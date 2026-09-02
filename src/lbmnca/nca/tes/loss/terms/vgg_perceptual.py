"""VGG 感知損失：深層特徵統計匹配（DyNCA / Texture-NCA 機制，破解 MSE→均勻場模糊）。

背景診斷
--------
mean_module 一直塌成均勻 / 過平滑場。根因 = NCA 擴散算子 + pixel-MSE 條件回歸雙重
平滑吸引子：pixel-MSE 對任何高頻細節都給「往平均靠」的梯度，NCA 的局部擴散則進一步
抹平結構。DyNCA / Texture-NCA 改用凍結 VGG 的深層特徵統計（feature statistics）而非
pixel MSE 當監督信號——模糊場的特徵統計（channel-wise mean/std、Gram matrix）與銳利
target 顯著不同，因此被罰，迫使模型維持高頻渦街結構。

本模組把此機制搬入 mean_step：
  - structure-matching：選定 relu 層特徵的 spatial L1（pred vs target 同位置）→ 保尾流位置。
  - moment-matching：各層特徵 channel-wise mean+std 差 → 抗模糊（特徵統計錯則被罰）。

輸入慣例
--------
pred / target 為 normalized mphy [B,9,H,W]，**必須先 ``denorm_mphy`` → vphy → colored
RGB**，不可對 ``Mphy_lattice_norm`` 直接算 vel/vort。colored RGB 由傳入的
``FieldColorizer`` 以 ``use_fixed_range=True`` 取得（值域對齊視覺面板）。

數值穩定性
----------
全程 fp32（``torch.amp.autocast("cuda", enabled=False)``）。VGG fp16 的 conv 累加與
近零梯度不穩，且 vel_mag 的 ``sqrt`` backward 在零速度像素溢出 inf 梯度，故 sqrt 一律
加 1e-8 eps。VGG 權重凍結（``requires_grad_(False)`` + ``eval()``），不回傳梯度到 VGG，
但梯度照常流回 pred。

MRT channel indices（9ch physics tensor）：
  RHO_CH = 0  (m0: 密度)
  JX_CH  = 3  (m3: x 動量)
  JY_CH  = 5  (m5: y 動量)

公開 API：
  vgg_perceptual_loss() — structure + moment 特徵統計 loss（scalar）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torchvision.models import VGG16_Weights

from ....utils.FieldNormalizer import FieldNormalizer
from ...utils.spatial_ops import ddx as _ddx, ddy as _ddy

RHO_CH: int = 0
JX_CH: int = 3
JY_CH: int = 5

_RHO_EPS: float = 1e-3

# ImageNet 統計（VGG16 預訓練分布；colored RGB∈[0,1] → 標準化）
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# torchvision 官方 VGG16 ImageNet 預訓練權重（``vgg16-397923af.pth`` 的正式來源）。
# 首次使用時由 torch.hub 下載並快取至 ``$TORCH_HOME/hub/checkpoints/``，之後離線可用。
_VGG16_WEIGHTS_URL: str = VGG16_Weights.IMAGENET1K_V1.url

__all__ = [
    "vgg_perceptual_loss",
    "vgg_ot_loss",
]


# ---------------------------------------------------------------------------
# VGG16 features（純 torch；module-level lazy singleton，per-device cache）
# ---------------------------------------------------------------------------

_VGG_CACHE: dict[torch.device, nn.Sequential] = {}


def _build_vgg16_features() -> nn.Sequential:
    """建構 torchvision-相容拓樸的 VGG16 ``features`` 子網。

    以 torchvision VGG cfg "D" 疊出 Conv/ReLU/MaxPool，使官方 ``vgg16-397923af.pth``
    權重（key 前綴 ``features.``）可直接對齊載入。手動建構而非呼叫
    ``torchvision.models.vgg16()``，是為了只取 features 子網（省去 classifier 的
    1.2 億參數），且 ``nn.Sequential`` 層索引與 ``_vgg_extract`` 的 relu 層編號一致。

    Returns:
        ``nn.Sequential``：未載權重的 VGG16 features（13 conv + relu + 5 maxpool）。
    """
    cfg = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ]
    layers: list[nn.Module] = []
    inc = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(2, 2)]
        else:
            layers += [nn.Conv2d(inc, int(v), 3, padding=1), nn.ReLU(inplace=True)]
            inc = int(v)
    return nn.Sequential(*layers)


def _get_vgg(device: torch.device) -> nn.Sequential:
    """取得（必要時建構並載權重）指定 device 上的凍結 VGG16 features singleton。

    首次呼叫：建網 → 由 torch.hub 取官方 state_dict（首次下載，之後讀快取；只取
    ``features.*``）→ ``eval()`` → ``requires_grad_(False)`` → 搬到 device → cache。
    後續同 device 直接回傳。

    Args:
        device: 目標裝置。

    Returns:
        凍結（eval + no-grad）的 VGG16 features ``nn.Sequential``。
    """
    feat = _VGG_CACHE.get(device)
    if feat is None:
        feat = _build_vgg16_features()
        sd = load_state_dict_from_url(_VGG16_WEIGHTS_URL, map_location="cpu")
        fsd = {
            k.replace("features.", ""): v
            for k, v in sd.items()
            if k.startswith("features.")
        }
        # strict=False：官方 state_dict 另含 classifier.*，此處只取 features 子網。
        feat.load_state_dict(fsd, strict=False)
        feat.eval()
        feat.requires_grad_(False)
        feat = feat.to(device)
        _VGG_CACHE[device] = feat
    return feat


def _vgg_extract(x_rgb: Tensor, layers: tuple[int, ...]) -> list[Tensor]:
    """在指定 relu 層索引擷取 VGG16 特徵圖。

    輸入 colored RGB∈[0,1] → ImageNet 標準化 → 逐層前傳，於 ``layers`` 指定的
    sequential index 後收集 activation。VGG 凍結，但保留 autograd 圖使梯度流回輸入。

    Args:
        x_rgb: ``[B,3,H,W]`` colored RGB，值域 [0,1]。
        layers: 要擷取的 ``nn.Sequential`` 層索引（如 relu1_2=3 / relu2_2=8 / relu3_3=15）。

    Returns:
        各指定層的特徵圖 list（``[B,C_l,H_l,W_l]``）。
    """
    feat = _get_vgg(x_rgb.device)
    mean = x_rgb.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x_rgb.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    h = (x_rgb - mean) / std
    max_layer = max(layers)
    out: list[Tensor] = []
    want = set(layers)
    for i, layer in enumerate(feat):
        h = layer(h)
        if i in want:
            out.append(h)
        if i >= max_layer:
            break
    return out


# ---------------------------------------------------------------------------
# mphy(norm) → vel_mag / vorticity（denorm 後 raw，sqrt 加 eps）
# ---------------------------------------------------------------------------


def _vel_vort(
    raw: Tensor,
    rho_eps: float = _RHO_EPS,
) -> tuple[Tensor, Tensor]:
    """denorm 後的 raw mphy → (vel_mag, vorticity)，均 ``[B,1,H,W]``。

    Args:
        raw: ``[B,9,H,W]`` 反正規化後的 raw mphy。
        rho_eps: 密度下限夾值（避免除零）。

    Returns:
        ``(velm, vort)``：vel_mag 與 vorticity，皆 ``[B,1,H,W]``。sqrt 加 1e-8 eps
        避免零速度像素 inf 梯度。
    """
    rho = raw[:, RHO_CH : RHO_CH + 1].clamp_min(rho_eps)
    vx = (raw[:, JX_CH : JX_CH + 1] / rho).clamp(-1.0, 1.0)
    vy = (raw[:, JY_CH : JY_CH + 1] / rho).clamp(-1.0, 1.0)
    velm = (vx * vx + vy * vy + 1e-8).sqrt()  # [B,1,H,W]
    vort = (_ddx(vy[:, 0]) - _ddy(vx[:, 0])).unsqueeze(1)  # [B,1,H,W]
    return velm, vort


def _project_sort(x: Tensor, proj: Tensor) -> Tensor:
    """特徵投影到隨機方向後沿空間排序（Sliced Wasserstein 用）。

    Args:
        x: ``[B,C,N]`` 特徵（N=H·W flatten）。
        proj: ``[C,P]`` 隨機單位投影方向。

    Returns:
        ``[B,P,N]`` 投影後沿 N 軸排序的值。
    """
    return torch.einsum("bcn,cp->bpn", x, proj).sort(dim=-1)[0]


def _sliced_ot_loss(fp: Tensor, ft: Tensor, proj_n: int = 32) -> Tensor:
    """Sliced Optimal Transport（Sliced Wasserstein）特徵分布匹配（DyNCA SOTA style loss）。

    比 Gram/moment 更佳：隨機投影 + 排序 + 比較 → 匹配特徵的**完整分布**（高階統計），
    模糊場分布錯 → 被罰。較 Gram 穩定、不色偏。

    Args:
        fp: pred 特徵 ``[B,C,H,W]``。
        ft: target 特徵 ``[B,C,H,W]``。
        proj_n: 隨機投影方向數（Monte-Carlo OT）。

    Returns:
        scalar：sorted 投影差的平方和（除以元素數正規化）。
    """
    b, c, h, w = fp.shape
    fp2 = fp.reshape(b, c, h * w)
    ft2 = ft.reshape(b, c, h * w)
    projs = F.normalize(torch.randn(c, proj_n, device=fp.device, dtype=fp.dtype), dim=0)
    sp = _project_sort(fp2, projs)
    tp = _project_sort(ft2, projs)
    if sp.shape[-1] != tp.shape[-1]:
        tp = F.interpolate(tp, sp.shape[-1], mode="nearest")
    return (sp - tp).square().sum() / (b * proj_n * sp.shape[-1])


def _feature_moment_diff(fp: Tensor, ft: Tensor, eps: float = 1e-8) -> Tensor:
    """單層特徵的 channel-wise mean+std 統計差（moment-matching，抗模糊核心）。

    模糊場的特徵能量分布（per-channel mean/std）與銳利 target 不同 → 被罰，迫使模型
    維持高頻結構。對空間維度（H,W）做統計，保留 batch 與 channel 維。

    Args:
        fp: pred 特徵 ``[B,C,H,W]``。
        ft: target 特徵 ``[B,C,H,W]``。
        eps: std 開根下限保護。

    Returns:
        scalar：|μ_p−μ_t| + |σ_p−σ_t| 的平均（L1）。
    """
    pm = fp.mean(dim=(2, 3))
    tm = ft.mean(dim=(2, 3))
    ps = (fp.var(dim=(2, 3), unbiased=False) + eps).sqrt()
    ts = (ft.var(dim=(2, 3), unbiased=False) + eps).sqrt()
    return (pm - tm).abs().mean() + (ps - ts).abs().mean()


# ---------------------------------------------------------------------------
# VGG 感知 loss（structure + moment；vel + vort 平均）
# ---------------------------------------------------------------------------


def vgg_perceptual_loss(
    pred_norm: Tensor,  # [B, 9, H, W] normalized mphy
    target_norm: Tensor,  # [B, 9, H, W] normalized mphy
    fluid_mask: Tensor,  # [B, 1, H, W] float（流體=1 固體=0）
    field_normalizer: FieldNormalizer,
    colorizer,  # FieldColorizer 實例
    structure_weight: float = 1.0,
    moment_weight: float = 1.0,
    ot_weight: float = 0.0,
    layers: tuple[int, ...] = (3, 8, 15),
    use_vel: bool = True,
    use_vort: bool = True,
    max_size: int = 256,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """VGG 深層特徵統計 loss（DyNCA 機制：破解 MSE→均勻場模糊）。

    管線：mphy(norm) → ``denorm_mphy`` → vel_mag / vorticity（sqrt 加 eps）→
    ``FieldColorizer`` fixed-range colored RGB∈[0,1] → ImageNet 標準化 → 凍結 VGG16，
    於 ``layers`` 指定 relu 層擷取特徵，計算兩種匹配：

      - **structure-matching**（保尾流位置）：選定層特徵的 spatial L1（pred vs target
        同位置）。提供空間對應監督，使尾流結構落在正確位置。
      - **moment-matching**（抗模糊）：各層特徵 channel-wise mean+std 差。模糊場特徵
        統計錯 → 被罰，是維持銳利度的核心信號。

    全程 fp32（VGG fp16 不穩）。VGG 凍結不回傳梯度到 VGG 權重，但梯度需流回 pred。
    vel 與 vort 兩 colored 通道各算一次，最後取平均。

    Args:
        pred_norm: ``[B,9,H,W]`` 正規化預測 mphy。
        target_norm: ``[B,9,H,W]`` 正規化目標 mphy。
        fluid_mask: ``[B,1,H,W]`` float，流體=1 固體=0（傳入 colorizer 作 solid 灰底）。
        field_normalizer: ``FieldNormalizer``（``denorm_mphy``）。
        colorizer: ``FieldColorizer``（``getVelColor`` / ``getVorColor``，use_fixed_range=True）。
        structure_weight: structure-matching 權重。
        moment_weight: moment-matching 權重。
        layers: 擷取特徵的 VGG ``nn.Sequential`` 層索引（relu1_2=3 / relu2_2=8 / relu3_3=15）。
        use_vel: 是否計入 vel_mag colored 通道。
        use_vort: 是否計入 vorticity colored 通道。
        rho_eps: 密度下限夾值。

    Returns:
        scalar Tensor = ``structure_weight·struct + moment_weight·moment``（vel/vort 平均）。
    """
    with torch.amp.autocast("cuda", enabled=False):
        pr = field_normalizer.denorm_mphy(pred_norm.float())
        tg = field_normalizer.denorm_mphy(target_norm.float())

        p_vel, p_vort = _vel_vort(pr, rho_eps=rho_eps)
        t_vel, t_vort = _vel_vort(tg, rho_eps=rho_eps)

        m = fluid_mask.float()

        def _channel_loss(p_f: Tensor, t_f: Tensor, get_color):
            pc = get_color(p_f, m, True)  # [B,3,H,W] fixed-range RGB∈[0,1]
            tc = get_color(t_f, m, True)
            # 降採樣至 max_size（VGG 全解析 360×640 太慢；特徵統計不需全解析）
            _hh, _ww = pc.shape[-2:]
            if max(_hh, _ww) > max_size:
                _sc = max_size / max(_hh, _ww)
                _nh, _nw = max(1, int(round(_hh * _sc))), max(1, int(round(_ww * _sc)))
                pc = F.interpolate(pc, (_nh, _nw), mode="bilinear", align_corners=False)
                tc = F.interpolate(tc, (_nh, _nw), mode="bilinear", align_corners=False)
            p_feats = _vgg_extract(pc, layers)
            t_feats = _vgg_extract(tc, layers)
            struct = pc.new_zeros(())
            moment = pc.new_zeros(())
            ot = pc.new_zeros(())
            for fp, ft in zip(p_feats, t_feats):
                if structure_weight > 0.0:
                    struct = struct + F.l1_loss(fp, ft)
                if moment_weight > 0.0:
                    moment = moment + _feature_moment_diff(fp, ft)
                if ot_weight > 0.0:
                    ot = ot + _sliced_ot_loss(fp, ft)
            n_l = max(len(p_feats), 1)
            return struct / n_l, moment / n_l, ot / n_l

        total = pred_norm.new_zeros(())
        n = 0
        if use_vel:
            s, mo, ot = _channel_loss(p_vel, t_vel, colorizer.getVelColor)
            total = total + structure_weight * s + moment_weight * mo + ot_weight * ot
            n += 1
        if use_vort:
            s, mo, ot = _channel_loss(p_vort, t_vort, colorizer.getVorColor)
            total = total + structure_weight * s + moment_weight * mo + ot_weight * ot
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# VGG cosine relaxed-OT（magnitude-free grain suppression）
# ---------------------------------------------------------------------------


def _masked_gather(feat: Tensor, fluid_mask_hw: Tensor, max_tokens: int) -> Tensor:
    """Feature map から流體 token を gather し max_tokens に random subsample する。

    Args:
        feat: ``[B, C, H, W]`` VGG 特徵圖。
        fluid_mask_hw: ``[B, 1, H', W']`` or ``[1, 1, H', W']`` 流體遮罩（>0.5=流體）。
        max_tokens: 每 block 流體 token 上限（bound matmul）。

    Returns:
        ``[B, N, C]``  N ≤ min(fluid_pixels, max_tokens)。
    """
    B, C, H, W = feat.shape
    m = fluid_mask_hw.float()
    if m.shape[-2:] != (H, W):
        m = F.interpolate(m, size=(H, W), mode="area")
    # 取 batch 間共有流體區（intersection），保 [B, N, C] 形狀一致
    common = (m > 0.5).all(dim=0).view(H * W)  # [HW]
    f_flat = feat.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
    tokens = f_flat[:, common, :]  # [B, N_common, C]
    N = tokens.shape[1]
    if N == 0:  # fallback: no common fluid tokens → use all pixels
        tokens = f_flat
        N = tokens.shape[1]
    if N > max_tokens:
        idx = torch.randperm(N, device=feat.device)[:max_tokens]
        tokens = tokens[:, idx, :]
    return tokens  # [B, N, C]


def _cosine_ot(X: Tensor, Y: Tensor) -> Tensor:
    """Kolkin cosine relaxed-OT: L_s = max(D(X→Y), D(Y→X))。

    magnitude-free：F.normalize 後 cosine 對 feature 幅度盲→grain(方向錯)與 ens(能量高)正交。

    Args:
        X: ``[B, Nx, C]`` 單位化 pred token。
        Y: ``[B, Ny, C]`` 單位化 GT token。

    Returns:
        scalar：对称 NN cosine distance。
    """
    S = torch.bmm(X, Y.permute(0, 2, 1))  # [B, Nx, Ny] cosine matrix
    dxy = (1.0 - S.max(dim=2).values).mean(dim=1)  # [B] D(X→Y)
    dyx = (1.0 - S.max(dim=1).values).mean(dim=1)  # [B] D(Y→X)
    return torch.maximum(dxy, dyx).mean()  # scalar


def _moment_unit(X: Tensor, Y: Tensor) -> Tensor:
    """Magnitude-free moment: (1/C)‖μXn−μYn‖₁ + (1/C²)‖var_Xn−var_Yn‖₁。

    Args:
        X: ``[B, N, C]`` 已單位化 token（F.normalize(dim=-1) 後）。
        Y: ``[B, N, C]``。

    Returns:
        scalar。
    """
    C = max(X.shape[-1], 1)
    mu_x, mu_y = X.mean(dim=1), Y.mean(dim=1)  # [B, C]
    var_x = X.var(dim=1, unbiased=False)  # [B, C]
    var_y = Y.var(dim=1, unbiased=False)
    return (mu_x - mu_y).abs().mean() / C + (var_x - var_y).abs().mean() / (C * C)


def vgg_ot_loss(
    pred: Tensor,  # [B, k, 9, H, W] rollout window (normalized mphy)
    targets: Tensor,  # [B, k, 9, H, W]
    fluid_mask: Tensor,  # [B, 1, H, W]
    field_normalizer: FieldNormalizer,
    colorizer,  # FieldColorizer
    weight_moment: float = 0.0,
    layers: tuple[int, ...] = (1, 6, 11, 18, 25),
    n_frames: int = 2,
    max_tokens: int = 2048,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """VGG cosine relaxed-OT perceptual loss（magnitude-free grain suppression）。

    管線：pred/targets ``[B,k,9,H,W]`` → 隨機選 n_frames 幀 → flatten ``[B·f,9,H,W]``
    → ``denorm_mphy`` → ``_vel_vort`` → ``colorizer``（solid=0.5 灰）
    → ImageNet norm → frozen VGG16（5 relu block）
    → ``_masked_gather``（只流體 token）→ ``F.normalize(C)`` → ``_cosine_ot``。

    magnitude-free 核心：feature 單位化後 cosine 對能量幅度盲；grain（texture 方向錯）
    被罰，真渦（同 GT 方向）不被罰 → grain vs ens 正交，壓 grain 不碰 ens。

    Args:
        pred: ``[B, k, 9, H, W]`` 歸一化預測 rollout（物理通道 9ch mphy）。
        targets: ``[B, k, 9, H, W]`` GT rollout。
        fluid_mask: ``[B, 1, H, W]`` 流體遮罩（流體=1 固體=0）。
        field_normalizer: 反歸一化 mphy。
        colorizer: ``FieldColorizer``（``getVelColor``/``getVorColor``，固體→0.5）。
        weight_moment: magnitude-free moment loss 權重（0 = 只用 cosine OT）。
        layers: VGG relu 層 sequential 索引（default=(1,6,11,18,25)=relu1_1..relu5_1）。
        n_frames: 隨機取幀數（horizon-agnostic，grad_tail 內期望）。
        max_tokens: 每 block 流體 token matmul 上限（bound O(Nx·Ny)）。
        rho_eps: 密度下限夾值。

    Returns:
        scalar Tensor：L_vggot（vel+vort 平均，n_frames 平均，n_block 加總）。
    """
    with torch.amp.autocast("cuda", enabled=False):
        B, k, _, H, W = pred.shape

        # 隨機選 n_frames 幀（horizon-agnostic）
        n_sel = min(n_frames, k)
        frame_idx = torch.randperm(k, device=pred.device)[:n_sel].sort().values

        # 取選中幀 + flatten batch×frame → [B*n_sel, 9, H, W]
        pred_f = pred[:, frame_idx, :, :, :].reshape(B * n_sel, 9, H, W).float()
        tgt_f = targets[:, frame_idx, :, :, :].reshape(B * n_sel, 9, H, W).float()
        # fluid_mask 擴展：每個 batch item 重複 n_sel 次 → [B*n_sel, 1, H, W]
        mask_f = fluid_mask.float().repeat_interleave(n_sel, dim=0)

        # denorm → vel/vort
        pr = field_normalizer.denorm_mphy(pred_f)
        tg = field_normalizer.denorm_mphy(tgt_f)
        p_vel, p_vort = _vel_vort(pr, rho_eps=rho_eps)
        t_vel, t_vort = _vel_vort(tg, rho_eps=rho_eps)

        def _ot_field(p_f: Tensor, t_f: Tensor, get_color) -> Tensor:
            pc = get_color(p_f, mask_f, True)  # [Bf, 3, H, W] ∈[0,1], solid=0.5
            tc = get_color(t_f, mask_f, True)
            p_feats = _vgg_extract(pc, layers)
            t_feats = _vgg_extract(tc, layers)
            blk = pc.new_zeros(())
            for fp, ft in zip(p_feats, t_feats):
                p_tok = _masked_gather(fp, mask_f, max_tokens)  # [Bf, N, C]
                t_tok = _masked_gather(ft, mask_f, max_tokens)
                if p_tok.shape[1] == 0 or t_tok.shape[1] == 0:
                    continue
                p_n = F.normalize(p_tok, dim=-1)
                t_n = F.normalize(t_tok, dim=-1)
                blk = blk + _cosine_ot(p_n, t_n)
                if weight_moment > 0.0:
                    blk = blk + weight_moment * _moment_unit(p_n, t_n)
            return blk / max(len(p_feats), 1)

        total = pred.new_zeros(()).float()
        total = total + _ot_field(p_vel, t_vel, colorizer.getVelColor)
        total = total + _ot_field(p_vort, t_vort, colorizer.getVorColor)

    return total / 2.0  # vel + vort 平均


# ---------------------------------------------------------------------------
# smoke test（自驗：random pred/target → backward → grad finite + nonzero）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "usage: python -m lbmnca.nca.tes.loss.terms.vgg_perceptual "
            "<path/to/<dataset>_stats.json>"
        )
        sys.exit(2)

    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _B, _H, _W = 2, 64, 64

    class _StubNorm:
        """denorm_mphy stub：tanh-norm 的近似反運算（smoke test 用）。"""

        def denorm_mphy(self, x: Tensor) -> Tensor:
            return torch.atanh(x.clamp(-0.9999, 0.9999)) * 0.1

    from lbmnca.nca.utils.FieldColorizer import FieldColorizer

    _norm = _StubNorm()
    _color = FieldColorizer.from_stats(sys.argv[1]).to(_dev)

    _pred = (torch.rand(_B, 9, _H, _W, device=_dev) * 1.6 - 0.8).requires_grad_(True)
    _tgt = torch.rand(_B, 9, _H, _W, device=_dev) * 1.6 - 0.8
    _mask = torch.ones(_B, 1, _H, _W, device=_dev)

    _loss = vgg_perceptual_loss(
        _pred,
        _tgt,
        _mask,
        _norm,
        _color,
        structure_weight=1.0,
        moment_weight=1.0,
        layers=(3, 8, 15),
    )
    _loss.backward()

    _lv = float(_loss.detach())
    _g = _pred.grad
    _finite = bool(torch.isfinite(_g).all())
    _nonzero = float(_g.abs().sum()) > 0.0
    print(f"[smoke] loss={_lv:.6f}  device={_dev}")
    print(
        f"[smoke] grad_finite={_finite}  grad_nonzero={_nonzero}  "
        f"grad_absmax={float(_g.abs().max()):.4e}"
    )
    if not (torch.isfinite(_loss).all() and _finite and _nonzero):
        print("[smoke] vgg_perceptual_loss FAIL")
        sys.exit(1)
    print("[smoke] vgg_perceptual_loss PASS")

    # -- vgg_ot_loss smoke --------------------------------------------------
    print("[smoke] testing vgg_ot_loss...")
    _B2, _K, _H2, _W2 = 2, 8, 64, 64

    _pred_ot = (
        torch.rand(_B2, _K, 9, _H2, _W2, device=_dev) * 1.6 - 0.8
    ).requires_grad_(True)
    _tgt_ot = torch.rand(_B2, _K, 9, _H2, _W2, device=_dev) * 1.6 - 0.8
    _mask_ot = torch.ones(_B2, 1, _H2, _W2, device=_dev)

    _loss_ot = vgg_ot_loss(
        _pred_ot,
        _tgt_ot,
        _mask_ot,
        _norm,
        _color,
        weight_moment=0.0,
        layers=(1, 6, 11, 18, 25),
        n_frames=2,
        max_tokens=2048,
    )
    _loss_ot.backward()

    _lv_ot = float(_loss_ot.detach())
    _g_ot = _pred_ot.grad
    _finite_ot = bool(torch.isfinite(_g_ot).all())
    _nonzero_ot = float(_g_ot.abs().sum()) > 0.0
    print(f"[smoke] vgg_ot loss={_lv_ot:.6f}  device={_dev}")
    print(
        f"[smoke] vgg_ot grad_finite={_finite_ot}  grad_nonzero={_nonzero_ot}  "
        f"grad_absmax={float(_g_ot.abs().max()):.4e}"
    )
    if not (torch.isfinite(_loss_ot).all() and _finite_ot and _nonzero_ot):
        print("[smoke] vgg_ot_loss FAIL")
        sys.exit(1)
    print("[smoke] vgg_ot_loss PASS")
