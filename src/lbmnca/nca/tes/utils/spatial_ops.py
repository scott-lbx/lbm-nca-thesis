"""共用空間算子（central / flux-form difference、Gaussian 低通、mask 工具）。

從 ``tes.loss.loss_terms_v2`` 與 ``tes.loss.loss_terms_m1a`` 抽取，避免重複定義。

形狀慣例（分類與計數由 ``tests/nca/tes/test_r5_countable_claims.py::test_spatial_ops_all_split_is_ten`` 斷言）：
  - 任意前置維度 ``[..., H, W]``、輸入輸出同形（6 個）：:func:`ddx`、:func:`ddy`、
    :func:`ddx_fwd`、:func:`ddy_fwd`、:func:`divergence_flux`、:func:`gaussian_lowpass`。
  - docstring 契約為 4D ``[B, 1, H, W]``（3 個）：:func:`compute_wall_band`、
    :func:`interior_core_mask`、:func:`handoff_reset_mask`（後者輸出為 **bool**，非 float）。
  - :func:`fluid_mask_from_static` 另有例外：吃 ``[B, 4, H, W]`` 回 ``[B, 1, H, W]``，
    且需傳入 ``FieldNormalizer``（**非**純張量運算）。

★差分算子兩家族：
  - **中央差分家族** :func:`ddx` / :func:`ddy`（既有 loss 端算子，勿改）——
    核 ``[-0.5, 0, 0.5]``，頻率響應 ``i·sin(k)``；k=π（2Δ 波長）增益 **0**，
    λ=4 px 低估約 36%（``sin(π/2)/(π/2)=0.6366``）。對 band/grain 這類 grid-scale
    結構結構性免疫。
    ⚠ 本對算子**不是**「所有既有 loss/指標」的共同實作：``loss/terms/`` 下只有部分模組
    import ``ddx``/``ddy``（模組數、消費者數與逐支名單由
    ``tests/nca/tes/test_r5_countable_claims.py::test_loss_terms_module_and_importer_counts`` 斷言）；
    而 ``eval.metrics`` 走 **Sobel/8**。這幾支同屬中央差分家族但
    **數值不可互比**；逐條差異（含 ``eval.physics_gate.divj_over_lbm_floor``
    的三個 region 變體）見
    ``tes.eval.metrics.mass_conservation_residual`` docstring 的「量測慣例」段。
  - **通量式前向/後向差分** :func:`ddx_fwd` / :func:`ddy_fwd` / :func:`divergence_flux`
    （對照用）——核 ``[-1, 1]``，k=π 增益 2，且在固/流界面上有定義。
  兩者並存，新增者不取代任何既有指標，以保歷史讀數可比。

⚠ **兩家族的讀數只有在** :func:`interior_core_mask` **給出的核心區才可相除比較**：
:func:`divergence_flux` 把域外一律視為固體（no-penetration），因此在有 inlet/outlet
的開放域上，域邊界與壁面格點會回報 O(|j|) 的單面通量項——那是邊界處理慣例，
**不是**離散化頻率響應差。

CPU 實測（360×640 網格 + 一棟 100×60 建物、解析上 ``div≡0`` 的均勻流
``jx=0.1``、``jy=0``，每個數字皆標明所用遮罩）。``mean|div|``：

  - ``interior_core_mask(fluid, 2)`` 核心區：通量式 0.000e+00／``ddx``/``ddy`` 0.000e+00／
    Sobel/8 0.000e+00
  - 全 ``fluid_mask``：通量式 2.050e-04／``ddx``/``ddy`` 4.456e-05／Sobel/8 2.050e-04
  - ``fluid ∖ core``（壁面 + 域邊界共 2 px 環，4640 格）：通量式 9.914e-03／
    ``ddx``/``ddy`` 2.155e-03／Sobel/8 9.914e-03
  - 最外圈 1 px ∩ ``fluid``：通量式 1.804e-02／``ddx``/``ddy`` 0.000e+00／Sobel/8 1.801e-02

即在一個**完美無散度**的場上，通量式對 ``ddx``/``ddy`` 的「全流體比值」已達 4.6×，
純屬邊界慣例差異。★該比值**不是常數**：同樣量法換成 64×96 網格 + 20×20 建物為 4.2×
（域周長／流體面積之比不同），不可外推到其他幾何。
★另注意同屬中央差分家族的 Sobel/8（``eval.metrics``）在同場全流體讀數為 2.050e-04、
對通量式比值 1.00——因為 ``F.conv2d(padding=1)`` 的零填補在域邊界等效於域外 ``j=0``，
與通量式注入同一個單面項；:func:`ddx` 在域邊界改用單側差分故不注入。
**「4.6×」專指通量式 vs** :func:`ddx`/:func:`ddy`，套到 Sobel 版上不成立。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ...utils.FieldNormalizer import FieldNormalizer


def ddx(x: Tensor) -> Tensor:
    """x-方向 central difference，邊界用 forward/backward difference。

    Args:
        x: 任意形狀張量，最後兩維為 ``(H, W)``。

    Returns:
        與 ``x`` 同形狀的 x 方向偏導數張量。
    """
    out = torch.empty_like(x)
    out[..., 1:-1] = 0.5 * (x[..., 2:] - x[..., :-2])
    out[..., 0] = x[..., 1] - x[..., 0]
    out[..., -1] = x[..., -1] - x[..., -2]
    return out


def ddy(x: Tensor) -> Tensor:
    """y-方向 central difference，邊界用 forward/backward difference。

    Args:
        x: 任意形狀張量，最後兩維為 ``(H, W)``。

    Returns:
        與 ``x`` 同形狀的 y 方向偏導數張量。
    """
    out = torch.empty_like(x)
    out[..., 1:-1, :] = 0.5 * (x[..., 2:, :] - x[..., :-2, :])
    out[..., 0, :] = x[..., 1, :] - x[..., 0, :]
    out[..., -1, :] = x[..., -1, :] - x[..., -2, :]
    return out


def ddx_fwd(x: Tensor) -> Tensor:
    """x-方向**前向**差分 ``x[i+1] − x[i]``，最後一欄 ``[..., :, -1]`` 補 0。

    與 :func:`ddx` 的中央差分互補：中央差分核 ``[-0.5, 0, 0.5]`` 的頻率響應為
    ``i·sin(k)``，在 2Δ 波長（k=π，Nyquist）增益**恰為 0**，對 grid-scale 的
    band/grain 結構結構性免疫；前向差分核 ``[-1, 1]`` 響應 ``1 − e^{-ik}``，
    在 k=π 增益 = 2。本函式提供該對照算子。

    Args:
        x: 任意形狀張量，最後兩維為 ``(H, W)``。

    Returns:
        與 ``x`` 同形狀的前向差分結果；最後一欄 ``[..., :, -1]``（x 方向最末，索引 ``W-1``）為 0。
    """
    out = torch.zeros_like(x)
    out[..., :-1] = x[..., 1:] - x[..., :-1]
    return out


def ddy_fwd(x: Tensor) -> Tensor:
    """y-方向**前向**差分 ``x[j+1] − x[j]``，最後一列 ``[..., -1, :]`` 補 0。

    參見 :func:`ddx_fwd` 的頻率響應說明。

    Args:
        x: 任意形狀張量，最後兩維為 ``(H, W)``。

    Returns:
        與 ``x`` 同形狀的前向差分結果；最後一列 ``[..., -1, :]``（y 方向最末，索引 ``H-1``）為 0。
    """
    out = torch.zeros_like(x)
    out[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return out


def divergence_flux(jx: Tensor, jy: Tensor, fluid_mask: Tensor) -> Tensor:
    """通量式（staggered / finite-volume）動量散度，對 grid-scale 結構有非零響應。

    定義（cell-centred，面通量取後向差分）::

        div_c = (jx_c − jx_west) + (jy_c − jy_south)

    西/南鄰為固體**或出界**時，該面通量視為 0（no-penetration 壁面條件）→ 本算子在固/流
    界面上仍有定義，不需要 erosion 就能在壁帶取到值（既有 ``divj_norm`` 的 ``erosion(2)``
    則把該圈排除）。壁帶讀數的解讀限制見下方警語。

    ⚠ **開放域警語**：本算子適用於「四周皆牆」的 Helmholtz 投影解，把域外一律
    當固體。本專案的域有 inlet/outlet，於是第 0 欄（西面出界）的流體格直接把整格
    ``jx`` 當成殘差（``j_c`` 而非 ``j_c − j_west``），第 0 列（南面出界）對 ``jy``
    同理，壁面格也回報 O(|j|)。CPU 實測（360×640 + 一棟 100×60 建物、解析上
    ``div≡0`` 的均勻流 ``jx=0.1``、``jy=0``，每個數字皆標明所用遮罩）：最外圈
    1 px ∩ ``fluid`` 1.804e-02、``fluid ∖ interior_core_mask(fluid, 2)``（壁面 +
    域邊界共 2 px 環）9.914e-03、全 ``fluid_mask`` 2.050e-04、
    :func:`interior_core_mask` 核心區 0.000e+00。
    ⚠ 該 2 px 環**不等於** :func:`compute_wall_band`（後者只取緊鄰固體的一圈，同場讀數
    3.086e-02）。
    → **全域讀數以邊界處理人工項為主，不可解讀成「中央差分低估了多少」**；要與中央差分
    家族相除比較，兩者都必須先限制在核心區（見 :func:`interior_core_mask`）。

    與 :func:`ddx`/:func:`ddy` 的中央差分家族並存、互為對照，**不取代**任何既有指標：
    既有 ``phys_mass_divj_absmean``（loss）與 ``eval.metrics``（Sobel/8）皆維持原算子，
    以保這兩者的歷史讀數可比。

    本函式的 torch 實作與獨立的 numpy 參考實作逐點等價，由
    ``tests/nca/tes/test_divergence_flux.py`` 的 ``test_matches_origin_probe_definition``
    與 ``test_matches_warmchain_probe_definition`` 守住。

    Args:
        jx: ``[..., H, W]`` x 方向動量（raw/denorm space）。
        jy: ``[..., H, W]`` y 方向動量，形狀須與 ``jx`` **完全相同**。
        fluid_mask: ``[..., H, W]`` float，流體=1 固體=0；形狀須與 ``jx`` **完全相同**。
            **不做廣播**——形狀不符直接 ``ValueError``（廣播會靜默給出錯誤形狀與錯誤數值，
            例如 ``mask=[B,1,H,W]`` 配 ``jx=[B,H,W]`` 會得到 ``[B,B,H,W]``）。

    Returns:
        ``[..., H, W]`` 散度場；固體格點恆為 0。

    Raises:
        ValueError: ``jx`` / ``jy`` / ``fluid_mask`` 三者形狀不完全一致時。
    """
    if not (jx.shape == jy.shape == fluid_mask.shape):
        raise ValueError(
            "divergence_flux 要求 jx/jy/fluid_mask 形狀完全一致（不做廣播），"
            f"得到 jx={tuple(jx.shape)}, jy={tuple(jy.shape)}, "
            f"fluid_mask={tuple(fluid_mask.shape)}"
        )
    f = fluid_mask
    jx_w = F.pad(jx, (1, 0, 0, 0))[..., :, :-1]        # 西鄰值（出界補 0）
    jy_s = F.pad(jy, (0, 0, 1, 0))[..., :-1, :]        # 南鄰值（出界補 0）
    f_w = f * F.pad(f, (1, 0, 0, 0))[..., :, :-1]      # 西面為 fluid-fluid 面
    f_s = f * F.pad(f, (0, 0, 1, 0))[..., :-1, :]      # 南面為 fluid-fluid 面
    dx = torch.where(f_w > 0.5, jx - jx_w, jx)         # 壁面 → 該面通量 0 → 殘差 = jx_c
    dy = torch.where(f_s > 0.5, jy - jy_s, jy)
    return (dx + dy) * f


def gaussian_lowpass(x: Tensor, sigma: float) -> Tensor:
    """可分離高斯低通濾波（band-limit 用）。

    擋 grid-scale 高頻（~2-3px grid 噪）、放行渦核尺度（~數十 px）。
    cutoff 由 sigma 控：sigma≈1.5-2px → 強衰減波長 <~4-6px、放行 >~10px。
    使用 replicate padding 保邊界一致性。

    Args:
        x: 任意前置維度 + ``(H, W)``；內部攤平為 ``[N, 1, H, W]`` 做 depthwise conv。
        sigma: 高斯標準差（px）；<= 0 直接回傳原張量（no-op，向後相容）。

    Returns:
        與 ``x`` 同形狀的低通濾波結果。
    """
    if sigma <= 0.0:
        return x
    shape = x.shape
    H, W = shape[-2], shape[-1]
    n = x.reshape(-1, 1, H, W)
    rad = max(1, int(round(3.0 * sigma)))
    ksize = 2 * rad + 1
    coords = torch.arange(ksize, device=x.device, dtype=torch.float32) - rad
    g = torch.exp(-coords.square() / (2.0 * sigma * sigma))
    g = (g / g.sum()).to(n.dtype)
    kx = g.view(1, 1, 1, ksize)
    ky = g.view(1, 1, ksize, 1)
    n = F.pad(n, (rad, rad, rad, rad), mode="replicate")
    n = F.conv2d(n, kx)
    n = F.conv2d(n, ky)
    return n.reshape(shape)


def compute_wall_band(fluid_mask: Tensor) -> Tensor:
    """從 fluid_mask 計算近壁帶（流體側與固體相鄰格點）。

    Args:
        fluid_mask: ``[B, 1, H, W]`` float 張量，流體格點=1，固體格點=0。

    Returns:
        ``[B, 1, H, W]`` float wall_band，僅流體中緊鄰固體的格點為 1。
    """
    solid_mask = 1.0 - fluid_mask
    solid_near = F.max_pool2d(solid_mask, kernel_size=3, stride=1, padding=1)
    return fluid_mask * (solid_near > 0.5).float()


def interior_core_mask(fluid_mask: Tensor, erode_iters: int = 2) -> Tensor:
    """流體核心遮罩：腐蝕 ``fluid_mask``，並把域外一律當固體（域邊界一併排除）。

    用途：:func:`divergence_flux`（通量式）與 :func:`ddx`/:func:`ddy`
    （中央差分）兩家族的讀數，只有在「西/南面必為 fluid–fluid、且不觸及域邊界」的格點上才
    看到同一組面，相除比較才只反映頻率響應差異。本函式即產生該區域的遮罩。
    在解析上 ``div≡0`` 的測試場中，兩家族在本遮罩內的 ``mean|div|`` 皆實測為 0.000e+00
    （CPU 實測，360×640 + 一棟建物）。

    實作細節：腐蝕採 3×3 方形結構元素（比 4-鄰接十字更嚴格），每次腐蝕前以 0 補邊
    → 域邊界 ``erode_iters`` px 一併被排除。與 scipy ``binary_erosion``
    （預設十字結構、``border_value=0``）語義相近但**不逐格相同**（本函式較嚴格），
    兩邊讀數不可逐格互比。

    Args:
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0（需 4D，比照
            :func:`compute_wall_band`）。
        erode_iters: 腐蝕次數（px）；``<=0`` 直接回傳輸入（不做任何排除，向後相容）。

    Returns:
        ``[B, 1, H, W]`` float 核心遮罩（核心=1）。
    """
    if erode_iters <= 0:
        return fluid_mask
    m = fluid_mask
    for _ in range(erode_iters):
        padded = F.pad(m, (1, 1, 1, 1), mode="constant", value=0.0)
        m = -F.max_pool2d(-padded, kernel_size=3, stride=1)   # min-pool = 腐蝕
    return m


def fluid_mask_from_static(
    static_norm: Tensor, normalizer: FieldNormalizer
) -> Tensor:
    """從正規化 static 場的 SDF channel（ch0）計算 fluid_mask。

    Args:
        static_norm: ``[B, 4, H, W]`` 正規化 static 場。
        normalizer: FieldNormalizer 實例，提供 denorm_sdf 方法。

    Returns:
        ``[B, 1, H, W]`` float fluid_mask，SDF > 0 的格點為流體（=1）。
    """
    sdf_raw = normalizer.denorm_sdf(static_norm[:, 0])
    return (sdf_raw > 0.0).float().unsqueeze(1)


def handoff_reset_mask(
    fluid_new: Tensor, fluid_old: Tensor | None, dilate_px: int
) -> Tensor:
    """warm-chain 換場 IC 重置遮罩：新幾何固體 ∪ 舊幾何固體 ∪ (變動邊界膨脹)。

    治「ghost-IC handoff」缺陷：
    舊版只清「新幾何判定為固體」的格點，「舊幾何是固體、新幾何變流體」的格點原封不動
    繼承上一 case 的殘留值（該值在舊 case 自身即已被歸零），呈現為與 GT 無關的矩形殘影。
    本函式把「新固體 ∪ 舊固體 ∪ 兩者變動邊界膨脹 dilate_px」一併歸零，把該區域交還模型
    既有的 damage-regrow 能力（``mean_pool_damage_*`` 已驗證此能力存在）重新長出正確流場，
    而非要求模型從一塊語意不明的殘留值精確修正。

    Args:
        fluid_new: ``[B, 1, H, W]`` float，新幾何流體遮罩（流體=1）。
        fluid_old: ``[B, 1, H, W]`` float 或 None（無前一 case / cold-start），舊幾何流體遮罩。
        dilate_px: 變動邊界膨脹半徑（px）；``<=0`` 或 ``fluid_old is None`` 時退化為僅清新固體
            （與舊版行為完全一致，零風險關閉）。

    Returns:
        ``[B, 1, H, W]`` bool，True＝該格點需歸零。
    """
    solid_new = fluid_new <= 0.5
    if dilate_px <= 0 or fluid_old is None:
        return solid_new
    solid_old = fluid_old <= 0.5
    changed = solid_new != solid_old
    k = 2 * dilate_px + 1
    changed_dil = F.max_pool2d(changed.float(), kernel_size=k, stride=1, padding=dilate_px) > 0.5
    return solid_new | solid_old | changed_dil


__all__ = [
    "ddx", "ddy", "ddx_fwd", "ddy_fwd", "divergence_flux", "gaussian_lowpass",
    "compute_wall_band", "interior_core_mask", "fluid_mask_from_static",
    "handoff_reset_mask",
]
