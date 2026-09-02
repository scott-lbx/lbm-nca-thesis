"""感知損失：SSIM + 頻域（固定數值範圍，強制 fp32 避免 fp16 NaN）。

提取自 loss_terms_v2.py，保留所有原始邏輯。

fp16 數值穩定性說明：
  - SSIM：fp16 方差計算（conv(x²) − μ²）存在消去誤差，sigma² 可能為負/nan，
    導致 loss=-inf，使 AMP scaler 連續跳過 optimizer step。
  - vel_mag sqrt：fp16 backward（1/2√vm）在近零速度像素溢出 inf 梯度。
  所有函式已強制轉為 float32 處理上述問題。

固定值域：
  _VEL_MAX   = 0.20   lattice units（對應 FieldColorizer.VEL_MAX）
  _VOR_RANGE = 0.04   lattice units（對應 FieldColorizer.VOR_RANGE）

公開 API：
  _ssim_loss_2d()        — 1 − mean(SSIM)，private helper
  struct_vel_ssim_loss() — vel_mag SSIM loss
  struct_vort_ssim_loss() — vorticity SSIM loss
  horiz_band_fft_loss()  — 水平帶頻域超量懲罰

MRT channel indices（9ch physics tensor）：
  RHO_CH = 0  (m0: 密度)
  JX_CH  = 3  (m3: x 動量)
  JY_CH  = 5  (m5: y 動量)

mask 慣例：
  fluid_mask [B,1,H,W] float，流體=1 固體=0
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ....utils.FieldNormalizer import FieldNormalizer
from ...utils.spatial_ops import (
    ddx as _ddx,
    ddy as _ddy,
)

RHO_CH: int = 0
JX_CH: int = 3
JY_CH: int = 5

_RHO_EPS: float = 1e-6
_VEL_MAX: float = 0.20    # matches FieldColorizer.VEL_MAX (lattice units)
_VOR_RANGE: float = 0.04  # matches FieldColorizer.VOR_RANGE

__all__ = [
    "struct_vel_ssim_loss",
    "struct_vort_ssim_loss",
    "horiz_band_fft_loss",
    "spectral_power_match_loss",
    "mean_spectral_lowmode_loss",
    "colored_residual_huber_loss",
]


# ---------------------------------------------------------------------------
# Colored-space Huber 殘差 loss（視覺優先；mphy→vphy→colored RGB 全程可微）
# ---------------------------------------------------------------------------


def colored_residual_huber_loss(
    pred_norm: Tensor,        # [B, 9, H, W] normalized mphy
    target_norm: Tensor,      # [B, 9, H, W] normalized mphy
    fluid_mask: Tensor,       # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    colorizer,                # FieldColorizer 實例
    beta: float = 0.1,
    wake_weight: "Tensor | None" = None,  # [B,1,H,W] soft 加權（None→fluid_mask）
    use_vel: bool = True,
    use_vort: bool = True,
) -> Tensor:
    """colored vel/vort RGB 空間的 wake-weighted Huber（smooth-L1）loss。

    管線：mphy(norm) → denorm → vel_mag/vorticity（sqrt 加 eps）→
    FieldColorizer fixed-range RGB → smooth_l1。全程可微（fixed_range 避開
    `.item()`、sqrt 加 1e-8 避免零速度 inf 梯度、強制 fp32）。

    Huber 對大誤差線性、不像 MSE 二次過懲 → 較不模糊、保住尾流銳利結構。
    colored 空間直接對齊視覺觀察（visual-first；與論文論述一致）。

    Args:
        pred_norm / target_norm: 正規化 mphy 場 [B,9,H,W]。
        fluid_mask: [B,1,H,W] 流體=1。
        normalizer: FieldNormalizer（denorm_mphy）。
        colorizer: FieldColorizer（getVelColor/getVorColor，use_fixed_range=True）。
        beta: smooth_l1 轉折點（colored∈[0,1]，0.1 合理）。
        wake_weight: 每像素 soft 權重（聚焦 child 偏轉區）；None→均勻 fluid。
        use_vel / use_vort: 是否計入該通道。

    Returns:
        scalar Tensor（vel+vort Huber 平均）。
    """
    with torch.amp.autocast("cuda", enabled=False):
        pr = normalizer.denorm_mphy(pred_norm.float())
        tg = normalizer.denorm_mphy(target_norm.float())

        def _vel_vort(raw: Tensor):
            rho = raw[:, RHO_CH:RHO_CH + 1].clamp_min(1e-3)
            vx = (raw[:, JX_CH:JX_CH + 1] / rho).clamp(-1.0, 1.0)
            vy = (raw[:, JY_CH:JY_CH + 1] / rho).clamp(-1.0, 1.0)
            velm = (vx * vx + vy * vy + 1e-8).sqrt()           # [B,1,H,W] eps→可微
            vort = (_ddx(vy[:, 0]) - _ddy(vx[:, 0])).unsqueeze(1)  # curl [B,1,H,W]
            return velm, vort

        p_vel, p_vort = _vel_vort(pr)
        t_vel, t_vort = _vel_vort(tg)

        m = fluid_mask.float()
        w = (wake_weight.float() if wake_weight is not None else m)

        def _huber_rgb(p_f: Tensor, t_f: Tensor, get_color) -> Tensor:
            pc = get_color(p_f, m, True)   # [B,3,H,W] fixed-range RGB
            tc = get_color(t_f, m, True)
            h = F.smooth_l1_loss(pc, tc, reduction="none", beta=beta)  # [B,3,H,W]
            wc = w.expand_as(h)
            return (h * wc).sum() / wc.sum().clamp_min(1e-8)

        total = pred_norm.new_zeros(())
        n = 0
        if use_vel:
            total = total + _huber_rgb(p_vel, t_vel, colorizer.getVelColor)
            n += 1
        if use_vort:
            total = total + _huber_rgb(p_vort, t_vort, colorizer.getVorColor)
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Radial power spectrum（cache）
# ---------------------------------------------------------------------------

_RADIAL_BIN_CACHE: dict = {}


def _radial_bin_index(H: int, W: int, n_bins: int, device: torch.device) -> Tensor:
    """rfft2 輸出 [H, W//2+1] 的 radial wavenumber → bin index（cache，shape 固定）。

    Args:
        H: 空間高度。
        W: 空間寬度。
        n_bins: radial bin 數量。
        device: 目標裝置。

    Returns:
        ``[H*(W//2+1)]`` 整數 Tensor，每個頻率格點對應的 bin 索引。
    """
    key = (H, W, n_bins)
    if key not in _RADIAL_BIN_CACHE:
        ky = torch.fft.fftfreq(H).abs()            # [H]
        kx = torch.fft.rfftfreq(W).abs()           # [W//2+1]
        kr = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        kr = kr / kr.max().clamp_min(1e-8)
        bins = torch.clamp((kr * n_bins).long(), 0, n_bins - 1).reshape(-1)  # [H*(W//2+1)]
        _RADIAL_BIN_CACHE[key] = bins
    return _RADIAL_BIN_CACHE[key].to(device)


# ---------------------------------------------------------------------------
# SSIM-based 結構 loss（fixed-range scalar field）
# ---------------------------------------------------------------------------


def _ssim_loss_2d(
    pred_01: Tensor,     # [N, 1, H, W] values in [0, 1]
    target_01: Tensor,   # [N, 1, H, W]
    window_size: int = 11,
    mask: Optional[Tensor] = None,   # [N, 1, H, W] 1=fluid；給定則只在 fluid 像素平均
) -> Tensor:
    """1 − mean(SSIM map)：SSIM loss 形式（低=好）。data_range=1。

    強制轉為 float32：fp16 下方差計算（conv(x²) − μ²）存在嚴重消去誤差，
    導致 sigma² 為負/nan，loss 輸出 -inf，使 AMP scaler 連續跳過 optimizer step。
    加 mask（fluid-only 平均，排除 solid 拖累；與 eval struct_vel_rgb_ssim 對齊）。

    Args:
        pred_01: ``[N, 1, H, W]`` 預測場，值域 [0, 1]。
        target_01: ``[N, 1, H, W]`` 目標場，值域 [0, 1]。
        window_size: 高斯窗大小（奇數）。
        mask: 可選 ``[N, 1, H, W]`` fluid mask（1=fluid）；給定則 SSIM map 只在 fluid 像素平均。

    Returns:
        scalar Tensor，= 1 − mean(SSIM)（鉗制 ≥0 避免浮點誤差造成負值）。
    """
    # 強制 float32：fp16 方差計算有消去誤差 → sigma² 負/nan → loss=-inf → AMP scaler 崩潰
    pred_01 = pred_01.float()
    target_01 = target_01.float()
    C1 = 1e-4   # (0.01)^2
    C2 = 9e-4   # (0.03)^2
    ws = window_size
    sigma = 1.5
    coords = torch.arange(ws, device=pred_01.device, dtype=torch.float32) - ws // 2
    gauss = torch.exp(-coords.square() / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window = (gauss.unsqueeze(1) * gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    pad = ws // 2
    mu1 = F.conv2d(pred_01, window, padding=pad)
    mu2 = F.conv2d(target_01, window, padding=pad)
    sigma1_sq = F.conv2d(pred_01 * pred_01, window, padding=pad) - mu1 * mu1
    sigma2_sq = F.conv2d(target_01 * target_01, window, padding=pad) - mu2 * mu2
    sigma12 = F.conv2d(pred_01 * target_01, window, padding=pad) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1.square() + mu2.square() + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    if mask is not None:
        m = (mask > 0.5).to(ssim_map.dtype)
        mean_ssim = (ssim_map * m).sum() / m.sum().clamp_min(1.0)
    else:
        mean_ssim = ssim_map.mean()
    return (1.0 - mean_ssim).clamp(min=0.0)  # ssim_map 浮點誤差可能略>1 → loss 負；鉗制避免優化錯方向


def struct_vel_ssim_loss(
    pred_norm: Tensor,      # [B, K, 9, H, W] normalized
    target_norm: Tensor,    # [B, K, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
    range_from_stats: bool = False,
) -> Tensor:
    """StructVel_SsimLoss：1 − SSIM on fixed-range vel_mag (0 to VEL_MAX)。

    固定值域 [0, VEL_MAX] → [0, 1] 後計算 SSIM，與視覺面板範圍一致。

    Args:
        pred_norm: ``[B, K, 9, H, W]`` 正規化預測軌跡。
        target_norm: ``[B, K, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0（此函式內未直接用於遮罩，保留介面一致性）。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。

    Returns:
        scalar Tensor（1 − mean SSIM on vel_mag）。
    """
    B, K, C, H, W = pred_norm.shape
    _vel_max, _ = _ssim_ranges(normalizer, range_from_stats)

    def _vel_mag_01(x_norm: Tensor) -> Tensor:
        # 強制 fp32：fp16 sqrt backward（1/2√vm）在近零速度像素溢出 inf 梯度，
        # 導致 AMP scaler 跳過每個 optimizer step。
        raw = normalizer.denorm_mphy(x_norm.float().reshape(B * K, C, H, W))
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        vm = ((raw[:, JX_CH] / rho).square() + (raw[:, JY_CH] / rho).square() + 1e-8).sqrt()
        return (vm / _vel_max).clamp(0.0, 1.0).unsqueeze(1)  # [B*K, 1, H, W]

    # fluid_mask [B,1,H,W] → [B*K,1,H,W]（對齊 flatten 的 K 軸）；fluid-only SSIM（排除 solid 拖累）
    _m = fluid_mask.unsqueeze(1).expand(B, K, 1, H, W).reshape(B * K, 1, H, W)
    return _ssim_loss_2d(_vel_mag_01(pred_norm), _vel_mag_01(target_norm), mask=_m)



def _ssim_ranges(normalizer: FieldNormalizer, from_stats: bool) -> tuple[float, float]:
    """回傳 SSIM 映射用的 ``(vel_max, vor_range)``。

    config: ``ssim_range_from_stats``——本模組原本硬編 ``_VEL_MAX=0.20``／
    ``_VOR_RANGE=0.04``，而 eval 側（gate、視覺面板）走的是 ``stats.json`` 的
    ``vis_range``（canonical 實測 0.14／0.016）。兩者不同，
    使 **訓練目標與 gate 是兩個不同的泛函**——同一個條紋缺陷在 loss 側被壓縮到
    較窄的動態範圍內，敏感度差 5.43 倍（0.04/0.016 × 0.20/0.14 ≈ 3.57 為渦度側，
    速度側 1.43）。訓練壓不到 gate 在意的東西。

    Args:
        normalizer: 反正規化器（``vis_range`` 由 ``stats.json`` 注入）。
        from_stats: True → 讀 SSOT；False → 沿用硬編常數（既有 run 零行為變更）。

    Returns:
        ``(vel_max, vor_range)``。

    Raises:
        RuntimeError: ``from_stats=True`` 但 normalizer 未注入 ``vis_range``
            （由 ``FieldNormalizer._require_vis`` 丟出，刻意不給 fallback）。
    """
    if not from_stats:
        return _VEL_MAX, _VOR_RANGE
    return float(normalizer.vel_vis_max), float(normalizer.vor_vis_range)

def struct_vort_ssim_loss(
    pred_norm: Tensor,      # [B, K, 9, H, W] normalized
    target_norm: Tensor,    # [B, K, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
    range_from_stats: bool = False,
) -> Tensor:
    """StructVort_SsimLoss：1 − SSIM on fixed-range vorticity ([-VOR_RANGE, VOR_RANGE])。

    映射至 [0, 1] 後計算 SSIM，與視覺面板範圍一致。

    Args:
        pred_norm: ``[B, K, 9, H, W]`` 正規化預測軌跡。
        target_norm: ``[B, K, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0（保留介面一致性）。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。

    Returns:
        scalar Tensor（1 − mean SSIM on vorticity）。
    """
    B, K, C, H, W = pred_norm.shape
    _, _vor_range = _ssim_ranges(normalizer, range_from_stats)

    def _vort_01(x_norm: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W))
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        vel_x = raw[:, JX_CH] / rho
        vel_y = raw[:, JY_CH] / rho
        vort = _ddx(vel_y) - _ddy(vel_x)
        return ((vort + _vor_range) / (2 * _vor_range)).clamp(0.0, 1.0).unsqueeze(1)

    _m = fluid_mask.unsqueeze(1).expand(B, K, 1, H, W).reshape(B * K, 1, H, W)
    return _ssim_loss_2d(_vort_01(pred_norm), _vort_01(target_norm), mask=_m)


# ---------------------------------------------------------------------------
# 頻域 loss（水平帶偽影懲罰）
# ---------------------------------------------------------------------------


def horiz_band_fft_loss(
    pred_norm: Tensor,      # [B, K, 9, H, W] normalized
    target_norm: Tensor,    # [B, K, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
    eps: float = 1e-8,
) -> Tensor:
    """StructHBandFFT_Loss：懲罰預測速度幅值中過多的 kx=0 能量（水平帶狀偽影）。

    horiz_ratio = kx=0 非-DC 功率 / 全部非-DC 功率。
    可微版 horizontal_band_score_fft（eval 指標）。取後半 rollout 步平均，降低逐步噪聲。
    回傳 clamp(horiz_ratio(pred) − horiz_ratio(target), min=0).mean(batch)。

    kx=0 欄位在 rfft2([B,H,W]) 輸出 [B,H,W//2+1] 中索引為 [:, :, 0]。

    Args:
        pred_norm: ``[B, K, 9, H, W]`` 正規化預測軌跡。
        target_norm: ``[B, K, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。
        eps: 分母保護。

    Returns:
        scalar Tensor（水平帶偽影超量懲罰）。
    """
    B, K, C, H, W = pred_norm.shape
    k_use = max(1, K // 2)
    pred_tail = pred_norm[:, -k_use:].reshape(B * k_use, C, H, W).float()
    tgt_tail = target_norm[:, -k_use:].reshape(B * k_use, C, H, W).float()

    def _vel_mag(x: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x)
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        # +1e-8 防 sqrt(0) backward=inf/nan（零速度像素，同 SSIM fp16 鐵律）
        return ((raw[:, JX_CH] / rho).square() + (raw[:, JY_CH] / rho).square() + 1e-8).sqrt()

    fm = fluid_mask.squeeze(1)[:, None, :, :].expand(B, k_use, H, W).reshape(B * k_use, H, W)
    vm_pred = (_vel_mag(pred_tail) * fm).reshape(B, k_use, H, W).mean(dim=1)
    vm_tgt = (_vel_mag(tgt_tail) * fm).reshape(B, k_use, H, W).mean(dim=1)

    def _horiz_ratio(vel: Tensor) -> Tensor:
        fft = torch.fft.rfft2(vel)               # [B, H, W//2+1] complex
        # |z|² via real²+imag²，避免 fft.abs() 在零頻分量 z=0 處 backward=0/0=nan
        # （數學等價 abs().square()，但 autograd 平滑；WF-W1 凍結 bug 根因）。
        power = fft.real.square() + fft.imag.square()   # [B, H, W//2+1]
        dc = power[:, 0, 0]                      # [B]
        kx0_power = power[:, 1:, 0].sum(dim=-1) # [B] kx=0, ky!=0
        total_non_dc = power.sum(dim=(-2, -1)) - dc  # [B]
        return kx0_power / total_non_dc.clamp_min(eps)

    excess = (_horiz_ratio(vm_pred) - _horiz_ratio(vm_tgt)).clamp_min(0.0)
    return excess.mean()


def spectral_power_match_loss(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    n_bins: int = 24,
    k_hi_frac: float = 1.0,
    rho_eps: float = _RHO_EPS,
    eps: float = 1e-8,
) -> Tensor:
    """SpecEnergy_RadialPowerMatch：2D radial power spectrum match。

    ``k_hi_frac<1.0`` = band-limit 只錨 [0, k_hi_frac·n_bins] 低/中頻能量。

    Args:
        pred_norm: ``[B,k,9,H,W]`` 正規化預測物理場。
        target_norm: ``[B,k,9,H,W]`` 正規化目標物理場。
        fluid_mask: ``[B,1,H,W]``（solid 置 0 避邊界假能量）。
        normalizer: 反正規化器。
        n_bins: radial k bin 數（空間尺度數）。
        k_hi_frac: 高頻截斷比例（1.0=全頻段）。
        rho_eps: 密度下限夾值。
        eps: 分母保護。

    Returns:
        scalar loss（fp32）。
    """
    B, K, C, H, W = pred_norm.shape

    def _vel(x_norm: Tensor) -> tuple[Tensor, Tensor]:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W)).float()
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        return raw[:, JX_CH] / rho, raw[:, JY_CH] / rho        # [B*K, H, W]

    vpx, vpy = _vel(pred_norm)
    vgx, vgy = _vel(target_norm)
    mask = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W).float()
    vpx, vpy, vgx, vgy = vpx * mask, vpy * mask, vgx * mask, vgy * mask

    with torch.amp.autocast("cuda", enabled=False):   # FFT/complex 不走 autocast（fp32）
        def _power(vx: Tensor, vy: Tensor) -> Tensor:
            fx = torch.fft.rfft2(vx.float())
            fy = torch.fft.rfft2(vy.float())
            return fx.real ** 2 + fx.imag ** 2 + fy.real ** 2 + fy.imag ** 2   # [BK, H, W//2+1]

        pp = _power(vpx, vpy).reshape(B * K, -1)
        pg = _power(vgx, vgy).reshape(B * K, -1)
        bins = _radial_bin_index(H, W, n_bins, pp.device)             # [H*(W//2+1)]
        pp_k = torch.zeros(B * K, n_bins, device=pp.device, dtype=pp.dtype).index_add_(1, bins, pp)
        pg_k = torch.zeros(B * K, n_bins, device=pg.device, dtype=pg.dtype).index_add_(1, bins, pg)
        k_hi = max(1, min(n_bins, int(round(k_hi_frac * n_bins))))
        ppk, pgk = pp_k[:, :k_hi], pg_k[:, :k_hi]
        # mean-floor 分母：tiny-Pg bin 致相對誤差爆炸 → 加 5% per-sample 均能量 floor
        pg_floor = 0.05 * pgk.mean(dim=1, keepdim=True)
        l = ((ppk - pgk).abs() / (pgk + pg_floor + eps)).mean()
    return l


def mean_spectral_lowmode_loss(
    pred: Tensor,
    target: Tensor,
    fluid_mask: Tensor,
    modes: int = 24,
    energy_weight: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """複數低-mode 頻譜匹配（FNO 風，mean_module 用）。

    對 norm 空間 physics 場逐 channel rfft2，只取低頻 ``modes`` 個複數係數
    （兩空間維；含 H 的低 +ky / 低 −ky 兩角，比照 ``SpectralConv2d``），
    算 ``|F_pred − F_tgt|²``。能量加權 ``w=|F_tgt|`` → 自動優先「重要 mode」
    （目標能量大者梯度大）。

    與 pixel-MSE **非冗餘**關鍵＝非均勻加權（Parseval：全頻均勻權即 MSE）：
    低-mode 截斷 + 能量加權使梯度集中於大尺度主導模態 → 釘 time-average 場
    結構**含相位**、破均勻吸引子（均勻場低頻係數≈0 → loss 高 → 推離）。

    Args:
        pred: ``[B, C, H, W]`` norm 空間預測 physics。
        target: ``[B, C, H, W]`` norm 空間目標 physics。
        fluid_mask: ``[B, 1, H, W]``（solid 置 0 避邊界假能量）。
        modes: 兩空間維保留低頻 mode 上限（隨輸入尺寸 clamp）。
        energy_weight: ``True``＝以 ``|F_tgt|`` 加權（優先重要 mode）；``False``＝均勻。
        eps: 分母保護。

    Returns:
        scalar loss（fp32）。
    """
    m = fluid_mask.float()
    p = pred.float() * m
    t = target.float() * m
    _B, _C, H, _W = p.shape
    with torch.amp.autocast("cuda", enabled=False):
        Fp = torch.fft.rfft2(p, norm="ortho")     # [B,C,H,W//2+1] complex
        Ft = torch.fft.rfft2(t, norm="ortho")
        w_half = Fp.shape[-1]
        m1 = max(1, min(modes, H // 2))
        m2 = max(1, min(modes, w_half))

        def _lo(F: Tensor) -> Tensor:
            # 低頻兩角：低 +ky（前 m1 列）+ 低 −ky（末 m1 列）× 低 kx（前 m2 欄）
            return torch.cat([F[..., :m1, :m2], F[..., -m1:, :m2]], dim=-2)

        Fp_lo, Ft_lo = _lo(Fp), _lo(Ft)
        diff2 = (Fp_lo - Ft_lo).abs().pow(2)
        tgt2 = Ft_lo.abs().pow(2)
        # ★相對誤差（除目標能量）→ O(1) 尺度無關（絕對量級在 norm 空間 ~30 會壓垮其他 loss + AMP overflow）。
        if energy_weight:
            w = Ft_lo.abs()        # 能量加權：高能量(重要)mode 主導
            l = (w * diff2).sum() / ((w * tgt2).sum() + eps)
        else:
            l = diff2.sum() / (tgt2.sum() + eps)
    return l.float()
