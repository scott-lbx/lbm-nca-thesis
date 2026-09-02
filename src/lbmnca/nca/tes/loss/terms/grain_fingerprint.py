"""grain 指紋 loss。

> turb clean-slate 重設計，治「視覺 grain」（結構化高頻斜向格子干涉 artifact）。
> 與相似度 loss（SSIM/VGG/OT）哲學不同：這是 **鑑別 loss** —— 找真渦沒有、grain
> 才有的可量化特徵（斜向各向異性 + Nyquist 翹起），**只罰那個**，不誤殺合法細渦。

設計原理（FFT 三指紋）：
  - 低頻區（k<k_low）：能量呈橢圓順主流 → 大尺度各向異性，**合法保留**（本 loss 不碰）。
  - 中高頻環（k_low≤k<k_nyq）：斜向異常能量帶 = **grain 本體** → 指紋 1 各向異性 P(θ)。
  - 高頻角落（k≥k_nyq）：checkerboard 翹起 → 指紋 2 Nyquist hinge（只罰超量）。
  - 中央水平+垂直十字亮線：邊界不連續 FFT artifact，**非物理** → Hann window + 軸排除雙保險。

作用對象：**denorm vorticity**（grain 訊噪比最佳）。本模組函式直接吃
vorticity 場 ``[B,1,H,W]``；若需從 norm mphy 取得 vorticity，呼叫端先 denorm
（``FieldNormalizer.denorm_mphy`` → ``vel=j/ρ`` → ``vorticity=∂vy/∂x−∂vx/∂y``）。

公開 API：
  grain_fingerprint_loss()  — w_aniso·L_aniso + w_nyq·L_nyq（標量）
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["grain_fingerprint_loss"]

_PI: float = 3.141592653589793

# 頻率座標 + Hann window 快取：(H, W, device, dtype) → dict
# 頻率/角度座標、軸排除遮罩、Hann window 與輸入內容無關，僅依空間尺寸；快取避免每步重算。
_GRID_CACHE: dict = {}


def _grain_grid(
    H: int,
    W: int,
    k_low: float,
    k_nyq: float,
    axis_excl_deg: float,
    n_theta: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """快取並回傳 grain 指紋所需的頻率座標、遮罩、theta bin 與 Hann window。

    fftshift 後的整數波數座標（kx, ky∈[-N/2, N/2)），徑向 ``kr``、角度 ``th``，
    以及中高頻環遮罩 ``band``、Nyquist 遮罩 ``nyq``、軸排除後的 theta bin。
    所有量僅依 ``(H, W, k_low, k_nyq, axis_excl_deg, n_theta)``，與場內容無關。

    Args:
        H: 空間高度。
        W: 空間寬度。
        k_low: 中高頻環內徑（各向同性恢復半徑；調參）。
        k_nyq: 中高頻環外徑 = Nyquist 帶起點（調參）。
        axis_excl_deg: θ≈0°/90° 排除窄帶半寬（度），殺中央假十字。
        n_theta: angular bin 數。
        device: 目標裝置。
        dtype: 目標浮點型別。

    Returns:
        dict，含 ``win`` [H,W]、``band`` [H,W] bool、``nyq`` [H,W] bool、
        ``theta_bins_band`` [M] long（僅 band 內格點的 bin 索引）、
        ``band_flat`` [H*W] bool（攤平的 band 遮罩，供 gather）。
    """
    key = (H, W, round(k_low, 4), round(k_nyq, 4), round(axis_excl_deg, 4), n_theta)
    cached = _GRID_CACHE.get(key)
    if cached is None:
        # fftshift 後的整數波數（cycles over the whole domain）
        ky = torch.fft.fftshift(torch.fft.fftfreq(H)) * H        # [H]
        kx = torch.fft.fftshift(torch.fft.fftfreq(W)) * W        # [W]
        ky2 = ky[:, None].expand(H, W)
        kx2 = kx[None, :].expand(H, W)
        kr = torch.sqrt(kx2 ** 2 + ky2 ** 2)
        th = torch.atan2(ky2, kx2)                               # [-π, π]

        # 軸排除遮罩（殺假十字）：θ 近 0/±π（水平軸）或 ±π/2（垂直軸）剔除
        excl = axis_excl_deg * _PI / 180.0
        near_horiz = (th.abs() < excl) | ((_PI - th.abs()) < excl)
        near_vert = ((th - 0.5 * _PI).abs() < excl) | ((th + 0.5 * _PI).abs() < excl)
        axis_mask = ~(near_horiz | near_vert)

        band = (kr >= k_low) & (kr < k_nyq) & axis_mask          # [H,W] bool 指紋 1 區
        nyq = kr >= k_nyq                                         # [H,W] bool 指紋 2 區

        # theta → bin（[0, n_theta)），僅保留 band 內格點（攤平後 gather/scatter）
        theta_bins = ((th + _PI) / (2.0 * _PI) * n_theta).long().clamp(0, n_theta - 1)
        band_flat = band.reshape(-1)
        theta_bins_band = theta_bins.reshape(-1)[band_flat]      # [M]

        # 2D Hann window（殺中央假十字 FFT artifact）
        win = (torch.hann_window(H, periodic=False)[:, None]
               * torch.hann_window(W, periodic=False)[None, :])  # [H,W]

        cached = {
            "win": win,
            "band_flat": band_flat,
            "nyq_flat": nyq.reshape(-1),
            "theta_bins_band": theta_bins_band,
        }
        _GRID_CACHE[key] = cached

    # 移到正確 device/dtype（bool/long 不轉 dtype）
    return {
        "win": cached["win"].to(device=device, dtype=dtype),
        "band_flat": cached["band_flat"].to(device=device),
        "nyq_flat": cached["nyq_flat"].to(device=device),
        "theta_bins_band": cached["theta_bins_band"].to(device=device),
    }


def grain_fingerprint_loss(
    pred_vort: Tensor,
    gt_vort: Tensor,
    k_low: float,
    k_nyq: float,
    n_theta: int = 36,
    axis_excl_deg: float = 8.0,
    w_aniso: float = 1.0,
    w_nyq: float = 0.3,
    eps: float = 1e-8,
) -> Tensor:
    """grain 指紋鑑別 loss：angular P(θ) 對齊 + Nyquist hinge。

    **鑑別 loss 非相似度 loss**：只罰 grain 的指紋（斜向各向異性 + Nyquist 翹起），
    不要求 pred 逐像素像 GT，故不誤殺合法細渦。

    - **指紋 1（主力）各向異性**：中高頻環 ``k_low≤k<k_nyq`` 的 angular power ``P(θ)``，
      pred 對齊 GT 的 ``P(θ)``。正規化成方向**分布**後 L1 → 比方向、不比總量
      （合法細渦使總量升但 P(θ) 仍平坦，不被罰；grain 使 P(θ) 斜向尖峰，被罰）。
      剔除 θ≈0°/90° 窄帶（``axis_excl_deg``，雙保險殺假十字）。
    - **指紋 2（配套）Nyquist hinge**：``k≥k_nyq`` 帶 ``max(0, E_pred−E_GT)``，只罰翹起
      （pred 高頻角落能量超過 GT 才罰，不罰 GT 本就有的合法高頻）。
    - **防呆**：先乘 2D Hann window（殺中央假十字 FFT artifact）。

    作用對象須為 **denorm vorticity**（grain 訊噪比最佳）。FFT 強制 fp32。

    Args:
        pred_vort: ``[B, 1, H, W]`` 預測 denorm vorticity。
        gt_vort: ``[B, 1, H, W]`` 目標 denorm vorticity。
        k_low: 中高頻環內徑（GT FFT 各向同性恢復半徑；調參）。整數波數單位
            （fftshift 後 ``kr=√(kx²+ky²)``，kx/ky∈[-N/2, N/2)）。
        k_nyq: 中高頻環外徑 = Nyquist 帶起點（建議 ≈ Nyquist 的 0.7–0.8；調參）。
        n_theta: angular bin 數。
        axis_excl_deg: θ≈0°/90° 排除窄帶半寬（度）。
        w_aniso: 指紋 1 權重。
        w_nyq: 指紋 2 權重。
        eps: 分布正規化分母保護。

    Returns:
        scalar Tensor = ``w_aniso·L_aniso + w_nyq·L_nyq``（fp32）。
    """
    if pred_vort.shape != gt_vort.shape:
        raise ValueError(
            f"pred_vort/gt_vort 形狀須相同，得 {tuple(pred_vort.shape)} vs {tuple(gt_vort.shape)}"
        )
    if pred_vort.ndim != 4 or pred_vort.shape[1] != 1:
        raise ValueError(f"vorticity 須為 [B,1,H,W]，得 {tuple(pred_vort.shape)}")

    B, _, H, W = pred_vort.shape

    # FFT/複數一律 fp32（與 perceptual.py 慣例一致：避免 fp16 複數運算 NaN）
    with torch.amp.autocast("cuda", enabled=False):
        pv = pred_vort.float()
        gv = gt_vort.float()
        grid = _grain_grid(
            H, W, k_low, k_nyq, axis_excl_deg, n_theta, pv.device, pv.dtype
        )
        win = grid["win"]                       # [H,W]
        band_flat = grid["band_flat"]           # [H*W] bool
        nyq_flat = grid["nyq_flat"]             # [H*W] bool
        theta_bins_band = grid["theta_bins_band"]   # [M] long

        def _power(x: Tensor) -> Tensor:
            # 防呆：先乘 Hann window（殺中央假十字）；fftshift 對齊上方頻率座標
            f2 = torch.fft.fftshift(torch.fft.fft2(x[:, 0] * win), dim=(-2, -1))
            return (f2.real ** 2 + f2.imag ** 2).reshape(B, -1)   # [B, H*W]

        pp = _power(pv)
        pg = _power(gv)

        # --- 指紋 1：中高頻環 angular power P(θ)（正規化成分布後 L1）---
        # 正規化成方向分布 → 量級恆 bounded [0,2]（與場振幅、local-RMS 放大無關）。
        def _angular(p_flat: Tensor) -> Tensor:
            band_p = p_flat[:, band_flat]                          # [B, M]
            out = p_flat.new_zeros(B, n_theta)
            idx = theta_bins_band[None, :].expand(B, -1)           # [B, M]
            out.scatter_add_(1, idx, band_p)
            return out / (out.sum(1, keepdim=True) + eps)          # 正規化成方向分布

        l_aniso = (_angular(pp) - _angular(pg)).abs().mean()

        # --- 指紋 2：Nyquist hinge（只罰翹起）---
        # Nyquist hinge 用相對比值而非原始功率差：`clamp(ep − eg, 0)` 是未 bounded
        # 的原始功率差，配 local_rms_norm 時 freestream 低能量區 vort 會被局部 RMS
        # 放大，功率可衝到極高量級，使 hinge 值與梯度一併爆量。改成**相對比值**
        # （除以 GT band 總功率 + eps）：
        #   l_nyq = relu(E_pred − E_gt) / (E_band_gt + eps)
        # 量級回到 O(1)、去 amplitude/local-RMS 放大敏感，與 l_aniso 同尺度，不再爆。
        ep = pp[:, nyq_flat].mean(1)                               # [B] pred Nyquist 帶能量
        eg = pg[:, nyq_flat].mean(1)                               # [B] GT Nyquist 帶能量
        # GT 中高頻環總功率作 scale（band 為 grain 主體區，反映該幀整體能量基準）。
        # ★detach：scale 為「量級正規化」非可微路徑（GT 端本就 no-grad，pred 端不應走分母）。
        band_scale = pg[:, band_flat].mean(1).detach().clamp_min(eps)  # [B]
        l_nyq = (torch.clamp(ep - eg, min=0.0) / band_scale).mean()

        loss = w_aniso * l_aniso + w_nyq * l_nyq
        # ★最終 finite guard：任何殘留 NaN/Inf（FFT 邊界、極端 denorm）→ 回 0 不污染反傳。
        if not torch.isfinite(loss):
            loss = pp.new_zeros(())
    return loss.float()
