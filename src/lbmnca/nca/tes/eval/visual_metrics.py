"""視覺正確性評估指標模組 — 論文 visual correctness 裁決工具。

報告 3.2 節定義的視覺指標，離線評估用。

**這些函式本身不被 optimizer 呼叫，但它們量的量正是 canonical 被訓練去最小化的量**
（本檔至少四項指標與 canonical 的訓練損失是同一泛函，最明確的是 column banding：
`loss/terms/struct_banding.py` 的 docstring 自陳是本檔 banding 指標的可微訓練版；
vel/vort SSIM 亦同核心、僅值域不同）。因此它們不可作為「獨立驗收」，尤其不可用來比較
「有訓該項」與「沒訓該項」的兩個模型。
所有指標在反正規化 raw space 計算，使用 FieldColorizer 固定視覺範圍。

指標（higher=better 除非標注）：
    StructVel_RGB_SSIM      — 固定範圍速度場結構相似度
    StructVort_RGB_SSIM     — 固定範圍渦量場結構相似度
    StructWake_Region_Mae   — wake 區域速度 MAE（lower=better）
    StructVort_HighMask_IoU — 高渦度區位置 IoU
    IC_Copy_Score           — IC-copy 診斷（1=完全 IC-copy，0=有演化）
    HorizontalBand_Score    — 水平條紋診斷（舊版 row_std/anisotropy，保留對比用）
    HorizontalBand_FFT      — 水平條紋診斷（2D FFT，kx=0 超額能量比，主要指標）
    HF_Excess_Ratio         — 高頻超額能量比（棋盤格／grain 偵測器，lower=better）
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ...utils.FieldNormalizer import FieldNormalizer
from .pod_metrics import compute_pod_metrics

_RHO_CH: int = 0
_JX_CH: int = 3
_JY_CH: int = 5
_EPS: float = 1e-6

# 固定視覺範圍 = stats.json `vis_range` SSOT，經 normalizer.vel_vis_max / vor_vis_range 取得
# （見各函式；不再用模組級硬編碼常數）。

# torch.quantile 有 ~16M 元素上限；大張量先隨機子採樣（不改變分位數估計的統計意義）
_QUANTILE_MAX_ELEMS: int = 8_000_000


def _quantile_safe(values: Tensor, q: float) -> float:
    """對超大張量安全計算分位數，避開 torch.quantile 元素上限。

    Args:
        values: 輸入張量（任意形狀）。
        q: 分位數，範圍 [0, 1]。

    Returns:
        分位數純量值。
    """
    flat = values.flatten()
    if flat.numel() > _QUANTILE_MAX_ELEMS:
        idx = torch.randint(0, flat.numel(), (_QUANTILE_MAX_ELEMS,), device=flat.device)
        flat = flat[idx]
    return float(torch.quantile(flat, q).item())


# ---------------------------------------------------------------------------
# 反正規化工具
# ---------------------------------------------------------------------------


def _denorm_vel_mag(x_norm: Tensor, normalizer: FieldNormalizer) -> Tensor:
    """將正規化場反正規化並計算速度大小。

    Args:
        x_norm: 正規化輸入場，形狀 [B, 9, H, W]。
        normalizer: 場正規化器。

    Returns:
        速度大小場，形狀 [B, H, W]，單位為 lattice units。
    """
    raw = normalizer.denorm_mphy(x_norm)
    rho = raw[:, _RHO_CH].clamp_min(_EPS)
    return ((raw[:, _JX_CH] / rho).square() + (raw[:, _JY_CH] / rho).square()).sqrt()


def _denorm_vorticity(x_norm: Tensor, normalizer: FieldNormalizer) -> Tensor:
    """將正規化場反正規化並計算渦量（ω = dv/dx - du/dy）。

    Args:
        x_norm: 正規化輸入場，形狀 [B, 9, H, W]。
        normalizer: 場正規化器。

    Returns:
        渦量場，形狀 [B, H, W]，單位為 lattice units。
    """
    raw = normalizer.denorm_mphy(x_norm)
    rho = raw[:, _RHO_CH].clamp_min(_EPS)
    vel_x = raw[:, _JX_CH] / rho
    vel_y = raw[:, _JY_CH] / rho
    # 中央差分：ω = dv/dx - du/dy
    dvdx = F.pad(vel_y, (1, 1, 0, 0))
    dvdx = (dvdx[:, :, 2:] - dvdx[:, :, :-2]) * 0.5
    dudy = F.pad(vel_x, (0, 0, 1, 1))
    dudy = (dudy[:, 2:, :] - dudy[:, :-2, :]) * 0.5
    return dvdx - dudy


# ---------------------------------------------------------------------------
# SSIM 工具
# ---------------------------------------------------------------------------


def _ssim_scalar(
    pred: Tensor,
    target: Tensor,
    window_size: int = 11,
    mask: Optional[Tensor] = None,
) -> float:
    """計算兩個場的平均結構相似度（SSIM）。

    加 ``mask`` 支援。SSIM map 在 ``mask>0.5`` 像素上平均（fluid-only），
    排除 solid 區（建物內部 + 邊界不連續）對 SSIM 的拖累 → 跨 case 公平、更貼 fluid 真實
    （fluid-only 比 full-field 高 ~0.04，solid 多的 case 拖累更大）。mask=None → 全場平均（向後相容）。

    Args:
        pred: 預測場，形狀 [N, H, W] 或 [N, 1, H, W]，值域 [0, 1]。
        target: 目標場，形狀同 pred。
        window_size: 高斯窗口大小（預設 11）。
        mask: 可選 fluid mask [N, H, W] 或 [N, 1, H, W]（1=fluid）；給定則只在 fluid 像素平均。

    Returns:
        平均 SSIM 純量（higher=better）。
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    C1, C2 = 1e-4, 9e-4
    ws = window_size
    sigma = 1.5
    coords = torch.arange(ws, device=pred.device, dtype=pred.dtype) - ws // 2
    gauss = torch.exp(-coords.square() / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window = (gauss.unsqueeze(1) * gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    pad = ws // 2

    mu1 = F.conv2d(pred, window, padding=pad)
    mu2 = F.conv2d(target, window, padding=pad)
    # 變異數用 E[x²]−E[x]² 相減式計算，在**近乎常數**的區域會發生
    #   災難性抵消而得到微小負值。此時分母 (σ1²+σ2²+C2) 會小於 C2，
    #   整個比值就能超過 1——SSIM 在數學上有界於 [0,1]，超過就是數值假象。
    #   渦度經固定範圍映射後全場擠在中點附近，正是 σ²→0 的區域，故該指標尤其
    #   容易觸發此假象；與 loss 端 `vort_ssim` 塌成 0 是**同一個根因**的兩種表現。
    #   標準 SSIM 實作都會把變異數夾到 ≥0；這裡補上。
    sigma1_sq = (F.conv2d(pred * pred, window, padding=pad) - mu1.square()).clamp_min(0.0)
    sigma2_sq = (F.conv2d(target * target, window, padding=pad) - mu2.square()).clamp_min(0.0)
    sigma12 = F.conv2d(pred * target, window, padding=pad) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1.square() + mu2.square() + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        m = (mask > 0.5).to(ssim_map.dtype)
        denom = m.sum().clamp_min(1.0)
        return float((ssim_map * m).sum().item() / denom.item())
    return float(ssim_map.mean().item())


# ---------------------------------------------------------------------------
# 主要 visual gate 指標
# ---------------------------------------------------------------------------


@torch.inference_mode()
def struct_vel_rgb_ssim(
    pred_norm: Tensor,
    target_norm: Tensor,
    normalizer: FieldNormalizer,
    fluid_mask: Optional[Tensor] = None,
) -> float:
    """StructVel_RGB_SSIM：固定範圍速度場結構相似度。

    速度大小正規化至 [0, VEL_MAX] 後計算 SSIM，對應 FieldColorizer 視覺範圍。
    ★給 fluid_mask 則只在 fluid 像素平均（排除 solid 拖累，跨 case 公平、更貼 wake 真實）。

    Args:
        pred_norm: 預測場（最終步），形狀 [B, 9, H, W]，已正規化。
        target_norm: 目標場，形狀 [B, 9, H, W]，已正規化。
        normalizer: 場正規化器。
        fluid_mask: 可選流體遮罩 [B, 1, H, W] 或 [B, H, W]（1=fluid）。

    Returns:
        SSIM 純量（higher=better）。
    """
    _vel_max = normalizer.vel_vis_max
    vm_pred = (_denorm_vel_mag(pred_norm, normalizer) / _vel_max).clamp(0.0, 1.0)
    vm_tgt = (_denorm_vel_mag(target_norm, normalizer) / _vel_max).clamp(0.0, 1.0)
    _m = fluid_mask.squeeze(1) if (fluid_mask is not None and fluid_mask.dim() == 4) else fluid_mask
    return _ssim_scalar(vm_pred, vm_tgt, mask=_m)


@torch.inference_mode()
def struct_vort_rgb_ssim(
    pred_norm: Tensor,
    target_norm: Tensor,
    normalizer: FieldNormalizer,
    fluid_mask: Optional[Tensor] = None,
    wake_only: bool = True,
    percentile: float = 85.0,
) -> float:
    """StructVort_RGB_SSIM：固定範圍渦量場結構相似度（**限定高渦量區**）。

    渦量正規化至 ``[-vor_range, vor_range]`` 後計算 SSIM，對應 FieldColorizer 視覺範圍。

    ``wake_only`` 預設為 True：全流體區平均 SSIM 對渦量這種稀疏場不敏感——大部分
    像素在 pred 與 target 兩邊都 ≈0，局部 SSIM ≈1，整體平均被這片近零背景灌到接近
    1，尾流本身的差異被稀釋掉；限定在高渦量區平均能大幅拉開動態範圍，更貼近渦量場
    的實際判別力。

    本軸**對棋盤格仍不敏感**。高頻偽影請看 ``vort_topk_mae``。

    Args:
        pred_norm: 預測場（最終步），形狀 [B, 9, H, W]，已正規化。
        target_norm: 目標場，形狀同上。
        normalizer: 場正規化器。
        fluid_mask: 可選流體遮罩 [B, 1, H, W] 或 [B, H, W]（1=fluid）。
        wake_only: True → 只在 target 的 |ω| 前 (100−percentile)% 像素上平均。
        percentile: 高渦量區的分位門檻（wake_only=True 時有效）。

    Returns:
        SSIM 純量（higher=better）。
    """
    _vor_range = normalizer.vor_vis_range
    vp = _denorm_vorticity(pred_norm, normalizer)
    vt = _denorm_vorticity(target_norm, normalizer)
    to01 = lambda v: ((v + _vor_range) / (2 * _vor_range)).clamp(0.0, 1.0)
    vort_pred, vort_tgt = to01(vp), to01(vt)
    _m = fluid_mask
    if _m is not None and _m.dim() == 4:
        _m = _m.squeeze(1)
    if wake_only:
        base = (_m > 0.5) if _m is not None else torch.ones_like(vt, dtype=torch.bool)
        fvals = vt.abs()[base]
        if fvals.numel() >= 2:
            thr = _quantile_safe(fvals, percentile / 100.0)
            _m = (base & (vt.abs() > thr)).float()
    return _ssim_scalar(vort_pred, vort_tgt, mask=_m)


def struct_wake_region_mae(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    percentile: float = 85.0,
) -> float:
    """StructWake_Region_Mae：**高活躍區**速度大小 MAE（非動量虧損尾流）。

    **名實不符**：本函式的「wake 區域」定義為「最快 15% ∪ 最高渦量 15%」的聯集，
    僅涵蓋約 28% 網格、只包住約 59.5% 的真實回流區。它量的是**高速與高渦量的活躍區**，
    不是流體力學意義下的尾流（動量虧損區）。真正的尾流量測見
    `tes/eval/physics_gate.py` 的回流面積比與再接著長度。引用本指標時不得稱之為
    「尾流精度」。
    metric key `wake_mae` 刻意不改名：既有落盤的證據 JSON 皆以此鍵存取，改鍵會使
    既有證據與程式碼脫節。

    Wake mask 定義為 target 速度大小或渦量超過 percentile 門檻的流體像素。

    Args:
        pred_norm: 預測場（最終步），形狀 [B, 9, H, W]，已正規化。
        target_norm: 目標場，形狀 [B, 9, H, W]，已正規化。
        fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，float。
        normalizer: 場正規化器。
        percentile: 定義 wake mask 的分位數門檻（預設 85）。

    Returns:
        wake 區域速度 MAE 純量（lower=better）；流體像素不足時回傳 NaN。
    """
    vm_pred = _denorm_vel_mag(pred_norm, normalizer)   # [B, H, W]
    vm_tgt = _denorm_vel_mag(target_norm, normalizer)
    vort_tgt = _denorm_vorticity(target_norm, normalizer).abs()

    fm = fluid_mask.squeeze(1)  # [B, H, W]
    fluid_vals_vm = vm_tgt[fm > 0.5]
    fluid_vals_vo = vort_tgt[fm > 0.5]
    if fluid_vals_vm.numel() < 2:
        return float("nan")

    q_vel = _quantile_safe(fluid_vals_vm, percentile / 100.0)
    q_vor = _quantile_safe(fluid_vals_vo, percentile / 100.0)
    wake_mask = fm * ((vm_tgt > q_vel) | (vort_tgt > q_vor)).float()

    n = wake_mask.sum().clamp_min(1.0)
    return float(((vm_pred - vm_tgt).abs() * wake_mask).sum().item() / n.item())


@torch.inference_mode()
def struct_vort_highmask_iou(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    percentile: float = 85.0,
    abs_floor_frac: float = 0.0,
) -> float:
    """StructVort_HighMask_IoU：預測與目標高渦度遮罩之 IoU。

    Args:
        pred_norm: 預測場（最終步），形狀 [B, 9, H, W]，已正規化。
        target_norm: 目標場，形狀 [B, 9, H, W]，已正規化。
        fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，float。
        normalizer: 場正規化器。
        percentile: high-mask 取 target |vort| 的此 percentile 為門檻（85=legacy top-15%；
            97/99=vortex core，過濾弱渦量/邊界 pixel）。
        abs_floor_frac: 絕對下限（VOR_RANGE 的倍數）。最終門檻 = max(percentile_thr,
            abs_floor_frac * VOR_RANGE)。near-zero 渦量場（早期 cold）其 percentile_thr
            極低 → 被 floor 抬高 → mask 近空 → 回 NaN，避免量測 noise 位置。0.0=不套 floor（legacy）。

    Returns:
        IoU 純量（higher=better）；流體像素不足或 target mask 近空時回傳 NaN。
    """
    fm = fluid_mask.squeeze(1)  # [B, H, W]
    vort_pred = _denorm_vorticity(pred_norm, normalizer).abs()
    vort_tgt = _denorm_vorticity(target_norm, normalizer).abs()

    fluid_tgt_vals = vort_tgt[fm > 0.5]
    if fluid_tgt_vals.numel() < 2:
        return float("nan")
    thresh = _quantile_safe(fluid_tgt_vals, percentile / 100.0)
    if abs_floor_frac > 0.0:
        thresh = max(thresh, abs_floor_frac * normalizer.vor_vis_range)

    tgt_high = ((vort_tgt > thresh) & (fm > 0.5)).float()
    # absolute-floor guard：target high-mask 太小（近-noise 場）→ 不計分（回 NaN）
    if abs_floor_frac > 0.0 and float(tgt_high.sum().item()) < 1.0:
        return float("nan")
    pred_high = ((vort_pred > thresh) & (fm > 0.5)).float()

    intersection = (pred_high * tgt_high).sum()
    union = ((pred_high + tgt_high).clamp_max(1.0)).sum()
    return float((intersection / union.clamp_min(1.0)).item())


@torch.inference_mode()
def vort_core_mae(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    percentile: float = 99.0,
    abs_floor_frac: float = 0.0,
) -> float:
    """VortCore_MAE：target vortex-core 像素上的有符號渦量 MAE。

    與 vort_topk_mae 相似，但以 percentile + 可選 abs_floor 定義 core，
    供 vortex-core gate 使用（高 percentile 聚焦真正 vortex core）。

    Args:
        pred_norm: 預測場（最終步），形狀 [B, 9, H, W]，已正規化。
        target_norm: 目標場，形狀 [B, 9, H, W]，已正規化。
        fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，float。
        normalizer: 場正規化器。
        percentile: 定義 vortex core 的 |vort| 分位數門檻（預設 99）。
        abs_floor_frac: 絕對下限倍數（VOR_RANGE 的倍數），0.0=不套（legacy）。

    Returns:
        有符號渦量 MAE 純量（lower=better）；core 近空時回傳 NaN。
    """
    vort_pred = _denorm_vorticity(pred_norm, normalizer)
    vort_tgt = _denorm_vorticity(target_norm, normalizer)
    fm = fluid_mask.squeeze(1)
    abs_tgt = vort_tgt.abs()
    fvals = abs_tgt[fm > 0.5]
    if fvals.numel() < 2:
        return float("nan")
    thresh = _quantile_safe(fvals, percentile / 100.0)
    if abs_floor_frac > 0.0:
        thresh = max(thresh, abs_floor_frac * normalizer.vor_vis_range)
    mask = fm * (abs_tgt >= thresh).float()
    n = float(mask.sum().item())
    if n < 1.0:
        return float("nan")
    return float(((vort_pred - vort_tgt).abs() * mask).sum().item() / n)


@torch.inference_mode()
def vort_core_signed_corr(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    percentile: float = 99.0,
    abs_floor_frac: float = 0.0,
) -> float:
    """VortCore_SignedCorr：target vortex-core 像素上的有符號渦量 Pearson 相關係數。

    抓取 core 的正負渦相位，core 近空或像素 <2 時回傳 NaN。

    Args:
        pred_norm: 預測場（最終步），形狀 [B, 9, H, W]，已正規化。
        target_norm: 目標場，形狀 [B, 9, H, W]，已正規化。
        fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，float。
        normalizer: 場正規化器。
        percentile: 定義 vortex core 的 |vort| 分位數門檻（預設 99）。
        abs_floor_frac: 絕對下限倍數（VOR_RANGE 的倍數），0.0=不套（legacy）。

    Returns:
        Pearson 相關係數純量（higher=better）；core 近空時回傳 NaN。
    """
    vort_pred = _denorm_vorticity(pred_norm, normalizer)
    vort_tgt = _denorm_vorticity(target_norm, normalizer)
    fm = fluid_mask.squeeze(1)
    abs_tgt = vort_tgt.abs()
    fvals = abs_tgt[fm > 0.5]
    if fvals.numel() < 2:
        return float("nan")
    thresh = _quantile_safe(fvals, percentile / 100.0)
    if abs_floor_frac > 0.0:
        thresh = max(thresh, abs_floor_frac * normalizer.vor_vis_range)
    region = (fm > 0.5) & (abs_tgt >= thresh)
    p = vort_pred[region]
    t = vort_tgt[region]
    if p.numel() < 2:
        return float("nan")
    pc = p - p.mean()
    tc = t - t.mean()
    denom = (pc.square().sum().sqrt() * tc.square().sum().sqrt()).clamp_min(1e-8)
    return float(((pc * tc).sum() / denom).item())


@torch.inference_mode()
def vort_topk_mae(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    topk_pct: float = 7.5,
) -> float:
    """VortTopK_MAE：target |vorticity| 前 k% 流體像素上的有符號渦量 MAE。

    嚴格 top-k of target |vort|（不混 velocity，與 wake 版區隔），
    解 full-domain mean 被背景近零渦量稀釋問題。

    Args:
        pred_norm: 預測場（最終步），形狀 [B, 9, H, W]，已正規化。
        target_norm: 目標場，形狀 [B, 9, H, W]，已正規化。
        fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，float。
        normalizer: 場正規化器。
        topk_pct: 取前 k% |vort| 像素（預設 7.5%）。

    Returns:
        有符號渦量 MAE 純量（lower=better）；流體像素不足時回傳 NaN。
    """
    vort_pred = _denorm_vorticity(pred_norm, normalizer)
    vort_tgt = _denorm_vorticity(target_norm, normalizer)
    fm = fluid_mask.squeeze(1)
    abs_tgt = vort_tgt.abs()
    fluid_vals = abs_tgt[fm > 0.5]
    if fluid_vals.numel() < 2:
        return float("nan")
    thresh = _quantile_safe(fluid_vals, 1.0 - topk_pct / 100.0)
    mask = fm * (abs_tgt >= thresh).float()
    n = mask.sum().clamp_min(1.0)
    return float(((vort_pred - vort_tgt).abs() * mask).sum().item() / n.item())


@torch.inference_mode()
def vort_signed_corr(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    wake_only: bool = True,
    percentile: float = 85.0,
    erode_solid: int = 3,
) -> float:
    """VortSignedCorr：wake（或全流體）區域的有符號渦量 Pearson 相關係數。

    抓正負渦的位置/相位（vortex core phase error），POD 對此不敏感，
    故獨立列入。

    Args:
        pred_norm: 預測場（最終步），形狀 [B, 9, H, W]，已正規化。
        target_norm: 目標場，形狀 [B, 9, H, W]，已正規化。
        fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，float。
        normalizer: 場正規化器。
        wake_only: 若 True，僅在 wake 區域（top percentile |vort|）計算；
            若 False，在全流體區域計算。
        percentile: 定義 wake mask 的 |vort| 分位數（wake_only=True 時有效，預設 85）。

    Returns:
        Pearson 相關係數純量（higher=better）；區域像素 <2 時回傳 NaN。
    """
    vort_pred = _denorm_vorticity(pred_norm, normalizer)
    vort_tgt = _denorm_vorticity(target_norm, normalizer)
    fm = fluid_mask.squeeze(1)
    # 侵蝕固體邊界。渦量用中央差分算，固體邊緣那一圈**必定**有
    #   不連續造成的假象，而 pred 與 target 在同樣的邊緣上有同號假象 →
    #   兩個場即使內部毫無關係也會被算出高相關。
    #   驗證：把 GT 換成「空間常數場」（渦量的真值恆為 0）仍得 corr=+0.8431；
    #   侵蝕 k=3 後降為 **恰好 0.0000**，而 GT 自比仍是 1.0000，只損失 1.3% 流體像素。
    if erode_solid and erode_solid > 1:
        k = int(erode_solid)
        keep = F.avg_pool2d(fm.unsqueeze(1), k, stride=1, padding=k // 2) > 0.999
        fm = (fm > 0.5).float() * keep.squeeze(1).float()
    if wake_only:
        abs_tgt = vort_tgt.abs()
        fvals = abs_tgt[fm > 0.5]
        if fvals.numel() < 2:
            return float("nan")
        thr = _quantile_safe(fvals, percentile / 100.0)
        region = (fm > 0.5) & (abs_tgt > thr)
    else:
        region = fm > 0.5
    p = vort_pred[region]
    t = vort_tgt[region]
    if p.numel() < 2:
        return float("nan")
    pc = p - p.mean()
    tc = t - t.mean()
    denom = (pc.square().sum().sqrt() * tc.square().sum().sqrt()).clamp_min(1e-8)
    return float(((pc * tc).sum() / denom).item())


@torch.inference_mode()
def ic_copy_score(
    pred_norm: Tensor,
    ic_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
) -> float:
    """IC-Copy Score：量化預測場與初始條件之間的演化幅度比值。

    score = ||pred - ic|| / (||target - ic|| + eps)

    - 接近 0 → 預測幾乎未從 IC 演化（IC-copy 失敗）
    - 接近 1 → 預測演化幅度與目標相當（正常）
    - 大於 1 → 預測過度演化或發散

    FAIL threshold：< 0.40（嚴重欠演化）
    PASS threshold：> 0.70（充分演化）

    Args:
        pred_norm: 預測場（最終步），形狀 [B, 9, H, W]，已正規化。
        ic_norm: 初始條件場（t=0），形狀 [B, 9, H, W]，已正規化。
        target_norm: 目標場，形狀 [B, 9, H, W]，已正規化。
        fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，float。

    Returns:
        IC-copy score 純量。
    """
    fm = fluid_mask
    diff_pred = ((pred_norm - ic_norm).abs() * fm).sum()
    diff_tgt = ((target_norm - ic_norm).abs() * fm).sum()
    return float((diff_pred / diff_tgt.clamp_min(1e-8)).item())


@torch.inference_mode()
def shear_peak_ratio(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    q: float = 0.99,
) -> float:
    """剪切層峰值比：預測的剪切強度峰值相對真值之比。

    剪切強度取 ``|du/dy|``（主流方向速度的橫向梯度），峰值取流體區內的
    第 `q` 分位數而非最大值——單點極值對數值雜訊極敏感，一個壞像素就能
    讓比值失真。

    Returns:
        ``peak(pred) / peak(target)``。1.0＝剪切層強度正確；
        <1 表示模型把剪切層抹平（過度耗散），>1 表示過銳。

    Args:
        pred_norm: 預測場 [B, 9, H, W]，已正規化。
        target_norm: 目標場 [B, 9, H, W]，已正規化。
        fluid_mask: 流體遮罩 [B, 1, H, W]。
        normalizer: 場正規化器（用於反正規化）。
        q: 峰值取的分位數，預設 0.99。
    """
    def _peak(x: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x)
        rho = raw[:, _RHO_CH].clamp_min(_EPS)
        u = raw[:, _JX_CH] / rho
        dudy = F.pad(u, (0, 0, 1, 1))
        dudy = (dudy[:, 2:, :] - dudy[:, :-2, :]) * 0.5
        m = fluid_mask[:, 0] > 0.5
        vals = dudy.abs()[m]
        return vals.new_tensor(_quantile_safe(vals, q)) if vals.numel() else vals.new_tensor(0.0)

    p_pk, t_pk = _peak(pred_norm), _peak(target_norm)
    return float((p_pk / t_pk.clamp_min(_EPS)).item())


@torch.inference_mode()
def hf_excess_ratio(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    k_frac: float = 0.20,
) -> float:
    """HF_Excess_Ratio：pred 相對 target 的**高頻超額能量比**（棋盤格／grain 偵測器）。

    既有的九項視覺指標對棋盤格類偽影普遍不敏感——高頻能量集中在 Nyquist 附近，
    而既有指標多半在較低頻段或整場平均上取值，容易被稀釋。本指標直接量測高頻
    能量占比，且與**合法**的退化（模糊、幅值縮放、場歸零等）分得開，避免低分
    「分不出」乾淨但錯與有偽影兩種情形。

    本指標取 2D FFT 中 ``|k| > k_frac × Nyquist`` 的功率占比，回傳
    ``clamp(hf_ratio(pred) − hf_ratio(target), min=0)``。
    棋盤格是純 Nyquist → 直接爆；模糊只會讓高頻**變少**（clamp 後為 0）。

    本指標**不抓中低頻條紋**，那不是它的職責；水平條紋看
    ``horizontal_band_score_fft``。

    Args:
        pred_norm: 預測場，形狀 [B, 9, H, W]，已正規化。
        target_norm: 目標場，形狀同上。
        fluid_mask: 流體遮罩 [B, 1, H, W]。
        normalizer: 場正規化器。
        k_frac: 高頻起算點，以 Nyquist 為單位。預設 **0.20**（週期 <10 px）由掃描
            區間定出：取太靠近 Nyquist（如 0.50）會漏掉較長週期的條紋類偽影；
            0.20 在能分離合法退化與偽影的同時，語義也清楚——週期 <10 px 是偽影
            尺度，真實尾流結構遠大於此。

    Returns:
        高頻超額能量比（0＝無超額，higher=worse）。
    """
    m = fluid_mask if fluid_mask.dim() == 4 else fluid_mask.unsqueeze(1)
    vp = _denorm_vel_mag(pred_norm, normalizer) * m.squeeze(1)
    vt = _denorm_vel_mag(target_norm, normalizer) * m.squeeze(1)

    def _hf(v: Tensor) -> Tensor:
        spec = torch.fft.rfft2(v)
        pw = spec.real.square() + spec.imag.square()
        H, W2 = pw.shape[-2], pw.shape[-1]
        ky = torch.fft.fftfreq(H, device=v.device).abs().view(-1, 1)
        kx = torch.fft.rfftfreq((W2 - 1) * 2, device=v.device).abs().view(1, -1)
        hi = ((ky.square() + kx.square()).sqrt() > k_frac * 0.5).to(pw.dtype)
        return (pw * hi).sum(dim=(-2, -1)) / pw.sum(dim=(-2, -1)).clamp_min(1e-20)

    return float((_hf(vp) - _hf(vt)).clamp_min(0.0).mean().item())


@torch.inference_mode()
def horizontal_band_score(
    pred_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    band_threshold: float = 3.0,
) -> float:
    """Horizontal-Band Score：以 row_std/anisotropy 診斷速度場水平條紋失效模式。

    結合兩個訊號：
    1. Relative：行 std 低於 median_std / band_threshold 的比例（原始版本）。
    2. Anisotropy：row_std << col_std → 即使全域平坦也能偵測水平條紋。
       anisotropy_score = 1 - clamp(mean_row_std / (mean_col_std * 0.4), 0, 1)

    Final score = max(relative, anisotropy)，高值（>0.40）代表條紋失效。

    此版本修正 v1 局限：當所有行均勻低 std（全域條紋）時，
    relative threshold median/3 也極低，原版指標失效。

    Args:
        pred_norm: 預測場（最終步），形狀 [B, 9, H, W]，已正規化。
        fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，float。
        normalizer: 場正規化器。
        band_threshold: 定義「低 std 行」的門檻倍數（預設 3.0）。

    Returns:
        水平條紋診斷分數（high=banded failure）；無有效行時回傳 NaN。
    """
    vm = _denorm_vel_mag(pred_norm, normalizer)  # [B, H, W]
    fm = fluid_mask.squeeze(1)

    scores: list[float] = []
    for b in range(vm.shape[0]):
        field = vm[b] * fm[b]  # mask solid
        fluid_rows = (fm[b].sum(dim=-1) > 5).float()
        fluid_cols = (fm[b].sum(dim=-2) > 5).float()
        if fluid_rows.sum() < 5:
            continue

        row_std = field.std(dim=-1)   # 沿 x 方向的行內標準差
        col_std = field.std(dim=-2)   # 沿 y 方向的列內標準差

        valid_row_stds = row_std[fluid_rows > 0.5]
        median_std = float(valid_row_stds.median().item())
        band_rows = (valid_row_stds < median_std / band_threshold).float().sum()
        relative_score = float(band_rows.item() / valid_row_stds.numel())

        # anisotropy：水平條紋 → 行 x 方向變異低，列 y 方向變異高
        mean_row_std = float(valid_row_stds.mean().item())
        if fluid_cols.sum() > 0:
            mean_col_std = float(col_std[fluid_cols > 0.5].mean().item())
        else:
            mean_col_std = mean_row_std
        # ratio < 0.4 代表行變異不到列變異的 40% → banded
        anisotropy_ratio = mean_row_std / (mean_col_std + _EPS)
        anisotropy_score = float(max(0.0, 1.0 - anisotropy_ratio / 0.4))

        scores.append(max(relative_score, anisotropy_score))

    return float(np.mean(scores)) if scores else float("nan")


@torch.inference_mode()
def horizontal_band_score_fft(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
) -> float:
    """2D FFT 水平條紋偵測器（pred vs target 超額能量比）。

    水平條紋 → 能量集中於 2D FFT 的 kx=0、ky≠0 位置。
    使用 pred-minus-target excess 排除物理上的水平結構。

    Score 解讀：
        ~0.00  : 無人工條紋
        >0.05  : 中等條紋 artifact
        >0.15  : 嚴重條紋失效

    相較 row_std 版本的優點：
    - 能偵測全域條紋（所有行均平坦時 row_std 給出假陰性）
    - target 相減可排除物理水平流場結構
    - 特定 ky 峰值可揭示條紋空間頻率

    Args:
        pred_norm: 預測場（最終步），形狀 [B, 9, H, W]，已正規化。
        target_norm: 目標場，形狀 [B, 9, H, W]，已正規化。
        fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，float。
        normalizer: 場正規化器。

    Returns:
        水平條紋超額能量比純量（higher=worse）。
    """
    pred_vm = _denorm_vel_mag(pred_norm, normalizer)    # [B, H, W]
    tgt_vm  = _denorm_vel_mag(target_norm, normalizer)  # [B, H, W]
    fm = fluid_mask.squeeze(1)                          # [B, H, W]

    def _horiz_ratio(field: Tensor) -> Tensor:
        """計算場的 kx=0 水平頻率能量佔比（排除 DC）。

        Args:
            field: 輸入速度大小場，形狀 [B, H, W]。

        Returns:
            各批次的水平頻率能量比，形狀 [B]。
        """
        # 流體區域均值中心化，固體區域設 0 以最小化邊緣 artifact
        fluid_sum = (field * fm).sum(dim=(-1, -2))
        fluid_cnt = fm.sum(dim=(-1, -2)).clamp_min(1.0)
        mean = (fluid_sum / fluid_cnt)[:, None, None]
        centered = (field - mean) * fm                  # [B, H, W]

        F_fft = torch.fft.rfft2(centered)               # [B, H, W//2+1]
        power = F_fft.abs().square()                    # [B, H, W//2+1]

        # kx=0 欄（kx index 0），排除 DC（ky=0, kx=0）
        horiz_power = power[:, 1:, 0].sum(dim=-1)       # [B]
        total_power = power.sum(dim=(-1, -2)) - power[:, 0, 0]  # [B] 非 DC
        return horiz_power / (total_power + _EPS)       # [B]

    pred_ratio = _horiz_ratio(pred_vm)   # [B]
    tgt_ratio  = _horiz_ratio(tgt_vm)    # [B]

    # 人工超額：pred 比 target 多出多少水平條紋能量
    excess = (pred_ratio - tgt_ratio).clamp(min=0.0)
    return float(excess.mean().item())


# ---------------------------------------------------------------------------
# 批次彙整
# ---------------------------------------------------------------------------


@torch.inference_mode()
def compute_visual_gate_metrics(
    pred_norm: Tensor,
    target_norm: Tensor,
    ic_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    pod_basis: dict[str, Tensor] | None = None,
    pod_modes: int | None = None,
) -> dict[str, float]:
    """計算一批最終步預測的全部 visual gate 指標。

    回傳的 key 與報告 §3.2 表格對應，所有值為純量 float。

    Args:
        pred_norm: 預測場（最終步），形狀 [B, 9, H, W]，已正規化。
        target_norm: 目標場，形狀 [B, 9, H, W]，已正規化。
        ic_norm: 初始條件場（t=0），形狀 [B, 9, H, W]，已正規化。
        fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，float。
        normalizer: 場正規化器。
        pod_basis: POD 基底字典（可選，None 則略過 POD 指標）。
        pod_modes: 要使用的 POD 模態數（可選）。

    Returns:
        指標名稱到純量值的字典。
    """
    metrics: dict[str, float] = {
        "StructVel_RGB_SSIM": struct_vel_rgb_ssim(pred_norm, target_norm, normalizer, fluid_mask),
        "StructVort_RGB_SSIM": struct_vort_rgb_ssim(pred_norm, target_norm, normalizer, fluid_mask),
        "StructWake_Region_Mae": struct_wake_region_mae(pred_norm, target_norm, fluid_mask, normalizer),
        "StructVort_HighMask_IoU": struct_vort_highmask_iou(pred_norm, target_norm, fluid_mask, normalizer),
        "VortTopK_MAE": vort_topk_mae(pred_norm, target_norm, fluid_mask, normalizer),
        "VortSignedCorr": vort_signed_corr(pred_norm, target_norm, fluid_mask, normalizer),
        "IC_Copy_Score": ic_copy_score(pred_norm, ic_norm, target_norm, fluid_mask),
        "HorizontalBand_Score": horizontal_band_score(pred_norm, fluid_mask, normalizer),
        "HorizontalBand_FFT": horizontal_band_score_fft(pred_norm, target_norm, fluid_mask, normalizer),
        # 補上既有九軸都不敏感的棋盤格／高頻偽影偵測。
        "HF_Excess_Ratio": hf_excess_ratio(pred_norm, target_norm, fluid_mask, normalizer),
    }
    if pod_basis is not None:
        metrics.update(
            compute_pod_metrics(
                pred_norm, target_norm, fluid_mask, normalizer, pod_basis, pod_modes
            )
        )
    return metrics
