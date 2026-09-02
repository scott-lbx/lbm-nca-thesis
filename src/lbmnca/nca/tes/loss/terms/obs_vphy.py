"""Vphy 速度空間觀測損失（需 denorm，M1A 延伸）。

M1A 核心修正：將 loss target 從 Mphy moment space 移至 Vphy velocity space，
直接監督視覺等價的物理量（vel_x, vel_y, vorticity）。

命名規則（與 loss_terms_v2 一致）：
  obs_vphy_vel_mae       — 全 fluid 區域 velocity MAE（主 loss）
  obs_vphy_delta_mae     — velocity 步差 MAE（演化方向監督）
  obs_vphy_wakevert_mae  — wake-weighted vorticity MAE（M1A_3 視覺精調用）

所有函式：
  - 輸入 pred/target 為 [B, K, 9, H, W] normalized Mphy
  - 強制 .float() 在 denorm 前，防止 fp16 catastrophic cancellation
  - 回傳 scalar Tensor（可微，支援 backward）

MRT channel indices（9ch physics tensor）：
  RHO_CH = 0  (m0: 密度)
  JX_CH  = 3  (m3: x 動量)
  JY_CH  = 5  (m5: y 動量)
"""
from __future__ import annotations

import torch
from torch import Tensor

from ....utils.FieldNormalizer import FieldNormalizer
from ...utils.spatial_ops import ddx as _ddx, ddy as _ddy

RHO_CH: int = 0
JX_CH: int = 3
JY_CH: int = 5

_RHO_EPS: float = 1e-6

# Sobel kernel normalizer: divide by 8 matches FieldConverter._sobel_curl convention
_SOBEL_DIV: float = 8.0


def _to_vel(
    mphy_norm: Tensor,
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> tuple[Tensor, Tensor]:
    """denorm → (vel_x, vel_y)。強制 fp32 防止 catastrophic cancellation。

    Args:
        mphy_norm: [N, 9, H, W] normalized Mphy tensor。
        normalizer: FieldNormalizer（denorm 用）。
        rho_eps: 密度 clamp 下限，防止除以零。

    Returns:
        (vel_x, vel_y)：各為 [N, H, W] fp32 速度分量。
    """
    raw = normalizer.denorm_mphy(mphy_norm.float())  # [N, 9, H, W] fp32
    rho = raw[:, RHO_CH].clamp_min(rho_eps)
    return raw[:, JX_CH] / rho, raw[:, JY_CH] / rho


def _to_vort(
    mphy_norm: Tensor,
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """denorm → vorticity（dvy/dx - dvx/dy）。強制 fp32。

    Args:
        mphy_norm: [N, 9, H, W] normalized Mphy tensor。
        normalizer: FieldNormalizer（denorm 用）。
        rho_eps: 密度 clamp 下限。

    Returns:
        vorticity：[N, H, W] fp32。
    """
    vel_x, vel_y = _to_vel(mphy_norm, normalizer, rho_eps)
    return _ddx(vel_y) - _ddy(vel_x)


def obs_vphy_vel_mae(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """ObsVphy_Vel_Mae：velocity field MAE（全 fluid 區域）。

    在 raw velocity space（vel_x, vel_y）計算 MAE，取代 Mphy moment MAE。
    視覺等價的 loss 函數，直接監督視覺等價的物理量。

    Args:
        pred_norm: [B, K, 9, H, W] normalized Mphy prediction。
        target_norm: [B, K, 9, H, W] normalized Mphy target。
        fluid_mask: [B, 1, H, W] float，流體=1 固體=0。
        normalizer: FieldNormalizer（denorm 用）。
        rho_eps: 密度 clamp 下限。

    Returns:
        scalar: mean(|vel_x_pred - vel_x_tgt| + |vel_y_pred - vel_y_tgt|) / 2，fluid only。
    """
    B, K, C, H, W = pred_norm.shape
    flat_pred = pred_norm.reshape(B * K, C, H, W)
    flat_tgt = target_norm.reshape(B * K, C, H, W)

    vx_pred, vy_pred = _to_vel(flat_pred, normalizer, rho_eps)
    vx_tgt, vy_tgt = _to_vel(flat_tgt, normalizer, rho_eps)

    diff = (vx_pred - vx_tgt).abs() + (vy_pred - vy_tgt).abs()   # [B*K, H, W]
    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    n = mask_bk.sum().clamp_min(1.0)
    return (diff * mask_bk).sum() / (2.0 * n)


def obs_vphy_delta_mae(
    pred_norm: Tensor,
    target_norm: Tensor,
    ic_physics_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """ObsVphy_Delta_Mae：velocity 步差 MAE（鼓勵正確演化方向）。

    定義：
      delta[0] = vel[0] - vel_ic
      delta[i] = vel[i] - vel[i-1]   (i >= 1)

    Args:
        pred_norm: [B, K, 9, H, W] normalized prediction trajectory。
        target_norm: [B, K, 9, H, W] normalized target trajectory。
        ic_physics_norm: [B, 9, H, W] IC physics（t=0 起點）。
        fluid_mask: [B, 1, H, W] float。
        normalizer: FieldNormalizer。
        rho_eps: 密度 clamp 下限。

    Returns:
        scalar: mean vel delta MAE，fluid only。
    """
    B, K, C, H, W = pred_norm.shape

    # [B, K+1, 9, H, W]：在 IC 前插一幀
    ic_t = ic_physics_norm.unsqueeze(1)
    pred_with_ic = torch.cat([ic_t, pred_norm], dim=1)
    tgt_with_ic = torch.cat([ic_t, target_norm], dim=1)

    # delta[i] = vel[i+1] - vel[i]
    BK1 = B * (K + 1)
    flat_p = pred_with_ic.reshape(BK1, C, H, W)
    flat_t = tgt_with_ic.reshape(BK1, C, H, W)

    vx_p, vy_p = _to_vel(flat_p, normalizer, rho_eps)   # [B*(K+1), H, W]
    vx_t, vy_t = _to_vel(flat_t, normalizer, rho_eps)

    vx_p = vx_p.reshape(B, K + 1, H, W)
    vy_p = vy_p.reshape(B, K + 1, H, W)
    vx_t = vx_t.reshape(B, K + 1, H, W)
    vy_t = vy_t.reshape(B, K + 1, H, W)

    dvx_pred = vx_p[:, 1:] - vx_p[:, :-1]   # [B, K, H, W]
    dvy_pred = vy_p[:, 1:] - vy_p[:, :-1]
    dvx_tgt = vx_t[:, 1:] - vx_t[:, :-1]
    dvy_tgt = vy_t[:, 1:] - vy_t[:, :-1]

    diff = (dvx_pred - dvx_tgt).abs() + (dvy_pred - dvy_tgt).abs()
    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W)
    n = mask_bk.sum().clamp_min(1.0)
    return (diff * mask_bk).sum() / (2.0 * n)


def obs_vphy_wakevert_mae(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    wake_weight: float = 10.0,
    wake_percentile: float = 85.0,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """ObsVphy_WakeVort_Mae：wake 區域加權渦量 MAE。

    Wake mask：target 渦度絕對值 top-(100-wake_percentile)% per sample。
    Wake 區域權重 = wake_weight；其他 fluid 區域權重 = 1.0。

    M1A_3 視覺精調才啟用，M1A_0/1/2 不使用。

    Args:
        pred_norm: [B, K, 9, H, W] normalized。
        target_norm: [B, K, 9, H, W] normalized。
        fluid_mask: [B, 1, H, W]。
        normalizer: FieldNormalizer。
        wake_weight: wake 區域損失倍率（預設 10×）。
        wake_percentile: 超過此分位數的渦度視為 wake（預設 85th）。
        rho_eps: 密度 clamp 下限。

    Returns:
        scalar: wake-weighted vorticity MAE，fluid only。
    """
    B, K, C, H, W = pred_norm.shape
    flat_pred = pred_norm.reshape(B * K, C, H, W)
    flat_tgt = target_norm.reshape(B * K, C, H, W)

    vort_pred = _to_vort(flat_pred, normalizer, rho_eps)   # [B*K, H, W]
    vort_tgt = _to_vort(flat_tgt, normalizer, rho_eps)

    # Wake mask：target 渦度 top-15% per sample (B*K independent)
    fm_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    abs_tgt = vort_tgt.abs()
    # compute per-sample threshold on fluid pixels only
    thresholds = []
    for b in range(B * K):
        fluid_vals = abs_tgt[b][fm_bk[b] > 0.5]
        if fluid_vals.numel() >= 2:
            q = torch.quantile(fluid_vals, wake_percentile / 100.0)
        else:
            q = abs_tgt[b].max()
        thresholds.append(q)
    thr = torch.stack(thresholds).to(vort_tgt.device)[:, None, None]  # [B*K, 1, 1]
    wake_mask = (abs_tgt > thr).float() * fm_bk   # [B*K, H, W]

    weight_map = 1.0 + wake_mask * (wake_weight - 1.0)
    diff = (vort_pred - vort_tgt).abs()
    n = (fm_bk * weight_map).sum().clamp_min(1.0)
    return (diff * weight_map * fm_bk).sum() / n
