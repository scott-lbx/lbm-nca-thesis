"""尾跡加權結構損失（動態計算尾跡遮罩）。

提取自 loss_terms_v2.py，保留所有原始邏輯。

命名規則：TargetScope_PhysicalMeaning_ErrorForm
  struct_wake_*  — wake 區域加權結構 loss
  struct_vort_*  — 渦量結構 loss（全 fluid 域）

公開 API：
  compute_wake_mask()       — 動態尾跡遮罩（helper）
  struct_wake_vel_mae()     — wake 區速度幅值 MAE
  struct_wake_vort_mae()    — wake 區渦量 MAE
  struct_vort_signed_huber() — 有符號渦量 Huber loss（全 fluid 域）

MRT channel indices（9ch physics tensor）：
  RHO_CH = 0  (m0: 密度)
  JX_CH  = 3  (m3: x 動量)
  JY_CH  = 5  (m5: y 動量)

mask 慣例：
  fluid_mask [B,1,H,W] float，流體=1 固體=0
"""
from __future__ import annotations

import torch
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

__all__ = [
    "compute_wake_mask",
    "struct_wake_vel_mae",
    "struct_wake_vort_mae",
    "struct_vort_signed_huber",
]


# ---------------------------------------------------------------------------
# Wake mask helper
# ---------------------------------------------------------------------------


def compute_wake_mask(
    target_norm: Tensor,       # [B, 9, H, W] or [B, K, 9, H, W] normalized
    fluid_mask: Tensor,        # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    vort_percentile: float = 85.0,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """Target-based wake mask：高渦度或高速度區域（fluid 內），避免背景稀釋尾流 loss 信號。

    以 target 場的速度幅值與渦量絕對值各取 ``vort_percentile`` 百分位門檻，
    union 後乘上 fluid_mask 得到尾流區遮罩。

    Args:
        target_norm: ``[B, 9, H, W]`` 或 ``[B, K, 9, H, W]`` 正規化目標場
            （5D 時取最終幀估算 wake）。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器。
        vort_percentile: 高渦度/高速度門檻百分位（預設 85%）。
        rho_eps: 密度下限夾值。

    Returns:
        ``[B, 1, H, W]`` float wake mask（高活躍區=1）。
    """
    if target_norm.dim() == 5:
        # 用最終幀估算 wake 分布
        t = target_norm[:, -1]   # [B, 9, H, W]
    else:
        t = target_norm          # [B, 9, H, W]

    raw = normalizer.denorm_mphy(t)  # [B, 9, H, W]
    rho = raw[:, RHO_CH].clamp_min(rho_eps)
    vel_x = raw[:, JX_CH] / rho
    vel_y = raw[:, JY_CH] / rho
    vel_mag = (vel_x.square() + vel_y.square()).sqrt()   # [B, H, W]
    vort = _ddx(vel_y) - _ddy(vel_x)                     # [B, H, W]

    fm = fluid_mask.squeeze(1)  # [B, H, W]
    fluid_vals_vel = vel_mag[fm > 0.5]
    fluid_vals_vor = vort.abs()[fm > 0.5]

    if fluid_vals_vel.numel() < 2:
        return fluid_mask

    q_vel = torch.quantile(fluid_vals_vel, vort_percentile / 100.0)
    q_vor = torch.quantile(fluid_vals_vor, vort_percentile / 100.0)

    high = ((vel_mag > q_vel) | (vort.abs() > q_vor)).float()  # [B, H, W]
    return (high * fm).unsqueeze(1)  # [B, 1, H, W]


# ---------------------------------------------------------------------------
# Wake-weighted 結構 loss
# ---------------------------------------------------------------------------


def struct_wake_vel_mae(
    pred_norm: Tensor,      # [B, K, 9, H, W] normalized
    target_norm: Tensor,    # [B, K, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """StructWake_VelMae：wake 區域速度幅值 MAE（raw lattice units）。

    只在 target 高渦度/高速度區計算，避免背景稀釋。

    Args:
        pred_norm: ``[B, K, 9, H, W]`` 正規化預測軌跡。
        target_norm: ``[B, K, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。

    Returns:
        scalar Tensor（wake 區速度幅值 MAE）。
    """
    B, K, C, H, W = pred_norm.shape
    wake_mask = compute_wake_mask(target_norm, fluid_mask, normalizer, rho_eps=rho_eps)

    def _vel_mag(x_norm: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W))
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        return (raw[:, JX_CH] / rho).square() + (raw[:, JY_CH] / rho).square()  # squared

    vm_pred = _vel_mag(pred_norm).sqrt()   # [B*K, H, W]
    vm_tgt = _vel_mag(target_norm).sqrt()

    mask_bk = wake_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    return ((vm_pred - vm_tgt).abs() * mask_bk).sum() / mask_bk.sum().clamp_min(1.0)


def struct_wake_vort_mae(
    pred_norm: Tensor,      # [B, K, 9, H, W] normalized
    target_norm: Tensor,    # [B, K, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """StructWake_VortMae：wake 區域渦量 MAE（raw lattice units）。

    Args:
        pred_norm: ``[B, K, 9, H, W]`` 正規化預測軌跡。
        target_norm: ``[B, K, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。

    Returns:
        scalar Tensor（wake 區渦量 MAE）。
    """
    B, K, C, H, W = pred_norm.shape
    wake_mask = compute_wake_mask(target_norm, fluid_mask, normalizer, rho_eps=rho_eps)

    def _vort(x_norm: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W))
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        vel_x = raw[:, JX_CH] / rho
        vel_y = raw[:, JY_CH] / rho
        return _ddx(vel_y) - _ddy(vel_x)  # [B*K, H, W]

    vort_pred = _vort(pred_norm)
    vort_tgt = _vort(target_norm)

    mask_bk = wake_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    return ((vort_pred - vort_tgt).abs() * mask_bk).sum() / mask_bk.sum().clamp_min(1.0)


def struct_vort_signed_huber(
    pred_norm: Tensor,      # [B, K, 9, H, W] normalized
    target_norm: Tensor,    # [B, K, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    delta: float = 0.01,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """StructVort_SignedHuber：fluid 區域有符號渦量 Huber loss。

    比純 MAE 更能懲罰符號錯誤的渦結構（正渦→負渦預測）。
    delta = 0.01 lattice units（約為 VOR_RANGE 的 25%）。

    Args:
        pred_norm: ``[B, K, 9, H, W]`` 正規化預測軌跡。
        target_norm: ``[B, K, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器。
        delta: Huber loss 轉換點（lattice units）。
        rho_eps: 密度下限夾值。

    Returns:
        scalar Tensor（有符號渦量 Huber loss）。
    """
    B, K, C, H, W = pred_norm.shape

    def _vort(x_norm: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W))
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        vel_x = raw[:, JX_CH] / rho
        vel_y = raw[:, JY_CH] / rho
        return _ddx(vel_y) - _ddy(vel_x)

    diff = _vort(pred_norm) - _vort(target_norm)  # [B*K, H, W] signed
    abs_diff = diff.abs()
    huber = torch.where(abs_diff <= delta, 0.5 * diff.square(), delta * (abs_diff - 0.5 * delta))

    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    return (huber * mask_bk).sum() / mask_bk.sum().clamp_min(1.0)
