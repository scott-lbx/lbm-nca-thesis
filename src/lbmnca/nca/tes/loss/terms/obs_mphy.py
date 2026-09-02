"""Mphy 觀測空間 MAE 損失（normalized tensor 空間，不需 denorm）。

此模組提供有成對目標的 Mphy 監督式 loss 函式（ObsMphy 系列），
以及 non-worsening 穩定性懲罰（Stab_Improve）。

所有函式操作於 normalized Mphy 空間，不需呼叫 FieldNormalizer。

命名規則（繼承自 loss_terms_v2）：
  obs_mphy_traj_mae   — 所有 rollout steps 的 MAE
  obs_mphy_delta_mae  — 逐步更新量的 MAE
  obs_mphy_final_mae  — 最終 rollout step 的 MAE（診斷用）
  stab_improve_loss   — non-worsening 懲罰
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from ..context import _LossContext  # noqa: F401（型別標注預留，_LossContext 尚未實作）

__all__ = [
    "obs_mphy_traj_mae",
    "obs_mphy_delta_mae",
    "obs_mphy_final_mae",
    "stab_improve_loss",
    "mean_flow_loss",
]


def obs_mphy_traj_mae(
    pred: Tensor,        # [B, k, 9, H, W] normalized
    target: Tensor,      # [B, k, 9, H, W] normalized
    fluid_mask: Tensor,  # [B, 1, H, W] float
) -> Tensor:
    """ObsMphy_Traj_Mae：所有 rollout steps 的 Mphy MAE（fluid 區域）。

    Args:
        pred: ``[B, k, 9, H, W]`` 正規化預測軌跡。
        target: ``[B, k, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。

    Returns:
        scalar Tensor（fluid 區域加權 MAE）。
    """
    mask = fluid_mask.unsqueeze(1)   # [B, 1, 1, H, W]
    diff = (pred - target).abs()
    mask_bc = mask.expand_as(diff)
    return (diff * mask_bc).sum() / mask_bc.sum().clamp_min(1.0)


def obs_mphy_delta_mae(
    pred: Tensor,        # [B, k, 9, H, W] normalized
    target: Tensor,      # [B, k, 9, H, W] normalized
    ic: Tensor,          # [B, 9, H, W] normalized（t=0 initial condition）
    fluid_mask: Tensor,  # [B, 1, H, W] float
) -> Tensor:
    """ObsMphy_Delta_Mae：逐步更新量的 MAE（fluid 區域）。

    定義：
      step 0:  DeltaPred[0] = Pred[0] - IC;  DeltaTarget[0] = Target[0] - IC
      step i:  DeltaPred[i] = Pred[i] - Pred[i-1];  DeltaTarget[i] = Target[i] - Target[i-1]

    只在 k>=2 時才有獨立貢獻；k=1 時與 ObsMphy_Traj_Mae 步驟等價。

    Args:
        pred: ``[B, k, 9, H, W]`` 正規化預測軌跡。
        target: ``[B, k, 9, H, W]`` 正規化目標軌跡。
        ic: ``[B, 9, H, W]`` 正規化初始條件（t=0）。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。

    Returns:
        scalar Tensor（逐步更新量 MAE）。
    """
    ic_t = ic.unsqueeze(1)                               # [B, 1, 9, H, W]
    pred_prev = torch.cat([ic_t, pred[:, :-1]], dim=1)   # [B, k, 9, H, W]
    tgt_prev = torch.cat([ic_t, target[:, :-1]], dim=1)
    delta_pred = pred - pred_prev
    delta_tgt = target - tgt_prev
    mask = fluid_mask.unsqueeze(1)
    diff = (delta_pred - delta_tgt).abs()
    mask_bc = mask.expand_as(diff)
    return (diff * mask_bc).sum() / mask_bc.sum().clamp_min(1.0)


def obs_mphy_final_mae(
    pred: Tensor,        # [B, k, 9, H, W] normalized
    target: Tensor,      # [B, k, 9, H, W] normalized
    fluid_mask: Tensor,  # [B, 1, H, W] float
) -> Tensor:
    """ObsMphy_Final_Mae：最終 rollout step 的 Mphy MAE（fluid 區域）。

    預設為診斷指標；只有在 final-step error 持續高於 trajectory average 時才列入消融。

    Args:
        pred: ``[B, k, 9, H, W]`` 正規化預測軌跡。
        target: ``[B, k, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。

    Returns:
        scalar Tensor（最終步 MAE）。
    """
    diff = (pred[:, -1] - target[:, -1]).abs()   # [B, 9, H, W]
    mask_bc = fluid_mask.expand_as(diff)
    return (diff * mask_bc).sum() / mask_bc.sum().clamp_min(1.0)


def stab_improve_loss(
    pred: Tensor,        # [B, k, 9, H, W] normalized
    target: Tensor,      # [B, k, 9, H, W] normalized
    fluid_mask: Tensor,  # [B, 1, H, W] float
) -> Tensor:
    """Stab_Improve / non-worsening loss（TV2 §1.4）：懲罰「誤差比上一步變大」的 rollout 步。

    per-step error e_i = fluid 區 9ch MAE(pred_i, target_i)（跨 batch/channel/space 平均），
    回傳 mean_i relu(e_i - e_{i-1})（i=1..k-1）。k<2 無相鄰步 → 0（保留 graph/device/dtype）。

    cold-0.2 safe：只罰「誤差增長」，不強迫 fixed-point（不會訓成 smooth-blob「不要動」）。

    Args:
        pred: ``[B, k, 9, H, W]`` 正規化預測軌跡。
        target: ``[B, k, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。

    Returns:
        scalar Tensor（non-worsening 懲罰；k<2 時為 0）。
    """
    if pred.shape[1] < 2:
        return (pred * 0.0).sum()
    m = fluid_mask.unsqueeze(1)                    # [B, 1, 1, H, W]
    diff = (pred - target).abs() * m               # [B, k, 9, H, W]
    num = diff.sum(dim=(0, 2, 3, 4))               # [k] 跨 batch/channel/space
    den = m.sum().clamp_min(1.0) * pred.shape[2]   # fluid cells * 9ch
    e = num / den                                  # [k] per-step error
    return torch.relu(e[1:] - e[:-1]).mean()


# ---------------------------------------------------------------------------
# mean_flow_loss（Phase 0 mean_module loss，需獨立 import 時在此）
# ---------------------------------------------------------------------------

_JX_CH: int = 3  # m3: x 動量（本模組局部別名，避免與 __all__ 衝突）
_JY_CH: int = 5  # m5: y 動量


def mean_flow_loss(
    pred_phys: Tensor,
    target_mean: Tensor,
    fluid_mask: Tensor,
    ssim_weight: float = 0.3,
    per_channel_norm: bool = False,
) -> tuple[Tensor, Tensor]:
    """Phase 0 mean_module loss：MSE + λ·(1−SSIM) on fluid pixels。

    Args:
        pred_phys: ``[B, 9, H, W]`` 正規化預測物理場。
        target_mean: ``[B, 9, H, W]`` mean_flow_post_conv 目標場。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        ssim_weight: SSIM 項的權重（λ）。
        per_channel_norm: True → 各通道 MSE 除以 target 變異數（fluid 像素）後再加總，
            使 9 通道等貢獻。預設 False（uniform）。

    Returns:
        ``(loss_mse, ssim_term)`` — total = loss_mse + ssim_term。
    """
    from .perceptual import _ssim_loss_2d  # 避免循環 import（在函式內引用）

    pred_f = pred_phys.float()
    tgt_f = target_mean.float()
    mask = fluid_mask.float().expand_as(pred_f)
    diff = (pred_f - tgt_f) * mask
    n_fluid = mask.sum().clamp_min(1.0)
    if per_channel_norm:
        n_fl_ch = mask.sum(dim=(0, 2, 3), keepdim=True).clamp_min(1.0)   # [1,9,1,1]
        with torch.no_grad():
            tgt_mean_ch = (tgt_f * mask).sum(dim=(0, 2, 3), keepdim=True) / n_fl_ch
            tgt_var_ch = (((tgt_f - tgt_mean_ch) * mask) ** 2).sum(dim=(0, 2, 3), keepdim=True) / n_fl_ch
            ch_w = 1.0 / (tgt_var_ch + 1e-4)
            ch_w = ch_w / ch_w.mean().clamp_min(1e-8)
        loss_mse = ((diff * diff) * ch_w).sum() / n_fluid
    else:
        loss_mse = (diff * diff).sum() / n_fluid

    # SSIM on jx + jy channels（JX_CH=3, JY_CH=5；移至 [0,1]，兩通道平均）。
    # 僅用 jx 單通道做結構監督時，jy／橫向結構的約束不足，會讓尾流過度平滑，
    # 故 SSIM 同時涵蓋 jx 與 jy。
    jx_p = ((pred_phys[:, _JX_CH:_JX_CH + 1].float() + 1.0) * 0.5).clamp(0.0, 1.0)
    jx_t = ((target_mean[:, _JX_CH:_JX_CH + 1].float() + 1.0) * 0.5).clamp(0.0, 1.0)
    jy_p = ((pred_phys[:, _JY_CH:_JY_CH + 1].float() + 1.0) * 0.5).clamp(0.0, 1.0)
    jy_t = ((target_mean[:, _JY_CH:_JY_CH + 1].float() + 1.0) * 0.5).clamp(0.0, 1.0)
    ssim_term = ssim_weight * 0.5 * (_ssim_loss_2d(jx_p, jx_t) + _ssim_loss_2d(jy_p, jy_t))

    return loss_mse, ssim_term
