"""診斷函式（@torch.inference_mode，不參與 loss 計算）。

提取自 loss_terms_v2.py（§ diag_phys_mass_continuity / diag_solid_energy）。

這些函式標注 ``@torch.inference_mode()``，純用於訓練監控與指標記錄，
不得加入 LossTotal 計算路徑。

LBM floor 參考值：
  diag_phys_mass_continuity  ~0.04–0.11（比 DivJ 高 268–479x）
  diag_solid_energy          ~0.667（raw space；solid zero-energy 假設無效）
"""
from __future__ import annotations

import torch
from torch import Tensor

from ....utils.FieldNormalizer import FieldNormalizer
from ...utils.spatial_ops import ddx as _ddx, ddy as _ddy

# ---------------------------------------------------------------------------
# 模組層級常數（與 loss_terms_v2 對齊）
# ---------------------------------------------------------------------------

RHO_CH: int = 0
JX_CH: int = 3
JY_CH: int = 5

_RHO_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# 診斷指標（不進 LossTotal）
# ---------------------------------------------------------------------------


@torch.inference_mode()
def diag_phys_mass_continuity(
    pred_norm: Tensor,           # [B, k, 9, H, W] normalized
    delta_ctus: Tensor,          # [B, k-1] step durations
    fluid_mask: Tensor,          # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """PhysMass_Continuity_AbsMean（診斷 only）。

    LBM floor: ~0.04–0.11（比 DivJ 高 268–479x）→ 禁止進 LossTotal。
    使用 ``@torch.inference_mode()`` 裝飾：純診斷，不需梯度。

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 正規化預測物理場。
        delta_ctus: ``[B, k-1]`` 各步時間間隔（CTU）。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。

    Returns:
        scalar Tensor（連續性方程殘差均絕對值，無梯度）。
    """
    B, K, C, H, W = pred_norm.shape
    flat = pred_norm.reshape(B * K, C, H, W)
    raw = normalizer.denorm_mphy(flat)          # [B*K, 9, H, W]
    rho = raw[:, RHO_CH].clamp_min(rho_eps).reshape(B, K, H, W)
    jx = raw[:, JX_CH].reshape(B, K, H, W)
    jy = raw[:, JY_CH].reshape(B, K, H, W)

    if K < 2:
        return torch.tensor(0.0, device=pred_norm.device)

    dt = delta_ctus[:, :K - 1].clamp_min(rho_eps)  # [B, k-1]
    drho_dt = (rho[:, 1:] - rho[:, :-1]) / dt[:, :, None, None]
    div_j_t = _ddx(jx[:, 1:]) + _ddy(jy[:, 1:])   # [B, k-1, H, W]
    continuity = drho_dt + div_j_t

    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K - 1, H, W)
    n = mask_bk.sum().clamp_min(1.0)
    return (continuity.abs() * mask_bk).sum() / n


@torch.inference_mode()
def diag_solid_energy(
    pred_norm: Tensor,      # [B, k, 9, H, W] normalized
    solid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
) -> Tensor:
    """BndSolid_MphyRawEnergy_Mse（診斷 only）。

    LBM floor: ~0.667（raw space，已修正）→ solid zero-energy 假設無效。
    S0 到 E1 禁止進 LossTotal。
    使用 ``@torch.inference_mode()`` 裝飾：純診斷，不需梯度。

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 正規化預測物理場。
        solid_mask: ``[B, 1, H, W]`` float，固體=1 流體=0。
        normalizer: 反正規化器。

    Returns:
        scalar Tensor（固體區 raw Mphy 能量均值，無梯度）。
    """
    B, K, C, H, W = pred_norm.shape
    flat = pred_norm.reshape(B * K, C, H, W)
    raw = normalizer.denorm_mphy(flat)
    raw_energy = raw.square().mean(dim=1).reshape(B, K, H, W)
    mask_bk = solid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W)
    n = mask_bk.sum().clamp_min(1.0)
    return (raw_energy * mask_bk).sum() / n
