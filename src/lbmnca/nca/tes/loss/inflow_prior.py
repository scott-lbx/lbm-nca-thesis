"""向後相容性重匯出 — 實作已移至 loss/terms/inflow_prior.py。"""
from __future__ import annotations

from .terms.inflow_prior import (
    compute_inflow_masks,
    inflow_weight_map_seq,
    obs_mphy_traj_mae_prior,
)

__all__ = ["compute_inflow_masks", "inflow_weight_map_seq", "obs_mphy_traj_mae_prior"]
