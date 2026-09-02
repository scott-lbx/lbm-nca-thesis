"""向後相容性重匯出 — 實作已移至 loss/terms/obs_vphy.py 與 obs_state.py。"""
from __future__ import annotations

from .terms.obs_vphy import obs_vphy_vel_mae, obs_vphy_delta_mae, obs_vphy_wakevert_mae
from .terms.obs_state import obs_state_rho_mae, obs_state_stress_mae, obs_state_stress_anchor

__all__ = [
    "obs_vphy_vel_mae", "obs_vphy_delta_mae", "obs_vphy_wakevert_mae",
    "obs_state_rho_mae", "obs_state_stress_mae", "obs_state_stress_anchor",
]
