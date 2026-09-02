"""向後相容性重匯出 — 實作已移至 loss/terms/。

此模組保留原始路徑，確保已存在的 import 語句不需修改仍可正常運作。
所有新程式碼請改用 ``from .terms import <function>``。
"""
from __future__ import annotations

# constants
RHO_CH: int = 0
JX_CH: int = 3
JY_CH: int = 5
PXX_CH: int = 7

# utils re-export
from ..utils.spatial_ops import (
    compute_wall_band,
    fluid_mask_from_static,
)

# terms re-exports
from .terms.obs_mphy import (
    obs_mphy_traj_mae,
    obs_mphy_delta_mae,
    obs_mphy_final_mae,
    stab_improve_loss,
    mean_flow_loss,
)
from .terms.phys_residual import (
    phys_mass_divj_absmean,
    phys_mass_divvel_absmean,
    fneq_match_loss,
)
from .terms.boundary import bnd_wall_normalflux_absmean
from .terms.struct_velocity import (
    motion_twoframe_loss,
    obs_vel_vorticity_mae,
    obs_vel_vorticity_bl_mae,
    enstrophy_match_loss,
    vel_field_decoupled_loss,
    ke_magnitude_match_loss,
    obs_vel_gradient_mae,
)
from .terms.struct_wake import (
    compute_wake_mask,
    struct_wake_vel_mae,
    struct_wake_vort_mae,
    struct_vort_signed_huber,
)
from .terms.perceptual import (
    _ssim_loss_2d,
    struct_vel_ssim_loss,
    struct_vort_ssim_loss,
    horiz_band_fft_loss,
    spectral_power_match_loss,
)
from .terms.temporal_ke import (
    domain_mean_ke,
    two_window_ke_loss,
    spectral_temporal_loss,
    temporal_ke_2ctu,
    jacobian_amp_loss,
)
from .terms.diagnostics import diag_phys_mass_continuity, diag_solid_energy

__all__ = [
    "RHO_CH", "JX_CH", "JY_CH", "PXX_CH",
    "compute_wall_band", "fluid_mask_from_static",
    "obs_mphy_traj_mae", "obs_mphy_delta_mae", "obs_mphy_final_mae",
    "stab_improve_loss", "mean_flow_loss",
    "phys_mass_divj_absmean", "phys_mass_divvel_absmean", "fneq_match_loss",
    "bnd_wall_normalflux_absmean",
    "motion_twoframe_loss", "obs_vel_vorticity_mae", "obs_vel_vorticity_bl_mae",
    "enstrophy_match_loss", "vel_field_decoupled_loss", "ke_magnitude_match_loss",
    "obs_vel_gradient_mae",
    "compute_wake_mask", "struct_wake_vel_mae", "struct_wake_vort_mae",
    "struct_vort_signed_huber",
    "_ssim_loss_2d", "struct_vel_ssim_loss", "struct_vort_ssim_loss",
    "horiz_band_fft_loss", "spectral_power_match_loss",
    "domain_mean_ke", "two_window_ke_loss", "spectral_temporal_loss",
    "temporal_ke_2ctu", "jacobian_amp_loss",
    "diag_phys_mass_continuity", "diag_solid_energy",
]
