"""loss.terms 子模組：按物理語義分類的損失函式集合（繁體中文說明）"""
from __future__ import annotations

from .obs_mphy import obs_mphy_traj_mae, obs_mphy_delta_mae, obs_mphy_final_mae, stab_improve_loss, mean_flow_loss
from .phys_residual import phys_mass_divj_absmean, phys_mass_divvel_absmean, mean_ke_log_anchor, fneq_match_loss
from .boundary import bnd_wall_normalflux_absmean
from .struct_velocity import (
    motion_twoframe_loss,
    obs_vel_vorticity_mae,
    obs_vel_vorticity_bl_mae,
    enstrophy_match_loss,
    vel_field_decoupled_loss,
    ke_magnitude_match_loss,
    obs_vel_gradient_mae,
    grad_energy_ratio_loss,
)
from .struct_wake import compute_wake_mask, struct_wake_vel_mae, struct_wake_vort_mae, struct_vort_signed_huber
from .perceptual import struct_vel_ssim_loss, struct_vort_ssim_loss, horiz_band_fft_loss, spectral_power_match_loss, mean_spectral_lowmode_loss
from .struct_banding import struct_column_banding_loss, struct_directional_band_loss
from .temporal_ke import domain_mean_ke, two_window_ke_loss, spectral_temporal_loss, temporal_ke_2ctu, jacobian_amp_loss
from .diagnostics import diag_phys_mass_continuity, diag_solid_energy
from .obs_vphy import obs_vphy_vel_mae, obs_vphy_delta_mae, obs_vphy_wakevert_mae
from .obs_state import obs_state_rho_mae, obs_state_stress_mae, obs_state_stress_anchor
from .inflow_prior import compute_inflow_masks, inflow_weight_map_seq, obs_mphy_traj_mae_prior
from .grain_fingerprint import grain_fingerprint_loss

__all__ = [
    # obs_mphy
    "obs_mphy_traj_mae",
    "obs_mphy_delta_mae",
    "obs_mphy_final_mae",
    "stab_improve_loss",
    "mean_flow_loss",
    # phys_residual
    "phys_mass_divj_absmean",
    "phys_mass_divvel_absmean",
    "mean_ke_log_anchor",
    "fneq_match_loss",
    # boundary
    "bnd_wall_normalflux_absmean",
    # struct_velocity
    "motion_twoframe_loss",
    "obs_vel_vorticity_mae",
    "obs_vel_vorticity_bl_mae",
    "enstrophy_match_loss",
    "vel_field_decoupled_loss",
    "ke_magnitude_match_loss",
    "obs_vel_gradient_mae",
    "grad_energy_ratio_loss",
    # struct_wake
    "compute_wake_mask",
    "struct_wake_vel_mae",
    "struct_wake_vort_mae",
    "struct_vort_signed_huber",
    # perceptual
    "struct_vel_ssim_loss",
    "struct_vort_ssim_loss",
    "struct_column_banding_loss",
    "struct_directional_band_loss",
    "horiz_band_fft_loss",
    "spectral_power_match_loss",
    "mean_spectral_lowmode_loss",
    # temporal_ke
    "domain_mean_ke",
    "two_window_ke_loss",
    "spectral_temporal_loss",
    "temporal_ke_2ctu",
    "jacobian_amp_loss",
    # diagnostics
    "diag_phys_mass_continuity",
    "diag_solid_energy",
    # obs_vphy
    "obs_vphy_vel_mae",
    "obs_vphy_delta_mae",
    "obs_vphy_wakevert_mae",
    # obs_state
    "obs_state_rho_mae",
    "obs_state_stress_mae",
    "obs_state_stress_anchor",
    # inflow_prior
    "compute_inflow_masks",
    "inflow_weight_map_seq",
    "obs_mphy_traj_mae_prior",
    # grain_fingerprint
    "grain_fingerprint_loss",
]
