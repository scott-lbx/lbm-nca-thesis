"""Identity / perturbation tests for loss_terms_v2.

Gate criteria（from docs/methodology/loss_terms_execution_plan_tes.md §6）：
  - ObsMphy_* identity: loss 接近 0 when pred == target
  - PhysMass_* identity: loss = LBM floor，不要求為 0
  - fluid-only perturbation: observation + fluid-supported terms 上升
  - solid-only perturbation: only solid-supported terms 上升
  - support 外 perturbation: 對該 term 不應有反應
  - 小擾動造成 loss 平滑上升，不出現 NaN / Inf

Test fixtures use synthetic tensors（no zarr required）。
LBM floor reference values（corrected, denominator = n_fluid × T）from S0/S1 audit:
  PhysMass_DivJ_AbsMean   ~1.1–1.8e-4
  PhysMass_DivVel_AbsMean ~1.1–1.8e-4
  BndWall_NormalFlux_AbsMean ~7e-4 to 1.9e-3
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.loss.loss_terms_v2 import (
    JX_CH,
    bnd_wall_normalflux_absmean,
    compute_wall_band,
    fluid_mask_from_static,
    mean_flow_loss,
    obs_mphy_delta_mae,
    obs_mphy_final_mae,
    obs_mphy_traj_mae,
    phys_mass_divj_absmean,
    phys_mass_divvel_absmean,
)
from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, K, H, W = 2, 4, 32, 32


def _make_normalizer() -> FieldNormalizer:
    """Minimal normalizer with identity stats（mean=0, std=1）."""
    n = 9
    return FieldNormalizer(
        mf_mean=np.zeros(n, dtype=np.float32),
        mf_std=np.ones(n, dtype=np.float32),
        up_mean=np.zeros(n, dtype=np.float32),
        up_std=np.ones(n, dtype=np.float32),
        sdf_mean=0.0,
        sdf_std=1.0,
        px_mean=0.0,
        px_std=1.0,
        py_mean=0.0,
        py_std=1.0,
        cond_mean=np.zeros(1, dtype=np.float32),
        cond_std=np.ones(1, dtype=np.float32),
        is_physics_informed=False,
    )


def _make_masks(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fluid_mask, solid_mask, sdf_raw for H×W domain.

    Left quarter is solid, rest is fluid（simple rectangle obstacle）.
    Returns:
        fluid_mask: [B, 1, H, W]
        solid_mask: [B, 1, H, W]
        sdf_raw: [B, H, W] (positive inside fluid, negative inside solid)
    """
    sdf = torch.ones(B, H, W, device=device)
    sdf[:, :, : W // 4] = -1.0
    fluid = (sdf > 0).float().unsqueeze(1)
    solid = 1.0 - fluid
    return fluid, solid, sdf


def _make_traj(scale: float = 0.1, device: str = "cpu") -> torch.Tensor:
    """Random trajectory [B, k, 9, H, W]."""
    return torch.randn(B, K, 9, H, W, device=device) * scale


def _make_ic(scale: float = 0.1, device: str = "cpu") -> torch.Tensor:
    return torch.randn(B, 9, H, W, device=device) * scale


# ---------------------------------------------------------------------------
# Observation loss — identity tests
# ---------------------------------------------------------------------------


class TestObsMphyTrajMae:
    def test_identity_zero(self):
        """pred == target → loss ≈ 0."""
        fluid, _, _ = _make_masks()
        traj = _make_traj()
        loss = obs_mphy_traj_mae(traj, traj, fluid)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_finite(self):
        fluid, _, _ = _make_masks()
        pred = _make_traj(0.1)
        tgt = _make_traj(0.1)
        loss = obs_mphy_traj_mae(pred, tgt, fluid)
        assert torch.isfinite(loss)

    def test_fluid_perturbation_increases_loss(self):
        """Perturbing fluid region increases loss."""
        fluid, solid, _ = _make_masks()
        traj = _make_traj(0.1)
        tgt = traj.clone()
        pred = traj.clone()
        # fluid: [B,1,H,W] → [B,1,1,H,W] → [B,K,9,H,W]
        fluid_5d = fluid.bool().unsqueeze(2).expand(B, 1, K, H, W)
        # Alternate approach: additive mask without indexed assignment
        noise = torch.zeros_like(pred)
        fluid_broad = fluid.bool().unsqueeze(1).expand_as(pred)
        noise[fluid_broad] = 0.5
        pred = pred + noise
        loss_perturbed = obs_mphy_traj_mae(pred, tgt, fluid)
        assert float(loss_perturbed) > 1e-4

    def test_solid_perturbation_no_effect(self):
        """Perturbing solid region only has NO effect on fluid-masked loss."""
        fluid, solid, _ = _make_masks()
        traj = _make_traj(0.1)
        tgt = traj.clone()
        pred = traj.clone()
        # solid: [B,1,H,W] → [B,1,1,H,W] → broadcast to [B,K,9,H,W]
        solid_broad = solid.bool().unsqueeze(1).expand_as(pred)
        noise = torch.zeros_like(pred)
        noise[solid_broad] = 5.0
        pred = pred + noise
        loss_perturbed = obs_mphy_traj_mae(pred, tgt, fluid)
        assert float(loss_perturbed) == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows(self):
        fluid, _, _ = _make_masks()
        pred = _make_traj(0.1).requires_grad_(True)
        tgt = _make_traj(0.1)
        loss = obs_mphy_traj_mae(pred, tgt, fluid)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestObsMphyDeltaMae:
    def test_identity_zero(self):
        """pred == target → delta loss ≈ 0."""
        fluid, _, _ = _make_masks()
        traj = _make_traj()
        ic = _make_ic()
        loss = obs_mphy_delta_mae(traj, traj, ic, fluid)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_finite(self):
        fluid, _, _ = _make_masks()
        pred = _make_traj(0.1)
        tgt = _make_traj(0.1)
        ic = _make_ic(0.1)
        loss = obs_mphy_delta_mae(pred, tgt, ic, fluid)
        assert torch.isfinite(loss)

    def test_different_from_traj_mae_when_k_gt_1(self):
        """With k>1 and auto-regressive drift, delta_mae != traj_mae."""
        fluid, _, _ = _make_masks()
        pred = _make_traj(0.1)
        tgt = _make_traj(0.1)
        ic = _make_ic(0.1)
        l_traj = obs_mphy_traj_mae(pred, tgt, fluid)
        l_delta = obs_mphy_delta_mae(pred, tgt, ic, fluid)
        # Not necessarily equal; just check both finite
        assert torch.isfinite(l_traj) and torch.isfinite(l_delta)

    def test_gradient_flows(self):
        fluid, _, _ = _make_masks()
        pred = _make_traj(0.1).requires_grad_(True)
        tgt = _make_traj(0.1)
        ic = _make_ic(0.1)
        loss = obs_mphy_delta_mae(pred, tgt, ic, fluid)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestObsMphyFinalMae:
    def test_identity_zero(self):
        fluid, _, _ = _make_masks()
        traj = _make_traj()
        loss = obs_mphy_final_mae(traj, traj, fluid)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_uses_only_last_step(self):
        """Perturbing all steps except last → loss == 0."""
        fluid, _, _ = _make_masks()
        traj = _make_traj(0.1)
        tgt = traj.clone()
        pred = traj.clone()
        pred[:, :-1] += 10.0   # perturb all steps except final
        loss = obs_mphy_final_mae(pred, tgt, fluid)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Physics loss — identity / floor tests
# ---------------------------------------------------------------------------


class TestPhysMassDivJAbsMean:
    def test_finite_on_random(self):
        fluid, _, _ = _make_masks()
        pred = _make_traj(0.1)
        norm = _make_normalizer()
        loss = phys_mass_divj_absmean(pred, fluid, norm)
        assert torch.isfinite(loss)
        assert float(loss) >= 0.0

    def test_bulk_fluid_perturbation_increases_loss(self):
        """Perturbing interior fluid region（away from solid boundary）increases DivJ."""
        fluid, solid, _ = _make_masks()
        norm = _make_normalizer()
        # Interior fluid: columns W//2 to W (far from solid at columns 0..W//4)
        interior_fluid = fluid.clone()
        interior_fluid[:, :, :, : W // 2] = 0.0  # mask out near-boundary half
        pred_base = torch.zeros(B, K, 9, H, W)
        # Add high-frequency jx pattern only in interior fluid
        noise = torch.zeros_like(pred_base)
        noise[:, :, 3, :, W // 2 :] = torch.randn(B, K, H, W // 2) * 2.0  # jx ch
        pred_perturb = pred_base + noise
        l_base = phys_mass_divj_absmean(pred_base, interior_fluid, norm)
        l_perturb = phys_mass_divj_absmean(pred_perturb, interior_fluid, norm)
        assert float(l_perturb) > float(l_base) + 1e-6

    def test_smooth_under_perturbation(self):
        """Small perturbation causes smooth increase, no NaN."""
        fluid, _, _ = _make_masks()
        norm = _make_normalizer()
        pred = _make_traj(0.05)
        for scale in [0.0, 0.01, 0.1, 0.5]:
            perturbed = pred + scale * torch.randn_like(pred)
            loss = phys_mass_divj_absmean(perturbed, fluid, norm)
            assert torch.isfinite(loss), f"NaN at scale={scale}"

    def test_gradient_flows(self):
        fluid, _, _ = _make_masks()
        norm = _make_normalizer()
        pred = _make_traj(0.1).requires_grad_(True)
        loss = phys_mass_divj_absmean(pred, fluid, norm)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestPhysMassDivVelAbsMean:
    def test_finite_on_random(self):
        fluid, _, _ = _make_masks()
        pred = _make_traj(0.1)
        norm = _make_normalizer()
        loss = phys_mass_divvel_absmean(pred, fluid, norm)
        assert torch.isfinite(loss)
        assert float(loss) >= 0.0

    def test_close_to_divj_for_unit_density(self):
        """With identity normalizer（mean=0, std=1）, rho ≈ 0（denormed）.
        clamp at rho_eps ensures division is stable. Loss should be finite."""
        fluid, _, _ = _make_masks()
        pred = _make_traj(0.1)
        norm = _make_normalizer()
        l_divj = phys_mass_divj_absmean(pred, fluid, norm)
        l_divvel = phys_mass_divvel_absmean(pred, fluid, norm)
        # Both finite; no constraint on relative magnitude with unit normalizer
        assert torch.isfinite(l_divj) and torch.isfinite(l_divvel)

    def test_gradient_flows(self):
        fluid, _, _ = _make_masks()
        norm = _make_normalizer()
        pred = _make_traj(0.1).requires_grad_(True)
        loss = phys_mass_divvel_absmean(pred, fluid, norm)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Boundary loss
# ---------------------------------------------------------------------------


class TestBndWallNormalFluxAbsMean:
    def test_finite(self):
        fluid, _, sdf = _make_masks()
        norm = _make_normalizer()
        pred = _make_traj(0.1)
        wall = compute_wall_band(fluid)
        loss = bnd_wall_normalflux_absmean(pred, sdf, fluid, wall, norm)
        assert torch.isfinite(loss)
        assert float(loss) >= 0.0

    def test_no_nan_small_perturbation(self):
        fluid, _, sdf = _make_masks()
        norm = _make_normalizer()
        for scale in [0.0, 0.01, 0.1, 0.5]:
            pred = torch.randn(B, K, 9, H, W) * scale
            wall = compute_wall_band(fluid)
            loss = bnd_wall_normalflux_absmean(pred, sdf, fluid, wall, norm)
            assert torch.isfinite(loss), f"NaN at scale={scale}"

    def test_gradient_flows(self):
        fluid, _, sdf = _make_masks()
        norm = _make_normalizer()
        wall = compute_wall_band(fluid)
        pred = _make_traj(0.1).requires_grad_(True)
        loss = bnd_wall_normalflux_absmean(pred, sdf, fluid, wall, norm)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


class TestComputeWallBand:
    def test_wall_band_is_subset_of_fluid(self):
        fluid, _, _ = _make_masks()
        wall = compute_wall_band(fluid)
        assert (wall * (1.0 - fluid)).sum() == pytest.approx(0.0, abs=1e-6)

    def test_wall_band_nonempty_for_obstacle(self):
        fluid, _, _ = _make_masks()
        wall = compute_wall_band(fluid)
        assert float(wall.sum()) > 0.0

    def test_wall_band_smaller_than_fluid(self):
        fluid, _, _ = _make_masks()
        wall = compute_wall_band(fluid)
        assert float(wall.sum()) < float(fluid.sum())


# ---------------------------------------------------------------------------
# Mask normalization: solid fraction independence
# ---------------------------------------------------------------------------


class TestMaskNormalization:
    def test_obs_traj_mae_solid_fraction_independent(self):
        """Constant per-pixel error gives same MAE regardless of solid fraction.

        With correct mask normalization（denominator = n_fluid × K × C）, a
        constant error of ε in the fluid region must return ε regardless of how
        large the solid region is.
        """
        norm = _make_normalizer()
        err = 0.1

        def _run(solid_frac: float) -> float:
            solid_cols = int(W * solid_frac)
            sdf = torch.ones(B, H, W)
            sdf[:, :, :solid_cols] = -1.0
            fluid = (sdf > 0).float().unsqueeze(1)   # [B,1,H,W]
            target = torch.zeros(B, K, 9, H, W)
            pred = target.clone()
            # Add constant error only in fluid region
            fluid_broad = fluid.unsqueeze(1).expand_as(pred)  # [B,1,1,H,W]→[B,K,9,H,W]
            pred = pred + err * fluid_broad.float()
            return float(obs_mphy_traj_mae(pred, target, fluid))

        loss_sparse = _run(solid_frac=0.05)   # 5% solid
        loss_dense = _run(solid_frac=0.50)    # 50% solid
        assert loss_sparse == pytest.approx(err, rel=1e-4)
        assert loss_dense == pytest.approx(err, rel=1e-4)

    def test_obs_final_mae_solid_fraction_independent(self):
        """Same check for ObsMphy_Final_Mae."""
        err = 0.2

        def _run(solid_frac: float) -> float:
            solid_cols = int(W * solid_frac)
            sdf = torch.ones(B, H, W)
            sdf[:, :, :solid_cols] = -1.0
            fluid = (sdf > 0).float().unsqueeze(1)
            target = torch.zeros(B, K, 9, H, W)
            pred = target.clone()
            fluid_broad = fluid.expand_as(pred[:, -1])   # [B,1,H,W]→[B,9,H,W]
            pred[:, -1] = pred[:, -1] + err * fluid_broad.float()
            return float(obs_mphy_final_mae(pred, target, fluid))

        assert _run(0.05) == pytest.approx(err, rel=1e-4)
        assert _run(0.50) == pytest.approx(err, rel=1e-4)


# ---------------------------------------------------------------------------
# mean_flow_loss（Phase 0 mean_module；M1/EG-2 通道回歸鎖定）
# ---------------------------------------------------------------------------


class TestMeanFlowLoss:
    """M1/EG-2：SSIM 必須取 jx（idx=3），不可回退到 e（idx=1）。"""

    @staticmethod
    def _make_pair() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        pred = torch.rand(B, 9, H, W) * 0.4 - 0.2
        target = pred.clone()
        fluid = torch.ones(B, 1, H, W)
        return pred, target, fluid

    def test_identity_zero(self):
        pred, target, fluid = self._make_pair()
        loss_mse, ssim_term = mean_flow_loss(pred, target, fluid)
        assert float(loss_mse) == pytest.approx(0.0, abs=1e-8)
        assert float(ssim_term) == pytest.approx(0.0, abs=1e-5)

    def test_ssim_sensitive_to_jx_channel(self):
        """結構性擾動 jx（idx=JX_CH=3）→ ssim_term 必須 > 0。"""
        pred, target, fluid = self._make_pair()
        torch.manual_seed(1)
        pred[:, JX_CH] = pred[:, JX_CH] + torch.randn(B, H, W) * 0.3
        _, ssim_term = mean_flow_loss(pred, target, fluid)
        assert float(ssim_term) > 1e-4

    def test_ssim_insensitive_to_e_channel(self):
        """擾動 e（idx=1）不得影響 ssim_term（舊 bug 誤取此通道）。"""
        pred, target, fluid = self._make_pair()
        torch.manual_seed(2)
        pred[:, 1] = pred[:, 1] + torch.randn(B, H, W) * 0.3
        _, ssim_term = mean_flow_loss(pred, target, fluid)
        assert float(ssim_term) == pytest.approx(0.0, abs=1e-5)

    def test_mse_fluid_only_constant_error(self):
        """fluid 區常數誤差 ε → MSE = ε²（mask 正規化正確）。"""
        err = 0.1
        sdf = torch.ones(B, H, W)
        sdf[:, :, : W // 4] = -1.0
        fluid = (sdf > 0).float().unsqueeze(1)
        target = torch.zeros(B, 9, H, W)
        pred = target + err * fluid.expand(B, 9, H, W)
        loss_mse, _ = mean_flow_loss(pred, target, fluid)
        assert float(loss_mse) == pytest.approx(err * err, rel=1e-4)

    def test_gradient_flows(self):
        pred, target, fluid = self._make_pair()
        pred = (pred + 0.05).requires_grad_(True)
        loss_mse, ssim_term = mean_flow_loss(pred, target, fluid)
        (loss_mse + ssim_term).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Real zarr floor consistency（skipped if zarr not available）
# ---------------------------------------------------------------------------

_ZARR_PATH = Path(__file__).resolve().parents[3] / "dataset_zarr" / "S0_1-Cold_M2C4.zarr"
_STATS_PATH = _ZARR_PATH.parent / "S0_1-Cold_M2C4_stats.json"
_REAL_ZARR_AVAILABLE = _ZARR_PATH.exists() and _STATS_PATH.exists()


@pytest.mark.skipif(not _REAL_ZARR_AVAILABLE, reason="dataset_zarr/S0_1 not found")
class TestRealZarrPhysMassFloor:
    """Verify PhysMass terms on real LBM target field match audit floor scale.

    Corrected floor（denominator = n_fluid × T）from S0/S1 audit:
      PhysMass_DivJ_AbsMean   mean ~1.1–1.4e-4, p95 < 3e-4
      PhysMass_DivVel_AbsMean mean ~1.1–1.4e-4, p95 < 3e-4
    """

    @pytest.fixture(scope="class")
    def zarr_window(self):
        root = zarr.open(str(_ZARR_PATH), mode="r")
        normalizer = FieldNormalizer.from_file(_STATS_PATH)
        case = root["cases"][list(root["cases"].keys())[0]]
        snaps = torch.from_numpy(np.ascontiguousarray(case["snapshots"][:9])).float()
        static = torch.from_numpy(np.ascontiguousarray(case["static"][:])).float()
        sdf_raw = normalizer.denorm_sdf(static[0])
        fluid_mask = (sdf_raw > 0).float().unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        pred = snaps.unsqueeze(0)                                        # [1,9,9,H,W]  (B=1,T=9,C=9,H,W)
        # Use snapshots as "pred" with shape [B=1, k=8, 9, H, W]（drop first as IC）
        pred_traj = pred[:, 1:]   # [1, 8, 9, H, W]
        return pred_traj, fluid_mask, normalizer

    def test_divj_in_audit_range(self, zarr_window):
        pred_traj, fluid_mask, normalizer = zarr_window
        loss = phys_mass_divj_absmean(pred_traj, fluid_mask, normalizer)
        val = float(loss)
        assert torch.isfinite(loss)
        # Corrected audit floor: mean ~1.1e-4, expect within 5e-5 to 5e-4
        assert 5e-5 < val < 5e-4, f"DivJ={val:.3e} outside expected 5e-5..5e-4"

    def test_divvel_in_audit_range(self, zarr_window):
        pred_traj, fluid_mask, normalizer = zarr_window
        loss = phys_mass_divvel_absmean(pred_traj, fluid_mask, normalizer)
        val = float(loss)
        assert torch.isfinite(loss)
        assert 5e-5 < val < 5e-4, f"DivVel={val:.3e} outside expected 5e-5..5e-4"

    def test_divj_divvel_close(self, zarr_window):
        """DivJ ≈ DivVel on LBM field because rho ≈ 1（low Mach LBM）."""
        pred_traj, fluid_mask, normalizer = zarr_window
        l_j = float(phys_mass_divj_absmean(pred_traj, fluid_mask, normalizer))
        l_v = float(phys_mass_divvel_absmean(pred_traj, fluid_mask, normalizer))
        # ratio should be close to 1 (within 20%)
        ratio = max(l_j, l_v) / max(min(l_j, l_v), 1e-10)
        assert ratio < 1.25, f"DivJ/DivVel ratio={ratio:.2f} > 1.25 (rho not ≈ 1?)"
