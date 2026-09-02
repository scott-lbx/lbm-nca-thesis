"""Shape and contract tests for H-series models.

Covers:
  * forward shape matches the TESNCA contract for return_trajectory False/True.
  * AMP autocast does not produce non-finite values.
  * ratio 4 and ratio 8 both run.
  * init_state / apply_ic_mask / get_fluid_mask are present and well-formed.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.config import TESModelConfig, TESMultiScaleConfig
from lbmnca.nca.tes.models.model_factory import build_tes_model


def _make_cfg(
    arch: str,
    ratio: int = 4,
    hidden_dim: int = 16,
    state_dim: int = 29,
    cond_dim: int = 19,
) -> TESModelConfig:
    return TESModelConfig(
        arch=arch,
        state_dim=state_dim,
        static_dim=4,
        physics_dim=9,
        hidden_dim=hidden_dim,
        cond_dim=cond_dim,
        encoder_hidden=32,
        nca_channels=16,
        g_vec_dim=0,
        gvec_mode="none",
        multiscale=TESMultiScaleConfig(
            levels=2,
            ratios=[ratio],
            share_core=(arch == "h_shared_scale"),
            scale_condition=True,
            topdown=True,
            bottomup=False,
        ),
    )


def _make_batch(cfg: TESModelConfig, h: int = 16, w: int = 16, batch: int = 1):
    static = torch.zeros(batch, cfg.static_dim, h, w)
    static[:, 0] = 1.0  # all fluid
    ic = torch.randn(batch, cfg.physics_dim, h, w) * 0.01
    cond = torch.randn(batch, cfg.cond_dim) * 0.1
    fluid = torch.ones(batch, 1, h, w)
    return static, ic, cond, fluid


H_ARCHES = [
    "h_shared_scale",
    "h_lapres",
    "h_pod_modal",
    "h_parent_leaf",
    "h_multirate",
    "h_block_refine",
]


@pytest.mark.parametrize("arch", H_ARCHES)
@pytest.mark.parametrize("ratio", [4, 8])
def test_h_forward_shape_no_trajectory(arch: str, ratio: int) -> None:
    cfg = _make_cfg(arch, ratio=ratio)
    model = build_tes_model(cfg)
    static, ic, cond, fluid = _make_batch(cfg, h=16, w=16)
    state = model.init_state(static, ic)
    out = model(state, cond, fluid, n_steps=3, return_trajectory=False)
    assert out.shape == state.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", H_ARCHES)
def test_h_forward_trajectory_length(arch: str) -> None:
    cfg = _make_cfg(arch, ratio=4)
    model = build_tes_model(cfg)
    static, ic, cond, fluid = _make_batch(cfg)
    state = model.init_state(static, ic)
    final, traj = model(state, cond, fluid, n_steps=4, return_trajectory=True)
    assert final.shape == state.shape
    assert len(traj) == 4
    for snap in traj:
        assert snap.shape == (1, cfg.physics_dim, 16, 16)
        assert torch.isfinite(snap).all()


@pytest.mark.parametrize("arch", H_ARCHES)
def test_h_amp_finite(arch: str) -> None:
    cfg = _make_cfg(arch, ratio=4)
    model = build_tes_model(cfg)
    static, ic, cond, fluid = _make_batch(cfg)
    state = model.init_state(static, ic)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        final, traj = model(state, cond, fluid, n_steps=2, return_trajectory=True)
    assert torch.isfinite(final.float()).all()
    for snap in traj:
        assert torch.isfinite(snap.float()).all()


@pytest.mark.parametrize("arch", H_ARCHES)
def test_h_state_utils(arch: str) -> None:
    cfg = _make_cfg(arch)
    model = build_tes_model(cfg)
    static, ic, _, _ = _make_batch(cfg)
    state = model.init_state(static, ic)
    assert state.shape[1] == cfg.state_dim
    # apply_ic_mask zeroes dynamic in solid cells
    solid = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    solid[..., :4, :4] = True
    masked = model.apply_ic_mask(state, solid)
    dyn = masked[:, cfg.static_dim:]
    assert torch.all(dyn[..., :4, :4] == 0)
    # get_fluid_mask derives from SDF (static[:,0])
    fluid = model.get_fluid_mask(static)
    assert fluid.shape == (1, 1, 16, 16)


def test_h_forward_with_tfc() -> None:
    """tf_ratio>0 with gt_traj should not crash and stay finite."""
    cfg = _make_cfg("h_shared_scale", ratio=4)
    model = build_tes_model(cfg)
    static, ic, cond, fluid = _make_batch(cfg)
    state = model.init_state(static, ic)
    gt = torch.randn(1, 3, cfg.physics_dim, 16, 16) * 0.01
    final, traj = model(
        state, cond, fluid, n_steps=3, return_trajectory=True,
        tf_ratio=0.5, gt_traj=gt, ss_detach=True,
    )
    assert len(traj) == 3
    assert torch.isfinite(final).all()


def test_hblockrefine_aux_metrics() -> None:
    cfg = _make_cfg("h_block_refine", ratio=4)
    cfg.multiscale.block_size = 8
    cfg.multiscale.active_ratio = 0.25
    model = build_tes_model(cfg)
    static, ic, cond, fluid = _make_batch(cfg)
    state = model.init_state(static, ic)
    _ = model(state, cond, fluid, n_steps=2, return_trajectory=False)
    aux = model.get_aux_metrics()
    assert "active_block_ratio" in aux
    assert 0.0 <= aux["active_block_ratio"] <= 1.0


def test_hpodmodal_aux_metrics() -> None:
    cfg = _make_cfg("h_pod_modal", ratio=4)
    model = build_tes_model(cfg)
    static, ic, cond, fluid = _make_batch(cfg)
    state = model.init_state(static, ic)
    _ = model(state, cond, fluid, n_steps=2, return_trajectory=False)
    aux = model.get_aux_metrics()
    assert "modal_coeff_l2_mean" in aux


def test_hlapres_aux_metrics() -> None:
    cfg = _make_cfg("h_lapres", ratio=4)
    model = build_tes_model(cfg)
    static, ic, cond, fluid = _make_batch(cfg)
    state = model.init_state(static, ic)
    _ = model(state, cond, fluid, n_steps=2, return_trajectory=False)
    aux = model.get_aux_metrics()
    assert "coarse_pred_energy" in aux
    assert "fine_residual_energy" in aux
    assert "residual_to_coarse_energy_ratio" in aux


def test_hmultirate_cond19_per_step_dcs_and_multiscale() -> None:
    """H3 cond19 refreshes per-step CTU fields without widening cond input."""
    cfg = _make_cfg("h_multirate", ratio=4)
    cfg.multiscale.ratios = [4, 8]
    cfg.multiscale.levels = 3
    cfg.multiscale.fine_substeps = 2
    model = build_tes_model(cfg)
    static, ic, cond, fluid = _make_batch(cfg, h=16, w=16)
    state = model.init_state(static, ic)

    seen: list[torch.Tensor] = []
    original_forward = model.cond_encoder.forward

    def _record_cond(x: torch.Tensor):
        seen.append(x.detach().clone())
        return original_forward(x)

    model.cond_encoder.forward = _record_cond  # type: ignore[method-assign]

    delta_ctu_seq = torch.tensor([[0.2, 2.0]])
    cumul_ctu_norm_seq = torch.tensor([[0.10, 0.25]])
    final, traj = model(
        state,
        cond,
        fluid,
        n_steps=2,
        return_trajectory=True,
        delta_ctu_seq=delta_ctu_seq,
        cumul_ctu_norm_seq=cumul_ctu_norm_seq,
    )

    assert final.shape == state.shape
    assert len(traj) == 2
    assert torch.isfinite(final).all()
    # two coarse scales plus one fine encode per outer step
    assert len(seen) == 2 * (len(cfg.multiscale.ratios) + 1)
    assert seen[0].shape[1] == cfg.cond_dim + 4
    expected_dt0 = math.log1p(0.2) / math.log1p(5.0)
    assert seen[0][0, 17].item() == pytest.approx(expected_dt0, rel=1e-6)
    assert seen[0][0, 18].item() == pytest.approx(0.10, rel=1e-6)
    aux = model.get_aux_metrics()
    assert aux["dcs_enabled"] == pytest.approx(1.0)
    assert aux["num_coarse_scales"] == pytest.approx(2.0)
    assert aux["max_ratio"] == pytest.approx(8.0)
