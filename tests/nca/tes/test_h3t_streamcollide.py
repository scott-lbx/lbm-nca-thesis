"""HE H3T-StreamCollide architecture tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.config import (  # noqa: E402
    MRPConfig,
    TESModelConfig,
    TESMultiScaleConfig,
    TESTransportConfig,
)
from lbmnca.nca.tes.models.h3t_streamcollide_nca import H3TStreamCollideNCA  # noqa: E402
from lbmnca.nca.tes.models.model_factory import build_tes_model  # noqa: E402


def _make_cfg() -> TESModelConfig:
    return TESModelConfig(
        arch="h3t_streamcollide",
        state_dim=29,
        static_dim=4,
        physics_dim=9,
        hidden_dim=16,
        cond_dim=19,
        encoder_hidden=32,
        nca_channels=16,
        g_vec_dim=0,
        gvec_mode="none",
        mrp=MRPConfig(enabled=True, num_learnable=8, hormone_dim=4),
        multiscale=TESMultiScaleConfig(ratios=[8, 32], fine_substeps=2, scale_condition=True),
        transport=TESTransportConfig(warp_ratio=8, enc_hidden=16, disp_max_px=20.0, gate_bias=-4.0),
    )


def _inputs(b: int = 2, h: int = 32, w: int = 32, cfg: TESModelConfig | None = None):
    cfg = cfg or _make_cfg()
    static = torch.randn(b, cfg.static_dim, h, w)
    static[:, 0] = 1.0
    ic = torch.randn(b, cfg.physics_dim, h, w) * 0.1
    cond = torch.randn(b, cfg.cond_dim)
    return static, ic, cond


def test_factory_builds_streamcollide() -> None:
    model = build_tes_model(_make_cfg())
    assert isinstance(model, H3TStreamCollideNCA)


def test_forward_shape_and_trajectory() -> None:
    cfg = _make_cfg()
    model = build_tes_model(cfg).eval()
    static, ic, cond = _inputs(cfg=cfg)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    dseq = torch.full((2, 4), 0.2)
    with torch.no_grad():
        final, traj = model(state, cond, fm, n_steps=4, return_trajectory=True, delta_ctu_seq=dseq)
    assert final.shape == state.shape
    assert len(traj) == 4
    assert torch.isfinite(final).all()
    assert model.get_aux_metrics()["collide_residual_ratio"] >= 0.0
    for t in traj:
        assert t.shape == (2, cfg.physics_dim, 32, 32)
        assert torch.isfinite(t).all()


def test_zero_init_stream_identity() -> None:
    cfg = _make_cfg()
    model = build_tes_model(cfg).eval()
    static, ic, _ = _inputs(cfg=cfg)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    prev_phys = state[:, cfg.static_dim:cfg.static_dim + cfg.physics_dim]
    with torch.no_grad():
        # collect_metrics 已改 opt-in（aux 按需收集），測試需顯式開啟
        warped, metrics = model._stream(state, prev_phys, fm, None, 0, collect_metrics=True)
    assert torch.allclose(warped, prev_phys, atol=1e-4)
    assert float(metrics["disp_max_px"]) == pytest.approx(0.0, abs=1e-7)


def test_delta_ctu_seq_scales_stream_displacement() -> None:
    cfg = _make_cfg()
    model = build_tes_model(cfg).eval()
    static, ic, _ = _inputs(cfg=cfg)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    prev_phys = state[:, cfg.static_dim:cfg.static_dim + cfg.physics_dim]
    torch.nn.init.normal_(model.disp_head.weight, std=0.5)
    torch.nn.init.normal_(model.disp_head.bias, std=0.5)
    with torch.no_grad():
        _, m_small = model._stream(state, prev_phys, fm, torch.full((2, 1), 0.2), 0,
                                   collect_metrics=True)
        _, m_big = model._stream(state, prev_phys, fm, torch.full((2, 1), 2.0), 0,
                                 collect_metrics=True)
    assert float(m_big["disp_max_px"]) > float(m_small["disp_max_px"])


def test_solid_mask_no_write() -> None:
    cfg = _make_cfg()
    model = build_tes_model(cfg).eval()
    static, ic, cond = _inputs(cfg=cfg)
    static[:, 0, :, :16] = -1.0
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    with torch.no_grad():
        _, traj = model(state, cond, fm, 3, True, delta_ctu_seq=torch.full((2, 3), 0.2))
    solid = fm[:, 0, :, :16] < 0.5
    for t in traj:
        phys_solid = t[:, :, :, :16]
        assert torch.allclose(
            phys_solid[solid.unsqueeze(1).expand_as(phys_solid)],
            torch.zeros(1),
            atol=1e-6,
        )
