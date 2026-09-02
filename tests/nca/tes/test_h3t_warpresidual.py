"""HD-4a H3T-WarpResidual 架構測試（§14.1 required tests）。

涵蓋：factory/config、forward shape、AMP finite、zero-init identity（disp=0/corr=0/gate≈sigmoid(bias)）、
delta_ctu_seq path、solid mask no-write。CPU 上以小尺寸跑。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.config import (
    MRPConfig, TESModelConfig, TESMultiScaleConfig, TESTransportConfig,
)
from lbmnca.nca.tes.models.h3t_warpresidual_nca import H3TWarpResidualNCA
from lbmnca.nca.tes.models.model_factory import build_tes_model


def _make_cfg() -> TESModelConfig:
    return TESModelConfig(
        arch="h3t_warpresidual",
        state_dim=29, static_dim=4, physics_dim=9, hidden_dim=16,
        cond_dim=19, encoder_hidden=32, nca_channels=16,
        mrp=MRPConfig(enabled=True, num_learnable=8, hormone_dim=4),
        multiscale=TESMultiScaleConfig(ratios=[8, 32], fine_substeps=2, scale_condition=True),
        transport=TESTransportConfig(warp_ratio=8, enc_hidden=16, disp_max_px=20.0, gate_bias=-4.0),
    )


def _inputs(b=2, h=32, w=32, cfg=None):
    cfg = cfg or _make_cfg()
    static = torch.randn(b, cfg.static_dim, h, w)
    static[:, 0] = 1.0  # SDF>0 → all fluid
    ic = torch.randn(b, cfg.physics_dim, h, w) * 0.1
    cond = torch.randn(b, cfg.cond_dim)
    return static, ic, cond


def test_factory_builds_h3t() -> None:
    model = build_tes_model(_make_cfg())
    assert isinstance(model, H3TWarpResidualNCA)


def test_forward_shape_and_trajectory() -> None:
    cfg = _make_cfg()
    model = build_tes_model(cfg).eval()
    static, ic, cond = _inputs(cfg=cfg)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    assert state.shape[1] == cfg.state_dim
    with torch.no_grad():
        final, traj = model(state, cond, fm, n_steps=4, return_trajectory=True)
    assert final.shape == state.shape
    assert len(traj) == 4
    for t in traj:
        assert t.shape == (2, cfg.physics_dim, 32, 32)


def test_amp_finite() -> None:
    if not torch.cuda.is_available():
        pytest.skip("AMP finite test 需 CUDA")
    cfg = _make_cfg()
    model = build_tes_model(cfg).cuda().eval()
    static, ic, cond = _inputs(cfg=cfg)
    static, ic, cond = static.cuda(), ic.cuda(), cond.cuda()
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    dseq = torch.full((2, 4), 0.2, device="cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        final, traj = model(state, cond, fm, 4, True, delta_ctu_seq=dseq)
    assert torch.isfinite(final).all()
    for t in traj:
        assert torch.isfinite(t).all()


def test_zero_init_identity_warp_branch() -> None:
    """初始：disp_head/corr_head zero-init → warped≈prev_phys、corr=0、gate≈sigmoid(gate_bias)。"""
    cfg = _make_cfg()
    model = build_tes_model(cfg).eval()
    static, ic, cond = _inputs(cfg=cfg)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    prev_phys = state[:, cfg.static_dim:cfg.static_dim + cfg.physics_dim]
    with torch.no_grad():
        warped, corr, gate, _ = model._warp_branch(state, prev_phys, fm, None, 0)
    # disp=0 → grid_sample 取 pixel center → warped == prev_phys（fluid 全開）
    assert torch.allclose(warped, prev_phys, atol=1e-4)
    assert torch.allclose(corr, torch.zeros_like(corr), atol=1e-6)
    expected_gate = torch.sigmoid(torch.tensor(cfg.transport.gate_bias))
    assert torch.allclose(gate, expected_gate.expand_as(gate), atol=1e-4)


def test_zero_init_output_near_residual() -> None:
    """zero-init 整體 forward 不爆，且 blended 與純 residual（gate≈0）接近。"""
    cfg = _make_cfg()
    model = build_tes_model(cfg).eval()
    static, ic, cond = _inputs(cfg=cfg)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    with torch.no_grad():
        final, traj = model(state, cond, fm, 2, True)
    assert torch.isfinite(final).all()
    # gate≈0.018 → blended 由 residual 主導，physics 量級合理（非爆炸）
    assert traj[0].abs().max() < 50.0


def test_delta_ctu_seq_scales_displacement() -> None:
    """大 delta_ctu → 大 disp（disp_max * delta_ctu scale）。"""
    cfg = _make_cfg()
    model = build_tes_model(cfg).eval()
    static, ic, cond = _inputs(cfg=cfg)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    # 把 disp_head 設非零，否則 disp 恆 0
    torch.nn.init.normal_(model.disp_head.weight, std=0.5)
    torch.nn.init.normal_(model.disp_head.bias, std=0.5)
    with torch.no_grad():
        _, _, _, m_small = model._warp_branch(state, state[:, 4:13], fm,
                                              torch.full((2, 1), 0.2), 0)
        _, _, _, m_big = model._warp_branch(state, state[:, 4:13], fm,
                                            torch.full((2, 1), 2.0), 0)
    assert float(m_big["disp_max_px"]) > float(m_small["disp_max_px"])


def test_solid_mask_no_write() -> None:
    """solid 區（fluid_mask=0）輸出 physics 必為 0。"""
    cfg = _make_cfg()
    model = build_tes_model(cfg).eval()
    static, ic, cond = _inputs(cfg=cfg)
    static[:, 0, :, :16] = -1.0  # 左半 solid (SDF<0)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    with torch.no_grad():
        _, traj = model(state, cond, fm, 3, True)
    solid = (fm[:, 0, :, :16] < 0.5)
    for t in traj:
        phys_solid = t[:, :, :, :16]
        assert torch.allclose(phys_solid[solid.unsqueeze(1).expand_as(phys_solid)],
                              torch.zeros(1), atol=1e-6)
