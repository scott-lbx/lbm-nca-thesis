"""HF-1b：P/G FiLM 分離 — separated G-FiLM path identity-init test。

HF-1b 在 H3TStreamCollideNCA 上加一條 separated geometry FiLM path
（gvec_mode='gated', g_vec_dim=14）。gated 模式下 alpha_g 為 nn.Parameter(zeros)，
故 init 時 gamma_g/beta_g 乘 0 → G-path 對 forward 貢獻為 0
→ forward 等同於關掉 G-path（= HF-1a baseline 行為）。

本 test 證明 identity-init：對同一個 gated model，餵 g_vec_norm 與餵 None
forward 輸出必須 allclose（因 alpha_g=0）。過不了不准進主跑（HF_code_plan §9）。
"""
from __future__ import annotations

import sys
from pathlib import Path

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
from lbmnca.nca.tes.models.model_factory import build_tes_model  # noqa: E402


def _make_cfg_gated() -> TESModelConfig:
    return TESModelConfig(
        arch="h3t_streamcollide",
        state_dim=29,
        static_dim=4,
        physics_dim=9,
        hidden_dim=16,
        cond_dim=19,
        encoder_hidden=32,
        nca_channels=16,
        g_vec_dim=14,
        gvec_mode="gated",
        mrp=MRPConfig(enabled=True, num_learnable=8, hormone_dim=4),
        multiscale=TESMultiScaleConfig(ratios=[8, 32], fine_substeps=2, scale_condition=True),
        transport=TESTransportConfig(warp_ratio=8, enc_hidden=16, disp_max_px=20.0, gate_bias=-4.0),
    )


def test_alpha_g_zero_init() -> None:
    """gated 模式 alpha_g 必須 zero-init（G-path 起步無貢獻的前提）。"""
    model = build_tes_model(_make_cfg_gated())
    assert model.alpha_g is not None
    assert torch.allclose(model.alpha_g.detach(), torch.zeros_like(model.alpha_g))


def test_separated_gfilm_identity_init() -> None:
    """init 時 separated G-FiLM path 貢獻為 0 → 有/無 g_vec_norm forward 一致。"""
    cfg = _make_cfg_gated()
    model = build_tes_model(cfg).eval()
    b, h, w = 2, 32, 32
    static = torch.randn(b, cfg.static_dim, h, w)
    static[:, 0] = 1.0
    ic = torch.randn(b, cfg.physics_dim, h, w) * 0.1
    cond = torch.randn(b, cfg.cond_dim)
    g_vec = torch.randn(b, cfg.g_vec_dim)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    dseq = torch.full((b, 4), 0.2)
    with torch.no_grad():
        out_with_g = model(state, cond, fm, n_steps=4, delta_ctu_seq=dseq, g_vec_norm=g_vec)
        out_no_g = model(state, cond, fm, n_steps=4, delta_ctu_seq=dseq, g_vec_norm=None)
    assert torch.isfinite(out_with_g).all()
    assert torch.allclose(out_with_g, out_no_g, atol=1e-6), (
        "alpha_g=0 時 separated G-FiLM path 應對 forward 無貢獻（identity-init 失敗）"
    )
