"""HF-1g：create / source branch identity-init test。

create_head 最終層 zero-init → create_residual=0 → forward 等同 streamcollide baseline。
過不了不准進主跑（HF_code_plan §9）。
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
from lbmnca.nca.tes.models.operator_gated_streamcollide_nca import (  # noqa: E402
    OperatorGatedStreamCollideNCA,
)


def _cfg() -> TESModelConfig:
    return TESModelConfig(
        arch="operator_gated_streamcollide",
        state_dim=29, static_dim=4, physics_dim=9, hidden_dim=16,
        cond_dim=19, encoder_hidden=32, nca_channels=16,
        g_vec_dim=0, gvec_mode="none",
        mrp=MRPConfig(enabled=True, num_learnable=8, hormone_dim=4),
        multiscale=TESMultiScaleConfig(ratios=[8, 32], fine_substeps=2, scale_condition=True),
        transport=TESTransportConfig(warp_ratio=8, enc_hidden=16, disp_max_px=20.0,
                                     gate_bias=-4.0, use_create=True),
    )


def test_create_wants_prior_ctx() -> None:
    model = OperatorGatedStreamCollideNCA(_cfg())
    assert model.wants_prior_ctx, "use_create 需 prior_ctx 取 front mask"
    # 最終層 zero-init
    assert torch.allclose(model.create_head[-1].weight.detach(),
                          torch.zeros_like(model.create_head[-1].weight))


def test_create_identity_at_init() -> None:
    cfg = _cfg()
    model = OperatorGatedStreamCollideNCA(cfg).eval()
    static = torch.zeros(2, 4, 32, 32); static[:, 0] = 1.0
    x = (torch.arange(32, dtype=torch.float32) + 0.5) / 32
    static[:, 2] = x.view(1, 1, 32).expand(2, 32, 32)
    ic = torch.randn(2, 9, 32, 32) * 0.1
    cond = torch.randn(2, 19)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    dseq = torch.full((2, 3), 0.2)
    pctx = {
        "current_ctu_raw_seq": torch.full((2, 3), 0.5),
        "l_char_px": torch.tensor([106.0, 80.0]),
        "prior_vel_norm": torch.tensor([0.3, 0.0]),
    }
    with torch.no_grad():
        o_create = model(state, cond, fm, 3, delta_ctu_seq=dseq, prior_ctx=pctx)
        model._use_create = False
        o_off = model(state, cond, fm, 3, delta_ctu_seq=dseq, prior_ctx=pctx)
    assert torch.allclose(o_create, o_off, atol=1e-6), "create head zero-init 應 == streamcollide（identity）"
