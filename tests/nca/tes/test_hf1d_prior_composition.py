"""HF-1d：deterministic inflow prior prediction composition 測試。

1. prior_mode='off' → forward 逐元素等於 H3TStreamCollideNCA（arch wrapper 本身 identity）。
2. prior_mode='composition' 但 W_prior=0（init）→ 有/無 prior_ctx forward allclose（identity-init）。
3. 方向（HF_code_plan §9 必加）：W_prior>0、transition 帶內，jx/jy 鬆弛向 prior_vel
   （往 inlet velocity 推；無 target 洩漏，僅由 current_ctu/L_char/X_01 決定 mask）。
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
from lbmnca.nca.tes.models.h3t_streamcollide_nca import H3TStreamCollideNCA  # noqa: E402
from lbmnca.nca.tes.models.operator_gated_streamcollide_nca import (  # noqa: E402
    OperatorGatedStreamCollideNCA,
)


def _cfg(prior_mode: str = "off") -> TESModelConfig:
    return TESModelConfig(
        arch="operator_gated_streamcollide",
        state_dim=29, static_dim=4, physics_dim=9, hidden_dim=16,
        cond_dim=19, encoder_hidden=32, nca_channels=16,
        g_vec_dim=0, gvec_mode="none",
        mrp=MRPConfig(enabled=True, num_learnable=8, hormone_dim=4),
        multiscale=TESMultiScaleConfig(ratios=[8, 32], fine_substeps=2, scale_condition=True),
        transport=TESTransportConfig(
            warp_ratio=8, enc_hidden=16, disp_max_px=20.0, gate_bias=-4.0,
            prior_mode=prior_mode, prior_w_front=0.1,
        ),
    )


def _inputs(cfg, b=2, h=32, w=32):
    static = torch.randn(b, cfg.static_dim, h, w)
    static[:, 0] = 1.0
    # ch2 = X_01 ∈[0,1]（inlet 在左）
    x = (torch.arange(w, dtype=torch.float32) + 0.5) / w
    static[:, 2] = x.view(1, 1, w).expand(b, h, w)
    ic = torch.randn(b, cfg.physics_dim, h, w) * 0.1
    cond = torch.randn(b, cfg.cond_dim)
    return static, ic, cond


def test_prior_off_equals_streamcollide() -> None:
    torch.manual_seed(0)
    parent = H3TStreamCollideNCA(_cfg("off")).eval()
    torch.manual_seed(0)
    child = OperatorGatedStreamCollideNCA(_cfg("off")).eval()
    cfg = _cfg("off")
    static, ic, cond = _inputs(cfg)
    fm = child.get_fluid_mask(static)
    state = child.init_state(static, ic)
    dseq = torch.full((2, 3), 0.2)
    with torch.no_grad():
        o_parent = parent(state, cond, fm, 3, delta_ctu_seq=dseq)
        o_child = child(state, cond, fm, 3, delta_ctu_seq=dseq)
    assert torch.allclose(o_parent, o_child, atol=1e-6), "prior_mode=off 應逐元素等於 streamcollide"
    assert not child.wants_prior_ctx


def test_prior_composition_identity_at_init() -> None:
    cfg = _cfg("composition")
    model = OperatorGatedStreamCollideNCA(cfg).eval()
    assert model.wants_prior_ctx
    assert torch.allclose(model.W_prior.detach(), torch.zeros(1))
    static, ic, cond = _inputs(cfg)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    dseq = torch.full((2, 3), 0.2)
    pctx = {
        "current_ctu_raw_seq": torch.full((2, 3), 0.5),
        "l_char_px": torch.tensor([106.0, 80.0]),
        "prior_vel_norm": torch.tensor([0.5, 0.0]),
    }
    with torch.no_grad():
        o_with = model(state, cond, fm, 3, delta_ctu_seq=dseq, prior_ctx=pctx)
        o_none = model(state, cond, fm, 3, delta_ctu_seq=dseq, prior_ctx=None)
    assert torch.allclose(o_with, o_none, atol=1e-6), "W_prior=0 時 prior 應無貢獻（identity-init）"


def test_prior_relaxes_velocity_toward_inlet() -> None:
    """W_prior>0、transition 帶內 → jx/jy 往 prior_vel 鬆弛（方向正確，無 target）。"""
    cfg = _cfg("composition")
    model = OperatorGatedStreamCollideNCA(cfg).eval()
    model.W_prior.data.fill_(1.0)
    b, h, w = 1, 16, 64
    static = torch.zeros(b, 4, h, w)
    static[:, 0] = 1.0
    x = (torch.arange(w, dtype=torch.float32) + 0.5) / w
    static[:, 2] = x.view(1, 1, w).expand(b, h, w)
    fm = torch.ones(b, 1, h, w)
    # prev physics：jx=0, jy=0.5（待被往 prior 推）
    prev = torch.zeros(b, 9, h, w)
    prev[:, 5] = 0.5
    hidden = torch.zeros(b, cfg.hidden_dim, h, w)
    state_in = torch.cat([static, prev, hidden], dim=1)
    model._prior_runtime = {
        "current_ctu_raw_seq": torch.tensor([[8.0]]),   # 大 ctu → front 在域內，transition 帶可見
        "l_char_px": torch.tensor([106.0]),
        "prior_vel_norm": torch.tensor([0.8, 0.0]),     # prior：jx→0.8, jy→0
    }
    new_phys, _ = model._compose_prior(state_in, prev, 0, fm)
    # 在 transition 帶（front 附近）jx 應被往 0.8 推（>0），jy 往 0 推（<0.5）
    from lbmnca.nca.tes.models.operator_gated_streamcollide_nca import _JX, _JY
    assert new_phys[:, _JX].max() > 1e-3, "transition 帶 jx 應往 prior_vel(+) 鬆弛"
    assert new_phys[:, _JY].min() < 0.5, "transition 帶 jy 應往 prior_vel(0) 鬆弛"
    # transition 帶外（極右 outlet，X_01≈1，front 未到）幾乎不動
    assert torch.allclose(new_phys[:, _JX, :, -1], torch.zeros(b, h), atol=1e-2)
