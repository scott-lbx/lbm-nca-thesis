"""HF-1f：operator gate（delta-conditioned stream 強度）identity-init test。

⚠️ cold-0.2 為單一 regime/delta → G0/G1 gate 輸入（delta）為常數 → 結構性 no-op；
gate 真正評估在 HF-3 multi-delta。本 test 只保證 identity-init（zero-init → W_stream=1
→ forward 等同 streamcollide baseline）。
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


def _cfg(gate_mode: str) -> TESModelConfig:
    return TESModelConfig(
        arch="operator_gated_streamcollide",
        state_dim=29, static_dim=4, physics_dim=9, hidden_dim=16,
        cond_dim=19, encoder_hidden=32, nca_channels=16,
        g_vec_dim=0, gvec_mode="none",
        mrp=MRPConfig(enabled=True, num_learnable=8, hormone_dim=4),
        multiscale=TESMultiScaleConfig(ratios=[8, 32], fine_substeps=2, scale_condition=True),
        transport=TESTransportConfig(warp_ratio=8, enc_hidden=16, disp_max_px=20.0,
                                     gate_bias=-4.0, gate_mode=gate_mode),
    )


def test_gate_w_stream_unity_at_init() -> None:
    model = OperatorGatedStreamCollideNCA(_cfg("g1")).eval()
    for dt in (0.2, 1.0, 2.0):
        model._cur_dt = torch.tensor([dt, dt])
        w = model._w_stream()
        assert torch.allclose(w.view(-1), torch.ones(2), atol=1e-5), f"W_stream 應 init=1（dt={dt}）"


def test_gate_identity_forward_at_init() -> None:
    """g1 gate zero-init → forward 等同關閉 gate（W_stream=1）。"""
    cfg = _cfg("g1")
    model = OperatorGatedStreamCollideNCA(cfg).eval()
    static = torch.randn(2, 4, 32, 32); static[:, 0] = 1.0
    ic = torch.randn(2, 9, 32, 32) * 0.1
    cond = torch.randn(2, 19)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    dseq = torch.full((2, 3), 1.0)
    with torch.no_grad():
        o_g1 = model(state, cond, fm, 3, delta_ctu_seq=dseq)
        model._gate_mode = "off"
        o_off = model(state, cond, fm, 3, delta_ctu_seq=dseq)
    assert torch.allclose(o_g1, o_off, atol=1e-6), "gate zero-init 應 == 關閉 gate（identity-init）"
