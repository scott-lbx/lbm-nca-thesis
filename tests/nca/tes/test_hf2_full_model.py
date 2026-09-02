"""HF-2：full operator-gated model 整合 identity-init test。

HF-2 arch = streamcollide + velocity-disp + gate(g1)（prior/create off）。
在 safe init（alpha_vel=0、gate W_stream=1）下，forward 必須等同 streamcollide baseline。
此為 HF-3 scale-up 的 expert 架構，integration 不可在 init 就偏離 baseline。
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


def _cfg_full() -> TESModelConfig:
    return TESModelConfig(
        arch="operator_gated_streamcollide",
        state_dim=29, static_dim=4, physics_dim=9, hidden_dim=16,
        cond_dim=19, encoder_hidden=32, nca_channels=16,
        g_vec_dim=0, gvec_mode="none",
        mrp=MRPConfig(enabled=True, num_learnable=8, hormone_dim=4),
        multiscale=TESMultiScaleConfig(ratios=[8, 32], fine_substeps=2, scale_condition=True),
        transport=TESTransportConfig(warp_ratio=8, enc_hidden=16, disp_max_px=20.0,
                                     gate_bias=-4.0, use_velocity_disp=True, gate_mode="g1"),
    )


def test_full_model_identity_at_init() -> None:
    """velocity-disp + gate 同開、safe init → forward 等同 streamcollide baseline。"""
    cfg = _cfg_full()
    model = OperatorGatedStreamCollideNCA(cfg).eval()
    assert torch.allclose(model.alpha_vel.detach(), torch.zeros(1))
    static = torch.randn(2, 4, 32, 32); static[:, 0] = 1.0
    ic = torch.randn(2, 9, 32, 32) * 0.1
    cond = torch.randn(2, 19)
    fm = model.get_fluid_mask(static)
    state = model.init_state(static, ic)
    dseq = torch.full((2, 3), 0.2)
    with torch.no_grad():
        o_full = model(state, cond, fm, 3, delta_ctu_seq=dseq)
        # 關閉兩 operator → 應與 full-at-init 逐元素相等
        model._use_velocity_disp = False
        model._gate_mode = "off"
        o_base = model(state, cond, fm, 3, delta_ctu_seq=dseq)
    assert torch.allclose(o_full, o_base, atol=1e-6), "full model safe-init 應 == streamcollide baseline"
