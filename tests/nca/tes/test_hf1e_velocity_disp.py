"""HF-1e：velocity-conditioned displacement 測試。

1. identity：use_velocity_disp=True 但 alpha_vel=0（init）→ _disp_coarse == 純 tanh(disp_head)，
   forward 逐元素等於 streamcollide baseline（過不了不准進主跑）。
2. 方向（HF_code_plan §9 必加，防 grid_sample sign 反向）：均勻 +jx（downstream 速度）、
   alpha_vel>0、disp_head=0 → warp 後 blob 質心往 +x（downstream）推進。
   （grid_sample backward gather：sample=base-disp，正 disp 應使內容往 +x。）
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
    _JX,
)


def _cfg(use_vel: bool) -> TESModelConfig:
    return TESModelConfig(
        arch="operator_gated_streamcollide",
        state_dim=29, static_dim=4, physics_dim=9, hidden_dim=16,
        cond_dim=19, encoder_hidden=32, nca_channels=16,
        g_vec_dim=0, gvec_mode="none",
        mrp=MRPConfig(enabled=True, num_learnable=8, hormone_dim=4),
        multiscale=TESMultiScaleConfig(ratios=[8, 32], fine_substeps=2, scale_condition=True),
        transport=TESTransportConfig(warp_ratio=8, enc_hidden=16, disp_max_px=20.0,
                                     gate_bias=-4.0, use_velocity_disp=use_vel),
    )


def test_velocity_disp_identity_at_init() -> None:
    """alpha_vel=0（init）→ 等於 streamcollide baseline（同 seed）。"""
    torch.manual_seed(0)
    parent = H3TStreamCollideNCA(_cfg(False)).eval()
    torch.manual_seed(0)
    child = OperatorGatedStreamCollideNCA(_cfg(True)).eval()
    assert torch.allclose(child.alpha_vel.detach(), torch.zeros(1))
    cfg = _cfg(True)
    static = torch.randn(2, 4, 32, 32); static[:, 0] = 1.0
    ic = torch.randn(2, 9, 32, 32) * 0.1
    cond = torch.randn(2, 19)
    fm = child.get_fluid_mask(static)
    state = child.init_state(static, ic)
    dseq = torch.full((2, 3), 0.2)
    with torch.no_grad():
        op = parent(state, cond, fm, 3, delta_ctu_seq=dseq)
        oc = child(state, cond, fm, 3, delta_ctu_seq=dseq)
    assert torch.allclose(op, oc, atol=1e-6), "alpha_vel=0 應等於 streamcollide baseline"


def test_velocity_disp_advects_downstream() -> None:
    """均勻 +jx + alpha_vel>0 → blob 質心往 +x（disp 方向正確）。"""
    cfg = _cfg(True)
    model = OperatorGatedStreamCollideNCA(cfg).eval()
    model.alpha_vel.data.fill_(0.6)
    # disp_head zero-init（無自由 disp）→ disp 純由 velocity 決定
    b, h, w = 1, 32, 32
    static = torch.zeros(b, 4, h, w); static[:, 0] = 1.0
    fm = torch.ones(b, 1, h, w)
    prev = torch.zeros(b, 9, h, w)
    # blob 在 ch0（rho）中央
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    blob = torch.exp(-(((xx - 12.0) ** 2 + (yy - 16.0) ** 2) / 12.0))
    prev[0, 0] = blob
    # 均勻 +jx 速度（downstream），jy=0
    prev[0, _JX] = 0.5
    hidden = torch.zeros(b, cfg.hidden_dim, h, w)
    state_in = torch.cat([static, prev, hidden], dim=1)
    dseq = torch.full((b, 1), 2.0)   # 大 Δt → 明顯位移
    with torch.no_grad():
        warped, _ = model._stream(state_in, prev, fm, dseq, 0)
    def centroid_x(field):
        m = field.clamp_min(0)
        xs = torch.arange(w, dtype=torch.float32)
        return float((m.sum(0) * xs).sum() / m.sum().clamp_min(1e-6))
    cx0 = centroid_x(prev[0, 0])
    cx1 = centroid_x(warped[0, 0])
    assert cx1 > cx0 + 0.5, f"blob 應往 downstream(+x) 推進：{cx0:.2f} -> {cx1:.2f}"
