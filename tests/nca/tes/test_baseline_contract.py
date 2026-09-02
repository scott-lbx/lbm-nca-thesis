"""Baseline 對照模型 contract / 形狀測試（CPU、小張量）。

驗證 `BaselineStepperNCA` / `SteadyStateUNetNCA` 滿足 trainer/eval forward
contract（見 docs/research_plan/baseline/03_port_spec.md），以及 model_factory
對 4 個 baseline arch 的建構與前向。
"""
from __future__ import annotations

import torch

from lbmnca.nca.tes.config import BaselineConfig, TESModelConfig
from lbmnca.nca.tes.models.baselines.baseline_stepper import (
    BaselineStepperNCA,
    SteadyStateUNetNCA,
)
from lbmnca.nca.tes.models.model_factory import build_tes_model


def _cfg(arch: str = "baseline_ar_fno", backbone: str = "dummy", hidden: int = 0) -> TESModelConfig:
    return TESModelConfig(
        arch=arch,
        state_dim=4 + 9 + hidden,
        static_dim=4,
        physics_dim=9,
        hidden_dim=hidden,
        cond_dim=19,
        g_vec_dim=0,
        gvec_mode="none",
        baseline=BaselineConfig(backbone=backbone),
    )


def _inputs(model, b: int = 2, h: int = 32, w: int = 48, k: int = 3):
    static = torch.randn(b, 4, h, w)
    ic = torch.randn(b, 9, h, w)
    state = model.init_state(static, ic)
    cond = torch.randn(b, 19)
    fm = (torch.rand(b, 1, h, w) > 0.3).float()
    dseq = torch.full((b, k), 1.0)
    return state, cond, fm, dseq


def test_stepper_dummy_contract():
    m = BaselineStepperNCA(_cfg(), backbone="dummy")
    state, cond, fm, dseq = _inputs(m)
    assert state.shape == (2, 13, 32, 48)
    out, traj = m(state, cond, fm, 3, return_trajectory=True, delta_ctu_seq=dseq)
    assert out.shape == (2, 13, 32, 48)
    assert len(traj) == 3 and traj[0].shape == (2, 9, 32, 48)
    # solid 區歸零
    assert torch.allclose(traj[-1] * (1.0 - fm), torch.zeros_like(traj[-1]))
    # 無 trajectory 路徑
    out2 = m(state, cond, fm, 2, delta_ctu_seq=dseq[:, :2])
    assert out2.shape == (2, 13, 32, 48)


def test_steady_dummy_contract():
    m = SteadyStateUNetNCA(_cfg(arch="baseline_unet_steady"))
    state, cond, fm, _ = _inputs(m)
    out, traj = m(state, cond, fm, 4, return_trajectory=True)
    assert out.shape == (2, 13, 32, 48)
    assert len(traj) == 4
    # 穩態：每步同值（結構上無法續推）
    for t in traj[1:]:
        assert torch.equal(traj[0], t)


def test_tfc_runs():
    m = BaselineStepperNCA(_cfg(), backbone="dummy")
    state, cond, fm, dseq = _inputs(m)
    gt = torch.randn(2, 3, 9, 32, 48)
    out, traj = m(
        state, cond, fm, 3, return_trajectory=True, delta_ctu_seq=dseq,
        tf_ratio=1.0, gt_traj=gt,
    )
    assert len(traj) == 3 and out.shape == (2, 13, 32, 48)


def test_factory_all_baselines():
    archs = [
        "baseline_ar_fno",
        "baseline_unet_modern",
        "baseline_dilated_resnet",
        "baseline_unet_steady",
    ]
    for arch in archs:
        # backbone 非 dummy → 真 backbone（stepper arch 由 factory 強制覆寫）
        m = build_tes_model(_cfg(arch=arch, backbone="fno"))
        state, cond, fm, dseq = _inputs(m, h=32, w=48, k=2)
        out, traj = m(state, cond, fm, 2, return_trajectory=True, delta_ctu_seq=dseq)
        assert out.shape == (2, 13, 32, 48), arch
        assert traj[0].shape == (2, 9, 32, 48), arch


def test_init_state_hidden0():
    m = BaselineStepperNCA(_cfg(hidden=0), backbone="dummy")
    s = m.init_state(torch.randn(2, 4, 16, 16), torch.zeros(2, 9, 16, 16))
    assert s.shape == (2, 13, 16, 16)
