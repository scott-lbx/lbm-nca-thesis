"""EG-1 mean_gate_eval 純函式測試（synthetic tensors，無 zarr/GPU）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.eval.mean_gate_eval import (
    ke_curve_convergence,
    mean_gate_metrics,
    self_rollout_drift,
)

B, H, W = 2, 32, 32


def _pair() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    tgt = torch.rand(B, 9, H, W) * 0.4 - 0.2
    fluid = torch.ones(B, 1, H, W)
    return tgt.clone(), tgt, fluid


class TestMeanGateMetrics:
    def test_identity(self):
        pred, tgt, fluid = _pair()
        m = mean_gate_metrics(pred, tgt, fluid)
        assert m["mean_mse"] == pytest.approx(0.0, abs=1e-10)
        assert m["mean_rel_l2"] == pytest.approx(0.0, abs=1e-6)
        assert m["mean_jx_ssim"] == pytest.approx(1.0, abs=1e-4)

    def test_constant_error_mse(self):
        """fluid 區常數誤差 ε → mean_mse = ε²。"""
        err = 0.1
        _, tgt, fluid = _pair()
        pred = tgt + err
        m = mean_gate_metrics(pred, tgt, fluid)
        assert m["mean_mse"] == pytest.approx(err * err, rel=1e-4)

    def test_rel_l2_scale(self):
        """pred = 2×tgt → rel_l2 = ||tgt||/||tgt|| = 1。"""
        _, tgt, fluid = _pair()
        m = mean_gate_metrics(2.0 * tgt, tgt, fluid)
        assert m["mean_rel_l2"] == pytest.approx(1.0, rel=1e-4)

    def test_solid_pixels_excluded(self):
        """誤差只在 solid 區 → 三指標不受影響。"""
        _, tgt, fluid = _pair()
        fluid[:, :, :, : W // 2] = 0.0
        pred = tgt.clone()
        pred[:, :, :, : W // 2] += 5.0  # solid 區大誤差
        m = mean_gate_metrics(pred, tgt, fluid)
        assert m["mean_mse"] == pytest.approx(0.0, abs=1e-10)
        assert m["mean_rel_l2"] == pytest.approx(0.0, abs=1e-6)


class TestKeCurveConvergence:
    def test_flat_curve_converged(self):
        ke = [1.0] * 50
        out = ke_curve_convergence(ke)
        assert out["ke_converged"] == 1.0
        assert out["ke_ratio"] == pytest.approx(0.0, abs=1e-9)
        assert out["ke_final"] == 1.0

    def test_ramp_curve_not_converged(self):
        ke = list(np.linspace(1.0, 3.0, 50))
        out = ke_curve_convergence(ke)
        assert out["ke_converged"] == 0.0
        assert out["ke_ratio"] > 0.05

    def test_short_curve_no_crash(self):
        out = ke_curve_convergence([1.0, 1.0])
        assert out["ke_converged"] == 0.0
        assert np.isnan(out["ke_ratio"])

    def test_decaying_to_plateau_converged(self):
        """前段衰減、後半平台 → 後半兩窗應判 converged（窗只看曲線後半）。"""
        ke = list(np.linspace(2.0, 1.0, 25)) + [1.0] * 25
        out = ke_curve_convergence(ke)
        assert out["ke_converged"] == 1.0


class TestSelfRolloutDrift:
    def test_drift_sign(self):
        out = self_rollout_drift(mse_at_k=0.10, mse_at_2k=0.15)
        assert out["mean_drift_2k"] == pytest.approx(0.05)
        out2 = self_rollout_drift(mse_at_k=0.10, mse_at_2k=0.08)
        assert out2["mean_drift_2k"] < 0.0
