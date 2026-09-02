"""EG-4 decompose_inference 兩階段 handoff 測試（fake model，無 GPU/ckpt）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.config import TESModelConfig
from lbmnca.nca.tes.inference.decompose_inference import (
    compose_field,
    decompose_rollout,
    run_mean_phase,
    run_turb_phase,
    seed_fluct,
)

B, SD, PD, H, W = 1, 4, 9, 8, 8


class _FakeMeanModel:
    """指數收斂 fake mean_module：phys ← phys + 0.5×(target − phys)。

    符合 TESNCA call 介面（state [B, SD+PD, H, W]，n 步推進）**與 ``_cfg`` 屬性**。

    ★為何需要 ``_cfg``（R3-G1，commit ba499aa「ke_anchor 收進 protocol」）：
    ``run_mean_phase`` 進迴圈前會呼叫
    ``protocol.forward_anchor_kwargs(model, state, fluid_mask)``，該函式經
    ``resolve_ke_anchor_from_model`` 直接讀 ``model._cfg``（禁 fallback，缺欄位即報錯）。
    真實模型的 ``_cfg`` 由 ``build_tes_model`` → 各 assembly 類別的
    ``self._cfg = cfg`` 設定（如 ``models/assembly/h_base.py:41``），所以 fake model
    也必須帶一份**真的** ``TESModelConfig``，否則測到的介面與生產路徑不同。
    """

    def __init__(self, target: torch.Tensor) -> None:
        self._target = target  # [B, PD, H, W]
        # transport.ke_anchor_mode 預設 'off'（協定建議值）→ forward_anchor_kwargs 回 {}，
        # 故下方 __call__ 不收 ke_anchor；欄位維度與 SD/PD 對齊，不用預設值矇混。
        self._cfg = TESModelConfig(state_dim=SD + PD, static_dim=SD, physics_dim=PD)

    def __call__(
        self, state, cond_norm, fluid_mask, n, *,
        return_trajectory=False, g_vec_norm=None,
    ):
        for _ in range(n):
            phys = state[:, SD : SD + PD]
            state = state.clone()
            state[:, SD : SD + PD] = phys + 0.5 * (self._target - phys)
        return state


def _setup() -> tuple[_FakeMeanModel, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    target = torch.rand(B, PD, H, W) * 0.4
    model = _FakeMeanModel(target)
    state = torch.zeros(B, SD + PD, H, W)
    state[:, SD : SD + PD] = torch.rand(B, PD, H, W)  # 任意 IC
    cond = torch.zeros(B, 19)
    fluid = torch.ones(B, 1, H, W)
    return model, state, cond, fluid, target


class TestRunMeanPhase:
    def test_converges_to_target(self):
        model, state, cond, fluid, target = _setup()
        res = run_mean_phase(
            model, state, cond, fluid,
            static_dim=SD, physics_dim=PD, max_steps=100,
        )
        assert res.converged
        assert res.n_steps < 100                      # 提前停
        assert float((res.mean_phys - target).abs().max()) < 1e-2
        assert len(res.ke_curve) == res.n_steps
        assert res.ke_ratio < 0.05

    def test_max_steps_cap_not_converged(self):
        """max_steps 太小 → converged=False 但仍回傳 mean。"""
        model, state, cond, fluid, _ = _setup()
        res = run_mean_phase(
            model, state, cond, fluid,
            static_dim=SD, physics_dim=PD, max_steps=4, min_steps=2,
        )
        assert not res.converged
        assert res.n_steps == 4


class TestSeedFluct:
    def test_nonzero_deterministic(self):
        mean = torch.zeros(B, PD, H, W)
        f1 = seed_fluct(mean, amplitude=1e-3, seed=42)
        f2 = seed_fluct(mean, amplitude=1e-3, seed=42)
        assert float(f1.abs().max()) > 0.0            # M7-1：非全 0
        assert torch.equal(f1, f2)                     # 固定 seed 可重現
        assert float(f1.std()) == pytest.approx(1e-3, rel=0.2)


class TestRunTurbPhase:
    def test_yields_composed_frames(self):
        """fake turb：fluct 相位旋轉 → composed = mean + fluct 逐步可迭代。"""
        mean = torch.full((B, PD, H, W), 0.5)

        def _rotate(fluct: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            return -fluct                              # 週期 2 的最小 limit-cycle

        init = torch.full((B, PD, H, W), 0.1)
        frames = list(run_turb_phase(_rotate, mean, init_fluct=init, n_steps=4))
        assert len(frames) == 4
        assert torch.allclose(frames[0], mean - 0.1)
        assert torch.allclose(frames[1], mean + 0.1)   # 反相震盪繞 mean
        assert torch.allclose(frames[2], frames[0])

    def test_solid_mask_zeroes_fluct(self):
        mean = torch.zeros(B, PD, H, W)
        fluid = torch.ones(B, 1, H, W)
        fluid[:, :, :, : W // 2] = 0.0

        def _ident(fluct, m):
            return fluct + 1.0

        frames = list(run_turb_phase(
            _ident, mean, init_fluct=torch.zeros(B, PD, H, W),
            n_steps=1, fluid_mask=fluid,
        ))
        assert float(frames[0][:, :, :, : W // 2].abs().max()) == 0.0
        assert float(frames[0][:, :, :, W // 2 :].abs().min()) == 1.0


class TestDecomposeRollout:
    def test_handoff_chain(self):
        """Phase1 收斂 mean → Phase2 generator 以該 mean 為基準震盪。"""
        model, state, cond, fluid, target = _setup()

        def _rotate(fluct, m):
            return -fluct

        mean_res, frames_gen = decompose_rollout(
            model, _rotate, state, cond, fluid,
            static_dim=SD, physics_dim=PD,
            mean_max_steps=100, turb_steps=6,
        )
        assert mean_res.converged
        frames = list(frames_gen)
        assert len(frames) == 6
        # 連續兩幀平均 = mean（繞 mean 對稱震盪），且接近 GT target
        pair_mean = 0.5 * (frames[0] + frames[1])
        assert float((pair_mean - mean_res.mean_phys).abs().max()) < 1e-6
        assert float((mean_res.mean_phys - target).abs().max()) < 1e-2

    def test_compose_field(self):
        m = torch.ones(1, 9, 4, 4)
        f = torch.full((1, 9, 4, 4), 0.25)
        assert torch.allclose(compose_field(m, f), torch.full((1, 9, 4, 4), 1.25))
