"""MeanPool 跨案例鏈式接力測試（A2，WARM_ACCUMULATION_PROPOSAL_2026-07-15）。

1. frac=0：chain_writeback 為 no-op（零行為變更迴歸）。
2. frac=1 + child 在池中：child slot 的 physics 應變成 parent 輸出經 init_state 建構的值；
   child 自己的 target/g_vec/p_vec 不變。
3. child 不在池中（無 case_id 命中）：略過，不報錯。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.training.mean_pool import MeanPool  # noqa: E402


class _FakeModel:
    """假 model：init_state 直接把 static/physics 拼起來（sd+pd 通道），無 hidden。"""

    @staticmethod
    def init_state(static: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        return torch.cat([static, physics], dim=1)


def _make_pool(n=4, sd=2, pd=3, h=4, w=4) -> MeanPool:
    pool = MeanPool(size=n, sd=sd, pd=pd, device=torch.device("cpu"),
                     seed_frac=0.0, corrupt_frac=0.0, damage_frac=0.0)
    state0 = torch.zeros(n, sd + pd, h, w)
    target = torch.zeros(n, pd, h, w)
    parent = torch.zeros(n, pd, h, w)
    # seed_ic 與 parent 是兩個不同的量：parent 供 wake 加權，seed_ic 才是
    # seed 重置時灌回去的 IC。本檔測的是 chain_writeback，兩者的值不影響結果，
    # 但仍分開傳，以免示範出「拿 parent 當 IC」的錯誤用法。
    seed_ic = torch.zeros(n, pd, h, w)
    g_vec = torch.zeros(n, 2)
    p_vec = torch.zeros(n, 1)
    case_ids = [f"case_{i}" for i in range(n)]
    pool.refill(state0, target, parent, seed_ic, g_vec, p_vec, case_ids=case_ids)
    return pool


def test_chain_writeback_noop_when_frac_zero() -> None:
    pool = _make_pool()
    before = pool._states.clone()
    pool.chain_writeback(
        parent_case_ids=["case_0"],
        final_state=torch.ones(1, pool.sd + pool.pd, 4, 4),
        model=_FakeModel(),
        child_map={"case_0": ["case_1"]},
        frac=0.0,
    )
    assert torch.equal(pool._states, before), "frac=0 應為 no-op"


def test_chain_writeback_injects_parent_output_into_child_slot() -> None:
    pool = _make_pool()
    child_target_before = pool._targets[1].clone()
    child_gvec_before = pool._gvec[1].clone()

    parent_final = torch.full((1, pool.sd + pool.pd, 4, 4), 7.0)  # case_0 的 model 輸出
    pool.chain_writeback(
        parent_case_ids=["case_0"],
        final_state=parent_final,
        model=_FakeModel(),
        child_map={"case_0": ["case_1"]},
        frac=1.0,
    )
    # child (case_1, slot 1) physics 應變成 init_state(child_static, parent_physics=7.0)
    child_static = pool._states[1:2, :pool.sd]  # child 自己的 static（未變動）
    expected = _FakeModel.init_state(child_static, parent_final[:, pool.sd:pool.sd + pool.pd])[0]
    assert torch.allclose(pool._states[1], expected), "child slot 應被 parent 輸出接力覆寫"
    # child 自己的 target/g_vec 不應被動到
    assert torch.equal(pool._targets[1], child_target_before)
    assert torch.equal(pool._gvec[1], child_gvec_before)
    assert pool._chain_count == 1


def test_chain_writeback_skips_child_not_in_pool() -> None:
    pool = _make_pool()
    before = pool._states.clone()
    pool.chain_writeback(
        parent_case_ids=["case_0"],
        final_state=torch.ones(1, pool.sd + pool.pd, 4, 4),
        model=_FakeModel(),
        child_map={"case_0": ["case_not_in_pool"]},
        frac=1.0,
    )
    assert torch.equal(pool._states, before), "child 不在池中應略過不報錯"
    assert pool._chain_count == 0


def test_sample_returns_case_ids() -> None:
    pool = _make_pool()
    ps = pool.sample(2, _FakeModel())
    assert len(ps.case_ids) == 2
    assert all(c is not None and c.startswith("case_") for c in ps.case_ids)
