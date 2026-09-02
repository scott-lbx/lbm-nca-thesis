"""幾何合法性回歸測試（GPU-free，純 numpy）。

驗證 DEEP_AUDIT_2026-07 Stage 1 / 1.4：warm 分支 `perturb_buildings` 補回
重疊 / 間距 / 越界的合法性約束，與 cold 分支 `generate.py` 共用同一組判定
（`src.param_pca.legality.check_layout_legal`，內部委派
`src.param_pca.generate.violates_gap`）。

涵蓋兩份 `perturb_buildings` 實作：
  - `scripts/qa/sample_warm_perturb.py`（C0-3/C1-3 perturbation-tree）
  - `src/lbmnca/lbm/config_gen/c2_2_graph_sampler.py`（C2-2 warm graph，生產路徑）

執行：
    python -m pytest tests/morphospace/test_legality.py -v   （cwd = workspace root）
    python tests/morphospace/test_legality.py                （直接執行）
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Windows 預設 cp950 console 對部分字元會噴 UnicodeEncodeError，強制 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from lbmnca.lbm.config_gen import c2_2_graph_sampler as _c22
from lbmnca.morphospace.dataset import sample_warm_perturb as _swp
from lbmnca.morphospace.param_pca import config as C
from lbmnca.morphospace.param_pca.generate import generate_layout, violates_gap
from lbmnca.morphospace.param_pca.legality import check_layout_legal, is_out_of_bounds


# ── Fixtures（純資料，非 pytest fixture，保持簡單）───────────────────────────

def _tight_cluster_parent(n: int = 10) -> list[list[float]]:
    """10 棟緊鄰排列的建物（30x30，間距僅 20px < MIN_GAP=12? 實際 20>12，
    但大幅擾動後極易互相重疊/間距不足），用於逼出 rejection sampling 邊界。
    """
    cx0, cy0 = 640, 540
    return [[cx0 + (i % 5) * 20, cy0 + (i // 5) * 20, 30, 30] for i in range(n)]


def _sparse_parent() -> list[list[float]]:
    """3 棟彼此相距甚遠的建物，代表典型正常擾動情境。"""
    return [[400.0, 400.0, 30.0, 30.0],
            [700.0, 700.0, 30.0, 30.0],
            [900.0, 300.0, 30.0, 30.0]]


# ── legality.py 原子判定 ──────────────────────────────────────────────────────

def test_check_layout_legal_detects_overlap() -> None:
    """兩棟嚴重重疊的建物 → 判定不合法，原因為 gap_violation。"""
    b1 = np.array([500.0, 500.0, 50.0, 50.0])
    b2 = np.array([510.0, 500.0, 50.0, 50.0])   # 中心距 10px << 邊緣需求
    assert violates_gap(b1, b2)
    is_legal, reason = check_layout_legal(np.stack([b1, b2]))
    assert not is_legal
    assert reason == "gap_violation"


def test_check_layout_legal_detects_out_of_bounds() -> None:
    """建物邊緣超出 inner box → 判定不合法，原因為 out_of_bounds。"""
    b = np.array([float(C.CX_HI) - 1.0, 500.0, 100.0, 50.0])  # 右邊界外溢
    assert is_out_of_bounds(b)
    is_legal, reason = check_layout_legal(np.stack([b]))
    assert not is_legal
    assert reason == "out_of_bounds"


def test_check_layout_legal_accepts_sparse_layout() -> None:
    """彼此遠離且在界內的建物 → 判定合法。"""
    buildings = np.array(_sparse_parent())
    is_legal, reason = check_layout_legal(buildings)
    assert is_legal
    assert reason == ""


def test_cold_generate_layout_always_legal() -> None:
    """cold 分支 `generate_layout` 產出的佈局必然通過 `check_layout_legal`
    （兩者理應同分布——這是本次修正要對齊 warm 分支的基準）。
    """
    rng = np.random.RandomState(7)
    n_checked = 0
    for _ in range(50):
        layout = generate_layout(rng)
        if layout is None:
            continue
        is_legal, reason = check_layout_legal(layout)
        assert is_legal, f"cold generate_layout 產出不合法佈局：{reason}"
        n_checked += 1
    assert n_checked > 0, "50 次嘗試都沒有產生合法 cold 佈局，測試母體有誤"


# ── sample_warm_perturb.perturb_buildings ────────────────────────────────────

def test_swp_perturb_never_returns_illegal_geometry() -> None:
    """緊密 parent + 最大擾動幅度：凡有回傳必合法；找不到合法解則顯式 raise，
    絕不靜默回退成重疊/越界幾何。
    """
    parent = _tight_cluster_parent()
    n_success = n_raised = n_illegal = 0
    for seed in range(30):
        rng = np.random.RandomState(seed)
        try:
            child = _swp.perturb_buildings(parent, 1.0, 1.0, rng)
        except RuntimeError:
            n_raised += 1
            continue
        n_success += 1
        is_legal, reason = check_layout_legal(np.array(child))
        if not is_legal:
            n_illegal += 1
    assert n_illegal == 0, f"{n_illegal}/30 次回傳不合法幾何"
    assert n_success + n_raised == 30
    print(f"[PASS] sample_warm_perturb tight-cluster: success={n_success} "
          f"raised={n_raised} illegal={n_illegal}")


def test_swp_perturb_succeeds_on_normal_case() -> None:
    """稀疏 parent + 適中擾動幅度：功能未被鎖死，仍能成功產生合法擾動。"""
    parent = _sparse_parent()
    n_success = 0
    for seed in range(30):
        rng = np.random.RandomState(seed)
        child = _swp.perturb_buildings(parent, 0.3, 0.5, rng)
        n_success += 1
        is_legal, reason = check_layout_legal(np.array(child))
        assert is_legal, f"seed={seed} 回傳不合法幾何：{reason}"
    assert n_success == 30
    print(f"[PASS] sample_warm_perturb normal case: {n_success}/30 全部合法成功")


# ── c2_2_graph_sampler.perturb_buildings（C2-2 生產路徑）───────────────────────

def test_c22_perturb_never_returns_illegal_geometry() -> None:
    """C2-2 生產路徑同一約束：緊密 parent + 大擾動 → 不回傳不合法幾何。"""
    parent = _tight_cluster_parent()
    n_success = n_raised = n_illegal = 0
    for seed in range(30):
        rng = np.random.RandomState(seed)
        try:
            child, _single = _c22.perturb_buildings(parent, 1.0, rng)
        except RuntimeError:
            n_raised += 1
            continue
        n_success += 1
        is_legal, reason = _c22.check_layout_legal(np.array(child))
        if not is_legal:
            n_illegal += 1
    assert n_illegal == 0, f"{n_illegal}/30 次回傳不合法幾何"
    assert n_success + n_raised == 30
    print(f"[PASS] c2_2_graph_sampler tight-cluster: success={n_success} "
          f"raised={n_raised} illegal={n_illegal}")


def test_c22_perturb_succeeds_on_normal_case() -> None:
    """C2-2 生產路徑：稀疏 parent 仍能成功產生合法擾動（功能未鎖死）。"""
    parent = _sparse_parent()
    n_success = 0
    for seed in range(30):
        rng = np.random.RandomState(seed)
        child, _single = _c22.perturb_buildings(parent, 0.3, rng)
        n_success += 1
        is_legal, reason = _c22.check_layout_legal(np.array(child))
        assert is_legal, f"seed={seed} 回傳不合法幾何：{reason}"
    assert n_success == 30
    print(f"[PASS] c2_2_graph_sampler normal case: {n_success}/30 全部合法成功")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
