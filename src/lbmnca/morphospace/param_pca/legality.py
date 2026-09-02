"""幾何合法性判定：重疊 / 間距 / 越界的單一真相（cold 與 warm 生成路徑共用）。

`generate.py`（cold rectangle-layout 母群體取樣器）以逐建物 placement-time
rejection 達成合法佈局，其原子幾何判定 `violates_gap()`（重疊 + 間距不足合併
判定）即為本模組沿用的單一真相，不重複實作。

warm 分支（`perturb_buildings`，見
`scripts/qa/sample_warm_perturb.py` 與
`src/lbmnca/lbm/config_gen/c2_2_graph_sampler.py`）擾動既有 parent 佈局後得到
的是「一整組已存在的建物」，而非逐棟增量放置，因此需要「驗證整組建物是否
合法」的判定，`generate.py` 原本沒有這個介面（它只在放置當下做逐棟拒絕）。
本模組補上這個介面（`check_layout_legal`），內部呼叫 `violates_gap` 與
`is_out_of_bounds`，供 warm 分支做 rejection sampling。
"""

from __future__ import annotations

import numpy as np

from lbmnca.morphospace.param_pca import config as C
from lbmnca.morphospace.param_pca.generate import violates_gap

__all__ = ["is_out_of_bounds", "check_layout_legal"]


def is_out_of_bounds(building: np.ndarray) -> bool:
    """building=[cx,cy,w,h] 是否超出可放置的 inner box（CX_LO..CX_HI × CY_LO..CY_HI）。

    Args:
        building: 長度 4 的陣列 [cx, cy, w, h]。

    Returns:
        True 表示建物邊緣超出 inner box（含 INNER_MARGIN 緩衝，與 cold 分支
        `generate_layout` 逐棟放置時使用的邊界完全一致）。
    """
    cx, cy, w, h = building
    hw, hh = w / 2.0, h / 2.0
    return bool(cx - hw < C.CX_LO or cx + hw > C.CX_HI or
                cy - hh < C.CY_LO or cy + hh > C.CY_HI)


def check_layout_legal(buildings: np.ndarray) -> tuple[bool, str]:
    """驗證一整組建物是否合法（無越界、無重疊/間距違規）。

    Args:
        buildings: (n,4) [cx,cy,w,h] 陣列（或可轉換為此形狀的巢狀 list）。

    Returns:
        (is_legal, reason)。is_legal=True 時 reason 為空字串；否則 reason 為
        第一個觸發的違規類別（"out_of_bounds" 或 "gap_violation"，後者涵蓋
        真重疊，因 `violates_gap` 的 MIN_GAP 緩衝恆為非負）。
    """
    buildings = np.asarray(buildings, dtype=float)
    for b in buildings:
        if is_out_of_bounds(b):
            return False, "out_of_bounds"
    n = len(buildings)
    for i in range(n):
        for j in range(i + 1, n):
            if violates_gap(buildings[i], buildings[j]):
                return False, "gap_violation"
    return True, ""
