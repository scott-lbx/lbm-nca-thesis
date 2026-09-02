"""order-stats featurize：變動棟數 (n,4) → 定長 21-d（permutation-invariant、平滑）。

21 = mean/std/min/max/median ×(cx,cy,w,h)=20 + n_buildings=1。
order-stats 對建物排序不變、移 1 棟只小幅動 mean/std → delta 平滑單調（已探針驗證）。
n_buildings 補上（order-stats 抓不到棟數）。
"""

from __future__ import annotations

import numpy as np

_STAT_NAMES = ("mean", "std", "min", "max", "median")
_COLS = ("cx", "cy", "w", "h")
FEATURE_NAMES: list[str] = [f"{s}_{c}" for s in _STAT_NAMES for c in _COLS] + ["n_bldg"]


def featurize(bld: np.ndarray) -> np.ndarray:
    """bld: (n,4) [cx,cy,w,h] → (21,)。"""
    bld = np.asarray(bld, dtype=np.float64)
    stats = [
        bld.mean(0), bld.std(0), bld.min(0), bld.max(0), np.percentile(bld, 50, 0),
    ]
    return np.concatenate(stats + [[len(bld)]]).astype(np.float64)


def iter_layouts(params: np.ndarray, counts: np.ndarray):
    """concatenated params (M,4) + counts (N,) → 逐佈局 (n_i,4)。"""
    off = 0
    for c in counts:
        c = int(c)
        yield params[off:off + c]
        off += c


def featurize_population(params: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """母群體 → X (N, 21)。"""
    return np.stack([featurize(b) for b in iter_layouts(params, counts)])


# ── 幾何描述子（解釋 PC 用，與 featurize 無關）─────────────────────────
DESCRIPTOR_NAMES: list[str] = ["centroid_x", "centroid_y", "area_frac_x1e3",
                               "n_bldg", "spread_x", "mean_w"]


def descriptors(bld: np.ndarray, w_total: int, h_total: int) -> np.ndarray:
    """簡單可解釋幾何量，供 PC 相關性解讀。"""
    bld = np.asarray(bld, dtype=np.float64)
    area = (bld[:, 2] * bld[:, 3]).sum() / (w_total * h_total)
    return np.array([bld[:, 0].mean() / w_total, bld[:, 1].mean() / h_total,
                     area * 1e3, len(bld), bld[:, 0].std() / w_total,
                     bld[:, 2].mean() / w_total])


def descriptors_population(params: np.ndarray, counts: np.ndarray,
                           w_total: int, h_total: int) -> np.ndarray:
    return np.stack([descriptors(b, w_total, h_total)
                     for b in iter_layouts(params, counts)])
