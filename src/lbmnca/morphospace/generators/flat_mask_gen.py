"""隨機走訪生成 flat 建物佈局 PNG masks（無 graph 關係）。

從 5 個起始佈局出發做隨機走訪，以 occupancy hash 去重，輸出 N 個多樣的
獨立建物佈局 masks，供 Geometry VAE 訓練或 LBM 模擬使用。

建物群置於 mask 左側 1/3 位置（短上游 + 長下游喚醒區），以利風場資料集
的下游 wake 結構捕捉。

Usage:
    python -m src.generators.flat_mask_gen
    python -m src.generators.flat_mask_gen --n_masks 300 --seed 99
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Domain constants（對齊 lbmnca.lbm v2.0 POI 規格，建物置於左側 1/3）
# ---------------------------------------------------------------------------

GRID_CELL_TO_LBM_PX: int = 96  # 每 grid cell = 96 LBM pixels
BUILDING_FOOTPRINT_PX: int = 77  # = round(20m / 25m × 96)
GRID_SIZE: int = 8  # 8×8 design grid
CELL_SIZE_M: int = 25  # 每 cell = 25m

_INNER_PX: int = GRID_SIZE * GRID_CELL_TO_LBM_PX  # 768

# POI domain = 1920×1080（1080p）；障礙物 min-bound 中心置於 (640, 540)
# 即左側 1/3（x=1920/3）、上下正中（y=1080/2）
# grid center = (PAD_LEFT + _INNER_PX/2, PAD_TOP + _INNER_PX/2) → (640, 540)
PAD_LEFT: int = 256  # 640 - 384 = 256（上游邊距）
PAD_RIGHT: int = 896  # 1920 - 256 - 768 = 896（下游 wake 區）
PAD_TOP: int = 156  # 540 - 384 = 156
PAD_BOT: int = 156  # 1080 - 156 - 768 = 156

# ext_pad（solver sponge+buffer）由 lbmnca.lbm master_config.domain_ext_pad 管理，
# 不存入 mask metadata。solver 域 = 2560×1440（2K）。
H_TOTAL: int = PAD_TOP + _INNER_PX + PAD_BOT  # 156 + 768 + 156 = 1080
W_TOTAL: int = PAD_LEFT + _INNER_PX + PAD_RIGHT  # 256 + 768 + 896 = 1920

# Grid validity：有效 cell 在 [BUFFER, GRID_SIZE - BUFFER)
BUFFER: int = 1
VALID_MIN: int = BUFFER
VALID_MAX: int = GRID_SIZE - BUFFER - 1  # 6（inclusive）
MIN_BUILDINGS: int = 4

_WORKSPACE_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
_OUTPUT_DIR: Path = _WORKSPACE_ROOT / "dataset" / "CEDS_dataset" / "flat200"


# ---------------------------------------------------------------------------
# 5 個起始佈局（只記錄 occupancy，height 對 2D mask 無影響）
# ---------------------------------------------------------------------------
# 類型：frozenset[tuple[int, int]]，每個 tuple 為 (grid_i, grid_j)

_SEED_LAYOUTS: list[frozenset[tuple[int, int]]] = [
    # 0：3×3 均勻格網（9 棟）
    frozenset({(2, 2), (2, 4), (2, 6), (4, 2), (4, 4), (4, 6), (6, 2), (6, 4), (6, 6)}),
    # 1：風通道——左右各一排（6 棟）
    frozenset({(2, 1), (4, 1), (6, 1), (2, 6), (4, 6), (6, 6)}),
    # 2：開放廣場（中央空缺，周邊 8 棟）
    frozenset({(2, 2), (2, 4), (2, 6), (4, 2), (4, 6), (6, 2), (6, 4), (6, 6)}),
    # 3：稀疏錯落（5 棟）
    frozenset({(2, 3), (4, 5), (5, 2), (6, 6), (3, 4)}),
    # 4：L 形排列（7 棟）
    frozenset({(2, 1), (2, 3), (2, 5), (4, 1), (4, 3), (6, 1), (6, 5)}),
]

_SEED_NAMES: list[str] = [
    "grid_3x3",
    "wind_channel",
    "open_plaza",
    "sparse",
    "l_shape",
]

MOVE_DIRS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


# ---------------------------------------------------------------------------
# 佈局驗證與編輯
# ---------------------------------------------------------------------------


def _is_valid(cells: frozenset[tuple[int, int]]) -> bool:
    """G1–G5 全規則檢查。

    G1: n_buildings >= MIN_BUILDINGS
    G2: 所有 cell 在 [VALID_MIN, VALID_MAX]
    G3: 無重疊（frozenset 自動保證）
    G5: 無 4-鄰域相鄰
    """
    if len(cells) < MIN_BUILDINGS:
        return False
    for i, j in cells:
        if not (VALID_MIN <= i <= VALID_MAX and VALID_MIN <= j <= VALID_MAX):
            return False
    # G5: 4-鄰域無相鄰
    for i, j in cells:
        if (i + 1, j) in cells or (i, j + 1) in cells:
            return False
    return True


def _enumerate_edits(
    cells: frozenset[tuple[int, int]],
) -> list[frozenset[tuple[int, int]]]:
    """列舉所有 valid 下一步狀態（add / remove / move）。"""
    results: list[frozenset[tuple[int, int]]] = []

    # add：所有 empty valid cells
    for i in range(VALID_MIN, VALID_MAX + 1):
        for j in range(VALID_MIN, VALID_MAX + 1):
            if (i, j) not in cells:
                candidate = cells | {(i, j)}
                if _is_valid(candidate):
                    results.append(candidate)

    # remove：移除每一棟
    for cell in cells:
        candidate = cells - {cell}
        if _is_valid(candidate):
            results.append(candidate)

    # move：每棟建物往 8 鄰域移動
    for cell in cells:
        i, j = cell
        rest = cells - {cell}
        for di, dj in MOVE_DIRS:
            ni, nj = i + di, j + dj
            if (ni, nj) not in rest:
                candidate = rest | {(ni, nj)}
                if _is_valid(candidate):
                    results.append(candidate)

    return results


# ---------------------------------------------------------------------------
# 去重 hash（occupancy only）
# ---------------------------------------------------------------------------


def _occ_hash(cells: frozenset[tuple[int, int]]) -> str:
    """SHA1[:8] based on sorted cell list（與高度無關）。"""
    key = b"".join(
        i.to_bytes(1, "big") + j.to_bytes(1, "big") for i, j in sorted(cells)
    )
    return hashlib.sha1(key).hexdigest()[:8]


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def render_mask(cells: frozenset[tuple[int, int]]) -> np.ndarray:
    """建物佈局 → LBM-ready PNG mask（uint8）。

    Convention：0 = 建物（固體），255 = 自由流。
    建物群置於左側 1/3（PAD_LEFT=500）。
    """
    mask = np.full((H_TOTAL, W_TOTAL), 255, dtype=np.uint8)
    half = BUILDING_FOOTPRINT_PX // 2
    for i, j in cells:
        cx = PAD_LEFT + (i + 0.5) * GRID_CELL_TO_LBM_PX
        cy = PAD_TOP + (j + 0.5) * GRID_CELL_TO_LBM_PX
        x0 = max(0, int(round(cx - half)))
        x1 = min(W_TOTAL, int(round(cx + half)))
        y0 = max(0, int(round(cy - half)))
        y1 = min(H_TOTAL, int(round(cy + half)))
        mask[y0:y1, x0:x1] = 0
    return mask


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _min_bldg_dist_px(cells: frozenset[tuple[int, int]]) -> float:
    """所有建物中心點兩兩最小 LBM pixel 距離；單棟回 0.0。"""
    centers = [
        (
            PAD_LEFT + (i + 0.5) * GRID_CELL_TO_LBM_PX,
            PAD_TOP + (j + 0.5) * GRID_CELL_TO_LBM_PX,
        )
        for (i, j) in cells
    ]
    if len(centers) < 2:
        return 0.0
    best = math.inf
    for a in range(len(centers)):
        for b in range(a + 1, len(centers)):
            d = math.hypot(centers[a][0] - centers[b][0], centers[a][1] - centers[b][1])
            best = min(best, d)
    return float(best)


def _build_metadata(
    mask_id: str,
    cells: frozenset[tuple[int, int]],
    occ_hash: str,
    source_seed: str,
) -> dict[str, Any]:
    """metadata.json 條目，對齊 lbmnca.lbm config_batch_gen.py 所需欄位。"""
    base_inner_m = GRID_SIZE * CELL_SIZE_M  # 200 m
    m_per_px = base_inner_m / _INNER_PX
    px_per_m = _INNER_PX / base_inner_m
    return {
        "file_name": f"{mask_id}.png",
        "mask_id": mask_id,
        "angle_deg": 0.0,
        "wind_dir_equiv": 0.0,
        "n_buildings": len(cells),
        "min_distance_px": round(_min_bldg_dist_px(cells), 4),
        "m_per_px": m_per_px,
        "px_per_m": px_per_m,
        "center_lon": 0.0,
        "center_lat": 0.0,
        "domain_W_total": W_TOTAL,
        "domain_H_total": H_TOTAL,
        "domain_W_inner": _INNER_PX,
        "domain_H_inner": _INNER_PX,
        "pad_top": PAD_TOP,
        "pad_bot": PAD_BOT,
        "pad_left": PAD_LEFT,
        "pad_right": PAD_RIGHT,
        "flat_meta": {
            "source_seed": source_seed,
            "occupancy_hash": occ_hash,
            "cells": sorted([list(c) for c in cells]),
        },
    }


# ---------------------------------------------------------------------------
# 隨機走訪
# ---------------------------------------------------------------------------


def generate_flat_masks(
    n_masks: int = 200,
    seed: int = 42,
    restart_prob: float = 0.15,
    steps_budget: int | None = None,
) -> tuple[list[frozenset[tuple[int, int]]], list[str], list[str]]:
    """隨機走訪從 5 個 seed layouts 出發，收集 n_masks 個 unique occupancy layouts。

    Args:
        n_masks: 目標 unique mask 數量。
        seed: numpy RNG seed。
        restart_prob: 每步以此機率重置回 seed layout（維持多樣性）。
        steps_budget: 最大走訪步數（預設 n_masks × 60）。

    Returns:
        (cells_list, occ_hashes, source_names) — 三個平行 list。
    """
    rng = np.random.RandomState(seed)
    if steps_budget is None:
        steps_budget = n_masks * 60

    collected: list[frozenset[tuple[int, int]]] = []
    hashes: list[str] = []
    sources: list[str] = []
    seen: set[str] = set()

    # 先加入所有 seed layouts 本身
    for layout, name in zip(_SEED_LAYOUTS, _SEED_NAMES):
        h = _occ_hash(layout)
        if h not in seen:
            seen.add(h)
            collected.append(layout)
            hashes.append(h)
            sources.append(name)

    walk_pos = list(_SEED_LAYOUTS)  # 每個 seed 的當前走訪位置
    n_seeds = len(_SEED_LAYOUTS)
    step = 0

    while len(collected) < n_masks and step < steps_budget:
        s = step % n_seeds
        cur = walk_pos[s]

        if rng.random_sample() < restart_prob:
            walk_pos[s] = _SEED_LAYOUTS[s]
            cur = _SEED_LAYOUTS[s]

        nexts = _enumerate_edits(cur)
        if not nexts:
            walk_pos[s] = _SEED_LAYOUTS[s]
            step += 1
            continue

        nxt = nexts[rng.randint(len(nexts))]
        walk_pos[s] = nxt

        h = _occ_hash(nxt)
        if h not in seen:
            seen.add(h)
            collected.append(nxt)
            hashes.append(h)
            sources.append(_SEED_NAMES[s])

        step += 1

    return collected, hashes, sources


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    n_masks: int = 200,
    seed: int = 42,
    output_dir: Path = _OUTPUT_DIR,
) -> int:
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"[flat_mask_gen] 目標 {n_masks} masks，seed={seed}")
    print(f"  PAD_LEFT={PAD_LEFT}  PAD_RIGHT={PAD_RIGHT}  (建物群於左側 1/3)")
    print(f"  mask size={W_TOTAL}×{H_TOTAL}  →  {output_dir}")

    cells_list, occ_hashes, seed_sources = generate_flat_masks(
        n_masks=n_masks, seed=seed
    )
    actual = len(cells_list)
    if actual < n_masks:
        print(
            f"[flat_mask_gen] 警告：steps_budget 耗盡，實際生成 {actual} 個（目標 {n_masks}）"
        )

    metadata_entries: list[dict[str, Any]] = []
    for idx, (cells, occ_hash, src) in enumerate(
        zip(cells_list, occ_hashes, seed_sources)
    ):
        mask_id = f"flat_{idx:04d}"
        mask = render_mask(cells)
        save_mask_png(mask, masks_dir / f"{mask_id}.png")
        metadata_entries.append(_build_metadata(mask_id, cells, occ_hash, src))

    meta_path = output_dir / "mask_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_entries, f, ensure_ascii=False, indent=2)

    from collections import Counter

    src_counts = Counter(e["flat_meta"]["source_seed"] for e in metadata_entries)
    print(f"[flat_mask_gen] 完成 {actual} masks")
    for name, cnt in sorted(src_counts.items()):
        print(f"  {name}: {cnt}")
    print(f"  metadata → {meta_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 flat 建物佈局 PNG masks。")
    parser.add_argument("--n_masks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-name",
        type=str,
        default="flat200",
        help="CEDS_dataset 批次名稱，輸出至 dataset/CEDS_dataset/{batch-name}/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="覆蓋輸出目錄（預設由 --batch-name 自動推導）",
    )
    args = parser.parse_args()

    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = _WORKSPACE_ROOT / "dataset" / "CEDS_dataset" / args.batch_name

    return run(n_masks=args.n_masks, seed=args.seed, output_dir=out)


if __name__ == "__main__":
    raise SystemExit(main())
