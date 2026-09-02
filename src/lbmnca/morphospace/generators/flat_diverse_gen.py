"""多樣性強化版 flat 建物佈局 PNG mask 生成器。

相較 flat_mask_gen.py，本生成器增加三個多樣性軸：
  1. 建物尺寸變化：寬/高獨立取樣 [W_MIN, W_MAX] px，支援方形/寬形/高形/混合四種類型
  2. 建物數量變化：每個場景 [N_MIN, N_MAX] 棟隨機
  3. 擺放自由化：rejection sampling 在 inner domain 自由散佈

保持矩形幾何（比 OSM 容易學），輸出格式與 flat_mask_gen.py 完全相容。

Usage:
    python -m src.generators.flat_diverse_gen
    python -m src.generators.flat_diverse_gen --n_masks 200 --seed 42 --batch-name flat_diverse200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image

# ── Domain（與 flat_mask_gen.py 完全相同）──────────────────────────────────
PAD_LEFT: int = 256
PAD_TOP: int = 156
PAD_RIGHT: int = 896
PAD_BOT: int = 156
_INNER_PX: int = 768
H_TOTAL: int = PAD_TOP + _INNER_PX + PAD_BOT   # 1080
W_TOTAL: int = PAD_LEFT + _INNER_PX + PAD_RIGHT  # 1920
M_PER_PX: float = (8 * 25) / _INNER_PX           # 200m / 768px ≈ 0.260

# ── 生成參數 ────────────────────────────────────────────────────────────────
W_MIN: int = 25    # 建物寬度範圍（mask pixels）— 小街屋
W_MAX: int = 220   # — 大型建物，尺寸差距 8.8×（vs 原來 3.5×）
H_MIN: int = 25    # 建物高度範圍（mask pixels）
H_MAX: int = 220
N_MIN: int = 3     # 每個場景建物棟數範圍（大樓場景棟數自然少）
N_MAX: int = 20
MIN_GAP: int = 12  # 建物邊緣最小間距（mask pixels）
INNER_MARGIN: int = 6  # 距 inner domain 邊界的最小距離

MAX_PLACE_TRIES: int = 200  # 每棟建物的擺放嘗試次數
MAX_TOTAL_FACTOR: int = 30   # 總嘗試次數 = n_masks × factor

_WORKSPACE_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent


# ── 建物資料結構 ────────────────────────────────────────────────────────────

class _Building(NamedTuple):
    cx: float   # 中心 x（mask 座標）
    cy: float   # 中心 y（mask 座標）
    w: int      # 寬度 px
    h: int      # 高度 px


# ── 幾何工具 ────────────────────────────────────────────────────────────────

def _violates_gap(a: _Building, b: _Building) -> bool:
    """若兩棟建物的邊緣間距 < MIN_GAP，回傳 True。

    使用 AABB expanded-box 近似：
      若擴大 MIN_GAP 後的 AABB 重疊，即違反間距。
    """
    dx = abs(a.cx - b.cx)
    dy = abs(a.cy - b.cy)
    return dx < (a.w + b.w) / 2 + MIN_GAP and dy < (a.h + b.h) / 2 + MIN_GAP


def _within_inner(b: _Building) -> bool:
    """建物完全在 inner domain 內（含 INNER_MARGIN 緩衝），回傳 True。"""
    return (
        b.cx - b.w / 2 >= PAD_LEFT + INNER_MARGIN
        and b.cx + b.w / 2 <= PAD_LEFT + _INNER_PX - INNER_MARGIN
        and b.cy - b.h / 2 >= PAD_TOP + INNER_MARGIN
        and b.cy + b.h / 2 <= PAD_TOP + _INNER_PX - INNER_MARGIN
    )


def _min_edge_dist(buildings: list[_Building]) -> float:
    """所有建物對的最小邊緣間距（Euclidean）。"""
    if len(buildings) < 2:
        return 0.0
    best = math.inf
    for i, a in enumerate(buildings):
        for b in buildings[i + 1:]:
            dx = max(0.0, abs(a.cx - b.cx) - (a.w + b.w) / 2)
            dy = max(0.0, abs(a.cy - b.cy) - (a.h + b.h) / 2)
            best = min(best, math.hypot(dx, dy))
    return float(best)


# ── 建物尺寸取樣 ────────────────────────────────────────────────────────────

def _sample_size(rng: np.random.RandomState) -> tuple[int, int]:
    """取樣建物尺寸，支援四種形狀類型。

    Returns:
        (w, h) 單位：mask pixels。
    """
    shape_type = rng.randint(5)
    if shape_type == 0:   # 小街屋（25–80px，近方形）
        base = rng.randint(W_MIN, 81)
        ratio = rng.uniform(0.7, 1.4)
        w, h = base, int(base * ratio)
    elif shape_type == 1:  # 中型方形（70–150px）
        base = rng.randint(70, 151)
        ratio = rng.uniform(0.75, 1.33)
        w, h = base, int(base * ratio)
    elif shape_type == 2:  # 大型建物（140–220px，近方形）
        base = rng.randint(140, W_MAX + 1)
        ratio = rng.uniform(0.7, 1.4)
        w, h = base, int(base * ratio)
    elif shape_type == 3:  # 寬形板狀（高/寬 比 1.8–5.0）
        short = rng.randint(W_MIN, 100)
        w = int(short * rng.uniform(1.8, 5.0))
        h = short
    else:                  # 高形板狀（寬/高 比 1.8–5.0）
        short = rng.randint(H_MIN, 100)
        h = int(short * rng.uniform(1.8, 5.0))
        w = short
    return max(W_MIN, min(W_MAX, w)), max(H_MIN, min(H_MAX, h))


# ── 佈局生成 ────────────────────────────────────────────────────────────────

def _generate_layout(rng: np.random.RandomState) -> list[_Building] | None:
    """嘗試生成一個有效的建物佈局。失敗時回傳 None。

    Args:
        rng: numpy RandomState。

    Returns:
        建物列表（已通過間距與邊界驗證），或 None。
    """
    n_target = rng.randint(N_MIN, N_MAX + 1)
    placed: list[_Building] = []

    for _ in range(n_target):
        w, h = _sample_size(rng)
        half_w, half_h = w / 2.0, h / 2.0

        x_lo = PAD_LEFT + INNER_MARGIN + half_w
        x_hi = PAD_LEFT + _INNER_PX - INNER_MARGIN - half_w
        y_lo = PAD_TOP + INNER_MARGIN + half_h
        y_hi = PAD_TOP + _INNER_PX - INNER_MARGIN - half_h

        if x_lo >= x_hi or y_lo >= y_hi:
            continue  # 建物尺寸過大，跳過

        for _ in range(MAX_PLACE_TRIES):
            cx = rng.uniform(x_lo, x_hi)
            cy = rng.uniform(y_lo, y_hi)
            b = _Building(cx, cy, w, h)
            if all(not _violates_gap(b, other) for other in placed):
                placed.append(b)
                break

    return placed if len(placed) >= N_MIN else None


# ── 渲染 ────────────────────────────────────────────────────────────────────

def render_mask(buildings: list[_Building]) -> np.ndarray:
    """建物列表 → mask uint8。0=建物，255=流體。

    Args:
        buildings: 建物列表。

    Returns:
        shape (H_TOTAL, W_TOTAL) uint8 array。
    """
    mask = np.full((H_TOTAL, W_TOTAL), 255, dtype=np.uint8)
    for b in buildings:
        x0 = max(0, int(round(b.cx - b.w / 2)))
        x1 = min(W_TOTAL, int(round(b.cx + b.w / 2)))
        y0 = max(0, int(round(b.cy - b.h / 2)))
        y1 = min(H_TOTAL, int(round(b.cy + b.h / 2)))
        mask[y0:y1, x0:x1] = 0
    return mask


# ── 去重 hash ────────────────────────────────────────────────────────────────

def _layout_hash(buildings: list[_Building]) -> str:
    """基於 10px 格點離散化的 SHA1[:8] hash，用於近似去重。"""
    key = b"".join(
        int(round(b.cx / 10)).to_bytes(2, "big")
        + int(round(b.cy / 10)).to_bytes(2, "big")
        + b.w.to_bytes(1, "big")
        + b.h.to_bytes(1, "big")
        for b in sorted(buildings, key=lambda b: (round(b.cx / 10), round(b.cy / 10)))
    )
    return hashlib.sha1(key).hexdigest()[:8]


# ── Metadata ─────────────────────────────────────────────────────────────────

def _build_metadata(
    mask_id: str,
    buildings: list[_Building],
    occ_hash: str,
) -> dict[str, Any]:
    """建立 mask_meta.json 條目，與 flat_mask_gen.py 格式相容。

    Args:
        mask_id: mask 唯一 ID（如 diverse_0000）。
        buildings: 已擺放的建物列表。
        occ_hash: 佈局 hash（去重用）。

    Returns:
        metadata dict。
    """
    total_building_px = sum(b.w * b.h for b in buildings)
    inner_area = _INNER_PX * _INNER_PX
    bcr = total_building_px / inner_area
    return {
        "file_name": f"{mask_id}.png",
        "mask_id": mask_id,
        "n_buildings": len(buildings),
        "min_distance_px": round(_min_edge_dist(buildings), 2),
        "m_per_px": M_PER_PX,
        "px_per_m": round(1.0 / M_PER_PX, 4),
        "bcr": round(bcr, 5),
        "angle_deg": 0.0,
        "wind_dir_equiv": 0.0,
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
        "diverse_meta": {
            "occupancy_hash": occ_hash,
            "buildings": [
                {"cx": round(b.cx, 1), "cy": round(b.cy, 1), "w": b.w, "h": b.h}
                for b in buildings
            ],
        },
    }


# ── 主 pipeline ──────────────────────────────────────────────────────────────

def run(
    n_masks: int = 200,
    seed: int = 42,
    output_dir: Path = _WORKSPACE_ROOT / "dataset" / "CEDS_dataset" / "flat_diverse200",
) -> int:
    """生成 n_masks 個多樣性 flat mask 並儲存。

    Args:
        n_masks: 目標 mask 數量。
        seed: RNG seed。
        output_dir: 輸出目錄（自動建立 masks/ 子目錄）。

    Returns:
        0 成功，1 失敗。
    """
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    max_total = n_masks * MAX_TOTAL_FACTOR
    print(f"[flat_diverse_gen] 目標 {n_masks} masks, seed={seed}")
    print(f"  建物尺寸 W:[{W_MIN},{W_MAX}] H:[{H_MIN},{H_MAX}] px")
    print(f"  棟數範圍 [{N_MIN}, {N_MAX}]，最小間距 {MIN_GAP} px")
    print(f"  輸出 -> {output_dir}")

    collected: list[list[_Building]] = []
    hashes: list[str] = []
    seen: set[str] = set()
    attempts = 0

    while len(collected) < n_masks and attempts < max_total:
        layout = _generate_layout(rng)
        attempts += 1
        if layout is None:
            continue
        h = _layout_hash(layout)
        if h in seen:
            continue
        seen.add(h)
        collected.append(layout)
        hashes.append(h)

        if len(collected) % 50 == 0:
            print(f"  {len(collected)}/{n_masks} 生成完畢（嘗試 {attempts} 次）...")

    actual = len(collected)
    if actual < n_masks:
        print(f"[flat_diverse_gen] 警告：只生成了 {actual} 個（目標 {n_masks}）")

    metadata_entries: list[dict[str, Any]] = []
    for idx, (layout, occ_hash) in enumerate(zip(collected, hashes)):
        mask_id = f"diverse_{idx:04d}"
        mask = render_mask(layout)
        Image.fromarray(mask).save(masks_dir / f"{mask_id}.png")
        metadata_entries.append(_build_metadata(mask_id, layout, occ_hash))

    meta_path = output_dir / "mask_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_entries, f, ensure_ascii=False, indent=2)

    n_bldgs = [len(l) for l in collected]
    bcrs = [e["bcr"] for e in metadata_entries]
    print(f"\n[flat_diverse_gen] 完成 {actual} masks（嘗試 {attempts} 次）")
    print(f"  棟數: min={min(n_bldgs)} max={max(n_bldgs)} mean={sum(n_bldgs)/len(n_bldgs):.1f}")
    print(f"  BCR:  min={min(bcrs):.3f} max={max(bcrs):.3f} mean={sum(bcrs)/len(bcrs):.3f}")
    print(f"  metadata -> {meta_path}")
    return 0


def main() -> int:
    """CLI 入口。"""
    p = argparse.ArgumentParser(description="多樣性強化版 flat mask 生成器")
    p.add_argument("--n_masks", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-name", type=str, default="flat_diverse200",
                   help="CEDS_dataset 批次名稱")
    p.add_argument("--output_dir", type=str, default=None,
                   help="覆蓋輸出目錄（預設由 --batch-name 推導）")
    args = p.parse_args()

    out = (
        Path(args.output_dir)
        if args.output_dir
        else _WORKSPACE_ROOT / "dataset" / "CEDS_dataset" / args.batch_name
    )
    return run(n_masks=args.n_masks, seed=args.seed, output_dir=out)


if __name__ == "__main__":
    raise SystemExit(main())
