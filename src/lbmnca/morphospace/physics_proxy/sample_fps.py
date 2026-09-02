"""Farthest Point Sampling (FPS) 採樣 — Phase G3 prep。

從 flat_diverse2000 pool（2000 mask）用 14 維 G_vec 的 robust z-score
距離跑 FPS，挑出 30 個最多樣的 mask 作為 LBM pilot 的輸入。

距離度量：
    1. 對 raw G_vec 每維 clip 至 [p1, p99]（去除 outlier 主導）
    2. 在 clipped 矩陣上計算 mean/std → z-score normalize
    3. L2 距離

理由：純 z-score 會被尾部 outlier 主導（如 C4/E3 的 10× p99 之外的尖峰）。
FPS 在這種情況下會變成「outlier picker」而非「diversity sampler」。
Clip [p1, p99] 後極端 outlier 仍有最大 z-score，但不會 unboundedly 拉大距離。

種子策略：以距 mean 最近的 mask 為初始（deterministic 可復現），
       後續每次挑距「已選集合」min distance 最大者。

產出（在 dataset/CEDS_dataset/g_vec_fps30/）：
    masks/                # 30 張 PNG（從源 pool 複製）
    mask_meta.json        # 30 個 entry（沿用源 pool schema）
    selection_log.json    # FPS 順序、距離、normalize 統計
    g_vec_table.csv       # 30 mask 的 raw + normalized G_vec 對照表

使用方式：
    python -m lbmnca.morphospace.physics_proxy.sample_fps \\
        --dataset-dir dataset/CEDS_dataset/flat_diverse2000 \\
        --out-dir dataset/CEDS_dataset/g_vec_fps30
    python -m lbmnca.morphospace.physics_proxy.sample_fps \\
        --dataset-dir dataset/CEDS_dataset/flat_diverse2000 \\
        --out-dir custom_dir --n-pick 50
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM, FEATURE_NAMES
from .validate_tier1 import (
    CACHE_DIR,
    DEFAULT_SIM_CFG,
    dataset_cache_key,
    load_meta,
    load_or_compute,
)


# ---------------------------------------------------------------------------
# 預設值（輸出目錄／來源 mask pool 皆無預設：由呼叫端指定）
# ---------------------------------------------------------------------------

DEFAULT_N_PICK: int = 30

# Robust z-score 的 clip 百分位數
CLIP_LOW_PCT: float = 1.0
CLIP_HIGH_PCT: float = 99.0


def robust_normalize(G_raw: np.ndarray) -> tuple[np.ndarray, dict]:
    """每維 clip 到 [p1, p99] 後做 z-score normalize。

    Returns:
        (G_norm [N, D], stats_dict)
        stats_dict 含每維的 p1/p99/mean(after clip)/std(after clip)
    """
    G_clipped = G_raw.copy().astype(np.float64)
    p_low = np.zeros(G_raw.shape[1])
    p_high = np.zeros(G_raw.shape[1])
    for d in range(G_raw.shape[1]):
        col = G_raw[:, d]
        finite = col[np.isfinite(col)]
        if finite.size < 10:
            continue
        lo = float(np.percentile(finite, CLIP_LOW_PCT))
        hi = float(np.percentile(finite, CLIP_HIGH_PCT))
        p_low[d] = lo
        p_high[d] = hi
        G_clipped[:, d] = np.clip(G_clipped[:, d], lo, hi)

    mean = G_clipped.mean(axis=0)
    std = G_clipped.std(axis=0)
    std = np.maximum(std, 1e-6)
    G_norm = (G_clipped - mean) / std

    stats = {
        "method": "clip_p1p99_then_zscore",
        "clip_low_pct": CLIP_LOW_PCT,
        "clip_high_pct": CLIP_HIGH_PCT,
        "p_low_per_dim": p_low.tolist(),
        "p_high_per_dim": p_high.tolist(),
        "mean_after_clip": mean.tolist(),
        "std_after_clip": std.tolist(),
    }
    return G_norm.astype(np.float32), stats


# ---------------------------------------------------------------------------
# FPS 演算法
# ---------------------------------------------------------------------------


@dataclass
class FPSResult:
    """FPS 採樣結果。"""

    indices: list[int]            # 在 source pool 中的 index
    min_distances: list[float]    # 每次選擇時對「已選集合」的 min distance
    seed_index: int               # 初始種子的 index


def fps_select(
    G_norm: np.ndarray,
    n_pick: int,
    seed_method: str = "centroid",
    rng_seed: int = 0,
) -> FPSResult:
    """Farthest Point Sampling。

    Args:
        G_norm: [N, D] z-score 正規化後的 G_vec。
        n_pick: 要選的數量。
        seed_method: "centroid"（距 mean 最近）或 "random"。
        rng_seed: random 模式的 seed。

    Returns:
        FPSResult。
    """
    N = G_norm.shape[0]
    if n_pick > N:
        raise ValueError(f"n_pick={n_pick} 超過 pool size N={N}")

    # 初始種子
    if seed_method == "centroid":
        centroid = G_norm.mean(axis=0)
        d_to_centroid = np.linalg.norm(G_norm - centroid, axis=1)
        seed = int(np.argmin(d_to_centroid))
    elif seed_method == "random":
        rng = np.random.default_rng(rng_seed)
        seed = int(rng.integers(0, N))
    else:
        raise ValueError(f"未知 seed_method: {seed_method}")

    selected: list[int] = [seed]
    min_dists: list[float] = [0.0]  # seed 自己對自己 dist=0

    # current_min[i] = i 到「目前已選集合」的最小距離
    current_min = np.linalg.norm(G_norm - G_norm[seed], axis=1)

    for _ in range(n_pick - 1):
        # 排除已選的 idx
        masked = current_min.copy()
        for s in selected:
            masked[s] = -np.inf
        next_idx = int(np.argmax(masked))
        next_min_dist = float(current_min[next_idx])

        selected.append(next_idx)
        min_dists.append(next_min_dist)

        # 更新 current_min
        new_dists = np.linalg.norm(G_norm - G_norm[next_idx], axis=1)
        current_min = np.minimum(current_min, new_dists)

    return FPSResult(indices=selected, min_distances=min_dists, seed_index=seed)


# ---------------------------------------------------------------------------
# Diversity 量化（給 selection_log 用）
# ---------------------------------------------------------------------------


def diversity_stats(G_norm: np.ndarray, indices: list[int]) -> dict:
    """計算選中集合的多樣性統計（與全 pool 對比）。"""
    sel = G_norm[indices]
    pool_std = G_norm.std(axis=0)
    sel_std = sel.std(axis=0)
    # diversity_gain = 選中集合在 z-score 空間的平均對 pool 的 std 比
    # 期望 ≈ 1.0（z-score 後 pool std=1）
    return {
        "n_picked": len(indices),
        "pool_std_per_dim": pool_std.tolist(),
        "selected_std_per_dim": sel_std.tolist(),
        "selected_std_mean": float(sel_std.mean()),
        "pairwise_min_dist": float(min(_pairwise_min_distances(sel))),
        "pairwise_mean_dist": float(_pairwise_mean_distance(sel)),
    }


def _pairwise_min_distances(X: np.ndarray) -> list[float]:
    """每個點到最近鄰的距離（用於檢查多樣性下限）。"""
    n = X.shape[0]
    out: list[float] = []
    for i in range(n):
        dists = np.linalg.norm(X - X[i], axis=1)
        dists[i] = np.inf
        out.append(float(dists.min()))
    return out


def _pairwise_mean_distance(X: np.ndarray) -> float:
    """所有 pair 的平均距離。"""
    n = X.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    cnt = 0
    for i in range(n - 1):
        d = np.linalg.norm(X[i + 1 :] - X[i], axis=1)
        total += float(d.sum())
        cnt += d.shape[0]
    return total / cnt


# ---------------------------------------------------------------------------
# 輸出寫入
# ---------------------------------------------------------------------------


def write_outputs(
    out_dir: Path,
    selected_meta: list[dict],
    selected_indices: list[int],
    G_raw: np.ndarray,
    G_norm: np.ndarray,
    fps_result: FPSResult,
    pool_meta_path: Path,
    pool_masks_dir: Path,
    div_stats: dict,
    seed_method: str,
    norm_stats: dict,
) -> None:
    """寫出所有產出檔案到 out_dir。"""
    out_masks = out_dir / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)

    # 1. 複製 PNG
    print(f"[output] 複製 {len(selected_meta)} 張 mask 到 {out_masks}")
    for entry in selected_meta:
        src = pool_masks_dir / entry["file_name"]
        dst = out_masks / entry["file_name"]
        shutil.copy2(src, dst)

    # 2. mask_meta.json
    meta_path = out_dir / "mask_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(selected_meta, f, indent=2, ensure_ascii=False)
    print(f"[output] mask_meta.json 已寫入")

    # 3. selection_log.json
    selection_log = {
        "source_pool": str(pool_meta_path),
        "n_pool": int(G_raw.shape[0]),
        "n_picked": len(selected_indices),
        "seed_method": seed_method,
        "seed_pool_index": fps_result.seed_index,
        "feature_names": FEATURE_NAMES,
        "feature_dim": FEATURE_DIM,
        "normalization": norm_stats,
        "fps_order": [
            {
                "step": i,
                "pool_index": idx,
                "mask_id": selected_meta[i].get("mask_id", selected_meta[i]["file_name"]),
                "min_dist_to_selected": fps_result.min_distances[i],
            }
            for i, idx in enumerate(selected_indices)
        ],
        "diversity_stats": div_stats,
    }
    log_path = out_dir / "selection_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(selection_log, f, indent=2, ensure_ascii=False)
    print(f"[output] selection_log.json 已寫入")

    # 4. g_vec_table.csv（raw + normalized）
    csv_path = out_dir / "g_vec_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = (
            ["fps_step", "mask_id", "file_name", "pool_index"]
            + [f"raw_{n}" for n in FEATURE_NAMES]
            + [f"znorm_{n}" for n in FEATURE_NAMES]
        )
        writer.writerow(header)
        for step, idx in enumerate(selected_indices):
            entry = selected_meta[step]
            row = (
                [step, entry.get("mask_id", entry["file_name"]), entry["file_name"], idx]
                + [f"{v:.6g}" for v in G_raw[idx]]
                + [f"{v:.6g}" for v in G_norm[idx]]
            )
            writer.writerow(row)
    print(f"[output] g_vec_table.csv 已寫入")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    masks_dir: Path,
    meta_path: Path,
    out_dir: Path,
    n_pick: int = DEFAULT_N_PICK,
    seed_method: str = "centroid",
    rng_seed: int = 0,
    force_recompute: bool = False,
) -> None:
    """跑 FPS 採樣。

    Args:
        masks_dir: 來源 mask pool 的 PNG 目錄。
        meta_path: 對應的 mask_meta.json 路徑。
        out_dir: FPS 選中結果輸出目錄。
        n_pick: 要挑選的 mask 數量。
        seed_method: FPS 初始種子選法。
        rng_seed: random 模式的 seed。
        force_recompute: 忽略既有 G_vec cache 強制重算。
    """
    print(f"[main] 讀取 metadata: {meta_path}")
    meta = load_meta(meta_path)
    print(f"[main] meta 共 {len(meta)} 筆")

    cache_key = dataset_cache_key(meta_path.parent)
    cache_path = CACHE_DIR / f"raw_native_{cache_key}.npy"
    G_raw = load_or_compute(
        cache_path,
        meta,
        masks_dir,
        sim_cfg=DEFAULT_SIM_CFG,
        force=force_recompute,
    )

    if G_raw.shape[1] != FEATURE_DIM:
        raise RuntimeError(
            f"Cache 維度不符（{G_raw.shape[1]} vs {FEATURE_DIM}）；"
            f"請刪除 {cache_path} 重新計算（features.py schema 已變更？）"
        )

    finite_rows = np.all(np.isfinite(G_raw), axis=1)
    n_finite = int(finite_rows.sum())
    print(f"[main] G_raw shape={G_raw.shape}, finite rows={n_finite}")

    if n_finite < n_pick:
        raise RuntimeError(f"finite 樣本 ({n_finite}) 少於 n_pick ({n_pick})")

    # robust z-score 正規化（clip [p1, p99] 後 z-score）
    print(f"[main] robust normalize（clip [p{CLIP_LOW_PCT}, p{CLIP_HIGH_PCT}] then z-score）...")
    G_norm, norm_stats = robust_normalize(G_raw)

    # 把 NaN/inf row 設成全 0（FPS 距離計算不受影響但會自動避開極端）
    valid_indices = np.where(finite_rows)[0]
    G_norm_valid = G_norm[valid_indices]

    # FPS（在 valid 子集上）
    print(f"[main] FPS 採樣 n_pick={n_pick}, seed_method={seed_method}...")
    fps_result_local = fps_select(G_norm_valid, n_pick, seed_method, rng_seed)

    # 對應回 pool 的全域 index
    pool_indices = [int(valid_indices[i]) for i in fps_result_local.indices]
    fps_result = FPSResult(
        indices=pool_indices,
        min_distances=fps_result_local.min_distances,
        seed_index=int(valid_indices[fps_result_local.seed_index]),
    )

    selected_meta = [meta[i] for i in pool_indices]
    div_stats = diversity_stats(G_norm, pool_indices)

    print(f"[main] FPS 完成，順序前 5 個：")
    for i in range(min(5, n_pick)):
        idx = pool_indices[i]
        mid = selected_meta[i].get("mask_id", selected_meta[i]["file_name"])
        d = fps_result.min_distances[i]
        print(f"  step {i}: pool_idx={idx}, mask_id={mid}, min_dist={d:.4f}")
    print(
        f"[main] selected pairwise min/mean dist = "
        f"{div_stats['pairwise_min_dist']:.4f} / {div_stats['pairwise_mean_dist']:.4f}"
    )

    # 輸出
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(
        out_dir,
        selected_meta,
        pool_indices,
        G_raw,
        G_norm,
        fps_result,
        meta_path,
        masks_dir,
        div_stats,
        seed_method,
        norm_stats,
    )

    print(f"\n[done] 共 {n_pick} 個 mask 已輸出到 {out_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="FPS 採樣 30 個 mask（Phase G3 prep）")
    p.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="來源 mask pool 目錄（含 masks/ 與 mask_meta.json）",
    )
    p.add_argument(
        "--out-dir", type=Path, required=True, help="FPS 選中結果輸出目錄",
    )
    p.add_argument("--n-pick", type=int, default=DEFAULT_N_PICK)
    p.add_argument("--seed-method", choices=["centroid", "random"], default="centroid")
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="強制重算 G_vec cache")
    args = p.parse_args()

    main(
        masks_dir=args.dataset_dir / "masks",
        meta_path=args.dataset_dir / "mask_meta.json",
        out_dir=args.out_dir,
        n_pick=args.n_pick,
        seed_method=args.seed_method,
        rng_seed=args.rng_seed,
        force_recompute=args.force,
    )
