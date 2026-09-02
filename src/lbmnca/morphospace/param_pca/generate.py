"""母群體生成：自包含 rectangle-layout 取樣器 → 凍結 NPZ（純參數，不 render）。

每佈局 = (n,4) [cx,cy,w,h]，n∈[N_MIN,N_MAX]，5 型尺寸（大小錯落）+ rejection（gap/邊界）。
NPZ 以 concatenated params (M,4) + counts (N,) 存（不用 pickle）。

用法：python -m lbmnca.morphospace.param_pca.generate \
            --n-pop 30000 --seed 0 --out data/design_space/population.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from lbmnca.morphospace.param_pca import config as C
from lbmnca.paths import design_space_dir


def _sample_size(rng: np.random.RandomState) -> tuple[int, int]:
    """5 型尺寸（小街屋/中方/大型/寬板/高板）→ (w,h)，clamp [W_MIN,W_MAX]。"""
    t = rng.randint(5)
    if t == 0:
        base = rng.randint(C.W_MIN, 81); w, h = base, int(base * rng.uniform(0.7, 1.4))
    elif t == 1:
        base = rng.randint(70, 151); w, h = base, int(base * rng.uniform(0.75, 1.33))
    elif t == 2:
        base = rng.randint(140, C.W_MAX + 1); w, h = base, int(base * rng.uniform(0.7, 1.4))
    elif t == 3:
        short = rng.randint(C.W_MIN, 100); w, h = int(short * rng.uniform(1.8, 5.0)), short
    else:
        short = rng.randint(C.H_MIN, 100); w, h = short, int(short * rng.uniform(1.8, 5.0))
    return max(C.W_MIN, min(C.W_MAX, w)), max(C.H_MIN, min(C.H_MAX, h))


def violates_gap(a: np.ndarray, b: np.ndarray) -> bool:
    """a,b = [cx,cy,w,h]；擴 MIN_GAP 的 AABB 重疊 → True。

    ★ 幾何合法性判定的單一真相（重疊 + 間距不足合併判定，MIN_GAP=0 等價於
    純重疊測試）。`src/param_pca/legality.py` 匯入本函式供 warm 擾動分支
    （`perturb_buildings`，scripts/qa/sample_warm_perturb.py 與
    src/lbmnca/lbm/config_gen/c2_2_graph_sampler.py）做 rejection sampling，
    不另外重複實作，避免 cold/warm 兩分支的合法性母體分布分歧。
    """
    return (abs(a[0] - b[0]) < (a[2] + b[2]) / 2 + C.MIN_GAP and
            abs(a[1] - b[1]) < (a[3] + b[3]) / 2 + C.MIN_GAP)


def generate_layout(rng: np.random.RandomState) -> np.ndarray | None:
    """回 (n,4) [cx,cy,w,h]；建物全在 inner box、滿足 gap；n<N_MIN → None。"""
    n_target = rng.randint(C.N_MIN, C.N_MAX + 1)
    placed: list[np.ndarray] = []
    for _ in range(n_target):
        w, h = _sample_size(rng)
        hw, hh = w / 2.0, h / 2.0
        x_lo, x_hi = C.CX_LO + hw, C.CX_HI - hw
        y_lo, y_hi = C.CY_LO + hh, C.CY_HI - hh
        if x_lo >= x_hi or y_lo >= y_hi:
            continue
        for _ in range(C.MAX_PLACE_TRIES):
            cand = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), w, h])
            if all(not violates_gap(cand, o) for o in placed):
                placed.append(cand)
                break
    if len(placed) < C.N_MIN:
        return None
    return np.asarray(placed, dtype=np.float32)


def _worker(args: tuple[int, int]) -> list[np.ndarray]:
    """子進程：給定 (seed, n) 生 n 張合法佈局（各自獨立 RNG → 可重現）。"""
    seed, n = args
    rng = np.random.RandomState(seed)
    out: list[np.ndarray] = []
    guard = 0
    while len(out) < n and guard < n * 200:
        guard += 1
        lay = generate_layout(rng)
        if lay is not None:
            out.append(lay)
    return out


def _try_tqdm(total: int, desc: str):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, ncols=80, unit="layout")
    except ImportError:
        return None


def build_population(n_pop: int, seed: int = 0, workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """生成 n_pop 個佈局 → (params (M,4) concatenated, counts (N,))。workers>1 → multiprocessing。"""
    layouts: list[np.ndarray] = []
    if workers and workers > 1:
        from multiprocessing import Pool
        n_tasks = workers * 20                              # 多切細塊 → pbar 更新更細
        sizes = [n_pop // n_tasks] * n_tasks
        for i in range(n_pop % n_tasks):
            sizes[i] += 1
        args = [(seed * 100003 + i, s) for i, s in enumerate(sizes) if s > 0]
        pbar = _try_tqdm(n_pop, f"生成佈局(x{workers})")
        with Pool(workers) as pool:
            for chunk in pool.imap_unordered(_worker, args):
                layouts.extend(chunk)
                if pbar is not None:
                    pbar.update(len(chunk))
        if pbar is not None:
            pbar.close()
    else:
        rng = np.random.RandomState(seed)
        pbar = _try_tqdm(n_pop, "生成佈局")
        tries = 0
        next_mark = 5000
        while len(layouts) < n_pop and tries < n_pop * 50:
            tries += 1
            lay = generate_layout(rng)
            if lay is not None:
                layouts.append(lay)
                if pbar is not None:
                    pbar.update(1)
                elif len(layouts) >= next_mark:
                    print(f"  {len(layouts)}/{n_pop} ...")
                    next_mark += 5000
        if pbar is not None:
            pbar.close()
    counts = np.array([len(l) for l in layouts], dtype=np.int32)
    params = np.concatenate(layouts, axis=0).astype(np.float32)
    print(f"[pop] {len(layouts)}/{n_pop} 佈局；建物總數 {len(params)}；"
          f"n: min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")
    return params, counts


def main() -> None:
    p = argparse.ArgumentParser(description="param_pca 母群體生成（純參數 NPZ）")
    p.add_argument("--n-pop", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="multiprocessing 進程數（預設 = CPU−1）")
    p.add_argument("--out", type=str, default=design_space_dir() / "population.npz")
    cfg = p.parse_args()

    print(f"[gen] n_pop={cfg.n_pop} workers={cfg.workers} seed={cfg.seed}")
    params, counts = build_population(cfg.n_pop, cfg.seed, cfg.workers)
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, params=params, counts=counts,
        n_pop=len(counts), seed=cfg.seed,
        ranges=np.array([C.W_MIN, C.W_MAX, C.H_MIN, C.H_MAX,
                         C.CX_LO, C.CX_HI, C.CY_LO, C.CY_HI], dtype=np.int32),
        param_cols=np.array(C.PARAM_COLS),
    )
    print(f"saved {out}（{out.stat().st_size/1e6:.1f} MB）")


if __name__ == "__main__":
    main()
