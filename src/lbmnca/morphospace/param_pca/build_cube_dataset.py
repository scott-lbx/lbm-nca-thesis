"""4D cube dataset pipeline：1M 候選 → project 4D → Sample-Elimination 300 →
LHS 16 root → multi-root MST → shortcuts。

步驟（對應原 qt_mst forward，空間換成 param-PCA 4D cube）：
  1. 母群體 1M 子抽 N_cand 候選（tractable）→ featurize → 指派 Re → project 4D cube
  2. Sample-Elimination（Yuksel 2015 加權貪婪）→ 300 cases（4D 均勻 blue-noise）
  3. LHS（Latin Hypercube）16 點 → snap 最近 case → 16 cold root
  4. 每 case 指派最近 root → 16 cluster；cluster 內 MST + BFS → parent/depth
  5. shortcut：非 MST 直連 + 4D 距離 < 門檻 → alt warm 邊
輸出 cases.json（params + re + cube + 圖結構）+ 2D 投影圖。mask render 為後續步驟。

用法：python -m lbmnca.morphospace.param_pca.build_cube_dataset \
            --pop data/design_space/population_big.npz \
            --cube data/design_space/cube4d.npz --n-cand 40000 --n-cases 300 --n-roots 16
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from lbmnca.morphospace.param_pca import config as C
from lbmnca.morphospace.param_pca.cube import Cube4D
from lbmnca.morphospace.param_pca.features import featurize
from lbmnca.paths import dataset_dir, design_space_dir


# ── Sample Elimination（Yuksel 2015，加權貪婪移除）────────────────────────
def sample_elimination(pts: np.ndarray, n_target: int, gamma: float = 2.0,
                       alpha: int = 8) -> np.ndarray:
    """4D blue-noise 子集：回保留 index（長 n_target）。"""
    N, d = pts.shape
    tree = cKDTree(pts)
    nn, _ = tree.query(pts, k=2)
    r_max = gamma * float(np.median(nn[:, 1])) * (N / n_target) ** (1.0 / d)
    nbrs = tree.query_ball_point(pts, r_max)

    def wij(i, j):
        dij = float(np.linalg.norm(pts[i] - pts[j]))
        return (1.0 - min(dij, r_max) / r_max) ** alpha

    w = np.zeros(N)
    nb = [[] for _ in range(N)]                         # (j, wij) 快取
    for i in range(N):
        for j in nbrs[i]:
            if j != i:
                c = wij(i, j)
                w[i] += c
                nb[i].append((j, c))
    alive = np.ones(N, bool)
    heap = [(-w[i], i) for i in range(N)]
    heapq.heapify(heap)
    n_alive = N
    while n_alive > n_target:
        negw, i = heapq.heappop(heap)
        if not alive[i] or -negw != w[i]:
            continue                                    # stale
        alive[i] = False
        n_alive -= 1
        for j, c in nb[i]:
            if alive[j]:
                w[j] -= c
                heapq.heappush(heap, (-w[j], j))
    return np.where(alive)[0]


# ── LHS（Latin Hypercube）──────────────────────────────────────────────
def latin_hypercube(n: int, d: int, rng: np.random.RandomState) -> np.ndarray:
    out = np.zeros((n, d))
    for k in range(d):
        out[:, k] = (rng.permutation(n) + rng.uniform(0, 1, n)) / n
    return out


def _layout_offsets(counts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def run(cfg: argparse.Namespace) -> None:
    rng = np.random.RandomState(cfg.seed)
    d = np.load(Path(cfg.pop))
    params, counts = d["params"], d["counts"]
    off = _layout_offsets(counts)
    M = len(counts)
    cube = Cube4D.load(cfg.cube)

    # 1. 子抽候選 + featurize + 指派 Re + project 4D
    n_cand = min(cfg.n_cand, M)
    cand = rng.choice(M, n_cand, replace=False)
    print(f"[1] 候選 {n_cand}/{M} → featurize + project 4D ...")
    X = np.stack([featurize(params[off[i]:off[i + 1]]) for i in cand])
    re = rng.uniform(cube.re_lo, cube.re_hi, n_cand)
    pts = cube.to_cube(X, re)                            # (n_cand, 4)

    # 2. Sample-Elimination → n_cases
    print(f"[2] Sample-Elimination {n_cand} → {cfg.n_cases} ...")
    keep = sample_elimination(pts, cfg.n_cases)
    cidx = cand[keep]                                   # 原 layout index
    P = pts[keep]                                       # (n_cases, 4)
    Re = re[keep]
    n = len(P)

    # 3. LHS 16 → snap 最近 case → roots
    print(f"[3] LHS {cfg.n_roots} → snap 最近 case = roots ...")
    lhs = latin_hypercube(cfg.n_roots, P.shape[1], rng)
    lo, hi = P.min(0), P.max(0)
    lhs_pts = lo + lhs * (hi - lo)
    roots, used = [], set()
    for q in lhs_pts:
        order = np.argsort(np.linalg.norm(P - q, axis=1))
        for j in order:
            if int(j) not in used:
                used.add(int(j)); roots.append(int(j)); break
    roots = np.array(roots)

    # 4. 每 case → 最近 root → cluster；cluster 內 MST + BFS
    print("[4] 指派最近 root → cluster MST + BFS ...")
    root_of = np.argmin(cdist(P, P[roots]), axis=1)     # 0..n_roots-1
    parent = np.full(n, -1)
    depth = np.zeros(n, int)
    D = cdist(P, P)
    for r in range(cfg.n_roots):
        members = np.where(root_of == r)[0]
        ridx_local = np.where(members == roots[r])[0][0]
        if len(members) == 1:
            continue
        sub = minimum_spanning_tree(D[np.ix_(members, members)]).toarray()
        sub = sub + sub.T
        adj = {i: np.nonzero(sub[i])[0].tolist() for i in range(len(members))}
        seen, q = {ridx_local}, [ridx_local]
        while q:
            u = q.pop(0)
            for w_ in adj[u]:
                if w_ not in seen:
                    seen.add(w_)
                    nd = depth[members[u]] + 1
                    if nd > cfg.max_depth:               # 超過 → re-root 成 cold（控 drift）
                        parent[members[w_]] = -1
                        depth[members[w_]] = 0
                    else:
                        parent[members[w_]] = int(members[u])
                        depth[members[w_]] = nd
                    q.append(w_)

    # 5. shortcuts：非 MST 直連 + 4D 距離 < 門檻，每 dst node 最多 cap 條（取最近）
    mst_set = {(min(i, parent[i]), max(i, parent[i])) for i in range(n) if parent[i] >= 0}
    edge_mean = float(np.mean([D[i, parent[i]] for i in range(n) if parent[i] >= 0]))
    thr = cfg.shortcut_mult * edge_mean
    cands = sorted((D[i, j], i, j)
                   for i in range(n) for j in range(i + 1, n)
                   if (i, j) not in mst_set and D[i, j] < thr)
    sc_of: dict = {}
    sc_cnt: dict = {}
    n_sc = 0
    for _, i, j in cands:                               # 近的先取
        src, dst = (i, j) if depth[i] <= depth[j] else (j, i)
        if sc_cnt.get(dst, 0) >= cfg.shortcut_cap:
            continue
        sc_of.setdefault(dst, []).append(src)
        sc_cnt[dst] = sc_cnt.get(dst, 0) + 1
        n_sc += 1
    print(f"[5] shortcuts={n_sc}（門檻={thr:.3f}=mean_edge×{cfg.shortcut_mult}, cap={cfg.shortcut_cap}/node）")

    # 輸出 cases.json
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = []
    for i in range(n):
        bld = params[off[cidx[i]]:off[cidx[i] + 1]]
        nodes.append({
            "case_id": f"{cfg.dataset_name}_c{i:04d}",
            "re": int(round(Re[i])),
            "cube": [float(x) for x in P[i]],
            "n_buildings": int(len(bld)),
            "buildings": [[float(v) for v in b] for b in bld],   # (cx,cy,w,h)
            "root_id": int(roots[root_of[i]]),
            "parent_idx": int(parent[i]),
            "depth": int(depth[i]),
            "ic_type": "cold" if parent[i] < 0 else "warm",
            "shortcut_parents": sorted(sc_of.get(i, [])),
        })
    n_roots_actual = int((parent < 0).sum())
    meta = {
        "dataset_id": cfg.dataset_name,
        "space": "param-PCA 4D cube (3 geom PC whiten + Re z)",
        "cube_geom_evr": [float(x) for x in cube.pca.explained_variance_ratio_[:cube.n_geom]],
        "n_cases": n, "n_roots": n_roots_actual, "n_shortcuts": n_sc,
        "max_depth": int(depth.max()),
        "method": "1M cand → SE(Yuksel) → LHS roots → multi-root MST → shortcuts",
        "n_candidates": n_cand, "shortcut_thr": thr,
        "nodes": nodes,
    }
    (out_dir / "cases.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {n} cases, {n_roots_actual} roots, {n_sc} shortcuts, max_depth={depth.max()} "
          f"→ {out_dir/'cases.json'}")
    _plot(P, roots, parent, depth, sc_of, cube, out_dir / "cube_dataset.png")


def _plot(P, roots, parent, depth, sc_of, cube, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 三個 2D 投影：(geom0,geom1)、(geom0,geom2)、(geom0,Re)
    pairs = [(0, 1, "PC1 密度", "PC2 大小"), (0, 2, "PC1 密度", "PC3 位置"),
             (0, 3, "PC1 密度", "Re")]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    for ax, (a, b, la, lb) in zip(axes, pairs):
        for i in range(len(P)):
            p = parent[i]
            if p >= 0:
                ax.plot([P[i, a], P[p, a]], [P[i, b], P[p, b]], "-", color="0.6", lw=0.5, alpha=0.5, zorder=1)
        for dst, srcs in sc_of.items():
            for s in srcs:
                ax.plot([P[dst, a], P[s, a]], [P[dst, b], P[s, b]], "--", color="red", lw=0.7, alpha=0.6, zorder=2)
        ax.scatter(P[:, a], P[:, b], s=14, c=depth, cmap="viridis", zorder=3)
        ax.scatter(P[roots, a], P[roots, b], s=70, c="k", zorder=4)
        ax.set_xlabel(la); ax.set_ylabel(lb)
    axes[0].set_title("cube 投影：geom0-1（黑=root, 灰=MST, 紅=shortcut, 色=depth）")
    axes[1].set_title("geom0-2")
    axes[2].set_title("geom0-Re")
    fig.suptitle(f"4D cube dataset：{len(P)} cases, {int((parent<0).sum())} roots", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="param-PCA 4D cube dataset（SE+LHS+MST+shortcut）")
    p.add_argument("--pop", type=str, default=design_space_dir() / "population_big.npz")
    p.add_argument("--cube", type=str, default=design_space_dir() / "cube4d.npz")
    p.add_argument("--dataset-name", type=str, default="S1_5-Cube4D_R16C300")
    p.add_argument("--out-dir", type=str, default=None,
                   help="預設 = dataset_dir(<dataset-name>)（見 paths.py）")
    p.add_argument("--n-cand", type=int, default=40000)
    p.add_argument("--n-cases", type=int, default=300)
    p.add_argument("--n-roots", type=int, default=16)
    p.add_argument("--shortcut-mult", type=float, default=0.9)
    p.add_argument("--shortcut-cap", type=int, default=2, help="每 node 最多 shortcut 數")
    p.add_argument("--max-depth", type=int, default=6, help="warm 鏈深度上限；超過 re-root 成 cold")
    p.add_argument("--seed", type=int, default=0)
    cfg = p.parse_args()
    if cfg.out_dir is None:
        cfg.out_dir = str(dataset_dir(cfg.dataset_name))
    run(cfg)


if __name__ == "__main__":
    main()
