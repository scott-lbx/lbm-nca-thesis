"""C2-2 WarmGraph sampler library — 逐案生成的 graph 資料集取樣器。

供 scripts/orchestrate_c2_2_bfs.py 逐案呼叫（gen-as-you-go，非整批 setup）。
規格：docs/DATASET_C2-2_WARMGRAPH_SPEC_2026-07-06.md

Graph 構成（300 edges / 200 nodes 預設）：
  A 骨架鏈    sample_chain_child()   full window  → 新 node
  B Re-only   sample_re_edge()       full window  → 新 node（同 mask、±ΔRe）
  C k-NN 加密 plan_knn_edges()       short window → 既有 node pair
  D backward  plan_backward_edges()  short window → 既有 edge 反向
  E clique    plan_clique_edges()    short window → 探針 clique 全連接

核心不變量：
  - canonical edge（chain/re）＝ dst node 首次模擬，h5 = raw/{node_id}.h5
  - relation edge（knn/backward/clique）＝ dst 已有 canonical，
    h5 = raw/{dst}__from_{src_tail}.h5，只錄 transient + 短驗證窗
  - parent_h5_path 永遠指向已 done 的 h5（orchestrator 保證）
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from lbmnca.paths import data_root, design_space_dir


def _import_morph_param_pca():
    """取得 morphospace 的設計空間元件（config / Cube4D / featurize / 合法性判定）。"""
    from lbmnca.morphospace.param_pca import config as morph_config
    from lbmnca.morphospace.param_pca.cube import Cube4D as morph_cube4d
    from lbmnca.morphospace.param_pca.features import featurize as morph_featurize
    from lbmnca.morphospace.param_pca.legality import (
        check_layout_legal as morph_check_layout_legal,
    )

    return morph_config, morph_cube4d, morph_featurize, morph_check_layout_legal


_C, Cube4D, featurize, check_layout_legal = _import_morph_param_pca()

_CUBE_NPZ = design_space_dir() / "cube4d.npz"
_WS_ROOT = data_root()

# ── 預設 graph 預算（spec §2.2；orchestrator CLI 可覆寫）──────────────────────
N_ROOTS = 20
CHAIN_DEPTH = 8  # A: N_ROOTS × CHAIN_DEPTH = 160 canonical edges
N_RE_EDGES = 40  # B
N_KNN_EDGES = 50  # C
N_BACKWARD_EDGES = 30  # D（geom 22 + re 8）
N_BACKWARD_RE = 8
N_CLIQUES = 2  # E: K=4 雙向全連接
CLIQUE_K = 4

# ── 擾動 / 編輯參數（spec §3.1-3.2）───────────────────────────────────────────
SINGLE_OBS_PROB = 0.7  # 單物件編輯機率（真實 GH 操作分布）
MAX_PARAM_SCALE = 1.0  # rand_scale ~ U(0, MAX)，normalized param space
NOISE_PROB = 0.5  # per-dim Bernoulli（多參數模式）
MIN_EDIT_PX = 3.0  # 最小可見編輯（3 mask px = 1 model cell）
MIN_EDIT_MASK_PX2 = 9  # rendered mask 至少 9 px 差異（3×3）
PERTURB_MAX_ATTEMPTS = 20  # 擾動 → 可見性 + 合法性 檢查的重試上限
RE_DELTA_CHOICES = (50, 100, -50, -100)
RE_MIN, RE_MAX = 50, 500
RE_DELTA_MIN_EFF = 25  # clip 後有效 ΔRe 下限

# ── 快照窗（spec §4）───────────────────────────────────────────────────────────
FULL_TOTAL_CTUS = 90.0
FULL_EXTRA_CTU = 50.0
SHORT_TOTAL_CTUS = 60.0
SHORT_EXTRA_CTU = 15.0

# ── Split（spec §3.7）─────────────────────────────────────────────────────────
N_HOLDOUT_CHAINS = 2

_PARAM_LO = np.array([_C.CX_LO, _C.CY_LO, _C.W_MIN, _C.H_MIN], dtype=float)
_PARAM_HI = np.array([_C.CX_HI, _C.CY_HI, _C.W_MAX, _C.H_MAX], dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# 幾何 helpers
# ══════════════════════════════════════════════════════════════════════════════


def render_mask(buildings: list) -> np.ndarray:
    """buildings [[cx,cy,w,h]...] → uint8 (H,W)，0=solid 255=fluid。"""
    mask = np.full((_C.H_TOTAL, _C.W_TOTAL), 255, np.uint8)
    for cx, cy, w, h in buildings:
        x0 = max(0, int(round(cx - w / 2)))
        x1 = min(_C.W_TOTAL, int(round(cx + w / 2)))
        y0 = max(0, int(round(cy - h / 2)))
        y1 = min(_C.H_TOTAL, int(round(cy + h / 2)))
        mask[y0:y1, x0:x1] = 0
    return mask


def load_cube() -> Cube4D:
    """載入凍結 cube4d.npz（只讀，永不 refit）。"""
    return Cube4D.load(_CUBE_NPZ)


def node_4d(buildings: list, re: int, cube: Cube4D) -> tuple[list, list]:
    """buildings + Re → (cube4d[4], cube_geom[3])。"""
    feat = featurize(np.array(buildings))
    cube4 = cube.to_cube(feat[None], np.array([re]))[0]
    geom3 = cube.geom_coords(feat[None])[0]
    return [float(x) for x in cube4], [float(x) for x in geom3]


def _perturb_flat(
    flat: np.ndarray, dim_mask: np.ndarray, scale: float, rng: np.random.RandomState
) -> np.ndarray:
    """normalized [0,1] 空間擾動（sample_warm_perturb 同公式）後反正規化。"""
    n = len(flat) // 4
    lo = np.tile(_PARAM_LO, n)
    hi = np.tile(_PARAM_HI, n)
    norm = (flat - lo) / (hi - lo)
    noise = rng.randn(len(flat))
    new_norm = np.clip(norm + scale * dim_mask * noise, 0.0, 1.0)
    return (new_norm * (hi - lo) + lo).round()


def perturb_buildings(
    buildings: list,
    scale: float,
    rng: np.random.RandomState,
    single_obs_prob: float = SINGLE_OBS_PROB,
    noise_prob: float = NOISE_PROB,
    max_attempts: int = PERTURB_MAX_ATTEMPTS,
) -> tuple[list, bool]:
    """擾動 buildings，強制 ≥3px 可見編輯，且拒絕不合法幾何。

    模式（spec §3.1）：
      - 機率 single_obs_prob：單物件編輯——只擾動一個矩形的 (cx,cy,w,h)
        子集（至少 1 維），match GH 使用者一次動一個矩形的操作分布。
      - 其餘：多參數擾動（Bernoulli(noise_prob) per dim，原 tree 行為）。

    每次 resample 依序檢查（任一不滿足即重試，≤ max_attempts）：
      1. 可見性：任一 param 位移 ≥ MIN_EDIT_PX，且 rendered mask 差異
         ≥ MIN_EDIT_MASK_PX2 px。
      2. 合法性：委派 `src.param_pca.legality.check_layout_legal`（與 cold
         分支 `generate.py` 共用同一組重疊/間距/越界判定），不重複實作。

    Returns:
        (new_buildings, single_obstacle_flag)

    Raises:
        RuntimeError: max_attempts 內找不到「可見且合法」的編輯（scale 過小
            → 不可見；scale 過大或母體過密 → 不合法）。拒絕靜默回退成不合法
            幾何。
    """
    flat = np.array(buildings, dtype=float).ravel()
    n_b = len(buildings)
    base_mask = render_mask(buildings)

    for _ in range(max_attempts):
        single = bool(rng.rand() < single_obs_prob)
        dim_mask = np.zeros(len(flat))
        if single:
            bi = int(rng.randint(0, n_b))
            sub = (rng.rand(4) < noise_prob).astype(float)
            if sub.sum() == 0:
                sub[rng.randint(0, 4)] = 1.0
            dim_mask[bi * 4 : (bi + 1) * 4] = sub
        else:
            dim_mask = (rng.rand(len(flat)) < noise_prob).astype(float)
            if dim_mask.sum() == 0:
                dim_mask[rng.randint(0, len(flat))] = 1.0

        new_flat = _perturb_flat(flat, dim_mask, scale, rng)
        if np.abs(new_flat - flat).max() < MIN_EDIT_PX:
            continue
        new_buildings_arr = new_flat.reshape(-1, 4)
        new_buildings = [[float(v) for v in row] for row in new_buildings_arr.tolist()]
        diff_px = int((render_mask(new_buildings) != base_mask).sum())
        if diff_px < MIN_EDIT_MASK_PX2:
            continue
        is_legal, _reason = check_layout_legal(new_buildings_arr)
        if not is_legal:
            continue
        return new_buildings, single

    raise RuntimeError(
        f"perturb_buildings: {max_attempts} 次內找不到可見（≥{MIN_EDIT_PX}px）"
        f"且合法的編輯 (scale={scale:.4f} 過小或過大？)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Node / Edge 建構
# ══════════════════════════════════════════════════════════════════════════════


def _src_tail(src_node_id: str) -> str:
    """src node id 尾 token（c0011），供 relation-edge case_id 命名。"""
    return src_node_id.split("_")[-1]


def make_node(
    node_id: str,
    mask_id: str,
    re: int,
    buildings: list,
    cube4: list,
    geom3: list,
    ic_type: str,
    tree_depth: int,
    root_id: str,
    canonical_edge_id: str = "",
) -> dict:
    """組合 nodes.json entry。"""
    return {
        "node_id": node_id,
        "mask_id": mask_id,
        "re": int(re),
        "buildings": buildings,
        "n_buildings": len(buildings),
        "cube": cube4,
        "cube_geom": geom3,
        "ic_type": ic_type,
        "tree_depth": int(tree_depth),
        "root_id": root_id,
        "canonical_edge_id": canonical_edge_id,
        "status": "pending",
    }


def make_edge(
    edge_id: str,
    src_node: str,
    dst_node: str,
    edge_type: str,
    edit_axis: str,
    window_type: str,
    case_id: str,
    single_obstacle: bool = False,
    rand_scale: float = -1.0,
    delta_re: int = 0,
    param_delta_norm: float = -1.0,
    replaces: str = "",
) -> dict:
    """組合 edges.json entry（spec §5 schema）。"""
    return {
        "edge_id": edge_id,
        "src_node": src_node,
        "dst_node": dst_node,
        "edge_type": edge_type,
        "edit_axis": edit_axis,
        "window_type": window_type,
        "case_id": case_id,
        "single_obstacle": bool(single_obstacle),
        "rand_scale": float(rand_scale),
        "delta_re": int(delta_re),
        "param_delta_norm": float(param_delta_norm),
        "split": "",
        "status": "pending",
        "replaces": replaces,
        "retry_count": 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# A — 骨架鏈 sampler（逐案）
# ══════════════════════════════════════════════════════════════════════════════


def sample_chain_child(
    parent_node: dict,
    dataset_name: str,
    node_idx: int,
    edge_idx: int,
    cube: Cube4D,
    rng: np.random.RandomState,
    scale: float,
    replaces: str = "",
) -> tuple[dict, dict]:
    """從 frontier parent 抽一個 chain child（geom-only、full window）。

    Args:
        parent_node: 已 done 的 parent node dict。
        dataset_name: C2-2 dataset id。
        node_idx / edge_idx: 流水號（journal 端遞增）。
        cube: 凍結 Cube4D。
        rng: seeded RNG。
        scale: 本次擾動幅度（orchestrator 依 retry 遞減）。
        replaces: 失敗重抽時填被取代的 edge_id。

    Returns:
        (node_dict, edge_dict)
    """
    child_buildings, single = perturb_buildings(parent_node["buildings"], scale, rng)
    child_re = int(parent_node["re"])  # geometry-only：Re 繼承
    node_id = f"{dataset_name}_c{node_idx:04d}"
    edge_id = f"{dataset_name}_e{edge_idx:04d}"
    cube4, geom3 = node_4d(child_buildings, child_re, cube)

    node = make_node(
        node_id=node_id,
        mask_id=node_id,
        re=child_re,
        buildings=child_buildings,
        cube4=cube4,
        geom3=geom3,
        ic_type="warm",
        tree_depth=parent_node["tree_depth"] + 1,
        root_id=parent_node["root_id"],
        canonical_edge_id=edge_id,
    )
    edge = make_edge(
        edge_id=edge_id,
        src_node=parent_node["node_id"],
        dst_node=node_id,
        edge_type="chain",
        edit_axis="geometry",
        window_type="full",
        case_id=node_id,
        single_obstacle=single,
        rand_scale=scale,
        param_delta_norm=float(
            np.linalg.norm(np.array(cube4) - np.array(parent_node["cube"]))
        ),
        replaces=replaces,
    )
    return node, edge


# ══════════════════════════════════════════════════════════════════════════════
# B — Re-only sampler（逐案）
# ══════════════════════════════════════════════════════════════════════════════


def sample_re_edge(
    anchor_node: dict,
    dataset_name: str,
    node_idx: int,
    edge_idx: int,
    cube: Cube4D,
    rng: np.random.RandomState,
    delta_re: int = 0,
    replaces: str = "",
) -> tuple[dict, dict]:
    """從 anchor node 抽 Re-only edge（同 mask、±ΔRe、full window → 新 node）。

    delta_re=0 → 從 RE_DELTA_CHOICES 隨機抽（clip 後有效 ΔRe ≥ RE_DELTA_MIN_EFF）。
    retry 遞減由 orchestrator 傳入減半後的 delta_re。

    Raises:
        RuntimeError: anchor Re 貼邊界使所有 choice 無效。
    """
    anchor_re = int(anchor_node["re"])
    if delta_re == 0:
        choices = list(RE_DELTA_CHOICES)
        rng.shuffle(choices)
        for d in choices:
            eff = int(np.clip(anchor_re + d, RE_MIN, RE_MAX)) - anchor_re
            if abs(eff) >= RE_DELTA_MIN_EFF:
                delta_re = d
                break
        else:
            raise RuntimeError(
                f"sample_re_edge: anchor Re={anchor_re} 無有效 ΔRe choice"
            )

    child_re = int(np.clip(anchor_re + delta_re, RE_MIN, RE_MAX))
    node_id = f"{dataset_name}_c{node_idx:04d}"
    edge_id = f"{dataset_name}_e{edge_idx:04d}"
    cube4, geom3 = node_4d(anchor_node["buildings"], child_re, cube)

    node = make_node(
        node_id=node_id,
        mask_id=anchor_node["mask_id"],
        re=child_re,
        buildings=anchor_node["buildings"],
        cube4=cube4,
        geom3=geom3,
        ic_type="warm",
        tree_depth=anchor_node["tree_depth"],
        root_id=anchor_node["root_id"],
        canonical_edge_id=edge_id,
    )
    edge = make_edge(
        edge_id=edge_id,
        src_node=anchor_node["node_id"],
        dst_node=node_id,
        edge_type="re",
        edit_axis="re",
        window_type="full",
        case_id=node_id,
        delta_re=child_re - anchor_re,
        param_delta_norm=float(
            np.linalg.norm(np.array(cube4) - np.array(anchor_node["cube"]))
        ),
        replaces=replaces,
    )
    return node, edge


# ══════════════════════════════════════════════════════════════════════════════
# C/D/E — relation-edge planners（骨架實現後批次計算）
# ══════════════════════════════════════════════════════════════════════════════


def _done_warm_nodes(nodes: list[dict]) -> list[dict]:
    """已 done 且非 root（canonical 在本 dataset 內）的 node。"""
    return [n for n in nodes if n["status"] == "done" and n["ic_type"] == "warm"]


def _existing_pairs(edges: list[dict]) -> set[tuple[str, str]]:
    """既有有向 (src, dst) pair（含 failed，避免重抽已知壞 pair）。"""
    return {(e["src_node"], e["dst_node"]) for e in edges}


def _relation_edge(
    dataset_name: str,
    edge_idx: int,
    src: dict,
    dst: dict,
    edge_type: str,
    edit_axis: str,
) -> dict:
    """建構 relation edge（short window，dst canonical 已存在）。

    knn/clique 依 4D 距離選 pair，可能跨 Re → edit_axis 標 mixed。
    """
    if edge_type in ("knn", "clique") and int(dst["re"]) != int(src["re"]):
        edit_axis = "mixed"
    case_id = f"{dst['node_id']}__from_{_src_tail(src['node_id'])}"
    return make_edge(
        edge_id=f"{dataset_name}_e{edge_idx:04d}",
        src_node=src["node_id"],
        dst_node=dst["node_id"],
        edge_type=edge_type,
        edit_axis=edit_axis,
        window_type="short",
        case_id=case_id,
        delta_re=int(dst["re"]) - int(src["re"]),
        param_delta_norm=float(
            np.linalg.norm(np.array(dst["cube"]) - np.array(src["cube"]))
        ),
    )


def plan_knn_edges(
    nodes: list[dict],
    edges: list[dict],
    dataset_name: str,
    next_edge_idx: int,
    n_edges: int = N_KNN_EDGES,
    k: int = 2,
) -> list[dict]:
    """C — 4D cube k-NN 局部加密（spec §3.3）。

    對每個已實現 warm node 取 k 最近鄰，排除既有 pair，全體候選按 4D
    距離排序取最短 n_edges 條。方向：creation order 較早者為 src。
    """
    pool = _done_warm_nodes(nodes)
    if len(pool) < 3:
        return []
    order = {n["node_id"]: i for i, n in enumerate(nodes)}
    pts = np.array([n["cube"] for n in pool])
    existing = _existing_pairs(edges)

    cands: dict[tuple[str, str], float] = {}
    for i, ni in enumerate(pool):
        d = np.linalg.norm(pts - pts[i], axis=1)
        d[i] = np.inf
        for j in np.argsort(d)[:k]:
            nj = pool[int(j)]
            a, b = ni, nj
            if order[a["node_id"]] > order[b["node_id"]]:
                a, b = b, a
            key = (a["node_id"], b["node_id"])
            if key in existing or (key[1], key[0]) in existing:
                continue
            cands[key] = min(cands.get(key, np.inf), float(d[j]))

    ranked = sorted(cands.items(), key=lambda kv: kv[1])[:n_edges]
    by_id = {n["node_id"]: n for n in pool}
    plans = []
    for (src_id, dst_id), _dist in ranked:
        plans.append(
            _relation_edge(
                dataset_name,
                next_edge_idx + len(plans),
                by_id[src_id],
                by_id[dst_id],
                "knn",
                "geometry",
            )
        )
    return plans


def plan_backward_edges(
    nodes: list[dict],
    edges: list[dict],
    dataset_name: str,
    next_edge_idx: int,
    rng: np.random.RandomState,
    n_edges: int = N_BACKWARD_EDGES,
    n_re: int = N_BACKWARD_RE,
) -> list[dict]:
    """D — 既有 canonical edge 反向（spec §3.4）。

    (A→B) ⇒ 跑 (B_steady → A)。dst（原 parent）須為 warm node（root 的
    canonical 在 cold dataset，mean_ref 跨 zarr 不支援 → 排除）。
    配額 geom : re = (n_edges - n_re) : n_re。
    """
    by_id = {n["node_id"]: n for n in nodes}
    done_warm = {n["node_id"] for n in _done_warm_nodes(nodes)}
    existing = _existing_pairs(edges)

    fwd = [
        e
        for e in edges
        if e["status"] == "done"
        and e["edge_type"] in ("chain", "re")
        and e["src_node"] in done_warm
        and e["dst_node"] in done_warm
        and (e["dst_node"], e["src_node"]) not in existing
    ]
    geom_pool = [e for e in fwd if e["edge_type"] == "chain"]
    re_pool = [e for e in fwd if e["edge_type"] == "re"]
    rng.shuffle(geom_pool)
    rng.shuffle(re_pool)

    picked = re_pool[:n_re] + geom_pool[: n_edges - min(n_re, len(re_pool))]
    picked = picked[:n_edges]

    plans = []
    for e in picked:
        src, dst = by_id[e["dst_node"]], by_id[e["src_node"]]  # 反向
        etype = "re_backward" if e["edge_type"] == "re" else "geom_backward"
        axis = "re" if e["edge_type"] == "re" else "geometry"
        plans.append(
            _relation_edge(
                dataset_name, next_edge_idx + len(plans), src, dst, etype, axis
            )
        )
    return plans


def plan_clique_edges(
    nodes: list[dict],
    edges: list[dict],
    dataset_name: str,
    next_edge_idx: int,
    n_cliques: int = N_CLIQUES,
    clique_k: int = CLIQUE_K,
) -> tuple[list[dict], list[list[str]]]:
    """E — 探針 clique：4D 半徑最小的 K-node 群，雙向全連接（spec §3.5）。

    Returns:
        (edge plans, clique 成員清單（供 split hold-out）)
    """
    pool = _done_warm_nodes(nodes)
    if len(pool) < clique_k:
        return [], []
    pts = np.array([n["cube"] for n in pool])
    existing = _existing_pairs(edges)
    used: set[int] = set()
    plans: list[dict] = []
    cliques: list[list[str]] = []

    for _ in range(n_cliques):
        best_group, best_r = None, np.inf
        for i in range(len(pool)):
            if i in used:
                continue
            d = np.linalg.norm(pts - pts[i], axis=1)
            d[list(used)] = np.inf
            d[i] = 0.0
            grp = np.argsort(d)[:clique_k]
            if any(int(g) in used for g in grp):
                continue
            r = float(d[grp].max())
            if r < best_r:
                best_r, best_group = r, [int(g) for g in grp]
        if best_group is None:
            break
        used.update(best_group)
        members = [pool[g] for g in best_group]
        cliques.append([m["node_id"] for m in members])
        for a in members:
            for b in members:
                if a["node_id"] == b["node_id"]:
                    continue
                if (a["node_id"], b["node_id"]) in existing:
                    continue
                plans.append(
                    _relation_edge(
                        dataset_name,
                        next_edge_idx + len(plans),
                        a,
                        b,
                        "clique",
                        "geometry",
                    )
                )
    return plans, cliques


# ══════════════════════════════════════════════════════════════════════════════
# Split（spec §3.7：root-subtree group split + clique hold-out）
# ══════════════════════════════════════════════════════════════════════════════


def assign_split(
    nodes: list[dict],
    edges: list[dict],
    clique_members: list[list[str]],
    rng: np.random.RandomState,
    n_holdout_chains: int = N_HOLDOUT_CHAINS,
    split_basis: str = "legacy_root_subtree",
) -> dict[str, Any]:
    """root-subtree group split：整鏈同組，clique 全 hold-out。

    test = 隨機 n_holdout_chains 條 root 鏈全 subtree（node + 觸及 edge）
           + 全部 clique edge。其餘 train。

    ──────────────────────────────────────────────────────────────────────
    ★★ DEEP_AUDIT_2026-07 R1（[S2] 分組鍵洩漏）2026-07-28 註記
    ──────────────────────────────────────────────────────────────────────
    本函式是**生成端的第二套 split 邏輯**，與 canonical 的
    `src/config_gen/lineage_split.py::assign_lineage_split` 判準不同：

    1. 只產 `train` / `test` 兩分，**沒有 val** —— 訓練期的 val 因此只能由
       loader 自行重算，那正是 [S2] 的起點。
    2. edge 規則是「任一端為 test 即 test」（見下方 `or` 條件），而裁決 D-3.1c
       要求「兩端同 split 才成立、跨群組一律 dropped」。
    3. 分組鍵只有 `root_id`，不看幾何指紋 —— 同一幾何出現在不同 lineage 時
       （C-series 實測存在）仍會跨 split。

    因此 canonical 流程是：生成端標好 `test` 承諾 → `scripts/backfill_graph_split.py`
    以 lineage_split 重新三分（既有 test 一個都不移出，D-3.1b）。

    `split_basis` 控制本函式的行為，**預設維持既有行為**：

    * `"legacy_root_subtree"`（預設）：照舊寫 train/test，並在回傳的 summary
      標記 `split_basis`，供下游辨識這份標記尚未經 lineage 三分。
    * `"defer"`：node/edge 的 `split` 一律寫空字串，交由 backfill 標定。
      lbmnca.nca 的 loader 遇到空字串會**直接報錯並指出補救指令**
      （`graph_split.assert_split_labelled`，D-3.1d），不會靜默當成 train。

    Args:
        nodes: node dict 列表（就地寫入 `split`）。
        edges: edge dict 列表（就地寫入 `split`）。
        clique_members: clique 成員 node_id 群組。
        rng: 決定性隨機源。
        n_holdout_chains: hold-out 的 root 鏈數。
        split_basis: `"legacy_root_subtree"` 或 `"defer"`（見上）。

    Returns:
        summary dict（含 `split_basis`；`defer` 模式下另有 `deferred=True`）。

    Raises:
        ValueError: `split_basis` 不是上述兩者之一。
    """
    if split_basis not in ("legacy_root_subtree", "defer"):
        raise ValueError(
            f"split_basis 必須是 'legacy_root_subtree' 或 'defer'，得到 {split_basis!r}"
        )
    if split_basis == "defer":
        for n in nodes:
            n["split"] = ""
        for e in edges:
            e["split"] = ""
        return {
            "split_basis": split_basis,
            "deferred": True,
            "n_edges": len(edges),
            "n_nodes": len(nodes),
            "note": "split 交由 scripts/backfill_graph_split.py 以 lineage 三分標定",
        }

    roots = sorted({n["root_id"] for n in nodes if n["ic_type"] == "warm"})
    rng.shuffle(roots)
    test_roots = set(roots[:n_holdout_chains])

    node_split: dict[str, str] = {}
    for n in nodes:
        n_split = "test" if n["root_id"] in test_roots else "train"
        node_split[n["node_id"]] = n_split
        n["split"] = n_split

    clique_nodes = {nid for grp in clique_members for nid in grp}
    n_test = 0
    for e in edges:
        if e["edge_type"] == "clique":
            e["split"] = "test"
        elif (
            node_split.get(e["src_node"]) == "test"
            or node_split.get(e["dst_node"]) == "test"
            or e["src_node"] in clique_nodes
            and e["dst_node"] in clique_nodes
        ):
            e["split"] = "test"
        else:
            e["split"] = "train"
        n_test += int(e["split"] == "test")

    return {
        "test_roots": sorted(test_roots),
        "n_test_edges": n_test,
        "n_edges": len(edges),
        "split_basis": split_basis,
        "note": (
            "兩分（無 val）、edge 只約束單端 —— 必須再跑 "
            "scripts/backfill_graph_split.py 以 lineage 三分定案（D-3.1a~c）"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Config YAML（wrap gen_c_series_configs.build_yaml_config）
# ══════════════════════════════════════════════════════════════════════════════


def build_edge_config(
    edge: dict,
    src_node: dict,
    dst_node: dict,
    dataset_name: str,
    cold_dataset: str,
    ds_dir: Path,
) -> dict[str, Any]:
    """為一條 edge 組 per-case YAML dict。

    canonical edge → full window（WARM_TOTAL/EXTRA 沿用 gen_c 預設）；
    relation edge → short window 覆寫 total_ctus / extra_ctu_after_conv。
    parent = src node canonical h5（root 在 cold dataset、warm 在本 dataset）。
    """
    from lbmnca.lbm.config_gen.case_config import build_yaml_config

    src_is_root = src_node["ic_type"] == "cold"
    node_dict = {
        "case_id": edge["case_id"],
        "mask_id": dst_node["mask_id"],
        "re": int(dst_node["re"]),
        "buildings": dst_node["buildings"],
        "n_buildings": dst_node["n_buildings"],
        "ic_type": "warm",
        "parent_case_id": src_node["node_id"],
        "parent_dataset": cold_dataset if src_is_root else dataset_name,
        "tree_depth": dst_node["tree_depth"],
    }
    cfg = build_yaml_config(node_dict, dataset_name, ds_dir / "masks", _WS_ROOT)

    tes = cfg["tes"]
    tes["dataset_name"] = dataset_name
    tes["edge_id"] = edge["edge_id"]
    tes["edge_type"] = edge["edge_type"]
    tes["edit_axis"] = edge["edit_axis"]
    tes["window_type"] = edge["window_type"]
    tes["canonical_case_id"] = (
        dst_node["node_id"] if edge["window_type"] == "short" else ""
    )
    tes["perturb_type"] = edge["edge_type"]
    tes["perturb_magnitude"] = float(edge["rand_scale"])

    if edge["window_type"] == "short":
        tes["total_ctus"] = SHORT_TOTAL_CTUS
        tes["extra_ctu_after_conv"] = SHORT_EXTRA_CTU
    else:
        tes["total_ctus"] = FULL_TOTAL_CTUS
        tes["extra_ctu_after_conv"] = FULL_EXTRA_CTU
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Journal I/O（atomic write；edges.json 為狀態 SSOT）
# ══════════════════════════════════════════════════════════════════════════════


def _atomic_write_json(path: Path, data: Any) -> None:
    """tmp + os.replace，防中斷寫壞 journal。"""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_journal(
    ds_dir: Path, meta: dict, nodes: list[dict], edges: list[dict]
) -> None:
    """flush nodes.json + edges.json（每條 edge settle 後呼叫）。"""
    _atomic_write_json(ds_dir / "nodes.json", {"meta": meta, "nodes": nodes})
    _atomic_write_json(ds_dir / "edges.json", {"meta": meta, "edges": edges})


def load_journal(ds_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    """載入 journal；不存在回傳空（fresh start）。

    resume 規則（spec §6.4）：殘留 status="running"（程序被殺）由
    orchestrator 視同 failed 重抽。
    """
    nodes_p, edges_p = ds_dir / "nodes.json", ds_dir / "edges.json"
    if not nodes_p.exists() or not edges_p.exists():
        return {}, [], []
    nd = json.loads(nodes_p.read_text(encoding="utf-8"))
    ed = json.loads(edges_p.read_text(encoding="utf-8"))
    return nd.get("meta", {}), nd["nodes"], ed["edges"]
