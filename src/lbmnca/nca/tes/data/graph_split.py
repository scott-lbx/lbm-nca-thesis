"""v4 graph dataset 的 train/val/test 讀取與多 zarr 組裝。

split 是資料的屬性，不是訓練期的函式。本模組只讀 `graph.json` 的 `split` 欄，不算。
切分由資料生成端一次算好、回填進 `graph.json`；
本模組只依 `split` 欄取集合。`split` 欄的合法值為 `"train"` / `"val"` / `"test"` /
`"dropped"`（`"dropped"` = 兩端點分屬不同 lineage 群組的 edge，任何 split 都不得
使用）。**遇到空字串或未知值一律 raise** —— 禁用 `n.get("split", "train")` 之類的
預設，那正是製造靜默崩壞的寫法。

兩個語義限制（皆為避免資料洩漏）：

1. **train 只收 `ic_source == "parent_mean"` 的 edge。**冷啟動 case 混進 train
   會使 `_collate_variable_length` 的 `all("parent_mean_renorm" in s ...)` 判斷
   失效——一個冷啟動 case 就讓整個 batch 掉到 `mean_step` 的冷啟動分支，同批的
   warm case 一併被降級。
2. **geometry-level split**：切分依據為 lineage 群組（幾何擾動樹的世系關係），
   而非單一節點的 `mask_id`——後者在 1:1 擾動樹下幾乎逐案唯一，等同不切分。

本模組另提供 `build_split_chains()` —— **邊走訪式**鏈建構（在單一 split 內以 visited
集合建簡單鏈，湊不滿即 raise）。鏈式評測請呼叫它，不要各自用 `out_edges()`
手刻隨機走訪。

本段可數宣稱的守衛：

    python -m pytest tests/nca/tes/test_r1_chain_claim_census.py -q
"""

from __future__ import annotations

import zlib
import json
import logging
import random
from pathlib import Path
from typing import Any, NamedTuple, Optional

from torch.utils.data import ConcatDataset, Dataset

from lbmnca.nca.tes.data.graph_dataset import (
    EdgeChainMeanDataset,
    EdgeMeanDataset,
    enumerate_chains,
)
from lbmnca.nca.tes.data.graph_index import GraphIndex

_log = logging.getLogger(__name__)

VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test", "dropped"})


def _role_from_dataset(graph) -> bool:
    """判斷該 graph 是否為「角色由 dataset 決定」的 unified 格式。

    ★不用 ``.get("split", default)`` —— 那會讓「欄位遺失」與「值為預設」不可區分，
    是製造靜默崩壞的寫法。這裡看的是 **鍵是否存在**。

    Args:
        graph: GraphIndex。

    Returns:
        True 表示資料層無 split 欄（unified／A 方案），角色須由呼叫端注入。
    """
    if not graph.nodes:
        return False
    any_node = next(iter(graph.nodes.values()))
    return "split" not in any_node

"""`graph.json` 的 `split` 欄合法值。`dropped` 表任何 split 都不得使用。"""


class GraphSplitDatasets(NamedTuple):
    """`split_graph_datasets()` 的回傳。

    Attributes:
        train: train 側 warm（`parent_mean`）edge dataset。
        val_cold: val 側 `cold_start` edge dataset（可能為 None）。
        val_warm: val 側 warm edge dataset（可能為 None）。
        test_cold: test 側 `cold_start` edge dataset（可能為 None）。
        test_warm: test 側 warm edge dataset（可能為 None）。
    """

    train: Dataset
    val_cold: Optional[Dataset]
    val_warm: Optional[Dataset]
    test_cold: Optional[Dataset]
    test_warm: Optional[Dataset]


def assert_split_labelled(graph: GraphIndex) -> None:
    """檢查 graph 的每個 node / edge 都帶合法 `split` 標記。

    ★ `split` 為空字串（v4 migration 的預設）時，
    `edges_by(split="train")` 會靜默回傳空集合、訓練集無聲崩掉。
    此處改為**立刻報錯並指出補救指令**。

    Args:
        graph: GraphIndex。

    Raises:
        RuntimeError: 任一 node/edge 的 `split` 不在 `VALID_SPLITS`。
    """
    # ★A 方案（tes_unified_v1）：split 已從資料層移除，角色由 config 的 zarr_paths
    #   決定。此時 node/edge 完全沒有 `split` 鍵——這是**預期狀態**，不是未填妥。
    #   判別依據是「鍵不存在」而非「值是某個預設」——後者正是本函式 docstring
    #   所禁止的靜默崩壞寫法。
    if _role_from_dataset(graph):
        return

    bad_nodes = [n for n, rec in graph.nodes.items() if rec["split"] not in VALID_SPLITS]
    bad_edges = [e for e, rec in graph.edges.items() if rec["split"] not in VALID_SPLITS]
    if not bad_nodes and not bad_edges:
        return
    raise RuntimeError(
        f"{graph.dataset_id}: graph.json 的 split 欄未填妥"
        f"（{len(bad_nodes)} node / {len(bad_edges)} edge 非法，"
        f"例：node={bad_nodes[:2]} edge={bad_edges[:2]}）。\n"
        "split 是資料的屬性、由生成端一次算定，loader 不重算。\n"
        "補救：於資料生成端重跑 split 回填（見 docs/DATA_GENERATION.md）。"
    )


def edges_of_split(
    graph: GraphIndex,
    split: str,
    *,
    ic_source: str,
    max_attractor_ratio: Optional[float] = None,
) -> list[str]:
    """取某 split 的 edge（只讀 `split` 欄，不重算）。

    Args:
        graph: GraphIndex。
        split: `"train"` / `"val"` / `"test"`。
        ic_source: `"parent_mean"` / `"cold_start"`。
        max_attractor_ratio: short edge 的 attractor 比值上限，None = 不篩。

    Returns:
        排序後的 edge_id 列表。

    Raises:
        ValueError: `split` 不是 train/val/test。
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"edges_of_split：split 必須是 train/val/test，得到 {split!r}")
    if _role_from_dataset(graph):
        # ★unified：資料層無 split 欄，整本屬同一角色。呼叫端已用 role 決定要不要
        #   取這一本，故此處不再按 split 篩，只按 ic_source。
        return graph.edges_by(
            split=None, ic_source=ic_source, max_attractor_ratio=max_attractor_ratio
        )
    return graph.edges_by(
        split=split, ic_source=ic_source, max_attractor_ratio=max_attractor_ratio
    )


def _graph_json_for(zarr_path: str | Path) -> Path:
    """v4 zarr 路徑 -> 其 graph.json 路徑。"""
    p = Path(zarr_path)
    return p.parent / f"{p.stem}_graph.json"


CANONICAL_SPLIT_METHOD: str = "root_lineage_group_split"
"""canonical 切分演算法的方法名（由資料生成端寫入）。"""


def split_provenance_of(zarr_path: str | Path) -> Optional[dict[str, Any]]:
    """讀 `graph.json` 的 `split_provenance` 區塊（切分來源出處）。

    `split_provenance` 是 2026-07-28 backfill 才加入的欄位，早於該日期產生的
    graph.json 沒有這一欄 —— 故此處**不**視為錯誤，回傳 None 由呼叫端記錄，
    不改變任何載入行為（見 `_log_split_provenance`）。

    Args:
        zarr_path: v4 主 zarr 路徑。

    Returns:
        provenance dict；graph.json 無該欄時為 None。
    """
    gj = _graph_json_for(zarr_path)
    if not gj.exists():
        # ★unified（A 方案）：split 不在資料層，故「切分出處」這個概念不適用。
        #   回 None 由呼叫端記錄，不視為錯誤——這與「早於 2026-07-28 的 graph.json
        #   沒有該欄」是同一種處置。
        return None
    with open(gj, encoding="utf-8") as f:
        raw = json.load(f)
    # 明確 optional 欄位（舊 graph.json 無此區塊）→ 以 in 檢查取代 fallback 取值
    if "split_provenance" not in raw:
        return None
    return raw["split_provenance"]


CANONICAL_SPLIT_BASIS: str = "lineage"
"""資料生成端蓋章時寫入的世代標記。"""


def _log_split_provenance(zarr_path: str | Path, dataset_id: str) -> None:
    """把切分出處寫進訓練 log；來源非 canonical 時發 warning（不中止）。

    ★ 只記錄、不改行為（不 raise）：`split_basis` 並非所有 graph.json 都具備
    （backfill 尚未對全部 dataset 重跑）。在此 raise 會讓缺此欄的訓練整批掛掉。
    要升級為硬性 assert，須先為全部 dataset 重跑 split 回填並蓋章。

    本函式會讀 `split_basis`（以 `in` 檢查，缺欄只記 INFO）；backfill 端與 loader
    端的對照點為 `method`（比 :data:`CANONICAL_SPLIT_METHOD`）與 `split_basis`
    （比 :data:`CANONICAL_SPLIT_BASIS`）。

    Args:
        zarr_path: v4 主 zarr 路徑。
        dataset_id: 資料集名稱（log 用）。
    """
    prov = split_provenance_of(zarr_path)
    if prov is None:
        if not _graph_json_for(zarr_path).exists():
            # unified：split 不在資料層，「切分出處」概念不適用，非缺漏。
            _log.info(
                "%s: unified 格式（split 不在資料層），角色由 config 的 zarr_paths 指定",
                dataset_id,
            )
            return
        _log.warning(
            "%s: graph.json 無 split_provenance —— 無法確認 split 由哪一支程式產生"
            "（補救：於資料生成端重跑 split 回填並蓋章）",
            dataset_id,
        )
        return
    method = prov["method"]
    if method != CANONICAL_SPLIT_METHOD:
        _log.warning(
            "%s: split 來源為 %r，非 canonical 的 %r —— 可能是生成端 "
            "c2_2_graph_sampler.assign_split 的舊二分切分（無 val、edge 只約束單端）",
            dataset_id, method, CANONICAL_SPLIT_METHOD,
        )
        return
    # 明確 optional 欄位（2026-07-28 之前蓋章的 graph.json 沒有）→ 以 in 檢查取代
    # fallback 取值；缺欄不是錯誤，但要在 log 裡看得見「這份是舊世代蓋章」。
    if "split_basis" in prov:
        basis = prov["split_basis"]
        if basis != CANONICAL_SPLIT_BASIS:
            _log.warning(
                "%s: split_basis=%r，非 canonical 的 %r —— 這份 graph.json 由另一套"
                "切分基準蓋章，跨 dataset 的群組一致性未經保證",
                dataset_id, basis, CANONICAL_SPLIT_BASIS,
            )
    else:
        basis = None
        _log.info(
            "%s: split_provenance 無 split_basis 欄（2026-07-28 之前蓋章）—— "
            "重跑 backfill_graph_split.py --promote 可補上世代標記",
            dataset_id,
        )
    _log.info(
        "%s: split 出處 method=%s split_basis=%s val_fraction=%s seed=%s scope=%s",
        dataset_id, method, basis, prov["val_fraction"], prov["seed"], prov["scope"],
    )


_MAX_DFS_EXPANSIONS: int = 200_000
"""單一 root 的鏈搜尋展開上限（防病態圖上的指數爆炸）。"""


class ChainSpec(NamedTuple):
    """一條鏈的完整規格（節點序列 + 實際走過的 edge）。

    Attributes:
        nodes: node_id 序列，長度 == chain_len。
        edge_ids: 逐段使用的 edge_id，長度 == chain_len - 1。
        edge_types: 對應的 `edge_type`（`chain` / `re` / `knn` / `clique` /
            `geom_backward` / `re_backward`）。
    """

    nodes: list[str]
    edge_ids: list[str]
    edge_types: list[str]


def _dfs_chain(
    graph: GraphIndex,
    root: str,
    adj: dict[str, list[str]],
    chain_len: int,
    rng: random.Random,
    longest: list[str],
) -> Optional[ChainSpec]:
    """自 `root` 以回溯 DFS 找一條長度為 `chain_len` 的**簡單**鏈。

    ★ 與舊 `rq2_chain_eval.build_chains` 的差異：舊版每步隨機挑一條出邊、
    無 visited 集合，實測 10 節點鏈只走過 4~5 個相異節點（在兩三點間來回振盪），
    等於把「長距離鏈式推進」量成了「同一對節點反覆互推」。此處以 visited
    集合禁止重訪，走不下去就回溯換邊。

    Args:
        graph: GraphIndex。
        root: 起點 node_id。
        adj: node_id -> 允許使用的 out edge_id 列表（已過 split 與 edge_type 篩選）。
        chain_len: 目標節點數。
        rng: 決定性隨機源。
        longest: 就地更新的「目前找到最長路徑」（診斷用）。

    Returns:
        `ChainSpec`；找不到時 None。
    """
    path = [root]
    used_edges: list[str] = []
    visited = {root}
    budget = [_MAX_DFS_EXPANSIONS]

    def _rec(cur: str) -> bool:
        if len(path) > len(longest):
            longest[:] = list(path)
        if len(path) == chain_len:
            return True
        if budget[0] <= 0:
            return False
        outs = sorted(adj.get(cur, []))
        rng.shuffle(outs)
        for eid in outs:
            dst = graph.dst_of(eid)
            if dst in visited:
                continue
            budget[0] -= 1
            visited.add(dst)
            path.append(dst)
            used_edges.append(eid)
            if _rec(dst):
                return True
            path.pop()
            used_edges.pop()
            visited.remove(dst)
        return False

    ok = _rec(root)
    if budget[0] <= 0:
        _log.warning("鏈搜尋自 %s 起超過 %d 次展開，放棄該 root", root, _MAX_DFS_EXPANSIONS)
    if not ok:
        return None
    return ChainSpec(
        nodes=list(path),
        edge_ids=list(used_edges),
        edge_types=[graph.edges[e]["edge_type"] for e in used_edges],
    )


BACKWARD_EDGE_TYPES: tuple[str, ...] = ("geom_backward", "re_backward")
"""反向 edge 的 `edge_type`（C2-2 實測 21 條 geom_backward + 9 條 re_backward）。"""


def build_split_chain_specs(
    graph: GraphIndex,
    split: str,
    n_chains: int,
    chain_len: int,
    seed: int,
    *,
    ic_source: str = "parent_mean",
    max_attractor_ratio: Optional[float] = None,
    exclude_edge_types: tuple[str, ...] = (),
) -> list[ChainSpec]:
    """在**單一 split 內**建 `n_chains` 條長度 `chain_len` 的簡單鏈（含用到的 edge）。

    鏈的每一段都必須是該 split 的 edge —— 由於 lineage 切分要求 edge 兩端同
    split，這同時保證鏈上每個節點都屬於該 split，鏈式評測才真的是 held-out。

    ★F7 揭露一：本函式走的是**任何** `ic_source == "parent_mean"` 的 edge，
    不限 `edge_type`。C2-2 的 warm edge 含反向邊（`geom_backward` /
    `re_backward`，見 :data:`BACKWARD_EDGE_TYPES`），其中 test split 就有 5 條
    geom_backward + 1 條 re_backward，因此鏈**可能沿反向邊推進**。這在語義上是
    合法的：v4 的 edge = 「由 src 收斂態出發、改成 dst 幾何後再收斂」，反向邊只是
    編輯方向相反的同一種 transit，不是時間倒流。要排除請顯式傳
    `exclude_edge_types=BACKWARD_EDGE_TYPES`；**預設不排除＝維持既有行為**，
    每條鏈實際用到的 `edge_type` 一律隨 `ChainSpec` 回傳供落盤。

    ★F7 揭露二：`max_attractor_ratio` 預設 `None`＝不篩，與訓練載入器
    (`split_graph_datasets` 預設 3.0) 不同。也就是鏈可以經過「訓練端判定未收斂到
    canonical attractor 而剔除」的 short edge。維持 None 是為了不改既有 RQ2 數字；
    傳 `None` 時會 log 一行明講本次未套用。

    Args:
        graph: GraphIndex。
        split: `"train"` / `"val"` / `"test"`。
        n_chains: 需要的鏈數。
        chain_len: 每條鏈的節點數。
        seed: 決定性隨機源種子。
        ic_source: edge 的 IC 型態（鏈式推進必為 `"parent_mean"`）。
        max_attractor_ratio: short edge 的 attractor 比值上限，None = 不篩。
        exclude_edge_types: 排除的 `edge_type`，預設空 tuple = 不排除。

    Returns:
        `n_chains` 個 `ChainSpec`。

    Raises:
        RuntimeError: 該 split 的拓撲湊不滿 `n_chains` x `chain_len`
            —— 刻意 raise 而非 warning：湊不滿卻繼續跑，等於在使用者不知情下
            改變了受測協定（舊版只 log.warning 後照跑）。
    """
    allowed = set(
        edges_of_split(graph, split, ic_source=ic_source,
                       max_attractor_ratio=max_attractor_ratio)
    )
    n_before = len(allowed)
    if exclude_edge_types:
        allowed = {e for e in allowed
                   if graph.edges[e]["edge_type"] not in exclude_edge_types}
    if max_attractor_ratio is None:
        _log.info(
            "%s: build_split_chains 未套用 attractor 篩選（max_attractor_ratio=None；"
            "訓練載入器用 3.0）—— 鏈可能經過訓練端剔除的 short edge",
            graph.dataset_id,
        )
    _log.info(
        "%s: split=%s 可用 warm edge %d 條（排除 %s 後 %d 條）",
        graph.dataset_id, split, n_before, exclude_edge_types or "無", len(allowed),
    )

    adj: dict[str, list[str]] = {}
    for eid in sorted(allowed):
        src = graph.src_of(eid)
        if src is None:      # cold_start edge 無 src，不可能是鏈的中段
            continue
        adj.setdefault(src, []).append(eid)

    roots = sorted(adj)
    rng = random.Random(seed)
    rng.shuffle(roots)

    longest: list[str] = []
    specs: list[ChainSpec] = []
    for root in roots:
        if len(specs) >= n_chains:
            break
        spec = _dfs_chain(graph, root, adj, chain_len, rng, longest)
        if spec is not None:
            specs.append(spec)

    if len(specs) < n_chains:
        raise RuntimeError(
            f"{graph.dataset_id}: split={split!r} 只湊到 {len(specs)}/{n_chains} 條 "
            f"len={chain_len} 簡單鏈（可用 warm edge {len(allowed)} 條、"
            f"可當起點的 node {len(roots)} 個、實際找到的最長鏈 {len(longest)} 節點）。\n"
            f"→ 本 split 在 chain_len <= {len(longest)} 時才可能湊到鏈；"
            "請降 --chain-len 或 --n-chains，或改用其他 split。\n"
            "★不自動放寬：湊不滿就繼續跑，等於在使用者不知情下改掉受測協定，"
            "故鏈數與鏈長只能由呼叫端明確指定。"
        )
    for i, spec in enumerate(specs):
        _log.info("chain %d edge_types = %s", i, spec.edge_types)
    return specs


def build_split_chains(
    graph: GraphIndex,
    split: str,
    n_chains: int,
    chain_len: int,
    seed: int,
    *,
    ic_source: str = "parent_mean",
    max_attractor_ratio: Optional[float] = None,
    exclude_edge_types: tuple[str, ...] = (),
) -> list[list[str]]:
    """`build_split_chain_specs` 的節點序列視圖（既有呼叫端相容）。

    Args:
        graph: GraphIndex。
        split: `"train"` / `"val"` / `"test"`。
        n_chains: 需要的鏈數。
        chain_len: 每條鏈的節點數。
        seed: 決定性隨機源種子。
        ic_source: edge 的 IC 型態。
        max_attractor_ratio: short edge 的 attractor 比值上限，None = 不篩。
        exclude_edge_types: 排除的 `edge_type`，預設空 tuple = 不排除。

    Returns:
        `n_chains` 條鏈（每條為 node_id 列表）。
    """
    return [s.nodes for s in build_split_chain_specs(
        graph, split, n_chains, chain_len, seed, ic_source=ic_source,
        max_attractor_ratio=max_attractor_ratio,
        exclude_edge_types=exclude_edge_types,
    )]


def build_far_src_map(graph: GraphIndex) -> dict[str, str]:
    # far_edge 消融臂不是單一定義的處理——「最遠」所在的距離空間隨資料集切換：
    # 有 cube_geom 座標的節點用凍結 4D cube 空間，沒有的退回 14 維 g_vec 空間。
    # 兩個空間的「遠」不可通約，故跨資料集聚合 far_edge 的結果時，等於把兩種
    # 不同的處理混在同一臂裡；沿用本函式做消融時須逐資料集分開報告或統一距離空間。
    """三臂消融 far_edge 臂：每個節點 -> 同資料集中距離最遠的節點。

    用於把「編輯距離」單獨隔離出來。use_edge 的 IC 取自 param-PCA 空間中的
    近鄰父節點，far_edge 改取同一空間中的最遠節點，dst 與監督目標不變。

    距離空間優先取 `cube_geom`（凍結的 param-PCA 投影，與建圖時的近鄰判準同源）。
    少數資料集之節點無此欄位（C2-1 迴圈集），退回 `g_vec`（14 維幾何條件向量）
    並記錄於 log，不靜默。

    Args:
        graph: GraphIndex。

    Returns:
        node_id -> 最遠 node_id 的對照表，涵蓋該圖全部節點。
    """
    import numpy as np

    node_ids = list(graph.nodes)
    use_cube = all(graph.nodes[n].get("cube_geom") for n in node_ids)
    key = "cube_geom" if use_cube else "g_vec"
    if not use_cube:
        _log.warning(
            "%s: 部分節點無 cube_geom，far_edge 距離改用 g_vec（14 維）計算",
            graph.dataset_id,
        )

    feats = np.asarray([graph.nodes[n][key] for n in node_ids], dtype=np.float64)
    # 各維量綱不同（g_vec 尤甚），先標準化再算歐氏距離，避免單一大量綱維度主導
    std = feats.std(axis=0)
    std[std < 1e-12] = 1.0
    z = (feats - feats.mean(axis=0)) / std

    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    far_idx = d2.argmax(axis=1)
    out = {node_ids[i]: node_ids[int(j)] for i, j in enumerate(far_idx)}

    med = float(np.median(np.sqrt(d2[np.arange(len(node_ids)), far_idx])))
    _log.info(
        "%s: far_edge 對照表建妥（%d 節點，距離空間 %s，最遠距離中位 %.3f）",
        graph.dataset_id, len(out), key, med,
    )
    return out


def build_random_src_map(graph: GraphIndex, seed: int = 20260813) -> dict[str, str]:
    """`rand_edge` 消融臂：每個節點 -> 同資料集中隨機抽取的另一節點。

    ## 與 `far_edge` 的關係

    兩者落在 2×2 析因的同一格（起點是真實流場、但與目標無設計上的鄰近關係），
    `rand_edge` 是該格較乾淨的實現。`build_far_src_map` 的「最遠」所在的距離空間
    隨資料集切換（見該函式 docstring），跨資料集不可通約；
    隨機抽取不依賴任何距離空間，故無此問題。兩臂結果一致即為穩健性支持。

    ## 為什麼是同資料集內而非跨資料集

    跨資料集抽取會同時改變雷諾數分布與生成器設定，那是第二個變因。
    同資料集內抽取只動「與目標的設計關係」這一項，其餘條件不變。

    Args:
        graph: GraphIndex。
        seed: 亂數種子。與 `dataset_id` 併入以使各資料集抽到不同但可重現的結果。

    Returns:
        node_id -> 隨機 node_id 的對照表，涵蓋該圖全部節點，**保證不映到自己**。
    """
    import numpy as np

    node_ids = list(graph.nodes)
    n = len(node_ids)
    if n < 2:
        raise RuntimeError(
            f"{graph.dataset_id}: 只有 {n} 個節點，無法建立不映到自己的隨機配對")

    # 原用 abs(hash(dataset_id))，但 Python 字串 hash 逐進程隨機化，同一字串連跑
    #   三次會得三個不同種子 → abl_rand 的配對表不可重現，且配對表未落檔，
    #   事後無從重建。改用 zlib.crc32（決定性）。
    _did = graph.dataset_id.encode("utf-8")
    rng = np.random.default_rng((zlib.crc32(_did) % (2**32) + seed) % (2**32))
    # 先抽 [0, n-2] 再把「大於等於自己」的往後推一格，等價於自等機率排除自己。
    draw = rng.integers(0, n - 1, size=n)
    draw = np.where(draw >= np.arange(n), draw + 1, draw)
    out = {node_ids[i]: node_ids[int(j)] for i, j in enumerate(draw)}

    _log.info("%s: rand_edge 對照表建妥（%d 節點，seed %d）",
              graph.dataset_id, len(out), seed)
    return out


def build_src_override_map(graph: GraphIndex,
                           ablation_arm: str) -> Optional[dict[str, str]]:
    """依消融臂別回傳 src 覆寫表；不需覆寫者回 `None`。

    `use_edge` 與 `no_edge` 不動 src（前者用真實近鄰邊，後者於 `mean_step` 把 IC
    歸零），`noise_edge` 亦不動 src（它保留 parent 場，只在 `mean_step` 加噪）。

    Args:
        graph: GraphIndex。
        ablation_arm: 臂別。

    Returns:
        node_id -> 覆寫後 node_id 的對照表，或 `None`。
    """
    if ablation_arm in ("far_edge", "farnoise_edge"):
        # farnoise_edge = far 的 src 覆寫（B✗）＋ mean_step 的加噪（A✗）。
        return build_far_src_map(graph)
    if ablation_arm == "rand_edge":
        return build_random_src_map(graph)
    return None


def split_graph_datasets(
    zarr_paths: list[str],
    vflip_aug_prob: float = 0.0,
    max_attractor_ratio: Optional[float] = 3.0,
    ablation_arm: str = "use_edge",
    role: Optional[str] = None,
) -> GraphSplitDatasets:
    """依 `graph.json` 的 `split` 欄組多個 v4 zarr 的 train / val / test dataset。

    `*_warm` = 該 split 的 `parent_mean` edge（warm transfer 監督）。
    `*_cold` = 該 split 的 `cold_start` edge（chain root + 孤兒 case）。
    train 只含 `parent_mean`（見模組 docstring §1）；train 側的 `cold_start` edge
    與所有 `dropped` edge 都不進任何 loader，數量一律 log。

    ★ 不再接受 `val_fraction` / `seed` —— 切分不由訓練期決定。

    Args:
        zarr_paths: v4 主 zarr 路徑列表；`source_stats_idx` 依此順序指派。
        vflip_aug_prob: 垂直翻轉 augmentation 機率（僅套用於 train）。
        max_attractor_ratio: short edge 的 attractor 比值上限，None = 不篩。
        ablation_arm: 三臂消融臂別（`use_edge` / `far_edge` / ...）。

    Returns:
        `GraphSplitDatasets`；val/test 側可能為 None（該 split 無資料時）。

    Raises:
        RuntimeError: train 集為空，或任一 graph.json 的 `split` 欄未填妥。
    """
    parts: dict[str, list[Dataset]] = {
        "train": [], "val_cold": [], "val_warm": [], "test_cold": [], "test_warm": [],
    }

    for idx, zp in enumerate(zarr_paths):
        graph = GraphIndex.for_zarr(zp)
        assert_split_labelled(graph)
        _log_split_provenance(zp, graph.dataset_id)

        # ★A 方案：unified 格式的資料層無 split 欄，整本屬同一角色。
        #   角色由呼叫端的 `role` 指定（來自 config 的 zarr_paths 分組），
        #   缺 role 即報錯 —— 不給預設，避免「沒指定就當 train」這種靜默錯誤。
        unified = _role_from_dataset(graph)
        if unified:
            if role not in ("train", "val", "test"):
                raise ValueError(
                    f"{graph.dataset_id} 是 unified 格式（資料層無 split 欄），"
                    f" 必須由呼叫端指定 role∈{{train,val,test}}，得到 {role!r}。"
                    " 角色來自 config 的 zarr_paths 分組，不從資料推導。"
                )
            _bucket = role
        else:
            _bucket = None

        def _pick(split_name: str, **kw) -> list[str]:
            """unified 下只有 `_bucket` 那一桶有內容，其餘為空。"""
            if unified and split_name != _bucket:
                return []
            return edges_of_split(graph, split_name, **kw)

        warm = {
            s: _pick(s, ic_source="parent_mean", max_attractor_ratio=max_attractor_ratio)
            for s in ("train", "val", "test")
        }
        cold = {
            s: _pick(s, ic_source="cold_start")
            for s in ("train", "val", "test")
        }
        n_dropped = 0 if unified else sum(
            1 for e in graph.edges.values() if e["split"] == "dropped")
        n_attractor_cut = (
            len(graph.edges_by(ic_source="parent_mean"))
            - sum(len(v) for v in warm.values())
            - (0 if unified else sum(
                1 for e in graph.edges.values()
                if e["ic_source"] == "parent_mean" and e["split"] == "dropped"))
        )

        # cold_start edge 不進 train（見模組 docstring §1）；跨 lineage 群組的
        # dropped edge 任何 split 都不用 —— 兩者都明確記錄，不靜默截斷。
        _log.info(
            "%s: warm parent_mean train %d / val %d / test %d  |  "
            "cold_start val %d / test %d（train 側 %d 條不進 train）  |  "
            "跨群組丟棄 %d  attractor 超標剔除 %d",
            graph.dataset_id, len(warm["train"]), len(warm["val"]), len(warm["test"]),
            len(cold["val"]), len(cold["test"]), len(cold["train"]),
            n_dropped, n_attractor_cut,
        )

        # far_edge／rand_edge 只改 train 側的 IC 來源；val/test 側維持真實近鄰邊，
        # 各臂才在同一把尺上被評比。
        far_map = build_src_override_map(graph, ablation_arm)

        if warm["train"]:
            parts["train"].append(EdgeMeanDataset(
                zp, graph, edge_ids=warm["train"], source_stats_idx=idx,
                vflip_aug_prob=vflip_aug_prob, src_override=far_map,
            ))
        for split, key in (("val", "val_warm"), ("test", "test_warm")):
            if warm[split]:
                parts[key].append(EdgeMeanDataset(
                    zp, graph, edge_ids=warm[split], source_stats_idx=idx,
                    vflip_aug_prob=0.0,
                ))
        # cold_start edge **不組成 val/test dataset**。
        #   mean module 的任務是 parent_mean(IC) → child_mean(target)；cold_start
        #   edge 依定義沒有 parent（src_node=None），batch 因此缺 parent_mean_renorm，
        #   `_validate_split_mean` 每次都只能印一行「批次皆缺 pair 欄位」再回空 dict，
        #   而 log 上仍出現一個看似存在、實則永遠空白的 `VAL cold:` 欄位。
        #   一個結構上不可能有指標的 split，不該以「有這個 loader」的形式存在。
        if cold["val"] or cold["test"]:
            _log.info(
                "%s: cold_start edge val %d / test %d **不組 dataset**"
                "（無 parent_mean → mean module 無從評估；非過濾失誤）",
                graph.dataset_id, len(cold["val"]), len(cold["test"]),
            )

    # ★unified（A 方案）：一次呼叫只組一個角色，val/test 角色本來就沒有 train 集。
    #   仍要擋「role=train 卻組不出東西」——那是真的錯誤。
    if not parts["train"] and role in (None, "train"):
        raise RuntimeError(
            "split_graph_datasets：train 集為空"
            + ("（role=train）" if role == "train" else "")
        )

    def _concat(items: list[Dataset]) -> Optional[Dataset]:
        if not items:
            return None
        return items[0] if len(items) == 1 else ConcatDataset(items)

    return GraphSplitDatasets(
        train=_concat(parts["train"]),
        val_cold=_concat(parts["val_cold"]),
        val_warm=_concat(parts["val_warm"]),
        test_cold=_concat(parts["test_cold"]),
        test_warm=_concat(parts["test_warm"]),
    )


def build_chain_train_dataset(
    zarr_paths: list[str],
    chain_len: int = 2,
    max_attractor_ratio: Optional[float] = 3.0,
    ablation_arm: str = "use_edge",
) -> Dataset:
    """組多 zarr 的 multi-edge pushforward 訓練集（缺陷 b lever）。

    只在 **train split 的 warm edge** 上枚舉連續 `chain_len` 邊的鏈 -> 鏈的每段
    皆 train-warm，與 val/test 無洩漏、逐 stage 皆有 GT mean_flow 監督。

    Args:
        zarr_paths: v4 主 zarr 路徑列表。
        chain_len: 鏈的 edge 數（stage 數）。
        max_attractor_ratio: short edge attractor 比值上限。
        ablation_arm: 三臂消融臂別。

    Returns:
        chain 訓練 Dataset（單 zarr 直回，多 zarr ConcatDataset）。

    Raises:
        RuntimeError: 枚舉不出任何鏈，或 graph.json 的 `split` 欄未填妥。
    """
    parts: list[Dataset] = []
    for idx, zp in enumerate(zarr_paths):
        # ★不可寫死 from_json(_graph_json_for(...))：unified 單檔沒有獨立 graph.json，
        #   topology 在 zarr 的 .zattrs。判別與載入一律走 for_zarr（同 split_graph_datasets）。
        graph = GraphIndex.for_zarr(zp)
        assert_split_labelled(graph)
        warm_train = edges_of_split(
            graph, "train", ic_source="parent_mean",
            max_attractor_ratio=max_attractor_ratio,
        )
        chains = enumerate_chains(graph, warm_train, chain_len)
        _log.info(
            "%s: chain-pushforward 枚舉 %d 條 len=%d 鏈（自 %d train warm edge）",
            graph.dataset_id, len(chains), chain_len, len(warm_train),
        )
        if chains:
            parts.append(EdgeChainMeanDataset(
                zp, graph, chains=chains, chain_len=chain_len, source_stats_idx=idx,
                src_override=build_src_override_map(graph, ablation_arm),
            ))
    if not parts:
        raise RuntimeError(
            f"build_chain_train_dataset：枚舉不出任何 len={chain_len} 鏈"
            "（train warm edge 拓撲無連續鏈？降 chain_len 或檢查 graph）"
        )
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)
