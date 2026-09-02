"""dataset_zarr v4 graph 索引層。

只讀 `{ds}_v4_graph.json` 一個檔案，建立 O(1) 的 node/edge 查表與鄰接表。
不碰 zarr、不碰 raw 資料目錄。

v4 語義：
    edge = transit   （src_node 收斂態 → dst_node 幾何，直到 KE 收斂）
    node = post-conv （attractor，與抵達路徑無關）
    warm IC = nodes/{graph.src_of(edge_id)}/mean_flow
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def is_v4_zarr(zarr_path: str | Path) -> bool:
    """該 zarr 是否為 v4（node/edge 分離）格式。

    ★單一判準：呼叫端一律用本函式判別 v3/v4，不得自行以「``_graph.json``
    存不存在」推論。舊佈局（``{ds}_v4.zarr`` ＋ 獨立 ``{ds}_v4_graph.json``）
    與 unified 單檔（``{ds}_unified.zarr``，topology 在 ``.zattrs``）是**同一個
    格式的兩種佈局**；只認前者會讓 D 系列被誤判成 v3，落進 ``cases/`` 分支後
    以 ``KeyError: 'cases'`` 收場——錯誤訊息完全指不到真因。

    判別順序與 :meth:`GraphIndex.for_zarr` 一致：先看有沒有同名 graph.json，
    再看 zarr 的 ``.zattrs`` 有沒有 ``topology``。

    Args:
        zarr_path: zarr 資料集路徑。

    Returns:
        True 表 v4（兩種佈局皆是）；False 表 v3 扁平 ``cases/`` 格式。

    Raises:
        FileNotFoundError: 路徑不存在——不回 False，否則「路徑打錯」會被
            當成「這是 v3」，錯誤在下游才炸且訊息無關。
    """
    p = Path(zarr_path)
    if not p.exists():
        raise FileNotFoundError(f"is_v4_zarr：zarr 路徑不存在：{p}")
    if (p.parent / f"{p.stem}_graph.json").exists():
        return True

    import zarr as _zarr

    return "topology" in _zarr.open(str(p), mode="r").attrs


class GraphIndex:
    """v4 graph.json 的純索引層。

    Attributes:
        dataset_id: dataset 名稱。
        stats_space: 本 graph 所屬的正規化 stats 空間 id。
        nodes: node_id → node record。
        edges: edge_id → edge record。
    """

    def __init__(self, graph: dict[str, Any]) -> None:
        """從 graph dict 建索引。

        Args:
            graph: `{ds}_v4_graph.json` 的內容。
        """
        if graph["version"] != "tes_v4":
            raise ValueError(f"預期 tes_v4，得到 {graph['version']}")

        self._raw = graph
        self.dataset_id: str = graph["dataset_id"]
        self.stats_space: str = graph["stats_space"]
        self.grid: dict[str, int] = graph["grid"]

        # ★格式分三代，數值意義**不同**，不可互換：
        #
        #   tes_unified_v1        postconv 併進同一個 zarr；正規化常數由資料量測而來
        #   tes_unified_mean_v1   mean-only：無 postconv、無過渡段 snapshots；
        #                         正規化常數為 `norm_frozen` 的凍結值
        #
        # 舊版的 `S_combined` 是掃資料算出來的（且掃描時未套流固遮罩），與凍結常數
        # 相差 4–40 倍。混用不會報錯，只會靜默產生不可比的結果 —— 故在此擋掉。
        self.fmt: str = graph.get("format", "")
        self.is_mean_only: bool = self.fmt == "tes_unified_mean_v1"
        self.is_unified: bool = self.fmt in ("tes_unified_v1", "tes_unified_mean_v1")
        if not self.is_unified:
            raise ValueError(
                f"未知的 zarr format {self.fmt!r}。"
                "預期 'tes_unified_mean_v1'（現行）或 'tes_unified_v1'（舊版）。")
        # mean-only 沒有 postconv；舊格式才有跨檔指標。
        self.postconv_zarr: Optional[str] = None

        self.nodes: dict[str, dict[str, Any]] = {n["node_id"]: n for n in graph["nodes"]}
        self.edges: dict[str, dict[str, Any]] = {e["edge_id"]: e for e in graph["edges"]}

        # 陣列順序即 index（spec §3.2 不變量 1）
        self._node_idx: dict[str, int] = {n["node_id"]: i for i, n in enumerate(graph["nodes"])}
        self._edge_idx: dict[str, int] = {e["edge_id"]: i for i, e in enumerate(graph["edges"])}

        # 鄰接表：不落盤，載入時建（不變量 2）
        self._in: dict[str, list[str]] = defaultdict(list)
        self._out: dict[str, list[str]] = defaultdict(list)
        for e in graph["edges"]:
            self._in[e["dst_node"]].append(e["edge_id"])
            if e["src_node"] is not None:  # cold_start edge 無 src
                self._out[e["src_node"]].append(e["edge_id"])

    @classmethod
    def from_json(cls, path: str | Path) -> "GraphIndex":
        """從 graph.json 檔案載入。

        Args:
            path: `{ds}_v4_graph.json` 路徑。

        Returns:
            GraphIndex 實例。
        """
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def for_zarr(cls, zarr_path: str | Path) -> "GraphIndex":
        """從 v4 zarr 路徑推導其 graph.json 並載入。

        例：`dataset_zarr/foo_v4.zarr` → `dataset_zarr/foo_v4_graph.json`。

        Args:
            zarr_path: v4 主 zarr 路徑。

        Returns:
            GraphIndex 實例。
        """
        p = Path(zarr_path)
        # ★unified 單檔格式（tes_unified_v1）：topology 存在 zarr 的 .zattrs，
        #   不再有獨立的 graph.json。先試單檔，退回舊佈局。
        if p.name.endswith("_unified.zarr"):
            return cls.from_unified_zarr(p)
        legacy = p.parent / f"{p.stem}_graph.json"
        if legacy.exists():
            return cls.from_json(legacy)
        return cls.from_unified_zarr(p)

    @classmethod
    def from_unified_zarr(cls, zarr_path: str | Path) -> "GraphIndex":
        """從 unified 單檔 zarr 的 ``.zattrs["topology"]`` 載入。

        單檔格式（``tes_unified_v1``）把 v4 主體、postconv 與 graph.json 併成一個
        ``{DATASET}_unified.zarr``——三者本來就是同一個 dataset 的三個面向，分開只是
        轉檔歷史的殘留，且會讓 ``postconv_zarr`` 這種跨檔指標存在。

        ★該格式**不含 `split` 欄**（A 方案）：實測六本 D 系列各自的 split 是 100%
        單一值，該欄只是把「這本是什麼」重複寫了 N 遍。角色改由 config 的
        ``zarr_paths`` 決定，由 :func:`graph_split.split_graph_datasets` 的
        ``role`` 參數注入。

        Args:
            zarr_path: unified zarr 路徑。

        Returns:
            GraphIndex 實例。

        Raises:
            KeyError: zarr 的 ``.zattrs`` 無 ``topology``（不是 unified 格式）。
        """
        import zarr as _zarr

        root = _zarr.open(str(zarr_path), mode="r")
        try:
            topo = root.attrs["topology"]
        except KeyError as exc:
            raise KeyError(
                f"{zarr_path} 的 .zattrs 無 'topology' —— 它不是 tes_unified_v1 格式，"
                " 也找不到同名的 _graph.json。"
            ) from exc
        return cls(topo)

    # ── id ↔ idx ─────────────────────────────────────────────────────────────

    def node_idx(self, node_id: str) -> int:
        """node_id → 陣列索引。"""
        return self._node_idx[node_id]

    def edge_idx(self, edge_id: str) -> int:
        """edge_id → 陣列索引。"""
        return self._edge_idx[edge_id]

    # ── 拓撲 ─────────────────────────────────────────────────────────────────

    def src_of(self, edge_id: str) -> Optional[str]:
        """edge → IC 來源 node_id；`None` 表冷啟動（無 parent mean）。"""
        return self.edges[edge_id]["src_node"]

    def ic_source_of(self, edge_id: str) -> str:
        """edge 的 IC 型態：`"parent_mean"` 或 `"cold_start"`。

        `"cold_start"` 涵蓋兩種 edge（spec §3.2.1）：chain dataset 的 cold root，
        以及 parent 未轉檔的孤兒 case。兩者都沒有 parent mean 可當 IC。
        """
        return self.edges[edge_id]["ic_source"]

    def dst_of(self, edge_id: str) -> str:
        """edge → 目標 node_id。"""
        return self.edges[edge_id]["dst_node"]

    def canonical_edge_of(self, node_id: str) -> Optional[str]:
        """node → 定義其 post-conv 的 full-window edge（external node 為 None）。"""
        return self.nodes[node_id]["canonical_edge_id"]

    def in_edges(self, node_id: str, window: Optional[str] = None) -> list[str]:
        """指向該 node 的所有 edge。

        Args:
            node_id: 目標 node。
            window: 篩 `"full"` / `"short"`，None = 不篩。

        Returns:
            edge_id 列表。
        """
        eids = self._in.get(node_id, [])
        if window is None:
            return list(eids)
        return [e for e in eids if self.edges[e]["window_type"] == window]

    def out_edges(self, node_id: str, window: Optional[str] = None) -> list[str]:
        """從該 node 出發的所有 edge。"""
        eids = self._out.get(node_id, [])
        if window is None:
            return list(eids)
        return [e for e in eids if self.edges[e]["window_type"] == window]

    # ── 張量定位 ──────────────────────────────────────────────────────────────

    def traj_key_of(self, edge_id: str) -> str:
        """edge → 實際落盤 trajectory 的 edge_id（處理 traj_ref alias）。

        v3 的 case_id 撞名使 5 對 edge 共用同一條 transit（同 src→dst，solver
        確定性）。alias edge 的 `traj_ref` 指向 owner，zarr 只存一份。

        Args:
            edge_id: 任一 edge。

        Returns:
            zarr `edges/` 下實際存在的 group key。
        """
        ref = self.edges[edge_id]["traj_ref"]
        return edge_id if ref is None else ref

    # ── 條件向量（graph.json 內，非 zarr）────────────────────────────────────

    def g_vec(self, node_id: str) -> list[float]:
        """node 的 14D 幾何條件向量（raw）。"""
        return self.nodes[node_id]["g_vec"]

    def p_vec(self, node_id: str) -> list[float]:
        """node 的 3D 物理參數向量 raw `[Re, U_inlet_lu, L_char_px]`。"""
        return self.nodes[node_id]["p_vec"]

    # ── 篩選 ─────────────────────────────────────────────────────────────────

    def edges_by(
        self,
        *,
        window: Optional[str] = None,
        split: Optional[str] = None,
        edge_type: Optional[str] = None,
        ic_source: Optional[str] = None,
        max_attractor_ratio: Optional[float] = None,
    ) -> list[str]:
        """依條件篩 edge。

        Args:
            window: `"full"` / `"short"`。
            split: `"train"` / `"val"` / `"test"`。
            edge_type: `"chain"` / `"re"` / `"knn"` / `"clique"` / `"*_backward"`。
            ic_source: `"parent_mean"` / `"cold_start"`。
            max_attractor_ratio: short edge 的 attractor 自校準比值上限；
                超過表示該 edge 未收斂到 canonical attractor，其 dst node 的
                mean_flow 監督訊號不可信。full edge 不受此篩選影響。

        Returns:
            排序後的 edge_id 列表。
        """
        out: list[str] = []
        for eid, e in self.edges.items():
            if window is not None and e["window_type"] != window:
                continue
            if split is not None and e["split"] != split:
                continue
            if edge_type is not None and e["edge_type"] != edge_type:
                continue
            if ic_source is not None and e["ic_source"] != ic_source:
                continue
            if max_attractor_ratio is not None and e["attractor"] is not None:
                if e["attractor"]["ratio"] > max_attractor_ratio:
                    continue
            out.append(eid)
        return sorted(out)

    def nodes_by(
        self,
        *,
        split: Optional[str] = None,
        depth: Optional[int] = None,
        external: Optional[bool] = None,
    ) -> list[str]:
        """依條件篩 node。

        Args:
            split: `"train"` / `"val"` / `"test"`。
            depth: `tree_depth` 精確值。
            external: True = 只取 cold root（跨 dataset 匯入）。

        Returns:
            排序後的 node_id 列表。
        """
        out: list[str] = []
        for nid, n in self.nodes.items():
            if split is not None and n["split"] != split:
                continue
            if depth is not None and n["tree_depth"] != depth:
                continue
            if external is not None and n["external"] != external:
                continue
            out.append(nid)
        return sorted(out)

    def __repr__(self) -> str:
        """摘要字串。"""
        n_ext = sum(1 for n in self.nodes.values() if n["external"])
        n_full = sum(1 for e in self.edges.values() if e["window_type"] == "full")
        return (
            f"GraphIndex({self.dataset_id}: {len(self.nodes)} nodes "
            f"[{n_ext} external], {len(self.edges)} edges [{n_full} full])"
        )
