"""dataset_zarr v4 的 Dataset 層（node / edge 分離）。

兩個 Dataset 對應 mean module 的兩種取樣：

    EdgeMeanDataset       單邊 warm transfer
        IC     = nodes/{src}/mean_flow
        target = nodes/{dst}/mean_flow
        cond   = nodes/{dst}/static + topology 的 g_vec / p_vec

    EdgeChainMeanDataset  多邊 pushforward（沿 lineage 連續 L 條 edge）

v4 的語義修正：IC 一律在同一 stats 空間（cold root 已 renorm 匯入為 external
node）→ `parent_mean_renorm` 不需跨 zarr 換算，直接讀 src node 的 mean_flow。

本檔只含 mean module 的 dataset 類別，不含 turb 取樣路徑。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from lbmnca.nca.tes.data.graph_index import GraphIndex

# short edge 的 attractor 自校準比值上限（tes_h5_to_zarr.ATTRACTOR_QA_RATIO_MAX）。
# 超過表示該 edge 未收斂到 canonical attractor，其 dst node 的 mean_flow 監督
# 訊號不可信 → 預設排除。
DEFAULT_MAX_ATTRACTOR_RATIO: float = 3.0


_STR_KEYS: tuple[str, ...] = ("case_key", "ic_type", "mask_id", "dataset_name")
_FIELD_KEYS: tuple[str, ...] = (
    "static",
    "mean_flow_post_conv",
    "parent_mean_renorm",
    # ★消融的損失空間權重一律以**覆寫前**的真實 parent 建立。
    #   漏掉這個鍵的話 collate 會把它丟掉，mean_step 的 batch.get 回 None 而
    #   退回用覆寫後的 ic_physics —— far/rand 兩臂的 ic_physics 在資料層就已被
    #   src_override 換掉，於是修正變成空操作。必須與 _maybe_vflip 成對維護。
    "true_parent_mean_renorm",
    "g_vec",
    "p_vec",
)
_SEQ_KEYS: tuple[str, ...] = ("trajectory", "ctu_values")


def collate_graph_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """v4 collate：只 stack 實際存在的鍵，序列以最短者截齊。

    與 v3 `_collate_variable_length` 的差別：v3 無條件索取 `trajectory` /
    `ctu_values` / `delta_ctu`，而 v4 的 mean batch（`ic_source=="parent_mean"`）
    根本不讀 transit，沒有這些鍵。

    同質性由 `EdgeMeanDataset` 在建構時保證（batch 內不會 parent_mean 與
    cold_start 混雜），故此處毋須再用 `all(...)` gate —— 那正是 v3 讓單一冷啟動
    case 汙染整個 batch 的機制。

    Args:
        samples: `__getitem__` 回傳的 dict 列表。

    Returns:
        batch dict；字串欄位保留為 list。
    """
    batch: dict[str, Any] = {
        k: [s[k] for s in samples] for k in _STR_KEYS if k in samples[0]
    }
    for k in _FIELD_KEYS:
        if k in samples[0]:
            batch[k] = torch.stack([s[k] for s in samples])

    if "trajectory" in samples[0]:
        min_t = min(s["trajectory"].shape[0] for s in samples)
        batch["trajectory"] = torch.stack([s["trajectory"][:min_t] for s in samples])
        if "ctu_values" in samples[0]:
            batch["ctu_values"] = torch.stack(
                [s["ctu_values"][:min_t] for s in samples]
            )
            batch["delta_ctu"] = torch.stack(
                [s["delta_ctu"][: min_t - 1] for s in samples]
            )

    batch["t0"] = torch.tensor(
        [s.get("t0", 0) for s in samples], dtype=torch.long
    )  # optional — 僅序列 dataset 有
    batch["source_stats_idx"] = torch.tensor(
        [s["source_stats_idx"] for s in samples], dtype=torch.long
    )
    return batch


class _V4Base(Dataset):
    """v4 zarr 共用的延遲開啟邏輯（DataLoader worker fork 安全）+ vflip aug。"""

    # y-odd moments：垂直翻轉後變號（jy=5, qy=6, pxy=8）
    _VFLIP_NEG_CH: tuple[int, ...] = (5, 6, 8)

    def __init__(
        self,
        zarr_path: str | Path,
        graph: GraphIndex,
        source_stats_idx: int,
        vflip_aug_prob: float = 0.0,
    ) -> None:
        """記錄路徑，不在此開 zarr（避免 fork 後 handle 失效）。

        Args:
            zarr_path: v4 主 zarr 路徑。
            graph: 對應的 GraphIndex。
            source_stats_idx: 多 zarr 混訓時對應全域 stats list 的索引。
            vflip_aug_prob: 垂直翻轉 augmentation 機率（僅 train，val 傳 0）。
        """
        self._zarr_path = str(zarr_path)
        self.graph = graph
        self._source_stats_idx = source_stats_idx
        self._vflip_aug_prob = float(vflip_aug_prob)
        self._root: Optional[zarr.Group] = None

    def _maybe_vflip(self, out: dict[str, Any]) -> dict[str, Any]:
        """依機率套用垂直翻轉 augmentation。

        CFD 左進右出 → 僅 y 鏡像物理合法（左右翻逆流、旋轉改入口）。
        `static[X, Y]` 為固定 grid coord、`g_vec` reflection-invariant、`p_vec`
        皆不變。`trajectory` 在 EdgeMeanDataset 的 parent_mean 分支不存在 → 跳過。

        Args:
            out: `__getitem__` 組好的 sample dict（就地修改）。

        Returns:
            同一個 dict。
        """
        if self._vflip_aug_prob <= 0.0 or random.random() >= self._vflip_aug_prob:
            return out

        st = out["static"].clone()
        st[0:2] = torch.flip(
            st[0:2], dims=[1]
        )  # ch0 SDF / ch1 upstream 翻 H；ch2/3 coord 不動
        out["static"] = st

        def _flip_field(f: torch.Tensor, h_dim: int, ch_dim: int) -> torch.Tensor:
            f = torch.flip(f, dims=[h_dim]).clone()
            idx: list[Any] = [slice(None)] * f.ndim
            for c in self._VFLIP_NEG_CH:
                idx[ch_dim] = c
                f[tuple(idx)] = -f[tuple(idx)]
            return f

        if "trajectory" in out:  # [T, 9, H, W]
            out["trajectory"] = _flip_field(out["trajectory"], h_dim=2, ch_dim=1)
        # ★true_parent_mean_renorm 必須與 parent 同步翻轉：它是損失空間權重的來源，
        #   只翻其中一個會讓權重高值落在流域的錯誤半邊（vflip_aug_prob 0.5，半數樣本受影響）。
        for k in ("mean_flow_post_conv", "parent_mean_renorm",
                  "true_parent_mean_renorm"):  # [9, H, W]
            if k in out:
                out[k] = _flip_field(out[k], h_dim=1, ch_dim=0)
        return out

    def _ensure_open(self) -> zarr.Group:
        """首次存取時開啟 zarr root。"""
        if self._root is None:
            self._root = zarr.open(self._zarr_path, mode="r")
        return self._root

    def _node_cond(self, node_id: str) -> dict[str, Any]:
        """讀 node 的 static / g_vec / p_vec（cond 三件組）。"""
        root = self._ensure_open()
        return {
            "static": torch.from_numpy(np.asarray(root["nodes"][node_id]["static"][:])),
            "g_vec": torch.tensor(self.graph.g_vec(node_id), dtype=torch.float32),
            "p_vec": torch.tensor(self.graph.p_vec(node_id), dtype=torch.float32),
        }

    def _mean_flow(self, node_id: str) -> torch.Tensor:
        """讀 node 的 post-conv 時均場 [9, H, W]。"""
        root = self._ensure_open()
        return torch.from_numpy(np.asarray(root["nodes"][node_id]["mean_flow"][:]))


class EdgeMeanDataset(_V4Base):
    """mean module（warm transfer）：src node mean → dst node mean。

    每個 item 是一條 edge。short edge 與 full edge 都可用 —— 兩者的 dst node
    共用同一份 mean_flow（attractor 只依賴 node），這正是 v4 分離的直接後果。

    **同質化**：預設只收 `ic_source == "parent_mean"` 的 edge。

    v3 的 `_collate_variable_length` 用 `all("parent_mean_renorm" in s ...)` 決定
    是否 stack 該欄 —— 只要 batch 內混進一個沒有 parent 的 case，**整個 batch**
    都會掉到 `mean_step` 的冷啟動分支，連同批有 parent 的 warm case 一起被降級成
    cold IC 監督。把 cold_start edge 排除在 dataset 之外可從源頭杜絕此汙染。

    需要冷啟動 edge 時明確傳 `ic_source="cold_start"` 另建一個 dataset，
    不要與 parent_mean 混進同一個 sampler。

    Args:
        zarr_path: v4 主 zarr 路徑。
        graph: GraphIndex。
        edge_ids: 要用的 edge_id 列表（None = 依條件全取）。
        split: 篩選 split（edge_ids 為 None 時生效）。
        ic_source: `"parent_mean"`（預設）或 `"cold_start"`。
        max_attractor_ratio: short edge 的 attractor 比值上限，None = 不篩。
        source_stats_idx: 多 zarr 混訓的 stats 索引。
    """

    def __init__(
        self,
        zarr_path: str | Path,
        graph: GraphIndex,
        edge_ids: Optional[list[str]] = None,
        split: Optional[str] = None,
        ic_source: str = "parent_mean",
        max_attractor_ratio: Optional[float] = DEFAULT_MAX_ATTRACTOR_RATIO,
        source_stats_idx: int = 0,
        vflip_aug_prob: float = 0.0,
        src_override: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(zarr_path, graph, source_stats_idx, vflip_aug_prob)
        if ic_source not in ("parent_mean", "cold_start"):
            raise ValueError(
                f"ic_source 只能是 parent_mean / cold_start，得到 {ic_source}"
            )
        self._ic_source = ic_source
        # 三臂消融 far_edge 用：原 src node → 改取之遠配對 node（見 graph_split.build_far_src_map）。
        # None＝不覆寫（use_edge／no_edge 兩臂）。dst 與監督目標一律不動。
        self._src_override = src_override

        if edge_ids is None:
            edge_ids = graph.edges_by(
                split=split,
                ic_source=ic_source,
                max_attractor_ratio=max_attractor_ratio,
            )
            n_excluded = len(graph.edges) - len(edge_ids)
            if n_excluded:
                print(
                    f"[EdgeMeanDataset] {graph.dataset_id}: 取 {len(edge_ids)} 條 edge，"
                    f"排除 {n_excluded} 條（ic_source!={ic_source} / split / attractor 超標）"
                )
        else:
            mixed = {graph.ic_source_of(e) for e in edge_ids}
            if len(mixed) > 1:
                raise ValueError(
                    f"edge_ids 混合了 ic_source={mixed}；同一 dataset 必須同質，"
                    "否則 collate 會讓整個 batch 掉到冷啟動分支"
                )
        if not edge_ids:
            raise ValueError(
                f"EdgeMeanDataset 篩選後為空（split={split}, ic_source={ic_source}）"
            )
        self._edge_ids = list(edge_ids)

    def __len__(self) -> int:
        """edge 數。"""
        return len(self._edge_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """組 sample 後統一在出口套 vflip aug（子類覆寫 `_build`，不覆寫本方法）。"""
        return self._maybe_vflip(self._build(idx))

    def _build(self, idx: int) -> dict[str, Any]:
        """回傳 mean_step 相容的 batch dict（未套 aug）。

        `ic_source == "parent_mean"` 時給 `parent_mean_renorm`（IC = src node 的
        mean_flow，同 stats 空間，無需 renorm）；`"cold_start"` 時改給
        `trajectory`（transit 首幀），對應 `mean_step` 既有的冷啟動分支。

        Returns:
            dict：
                case_key            str（edge_id）
                ic_type             str（dst node 的 ic_type）
                mask_id             str
                dataset_name        str
                static              Tensor [4, H, W]     float32（dst node）
                mean_flow_post_conv Tensor [9, H, W]     float32（dst node，target）
                parent_mean_renorm  Tensor [9, H, W]     float32（僅 parent_mean）
                trajectory          Tensor [1, 9, H, W]  float16（僅 cold_start）
                g_vec               Tensor [14]          float32（raw）
                p_vec               Tensor [3]           float32（raw）
                source_stats_idx    int
        """
        eid = self._edge_ids[idx]
        dst = self.graph.dst_of(eid)
        dst_node = self.graph.nodes[dst]

        out: dict[str, Any] = {
            "case_key": eid,
            "ic_type": dst_node["ic_type"],
            "mask_id": dst_node["mask_id"],
            "dataset_name": self.graph.dataset_id,
            "mean_flow_post_conv": self._mean_flow(dst),
            "source_stats_idx": self._source_stats_idx,
        }

        src = self.graph.src_of(eid)
        true_src = src
        if src is not None and self._src_override is not None:
            # far_edge 臂：IC 改取遠配對節點之時均場；dst、監督目標與條件向量皆不動，
            # 故本臂與 use_edge 的唯一差異即「編輯距離」。
            src = self._src_override[src]
        if src is not None:
            out["parent_mean_renorm"] = self._mean_flow(src)
            # ★消融可比性：損失的空間權重（Δ-focus 與 wake）必須跨臂一致，
            # 否則各臂最佳化的是不同的目標函數，效應量無從歸因。故一律另外提供
            # **覆寫前**的真實 parent 供建權重用；送進模型的 IC 仍是覆寫後的場。
            out["true_parent_mean_renorm"] = self._mean_flow(true_src)
        elif getattr(self.graph, "is_mean_only", False):
            # ★ mean-only zarr 不存任何幀。`cold_start` edge 依定義沒有 parent，
            #   而 `graph_split.py` 對它的處置是**不進 train、也不組 val/test**
            #   （mean module 的任務是 parent_mean(IC) → child_mean(target)）。
            #   走到這裡代表呼叫端手動建了一個訓練路徑不會建的 dataset。
            raise RuntimeError(
                f"{eid}: mean-only zarr 沒有冷啟動 IC 幀。"
                "`cold_start` edge 在 mean module 下不被消費（見 graph_split.py），"
                "故轉檔時未保存任何快照。要用冷啟動 IC 需先決定它的正規化尺度 ——"
                "瞬時快照的振幅是時均場的 4–5 倍，凍結尺度表達不了。")
        else:
            root = self._ensure_open()
            traj_key = self.graph.traj_key_of(eid)
            first = np.asarray(
                root["edges"][traj_key]["snapshots"][0:1]
            )  # [1, 9, H, W]
            out["trajectory"] = torch.from_numpy(first)

        out.update(self._node_cond(dst))
        return out


class NodePairDataset(_V4Base):
    """任意 **(IC node, target node)** 配對 —— 不受圖上是否有邊限制。

    ★為什麼需要這個：`EdgeMeanDataset` 只吃圖上既有的 warm edge，於是 `DT_Stress`
    只剩 **9** 個可評樣本。但實際使用情境是「每個 case 都可以是前一個的 IC、也可以
    是下一個的 target」——使用者在前端連續編輯，狀態一路傳遞下去，
    圖上有沒有那條邊跟能不能這樣用**無關**。
    本類直接由 node 組配對，把 DT 的 59 個 node 全部納入評測。

    幾何／條件一律取自 **target node**（要求算子在 target 的幾何下把 IC 演化過去）；
    IC 只提供起始場。

    Args:
        zarr_path: v4 unified zarr 路徑。
        graph: GraphIndex。
        pairs: ``(ic_node_id, target_node_id)`` 列表。
        source_stats_idx: 多 zarr 混訓的 stats 索引。
    """

    def __init__(
        self,
        zarr_path: str | Path,
        graph: GraphIndex,
        pairs: list[tuple[str, str]],
        source_stats_idx: int = 0,
    ) -> None:
        super().__init__(zarr_path, graph, source_stats_idx, vflip_aug_prob=0.0)
        if not pairs:
            raise ValueError("pairs 為空")
        missing = {n for p in pairs for n in p if n not in graph.nodes}
        if missing:
            raise KeyError(f"pairs 含不存在的 node：{sorted(missing)[:5]}")
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """回傳與 `EdgeMeanDataset` 同鍵的 sample（`collate_graph_batch` 相容）。"""
        return self._build(idx)

    def _build(self, idx: int) -> dict[str, Any]:
        """組 (ic_node → tgt_node) 的 mean_step 相容 sample。"""
        ic_node, tgt = self._pairs[idx]
        tgt_node = self.graph.nodes[tgt]
        out: dict[str, Any] = {
            "case_key": f"{ic_node}->{tgt}",
            "ic_type": tgt_node["ic_type"],
            "mask_id": tgt_node["mask_id"],
            "dataset_name": self.graph.dataset_id,
            "mean_flow_post_conv": self._mean_flow(tgt),
            "parent_mean_renorm": self._mean_flow(ic_node),
            "source_stats_idx": self._source_stats_idx,
        }
        out.update(self._node_cond(tgt))
        return out


_CHAIN_STACK_KEYS: tuple[str, ...] = (
    "parent_mean_renorm",
    "chain_static",
    "chain_target",
    "chain_g_vec",
    "chain_p_vec",
)


def collate_chain_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """multi-edge chain collate：stack 各 chain 張量（固定 chain_len，直接 stack）。

    產生 batch dict（供 ``MeanStep.run_chain`` 消費）：
        parent_mean_renorm [B, 9, H, W]      stage1 IC（N0 mean_flow）
        chain_static       [B, L, 4, H, W]   逐 stage dst node static
        chain_target       [B, L, 9, H, W]   逐 stage dst node mean_flow（target）
        chain_g_vec        [B, L, 14]
        chain_p_vec        [B, L, 3]
        source_stats_idx   [B]

    Args:
        samples: ``EdgeChainMeanDataset.__getitem__`` 回傳的 dict 列表。

    Returns:
        chain batch dict。
    """
    batch: dict[str, Any] = {
        "case_key": [s["case_key"] for s in samples],
        "dataset_name": [s["dataset_name"] for s in samples],
    }
    for k in _CHAIN_STACK_KEYS:
        batch[k] = torch.stack([s[k] for s in samples])
    batch["source_stats_idx"] = torch.tensor(
        [s["source_stats_idx"] for s in samples], dtype=torch.long
    )
    return batch


def enumerate_chains(
    graph: GraphIndex,
    edge_ids: list[str],
    chain_len: int = 2,
) -> list[list[str]]:
    """從 train warm edge 集合枚舉長度 ``chain_len``（edge 數）的連續 warm 鏈。

    鏈 = ``[N0, N1, ..., N_L]``，每段 edge ``N_{i}→N_{i+1}`` 皆須在 ``edge_ids``
    內（同 train split + parent_mean）→ 防 val 洩漏、確保逐 stage 皆有 GT mean_flow。

    Args:
        graph: GraphIndex。
        edge_ids: train warm edge_id 集合。
        chain_len: 鏈的 edge 數（stage 數）；預設 2（=攻 hand-off 本體）。

    Returns:
        node-id 鏈列表，每條長度 ``chain_len + 1``。
    """
    edge_set = set(edge_ids)
    # 以 (src, dst) 建 node→在集合內的後繼 edge
    chains: list[list[str]] = []

    def _extend(path_nodes: list[str], depth: int) -> None:
        if depth == chain_len:
            chains.append(list(path_nodes))
            return
        tail = path_nodes[-1]
        for e in graph.out_edges(tail):
            if e in edge_set:
                nxt = graph.dst_of(e)
                if nxt not in path_nodes:  # 防環
                    _extend(path_nodes + [nxt], depth + 1)

    for e1 in edge_ids:
        n0 = graph.src_of(e1)
        if n0 is None:  # 需 parent 當 stage1 IC
            continue
        _extend([n0, graph.dst_of(e1)], 1)
    return chains


class EdgeChainMeanDataset(_V4Base):
    """multi-edge pushforward（缺陷 b）：連續 ``chain_len`` 個 warm edge 一條鏈。

    stage1 = N0 mean_flow 注入為 IC → 模型 rollout → 對 N1 mean_flow；stage j>1 =
    上一 stage 模型輸出 hand-off（``apply_ic_mask`` + static 置換，鏡射部署推論）→
    對 N_j mean_flow。逐 stage 皆錨 GT，直接訓練跨幾何 warm 累積行為（單邊 loss
    訓不到的缺陷 b）。

    Args:
        zarr_path: v4 主 zarr 路徑。
        graph: GraphIndex。
        chains: node-id 鏈列表（``enumerate_chains`` 產）。
        chain_len: edge 數（stage 數），與 chains 長度一致性由 caller 保證。
        source_stats_idx: 多 zarr 混訓的 stats 索引。
    """

    def __init__(
        self,
        zarr_path: str | Path,
        graph: GraphIndex,
        chains: list[list[str]],
        chain_len: int = 2,
        source_stats_idx: int = 0,
        src_override: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(zarr_path, graph, source_stats_idx, vflip_aug_prob=0.0)
        # far_edge 臂：僅覆寫 stage1 的 IC 來源節點；後續 stage 為模型自身 hand-off，不受影響。
        self._src_override = src_override
        if not chains:
            raise ValueError("EdgeChainMeanDataset：chains 為空")
        _bad = [c for c in chains if len(c) != chain_len + 1]
        if _bad:
            raise ValueError(
                f"EdgeChainMeanDataset：{len(_bad)} 條鏈長度 != chain_len+1={chain_len + 1}"
            )
        self._chains = list(chains)
        self._chain_len = chain_len

    def __len__(self) -> int:
        return len(self._chains)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        chain = self._chains[idx]
        n0, stage_nodes = chain[0], chain[1:]
        if self._src_override is not None:
            n0 = self._src_override[n0]
        statics, targets, gvecs, pvecs = [], [], [], []
        for nid in stage_nodes:
            cond = self._node_cond(nid)
            statics.append(cond["static"])
            gvecs.append(cond["g_vec"])
            pvecs.append(cond["p_vec"])
            targets.append(self._mean_flow(nid))
        return {
            "case_key": "→".join(chain),
            "dataset_name": self.graph.dataset_id,
            "parent_mean_renorm": self._mean_flow(n0),  # [9,H,W] stage1 IC
            "chain_static": torch.stack(statics),  # [L,4,H,W]
            "chain_target": torch.stack(targets),  # [L,9,H,W]
            "chain_g_vec": torch.stack(gvecs),  # [L,14]
            "chain_p_vec": torch.stack(pvecs),  # [L,3]
            "source_stats_idx": self._source_stats_idx,
        }

