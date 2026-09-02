"""零模型基線與 skill score。

## 為什麼有這個模組

原始指標值（如 SSIM）不足以判斷模型的貢獻——如果任務本身很容易，一個什麼都不做的
零模型也能拿到高分。既有 canonical 評測路徑沒有任何 null 分支，無從回答「模型比
零模型好多少」這個問題。本模組提供零模型基線與 skill score，把模型的表現拆成
「任務本身多容易」與「模型額外貢獻多少」兩部分。

## 三個零模型

1. `null_do_nothing` —— 直接輸出 IC。暖啟動情境下 IC ＝ 父節點的收斂時均場，是部署端
   **本來就已經握在手上**的東西。模型若贏不過它，暖啟動推論就沒有產生任何資訊。
2. `null_persistence` —— 同樣輸出「最後可得的場」。在 warm split 上與 do-nothing 等價
   （IC 即最後可得幀），保留獨立入口是因為 cold split 的 IC 語義不同（transit 首幀），
   兩者不該共用一個名字而讓讀者以為量的是同一件事。
3. `null_geom_nn` —— 幾何最近鄰檢索：以 ``g_vec_norm`` 的 L2 距離在**訓練 split**
   內找最近的節點，直接搬它的時均場。這個 null 回答的是「這個任務能不能用查表解決」。

## 洩漏防護

`null_geom_nn` 的語料**必須**限定在無洩漏的 split，即 `graph.json` 每個 node 的
`split` 欄為 ``"train"``。本模組只讀該欄、遇到空字串或未知值一律 raise（對齊
`data/graph_split.py` 的裁決），不接受 `.get("split", ...)` 之類的預設——那正是讓
null 也被污染、進而虛報 skill 的寫法。

## skill score

``skill = (m_model − m_null) / (m_perfect − m_null)``

1.0 ＝ 完美、0.0 ＝ 與零模型同等、負值 ＝ 比零模型差。SSIM/IoU 類 ``m_perfect = 1.0``、
誤差類（MSE / rel-L2 / MAE）``m_perfect = 0.0``。分母 ``|m_perfect − m_null| < 1e-8``
時回 nan 並標 ``degenerate``——**禁止靜默 clamp**，因為那會把「零模型已經完美、此指標
無鑑別力」偽裝成「模型技能 1.0」。

★本模組是純後處理，不進入任何損失。import 條數（含 `build_geom_nn_corpus` 內那條
延遲 import 的 `zarr`）與「`lbmnca.nca.tes.loss.*` 不在其中」由
`tests/nca/tes/test_r4_countable_claims.py` 以 AST 機器複驗，數字不寫進本 docstring。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from lbmnca.nca.tes.data.graph_index import GraphIndex
from lbmnca.nca.tes.data.graph_split import VALID_SPLITS

__all__ = [
    "ERROR_LIKE_METRICS",
    "GeomNNCorpus",
    "SIMILARITY_LIKE_METRICS",
    "NullPrediction",
    "build_geom_nn_corpus",
    "null_do_nothing",
    "null_geom_nn",
    "null_persistence",
    "skill_score",
    "skill_table",
]

_EPS: float = 1e-8

SIMILARITY_LIKE_METRICS: frozenset[str] = frozenset(
    {
        "mean_jx_ssim",
        "vel_ssim",
        "vort_ssim",
        "vort_iou",
        "ke_converged",
    }
)
"""``m_perfect = 1.0`` 的指標（相似度/命中率類，越大越好）。"""

ERROR_LIKE_METRICS: frozenset[str] = frozenset(
    {
        "mean_mse",
        "mean_rel_l2",
        "wake_mae",
    }
)
"""``m_perfect = 0.0`` 的指標（誤差類，越小越好）。"""


# ── 零模型 ────────────────────────────────────────────────────────────────────


@dataclass
class NullPrediction:
    """零模型的一筆輸出。

    Attributes:
        name: 零模型名稱（``do_nothing`` / ``persistence`` / ``geom_nn``）。
        phys: 預測場 ``[1, 9, H, W]``（與 model 輸出同 norm 空間、同 grid）。
        meta: 該零模型的診斷資訊（如 geom_nn 命中的 node_id 與距離）。
    """

    name: str
    phys: torch.Tensor
    meta: dict[str, Any]


@torch.inference_mode()
def null_do_nothing(case: Any) -> NullPrediction:
    """零模型一：直接輸出 IC（部署端本已握有的父節點時均場）。

    Args:
        case: `_v4_infer.Case`（需有 ``ic`` 與 ``has_parent``）。

    Returns:
        `NullPrediction`，``phys`` 為 ``case.ic`` 的複本。
    """
    return NullPrediction(
        name="do_nothing",
        phys=case.ic.clone(),
        meta={"has_parent": bool(case.has_parent)},
    )


@torch.inference_mode()
def null_persistence(case: Any) -> NullPrediction:
    """零模型二：persistence（輸出最後可得的場）。

    warm split 上 IC 即最後可得幀 → 與 `null_do_nothing` 數值等價；cold split 上 IC
    是 transit 首幀，語義不同。刻意保留獨立入口，避免兩種語義共用一個名字。

    Args:
        case: `_v4_infer.Case`。

    Returns:
        `NullPrediction`，``phys`` 為 ``case.ic`` 的複本。
    """
    return NullPrediction(
        name="persistence",
        phys=case.ic.clone(),
        meta={
            "has_parent": bool(case.has_parent),
            "equivalent_to_do_nothing": bool(case.has_parent),
        },
    )


@dataclass
class GeomNNCorpus:
    """`null_geom_nn` 的檢索語料（僅訓練 split 節點）。

    Attributes:
        g_vec_norm: ``[N, G]`` 語料的正規化幾何條件向量（與 query 同一 normalizer）。
        node_ids: ``[N]`` 對應 node_id。
        dataset_ids: ``[N]`` 對應 dataset_id。
        mean_flow: ``[N, 9, H, W]`` 已 renorm 到 dataset-global 空間的時均場。
        split: 語料所屬 split（固定 ``"train"``，寫進 meta 供事後核對）。
    """

    g_vec_norm: torch.Tensor
    node_ids: list[str]
    dataset_ids: list[str]
    mean_flow: torch.Tensor
    split: str = "train"

    def __len__(self) -> int:
        """語料節點數。"""
        return len(self.node_ids)


def _node_split(graph: GraphIndex, node_id: str) -> str:
    """讀 node 的 `split` 欄；空字串或未知值一律 raise。

    Args:
        graph: 已載入的 GraphIndex。
        node_id: node_id。

    Returns:
        split 字串。

    Raises:
        KeyError: node 沒有 `split` 欄時（bracket 存取直接拋出）。
        ValueError: split 為空字串或不在合法集合內時拋出。
    """
    sp = graph.nodes[node_id]["split"]
    # 合法值直接取 `data/graph_split.VALID_SPLITS`（該檔是 split 欄的 SSOT），
    # 不在本檔另抄一份字面集合，免得日後兩邊漂移。
    if not sp or sp not in VALID_SPLITS:
        raise ValueError(
            f"node {node_id} 的 split 欄非法：{sp!r}。"
            "null_geom_nn 的語料必須來自 Stage 3.1-3.4 回填的無洩漏 split，"
            "不接受任何預設值。"
        )
    return str(sp)


@torch.inference_mode()
def build_geom_nn_corpus(
    inf: Any,
    zarr_paths: list[str],
    *,
    max_nodes_per_dataset: Optional[int] = None,
) -> GeomNNCorpus:
    """從 `graph.json` 的 ``split == "train"`` 節點建幾何最近鄰語料。

    每個節點以 `V4Infer.prep_raw` 走**完全相同**的 renorm / mask / cond 路徑組出
    `Case`，取其 ``ic``（即該節點自身的時均場）作為語料場——這樣語料與 query 一定在
    同一個 norm 空間，不會出現「null 在別的 stats 空間」的隱形不公平。

    ★檢索鍵：``TESCondNormalizer.normalize_g(raw g_vec)``，**不是**
    ``Case.g_vec_norm``——後者是模型 dual-pathway FiLM 的輸入，canonical config 為
    ``gvec_mode: none`` / ``g_vec_dim: 0`` 時恆 None。幾何相似度檢索與「模型是否
    消費獨立 g_vec 通路」無關，normalizer 對兩者都存在且同一份（退化維同樣歸零），
    故以它為鍵。

    Args:
        inf: `_v4_infer.V4Infer` 實例（提供 `prep_raw` 與 device）。
        zarr_paths: canonical config 的 ``zarr_paths``；``source_stats_idx`` 依此
            順序指派（對齊 `data/graph_split.split_graph_datasets`）。
        max_nodes_per_dataset: 每個 dataset 取樣上限（None = 全取）。用於在記憶體
            吃緊時降規模；取樣為**前 N 個（node_id 排序）**，確定性、不隨機。

    Returns:
        `GeomNNCorpus`。

    Raises:
        RuntimeError: 沒有任何 train 節點時拋出。
    """
    import zarr

    g_list: list[torch.Tensor] = []
    f_list: list[torch.Tensor] = []
    node_ids: list[str] = []
    dataset_ids: list[str] = []

    for idx, zp in enumerate(zarr_paths):
        zpath = Path(zp)
        # for_zarr 同時認舊佈局與 unified 單檔；寫死 graph.json 會漏掉後者。
        graph = GraphIndex.for_zarr(zpath)
        root = zarr.open(str(zpath), mode="r")
        train_nodes = sorted(
            nid for nid in graph.nodes if _node_split(graph, nid) == "train"
        )
        if max_nodes_per_dataset is not None:
            train_nodes = train_nodes[: int(max_nodes_per_dataset)]
        for nid in train_nodes:
            static = torch.from_numpy(
                np.asarray(root["nodes"][nid]["static"][:])
            ).unsqueeze(0)
            mean_flow = torch.from_numpy(
                np.asarray(root["nodes"][nid]["mean_flow"][:])
            ).unsqueeze(0)
            g_vec = torch.tensor([graph.g_vec(nid)], dtype=torch.float32)
            p_vec = torch.tensor([graph.p_vec(nid)], dtype=torch.float32)
            corpus_case = inf.prep_raw(
                static=static,
                ic=mean_flow,
                target=mean_flow,
                g_vec=g_vec,
                p_vec=p_vec,
                source_idx=idx,
                case_key=f"{graph.dataset_id}/{nid}",
            )
            # 檢索鍵與模型 gvec_mode 無關（見 docstring ★）；normalize_g 對
            # canonical（gvec_mode: none）同樣定義良好。
            key = inf._b._normalizer.normalize_g(g_vec.to(inf.device))
            g_list.append(key.reshape(1, -1).float().cpu())
            f_list.append(corpus_case.ic.float().cpu())
            node_ids.append(nid)
            dataset_ids.append(graph.dataset_id)

    if not node_ids:
        raise RuntimeError(
            "geom-NN 語料為空：zarr_paths 內沒有任何 split=='train' 的節點。"
        )
    return GeomNNCorpus(
        g_vec_norm=torch.cat(g_list, dim=0),
        node_ids=node_ids,
        dataset_ids=dataset_ids,
        mean_flow=torch.cat(f_list, dim=0),
    )


@torch.inference_mode()
def null_geom_nn(
    case: Any,
    corpus: GeomNNCorpus,
    *,
    query_g_norm: Optional[torch.Tensor] = None,
) -> NullPrediction:
    """零模型三：以 normalize_g 後的 g_vec L2 最近鄰，搬訓練 split 節點的時均場。

    Args:
        case: `_v4_infer.Case`（需有 ``fluid_mask`` 與 ``ic``）。
        corpus: `build_geom_nn_corpus` 的回傳（僅 train split）。
        query_g_norm: query 幾何鍵＝``TESCondNormalizer.normalize_g(raw g_vec)``。
            ★canonical（``gvec_mode: none``）下 ``Case.g_vec_norm`` 恆 None，
            呼叫端必須自備 raw g_vec（如自 graph.json 的 dst node）算出本鍵傳入；
            未傳且 ``case.g_vec_norm`` 也是 None 時直接 raise，不設 fallback。

    Returns:
        `NullPrediction`；``phys`` 已套上 **query 的** fluid mask（固體格歸零，
        與模型輸出的評測前處理一致），``meta`` 記錄命中 node 與 L2 距離。

    Raises:
        ValueError: 檢索鍵無處可取時拋出。
    """
    src = query_g_norm if query_g_norm is not None else case.g_vec_norm
    if src is None:
        raise ValueError(
            "null_geom_nn 無檢索鍵：query_g_norm 未傳且 case.g_vec_norm 為 None"
            "（canonical gvec_mode: none 即此情形）。請以 normalizer.normalize_g("
            "raw g_vec) 自備 query_g_norm。"
        )
    q = src.reshape(1, -1).float().cpu()
    d = torch.linalg.vector_norm(corpus.g_vec_norm - q, dim=1)
    j = int(torch.argmin(d).item())
    phys = corpus.mean_flow[j : j + 1].to(case.ic.device).clone()
    solid = case.fluid_mask <= 0.5
    phys = phys.masked_fill(solid.expand_as(phys), 0.0)
    return NullPrediction(
        name="geom_nn",
        phys=phys,
        meta={
            "nn_node_id": corpus.node_ids[j],
            "nn_dataset_id": corpus.dataset_ids[j],
            "nn_g_vec_l2": float(d[j].item()),
            "corpus_split": corpus.split,
            "corpus_size": len(corpus),
        },
    )


# ── skill score ──────────────────────────────────────────────────────────────


def skill_score(m_model: float, m_null: float, m_perfect: float) -> dict[str, Any]:
    """``(m_model − m_null) / (m_perfect − m_null)``。

    Args:
        m_model: 模型的指標值。
        m_null: 零模型的指標值。
        m_perfect: 完美值（相似度類 1.0、誤差類 0.0）。

    Returns:
        包含以下鍵的字典：
        - ``skill``: 技能分數（分母退化時為 nan）。
        - ``degenerate``: 分母 ``|m_perfect − m_null| < 1e-8`` 或任一輸入非有限時為 True。

    Note:
        分母退化時**不 clamp、不代 0**。零模型已經達到完美值意味著該指標對這個任務
        沒有鑑別力；靜默 clamp 會把這件事偽裝成模型技能。
    """
    vals = (m_model, m_null, m_perfect)
    if not all(np.isfinite(v) for v in vals):
        return {"skill": float("nan"), "degenerate": True}
    den = float(m_perfect) - float(m_null)
    if abs(den) < _EPS:
        return {"skill": float("nan"), "degenerate": True}
    return {"skill": (float(m_model) - float(m_null)) / den, "degenerate": False}


def skill_table(
    model_metrics: dict[str, float],
    null_metrics: dict[str, float],
    *,
    similarity_like: frozenset[str] = SIMILARITY_LIKE_METRICS,
    error_like: frozenset[str] = ERROR_LIKE_METRICS,
) -> dict[str, dict[str, Any]]:
    """一次產出 ``{metric: {raw, null, m_perfect, skill, degenerate}}`` 三欄表。

    只處理**兩邊都有、且已登記完美值方向**的 metric key。被跳過的 key 分兩類，
    **各有獨立的可稽核出口，不靜默丟棄**：

    - ``_missing_in_null``: model 端有、null 端沒有這個 key（兩邊 metric 集合不一致）。
    - ``_unclassified``: 兩邊都有，但方向未登記於 `SIMILARITY_LIKE_METRICS` /
      `ERROR_LIKE_METRICS`。不猜方向——猜錯會把技能算成反的。

    Args:
        model_metrics: 模型端 metric dict。
        null_metrics: 零模型端 metric dict（同 key 命名）。
        similarity_like: ``m_perfect = 1.0`` 的 key 集合。
        error_like: ``m_perfect = 0.0`` 的 key 集合。

    Returns:
        每個 metric 一筆 ``{raw, null, m_perfect, skill, degenerate}``；另含兩個
        底線開頭的稽核欄位 ``_missing_in_null`` 與 ``_unclassified``，形狀皆為
        ``{"keys": [...]}``（排序後的 key 清單，可能為空）。★底線開頭的鍵不是 metric，
        逐 metric 迭代回傳值時需排除。
    """
    out: dict[str, dict[str, Any]] = {}
    unclassified: list[str] = []
    missing_in_null: list[str] = []
    for k, v in model_metrics.items():
        if k not in null_metrics:
            missing_in_null.append(k)
            continue
        if k in similarity_like:
            perfect = 1.0
        elif k in error_like:
            perfect = 0.0
        else:
            unclassified.append(k)
            continue
        s = skill_score(float(v), float(null_metrics[k]), perfect)
        out[k] = {
            "raw": float(v),
            "null": float(null_metrics[k]),
            "m_perfect": perfect,
            "skill": s["skill"],
            "degenerate": s["degenerate"],
        }
    out["_missing_in_null"] = {"keys": sorted(missing_in_null)}
    out["_unclassified"] = {"keys": sorted(unclassified)}
    return out
