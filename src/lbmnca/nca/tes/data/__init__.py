"""TES-NCA 資料層：v4 unified graph dataset、條件向量正規化、runtime stats 合併。

本層只認 `tes_unified_v1`（topology 在 zarr 的 `.zattrs`），沒有格式分支、
沒有向後相容 shim —— 讀不到 topology 就是 raise，不退回任何舊路徑。
"""

from __future__ import annotations

from .graph_dataset import (
    EdgeChainMeanDataset,
    EdgeMeanDataset,
    collate_chain_batch,
    collate_graph_batch,
)
from .graph_index import GraphIndex, is_v4_zarr
from .graph_split import build_chain_train_dataset, split_graph_datasets
from .loader import build_tes_dataloader
from .normalizer import TESCondNormalizer
from .runtime_global_stats import build_runtime_global_stats, resolve_stats_paths

__all__ = [
    "EdgeChainMeanDataset",
    "EdgeMeanDataset",
    "GraphIndex",
    "TESCondNormalizer",
    "build_chain_train_dataset",
    "build_runtime_global_stats",
    "build_tes_dataloader",
    "collate_chain_batch",
    "collate_graph_batch",
    "is_v4_zarr",
    "resolve_stats_paths",
    "split_graph_datasets",
]
