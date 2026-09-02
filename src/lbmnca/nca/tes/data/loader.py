"""DataLoader 建構與 worker 播種（格式無關）。

本模組只負責「把 Dataset 包成 DataLoader 並封住 RNG 逃逸」，不知道資料格式。
collate 由呼叫端指定（v4 graph 用 `graph_dataset.collate_graph_batch` /
`collate_chain_batch`）——**沒有預設 collate**：預設值曾是 v3 的
`_collate_variable_length`，忘了傳就會靜默走錯 collate、鍵集對不上。

v3（`cases/` 扁平格式）已歸檔，本專案只餘 v4 unified 一條路徑。
"""
from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker 的 RNG 播種。

    ★已實測校正（Stage 0.5）：目前釘選的 torch（``_worker_loop``）其實已無條件
    對每個 worker 重播 ``random`` / ``torch`` / ``numpy`` 三者——並非稽核原敘述
    「只重播 torch」。此函式對這三者而言是**冗餘但無害的顯式保險**（同值重設、
    防禦舊版 torch 不重播 numpy/random、也讓意圖在程式碼中自我說明）。

    真正的漏洞在 ``base_seed`` 本身：``DataLoader.__iter__`` 若未收到明確
    ``generator``，會向**主行程共用、可變動的全域 default torch generator**
    抽一個值當 ``base_seed``。抽樣時機在第一次疊代，此時全域 generator 已被
    ``seed_everything()`` 之後、迭代之前的任何 torch 隨機呼叫（典型如
    ``build_tes_model()`` 內大量 ``nn.init.*``）推進過 —— 只要那段路徑消耗的
    隨機數次數在兩次「同 config 同 seed」訓練間有任何差異，``base_seed`` 就會不同。

    本專案的 vflip augmentation 判定用裸 ``random.*``（``graph_dataset.__getitem__``），
    故 ``base_seed`` 漂移會直接逃逸 ``seed_everything()`` 的保證。封住它的是
    :func:`build_tes_dataloader` 的專屬 ``generator``，不是本函式。

    ★必須是模組頂層函式（可 pickle）：Windows spawn 不支援 lambda / closure。

    Args:
        worker_id: torch 傳入的 worker 序號（本函式不使用，由 ``torch.initial_seed()``
            取得已含 worker 偏移的種子）。
    """
    worker_seed: int = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_tes_dataloader(
    dataset: Dataset,
    collate_fn: Callable[[list[dict]], dict],
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: Optional[int] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """建立 DataLoader。

    Args:
        dataset: v4 graph dataset 實例（``EdgeMeanDataset`` / ``EdgeChainMeanDataset``
            或其 ``ConcatDataset``）。
        collate_fn: **必填**，無預設。v4 單邊用 ``collate_graph_batch``，
            多邊鏈用 ``collate_chain_batch``。
        batch_size: 每 batch 樣本數。
        shuffle: 訓練時 True，val 一律 False。
        num_workers: ≥1 啟用平行預取（lazy zarr open 已支援 spawn）。
        seed: 設定時建立專屬 ``torch.Generator`` 掛給 DataLoader（同時決定 shuffle
            順序與各 worker 的 ``base_seed``），使 vflip augmentation 可由此 seed
            重現。★``num_workers > 0`` 時**必填**（見 Raises）。
        pin_memory: 預設 True（真 non_blocking H2D 重疊）。純 CPU 測試設 False
            可避免疊代時建立 CUDA context。

    Returns:
        配置好的 DataLoader 實例。

    Raises:
        ValueError: ``num_workers > 0`` 但未給 ``seed``。多行程時 worker 的
            ``base_seed`` 由主行程全域 torch generator 抽取，會受呼叫前任意 torch
            隨機呼叫次數影響 → vflip 逃逸 seed 控制。單行程不走該路徑，故不強制。
    """
    if num_workers > 0 and seed is None:
        raise ValueError(
            "build_tes_dataloader：num_workers>0 時必須傳 seed。"
            "否則各 worker 的 base_seed 由主行程全域 torch generator 抽取，"
            "會隨呼叫前的 torch 隨機呼叫次數漂移，使 vflip augmentation "
            "逃逸 seed_everything() 的可重現性保證。"
            "純推論／評估腳本請傳 cfg.seed 或改用 num_workers=0。"
        )

    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
    )
