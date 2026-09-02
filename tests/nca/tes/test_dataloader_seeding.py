"""DataLoader 播種回歸測試（R7-G7；CPU-only、不碰 zarr / GPU）。

鎖住 DEEP_AUDIT_2026-07 根因 R7 的三條性質，避免 Stage 0.5 的修正被無聲回退：

1. **同 seed 可重現**：``build_tes_dataloader(seed=S, num_workers>0)`` 兩次疊代
   的樣本序列逐位相同——樣本值由 worker 行程內的裸 ``random.random()`` 產生，
   與 vflip augmentation 判定／window 起點取樣所倚賴的是**同一個 worker 端 RNG
   機制**。★但本檔用的是 ``TinyRandomDataset``，並未實際疊代 zarr dataset，
   故只鎖住該機制本身，不涵蓋真實 dataset 的其他隨機來源。
2. **不受呼叫前 torch 隨機呼叫影響**：疊代前插入額外 ``torch.rand(1)`` 後序列
   不變（專屬 ``torch.Generator`` 已與主行程全域 default generator 解耦）。
   這正是稽核實測到的逃逸路徑：``base_seed`` 原本自全域 generator 抽取。
3. **守衛存在**：``num_workers>0`` 而未給 ``seed`` 直接 ``ValueError``（R7-G5）。

★``TinyRandomDataset`` 必須是**模組層**類別：Windows spawn 會在各 worker 重新
import 本模組，closure / 區域類別無法 pickle。

★純 CPU：所有 loader 皆以 ``pin_memory=False`` 建立（``build_tes_dataloader`` 的
預設仍是 True，訓練路徑不受影響），故疊代時**不會建立 CUDA context**，可在訓練
進行中安全執行（workspace 規則「高 VRAM 訓練期禁 GPU pytest」）。本檔不含任何
``.cuda()`` / device 搬移。

⚠️ Windows spawn 會為每個 worker 重新 import torch；實測全檔 6 個測試約 13 秒。
"""

from __future__ import annotations

import random

import pytest
import torch
from torch.utils.data import Dataset

from lbmnca.nca.tes.data.loader import build_tes_dataloader

_N_ITEMS: int = 12
_NUM_WORKERS: int = 2


class TinyRandomDataset(Dataset):
    """每個 item 回傳一個由行程內全域 ``random`` 抽出的浮點數。

    模擬 ``graph_dataset`` 的 ``__getitem__`` 中以裸
    ``random.*`` 決定 vflip 與 window 起點的行為：回傳值完全由該 worker
    行程的 Python RNG 狀態決定。

    Args:
        n: 樣本數。
    """

    def __init__(self, n: int = _N_ITEMS) -> None:
        self._n = n

    def __len__(self) -> int:
        """樣本數。

        Returns:
            資料集長度。
        """
        return self._n

    def __getitem__(self, idx: int) -> dict[str, object]:
        """回傳第 idx 筆（含 worker 端亂數）。

        Args:
            idx: 樣本索引。

        Returns:
            含 ``idx``（索引）與 ``r``（worker RNG 抽樣值）的 dict。
        """
        return {"idx": int(idx), "r": float(random.random())}


def _passthrough_collate(samples: list[dict]) -> list[dict]:
    """不做 tensor stack 的 collate（本測試只比對純量序列）。

    Args:
        samples: ``__getitem__`` 回傳的 dict 列表。

    Returns:
        原樣的 dict 列表。
    """
    return samples


def _drain(seed: int, shuffle: bool, num_workers: int) -> list[tuple[int, float]]:
    """建 loader、完整疊代一次，回傳 ``(idx, r)`` 序列。

    Args:
        seed: 傳給 ``build_tes_dataloader`` 的 seed。
        shuffle: 是否打亂。
        num_workers: worker 行程數。

    Returns:
        依疊代順序展平的 ``(idx, r)`` 列表。
    """
    loader = build_tes_dataloader(
        TinyRandomDataset(),
        batch_size=3,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_passthrough_collate,
        seed=seed,
        pin_memory=False,     # 純 CPU：避免疊代時建立 CUDA context
    )
    out: list[tuple[int, float]] = []
    for batch in loader:
        out.extend((int(s["idx"]), float(s["r"])) for s in batch)
    del loader  # persistent_workers：明確釋放 worker 行程
    return out


@pytest.mark.parametrize("shuffle", [False, True])
def test_same_seed_reproduces_worker_sequence(shuffle: bool) -> None:
    """(a) 同 seed 兩次疊代的 (順序, worker 亂數) 序列逐位相同。"""
    first = _drain(seed=1234, shuffle=shuffle, num_workers=_NUM_WORKERS)
    second = _drain(seed=1234, shuffle=shuffle, num_workers=_NUM_WORKERS)
    assert first == second
    assert len(first) == _N_ITEMS


def test_sequence_immune_to_prior_global_torch_draws() -> None:
    """(b) 疊代前插入額外 torch 隨機呼叫，序列不得改變（base_seed 已解耦）。"""
    torch.manual_seed(7)
    baseline = _drain(seed=99, shuffle=True, num_workers=_NUM_WORKERS)

    torch.manual_seed(7)
    torch.rand(1)          # ★逃逸路徑：舊實作會讓 base_seed 整條位移
    torch.rand(3)
    perturbed = _drain(seed=99, shuffle=True, num_workers=_NUM_WORKERS)

    assert baseline == perturbed


def test_different_seed_changes_worker_sequence() -> None:
    """seed 不同 → worker 亂數序列必須不同（否則 seed 根本沒接上）。"""
    a = _drain(seed=11, shuffle=False, num_workers=_NUM_WORKERS)
    b = _drain(seed=12, shuffle=False, num_workers=_NUM_WORKERS)
    assert [x[1] for x in a] != [x[1] for x in b]


def test_multiprocess_without_seed_is_rejected() -> None:
    """(c) num_workers>0 且 seed=None → 守衛 ValueError（R7-G5）。"""
    with pytest.raises(ValueError, match="num_workers>0"):
        build_tes_dataloader(
            TinyRandomDataset(),
            batch_size=3,
            shuffle=False,
            num_workers=_NUM_WORKERS,
            collate_fn=_passthrough_collate,
            seed=None,
        )


def test_single_process_without_seed_is_allowed() -> None:
    """num_workers=0 不走 worker base_seed 路徑，維持可不傳 seed（既有 20+ 呼叫點）。"""
    loader = build_tes_dataloader(
        TinyRandomDataset(),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=_passthrough_collate,
        seed=None,
        pin_memory=False,
    )
    assert sum(len(b) for b in loader) == _N_ITEMS
