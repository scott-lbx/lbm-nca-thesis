"""
PatchUtils — 場域切割與重建工具
================================
純粹負責滑動視窗切割 (data_to_patches) 與 Hann 加權重建 (patches_to_full)。

不包含任何正規化邏輯。輸入幾個通道就輸出幾個通道。
支援任意通道數的 torch.Tensor 或 numpy.ndarray，輸出統一為 torch.Tensor。

Usage:
    from lbmnca.nca.utils.PatchUtils import data_to_patches, patches_to_full

    # 切割
    patches, locs = data_to_patches(field, patch_size=64, stride=32)
    # patches: [B, C, 64, 64]

    # 重建 (Hann window 加權融合，消除接縫)
    full = patches_to_full(patches, locs, full_hw=(H, W))
    # full: [C, H, W]
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch


# ── 切割 ─────────────────────────────────────────────────────────────────────


def data_to_patches(
    field: Union[torch.Tensor, np.ndarray],
    patch_size: int,
    stride: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    將 (C, H, W) 場域按滑動視窗切割為 patches。

    Args:
        field      : [C, H, W]  torch.Tensor 或 numpy.ndarray，任意通道數
        patch_size : patch 邊長 (正方形)
        stride     : 滑動步長

    Returns:
        patches   : [B, C, patch_size, patch_size] torch.Tensor, float32
        locations : List[(h_start, w_start)]  長度 == B
    """
    if isinstance(field, torch.Tensor):
        t = field.detach().cpu().float()
    else:
        t = torch.from_numpy(np.asarray(field, dtype=np.float32))

    C, H, W = t.shape

    patches: List[torch.Tensor] = []
    locations: List[Tuple[int, int]] = []

    for h_s in range(0, H - patch_size + 1, stride):
        for w_s in range(0, W - patch_size + 1, stride):
            patches.append(t[:, h_s : h_s + patch_size, w_s : w_s + patch_size])
            locations.append((h_s, w_s))

    return torch.stack(patches, dim=0), locations  # [B, C, P, P]


# ── 重建 ─────────────────────────────────────────────────────────────────────


def patches_to_full(
    patches: Union[torch.Tensor, np.ndarray],
    locations: List[Tuple[int, int]],
    full_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    將 patches 以 Hann window 加權融合，重建完整場域。

    重疊區域取加權平均，消除 patch 邊界接縫。

    Args:
        patches   : [B, C, patch_size, patch_size]  torch.Tensor 或 ndarray
        locations : List[(h_start, w_start)]         與 data_to_patches 回傳一致
        full_hw   : (H, W)                           完整場域的空間尺寸

    Returns:
        full_field : [C, H, W]  torch.Tensor, float32
    """
    if isinstance(patches, np.ndarray):
        patches = torch.from_numpy(patches.astype(np.float32))
    else:
        patches = patches.detach().cpu().float()

    B, C, patch_size, _ = patches.shape
    full_H, full_W = full_hw

    # Hann window [patch_size, patch_size]
    hann_1d = torch.hann_window(patch_size, periodic=False)   # [P]
    hann_2d = torch.outer(hann_1d, hann_1d)                   # [P, P]
    hann_2d = hann_2d.unsqueeze(0)                            # [1, P, P]

    canvas = torch.zeros(C, full_H, full_W, dtype=torch.float32)
    weight = torch.zeros(1, full_H, full_W, dtype=torch.float32)

    for i, (h_s, w_s) in enumerate(locations):
        h_e = h_s + patch_size
        w_e = w_s + patch_size
        canvas[:, h_s:h_e, w_s:w_e] += patches[i] * hann_2d
        weight[:,  h_s:h_e, w_s:w_e] += hann_2d

    weight.clamp_(min=1e-8)
    return canvas / weight   # [C, H, W]
