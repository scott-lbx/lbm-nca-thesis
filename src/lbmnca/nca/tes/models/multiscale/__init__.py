"""多尺度操作子套件。

從 ops 模組重新匯出所有公開函式，使外部程式可直接 import：

    from lbmnca.nca.tes.models.multiscale import downsample_state, restrict_fine_to_coarse
"""
from __future__ import annotations

from .ops import (
    downsample_mask,
    downsample_state,
    expand_block_halo,
    make_block_mask,
    prolong_coarse_to_fine,
    restrict_fine_to_coarse,
    upsample_state,
)

__all__ = [
    "downsample_state",
    "upsample_state",
    "downsample_mask",
    "make_block_mask",
    "expand_block_halo",
    "restrict_fine_to_coarse",
    "prolong_coarse_to_fine",
]
