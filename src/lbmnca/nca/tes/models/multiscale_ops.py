"""向後相容轉發層（Backward-compatible re-export）。

此模組已重組為 multiscale/ 子套件。
所有符號從 multiscale.ops 重新匯出，既有 import 路徑維持不變：

    from .multiscale_ops import downsample_state  # 仍可正常運作

新程式碼請改用：

    from .multiscale import downsample_state
"""
from __future__ import annotations

from .multiscale.ops import (  # noqa: F401  (re-export)
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
