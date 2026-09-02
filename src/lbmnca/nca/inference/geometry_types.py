"""GeometryInputs — model-agnostic 幾何 + 條件資料結構。

正規化由 backend 內部處理；呼叫端傳 raw 值。
cond_raw 為 superset dict，各 backend 取所需鍵。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GeometryInputs:
    """Model-agnostic raw geometry，由呼叫端傳給 backend。

    所有 tensor 為 CUDA float32。backend 負責映射至模型所需輸入（
    cond vectors、正規化、state seeding）。
    """

    mask: torch.Tensor          # [H, W] 0=solid, 1=fluid
    static_raw: torch.Tensor    # [4, H, W]（SDF px, upstream, X_norm, Y_norm）raw
    cond_raw: dict[str, Any]    # superset 條件參數（g_vec list[14], p_vec list[3],
                                # L_char, u_inlet, Re, wind_level, BCR, ...）
    H: int
    W: int


__all__ = ["GeometryInputs"]
