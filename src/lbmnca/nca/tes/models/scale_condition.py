"""向後相容性重匯出 — 實作已移至 models/conditioning/scale_encoder.py。

此模組保留原始路徑，確保已存在的 import 語句不需修改仍可正常運作。
所有新程式碼請改用 ``from .conditioning.scale_encoder import ...``。
"""
from __future__ import annotations

from .conditioning.scale_encoder import (
    SCALE_FEAT_DIM,
    ScaleConditionEncoder,
    build_scale_features,
)

__all__ = ["SCALE_FEAT_DIM", "build_scale_features", "ScaleConditionEncoder"]
