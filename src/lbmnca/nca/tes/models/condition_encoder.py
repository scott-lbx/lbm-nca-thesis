"""向後相容性重匯出 — 實作已移至 models/conditioning/。

此模組保留原始路徑，確保已存在的 import 語句不需修改仍可正常運作。
所有新程式碼請改用 ``from .conditioning import TESConditionEncoder``。
"""
from __future__ import annotations

from .conditioning import GVecEncoder, MeanFlowEncoder, TESConditionEncoder

__all__ = ["TESConditionEncoder", "GVecEncoder", "MeanFlowEncoder"]
