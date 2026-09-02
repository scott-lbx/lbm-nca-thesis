"""向後相容性重匯出 — 實作已移至 models/update/markov_cell.py。

此模組保留原始路徑，確保已存在的 import 語句不需修改仍可正常運作。
所有新程式碼請改用 ``from .update.markov_cell import MarkovNCACell``。
"""
from __future__ import annotations

from .update.markov_cell import MarkovNCACell

__all__ = ["MarkovNCACell"]
