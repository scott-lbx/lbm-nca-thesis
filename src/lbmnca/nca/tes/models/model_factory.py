"""向後相容性重匯出 — 實作已移至 models/factory.py。

此模組保留原始路徑，確保已存在的 import 語句（訓練腳本、checkpoint 載入邏輯等）
不需修改仍可正常運作。所有新程式碼請改用 ``from .factory import build_tes_model``。
"""
from __future__ import annotations

from .factory import _build_arch, build_tes_model

__all__ = ["build_tes_model"]
