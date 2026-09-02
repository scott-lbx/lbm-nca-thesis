"""TES 損失函式模組入口。

匯出 :class:`TESSemanticLoss`（語義複合損失路由）、``_LossContext``
（loss term 共享張量容器），以及 ``terms`` 子套件的所有 public loss 函式
（供進階消融與獨立測試直接 import）。
"""
from __future__ import annotations

from .semantic_loss import TESSemanticLoss
from .context import _LossContext
from .terms import *  # noqa: F401,F403（re-export terms 子套件全部 public loss 函式）

__all__ = ["TESSemanticLoss", "_LossContext"]
