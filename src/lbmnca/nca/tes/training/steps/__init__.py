"""training/steps — TESTrainer 訓練步驟子模組。

每個子元件接受 TrainerContext，封裝一類訓練步邏輯，回傳 dict[str, float] metrics。

只有 mean 一條路徑：``MeanStep``。
"""
from .base_step import BaseStep
from .mean_step import MeanStep

__all__ = ["BaseStep", "MeanStep"]
