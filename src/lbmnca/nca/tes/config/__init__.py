"""訓練設定套件。

  - ``_legacy``：``TESTrainConfig`` 與 ``load_tes_config``（訓練入口實際使用的載入器）
  - ``base_config``：``BaseArchConfig`` / ``DataConfig`` / ``OptimConfig``
  - ``mean_config``：``MeanTrainConfig`` + ``load_mean_config``

兩個載入器都在 ``yaml.safe_load`` 後立即呼叫 ``expand_config_placeholders``，
把 ``${ZARR_ROOT}`` 這類佔位符展開成本機絕對路徑（見 ``lbmnca.paths``）。
"""
from __future__ import annotations

# ── 主設定（訓練入口用）──
from ._legacy import (
    BaselineConfig,
    MRPConfig,
    TESCurriculumConfig,
    TESEarlyStopConfig,
    TESModelConfig,
    TESMultiScaleConfig,
    TESPoolConfig,
    TESRPConfig,
    TESSScheduleConfig,
    TESTFCConfig,
    TESTemporalConfig,
    TESTransportConfig,
    TESTrainConfig,
    expand_config_placeholders,
    load_tes_config,
)
from ._legacy import _check  # 內部 helper，部分新模組共用

# ── 拆分設定 ──
from .base_config import BaseArchConfig, DataConfig, OptimConfig
from .mean_config import MeanTrainConfig, load_mean_config

__all__ = [
    "MRPConfig",
    "TESEarlyStopConfig",
    "TESTFCConfig",
    "TESSScheduleConfig",
    "TESRPConfig",
    "TESTemporalConfig",
    "TESPoolConfig",
    "TESMultiScaleConfig",
    "TESTransportConfig",
    "BaselineConfig",
    "TESModelConfig",
    "TESCurriculumConfig",
    "TESTrainConfig",
    "expand_config_placeholders",
    "load_tes_config",
    "BaseArchConfig",
    "DataConfig",
    "OptimConfig",
    "MeanTrainConfig",
    "load_mean_config",
]
