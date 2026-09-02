"""TES-NCA 模型子套件：匯出所有公開模型類別與工廠函式。

匯出來源已遷移至組裝層子目錄（assembly / conditioning / perception /
update / baselines / transport）；舊的扁平模組路徑（tes_nca / markov_nca_cell /
condition_encoder / model_factory）仍保留為向後相容 shim，既有 import 不受影響。
"""
from __future__ import annotations

from .assembly import (
    H3TStreamCollideNCA,
    H3TWarpResidualNCA,
    H3TWarpResidualSerialNCA,
    HBaseNCA,
    HBlockRefineNCA,
    HLapResNCA,
    HMultirateNCA,
    HParentLeafNCA,
    HPODModalNCA,
    HSharedScaleNCA,
    OperatorGatedStreamCollideNCA,
    TESNCA,
)
from .baselines import BaselineStepperNCA, SteadyStateUNetNCA
from .conditioning import (
    GVecEncoder,
    MeanFlowEncoder,
    SCALE_FEAT_DIM,
    ScaleConditionEncoder,
    TESConditionEncoder,
    build_scale_features,
)
from .factory import build_tes_model
from .perception import TES_MRP
from .transport import KErescaler
from .update import MarkovNCACell

__all__ = [
    # 工廠
    "build_tes_model",
    # 組裝層：flat + h-series + transport variants
    "TESNCA",
    "HBaseNCA",
    "HSharedScaleNCA",
    "HLapResNCA",
    "HParentLeafNCA",
    "HMultirateNCA",
    "HBlockRefineNCA",
    "HPODModalNCA",
    "H3TWarpResidualNCA",
    "H3TWarpResidualSerialNCA",
    "H3TStreamCollideNCA",
    "OperatorGatedStreamCollideNCA",
    # baselines
    "BaselineStepperNCA",
    "SteadyStateUNetNCA",
    # conditioning
    "TESConditionEncoder",
    "GVecEncoder",
    "MeanFlowEncoder",
    "ScaleConditionEncoder",
    "build_scale_features",
    "SCALE_FEAT_DIM",
    # perception / update / transport
    "TES_MRP",
    "MarkovNCACell",
    "KErescaler",
]
