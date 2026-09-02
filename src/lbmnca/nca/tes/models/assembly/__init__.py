"""NCA 模型組裝層：對外接口，組合各子模組。

此套件將各子模組的公開模型類別集中匯出，
呼叫端只需從此處匯入，不必了解底層目錄結構。
"""
from __future__ import annotations

from .flat_nca import TESNCA
from .h_base import HBaseNCA
from .h_variants import (
    HBlockRefineNCA,
    HLapResNCA,
    HMultirateNCA,
    HPODModalNCA,
    HParentLeafNCA,
    HSharedScaleNCA,
)
from .transport_variants import (
    H3TStreamCollideNCA,
    H3TWarpResidualNCA,
    H3TWarpResidualSerialNCA,
    OperatorGatedStreamCollideNCA,
)

__all__ = [
    # flat
    "TESNCA",
    # h-series base
    "HBaseNCA",
    # h-series variants
    "HSharedScaleNCA",
    "HLapResNCA",
    "HParentLeafNCA",
    "HMultirateNCA",
    "HBlockRefineNCA",
    "HPODModalNCA",
    # transport variants
    "H3TWarpResidualNCA",
    "H3TWarpResidualSerialNCA",
    "H3TStreamCollideNCA",
    "OperatorGatedStreamCollideNCA",
]
