"""向後相容性重匯出 — 實作已移至 models/assembly/transport_variants/operator_gated.py。"""
from __future__ import annotations

from .assembly.transport_variants.operator_gated import (
    OperatorGatedStreamCollideNCA,
    _JX,
    _JY,
)

__all__ = ["OperatorGatedStreamCollideNCA", "_JX", "_JY"]
