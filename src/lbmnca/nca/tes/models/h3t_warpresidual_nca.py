"""向後相容性重匯出 — 實作已移至 models/assembly/transport_variants/h3t_warpresidual.py。"""
from __future__ import annotations

from .assembly.transport_variants.h3t_warpresidual import (
    H3TWarpResidualNCA,
    H3TWarpResidualSerialNCA,
)

__all__ = ["H3TWarpResidualNCA", "H3TWarpResidualSerialNCA"]
