"""transport_variants — H3T 系列與 OperatorGated transport 架構集合。

本子目錄收錄所有以「STREAM → COLLIDE」operator splitting 為核心的 NCA 變體：

  - H3TWarpResidualNCA：並行 warp + residual，sigmoid gate 混合（HD-4a）。
  - H3TWarpResidualSerialNCA：串行消融版，先 STREAM 再 COLLIDE（LBM 對照）。
  - H3TStreamCollideNCA：純串行 streamcollide，支援 KE rescale + mean_film。
  - OperatorGatedStreamCollideNCA：當前主力（C-series），flag 驅動算子組合。
"""
from __future__ import annotations

from .h3t_warpresidual import H3TWarpResidualNCA, H3TWarpResidualSerialNCA
from .h3t_streamcollide import H3TStreamCollideNCA
from .operator_gated import OperatorGatedStreamCollideNCA

__all__ = [
    "H3TWarpResidualNCA",
    "H3TWarpResidualSerialNCA",
    "H3TStreamCollideNCA",
    "OperatorGatedStreamCollideNCA",
]
