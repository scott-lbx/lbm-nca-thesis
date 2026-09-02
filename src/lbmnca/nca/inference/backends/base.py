"""InferenceBackend Protocol — 內部合約，backends 實作，Engine 呼叫。"""
from __future__ import annotations

from typing import Protocol

import torch

from lbmnca.nca.inference.geometry_types import GeometryInputs


class InferenceBackend(Protocol):
    """Internal contract — all backends implement this."""

    H: int
    W: int
    phy_channels: int       # invariant: 9 (MRT moments)
    has_convergence: bool

    def prepare(self, geo: GeometryInputs, warm: bool) -> None:
        """幾何/條件改變時呼叫。warm=True：從前次收斂態出發。"""
        ...

    def step(self, n: int = 1) -> None:
        """推進 n 步，狀態留 GPU。"""
        ...

    def phy_field(self) -> torch.Tensor:
        """[9, H, W] CUDA float32，denorm 物理單位（MRT moments）。"""
        ...

    def convergence_field(self) -> torch.Tensor | None:
        """[1, H, W] MEAN_CONVERGING 期有效；其他 None。"""
        ...

    def is_diverged(self) -> bool:
        """NaN / 爆炸偵測。"""
        ...

    def dynamic_dim(self) -> int:
        """phy + hidden 通道數，用於 warm-start IC masking。"""
        ...


__all__ = ["InferenceBackend"]
