"""ModelBackend — the universal model boundary contract for lbmnca.bridge.

03 talks ONLY to this Protocol. Concrete models (dummy / GC-NCA / TES /
precomputed field store) each provide an adapter implementing it. The render
layer is model-agnostic: every backend exposes a 9ch MRT phy field
(`[rho,e,eps,jx,qx,jy,qy,pxx,pxy]`) on GPU plus an optional convergence field.

See docs/MODEL_BOUNDARY_CONTRACT.md for the full spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class GeometryInputs:
    """Model-agnostic raw geometry stats handed from 03 to an adapter.

    All tensors are CUDA, float32. The adapter maps these to model-specific
    inputs (cond vectors, normalization, state seeding).
    """

    mask: torch.Tensor          # [H, W]  0=solid, 1=fluid
    static_raw: torch.Tensor    # [4, H, W]  SDF, upstream, X_norm, Y_norm (raw)
    cond_raw: dict[str, Any]    # superset regime params (see regime_features.py)
    H: int
    W: int


@runtime_checkable
class ModelBackend(Protocol):
    """Contract every model adapter implements. 03's only call surface."""

    H: int
    W: int
    phy_channels: int           # invariant: 9 (MRT moments)
    has_convergence: bool       # GC-NCA/dummy True; TES False

    def prepare(self, geo: GeometryInputs, warm: bool) -> None:
        """Event-driven setup on geometry/regime change.

        warm=False : cold start (flow from rest).
        warm=True  : continuation (reuse previous field + IC-mask new solids).
        """
        ...

    def step(self, n: int = 1) -> None:
        """Advance the field n steps. State stays on GPU."""
        ...

    def phy_field(self) -> torch.Tensor:
        """Current physics field [9, H, W] CUDA, denormalized physical units."""
        ...

    def convergence_field(self) -> torch.Tensor | None:
        """Per-cell convergence [1, H, W] CUDA (1=active, 0=settled), or None."""
        ...

    def is_diverged(self) -> bool:
        """True if the field went NaN / exploded (feeds ConvergenceGuard)."""
        ...

    def dynamic_dim(self) -> int:
        """phy + hidden channel count (used for warm-start IC masking)."""
        ...


__all__ = ["GeometryInputs", "ModelBackend"]
