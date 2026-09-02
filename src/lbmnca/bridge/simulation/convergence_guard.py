"""ConvergenceGuard — runtime safety net + degradation trigger.

Monitors the active backend each tick (scalar checks throttled to every N
frames to preserve the zero-copy hot loop). Two jobs:
  A. report convergence_occupancy (for the convergence-front overlay + status)
  B. detect NaN / explosion / drift -> request tier demotion (never show NaN)

See PLAN_03 §3-4.
"""
from __future__ import annotations

import torch


class ConvergenceGuard:
    def __init__(self, check_every: int = 5, drift_tau: float = 6.0) -> None:
        self.check_every = max(1, check_every)
        self.drift_tau = drift_tau
        self._tick = 0
        self._prev_rms: float | None = None
        self.tripped = False
        self.occupancy = 0.0          # fraction of cells still active (conv>0.5)
        self.reason = ""

    def reset(self) -> None:
        self._prev_rms = None
        self.tripped = False
        self.occupancy = 0.0
        self.reason = ""

    def check(self, backend, phy: torch.Tensor, convergence: torch.Tensor | None) -> bool:
        """Return True if the field is safe; False means tripped (demote tier).

        phy: [9,H,W] CUDA. convergence: [1,H,W] CUDA or None.
        """
        self._tick += 1
        
        # fast model-side flag first (always check to prevent NaNs)
        if backend.is_diverged():
            self._trip("NaN/explosion (backend)")
            return False

        if self._tick % self.check_every != 0:
            return not self.tripped

        rms = float(phy.float().pow(2).mean().sqrt().item())
        if rms != rms or rms > 1e3:                      # NaN or blow-up
            self._trip(f"rms={rms:.1f}")
            return False
        if self._prev_rms is not None and rms - self._prev_rms > self.drift_tau:
            self._trip(f"drift {self._prev_rms:.2f}->{rms:.2f}")
            return False
        self._prev_rms = rms

        if convergence is not None:
            self.occupancy = float((convergence > 0.5).float().mean().item())
        return True

    def _trip(self, reason: str) -> None:
        self.tripped = True
        self.reason = reason
