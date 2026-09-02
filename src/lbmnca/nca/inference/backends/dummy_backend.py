"""DummyBackend — 零依賴偽物理場產生器（無 checkpoint 時的測試替代品）。

"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from lbmnca.nca.inference.geometry_types import GeometryInputs


class _BuiltinDummyNCA:
    """內建偽物理場產生器。

    jx 趨向入流速 + 平滑擴散 + 小幅週期擺動（開發/展示視覺用，無物理意義）。
    """

    def __init__(self, hidden_channels: int, device: str) -> None:
        self.static_channels = 4
        self.dyn_ch = 9 + 1 + hidden_channels   # 9 phy + 1 convergence + hidden
        self.device = torch.device(device)
        self._t = 0

    def init_state(self, static: torch.Tensor) -> torch.Tensor:
        b, _, h, w = static.shape
        dyn = torch.zeros(b, self.dyn_ch, h, w, device=static.device)
        dyn[:, 9] = 1.0
        self._t = 0
        return torch.cat([static, dyn], dim=1)

    def encode_once(self, state: torch.Tensor, cond_raw: dict):
        mask = (state[:, 0:1] > 0).float()
        u = float(cond_raw.get("u_inlet") or 0.05)
        return {"u": u}, mask

    def apply_ic_mask(self, x: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, self.static_channels:] = out[:, self.static_channels:] * (~solid).float()
        return out

    def step_one(self, x: torch.Tensor, cc: dict, mask: torch.Tensor) -> torch.Tensor:
        sc = self.static_channels
        self._t += 1
        phy = x[:, sc : sc + 9].clone()
        jx = phy[:, 3:4]
        jx = jx + 0.15 * (cc["u"] - jx)
        blur = F.avg_pool2d(F.pad(jx, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        jx = (0.7 * jx + 0.3 * blur) * mask
        phy[:, 0:1] = mask
        phy[:, 3:4] = jx
        phy[:, 5:6] = 0.3 * math.sin(self._t * 0.3) * jx
        out = x.clone()
        out[:, sc : sc + 9] = phy
        out[:, sc + 9 : sc + 10] = x[:, sc + 9 : sc + 10] * 0.93
        return out


class DummyBackend:
    """InferenceBackend backed by built-in dummy NCA（零 checkpoint 依賴）。"""

    def __init__(
        self,
        H: int,
        W: int,
        device: str = "cuda",
        hidden_channels: int = 16,
    ) -> None:
        self.H = H
        self.W = W
        self.phy_channels = 9
        self.has_convergence = True
        self.device = torch.device(device)

        self.model = _BuiltinDummyNCA(hidden_channels, device)
        self._sc = self.model.static_channels
        self._x: torch.Tensor | None = None
        self._cc = None
        self._mask: torch.Tensor | None = None
        self._steady: torch.Tensor | None = None

    def prepare(self, geo: GeometryInputs, warm: bool) -> None:
        static = geo.static_raw.unsqueeze(0).to(self.device)
        self._cc, self._mask = self.model.encode_once(
            self.model.init_state(static), geo.cond_raw
        )
        if warm and self._steady is not None:
            solid = (self._mask < 0.5)
            x = self.model.apply_ic_mask(self._steady.clone(), solid)
            x[:, : self._sc] = static
            self._x = x
        else:
            self._x = self.model.init_state(static)

    def step(self, n: int = 1) -> None:
        assert self._x is not None, "call prepare() first"
        for _ in range(max(1, n)):
            self._x = self.model.step_one(self._x, self._cc, self._mask)

    def phy_field(self) -> torch.Tensor:
        assert self._x is not None
        return self._x[0, self._sc : self._sc + 9].contiguous()

    def convergence_field(self) -> torch.Tensor | None:
        assert self._x is not None
        return self._x[0, self._sc + 9 : self._sc + 10].contiguous()

    def is_diverged(self) -> bool:
        if self._x is None:
            return False
        phy = self._x[0, self._sc : self._sc + 9]
        return bool(torch.isnan(phy).any() or phy.abs().max() > 1e4)

    def dynamic_dim(self) -> int:
        return self.model.dyn_ch

    def save_steady(self) -> None:
        if self._x is not None:
            self._steady = self._x.clone()


__all__ = ["DummyBackend"]
