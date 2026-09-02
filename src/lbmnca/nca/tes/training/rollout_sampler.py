"""rollout_sampler.py — Adaptive Rollout Length Sampler。

常態分布隨訓練進度偏移的 rollout 長度抽樣器。
訓練初期偏向短 rollout（穩定梯度），後期偏向長 rollout（穩定性壓力）。
K(t) ~ N(μ(t), σ(t)²)，clamp 至 [K_min, K_max]。μ(t) 由 sigmoid/linear/exp 從 K_min 滑向 K_max。

Usage:
    sampler = RolloutSampler.build(K_min=1, K_max=12, total_steps=10000,
                                   mu_schedule='sigmoid', sigma_schedule='decay')
    K = sampler.get_k(step)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .rng_streams import stream


@dataclass
class RolloutSampler:
    """常態分布偏移的 rollout 長度抽樣器（μ 隨進度滑移、σ 控制隨機範圍）。"""

    K_min: int = 2
    K_max: int = 64
    total_steps: int = 100_000

    mu_schedule: Literal["linear", "sigmoid", "exponential"] = "sigmoid"
    sigmoid_midpoint: float = 0.4
    sigmoid_sharpness: float = 8.0
    exp_gamma: float = 3.0

    sigma_schedule: Literal["fixed_ratio", "decay", "bell"] = "decay"
    sigma_max: float = None  # type: ignore[assignment]  # 自動 = K_max/4
    sigma_min: float = 2.0
    sigma_ratio: float = 0.3

    _history: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.sigma_max is None:
            self.sigma_max = self.K_max / 4.0

    @classmethod
    def build(
        cls, K_min: int, K_max: int, total_steps: int, **kwargs
    ) -> "RolloutSampler":
        return cls(K_min=K_min, K_max=K_max, total_steps=total_steps, **kwargs)

    def get_k(self, current_step: int) -> int:
        alpha = min(current_step / max(self.total_steps, 1), 1.0)
        mu = self._compute_mu(alpha)
        sigma = self._compute_sigma(alpha)
        # rollout 長度抽樣走具名串流（isolation 關閉時＝全域 random，行為不變）
        K_raw = stream("rollout").gauss(mu, sigma)
        K = int(round(max(self.K_min, min(self.K_max, K_raw))))
        self._history.append(K)
        return K

    def get_k_batch(self, current_step: int, batch_size: int) -> list[int]:
        return [self.get_k(current_step) for _ in range(batch_size)]

    def _compute_mu(self, alpha: float) -> float:
        return self.K_min + (self.K_max - self.K_min) * self._g(alpha)

    def _g(self, alpha: float) -> float:
        if self.mu_schedule == "linear":
            return alpha
        if self.mu_schedule == "sigmoid":
            raw = 1.0 / (
                1.0
                + math.exp(-self.sigmoid_sharpness * (alpha - self.sigmoid_midpoint))
            )
            g0 = 1.0 / (1.0 + math.exp(self.sigmoid_sharpness * self.sigmoid_midpoint))
            g1 = 1.0 / (
                1.0 + math.exp(-self.sigmoid_sharpness * (1.0 - self.sigmoid_midpoint))
            )
            return (raw - g0) / (g1 - g0 + 1e-8)
        if self.mu_schedule == "exponential":
            return (math.exp(self.exp_gamma * alpha) - 1.0) / (
                math.exp(self.exp_gamma) - 1.0
            )
        raise ValueError(f"Unknown mu_schedule: {self.mu_schedule}")

    def _compute_sigma(self, alpha: float) -> float:
        if self.sigma_schedule == "fixed_ratio":
            return max(self.sigma_min, self.sigma_ratio * self._compute_mu(alpha))
        if self.sigma_schedule == "decay":
            return self.sigma_max * (1.0 - alpha) + self.sigma_min * alpha
        if self.sigma_schedule == "bell":
            return max(self.sigma_min, self.sigma_max * 4.0 * alpha * (1.0 - alpha))
        raise ValueError(f"Unknown sigma_schedule: {self.sigma_schedule}")

    def get_schedule_at(self, current_step: int) -> dict:
        alpha = min(current_step / max(self.total_steps, 1), 1.0)
        return {
            "alpha": round(alpha, 4),
            "mu": round(self._compute_mu(alpha), 2),
            "sigma": round(self._compute_sigma(alpha), 2),
            "K_min": self.K_min,
            "K_max": self.K_max,
        }


__all__ = ["RolloutSampler"]
