"""尺度特徵編碼器（H-series 多尺度條件調製）。

Phase Code-1 使用 concat 路徑（``build_scale_features`` → concat 至
``cond_norm`` → 送入既有的 TESConditionEncoder）。
:class:`ScaleConditionEncoder` 為後續階段保留，提供加法式 FiLM 調製頭替代選項。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


SCALE_FEAT_DIM: int = 4


def build_scale_features(
    level_id: float,
    ratio: int,
    dt_scale: float,
    ctu_scale: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """建構單一尺度層級的條件特徵向量，長度為 ``SCALE_FEAT_DIM``。

    Args:
        level_id: 尺度層級編號，粗格網為 0.0、細格網為 1.0，依此類推。
        ratio: 當前層級的空間降採樣倍率（細格網為 1）。
        dt_scale: 當前層級的 Δt 縮放係數（基底為 1.0，substep < 1.0）。
        ctu_scale: 累積 CTU 縮放標記（保留供未來擴充使用）。
        batch_size: 目標批次大小，用於 batch 維度廣播。
        device: 目標張量所在裝置。

    Returns:
        尺度特徵張量，形狀 [B, SCALE_FEAT_DIM]，沿批次維度廣播。
    """
    log_ratio = math.log2(float(max(ratio, 1)))
    feats = torch.tensor(
        [float(level_id), log_ratio, float(dt_scale), float(ctu_scale)],
        device=device,
        dtype=torch.float32,
    )
    return feats.unsqueeze(0).expand(batch_size, -1)


class ScaleConditionEncoder(nn.Module):
    """尺度特徵的加法式 FiLM 調製頭（選用）。

    Phase Code-1 尚未啟用（採用 concat 路徑通過 TESConditionEncoder），
    保留供後續階段需要 additive FiLM 調製時使用。

    zero-init 確保訓練初期輸出接近恆等映射（gamma≈0, beta≈0），
    避免早期 FiLM 調製干擾主幹收斂。

    Args:
        n_features: 輸入特徵維度，預設為 SCALE_FEAT_DIM (4)。
        hidden_dim: MLP 隱藏層維度，預設 32。
        out_dim: 輸出 gamma/beta 的維度，預設 64。
    """

    def __init__(
        self,
        n_features: int = SCALE_FEAT_DIM,
        hidden_dim: int = 32,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.SiLU(),
        )
        self.to_gamma = nn.Linear(hidden_dim, out_dim)
        self.to_beta = nn.Linear(hidden_dim, out_dim)
        # zero-init 使訓練起始 FiLM 效果為恆等，避免初期 loss 爆炸
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(
        self,
        scale_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向傳播：從尺度特徵產生 FiLM 調製參數。

        Args:
            scale_feats: 尺度條件特徵張量，形狀 [B, n_features]。

        Returns:
            gamma: FiLM 縮放參數，形狀 [B, out_dim]。
            beta: FiLM 位移參數，形狀 [B, out_dim]。
        """
        h = self.mlp(scale_feats)
        return self.to_gamma(h), self.to_beta(h)


__all__ = ["SCALE_FEAT_DIM", "build_scale_features", "ScaleConditionEncoder"]
