"""物理條件向量（17D）→ FiLM (γ, β) 參數，identity 初始化保留熱啟動相容性。

TESConditionEncoder 接收已正規化的條件向量（p_vec z-score，或
G_vec+p_vec z-score 拼接），輸出 γ 與 β 供 MarkovNCACell FiLM 調製。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TESConditionEncoder(nn.Module):
    """物理條件向量 → (γ_p, β_p) FiLM 調製參數。

    輸入為已正規化的條件向量（p_vec z-score，或 G_vec+p_vec z-score 拼接），
    輸出 γ 與 β 用於 MarkovNCACell 的 FiLM 調製。
    恆等初始化（γ=1, β=0）確保熱啟動時不破壞原有模型行為。

    Args:
        cond_dim: 條件向量維度，預設 17（14D 幾何 + 3D 物理）。
        hidden_dim: MLP 隱藏層維度，預設 128。
        out_dim: γ/β 各自的維度，等於 MarkovNCACell.nca_channels，預設 64。
    """

    def __init__(
        self,
        cond_dim: int = 17,
        hidden_dim: int = 128,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.to_gamma = nn.Linear(hidden_dim, out_dim)
        self.to_beta = nn.Linear(hidden_dim, out_dim)

        # FiLM 恆等初始化：γ = 0·h + 1 = 1，β = 0（訓練開始即為 identity transform）
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(
        self,
        cond_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """計算 FiLM 調製參數。

        Args:
            cond_norm: [B, cond_dim] 已正規化條件向量。

        Returns:
            gamma: [B, out_dim] 縮放參數。
            beta:  [B, out_dim] 平移參數。
        """
        h = self.mlp(cond_norm)
        gamma = self.to_gamma(h)
        beta = self.to_beta(h)
        return gamma, beta


__all__ = ["TESConditionEncoder"]
