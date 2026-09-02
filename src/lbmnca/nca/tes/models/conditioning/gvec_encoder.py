"""幾何先驗向量（G_vec 14D）→ 雙路徑 FiLM 參數，zero-init 確保初始等同 baseline。

GVecEncoder 每次 rollout 計算一次（G_vec 為 per-case 靜態常數，不隨時間步變化）。
組合方式由 TESNCA 負責：γ = γ_p + γ_g，β = β_p + β_g。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GVecEncoder(nn.Module):
    """幾何先驗向量 G_vec → (γ_g, β_g) FiLM 調製參數（dual-pathway 幾何分支）。

    每次 rollout 只計算一次（G_vec 為 per-case 靜態常數，不隨時間步變化）。
    γ_g/β_g 以 zero-init 開始，確保熱啟動時不干擾既有 p_vec 路徑的 FiLM 輸出。
    組合方式：γ = γ_p + γ_g，β = β_p + β_g（加法 FiLM，TESNCA 負責相加）。

    Args:
        g_vec_dim: G_vec 輸入維度，預設 14。
        hidden_dim: MLP 隱藏層維度，預設 128。
        out_dim: γ_g/β_g 各自的維度，等於 nca_channels，預設 64。
    """

    def __init__(
        self,
        g_vec_dim: int = 14,
        hidden_dim: int = 128,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(g_vec_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.to_gamma = nn.Linear(hidden_dim, out_dim)
        self.to_beta = nn.Linear(hidden_dim, out_dim)

        # zero-init：熱啟動時不干擾既有 p_vec 路徑的 FiLM 輸出
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(
        self,
        g_vec_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """計算幾何先驗 FiLM 調製參數。

        Args:
            g_vec_norm: [B, g_vec_dim] 已正規化 G_vec。

        Returns:
            gamma_g: [B, out_dim] 幾何縮放參數。
            beta_g:  [B, out_dim] 幾何平移參數。
        """
        h = self.mlp(g_vec_norm)
        return self.to_gamma(h), self.to_beta(h)


__all__ = ["GVecEncoder"]
