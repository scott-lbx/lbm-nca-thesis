"""平均流場空間調製器（LC decompose turb_module 用），每像素 FiLM，zero-init 保持基線。

MeanFlowEncoder 設計：turb input 維持 9ch fluct
不 concat，mean_flow 經小型 encoder 產生 per-pixel 雙通道 FiLM 調製（γ_m, β_m）。
per-rollout 計算一次（mean 在 rollout 內為常數）。

採固定受體（Sobel-x/y + Laplacian，非 learnable buffer）+ 1×1 learnable fuse 設計，
參數量 ≈2.8k（in=9/hidden=32/out=24），且基底語彙與 turb MRP 的 TopoR 受體一致，
降低不可解釋模組。固定核直擊 mean 場邊緣/曲率（wake 結構感知），
learnable 部分只剩跨通道 1×1 融合 + γ_m/β_m heads。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanFlowEncoder(nn.Module):
    """mean_flow 場 → 空間 FiLM (γ_m, β_m)（turb_module mean conditioning）。

    turb input 維持 9ch fluct **不 concat**，
    mean_flow（訓練=GT ``mean_flow_post_conv``，推理=mean_module 輸出）經 encoder
    產生 **per-pixel** 雙通道 FiLM 調製（γ_m, β_m），保輸入 condition 可解釋性。

    ★lite 結構（固定受體 + 1×1 fuse）：
      1. 固定 depthwise 受體（buffer，0 learnable）：每個 mean 通道套 Sobel-x、
         Sobel-y、Laplacian → in_channels×3 邊緣/曲率特徵（與 MRP TopoR 同基底）。
      2. raw 9ch ⊕ 受體特徵 → in_channels×4 通道 → 1×1 learnable fuse → hidden。
      3. 1×1 to_gamma / to_beta heads（zero-init）→ γ_m, β_m。

    組合（TESNCA P1 接線）：向量路徑 γ_p/γ_g [B,C] broadcast 至 [B,C,H,W]
    後與 γ_m 相加——γ = γ_p + γ_g + γ_m，β 同理。

    zero-init：γ_m/β_m 輸出層全零 → 未接線/剛接線時不改變既有 FiLM 行為
    （模型修改規則：新增 head 一律 zero-initialization）。

    Args:
        in_channels: mean_flow 通道數，預設 9（Mphy norm 空間）。
        hidden_dim: 1×1 fuse 隱藏通道數，預設 32。
        out_dim: γ_m/β_m 通道數，等於 nca_channels，預設 64。
    """

    _N_RECEPTORS: int = 3   # Sobel-x, Sobel-y, Laplacian

    def __init__(
        self,
        in_channels: int = 9,
        hidden_dim: int = 32,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        # ── 固定 depthwise 受體（非 learnable，buffer）：Sobel-x/y（÷8 正規化）+ Laplacian ──
        sob_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sob_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        recep = torch.stack([sob_x, sob_y, lap], dim=0)              # [3, 3, 3]
        # grouped(depthwise) weight：[in_channels*3, 1, 3, 3]，group g（通道 g）→ 3 受體輸出。
        # repeat 後通道序 = [c0_sobx, c0_soby, c0_lap, c1_sobx, ...]，對齊 groups=in_channels 期望。
        weight = recep.repeat(in_channels, 1, 1).unsqueeze(1)        # [in_channels*3, 1, 3, 3]
        self.register_buffer("recep_weight", weight)

        fuse_in: int = in_channels * (1 + self._N_RECEPTORS)         # raw ⊕ 受體
        # ``encoder`` 命名保留（向後相容 test：mean_encoder.encoder[0]）；首層即 learnable 1×1 fuse。
        self.encoder = nn.Sequential(
            nn.Conv2d(fuse_in, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        self.to_gamma = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
        self.to_beta = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

        # zero-init：γ_m = 0、β_m = 0 → 加法組合下不干擾既有 pathway
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(
        self,
        mean_flow_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """計算 mean conditioning 空間 FiLM 調製參數。

        Args:
            mean_flow_norm: [B, in_channels, H, W] norm 空間 mean_flow。

        Returns:
            gamma_m: [B, out_dim, H, W] 空間縮放參數。
            beta_m:  [B, out_dim, H, W] 空間平移參數。
        """
        x = mean_flow_norm
        # 固定受體：replicate pad 避邊界偏壓；buffer 對齊 x dtype（AMP fp16 安全）。
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        feat = F.conv2d(x_pad, self.recep_weight.to(x.dtype), groups=self.in_channels)
        h = self.encoder(torch.cat([x, feat], dim=1))
        return self.to_gamma(h), self.to_beta(h)


__all__ = ["MeanFlowEncoder"]
