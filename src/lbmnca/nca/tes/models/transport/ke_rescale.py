"""動能重縮放模組（KErescaler）。

從 H3TStreamCollideNCA.forward 的 ke_rescale 區塊提取，封裝為獨立可重用模組。
負責防止長程 rollout KE 爆炸或衰減（V2-C / V2-C2 / V2-D 聯合策略）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class KErescaler(nn.Module):
    """動能重縮放（防止長程 rollout KE 爆炸/衰減；start_ctu 後才啟用）。

    實作 LC-1 [A12] + [A15] + [A17] 策略：
        - [A12]：alpha fixed=1.0（buffer，非 learnable）→ s=g → 精確每步 KE 守恆，不漂回 0。
        - [A15]：KE rescale 移到 fine substeps 後（完整 step KE 守恆）。
        - [A17]：spin-up warmup gate——cumulative CTU < start_ctu 時 bypass（spin-up 不限 KE）。

    V2-C2 對稱 clamp：gmin<1 → 既補消散又壓過幅（移除 1.15^步 單向爆幅）。
    default gmin=1.0 = 非對稱 V2-C 原行為（只補不壓），零行為變更。

    rescale 僅作用在 physics channels 中的 jx（ch 3）與 jy（ch 5），
    其餘 channels 原樣保留，最後套 fluid_mask 清除 solid 區。

    注意：rescale 本身是 no_grad 操作（g 從 detached KE 計算），
    但 s 作用在 full_out 上，梯度仍可流過乘法（s.detach() 不需要額外標記）。
    """

    def __init__(
        self,
        ke_rescale_gmax: float,
        ke_rescale_gmin: float,
        ke_rescale_start_ctu: float,
    ) -> None:
        """初始化 KErescaler。

        Args:
            ke_rescale_gmax: 增益 g 的上限（防止 KE 過幅）。
            ke_rescale_gmin: 增益 g 的下限（V2-C2：<1.0 可壓過幅；1.0 = 只補不壓）。
            ke_rescale_start_ctu: cumulative CTU 達此值後才啟用 rescale（spin-up warmup）；
                0.0 = 永遠啟用。
        """
        super().__init__()
        self._ke_rescale_gmax = ke_rescale_gmax
        self._ke_rescale_gmin = ke_rescale_gmin
        self._ke_rescale_start_ctu = ke_rescale_start_ctu
        # LC-1 [A12]：α fixed=1.0（buffer，非 learnable）→ s=g → 精確每步 KE 守恆，不漂回 0。
        self.register_buffer("ke_rescale_alpha", torch.ones(1))

    # ------------------------------------------------------------------
    def forward(
        self,
        pred_phys: torch.Tensor,
        prev_phys: torch.Tensor,
        cumul_ctu: float,
        fluid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """執行動能重縮放，回傳修正後的 physics tensor。

        以步首（STREAM 前）KE 為參考，修正 fine substeps 後的完整輸出，
        使各 batch item 的 KE 盡量守恆（在 gmin/gmax 範圍內）。

        cumul_ctu < ke_rescale_start_ctu 時直接回傳 pred_phys（bypass）。

        Args:
            pred_phys: fine substeps 後完整 physics 輸出，形狀 [B, physics_dim, H, W]。
            prev_phys: 步首 physics（STREAM 前），形狀 [B, physics_dim, H, W]，
                作為 KE 參考（detached）。
            cumul_ctu: 當前累計 CTU（float），用於 spin-up warmup gate。
            fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，solid 區為 0。

        Returns:
            rescaled: 修正後的 physics tensor，形狀 [B, physics_dim, H, W]，
                已套 fluid_mask。若 cumul_ctu < start_ctu 則直接回傳 pred_phys（未修改）。
        """
        # LC-1 [A17]：warmup gate——cumul_ctu < start_ctu 時 bypass（spin-up 不限 KE）。
        if cumul_ctu < self._ke_rescale_start_ctu:
            return pred_phys

        bsz = pred_phys.shape[0]
        pd = pred_phys.shape[1]

        with torch.no_grad():
            with torch.autocast(device_type=pred_phys.device.type, enabled=False):
                jp = prev_phys.float()   # 步首幅值參考（STREAM 前）
                jn = pred_phys.float()   # fine substeps 後完整輸出
                fm = fluid_mask.float()
                denom = fm.sum(dim=(1, 2, 3)).clamp_min(1.0)  # [B]
                # jx ch=3, jy ch=5；KE ∝ jx² + jy²（密度 ρ=1 簡化）
                ke_in = (
                    (jp[:, 3:4] ** 2 + jp[:, 5:6] ** 2) * fm
                ).sum(dim=(1, 2, 3)) / denom
                ke_out = (
                    (jn[:, 3:4] ** 2 + jn[:, 5:6] ** 2) * fm
                ).sum(dim=(1, 2, 3)) / denom
                # g = sqrt(ke_in / ke_out)：使輸出 KE 縮回輸入幅值
                g = (ke_in / (ke_out + 1e-8)).sqrt().clamp(
                    self._ke_rescale_gmin, self._ke_rescale_gmax
                )  # [B]

        a = self.ke_rescale_alpha
        # α=1 → s=g（精確守恆）；α<1 → 軟化修正（較保守）
        s = (1.0 - a + a * g.view(bsz, 1, 1, 1)).to(pred_phys.dtype)  # [B,1,1,1]，detached
        # 僅對 jx（ch 3）與 jy（ch 5）套 s，其餘 channels 不動
        rescaled = torch.cat(
            [
                pred_phys[:, c:c + 1] * s if c in (3, 5) else pred_phys[:, c:c + 1]
                for c in range(pd)
            ],
            dim=1,
        ) * fluid_mask
        return rescaled


__all__ = ["KErescaler"]
