"""條件向量建構子模組。

本模組提供 ConditionBuilder，負責依 cfg.model.cond_dim 組合正規化後的
條件向量（cond_norm）及 G_vec 雙路徑正規化輸出（g_vec_norm），
從 TESTrainer._build_cond_norm 與 _build_g_vec_norm、_uses_per_step_delta_ctu 提取。
"""

from __future__ import annotations

from typing import Optional

import torch

from ..context import TrainerContext


class ConditionBuilder:
    """條件向量建構器，依 cfg.model.cond_dim 組合 cond_norm 與 g_vec_norm。

    從 TESCondNormalizer 委派實際正規化計算，本類別負責路由分派與例外提示。

    Args:
        ctx: TrainerContext 共享狀態容器。
    """

    def __init__(self, ctx: TrainerContext) -> None:
        """初始化 ConditionBuilder。

        Args:
            ctx: TrainerContext 共享狀態容器。
        """
        self.ctx = ctx

    def build_cond_norm(
        self,
        g_vec: torch.Tensor,
        p_vec: torch.Tensor,
    ) -> torch.Tensor:
        """組合條件向量 ``cond_17D = [G_vec_norm(14), P_vec_norm(3)]``。

        本函式簽名不含時間參數（``delta_ctu`` / ``cumulative_ctu``）：它們恆為
        None，屬於 `build_cond` 內以常數填補的 fallback，而非真實條件。

        Args:
            g_vec: 幾何特徵向量 [B, 14]。
            p_vec: 物理參數向量 [B, 3]（Re、U_inlet、L_char）。

        Returns:
            正規化後的條件向量 [B, 17]。

        Raises:
            ValueError: cfg.model.cond_dim != 17。
        """
        cfg = self.ctx.cfg
        cond_dim = cfg["model"]["cond_dim"] if isinstance(cfg, dict) else cfg.model.cond_dim
        return self.ctx.normalizer.build_cond(cond_dim, g_vec, p_vec)

    def build_g_vec_norm(
        self,
        g_vec: torch.Tensor,
    ) -> torch.Tensor | None:
        """Dual-pathway G_vec 正規化（gvec_mode=additive/gated 且 g_vec_dim>0 時回傳正規化 G_vec）。

        Args:
            g_vec: 幾何特徵向量 [B, g_vec_dim]。

        Returns:
            正規化後的 G_vec [B, g_vec_dim]；gvec_mode 非 additive/gated 或
            g_vec_dim=0 時回傳 None。
        """
        cfg = self.ctx.cfg
        gvec_mode = cfg.model.gvec_mode
        if gvec_mode in ("additive", "gated") and cfg.model.g_vec_dim > 0:
            return self.ctx.normalizer.normalize_g(g_vec)
        return None

    def uses_per_step_delta_ctu(self) -> bool:
        """判斷 model.forward 是否應接收 per-step delta_ctu_seq。

        Returns:
            time_conditioned 或（arch=h_multirate 且 cond_dim=19）時為 True。
        """
        cfg = self.ctx.cfg
        return bool(
            cfg.time_conditioned
            or (
                cfg.model.arch == "h_multirate"
                and cfg.model.cond_dim == 19
            )
        )
