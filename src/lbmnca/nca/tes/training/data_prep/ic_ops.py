"""初始條件（IC）操作子模組。

本模組提供 ICOps，負責流體遮罩推導、訓練雜訊注入、均勻入口速度先驗
以及 GT injection 機率排程，從 TESTrainer._get_fluid_mask、_maybe_train_noise、
_get_prior_vel_norm、_get_mean_p_gt 提取。
"""

from __future__ import annotations

import torch

from ..context import TrainerContext


class ICOps:
    """初始條件（IC）相關操作集合。

    涵蓋：SDF 推導流體遮罩、訓練時加噪、均勻入口動量先驗 cache、
    mean/turb scheduled sampling GT injection 機率計算。

    Args:
        ctx: TrainerContext 共享狀態容器。
    """

    def __init__(self, ctx: TrainerContext) -> None:
        """初始化 ICOps。

        Args:
            ctx: TrainerContext 共享狀態容器。
        """
        self.ctx = ctx
        self._prior_vel_norm_cache: torch.Tensor | None = None

    def get_fluid_mask(self, static: torch.Tensor) -> torch.Tensor:
        """由 SDF_norm 依 dataset SDF 統計推導正確的二值流體遮罩。

        ``SDF_norm > threshold`` → 流體（1.0），否則固體（0.0）。

        Args:
            static: 靜態通道張量 [B, 4, H, W]（channel 0 為 SDF_norm）。

        Returns:
            流體遮罩 [B, 1, H, W]（float，1.0=流體 / 0.0=固體）。
        """
        return (static[:, 0:1] > self.ctx.sdf_threshold).float()

    def apply_noise(
        self,
        ic_physics: torch.Tensor,
        fluid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Dy-A：訓練時對 IC physics 加 N(0,σ²)（fluid 區）。eval / σ=0 時不動。

        教模型「偏離→拉回」→ GT 軌跡 manifold 變 attractor
        （limit-cycle 無定點靠此拘束）。

        Args:
            ic_physics: IC physics 張量 [B, 9, H, W]（model state 空間）。
            fluid_mask: 流體遮罩 [B, 1, H, W]（1.0=流體）。

        Returns:
            加噪後的 IC physics 張量；σ=0 或 eval 模式時回傳原張量。
        """
        sigma = float(self.ctx.cfg.train_noise_sigma or 0.0)
        if sigma > 0.0 and self.ctx.model.training:
            ic_physics = ic_physics + sigma * torch.randn_like(ic_physics) * fluid_mask
        return ic_physics

    def get_prior_vel_norm(self) -> torch.Tensor:
        """HF-1d：均勻入口 momentum 的 normalized 值 [jx_inlet_norm, jy_inlet_norm]（cache）。

        raw jx_inlet = rho(≈1)·U_inlet(0.05)、jy_inlet = 0；以 physics normalizer
        tanh((raw−M_REF)/S_combined) 轉至 model state 空間。deterministic prior 的目標速度。

        Returns:
            [2] float tensor（jx_inlet_norm, jy_inlet_norm），已搬至 ctx.device。
        """
        if self._prior_vel_norm_cache is None:
            nz = self.ctx.loss_fn._get_normalizer()
            mref, scomb = nz._mf_mean, nz._mf_std            # [9] M_REF / S_combined
            u_inlet = 0.05                                    # P_vec[1] 固定（lattice）
            jx_n = torch.tanh((u_inlet - mref[3]) / scomb[3])
            jy_n = torch.tanh((0.0 - mref[5]) / scomb[5])
            self._prior_vel_norm_cache = (
                torch.stack([jx_n, jy_n]).to(self.ctx.device).float()
            )
        return self._prior_vel_norm_cache

    def get_mean_p_gt(self) -> float:
        """mean/turb scheduled sampling 的 GT injection 機率。

        排程（progress = step / max_steps）：
            progress ∈ [0, p1_end]      → p_gt=1.0（全 GT）
            progress ∈ [p1_end, p2_end] → 線性 1→0
            progress ∈ [p2_end, 1]      → p_gt=0.0（全 TBPTT）

        Returns:
            當前步的 GT injection 機率 p_gt（0.0 ~ 1.0）。
        """
        cfg = self.ctx.cfg
        p1 = float(cfg.mean_gt_phase1_end)
        p2 = float(cfg.mean_gt_phase2_end)
        progress = self.ctx.step / max(1, cfg.max_steps)
        if progress <= p1:
            return 1.0
        if progress >= p2:
            return 0.0
        return 1.0 - (progress - p1) / max(1e-6, p2 - p1)
