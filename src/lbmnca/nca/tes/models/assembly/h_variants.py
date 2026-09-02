"""H-series 分層 NCA 變體（H0-H5）

所有變體繼承 HBaseNCA，提供不同的多尺度分解策略。
每個 class 保持與 TESNCA 相同的 forward() 外部接口。

變體列表：
  H0 — HSharedScaleNCA：粗、細尺度共用同一 MarkovNCACell，以尺度條件特徵區分。
  H1 — HLapResNCA：粗尺度預測物理增量，細尺度預測殘差，加法合併。
  H2 — HParentLeafNCA：遞歸父狀態在粗網格更新，透過零初始化 actuator 注入葉節點。
  H3 — HMultirateNCA：多組粗網格尺度各自貢獻殘差，細網格執行 fine_substeps 次子步。
  H4 — HBlockRefineNCA：根據粗增量幅度評分，動態選取活躍 block 執行細網格精化。
  H5 — HPODModalNCA：POD 基底低維模態係數空間動態學，重建至全解析度物理殘差。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import TESModelConfig
from ..h_base import HBaseNCA
from ..markov_nca_cell import MarkovNCACell
from ..multiscale_ops import (
    downsample_mask,
    downsample_state,
    expand_block_halo,
    make_block_mask,
    upsample_state,
)
from ..scale_condition import SCALE_FEAT_DIM, build_scale_features


# ---------------------------------------------------------------------------
# H0：HSharedScaleNCA
# ---------------------------------------------------------------------------

class HSharedScaleNCA(HBaseNCA):
    """H0 共享尺度 NCA。Forward 介面與 :class:`TESNCA` 相同。

    粗尺度與細尺度共用同一個 MarkovNCACell，透過尺度條件特徵區分兩者。
    可選 top-down 注入：零初始化投影將粗隱藏狀態注入細尺度隱藏區塊。

    每個外層步驟依序執行一次粗尺度更新與一次細尺度更新，兩個尺度使用相同的 cell。
    兩者的差異僅在於：(a) 輸入解析度，以及 (b) 在 FiLM encoder 前拼接到
    ``cond_norm`` 的尺度條件特徵。

    可選的 top-down 注入：以零初始化的 1×1 conv 將上採樣後的粗隱藏通道投影到
    細尺度隱藏區塊，供細 cell 執行前使用。初始化時等價於平坦基線，
    模型可自行學習注入量。
    """

    def __init__(self, cfg: TESModelConfig) -> None:
        ms = cfg.multiscale
        n_scale_feats = SCALE_FEAT_DIM if ms.scale_condition else 0
        super().__init__(cfg, scale_feat_dim=n_scale_feats)

        if ms.levels != 2:
            # H0 第一版僅支援嚴格 2 層；更深的層次結構由 H2/H3 處理。
            raise ValueError(
                f"HSharedScaleNCA: levels must be 2 in this version, got {ms.levels}"
            )
        if len(ms.ratios) < 1:
            raise ValueError("HSharedScaleNCA: multiscale.ratios must have >=1 entry")
        self._ratio: int = int(ms.ratios[0])
        self._scale_cond: bool = bool(ms.scale_condition)
        self._topdown: bool = bool(ms.topdown)

        self.nca_cell = MarkovNCACell(
            state_dim=cfg.state_dim,
            nca_channels=cfg.nca_channels,
            physics_dim=cfg.physics_dim,
            hidden_dim=cfg.hidden_dim,
            mrp_config=cfg.mrp,
        )

        # 零初始化 top-down 投影：粗隱藏 → 細隱藏。
        # 初始化時等價於平坦基線，讓模型學習注入量。
        self.topdown_proj: Optional[nn.Conv2d] = None
        if self._topdown and cfg.hidden_dim > 0:
            self.topdown_proj = nn.Conv2d(
                cfg.hidden_dim, cfg.hidden_dim, kernel_size=1, bias=True
            )
            nn.init.zeros_(self.topdown_proj.weight)
            nn.init.zeros_(self.topdown_proj.bias)

    # ------------------------------------------------------------------
    def _encode_per_step(
        self,
        cond_norm: torch.Tensor,
        delta_ctu_seq: Optional[torch.Tensor],
        cumul_ctu_norm_seq: Optional[torch.Tensor],
        step_idx: int,
        time_conditioned: bool,
        per_step_cumul: bool,
        level_id: float,
        ratio: int,
        dt_scale: float,
        gamma_g: Optional[torch.Tensor],
        beta_g: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """建立每步的 (γ, β) FiLM 調製係數與可選的 Δt 空間純量。

        Args:
            cond_norm: 正規化條件向量，shape (B, cond_dim)。
            delta_ctu_seq: 逐步 Δctu 序列，shape (B, n_steps)，可為 None。
            cumul_ctu_norm_seq: 累積 ctu 正規化序列，shape (B, n_steps)，可為 None。
            step_idx: 當前步驟索引。
            time_conditioned: 是否啟用逐步 Δctu 條件化。
            per_step_cumul: 是否啟用逐步累積 ctu 條件化。
            level_id: 尺度識別符（粗=0.0，細=1.0）。
            ratio: 下採樣比率。
            dt_scale: Δt 縮放係數。
            gamma_g: G_vec FiLM gamma 偏移，可為 None。
            beta_g: G_vec FiLM beta 偏移，可為 None。

        Returns:
            tuple: (gamma, beta, dt_spatial)，dt_spatial 可為 None。
        """
        b = cond_norm.shape[0]
        device = cond_norm.device
        cond_i = cond_norm
        dt_spatial: Optional[torch.Tensor] = None
        if time_conditioned and delta_ctu_seq is not None:
            dt_i = delta_ctu_seq[:, step_idx:step_idx + 1]
            cond_i = torch.cat([cond_norm, dt_i], dim=1)
            dt_spatial = dt_i[:, :, None, None]
        elif per_step_cumul and cumul_ctu_norm_seq is not None:
            cond_i = cond_norm.clone()
            cond_i[:, 4] = cumul_ctu_norm_seq[:, step_idx]
        if self._scale_cond:
            scale_feats = build_scale_features(
                level_id=level_id,
                ratio=ratio,
                dt_scale=dt_scale,
                ctu_scale=1.0,
                batch_size=b,
                device=device,
            )
            cond_i = torch.cat([cond_i, scale_feats], dim=1)
        gamma, beta = self.cond_encoder(cond_i)
        gamma, beta = self._add_gvec(gamma, beta, gamma_g, beta_g)
        return gamma, beta, dt_spatial

    # ------------------------------------------------------------------
    def forward(
        self,
        state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        n_steps: int,
        return_trajectory: bool = False,
        delta_ctu_seq: Union[torch.Tensor, None] = None,
        g_vec_norm: Union[torch.Tensor, None] = None,
        tf_ratio: float = 0.0,
        gt_traj: Union[torch.Tensor, None] = None,
        ss_detach: bool = False,
        cumul_ctu_norm_seq: Union[torch.Tensor, None] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """執行 n_steps 步的共享尺度 NCA 推進。

        Args:
            state: 初始狀態張量，shape (B, state_dim, H, W)。
            cond_norm: 正規化條件向量，shape (B, cond_dim)。
            fluid_mask: 流體區域遮罩，shape (B, 1, H, W)。
            n_steps: NCA 推進步數。
            return_trajectory: 若為 True，同時回傳物理通道的步驟軌跡。
            delta_ctu_seq: 逐步 Δctu 序列，shape (B, n_steps)，可為 None。
            g_vec_norm: 正規化 G_vec，shape (B, g_vec_dim)，可為 None。
            tf_ratio: Teacher forcing 比例（0.0 表示不使用）。
            gt_traj: Ground truth 軌跡（teacher forcing 用），可為 None。
            ss_detach: 是否在 scheduled sampling 時 detach 狀態。
            cumul_ctu_norm_seq: 累積 ctu 正規化序列，可為 None。

        Returns:
            若 return_trajectory 為 False，回傳最終狀態 (B, state_dim, H, W)。
            若 return_trajectory 為 True，回傳 (最終狀態, 物理通道軌跡列表)。
        """
        cfg = self._cfg
        sd, pd, hd = cfg.static_dim, cfg.physics_dim, cfg.hidden_dim
        ratio = self._ratio
        h_full, w_full = state.shape[-2:]

        time_conditioned = delta_ctu_seq is not None
        per_step_cumul = (
            cumul_ctu_norm_seq is not None
            and cond_norm.shape[1] == 5
            and not time_conditioned
        )

        gamma_g, beta_g = self._compute_gvec_film(g_vec_norm)
        coarse_mask = downsample_mask(fluid_mask, ratio, threshold=0.5)

        trajectory: list[torch.Tensor] = []

        for i in range(n_steps):
            # 粗尺度更新
            gamma_c, beta_c, dt_spatial_c = self._encode_per_step(
                cond_norm,
                delta_ctu_seq,
                cumul_ctu_norm_seq,
                i,
                time_conditioned,
                per_step_cumul,
                level_id=0.0,
                ratio=ratio,
                dt_scale=1.0,
                gamma_g=gamma_g,
                beta_g=beta_g,
            )
            coarse_state = downsample_state(state, ratio, mode="area")
            coarse_state_new = self.nca_cell(
                coarse_state, gamma_c, beta_c, coarse_mask, dt_spatial_c
            )

            # top-down：將上採樣後的粗隱藏通道投影注入細尺度隱藏區塊（零初始化）
            fine_state_input = state
            if self.topdown_proj is not None and hd > 0:
                coarse_hidden = coarse_state_new[:, sd + pd:]
                up_hidden = upsample_state(
                    coarse_hidden, (h_full, w_full), mode="bilinear"
                )
                injected = self.topdown_proj(up_hidden)
                fine_state_input = torch.cat(
                    [
                        state[:, :sd + pd],
                        state[:, sd + pd:] + injected,
                    ],
                    dim=1,
                )

            # 細尺度更新
            gamma_f, beta_f, dt_spatial_f = self._encode_per_step(
                cond_norm,
                delta_ctu_seq,
                cumul_ctu_norm_seq,
                i,
                time_conditioned,
                per_step_cumul,
                level_id=1.0,
                ratio=1,
                dt_scale=1.0,
                gamma_g=gamma_g,
                beta_g=beta_g,
            )
            state = self.nca_cell(
                fine_state_input, gamma_f, beta_f, fluid_mask, dt_spatial_f
            )

            if return_trajectory:
                trajectory.append(state[:, sd:sd + pd])

            state = self._apply_tfc(
                state, gt_traj, i, tf_ratio, fluid_mask, ss_detach
            )

        if return_trajectory:
            return state, trajectory
        return state


# ---------------------------------------------------------------------------
# H1：HLapResNCA
# ---------------------------------------------------------------------------

class HLapResNCA(HBaseNCA):
    """H1 Laplacian-Residual NCA。Forward 介面與 :class:`TESNCA` 相同。

    粗 cell 在粗網格上預測物理增量，細 cell 在完整網格預測殘差，
    兩者相加得到最終物理更新量：

        new_physics = prev_physics + prolong(coarse_delta) + fine_residual

    兩個 cell 的輸出投影均以零初始化，確保步驟 0 的行為等價於平坦基線。

    訓練器 logging 的輔助指標：
      * coarse_pred_energy — 粗預測能量
      * fine_residual_energy — 細殘差能量
      * residual_to_coarse_energy_ratio — 殘差對粗預測的能量比
    """

    def __init__(self, cfg: TESModelConfig) -> None:
        ms = cfg.multiscale
        n_scale_feats = SCALE_FEAT_DIM if ms.scale_condition else 0
        super().__init__(cfg, scale_feat_dim=n_scale_feats)

        if ms.levels != 2:
            raise ValueError(
                f"HLapResNCA: levels must be 2 in this version, got {ms.levels}"
            )
        if len(ms.ratios) < 1:
            raise ValueError("HLapResNCA: multiscale.ratios must have >=1 entry")
        self._ratio: int = int(ms.ratios[0])
        self._scale_cond: bool = bool(ms.scale_condition)

        # 粗、細各自獨立的 cell；輸出 delta 投影均以零初始化。
        self.coarse_cell = MarkovNCACell(
            state_dim=cfg.state_dim,
            nca_channels=cfg.nca_channels,
            physics_dim=cfg.physics_dim,
            hidden_dim=cfg.hidden_dim,
            mrp_config=cfg.mrp,
        )
        self.fine_cell = MarkovNCACell(
            state_dim=cfg.state_dim,
            nca_channels=cfg.nca_channels,
            physics_dim=cfg.physics_dim,
            hidden_dim=cfg.hidden_dim,
            mrp_config=cfg.mrp,
        )

    # ------------------------------------------------------------------
    def _encode_per_step(
        self,
        cond_norm: torch.Tensor,
        delta_ctu_seq: Optional[torch.Tensor],
        cumul_ctu_norm_seq: Optional[torch.Tensor],
        step_idx: int,
        time_conditioned: bool,
        per_step_cumul: bool,
        level_id: float,
        ratio: int,
        gamma_g: Optional[torch.Tensor],
        beta_g: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """建立每步的 (γ, β) FiLM 調製係數與可選的 Δt 空間純量。

        Args:
            cond_norm: 正規化條件向量，shape (B, cond_dim)。
            delta_ctu_seq: 逐步 Δctu 序列，shape (B, n_steps)，可為 None。
            cumul_ctu_norm_seq: 累積 ctu 正規化序列，shape (B, n_steps)，可為 None。
            step_idx: 當前步驟索引。
            time_conditioned: 是否啟用逐步 Δctu 條件化。
            per_step_cumul: 是否啟用逐步累積 ctu 條件化。
            level_id: 尺度識別符（粗=0.0，細=1.0）。
            ratio: 下採樣比率。
            gamma_g: G_vec FiLM gamma 偏移，可為 None。
            beta_g: G_vec FiLM beta 偏移，可為 None。

        Returns:
            tuple: (gamma, beta, dt_spatial)，dt_spatial 可為 None。
        """
        b = cond_norm.shape[0]
        device = cond_norm.device
        cond_i = cond_norm
        dt_spatial: Optional[torch.Tensor] = None
        if time_conditioned and delta_ctu_seq is not None:
            dt_i = delta_ctu_seq[:, step_idx:step_idx + 1]
            cond_i = torch.cat([cond_norm, dt_i], dim=1)
            dt_spatial = dt_i[:, :, None, None]
        elif per_step_cumul and cumul_ctu_norm_seq is not None:
            cond_i = cond_norm.clone()
            cond_i[:, 4] = cumul_ctu_norm_seq[:, step_idx]
        if self._scale_cond:
            scale_feats = build_scale_features(
                level_id=level_id,
                ratio=ratio,
                dt_scale=1.0,
                ctu_scale=1.0,
                batch_size=b,
                device=device,
            )
            cond_i = torch.cat([cond_i, scale_feats], dim=1)
        gamma, beta = self.cond_encoder(cond_i)
        gamma, beta = self._add_gvec(gamma, beta, gamma_g, beta_g)
        return gamma, beta, dt_spatial

    # ------------------------------------------------------------------
    def forward(
        self,
        state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        n_steps: int,
        return_trajectory: bool = False,
        delta_ctu_seq: Union[torch.Tensor, None] = None,
        g_vec_norm: Union[torch.Tensor, None] = None,
        tf_ratio: float = 0.0,
        gt_traj: Union[torch.Tensor, None] = None,
        ss_detach: bool = False,
        cumul_ctu_norm_seq: Union[torch.Tensor, None] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """執行 n_steps 步的 Laplacian-Residual NCA 推進。

        Args:
            state: 初始狀態張量，shape (B, state_dim, H, W)。
            cond_norm: 正規化條件向量，shape (B, cond_dim)。
            fluid_mask: 流體區域遮罩，shape (B, 1, H, W)。
            n_steps: NCA 推進步數。
            return_trajectory: 若為 True，同時回傳物理通道的步驟軌跡。
            delta_ctu_seq: 逐步 Δctu 序列，shape (B, n_steps)，可為 None。
            g_vec_norm: 正規化 G_vec，shape (B, g_vec_dim)，可為 None。
            tf_ratio: Teacher forcing 比例（0.0 表示不使用）。
            gt_traj: Ground truth 軌跡（teacher forcing 用），可為 None。
            ss_detach: 是否在 scheduled sampling 時 detach 狀態。
            cumul_ctu_norm_seq: 累積 ctu 正規化序列，可為 None。

        Returns:
            若 return_trajectory 為 False，回傳最終狀態 (B, state_dim, H, W)。
            若 return_trajectory 為 True，回傳 (最終狀態, 物理通道軌跡列表)。
        """
        cfg = self._cfg
        sd, pd, hd = cfg.static_dim, cfg.physics_dim, cfg.hidden_dim
        ratio = self._ratio
        h_full, w_full = state.shape[-2:]

        time_conditioned = delta_ctu_seq is not None
        per_step_cumul = (
            cumul_ctu_norm_seq is not None
            and cond_norm.shape[1] == 5
            and not time_conditioned
        )

        gamma_g, beta_g = self._compute_gvec_film(g_vec_norm)
        coarse_mask = downsample_mask(fluid_mask, ratio, threshold=0.5)

        trajectory: list[torch.Tensor] = []
        # 使用純量 tensor 累積，避免每步 .item() 造成 CPU 同步
        coarse_energy_acc = state.new_zeros(1)
        fine_energy_acc = state.new_zeros(1)

        for i in range(n_steps):
            # 粗尺度更新
            gamma_c, beta_c, dt_spatial_c = self._encode_per_step(
                cond_norm,
                delta_ctu_seq,
                cumul_ctu_norm_seq,
                i,
                time_conditioned,
                per_step_cumul,
                level_id=0.0,
                ratio=ratio,
                gamma_g=gamma_g,
                beta_g=beta_g,
            )
            coarse_state = downsample_state(state, ratio, mode="area")
            coarse_state_new = self.coarse_cell(
                coarse_state, gamma_c, beta_c, coarse_mask, dt_spatial_c
            )
            coarse_phys_delta = (
                coarse_state_new[:, sd:sd + pd] - coarse_state[:, sd:sd + pd]
            )
            coarse_phys_delta_up = upsample_state(
                coarse_phys_delta, (h_full, w_full), mode="bilinear"
            )

            # 完整網格的細尺度更新
            gamma_f, beta_f, dt_spatial_f = self._encode_per_step(
                cond_norm,
                delta_ctu_seq,
                cumul_ctu_norm_seq,
                i,
                time_conditioned,
                per_step_cumul,
                level_id=1.0,
                ratio=1,
                gamma_g=gamma_g,
                beta_g=beta_g,
            )
            fine_state_new = self.fine_cell(
                state, gamma_f, beta_f, fluid_mask, dt_spatial_f
            )
            fine_phys_resid = (
                fine_state_new[:, sd:sd + pd] - state[:, sd:sd + pd]
            )

            new_physics = (
                state[:, sd:sd + pd] + coarse_phys_delta_up + fine_phys_resid
            )
            new_physics = new_physics * fluid_mask

            # 隱藏通道：採用細 cell 的隱藏更新（已遮罩）。
            if hd > 0:
                new_hidden = fine_state_new[:, sd + pd:]
                state = torch.cat([state[:, :sd], new_physics, new_hidden], dim=1)
            else:
                state = torch.cat([state[:, :sd], new_physics], dim=1)

            if return_trajectory:
                trajectory.append(state[:, sd:sd + pd])

            # 累積尺度指標（在 inference_mode 下減少梯度記憶體佔用）
            with torch.inference_mode():
                coarse_energy_acc += coarse_phys_delta_up.detach().float().pow(2).mean()
                fine_energy_acc += fine_phys_resid.detach().float().pow(2).mean()

            state = self._apply_tfc(
                state, gt_traj, i, tf_ratio, fluid_mask, ss_detach
            )

        # 迴圈結束後一次性轉換為 Python float，避免每步 CPU 同步
        n = max(1, n_steps)
        coarse_avg = coarse_energy_acc.item() / n
        fine_avg = fine_energy_acc.item() / n
        self._aux_metrics = {
            "coarse_pred_energy": coarse_avg,
            "fine_residual_energy": fine_avg,
            "residual_to_coarse_energy_ratio": fine_avg / max(coarse_avg, 1e-12),
        }

        if return_trajectory:
            return state, trajectory
        return state


# ---------------------------------------------------------------------------
# H2：HParentLeafNCA
# ---------------------------------------------------------------------------

class HParentLeafNCA(HBaseNCA):
    """H2 Parent-Leaf NCA。Forward 介面與 :class:`TESNCA` 相同。

    父節點在粗網格遞迴更新，透過零初始化 actuator 對葉節點施加 top-down 影響。
    可選 bottom-up feedback 將葉節點訊號回饋至父節點。

    粗尺度父狀態（parent state）以 full_grid // ratio 的解析度存在，
    並在每個外層步驟中遞迴更新。每個步驟中，父節點透過 AvgPool + 1×1 conv
    感知葉節點（leaf），執行一個小型 3×3 conv block，再透過零初始化的
    actuator → upsample 對葉節點施加影響。可選的 bottom-up feedback
    將葉節點池化後回饋至父節點。

    Top-down 注入在 ``hidden_dim > 0`` 時作用於葉節點隱藏區塊，
    否則作用於物理通道。Actuator 以零初始化，確保步驟 0 等價於平坦基線。
    """

    def __init__(self, cfg: TESModelConfig) -> None:
        ms = cfg.multiscale
        n_scale_feats = SCALE_FEAT_DIM if ms.scale_condition else 0
        super().__init__(cfg, scale_feat_dim=n_scale_feats)

        if len(ms.ratios) < 1:
            raise ValueError("HParentLeafNCA: multiscale.ratios must have >=1 entry")
        self._ratio: int = int(ms.ratios[0])
        self._scale_cond: bool = bool(ms.scale_condition)
        self._topdown: bool = bool(ms.topdown)
        self._bottomup: bool = bool(ms.bottomup)
        self._parent_channels: int = int(ms.parent_channels)

        # 感測器：葉節點 → 粗網格 → 1×1 conv → 父節點輸入。
        self.sensor = nn.Conv2d(
            cfg.state_dim, self._parent_channels, kernel_size=1, bias=True
        )

        # 父節點遞迴更新（3×3 conv block，delta 投影零初始化）。
        self.parent_conv = nn.Conv2d(
            self._parent_channels, cfg.nca_channels, kernel_size=3, padding=1, bias=True
        )
        self.parent_project = nn.Conv2d(
            cfg.nca_channels, self._parent_channels, kernel_size=1, bias=True
        )
        nn.init.zeros_(self.parent_project.weight)
        nn.init.zeros_(self.parent_project.bias)

        # Actuator：父節點 → 葉節點訊號。有隱藏通道時注入隱藏區塊，
        # 否則注入物理通道。零初始化保持步驟 0 的 identity 特性。
        self._inject_to_hidden: bool = cfg.hidden_dim > 0
        out_dim: int = cfg.hidden_dim if self._inject_to_hidden else cfg.physics_dim
        self.actuator = nn.Conv2d(
            self._parent_channels, out_dim, kernel_size=1, bias=True
        )
        nn.init.zeros_(self.actuator.weight)
        nn.init.zeros_(self.actuator.bias)

        # 葉節點 cell — 標準 MarkovNCACell，使用完整狀態維度。
        self.leaf_cell = MarkovNCACell(
            state_dim=cfg.state_dim,
            nca_channels=cfg.nca_channels,
            physics_dim=cfg.physics_dim,
            hidden_dim=cfg.hidden_dim,
            mrp_config=cfg.mrp,
        )

        # Bottom-up feedback（零初始化）。
        self.bottomup_conv: Optional[nn.Conv2d]
        if self._bottomup:
            self.bottomup_conv = nn.Conv2d(
                out_dim, self._parent_channels, kernel_size=1, bias=True
            )
            nn.init.zeros_(self.bottomup_conv.weight)
            nn.init.zeros_(self.bottomup_conv.bias)
        else:
            self.bottomup_conv = None

    # ------------------------------------------------------------------
    def _encode_leaf_cond(
        self,
        cond_norm: torch.Tensor,
        delta_ctu_seq: Optional[torch.Tensor],
        cumul_ctu_norm_seq: Optional[torch.Tensor],
        step_idx: int,
        time_conditioned: bool,
        per_step_cumul: bool,
        gamma_g: Optional[torch.Tensor],
        beta_g: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """建立葉節點每步的 (γ, β) FiLM 調製係數與可選的 Δt 空間純量。

        Args:
            cond_norm: 正規化條件向量，shape (B, cond_dim)。
            delta_ctu_seq: 逐步 Δctu 序列，shape (B, n_steps)，可為 None。
            cumul_ctu_norm_seq: 累積 ctu 正規化序列，shape (B, n_steps)，可為 None。
            step_idx: 當前步驟索引。
            time_conditioned: 是否啟用逐步 Δctu 條件化。
            per_step_cumul: 是否啟用逐步累積 ctu 條件化。
            gamma_g: G_vec FiLM gamma 偏移，可為 None。
            beta_g: G_vec FiLM beta 偏移，可為 None。

        Returns:
            tuple: (gamma, beta, dt_spatial)，dt_spatial 可為 None。
        """
        b = cond_norm.shape[0]
        device = cond_norm.device
        cond_i = cond_norm
        dt_spatial: Optional[torch.Tensor] = None
        if time_conditioned and delta_ctu_seq is not None:
            dt_i = delta_ctu_seq[:, step_idx:step_idx + 1]
            cond_i = torch.cat([cond_norm, dt_i], dim=1)
            dt_spatial = dt_i[:, :, None, None]
        elif per_step_cumul and cumul_ctu_norm_seq is not None:
            cond_i = cond_norm.clone()
            cond_i[:, 4] = cumul_ctu_norm_seq[:, step_idx]
        if self._scale_cond:
            scale_feats = build_scale_features(
                level_id=1.0, ratio=1, dt_scale=1.0, ctu_scale=1.0,
                batch_size=b, device=device,
            )
            cond_i = torch.cat([cond_i, scale_feats], dim=1)
        gamma, beta = self.cond_encoder(cond_i)
        gamma, beta = self._add_gvec(gamma, beta, gamma_g, beta_g)
        return gamma, beta, dt_spatial

    # ------------------------------------------------------------------
    def forward(
        self,
        state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        n_steps: int,
        return_trajectory: bool = False,
        delta_ctu_seq: Union[torch.Tensor, None] = None,
        g_vec_norm: Union[torch.Tensor, None] = None,
        tf_ratio: float = 0.0,
        gt_traj: Union[torch.Tensor, None] = None,
        ss_detach: bool = False,
        cumul_ctu_norm_seq: Union[torch.Tensor, None] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """執行 n_steps 步的 Parent-Leaf NCA 推進。

        Args:
            state: 初始狀態張量，shape (B, state_dim, H, W)。
            cond_norm: 正規化條件向量，shape (B, cond_dim)。
            fluid_mask: 流體區域遮罩，shape (B, 1, H, W)。
            n_steps: NCA 推進步數。
            return_trajectory: 若為 True，同時回傳物理通道的步驟軌跡。
            delta_ctu_seq: 逐步 Δctu 序列，shape (B, n_steps)，可為 None。
            g_vec_norm: 正規化 G_vec，shape (B, g_vec_dim)，可為 None。
            tf_ratio: Teacher forcing 比例（0.0 表示不使用）。
            gt_traj: Ground truth 軌跡（teacher forcing 用），可為 None。
            ss_detach: 是否在 scheduled sampling 時 detach 狀態。
            cumul_ctu_norm_seq: 累積 ctu 正規化序列，可為 None。

        Returns:
            若 return_trajectory 為 False，回傳最終狀態 (B, state_dim, H, W)。
            若 return_trajectory 為 True，回傳 (最終狀態, 物理通道軌跡列表)。
        """
        cfg = self._cfg
        sd, pd, hd = cfg.static_dim, cfg.physics_dim, cfg.hidden_dim
        ratio = self._ratio
        h_full, w_full = state.shape[-2:]

        time_conditioned = delta_ctu_seq is not None
        per_step_cumul = (
            cumul_ctu_norm_seq is not None
            and cond_norm.shape[1] == 5
            and not time_conditioned
        )
        gamma_g, beta_g = self._compute_gvec_film(g_vec_norm)

        # 初始父狀態由感測葉節點衍生。
        parent_state = self.sensor(F.avg_pool2d(state, kernel_size=ratio, stride=ratio))

        trajectory: list[torch.Tensor] = []
        # 使用純量 tensor 累積，避免每步 .item() 造成 CPU 同步
        topdown_norm_acc = state.new_zeros(1)

        for i in range(n_steps):
            # 1. 父節點從感測葉節點進行遞迴更新
            sensed = self.sensor(
                F.avg_pool2d(state, kernel_size=ratio, stride=ratio)
            )
            parent_in = parent_state + sensed
            h_parent = F.silu(self.parent_conv(parent_in))
            parent_state = parent_state + self.parent_project(h_parent)

            # 2. actuator → 葉節點訊號（訓練初期為零初始化）
            signal_coarse = self.actuator(parent_state)
            signal_fine = upsample_state(signal_coarse, (h_full, w_full), mode="bilinear")

            # 3. 注入 top-down 訊號（僅在 topdown 啟用時執行）
            if self._topdown:
                if self._inject_to_hidden and hd > 0:
                    leaf_input = torch.cat(
                        [state[:, :sd + pd], state[:, sd + pd:] + signal_fine], dim=1
                    )
                else:
                    new_phys = (state[:, sd:sd + pd] + signal_fine) * fluid_mask
                    if hd > 0:
                        leaf_input = torch.cat(
                            [state[:, :sd], new_phys, state[:, sd + pd:]], dim=1
                        )
                    else:
                        leaf_input = torch.cat([state[:, :sd], new_phys], dim=1)
            else:
                leaf_input = state

            # 4. 葉節點 cell 更新
            gamma, beta, dt_spatial = self._encode_leaf_cond(
                cond_norm, delta_ctu_seq, cumul_ctu_norm_seq, i,
                time_conditioned, per_step_cumul, gamma_g, beta_g,
            )
            state = self.leaf_cell(leaf_input, gamma, beta, fluid_mask, dt_spatial)

            # 5. 可選的 bottom-up feedback（零初始化）注回父節點供下一步使用
            if self.bottomup_conv is not None:
                leaf_signal_src = state[:, sd + pd:] if (
                    self._inject_to_hidden and hd > 0
                ) else state[:, sd:sd + pd]
                pooled = F.avg_pool2d(
                    leaf_signal_src, kernel_size=ratio, stride=ratio
                )
                parent_state = parent_state + self.bottomup_conv(pooled)

            if return_trajectory:
                trajectory.append(state[:, sd:sd + pd])

            # 累積 top-down 能量指標（在 inference_mode 下減少梯度記憶體佔用）
            with torch.inference_mode():
                topdown_norm_acc += signal_fine.detach().float().pow(2).mean()

            state = self._apply_tfc(
                state, gt_traj, i, tf_ratio, fluid_mask, ss_detach
            )

        # 迴圈結束後一次性轉換為 Python float，避免每步 CPU 同步
        n = max(1, n_steps)
        self._aux_metrics = {
            "topdown_signal_energy": topdown_norm_acc.item() / n,
        }

        if return_trajectory:
            return state, trajectory
        return state


# ---------------------------------------------------------------------------
# H3：HMultirateNCA
# ---------------------------------------------------------------------------

_DELTA_CTU_LOG_MAX = math.log1p(5.0)


def _normalize_delta_ctu(delta_ctu: torch.Tensor) -> torch.Tensor:
    """不攜帶 normalizer 的情況下，複製 TESCondNormalizer.normalize_delta_ctu 的行為。

    Args:
        delta_ctu: 原始 delta CTU 張量。

    Returns:
        對數正規化後的張量。
    """
    return torch.log1p(delta_ctu.float().clamp_min(0.0)) / _DELTA_CTU_LOG_MAX


class HMultirateNCA(HBaseNCA):
    """H3 多速率 NCA。Forward 介面與 :class:`TESNCA` 相同。

    透過多個粗網格尺度（ratios）分別計算物理增量，
    疊加後加入細網格狀態，再執行 fine_substeps 次細步更新。

    每個外層步驟執行一次粗網格更新與 ``fine_substeps`` 次細網格更新。
    粗網格的物理增量透過插值延伸至完整格網，並作為殘差加入細層物理狀態，
    再執行細網格子步。細網格的 ``dt_scale`` 透過 scale features 傳入，
    使每個 CTU 能學習各自的速率。

    軌跡快照每個外層步驟只取一次（非每個細步），以保持 loss 目標長度一致。
    """

    def __init__(self, cfg: TESModelConfig) -> None:
        """初始化 HMultirateNCA。

        Args:
            cfg: TES 模型設定物件。

        Raises:
            ValueError: 若 ratios 為空或 fine_substeps < 1。
        """
        ms = cfg.multiscale
        n_scale_feats = SCALE_FEAT_DIM if ms.scale_condition else 0
        super().__init__(cfg, scale_feat_dim=n_scale_feats)

        if len(ms.ratios) < 1:
            raise ValueError("HMultirateNCA: multiscale.ratios must have >=1 entry")
        self._ratios: list[int] = [int(r) for r in ms.ratios]
        self._ratio: int = self._ratios[0]
        self._fine_substeps: int = int(ms.fine_substeps)
        self._scale_cond: bool = bool(ms.scale_condition)
        if self._fine_substeps < 1:
            raise ValueError(
                f"HMultirateNCA: fine_substeps must be >=1, got {self._fine_substeps}"
            )

        _msp = list(cfg.nca_msp_scales)
        _psi_curl = bool(cfg.psi_curl_momentum)
        self.coarse_cells: nn.ModuleList = nn.ModuleList(
            [
                MarkovNCACell(
                    state_dim=cfg.state_dim,
                    nca_channels=cfg.nca_channels,
                    physics_dim=cfg.physics_dim,
                    hidden_dim=cfg.hidden_dim,
                    mrp_config=cfg.mrp,
                    spectral_update=bool(cfg.nca_spectral_update),
                    spectral_modes=int(cfg.nca_spectral_modes),
                    msp_scales=_msp,
                    msp_mode=cfg.nca_msp_mode,
                    msp_down_mode=cfg.nca_msp_down_mode,
                    msp_up_mode=cfg.nca_msp_up_mode,
                    psi_curl_momentum=_psi_curl,
                )
                for _ in self._ratios
            ]
        )
        self.fine_cell: MarkovNCACell = MarkovNCACell(
            state_dim=cfg.state_dim,
            nca_channels=cfg.nca_channels,
            physics_dim=cfg.physics_dim,
            hidden_dim=cfg.hidden_dim,
            mrp_config=cfg.mrp,
            spectral_update=bool(cfg.nca_spectral_update),
            spectral_modes=int(cfg.nca_spectral_modes),
            msp_scales=_msp,
                    msp_mode=cfg.nca_msp_mode,
                    msp_down_mode=cfg.nca_msp_down_mode,
                    msp_up_mode=cfg.nca_msp_up_mode,
            psi_curl_momentum=_psi_curl,
        )

    @property
    def coarse_cell(self) -> MarkovNCACell:
        """向後相容的別名，指向第一個粗尺度 cell。"""
        return self.coarse_cells[0]

    # ------------------------------------------------------------------
    def _encode_cond(
        self,
        cond_norm: torch.Tensor,
        delta_ctu_seq: Optional[torch.Tensor],
        cumul_ctu_norm_seq: Optional[torch.Tensor],
        step_idx: int,
        time_conditioned: bool,
        per_step_cumul: bool,
        level_id: float,
        ratio: int,
        dt_scale: float,
        gamma_g: Optional[torch.Tensor],
        beta_g: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """編碼每步條件向量，生成 FiLM 參數與空間 dt。

        Args:
            cond_norm: 正規化條件向量，shape [B, C]。
            delta_ctu_seq: 每步 delta CTU 序列，shape [B, T] 或 None。
            cumul_ctu_norm_seq: 累積 CTU 正規化序列，shape [B, T] 或 None。
            step_idx: 當前步驟索引。
            time_conditioned: 是否啟用時間條件。
            per_step_cumul: 是否逐步更新累積 CTU。
            level_id: 層級識別碼（float，供 scale_condition 使用）。
            ratio: 當前粗網格縮放比例。
            dt_scale: 細步時間縮放因子。
            gamma_g: G_vec FiLM gamma，shape [B, C] 或 None。
            beta_g: G_vec FiLM beta，shape [B, C] 或 None。

        Returns:
            gamma: FiLM gamma 張量。
            beta: FiLM beta 張量。
            dt_spatial: 空間 dt 張量，shape [B, 1, 1, 1] 或 None。
        """
        b = cond_norm.shape[0]
        device = cond_norm.device
        cond_i = cond_norm
        dt_spatial: Optional[torch.Tensor] = None
        if delta_ctu_seq is not None:
            dt_i = delta_ctu_seq[:, step_idx:step_idx + 1].to(
                device=device, dtype=cond_norm.dtype
            )
            dt_spatial = dt_i[:, :, None, None] * dt_scale
            if cond_norm.shape[1] == 19:
                # cond19 = [G_norm(14), P_norm(3), delta_ctu_norm, cumul_ctu_norm].
                # 維持向量寬度不變，每步只刷新時間欄位。
                cond_i = cond_norm.clone()
                cond_i[:, 17] = _normalize_delta_ctu(dt_i[:, 0]).to(cond_norm.dtype)
                if cumul_ctu_norm_seq is not None:
                    cond_i[:, 18] = cumul_ctu_norm_seq[:, step_idx].to(cond_norm.dtype)
            elif cond_norm.shape[1] == 5:
                # Legacy cond5 路徑：[P_norm(3), delta_ctu_norm, cumul_ctu_norm]。
                cond_i = cond_norm.clone()
                cond_i[:, 3] = _normalize_delta_ctu(dt_i[:, 0]).to(cond_norm.dtype)
                if cumul_ctu_norm_seq is not None:
                    cond_i[:, 4] = cumul_ctu_norm_seq[:, step_idx].to(cond_norm.dtype)
            elif time_conditioned:
                # cond18 風格路徑：呼叫端提供 17D 基底向量，模型在末尾拼接原始 delta_ctu。
                cond_i = torch.cat([cond_norm, dt_i], dim=1)
        elif per_step_cumul and cumul_ctu_norm_seq is not None:
            cond_i = cond_norm.clone()
            if cond_norm.shape[1] == 19:
                cond_i[:, 18] = cumul_ctu_norm_seq[:, step_idx].to(cond_norm.dtype)
            else:
                cond_i[:, 4] = cumul_ctu_norm_seq[:, step_idx].to(cond_norm.dtype)
        if self._scale_cond:
            scale_feats = build_scale_features(
                level_id=level_id, ratio=ratio, dt_scale=dt_scale, ctu_scale=1.0,
                batch_size=b, device=device,
            )
            cond_i = torch.cat([cond_i, scale_feats], dim=1)
        gamma, beta = self.cond_encoder(cond_i)
        gamma, beta = self._add_gvec(gamma, beta, gamma_g, beta_g)
        return gamma, beta, dt_spatial

    # ------------------------------------------------------------------
    def forward(
        self,
        state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        n_steps: int,
        return_trajectory: bool = False,
        delta_ctu_seq: Union[torch.Tensor, None] = None,
        g_vec_norm: Union[torch.Tensor, None] = None,
        tf_ratio: float = 0.0,
        gt_traj: Union[torch.Tensor, None] = None,
        ss_detach: bool = False,
        cumul_ctu_norm_seq: Union[torch.Tensor, None] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """執行多速率 NCA 前向推論。

        Args:
            state: 初始狀態張量，shape [B, state_dim, H, W]。
            cond_norm: 正規化條件向量，shape [B, cond_dim]。
            fluid_mask: 流體遮罩，shape [B, 1, H, W]，1=流體，0=固體。
            n_steps: 外層推進步驟數。
            return_trajectory: 若 True，回傳物理場軌跡（每外層步驟一筆）。
            delta_ctu_seq: 每步 delta CTU 序列，shape [B, T] 或 None。
            g_vec_norm: 正規化幾何向量，shape [B, g_dim] 或 None。
            tf_ratio: Teacher forcing 比例，0.0 表示完全自回歸。
            gt_traj: Ground truth 軌跡，供 teacher forcing 使用。
            ss_detach: 是否對 scheduled sampling 進行 detach。
            cumul_ctu_norm_seq: 累積 CTU 正規化序列，shape [B, T] 或 None。

        Returns:
            若 return_trajectory=False，回傳最終 state 張量。
            若 return_trajectory=True，回傳 (state, trajectory) tuple。
        """
        cfg = self._cfg
        sd, pd, hd = cfg.static_dim, cfg.physics_dim, cfg.hidden_dim
        ratios = self._ratios
        h_full, w_full = state.shape[-2:]
        sub = self._fine_substeps
        dt_fine = 1.0 / sub

        time_conditioned = delta_ctu_seq is not None
        per_step_cumul = (
            cumul_ctu_norm_seq is not None
            and cond_norm.shape[1] in (5, 19)
            and not time_conditioned
        )
        gamma_g, beta_g = self._compute_gvec_film(g_vec_norm)
        coarse_masks: list[torch.Tensor] = [
            downsample_mask(fluid_mask, ratio, threshold=0.5)
            for ratio in ratios
        ]

        trajectory: list[torch.Tensor] = []
        # 監控用累積量：以 Tensor 累積，最後統一 .item() 取純量，減少同步開銷
        coarse_energy_acc: torch.Tensor = state.new_zeros(())
        fine_substep_energy_acc: torch.Tensor = state.new_zeros(())
        coarse_phys_norm_acc: torch.Tensor = state.new_zeros(())
        fine_phys_norm_acc: torch.Tensor = state.new_zeros(())

        for i in range(n_steps):
            # 粗網格更新：每個設定的 ratio 貢獻同一步的殘差，
            # 感受野範圍來自尺度而非 rollout 長度。
            coarse_phys_delta_up_sum = state.new_zeros(
                state.shape[0], pd, h_full, w_full
            )
            for level_idx, (ratio, coarse_cell, coarse_mask) in enumerate(
                zip(ratios, self.coarse_cells, coarse_masks)
            ):
                gamma_c, beta_c, dt_spatial_c = self._encode_cond(
                    cond_norm, delta_ctu_seq, cumul_ctu_norm_seq, i,
                    time_conditioned, per_step_cumul,
                    level_id=float(level_idx), ratio=ratio, dt_scale=1.0,
                    gamma_g=gamma_g, beta_g=beta_g,
                )
                coarse_state = downsample_state(state, ratio, mode="area")
                coarse_state_new = coarse_cell(
                    coarse_state, gamma_c, beta_c, coarse_mask, dt_spatial_c
                )
                coarse_phys_delta = (
                    coarse_state_new[:, sd:sd + pd] - coarse_state[:, sd:sd + pd]
                )
                coarse_phys_delta_up = upsample_state(
                    coarse_phys_delta, (h_full, w_full), mode="bilinear"
                )
                coarse_phys_delta_up_sum = (
                    coarse_phys_delta_up_sum + coarse_phys_delta_up
                )
                with torch.no_grad():
                    coarse_energy_acc = coarse_energy_acc + coarse_phys_delta_up.detach().float().pow(2).mean()
            new_physics = (
                state[:, sd:sd + pd] + coarse_phys_delta_up_sum
            ) * fluid_mask
            with torch.no_grad():
                coarse_phys_norm_acc = coarse_phys_norm_acc + new_physics.detach().float().pow(2).mean()
            if hd > 0:
                state = torch.cat(
                    [state[:, :sd], new_physics, state[:, sd + pd:]], dim=1
                )
            else:
                state = torch.cat([state[:, :sd], new_physics], dim=1)

            # 細網格子步
            gamma_f, beta_f, dt_spatial_f = self._encode_cond(
                cond_norm, delta_ctu_seq, cumul_ctu_norm_seq, i,
                time_conditioned, per_step_cumul,
                level_id=float(len(ratios)), ratio=1, dt_scale=dt_fine,
                gamma_g=gamma_g, beta_g=beta_g,
            )
            for _ in range(sub):
                pre_phys = state[:, sd:sd + pd].detach()
                state = self.fine_cell(
                    state, gamma_f, beta_f, fluid_mask, dt_spatial_f
                )
                with torch.no_grad():
                    fine_delta = state[:, sd:sd + pd].detach() - pre_phys
                    fine_substep_energy_acc = fine_substep_energy_acc + fine_delta.float().pow(2).mean()
                    fine_phys_norm_acc = fine_phys_norm_acc + state[:, sd:sd + pd].detach().float().pow(2).mean()

            if return_trajectory:
                trajectory.append(state[:, sd:sd + pd])

            state = self._apply_tfc(
                state, gt_traj, i, tf_ratio, fluid_mask, ss_detach
            )

        n_outer = max(1, n_steps)
        n_fine_calls = max(1, n_outer * sub)
        coarse_avg = coarse_energy_acc / max(1, n_outer * len(ratios))
        fine_avg = fine_substep_energy_acc / n_fine_calls
        self._aux_metrics = {
            "coarse_update_energy": coarse_avg,
            "fine_substep_energy": fine_avg,
            "fine_to_coarse_energy_ratio": fine_avg / max(coarse_avg, 1e-12),
            "coarse_phys_norm_mean": coarse_phys_norm_acc / n_outer,
            "fine_phys_norm_mean": fine_phys_norm_acc / n_fine_calls,
            "fine_substeps": float(sub),
            "ratio": float(ratios[0]),
            "max_ratio": float(max(ratios)),
            "num_coarse_scales": float(len(ratios)),
            "dcs_enabled": float(delta_ctu_seq is not None),
            # 單步感受野直徑估計 ~= 4 * max_ratio。
            # 細路徑 3x3 加子步僅 ~2 px；粗 cell 3x3 更新後雙線性上採樣
            # 於比例 r 下覆蓋實際直徑 ~3r 至 4r，單位為像素，
            # 供 HC RF gate 比較 delta_ctu_i * L_char_px 使用。
            "single_step_rf_diameter_px_est": float(4 * max(ratios)),
        }

        if return_trajectory:
            return state, trajectory
        return state


# ---------------------------------------------------------------------------
# H4：HBlockRefineNCA
# ---------------------------------------------------------------------------

class HBlockRefineNCA(HBaseNCA):
    """H4 Block-Refinement NCA（PyTorch 密集遮罩版本）。Forward 介面與 :class:`TESNCA` 相同。

    根據粗網格增量幅度動態選取活躍 block，
    在活躍區域使用細網格 cell 輸出，其餘區域保留粗網格預測值。

    每個外層步驟：
      1. 粗網格 cell 在粗格網預測物理增量
      2. 將粗增量延伸至完整格網 → 非活躍區域預測
      3. 依粗增量平均幅度對 block 評分；選取前 k 個活躍 block（含 halo 擴展）
      4. 細網格 cell 預測完整格網更新；僅活躍 block 遮罩內的值被寫回；
         遮罩外使用非活躍的粗網格預測值

    兩個 cell 的輸出投影均以零初始化，確保步驟 0 的行為與平坦 baseline 相同（恆等映射）。

    未來 Code-4b 將提供 Taichi 稀疏後端以處理活躍 block stencil；
    本 class 為 Code-4 驗證用的密集遮罩參考實作。
    """

    def __init__(self, cfg: TESModelConfig) -> None:
        """初始化 HBlockRefineNCA。

        Args:
            cfg: TES 模型設定物件。

        Raises:
            ValueError: 若 ratios 為空或 active_ratio 超出 (0, 1] 範圍。
        """
        ms = cfg.multiscale
        n_scale_feats = SCALE_FEAT_DIM if ms.scale_condition else 0
        super().__init__(cfg, scale_feat_dim=n_scale_feats)

        if len(ms.ratios) < 1:
            raise ValueError("HBlockRefineNCA: multiscale.ratios must have >=1 entry")
        self._ratio: int = int(ms.ratios[0])
        self._block_size: int = int(ms.block_size)
        self._active_ratio: float = float(ms.active_ratio)
        self._scale_cond: bool = bool(ms.scale_condition)
        if not 0.0 < self._active_ratio <= 1.0:
            raise ValueError(
                f"HBlockRefineNCA: active_ratio must be in (0,1], got {self._active_ratio}"
            )

        self.coarse_cell: MarkovNCACell = MarkovNCACell(
            state_dim=cfg.state_dim,
            nca_channels=cfg.nca_channels,
            physics_dim=cfg.physics_dim,
            hidden_dim=cfg.hidden_dim,
            mrp_config=cfg.mrp,
        )
        self.fine_cell: MarkovNCACell = MarkovNCACell(
            state_dim=cfg.state_dim,
            nca_channels=cfg.nca_channels,
            physics_dim=cfg.physics_dim,
            hidden_dim=cfg.hidden_dim,
            mrp_config=cfg.mrp,
        )

    # ------------------------------------------------------------------
    def _encode_cond(
        self,
        cond_norm: torch.Tensor,
        delta_ctu_seq: Optional[torch.Tensor],
        cumul_ctu_norm_seq: Optional[torch.Tensor],
        step_idx: int,
        time_conditioned: bool,
        per_step_cumul: bool,
        level_id: float,
        ratio: int,
        gamma_g: Optional[torch.Tensor],
        beta_g: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """編碼每步條件向量，生成 FiLM 參數與空間 dt。

        Args:
            cond_norm: 正規化條件向量，shape [B, C]。
            delta_ctu_seq: 每步 delta CTU 序列，shape [B, T] 或 None。
            cumul_ctu_norm_seq: 累積 CTU 正規化序列，shape [B, T] 或 None。
            step_idx: 當前步驟索引。
            time_conditioned: 是否啟用時間條件。
            per_step_cumul: 是否逐步更新累積 CTU。
            level_id: 層級識別碼（float，供 scale_condition 使用）。
            ratio: 當前粗網格縮放比例。
            gamma_g: G_vec FiLM gamma，shape [B, C] 或 None。
            beta_g: G_vec FiLM beta，shape [B, C] 或 None。

        Returns:
            gamma: FiLM gamma 張量。
            beta: FiLM beta 張量。
            dt_spatial: 空間 dt 張量，shape [B, 1, 1, 1] 或 None。
        """
        b = cond_norm.shape[0]
        device = cond_norm.device
        cond_i = cond_norm
        dt_spatial: Optional[torch.Tensor] = None
        if time_conditioned and delta_ctu_seq is not None:
            dt_i = delta_ctu_seq[:, step_idx:step_idx + 1]
            cond_i = torch.cat([cond_norm, dt_i], dim=1)
            dt_spatial = dt_i[:, :, None, None]
        elif per_step_cumul and cumul_ctu_norm_seq is not None:
            cond_i = cond_norm.clone()
            cond_i[:, 4] = cumul_ctu_norm_seq[:, step_idx]
        if self._scale_cond:
            scale_feats = build_scale_features(
                level_id=level_id, ratio=ratio, dt_scale=1.0, ctu_scale=1.0,
                batch_size=b, device=device,
            )
            cond_i = torch.cat([cond_i, scale_feats], dim=1)
        gamma, beta = self.cond_encoder(cond_i)
        gamma, beta = self._add_gvec(gamma, beta, gamma_g, beta_g)
        return gamma, beta, dt_spatial

    # ------------------------------------------------------------------
    def forward(
        self,
        state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        n_steps: int,
        return_trajectory: bool = False,
        delta_ctu_seq: Union[torch.Tensor, None] = None,
        g_vec_norm: Union[torch.Tensor, None] = None,
        tf_ratio: float = 0.0,
        gt_traj: Union[torch.Tensor, None] = None,
        ss_detach: bool = False,
        cumul_ctu_norm_seq: Union[torch.Tensor, None] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """執行 Block-Refinement NCA 前向推論。

        Args:
            state: 初始狀態張量，shape [B, state_dim, H, W]。
            cond_norm: 正規化條件向量，shape [B, cond_dim]。
            fluid_mask: 流體遮罩，shape [B, 1, H, W]，1=流體，0=固體。
            n_steps: 外層推進步驟數。
            return_trajectory: 若 True，回傳物理場軌跡（每步一筆）。
            delta_ctu_seq: 每步 delta CTU 序列，shape [B, T] 或 None。
            g_vec_norm: 正規化幾何向量，shape [B, g_dim] 或 None。
            tf_ratio: Teacher forcing 比例，0.0 表示完全自回歸。
            gt_traj: Ground truth 軌跡，供 teacher forcing 使用。
            ss_detach: 是否對 scheduled sampling 進行 detach。
            cumul_ctu_norm_seq: 累積 CTU 正規化序列，shape [B, T] 或 None。

        Returns:
            若 return_trajectory=False，回傳最終 state 張量。
            若 return_trajectory=True，回傳 (state, trajectory) tuple。

        Raises:
            ValueError: 若空間尺寸小於 block_size。
        """
        cfg = self._cfg
        sd, pd, hd = cfg.static_dim, cfg.physics_dim, cfg.hidden_dim
        ratio = self._ratio
        bs = self._block_size
        h_full, w_full = state.shape[-2:]
        if h_full < bs or w_full < bs:
            raise ValueError(
                f"HBlockRefineNCA: spatial dims ({h_full},{w_full}) smaller than "
                f"block_size {bs}"
            )

        time_conditioned = delta_ctu_seq is not None
        per_step_cumul = (
            cumul_ctu_norm_seq is not None
            and cond_norm.shape[1] == 5
            and not time_conditioned
        )
        gamma_g, beta_g = self._compute_gvec_film(g_vec_norm)
        coarse_mask = downsample_mask(fluid_mask, ratio, threshold=0.5)

        trajectory: list[torch.Tensor] = []
        # 監控用純量，僅供 _aux_metrics 記錄，不參與梯度計算
        active_ratio_acc: float = 0.0

        for i in range(n_steps):
            # 粗網格更新
            gamma_c, beta_c, dt_spatial_c = self._encode_cond(
                cond_norm, delta_ctu_seq, cumul_ctu_norm_seq, i,
                time_conditioned, per_step_cumul,
                level_id=0.0, ratio=ratio,
                gamma_g=gamma_g, beta_g=beta_g,
            )
            coarse_state = downsample_state(state, ratio, mode="area")
            coarse_state_new = self.coarse_cell(
                coarse_state, gamma_c, beta_c, coarse_mask, dt_spatial_c
            )
            coarse_phys_delta = (
                coarse_state_new[:, sd:sd + pd] - coarse_state[:, sd:sd + pd]
            )
            coarse_phys_delta_up = upsample_state(
                coarse_phys_delta, (h_full, w_full), mode="bilinear"
            )

            # block 活躍度評分：上採樣粗增量的平均幅度
            with torch.no_grad():
                score = coarse_phys_delta_up.detach().float().abs().mean(dim=1, keepdim=True)
                block_mask = make_block_mask(score, bs, self._active_ratio)
                block_mask_halo = expand_block_halo(block_mask, halo_blocks=1)
                fine_mask = F.interpolate(
                    block_mask_halo, size=(h_full, w_full), mode="nearest"
                )
                active_ratio_acc += float(block_mask_halo.mean().item())

            # 完整格網細 cell（零初始化：初期增量極小）
            gamma_f, beta_f, dt_spatial_f = self._encode_cond(
                cond_norm, delta_ctu_seq, cumul_ctu_norm_seq, i,
                time_conditioned, per_step_cumul,
                level_id=1.0, ratio=1,
                gamma_g=gamma_g, beta_g=beta_g,
            )
            fine_state_new = self.fine_cell(
                state, gamma_f, beta_f, fluid_mask, dt_spatial_f
            )

            # 合併：非活躍區 → 僅使用粗增量，活躍區 → 細 cell 輸出
            inactive_phys = state[:, sd:sd + pd] + coarse_phys_delta_up
            active_phys = fine_state_new[:, sd:sd + pd]
            new_physics = fine_mask * active_phys + (1.0 - fine_mask) * inactive_phys
            new_physics = new_physics * fluid_mask

            if hd > 0:
                new_hidden = fine_state_new[:, sd + pd:]
                state = torch.cat(
                    [state[:, :sd], new_physics, new_hidden], dim=1
                )
            else:
                state = torch.cat([state[:, :sd], new_physics], dim=1)

            if return_trajectory:
                trajectory.append(state[:, sd:sd + pd])

            state = self._apply_tfc(
                state, gt_traj, i, tf_ratio, fluid_mask, ss_detach
            )

        n = max(1, n_steps)
        self._aux_metrics = {
            "active_block_ratio": active_ratio_acc / n,
        }

        if return_trajectory:
            return state, trajectory
        return state


# ---------------------------------------------------------------------------
# H5：HPODModalNCA
# ---------------------------------------------------------------------------

class HPODModalNCA(HBaseNCA):
    """H5 POD Modal NCA。Forward 介面與 :class:`TESNCA` 相同。

    以 POD（Proper Orthogonal Decomposition）模態基底在低維係數空間中
    進行狀態更新，再重建至全解析度物理場。

    每個外層步驟：
      1. 透過 1×1 卷積將當前物理場投影至純量代理場（scalar proxy）
      2. 將純量場投影至 POD 基底 → 模態係數
      3. 以小型 MLP（以 FiLM γ 向量作為條件）更新模態係數
      4. 以更新後的模態係數重建純量場
      5. 透過 1×1 卷積將純量場解碼為 9 通道物理殘差
      6. 將殘差加入物理場，hidden 保持不變（無細殘差分支）

    所有輸出層均以零初始化，確保步驟 0 的行為等價於平坦 baseline（恆等映射）。
    若 ``cfg.pod_basis_path`` 指向已儲存的基底，則載入之；
    否則在第一次 forward 時延遲建立隨機正交規一基底（足以應付合成煙霧測試）。
    """

    def __init__(self, cfg: TESModelConfig) -> None:
        """初始化 HPODModalNCA。

        Args:
            cfg: TES 模型設定物件。
        """
        super().__init__(cfg, scale_feat_dim=0)
        self._n_modes: int = int(cfg.pod_modes)
        self._basis_path: str = str(cfg.pod_basis_path or "")

        # 物理場 → 純量代理（零初始化：初始純量為 0）
        self.physics_to_scalar: nn.Conv2d = nn.Conv2d(
            cfg.physics_dim, 1, kernel_size=1, bias=True
        )
        nn.init.zeros_(self.physics_to_scalar.weight)
        nn.init.zeros_(self.physics_to_scalar.bias)

        # 模態係數更新 MLP：[coeffs ‖ gamma] → Δcoeffs
        self.coeff_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(self._n_modes + cfg.nca_channels, cfg.encoder_hidden),
            nn.SiLU(),
            nn.Linear(cfg.encoder_hidden, self._n_modes),
        )
        # 最後一層零初始化 → 初始 Δcoeffs = 0
        nn.init.zeros_(self.coeff_mlp[-1].weight)
        nn.init.zeros_(self.coeff_mlp[-1].bias)

        # 純量重建 → 物理殘差（零初始化）
        self.scalar_to_physics: nn.Conv2d = nn.Conv2d(
            1, cfg.physics_dim, kernel_size=1, bias=True
        )
        nn.init.zeros_(self.scalar_to_physics.weight)
        nn.init.zeros_(self.scalar_to_physics.bias)

        # POD 基底 tensor，第一次 forward 時延遲初始化
        self._basis_mean: torch.Tensor | None = None
        self._basis_modes: torch.Tensor | None = None
        self._basis_source: str = "uninitialised"

    # ------------------------------------------------------------------
    def _ensure_basis(
        self,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """確保 POD 基底已載入且與指定空間尺寸、裝置、資料型別對齊。

        若基底尚未初始化或尺寸不符，則嘗試從 ``pod_basis_path`` 載入；
        路徑不存在時退而建立隨機正交規一基底（固定 seed，供煙霧測試使用）。

        Args:
            h: 目標特徵圖高度（像素）。
            w: 目標特徵圖寬度（像素）。
            device: 目標裝置。
            dtype: 目標資料型別。

        Raises:
            ValueError: 若已載入基底的空間尺寸或模態數與要求不符。
        """
        if (
            self._basis_mean is not None
            and self._basis_modes is not None
            and self._basis_mean.shape == (h, w)
            and self._basis_modes.shape[0] == self._n_modes
        ):
            # 確保裝置與資料型別對齊（例如 .cuda() 後需重新繫結）
            self._basis_mean = self._basis_mean.to(device=device, dtype=dtype)
            self._basis_modes = self._basis_modes.to(device=device, dtype=dtype)
            return

        if self._basis_path and Path(self._basis_path).exists():
            from ...eval.pod_metrics import load_pod_basis

            data = load_pod_basis(self._basis_path)
            mean = data["mean"]
            modes = data["modes"]
            if mean.shape != (h, w):
                raise ValueError(
                    f"HPODModalNCA: POD basis spatial shape {tuple(mean.shape)} "
                    f"does not match field shape ({h}, {w})"
                )
            if modes.shape[0] < self._n_modes:
                raise ValueError(
                    f"HPODModalNCA: POD basis has {modes.shape[0]} modes, "
                    f"need {self._n_modes}"
                )
            self._basis_mean = mean.to(device=device, dtype=dtype)
            self._basis_modes = modes[: self._n_modes].to(device=device, dtype=dtype)
            self._basis_source = f"file:{self._basis_path}"
            return

        # 後備路徑：固定 seed 隨機正交規一基底，足以用於合成煙霧測試
        flat = h * w
        gen = torch.Generator(device="cpu").manual_seed(0)
        rand = torch.randn(flat, self._n_modes, generator=gen)
        q, _ = torch.linalg.qr(rand)  # [flat, n_modes]
        modes = q.t().reshape(self._n_modes, h, w)
        self._basis_mean = torch.zeros(h, w, device=device, dtype=dtype)
        self._basis_modes = modes.to(device=device, dtype=dtype)
        self._basis_source = "random_orthonormal"

    # ------------------------------------------------------------------
    def _project_to_coeffs(self, scalar: torch.Tensor) -> torch.Tensor:
        """將純量場投影至 POD 模態係數。

        Args:
            scalar: 純量場張量，shape [B, 1, H, W]。

        Returns:
            模態係數張量，shape [B, n_modes]。
        """
        b, _, h, w = scalar.shape
        assert self._basis_mean is not None and self._basis_modes is not None
        centered = scalar.squeeze(1) - self._basis_mean.unsqueeze(0)
        modes_flat = self._basis_modes.reshape(self._n_modes, h * w)
        return centered.reshape(b, h * w) @ modes_flat.t()

    def _reconstruct_scalar(
        self,
        coeffs: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """從 POD 模態係數重建純量場。

        Args:
            coeffs: 模態係數張量，shape [B, n_modes]。
            h: 輸出特徵圖高度。
            w: 輸出特徵圖寬度。

        Returns:
            重建後的純量場，shape [B, 1, H, W]。
        """
        assert self._basis_mean is not None and self._basis_modes is not None
        modes_flat = self._basis_modes.reshape(self._n_modes, h * w)
        flat = coeffs @ modes_flat + self._basis_mean.reshape(1, h * w)
        return flat.reshape(coeffs.shape[0], 1, h, w)

    def _encode_per_step(
        self,
        cond_norm: torch.Tensor,
        delta_ctu_seq: Optional[torch.Tensor],
        cumul_ctu_norm_seq: Optional[torch.Tensor],
        step_idx: int,
        time_conditioned: bool,
        per_step_cumul: bool,
        gamma_g: Optional[torch.Tensor],
        beta_g: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """編碼每步條件向量，生成 FiLM gamma 與 beta。

        Args:
            cond_norm: 正規化條件向量，shape [B, C]。
            delta_ctu_seq: 每步 delta CTU 序列，shape [B, T] 或 None。
            cumul_ctu_norm_seq: 累積 CTU 正規化序列，shape [B, T] 或 None。
            step_idx: 當前步驟索引。
            time_conditioned: 是否啟用時間條件。
            per_step_cumul: 是否逐步更新累積 CTU。
            gamma_g: G_vec FiLM gamma，shape [B, C] 或 None。
            beta_g: G_vec FiLM beta，shape [B, C] 或 None。

        Returns:
            gamma: FiLM gamma 張量。
            beta: FiLM beta 張量。
        """
        cond_i = cond_norm
        if time_conditioned and delta_ctu_seq is not None:
            dt_i = delta_ctu_seq[:, step_idx:step_idx + 1]
            cond_i = torch.cat([cond_norm, dt_i], dim=1)
        elif per_step_cumul and cumul_ctu_norm_seq is not None:
            cond_i = cond_norm.clone()
            cond_i[:, 4] = cumul_ctu_norm_seq[:, step_idx]
        gamma, beta = self.cond_encoder(cond_i)
        gamma, beta = self._add_gvec(gamma, beta, gamma_g, beta_g)
        return gamma, beta

    # ------------------------------------------------------------------
    def forward(
        self,
        state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        n_steps: int,
        return_trajectory: bool = False,
        delta_ctu_seq: Union[torch.Tensor, None] = None,
        g_vec_norm: Union[torch.Tensor, None] = None,
        tf_ratio: float = 0.0,
        gt_traj: Union[torch.Tensor, None] = None,
        ss_detach: bool = False,
        cumul_ctu_norm_seq: Union[torch.Tensor, None] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """執行 POD Modal NCA 前向推論。

        Args:
            state: 初始狀態張量，shape [B, state_dim, H, W]。
            cond_norm: 正規化條件向量，shape [B, cond_dim]。
            fluid_mask: 流體遮罩，shape [B, 1, H, W]，1=流體，0=固體。
            n_steps: 外層推進步驟數。
            return_trajectory: 若 True，回傳物理場軌跡（每步一筆）。
            delta_ctu_seq: 每步 delta CTU 序列，shape [B, T] 或 None。
            g_vec_norm: 正規化幾何向量，shape [B, g_dim] 或 None。
            tf_ratio: Teacher forcing 比例，0.0 表示完全自回歸。
            gt_traj: Ground truth 軌跡，供 teacher forcing 使用。
            ss_detach: 是否對 scheduled sampling 進行 detach。
            cumul_ctu_norm_seq: 累積 CTU 正規化序列，shape [B, T] 或 None。

        Returns:
            若 return_trajectory=False，回傳最終 state 張量。
            若 return_trajectory=True，回傳 (state, trajectory) tuple。
        """
        cfg = self._cfg
        sd, pd = cfg.static_dim, cfg.physics_dim
        h, w = state.shape[-2:]
        self._ensure_basis(h, w, state.device, state.dtype)

        time_conditioned = delta_ctu_seq is not None
        per_step_cumul = (
            cumul_ctu_norm_seq is not None
            and cond_norm.shape[1] == 5
            and not time_conditioned
        )
        gamma_g, beta_g = self._compute_gvec_film(g_vec_norm)

        trajectory: list[torch.Tensor] = []
        # 以 Tensor 累積模態係數 L2，最後統一取純量，減少 GPU→CPU 同步次數
        modal_coeff_l2_acc: torch.Tensor = state.new_zeros(())

        for i in range(n_steps):
            physics_orig = state[:, sd:sd + pd]
            scalar = self.physics_to_scalar(physics_orig)
            coeffs = self._project_to_coeffs(scalar)

            gamma, _beta = self._encode_per_step(
                cond_norm,
                delta_ctu_seq,
                cumul_ctu_norm_seq,
                i,
                time_conditioned,
                per_step_cumul,
                gamma_g,
                beta_g,
            )
            coeff_in = torch.cat([coeffs, gamma], dim=1)
            coeffs_new = coeffs + self.coeff_mlp(coeff_in)

            scalar_new = self._reconstruct_scalar(coeffs_new, h, w)
            phys_residual = self.scalar_to_physics(scalar_new)
            new_physics = (physics_orig + phys_residual) * fluid_mask

            if cfg.hidden_dim > 0:
                state = torch.cat(
                    [state[:, :sd], new_physics, state[:, sd + pd:]], dim=1
                )
            else:
                state = torch.cat([state[:, :sd], new_physics], dim=1)

            if return_trajectory:
                trajectory.append(state[:, sd:sd + pd])

            with torch.no_grad():
                modal_coeff_l2_acc = (
                    modal_coeff_l2_acc
                    + coeffs_new.detach().float().pow(2).mean()
                )

            state = self._apply_tfc(
                state, gt_traj, i, tf_ratio, fluid_mask, ss_detach
            )

        n = max(1, n_steps)
        # 迴圈結束後僅呼叫一次 .item()，避免每步 GPU→CPU 同步
        self._aux_metrics = {
            "modal_coeff_l2_mean": float(modal_coeff_l2_acc.item()) / n,
        }

        if return_trajectory:
            return state, trajectory
        return state


# ---------------------------------------------------------------------------
# 公開介面
# ---------------------------------------------------------------------------

__all__ = [
    "HSharedScaleNCA",
    "HLapResNCA",
    "HParentLeafNCA",
    "HMultirateNCA",
    "HBlockRefineNCA",
    "HPODModalNCA",
]
