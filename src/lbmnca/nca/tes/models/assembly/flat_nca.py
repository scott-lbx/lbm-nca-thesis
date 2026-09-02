"""TESNCA — TES-NCA 頂層模型，整合 ConditionEncoder + MarkovNCACell。

組裝層版本：import 路徑指向 conditioning／update 子模組。
原始邏輯與算法行為完全保留。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...config import TESModelConfig
from ..conditioning.cond_encoder import TESConditionEncoder
from ..conditioning.gvec_encoder import GVecEncoder
from ..markov_nca_cell import MarkovNCACell


class TESNCA(nn.Module):
    """TES-NCA 頂層模型。

    整合流程（標準模式）：
        1. TESConditionEncoder(cond_norm) → (γ_p, β_p)  [每次 forward 計算一次]
        2. MarkovNCACell(state, γ, β, fluid_mask)  [重複 n_steps 次]

    Dual-pathway 模式（cfg.g_vec_dim > 0）：
        1. GVecEncoder(g_vec_norm) → (γ_g, β_g)  [每次 forward 計算一次，per-case]
        2. TESConditionEncoder(cond_norm[+dt_i]) → (γ_p, β_p)  [time_conditioned 時每步]
        3. γ = γ_p + γ_g，β = β_p + β_g（加法 FiLM）
        4. MarkovNCACell(state, γ, β, fluid_mask)

    狀態初始化（`init_state`）：
        Cold-start: physics = zeros，hidden = zeros（hidden 為 within-trajectory 內部記憶，
            非跨 trajectory 狀態，因此每次 init_state 均重置為零）。
        Warm-start: physics = 前版穩態場（需搭配 `apply_ic_mask`），hidden 仍初始化為零。
        當 cfg.hidden_dim > 0，狀態形狀為 [B, 4+9+hidden_dim, H, W]；
        hidden_dim=0 時行為與 baseline 13ch 完全相同。

    IC masking（`apply_ic_mask`）：
        Warm-start IC 中，目標幾何的固體區必須歸零，
        以確保初始狀態物理合法。

    rate_mode（arm-F vs arm-FΔ 消融軸）：
        ``"fdelta"``（default，現況／arm-FΔ）：time_conditioned 時每步把 per-step
            ΔCTU 附加進 cond 編碼（cond = concat(cond_norm, dt_i)），且 NCA cell
            輸出 rate × Δt（explicit physical scaling）。行為與既往完全相同。
        ``"f"``（arm-F，rate=f(x)）：模型輸出與 Δ **無關** 的 rate
            —— cond 編碼 **不** 附加 per-step ΔCTU（步長不變的純條件），
            但仍由 NCA cell 外部 × Δt 積分（Euler 連續時間視角）。
        以 ``self.rate_mode = getattr(cfg, "rate_mode", "fdelta")`` 設定，
        cfg 未定義該欄位時安全退回 "fdelta"，default 行為不變。

    ⚠ 此 arm-F flag 僅在 flat ``TESNCA`` 的 forward 內生效。H 系列
        （``h_multirate`` → ``h3t_streamcollide`` → ``operator_gated_streamcollide``，
        即現行 active arch）的 cond-encode 在 ``h_multirate_nca._encode_film``
        進行（見該檔 ~L104-107 的 ``cond_i[:, 17] = _normalize_delta_ctu(dt_i)``），
        **不經過本檔 forward**。

    Args:
        cfg: TESModelConfig，包含所有架構超參數。
    """

    def __init__(self, cfg: TESModelConfig) -> None:
        super().__init__()
        self._cfg = cfg
        # arm-F / arm-FΔ 切換；cfg 未定義時安全退回 "fdelta"（default 不變）。
        self.rate_mode: str = cfg.rate_mode
        self.cond_encoder = TESConditionEncoder(
            cond_dim=cfg.cond_dim,
            hidden_dim=cfg.encoder_hidden,
            out_dim=cfg.nca_channels,
        )
        self._gvec_mode: str = cfg.gvec_mode
        self.g_encoder: GVecEncoder | None = None
        self.alpha_g: nn.Parameter | None = None
        if cfg.gvec_mode in ("additive", "gated") and cfg.g_vec_dim > 0:
            self.g_encoder = GVecEncoder(
                g_vec_dim=cfg.g_vec_dim,
                hidden_dim=cfg.encoder_hidden,
                out_dim=cfg.nca_channels,
            )
        if cfg.gvec_mode == "gated":
            # alpha_g init=0：初期 G_vec 貢獻為零，訓練後自然成長
            self.alpha_g = nn.Parameter(torch.zeros(1))
        self.nca_cell = MarkovNCACell(
            state_dim=cfg.state_dim,
            nca_channels=cfg.nca_channels,
            physics_dim=cfg.physics_dim,
            hidden_dim=cfg.hidden_dim,
            mrp_config=cfg.mrp,
        )

    # ------------------------------------------------------------------
    # 狀態操作工具
    # ------------------------------------------------------------------

    def init_state(
        self,
        static: torch.Tensor,
        ic_physics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """建立初始狀態 [B, state_dim, H, W]。

        hidden 通道（若 cfg.hidden_dim > 0）一律初始化為零，無論 cold/warm start。
        hidden 為 within-trajectory 內部記憶，非跨 trajectory 狀態。

        Args:
            static: [B, 4, H, W] 靜態通道（已正規化）。
            ic_physics: [B, 9, H, W] warm-start 物理初值；
                        None 表示 cold-start（physics=0）。

        Returns:
            state: [B, state_dim, H, W]。
        """
        B, _, H, W = static.shape
        if ic_physics is None:
            physics = torch.zeros(
                B, self._cfg.physics_dim, H, W,
                device=static.device, dtype=static.dtype,
            )
        else:
            physics = ic_physics

        if self._cfg.hidden_dim > 0:
            hidden = torch.zeros(
                B, self._cfg.hidden_dim, H, W,
                device=static.device, dtype=static.dtype,
            )
            return torch.cat([static, physics, hidden], dim=1)
        return torch.cat([static, physics], dim=1)

    def apply_ic_mask(
        self,
        state: torch.Tensor,
        solid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """將 warm-start IC 中目標幾何的固體區物理通道歸零。

        IC masking 必要性：warm-start IC 來自前版穩態場（幾何 A），
        若幾何 B 的固體位置與 A 不同，A 的物理場在 B 的固體區不合法，
        必須清零才能保持物理初值合法。

        Args:
            state: [B, state_dim, H, W]。
            solid_mask: [B, 1, H, W] bool，True 表示固體。

        Returns:
            state: [B, state_dim, H, W]，固體區 physics+hidden=0，static 不變。
        """
        static_dim: int = self._cfg.static_dim
        dynamic = state[:, static_dim:].clone()   # physics + hidden
        dynamic_dim: int = self._cfg.physics_dim + self._cfg.hidden_dim
        mask_expanded = solid_mask.expand(-1, dynamic_dim, -1, -1)
        dynamic[mask_expanded] = 0.0
        return torch.cat([state[:, :static_dim], dynamic], dim=1)

    def get_fluid_mask(self, static: torch.Tensor) -> torch.Tensor:
        """從 SDF_norm (C0) 推導流體遮罩。

        v3 tanh_lchar: SDF_norm = tanh(SDF_raw / L_char)，tanh(0)=0 = 障礙物表面。
        SDF_norm > 0 → 流體，SDF_norm ≤ 0 → 固體（threshold=0.0 精確）。

        Args:
            static: [B, 4, H, W] 靜態通道，C0 = SDF_norm。

        Returns:
            fluid_mask: [B, 1, H, W] float，流體=1，固體=0。
        """
        return (static[:, 0:1] > 0.0).float()

    # ------------------------------------------------------------------
    # 前向傳播
    # ------------------------------------------------------------------

    def forward(
        self,
        state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        n_steps: int,
        return_trajectory: bool = False,
        delta_ctu_seq: torch.Tensor | None = None,
        g_vec_norm: torch.Tensor | None = None,
        tf_ratio: float = 0.0,
        gt_traj: torch.Tensor | None = None,
        ss_detach: bool = False,
        cumul_ctu_norm_seq: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """執行 n_steps 步 NCA 動力算子推進。

        Args:
            state: [B, state_dim, H, W] 初始狀態。
            cond_norm: [B, cond_dim] 已正規化條件向量（p_vec z-score，或拼接 G_vec）。
            fluid_mask: [B, 1, H, W] 流體遮罩（float）。
            n_steps: rollout 步數（k）。
            return_trajectory: 若 True，回傳每步的 physics 快照列表。
            delta_ctu_seq: [B, k] float 或 None。time_conditioned 模式下每步的 Δt（CTU）。
                若提供，行為依 ``self.rate_mode``：
                ``"fdelta"``（default）：
                    ① 每步重新編碼 cond = concat(cond_norm, delta_ctu_i)（per-step FiLM，Δ 進 cond）
                    ② NCA cell 輸出 rate × delta_ctu（explicit physical scaling）
                ``"f"``（arm-F，rate=f(x)）：
                    ① cond 不附加 per-step Δ —— 改 append 零佔位（維持 encoder 輸入寬度
                       一致，但 rate 與 Δ 無關）
                    ② NCA cell 仍 × delta_ctu 外部積分（Euler，步長語意不變）
            g_vec_norm: [B, 14] 已正規化 G_vec，dual-pathway 模式（cfg.g_vec_dim>0）時傳入。
                GVecEncoder 只計算一次，輸出 (γ_g, β_g) 與 p_vec 路徑合併（additive 或 gated）。
            tf_ratio: Teacher Forcing 比例 [0.0, 1.0]。>0 時需傳入 gt_traj。
                每步 rollout 後，以 Bernoulli(tf_ratio) 機率將 physics channels 替換為 GT，
                作為下一步 IC；trajectory 記錄替換前的 model 輸出（不影響 loss target）。
            gt_traj: [B, n_steps, physics_dim, H, W] GT physics trajectory，tf_ratio>0 時必傳。
            ss_detach: Scheduled Sampling 模式。True 時 free-rollout 路徑（use_gt=False）
                的 physics channels 在作為下一步 IC 前 detach()，阻斷跨步梯度流。
                此為 DAgger 屬性的關鍵：每步各自接受梯度，但跨步 state 不傳梯度。
                僅當 tf_ratio > 0 且 gt_traj is not None 時生效。
            cumul_ctu_norm_seq: [B, k] float 或 None。M1A cond_5D per-step phase awareness，
                cond_dim==5 且非 time_conditioned 時生效，每步替換 cumul_ctu_log（index 4）。

        Returns:
            若 return_trajectory=False：final_state [B, state_dim, H, W]
            若 return_trajectory=True：(final_state, trajectory)
                trajectory: list[Tensor [B, 9, H, W]]，長度 n_steps。
        """
        time_conditioned: bool = delta_ctu_seq is not None
        # M1A cond_5D per-step phase awareness：每步用 cumul_ctu[i] 重編碼 cond
        # 條件：cond_dim==5（cond_5D 模式）且 cumul_ctu_norm_seq 提供，且非 time_conditioned
        per_step_cumul: bool = (
            cumul_ctu_norm_seq is not None
            and cond_norm.shape[1] == 5
            and not time_conditioned
        )
        static_dim: int = self._cfg.static_dim
        trajectory: list[torch.Tensor] = []

        # Dual-pathway: G_vec 幾何先驗，每 rollout 只算一次
        gamma_g: torch.Tensor | None = None
        beta_g: torch.Tensor | None = None
        if self.g_encoder is not None and g_vec_norm is not None:
            gamma_g, beta_g = self.g_encoder(g_vec_norm)
            if self._gvec_mode == "gated" and self.alpha_g is not None:
                # 殘差門控：alpha_g init≈0，初期幾乎無 G_vec 影響
                gamma_g = self.alpha_g * gamma_g
                beta_g = self.alpha_g * beta_g

        # 宣告 gamma/beta 供 loop 使用（time_conditioned/per_step_cumul 時每步覆寫）
        gamma: torch.Tensor
        beta: torch.Tensor

        if not time_conditioned and not per_step_cumul:
            # 標準模式：條件編碼一次，整個 rollout 共用
            gamma, beta = self.cond_encoder(cond_norm)
            if gamma_g is not None and beta_g is not None:
                gamma = gamma + gamma_g
                beta = beta + beta_g

        for i in range(n_steps):
            if time_conditioned:
                # 每步用 (cond_norm ‖ Δ-feature) 重新計算 p_vec FiLM
                dt_i = delta_ctu_seq[:, i:i + 1]                    # [B, 1]
                if self.rate_mode == "f":
                    # arm-F：rate=f(x)，Δ 不進 cond。append 零佔位以維持
                    # encoder 輸入寬度一致（cond_dim+1），但 rate 與 Δ 無關。
                    cond_feat = torch.zeros_like(dt_i)
                else:
                    # arm-FΔ（default）：Δ 進 cond（per-step time-conditioned FiLM）
                    cond_feat = dt_i
                cond_i = torch.cat([cond_norm, cond_feat], dim=1)   # [B, cond_dim+1]
                gamma, beta = self.cond_encoder(cond_i)
                if gamma_g is not None and beta_g is not None:
                    gamma = gamma + gamma_g
                    beta = beta + beta_g
                dt_spatial = dt_i[:, :, None, None]                  # [B, 1, 1, 1]
                state = self.nca_cell(state, gamma, beta, fluid_mask, dt_spatial)
            elif per_step_cumul:
                # M1A: per-step cond_5D，替換 cumul_ctu_log（index 4）為當步累積值
                cond_i = cond_norm.clone()
                cond_i[:, 4] = cumul_ctu_norm_seq[:, i]
                gamma, beta = self.cond_encoder(cond_i)
                if gamma_g is not None and beta_g is not None:
                    gamma = gamma + gamma_g
                    beta = beta + beta_g
                state = self.nca_cell(state, gamma, beta, fluid_mask)
            else:
                state = self.nca_cell(state, gamma, beta, fluid_mask)

            # 記錄 trajectory（TFC 替換前，即 model 實際輸出）
            if return_trajectory:
                trajectory.append(state[:, static_dim: static_dim + self._cfg.physics_dim])

            # TFC / SS：用 GT physics 替換下一步 IC（不影響 loss target，只改 IC）
            # ss_detach=True 時，free-rollout 路徑 physics detach()，阻斷跨步梯度。
            if tf_ratio > 0.0 and gt_traj is not None and i < gt_traj.shape[1]:
                _gt_i = gt_traj[:, i]                                    # [B, pd, H, W]
                _use_gt = torch.bernoulli(
                    torch.full((state.shape[0],), tf_ratio, device=state.device)
                ).bool()                                                  # [B]
                _use_gt_exp = _use_gt[:, None, None, None].expand_as(_gt_i)
                _sd: int = self._cfg.static_dim
                _pd: int = self._cfg.physics_dim
                _phys_model = state[:, _sd:_sd + _pd]
                if ss_detach:
                    _phys_model = _phys_model.detach()
                _phys_new = torch.where(_use_gt_exp, _gt_i, _phys_model)
                _phys_new = _phys_new * fluid_mask   # zero solid-region values injected by TF
                if self._cfg.hidden_dim > 0:
                    state = torch.cat(
                        [state[:, :_sd], _phys_new, state[:, _sd + _pd:]], dim=1
                    )
                else:
                    state = torch.cat([state[:, :_sd], _phys_new], dim=1)

        if return_trajectory:
            return state, trajectory
        return state


__all__ = ["TESNCA"]
