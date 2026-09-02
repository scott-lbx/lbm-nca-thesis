"""H-series 分層 NCA 模型的共用基底類別（組裝層版本）。

所有 H-series 模型（HCoarseNCA、HFineNCA 等）均繼承此類別，
提供狀態初始化、IC 遮罩套用、流體遮罩取得、G_vec FiLM 編碼，
以及 Teacher-Forcing Curriculum（TFC）/Scheduled Sampling（SS）機制。
子類別只需實作 ``forward``。

組裝層版本：import 路徑指向 conditioning 子模組。
原始邏輯與算法行為完全保留。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ...config import TESModelConfig
from ..conditioning.cond_encoder import TESConditionEncoder
from ..conditioning.gvec_encoder import GVecEncoder
from ..transport.admissibility import PhysicsAdmissibility


class HBaseNCA(nn.Module):
    """所有 H-series 模型的基底類別，子類別實作 ``forward``。

    封裝每個 H-series 模型重複使用的功能：
    - TESConditionEncoder（含可選尺度特徵維度擴充）
    - GVecEncoder 與 gated/additive FiLM 融合
    - 狀態向量初始化（static + physics + hidden）
    - IC 遮罩套用（固體邊界歸零）
    - TFC/SS 課程取樣

    Args:
        cfg: 完整模型配置物件。
        scale_feat_dim: 子類別額外拼接至 ``cond_norm`` 的尺度條件特徵維度數，
            TESConditionEncoder 輸入維度將設為 ``cfg.cond_dim + scale_feat_dim``。
    """

    def __init__(self, cfg: TESModelConfig, scale_feat_dim: int = 0) -> None:
        super().__init__()
        self._cfg = cfg
        self._scale_feat_dim = int(scale_feat_dim)
        self.cond_encoder = TESConditionEncoder(
            cond_dim=cfg.cond_dim + self._scale_feat_dim,
            hidden_dim=cfg.encoder_hidden,
            out_dim=cfg.nca_channels,
        )
        self._gvec_mode = cfg.gvec_mode
        self.g_encoder: GVecEncoder | None = None
        self.alpha_g: nn.Parameter | None = None
        if cfg.gvec_mode in ("additive", "gated") and cfg.g_vec_dim > 0:
            self.g_encoder = GVecEncoder(
                g_vec_dim=cfg.g_vec_dim,
                hidden_dim=cfg.encoder_hidden,
                out_dim=cfg.nca_channels,
            )
        if cfg.gvec_mode == "gated":
            self.alpha_g = nn.Parameter(torch.zeros(1))

        # 物理可容許性投影。cfg.phys_adm_enabled=False 時為恆等（既有 ckpt 零行為變更）。
        # 建在共用基底而非個別 arch：所有 H-series 與 baseline_* 共享同一個算子，
        # 消融臂之間才具可比性。
        self.phys_adm = PhysicsAdmissibility(cfg)

        # 狀態空間閉包半徑（0 = 停用）。見 TESModelConfig.state_clip。
        self.state_clip: float = float(cfg.state_clip)

        self._aux_metrics: dict[str, float] = {}
        self._collect_aux_metrics = True

    # ---- 狀態工具方法（與 TESNCA 對稱） --------------------------------

    def init_state(
        self,
        static: torch.Tensor,
        ic_physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """初始化 NCA 狀態向量（static ++ physics ++ hidden）。

        Args:
            static: 靜態通道張量，shape ``(B, static_dim, H, W)``。
            ic_physics: 初始物理通道張量，shape ``(B, physics_dim, H, W)``；
                為 ``None`` 時填零初始化。

        Returns:
            初始狀態張量，shape ``(B, static_dim + physics_dim [+ hidden_dim], H, W)``。
        """
        cfg = self._cfg
        b, _, h, w = static.shape
        if ic_physics is None:
            physics = torch.zeros(
                b, cfg.physics_dim, h, w, device=static.device, dtype=static.dtype
            )
        else:
            physics = ic_physics
        if cfg.hidden_dim > 0:
            hidden = torch.zeros(
                b, cfg.hidden_dim, h, w, device=static.device, dtype=static.dtype
            )
            return torch.cat([static, physics, hidden], dim=1)
        return torch.cat([static, physics], dim=1)

    def apply_ic_mask(
        self, state: torch.Tensor, solid_mask: torch.Tensor
    ) -> torch.Tensor:
        """對固體邊界套用 IC 遮罩，將固體區域的動態通道歸零。

        ★保證範圍：本函式**只做一件事**——把
        `solid_mask` 標記處的 dynamic 通道設為 0。它
          (a) 不修復「舊幾何處已移除的固體」留下的 ghost 尾流；
          (b) 不對新舊固體交界做任何擴張處理——ghost 修復依賴呼叫端傳入
              `spatial_ops.handoff_reset_mask(..., dilate_px)`，而 canonical 的
              `mean_chain_handoff_dilate_px = 0` 使該擴張退化為無作用；
          (c) 不使 IC 滿足任何物理條件（散度、邊界條件皆未處理）。
        因此**不得**據本函式宣稱「暖啟動 IC 已物理合法化」。

        Args:
            state: 完整狀態張量，shape ``(B, total_dim, H, W)``。
            solid_mask: 布林遮罩，shape ``(B, 1, H, W)``，True 代表固體格點。

        Returns:
            套用遮罩後的狀態張量，shape 與輸入相同。
        """
        cfg = self._cfg
        static_dim = cfg.static_dim
        dynamic = state[:, static_dim:].clone()
        dynamic_dim = cfg.physics_dim + cfg.hidden_dim
        mask_expanded = solid_mask.expand(-1, dynamic_dim, -1, -1)
        dynamic[mask_expanded] = 0.0
        return torch.cat([state[:, :static_dim], dynamic], dim=1)

    def get_fluid_mask(self, static: torch.Tensor) -> torch.Tensor:
        """從靜態通道第 0 軸萃取流體遮罩（SDF > 0 為流體）。

        Args:
            static: 靜態通道張量，shape ``(B, static_dim, H, W)``。

        Returns:
            流體遮罩（0/1 float），shape ``(B, 1, H, W)``。
        """
        return (static[:, 0:1] > 0.0).float()

    def get_aux_metrics(self) -> dict[str, float]:
        """取得用於 trainer 記錄的輔助診斷指標（如殘差能量）。

        Returns:
            指標名稱到純量 float 的字典；無指標時回傳空字典。
        """
        out: dict[str, float] = {}
        for key, val in self._aux_metrics.items():
            if torch.is_tensor(val):
                out[key] = float(val.detach().float().cpu().item())
            else:
                out[key] = float(val)
        return out

    def set_aux_metrics_enabled(self, enabled: bool) -> bool:
        """切換輔助診斷指標收集開關，回傳切換前的狀態。

        Args:
            enabled: ``True`` 啟用收集，``False`` 停用。

        Returns:
            切換前的啟用狀態。
        """
        prev = bool(getattr(self, "_collect_aux_metrics", True))
        self._collect_aux_metrics = bool(enabled)
        return prev

    # ---- 私有輔助方法 ---------------------------------------------------

    def _compute_gvec_film(
        self, g_vec_norm: Optional[torch.Tensor]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """計算 G_vec FiLM 調制參數（gamma_g, beta_g）。

        當 ``g_encoder`` 未初始化或 ``g_vec_norm`` 為 ``None`` 時回傳 ``(None, None)``。
        gated 模式下以可學習純量 ``alpha_g`` 縮放輸出。

        Args:
            g_vec_norm: 正規化後的幾何向量，shape ``(B, g_vec_dim)``；
                或 ``None`` 表示不使用 G_vec 條件。

        Returns:
            ``(gamma_g, beta_g)`` 各為 shape ``(B, nca_channels)`` 的張量，
            或 ``(None, None)``。
        """
        if self.g_encoder is None or g_vec_norm is None:
            return None, None
        gamma_g, beta_g = self.g_encoder(g_vec_norm)
        if self._gvec_mode == "gated" and self.alpha_g is not None:
            gamma_g = self.alpha_g * gamma_g
            beta_g = self.alpha_g * beta_g
        return gamma_g, beta_g

    @staticmethod
    def _add_gvec(
        gamma: torch.Tensor,
        beta: torch.Tensor,
        gamma_g: Optional[torch.Tensor],
        beta_g: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """將 G_vec FiLM 參數疊加至條件編碼器的 gamma/beta。

        ``gamma_g`` 或 ``beta_g`` 為 ``None`` 時直接回傳原值（跳過 G_vec 融合）。

        Args:
            gamma: 條件編碼器輸出的 FiLM scale，shape ``(B, nca_channels)``。
            beta: 條件編碼器輸出的 FiLM shift，shape ``(B, nca_channels)``。
            gamma_g: G_vec FiLM scale，shape ``(B, nca_channels)`` 或 ``None``。
            beta_g: G_vec FiLM shift，shape ``(B, nca_channels)`` 或 ``None``。

        Returns:
            融合後的 ``(gamma, beta)``，shape 與輸入相同。
        """
        if gamma_g is None or beta_g is None:
            return gamma, beta
        return gamma + gamma_g, beta + beta_g

    def _apply_tfc(
        self,
        state: torch.Tensor,
        gt_traj: Optional[torch.Tensor],
        i: int,
        tf_ratio: float,
        fluid_mask: torch.Tensor,
        ss_detach: bool,
    ) -> torch.Tensor:
        """依 TFC/SS 課程比例，以 GT 替換下一步 IC 物理通道（與 TESNCA 語意相同）。

        以伯努利採樣決定每個 batch 樣本是否使用 GT 取代模型自回歸輸出，
        並在流體遮罩外歸零以維持邊界條件一致性。

        Args:
            state: 當前步驟狀態，shape ``(B, total_dim, H, W)``。
            gt_traj: 完整 GT 軌跡，shape ``(B, T, physics_dim, H, W)``；
                為 ``None`` 或步驟索引超界時略過替換。
            i: 當前時間步索引（0-based），用於取 ``gt_traj[:, i]``。
            tf_ratio: Teacher-Forcing 比例，範圍 ``[0, 1]``；
                0.0 表示純自回歸，1.0 表示全 GT 驅動。
            fluid_mask: 流體遮罩（0/1 float），shape ``(B, 1, H, W)``。
            ss_detach: ``True`` 時對模型物理通道呼叫 ``.detach()``
                以切斷梯度（Scheduled Sampling 模式）。

        Returns:
            替換後的狀態張量，shape 與 ``state`` 相同。
        """
        if not (tf_ratio > 0.0 and gt_traj is not None and i < gt_traj.shape[1]):
            return state
        sd = self._cfg.static_dim
        pd = self._cfg.physics_dim
        hd = self._cfg.hidden_dim
        gt_i = gt_traj[:, i]
        use_gt = torch.bernoulli(
            torch.full((state.shape[0],), tf_ratio, device=state.device)
        ).bool()
        use_gt_exp = use_gt[:, None, None, None].expand_as(gt_i)
        phys_model = state[:, sd:sd + pd]
        if ss_detach:
            phys_model = phys_model.detach()
        phys_new = torch.where(use_gt_exp, gt_i, phys_model)
        phys_new = phys_new * fluid_mask
        if hd > 0:
            return torch.cat([state[:, :sd], phys_new, state[:, sd + pd:]], dim=1)
        return torch.cat([state[:, :sd], phys_new], dim=1)


__all__ = ["HBaseNCA"]
