"""Baseline 自回歸 stepper 與穩態 adapter，均符合 HBaseNCA forward 契約。

`BaselineStepperNCA`：自回歸推進 baseline（FNO / modern U-Net / Dilated-ResNet
backbone）= P_new（state→state）。
`SteadyStateUNetNCA`：穩態 geometry→field baseline（Thuerey 範式）= P_old，
結構上不收前次場（忽略 state physics 與 n_steps）。

兩者滿足 trainer / eval 的 forward contract：

    forward(state, cond_norm, fluid_mask, n_steps, return_trajectory=False,
            delta_ctu_seq=None, g_vec_norm=None, tf_ratio=0.0, gt_traj=None,
            ss_detach=False, cumul_ctu_norm_seq=None)

回傳 physics-only trajectory（每步 [B, physics_dim, H, W]，已 ×fluid_mask）。

協定：baseline `hidden_dim=0`、`physics_dim=9` 輸出、條件靠
channel concat（`_cond_inject`）、公平靠 param budget。backbone 末層由 stepper
以殘差（`out_residual`）組合；backbone 自身只做 [B,in,H,W]→[B,out,H,W] 映射。
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from ...config import BaselineConfig, TESModelConfig
from ..h_base import HBaseNCA
from ..transport.admissibility import PhysicsAdmissibility
from ._cond_inject import build_cond_channels, cond_channel_count


class _DummyBackbone(nn.Module):
    """骨架／contract 驗證用 backbone（3×3 conv → ReLU → 1×1 conv）。末層 zero-init。"""

    def __init__(self, in_ch: int, out_ch: int, bcfg: BaselineConfig) -> None:
        """初始化 DummyBackbone。

        Args:
            in_ch: 輸入通道數。
            out_ch: 輸出通道數。
            bcfg: baseline config 物件（僅佔位，此 dummy 不使用）。
        """
        super().__init__()
        hid = 32
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, out_ch, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向推論。

        Args:
            x: 輸入特徵圖，shape [B, in_ch, H, W]。

        Returns:
            shape [B, out_ch, H, W] 的輸出特徵圖。
        """
        return self.net(x)


def _build_backbone(name: str, in_ch: int, out_ch: int, bcfg: BaselineConfig) -> nn.Module:
    """Lazy import 並建構指定 backbone（半成品檔不影響骨架 import）。

    Args:
        name: backbone 名稱，支援 "dummy"、"fno"、"unet"、"drn"、"unet_steady"。
        in_ch: 輸入通道數。
        out_ch: 輸出通道數。
        bcfg: baseline config 物件，傳遞給各 backbone 建構子。

    Returns:
        已初始化的 backbone nn.Module。

    Raises:
        ValueError: 若 `name` 不在已知清單中。
    """
    if name == "dummy":
        return _DummyBackbone(in_ch, out_ch, bcfg)
    if name == "fno":
        from .ar_fno import FNO2DBackbone
        return FNO2DBackbone(in_ch, out_ch, bcfg)
    if name == "unet":
        from .unet_modern import ModernUNetBackbone
        return ModernUNetBackbone(in_ch, out_ch, bcfg)
    if name == "drn":
        from .dilated_resnet import DilatedResNetBackbone
        return DilatedResNetBackbone(in_ch, out_ch, bcfg)
    if name == "unet_steady":
        from .unet_steady import SteadyUNetBackbone
        return SteadyUNetBackbone(in_ch, out_ch, bcfg)
    raise ValueError(f"unknown baseline backbone: {name!r}")


class _BaselineBase(HBaseNCA):
    """共用初始化：略過 HBaseNCA 的 cond/g encoder（baseline 用 channel concat）。

    只保留 init_state / apply_ic_mask / get_fluid_mask / get_aux_metrics /
    _apply_tfc 所需的 self._cfg 與 self._aux_metrics。
    """

    def __init__(self, cfg: TESModelConfig) -> None:
        """初始化 _BaselineBase，略過 HBaseNCA 的條件編碼器建構。

        Args:
            cfg: TES 模型設定物件。
        """
        nn.Module.__init__(self)          # 不呼叫 HBaseNCA.__init__（不建 cond_encoder）
        self._cfg = cfg
        self._scale_feat_dim = 0
        self.g_encoder = None
        self.alpha_g = None
        self._gvec_mode = cfg.gvec_mode
        self._aux_metrics: dict[str, float] = {}
        # HBaseNCA.__init__ 被跳過，故此處自行建。三臂必須套同一組物理約束，
        # 少建這一個會讓 baseline 悄悄變成「沒有可容許性保證」的對照組。
        self.phys_adm = PhysicsAdmissibility(cfg)
        self.state_clip: float = float(cfg.state_clip)
        self._bcfg: BaselineConfig = cfg.baseline
        self._cond_mode: str = cfg.baseline.cond_features
        self._n_cond_ch: int = cond_channel_count(self._cond_mode)


class BaselineStepperNCA(_BaselineBase):
    """自回歸 baseline：每步 backbone(physics + static + cond) → next physics。

    對應論文範式 P_new（state→state），用於與 TES-NCA 公平比較。
    Backbone 可替換為 FNO、modern U-Net、Dilated-ResNet 或 dummy。
    """

    def __init__(self, cfg: TESModelConfig, backbone: Optional[str] = None) -> None:
        """初始化 BaselineStepperNCA。

        Args:
            cfg: TES 模型設定物件。
            backbone: backbone 名稱；若為 None 則從 cfg.baseline.backbone 讀取。
        """
        super().__init__(cfg)
        bk = backbone or cfg.baseline.backbone
        sd, pd = cfg.static_dim, cfg.physics_dim
        in_ch = pd + sd + self._n_cond_ch
        self.backbone = _build_backbone(bk, in_ch, pd, cfg.baseline)
        self._residual: bool = bool(cfg.baseline.out_residual)
        self._backbone_name: str = bk

    def forward(
        self,
        state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        n_steps: int,
        return_trajectory: bool = False,
        delta_ctu_seq: Optional[torch.Tensor] = None,
        g_vec_norm: Optional[torch.Tensor] = None,
        tf_ratio: float = 0.0,
        gt_traj: Optional[torch.Tensor] = None,
        ss_detach: bool = False,
        cumul_ctu_norm_seq: Optional[torch.Tensor] = None,
        **kwargs,  # 容忍 adapter 傳入 ke_anchor 等（baseline 不支援 ceiling，直接忽略）
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """自回歸推進 n_steps 步，回傳最終 state 或完整 trajectory。

        Args:
            state: 初始 state，shape [B, static_dim+physics_dim, H, W]。
            cond_norm: 正規化後的全域條件向量，shape [B, cond_dim]。
            fluid_mask: 流體遮罩，shape [B, 1, H, W]，非流體區強制歸零。
            n_steps: 自回歸推進步數。
            return_trajectory: 若為 True，同時回傳每步 physics 張量組成的 list。
            delta_ctu_seq: per-step 時間增量序列，shape [B, n_steps, ...]；可為 None。
            g_vec_norm: 正規化幾何向量（baseline 不使用，保留契約相容）。
            tf_ratio: Teacher Forcing 比率（0.0 表示純自回歸）。
            gt_traj: Ground truth trajectory，Teacher Forcing 用；可為 None。
            ss_detach: 是否在 Teacher Forcing 切換時 detach state。
            cumul_ctu_norm_seq: 累積時間步序列（baseline 不使用，保留契約相容）。

        Returns:
            若 return_trajectory 為 False：最終 state，shape [B, static_dim+physics_dim, H, W]。
            若 return_trajectory 為 True：(最終 state, trajectory list)，
                trajectory 為長度 n_steps 的 list，每元素 shape [B, physics_dim, H, W]。
        """
        cfg = self._cfg
        sd, pd = cfg.static_dim, cfg.physics_dim
        h, w = state.shape[-2:]
        static = state[:, :sd]
        trajectory: list[torch.Tensor] = []
        for i in range(n_steps):
            phys_in = state[:, sd:sd + pd]
            dt_i = delta_ctu_seq[:, i] if delta_ctu_seq is not None else None
            cond_ch = build_cond_channels(cond_norm, dt_i, h, w, self._cond_mode).to(phys_in.dtype)
            x_in = torch.cat([phys_in, static, cond_ch], dim=1)
            out = self._maybe_ckpt(x_in)
            new_phys = (phys_in + out) if self._residual else out
            new_phys = new_phys * fluid_mask
            # 物理可容許性投影：與主架構同一個算子（見 models/transport/admissibility.py）。
            # 消融臂與 baseline 必須套同一組物理約束，否則臂間比較不受控。停用時逐位恆等。
            new_phys = self.phys_adm(new_phys, fluid_mask)
            state = torch.cat([static, new_phys], dim=1)   # hidden_dim=0
            if return_trajectory:
                trajectory.append(new_phys)
            state = self._apply_tfc(state, gt_traj, i, tf_ratio, fluid_mask, ss_detach)
        self._aux_metrics = {"baseline_residual": float(self._residual)}
        if return_trajectory:
            return state, trajectory
        return state


    def _maybe_ckpt(self, x_in: torch.Tensor) -> torch.Tensor:
        """訓練時對 backbone forward 套 gradient checkpointing（反向重算、不存 activation）。

        序列迭代下 `n_steps` 最高可達 24（`mean_rollout_k_max`），若不做
        checkpointing，完整計算圖會全部留在顯存裡，容易在較大 batch 下觸發
        `torch.OutOfMemoryError`。主模型（`h3t_streamcollide._maybe_ckpt`）與
        baseline 需支援「三模型同 batch、同 rollout_k」的受控比較，故此處
        同樣需要這層 checkpointing。

        數值等價性：`unet_modern` 用 GroupNorm（無 running stats）＋ SiLU，
        無 dropout、無 batch 統計量，recompute 為 deterministic，故與不 checkpoint
        逐位元等價。若日後換成含 BatchNorm 或 dropout 的 backbone，這個等價性
        不再成立（BN 的 running stats 會在重算時被更新兩次）。
        `unet_steady` 確實含 BatchNorm，但 `SteadyStateUNetNCA` 只做單次前向、
        不走本函式。

        Args:
            x_in: backbone 輸入 tensor。

        Returns:
            backbone 輸出。
        """
        if bool(self._cfg.grad_checkpoint) and self.training:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(self.backbone, x_in, use_reentrant=False)
        return self.backbone(x_in)


class SteadyStateUNetNCA(_BaselineBase):
    """穩態 geometry→field baseline（P_old）。忽略 state physics 與 n_steps。

    每次從 geometry+cond 一次預測 developed field；trajectory 每步同值，
    餵自身輸出當 warm-init 無效（結構上無法續推），實證 CH1 gap。
    對應論文範式 P_old（Thuerey 範式）。
    """

    def __init__(self, cfg: TESModelConfig, backbone: Optional[str] = None) -> None:
        """初始化 SteadyStateUNetNCA。

        Args:
            cfg: TES 模型設定物件。
            backbone: backbone 名稱；若為 None 且非 dummy，預設使用 "unet_steady"。
        """
        super().__init__(cfg)
        sd, pd = cfg.static_dim, cfg.physics_dim
        # 穩態無時間推進 → 條件不含 per-step delta
        if self._cond_mode == "re_lchar_delta":
            self._cond_mode = "re_lchar"
            self._n_cond_ch = cond_channel_count(self._cond_mode)
        in_ch = sd + self._n_cond_ch
        # backbone 可換（同 backbone 可跑 P_old vs P_new）；dummy 供 contract test
        if cfg.baseline.backbone == "dummy":
            bk = "dummy"
        else:
            bk = backbone or "unet_steady"
        self.backbone = _build_backbone(bk, in_ch, pd, cfg.baseline)
        self._backbone_name: str = bk

    def forward(
        self,
        state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        n_steps: int,
        return_trajectory: bool = False,
        delta_ctu_seq: Optional[torch.Tensor] = None,
        g_vec_norm: Optional[torch.Tensor] = None,
        tf_ratio: float = 0.0,
        gt_traj: Optional[torch.Tensor] = None,
        ss_detach: bool = False,
        cumul_ctu_norm_seq: Optional[torch.Tensor] = None,
        **kwargs,  # 容忍 adapter 傳入 ke_anchor 等（baseline 不支援 ceiling，直接忽略）
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """一次前向預測穩態場，回傳固定輸出（所有 trajectory 步驟相同）。

        Args:
            state: 初始 state，shape [B, static_dim+physics_dim, H, W]；physics 部分被忽略。
            cond_norm: 正規化後的全域條件向量，shape [B, cond_dim]。
            fluid_mask: 流體遮罩，shape [B, 1, H, W]，非流體區強制歸零。
            n_steps: trajectory 長度（穩態模型輸出重複 n_steps 次，不推進）。
            return_trajectory: 若為 True，回傳長度 n_steps 的重複 list。
            delta_ctu_seq: per-step 時間增量（穩態不使用，保留契約相容）。
            g_vec_norm: 正規化幾何向量（baseline 不使用，保留契約相容）。
            tf_ratio: Teacher Forcing 比率（穩態不使用，保留契約相容）。
            gt_traj: Ground truth trajectory（穩態不使用，保留契約相容）。
            ss_detach: detach 旗標（穩態不使用，保留契約相容）。
            cumul_ctu_norm_seq: 累積時間步序列（穩態不使用，保留契約相容）。

        Returns:
            若 return_trajectory 為 False：預測的 state，shape [B, static_dim+physics_dim, H, W]。
            若 return_trajectory 為 True：(state, trajectory list)，
                trajectory 為長度 max(1, n_steps) 的 list，每元素相同 shape [B, physics_dim, H, W]。
        """
        cfg = self._cfg
        sd, pd = cfg.static_dim, cfg.physics_dim
        h, w = state.shape[-2:]
        static = state[:, :sd]
        cond_ch = build_cond_channels(cond_norm, None, h, w, self._cond_mode).to(static.dtype)
        x_in = torch.cat([static, cond_ch], dim=1)
        phys_out = self.backbone(x_in) * fluid_mask
        phys_out = self.phys_adm(phys_out, fluid_mask)   # 同上，三臂共用同一組物理約束
        state_out = torch.cat([static, phys_out], dim=1)   # hidden_dim=0
        self._aux_metrics = {"baseline_steady": 1.0}
        if return_trajectory:
            return state_out, [phys_out for _ in range(max(1, n_steps))]
        return state_out


__all__ = ["BaselineStepperNCA", "SteadyStateUNetNCA"]
