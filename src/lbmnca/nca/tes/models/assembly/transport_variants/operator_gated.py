"""Operator-Gated Stream-Collide NCA — flag 驅動的增量 operator 整合模組。

當前主力架構（C-series 訓練），旗標驅動算子組合：
入流先驗 / 速度位移 / stream gating / creation 算子。

繼承 `H3TStreamCollideNCA`，以 `TESTransportConfig` 的 flag 逐一啟用 operator：

    prior_mode      off | composition   # HF-1d：deterministic inflow prior 注入 state_create
    use_velocity_disp                   # HF-1e：速度場條件化位移
    gate_mode       off | g0 | g1 | g2  # HF-1f：delta 條件化 stream 強度閘控
    use_create                          # HF-1g：create / source branch
    warp_substeps                       # HF-3f（待實作）

**所有 flag = 預設（off/1）時，forward 必須與 H3TStreamCollideNCA 逐元素相等**（identity-init，
見 `tests/nca/tes/test_hf1d_prior_composition.py`）。HF-2 全開即 full model。

設計：不複製 parent 的 forward。prior composition 透過 override `_stream` 在 warp **之前**
修改 `prev_phys` / `state_in`（operator 順序 CREATE/PRIOR → STREAM → COLLIDE，HF_plan §1.2）。
runtime metadata（current_ctu / L_char / prior_vel）由 `forward` wrapper 存入 `self._prior_runtime`，
override `_stream` 讀取。
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from ....config import TESModelConfig
from .h3t_streamcollide import H3TStreamCollideNCA
from ....loss.inflow_prior import compute_inflow_masks

# physics 通道（相對 physics block 起點）：jx=3, jy=5（ARCHITECTURE §5.1）
_JX: int = 3
_JY: int = 5
_X01_STATE_CH: int = 2  # state static block：[SDF, upstream, X_01, Y_01]


class OperatorGatedStreamCollideNCA(H3TStreamCollideNCA):
    """flag 驅動的 operator-gated streamcollide NCA（HF-1d 起）。

    以 `TESTransportConfig` 中的 flag 控制各 operator 的啟閉。
    所有 flag 為 off/預設時，forward 行為與 `H3TStreamCollideNCA` 逐元素相等（identity-init）。

    算子執行順序（每 outer step）：
        1. CREATE residual（use_create）：active 區源注入
        2. PRIOR composition（prior_mode="composition"）：鬆弛式 inflow prior
        3. STREAM（含 velocity-disp HF-1e + gate HF-1f）
        4. COLLIDE（multirate residual + fine substeps + KE rescale）

    Attributes:
        wants_prior_ctx: 訓練器是否需傳入 prior_ctx 的旗標（prior 或 create 皆需 front mask）。
        W_prior: 鬆弛注入係數（prior_mode="composition" 時啟用），zero-init → identity。
        create_head: source residual head（use_create 時啟用），zero-init → identity。
        alpha_vel: 速度場條件化強度（use_velocity_disp 時啟用），zero-init → identity。
        gate_stream: stream 強度閘控線性層（gate_mode="g1" 時啟用），zero-init → identity。
    """

    def __init__(self, cfg: TESModelConfig) -> None:
        """初始化 OperatorGatedStreamCollideNCA。

        Args:
            cfg: TES 模型完整設定，包含 transport flags 與 architecture 參數。
        """
        super().__init__(cfg)
        tc = cfg.transport
        self._prior_mode: str = str(tc.prior_mode)
        self._prior_w_front: float = float(tc.prior_w_front)
        self._use_velocity_disp: bool = bool(tc.use_velocity_disp)
        self._use_create: bool = bool(tc.use_create)
        # runtime metadata（每次 forward 設定）：current_ctu_raw_seq / l_char_px / prior_vel_norm
        self._prior_runtime: Optional[dict] = None
        # trainer 用此旗標決定是否傳 prior_ctx（avoid 影響其他 arch；prior 或 create 皆需 front mask）
        self.wants_prior_ctx: bool = (self._prior_mode == "composition") or self._use_create
        if self._prior_mode == "composition":
            # W_prior zero-init → 起步 state_create = state_t → identity（= streamcollide baseline）
            self.W_prior: nn.Parameter = nn.Parameter(torch.zeros(1))
        if self._use_create:
            # HF-1g：create / source branch。low-cap conv → physics residual，限 active 區（front 後方）。
            # 最終層 zero-init → residual=0 → identity（= streamcollide baseline）。
            ch = int(tc.enc_hidden)
            self.create_head: nn.Sequential = nn.Sequential(
                nn.Conv2d(cfg.state_dim, ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, cfg.physics_dim, kernel_size=1),
            )
            nn.init.zeros_(self.create_head[-1].weight)
            nn.init.zeros_(self.create_head[-1].bias)
        if self._use_velocity_disp:
            # HF-1e：velocity-conditioned displacement。alpha_vel zero-init → 起步 disp_prior=0
            # → disp = 純自由學 tanh(disp_head)（= streamcollide baseline，identity-init）。
            self.alpha_vel: nn.Parameter = nn.Parameter(torch.zeros(1))

        # HF-1f：operator gate（delta-conditioned stream 強度）。
        self._gate_mode: str = str(tc.gate_mode)
        self._cur_dt: Optional[torch.Tensor] = None   # 由 _stream override 設定（delta_ctu per step）
        if self._gate_mode == "g1":
            # G1 low-capacity：delta_ctu_norm → W_stream multiplier。zero-init → sigmoid(0)*2=1
            # → W_stream=1 → disp 不變（identity-init）。不吃 raw absolute ctu（防 time-lookup）。
            self.gate_stream: nn.Linear = nn.Linear(1, 1)
            nn.init.zeros_(self.gate_stream.weight)
            nn.init.zeros_(self.gate_stream.bias)

    # ------------------------------------------------------------------
    def _w_stream(self) -> Union[float, torch.Tensor]:
        """計算 HF-1f stream 強度權重 W_stream。

        off → 恆等 1.0；g0 → deterministic ramp（目前同 1.0）；g1 → learned multiplier。
        identity-init 保證：所有 mode 在初始化時輸出 1.0。

        Returns:
            純量 1.0 或形狀 [B,1,1,1] 的 Tensor（供 broadcast 乘 disp）。
        """
        if self._gate_mode == "off" or self._cur_dt is None:
            return 1.0
        dt = self._cur_dt.float().view(-1, 1)                     # [B,1] delta_ctu（步幅，非 absolute ctu）
        if self._gate_mode == "g0":
            return 1.0   # G0 deterministic 預設恆等（closed-form ramp 留 HF-3 multi-delta 再定）
        # g1：W_stream = 2*sigmoid(linear(delta_norm))，init→1
        dt_norm = (dt / 2.0).clamp(0.0, 1.5)                      # delta∈{0.2,1,2} → 粗略 [0.1,1]
        w = 2.0 * torch.sigmoid(self.gate_stream(dt_norm))        # [B,1]，init=1
        return w.view(-1, 1, 1, 1)

    # ------------------------------------------------------------------
    def _disp_coarse(
        self,
        feat: torch.Tensor,
        state_coarse: torch.Tensor,
    ) -> torch.Tensor:
        """計算 warp 位移場（HF-1e velocity-disp + HF-1f gate 整合）。

        最終位移：disp = W_stream * [ tanh(disp_head) + alpha_vel * normalized_momentum_dir ]

        alpha_vel=0、W_stream=1（init）時，等同 streamcollide baseline（identity-init）。

        turb_module 診斷（M7-2）：state physics = fluct，但渦街被 mean+fluct **合成流場**
        平流 → momentum 條件補上 coarse mean jx/jy（mom = fluct + mean）。
        mean runtime 不存在時行為不變。

        Args:
            feat: encoder 輸出特徵圖，形狀 [B, feat_ch, Hr, Wr]。
            state_coarse: downsample 後的 state，形狀 [B, state_dim, Hr, Wr]。

        Returns:
            位移場，形狀 [B, 2, Hr, Wr]，值域由 tanh 限制。
        """
        disp = torch.tanh(self.disp_head(feat))                  # [B,2,hr,wr]
        if self._use_velocity_disp:
            sd = self._cfg.static_dim
            mom = state_coarse[:, [sd + _JX, sd + _JY]]           # [B,2,hr,wr] normalized (jx,jy)
            rt = getattr(self, "_mean_runtime", None)
            if rt is not None:
                # .get() 合法例外：mean_coarse_warp 為 per-forward 可選快取，不存在時按需計算
                mean_c = rt.get("mean_coarse_warp")
                if mean_c is None:
                    from ...multiscale_ops import downsample_state
                    mean_c = downsample_state(
                        rt["mean_full"], self._warp_ratio, mode="area"
                    )
                    rt["mean_coarse_warp"] = mean_c               # per-forward 快取
                mom = mom + mean_c[:, [_JX, _JY]]
            disp = disp + self.alpha_vel * mom
        w_stream = self._w_stream()
        if not isinstance(w_stream, float):
            disp = disp * w_stream
        return disp

    # ------------------------------------------------------------------
    def forward(
        self,
        *args: object,
        prior_ctx: Optional[dict] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """thin wrapper：存入 prior runtime metadata 後委派給 parent forward（不改 loop）。

        prior_ctx 在本次 forward 期間存入 `_prior_runtime`，供 `_stream` override 讀取；
        forward 結束後（含例外路徑）一律清除，避免跨步污染。

        Args:
            *args: 同 parent forward 位置參數。
            prior_ctx: 可選的 prior context dict，包含：
                - ``current_ctu_raw_seq``: 時間步 CTU 序列，形狀 [B, T]。
                - ``l_char_px``: 特徵長度（像素），用於 front mask 計算。
                - ``prior_vel_norm``: 入口正規化速度向量 [2]，(jx_norm, jy_norm)。
            **kwargs: 同 parent forward 關鍵字參數。

        Returns:
            parent forward 的輸出 Tensor。
        """
        self._prior_runtime = prior_ctx
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._prior_runtime = None

    # ------------------------------------------------------------------
    def _compose_prior(
        self,
        state_in: torch.Tensor,
        prev_phys: torch.Tensor,
        step_idx: int,
        fluid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """在 transition 鋒面帶、僅 jx/jy 通道，鬆弛式注入 inflow prior（無 target 洩漏）。

        更新公式：
            prev_phys[vel] += W_prior * transition * (prior_vel - prev_phys[vel])

        W_prior=0（init）→ 無修改 → identity。

        Args:
            state_in: 當前 NCA state，形狀 [B, state_dim, H, W]。
            prev_phys: 當前 physics block，形狀 [B, physics_dim, H, W]。
            step_idx: 當前 rollout 步索引，用於從 cc_seq 取對應 CTU。
            fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，1 為流體區域。

        Returns:
            更新後的 (prev_phys, state_in) tuple，形狀不變。
            若 runtime metadata 不完整或 step_idx 越界，原樣回傳。
        """
        rt = self._prior_runtime
        # .get() 合法例外：以下三個鍵皆為可選的 Optional prior context 欄位，缺少時提早返回
        cc_seq = rt.get("current_ctu_raw_seq")
        l_char = rt.get("l_char_px")
        prior_vel = rt.get("prior_vel_norm")        # [2] = (jx_inlet_norm, jy_inlet_norm)
        if cc_seq is None or l_char is None or prior_vel is None:
            return prev_phys, state_in
        if step_idx >= cc_seq.shape[1]:
            return prev_phys, state_in

        sd = self._cfg.static_dim
        x01 = state_in[:, _X01_STATE_CH:_X01_STATE_CH + 1]          # [B,1,H,W] ∈[0,1]
        _active, transition, _inactive = compute_inflow_masks(
            cc_seq[:, step_idx], l_char, x01, fluid_mask, w_front=self._prior_w_front,
        )
        t = transition                                              # [B,1,H,W]
        pv = prior_vel.to(prev_phys.dtype).view(1, 2, 1, 1)         # broadcast jx/jy 入口常數
        # clone 保留 non-jx/jy 通道不變，只修改速度通道（維持完整梯度鏈）
        new_phys = prev_phys.clone()
        w = self.W_prior
        new_phys[:, _JX:_JX + 1] = prev_phys[:, _JX:_JX + 1] + w * t * (pv[:, 0:1] - prev_phys[:, _JX:_JX + 1])
        new_phys[:, _JY:_JY + 1] = prev_phys[:, _JY:_JY + 1] + w * t * (pv[:, 1:2] - prev_phys[:, _JY:_JY + 1])
        # state_in 的 physics block 同步更新（讓 warp encoder 也見到 prior-modified 場）
        hd = self._cfg.hidden_dim
        if hd > 0:
            new_state_in = torch.cat([state_in[:, :sd], new_phys, state_in[:, sd + new_phys.shape[1]:]], dim=1)
        else:
            new_state_in = torch.cat([state_in[:, :sd], new_phys], dim=1)
        return new_phys, new_state_in

    # ------------------------------------------------------------------
    def _compose_create(
        self,
        state_in: torch.Tensor,
        prev_phys: torch.Tensor,
        step_idx: int,
        fluid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """HF-1g：create / source residual，限 active 區（front 後方）。

        head zero-init → residual=0 → identity（= streamcollide baseline）。
        residual 僅注入已被 front 通過的 active 區域，避免污染 inflow 鋒前尚未激活的區域。

        Args:
            state_in: 當前 NCA state，形狀 [B, state_dim, H, W]。
            prev_phys: 當前 physics block，形狀 [B, physics_dim, H, W]。
            step_idx: 當前 rollout 步索引，用於從 cc_seq 取對應 CTU。
            fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，1 為流體區域。

        Returns:
            更新後的 (prev_phys, state_in) tuple，形狀不變。
            若 runtime metadata 不完整或 step_idx 越界，原樣回傳。
        """
        rt = self._prior_runtime
        if rt is None:
            return prev_phys, state_in
        # .get() 合法例外：cc_seq / l_char 為可選的 Optional prior context 欄位，缺少時提早返回
        cc_seq, l_char = rt.get("current_ctu_raw_seq"), rt.get("l_char_px")
        if cc_seq is None or l_char is None or step_idx >= cc_seq.shape[1]:
            return prev_phys, state_in
        sd = self._cfg.static_dim
        x01 = state_in[:, _X01_STATE_CH:_X01_STATE_CH + 1]
        active, _transition, _inactive = compute_inflow_masks(
            cc_seq[:, step_idx], l_char, x01, fluid_mask, w_front=self._prior_w_front,
        )
        residual = self.create_head(state_in)                     # [B,physics,H,W]，zero-init→0
        new_phys = prev_phys + active * residual                  # active 區源注入
        hd = self._cfg.hidden_dim
        if hd > 0:
            new_state_in = torch.cat([state_in[:, :sd], new_phys, state_in[:, sd + new_phys.shape[1]:]], dim=1)
        else:
            new_state_in = torch.cat([state_in[:, :sd], new_phys], dim=1)
        return new_phys, new_state_in

    # ------------------------------------------------------------------
    def _stream(
        self,
        state_in: torch.Tensor,
        prev_phys: torch.Tensor,
        fluid_mask: torch.Tensor,
        delta_ctu_seq: Optional[torch.Tensor],
        step_idx: int,
        collect_metrics: bool = False,
    ) -> torch.Tensor:
        """override：CREATE / PRIOR（warp 前）→ parent STREAM（內部 _disp_coarse 含 gate）。

        operator 執行順序（HF_plan §1.2）：
            1. CREATE residual（use_create）
            2. PRIOR composition（prior_mode="composition"）
            3. 委派 parent STREAM（含 _disp_coarse → gate）

        collect_metrics 須與 parent 簽章一致並 passthrough（parent forward 依
        ``_collect_aux_metrics`` 傳入；不接收此參數會 TypeError）。

        Args:
            state_in: 當前 NCA state，形狀 [B, state_dim, H, W]。
            prev_phys: 當前 physics block，形狀 [B, physics_dim, H, W]。
            fluid_mask: 流體遮罩，形狀 [B, 1, H, W]。
            delta_ctu_seq: 步幅序列（非 absolute ctu），形狀 [B, T] 或 None。
            step_idx: 當前 rollout 步索引。
            collect_metrics: 是否收集 auxiliary metrics，passthrough 給 parent。

        Returns:
            parent _stream 的輸出 Tensor（STREAM 後的 physics state）。
        """
        # gate 用 delta_ctu（步幅，非 absolute ctu）；供 _disp_coarse 的 W_stream
        self._cur_dt = (
            delta_ctu_seq[:, step_idx] if delta_ctu_seq is not None else None
        )
        if self._prior_mode == "composition" and self._prior_runtime is not None:
            prev_phys, state_in = self._compose_prior(state_in, prev_phys, step_idx, fluid_mask)
        if self._use_create:
            prev_phys, state_in = self._compose_create(state_in, prev_phys, step_idx, fluid_mask)
        return super()._stream(state_in, prev_phys, fluid_mask, delta_ctu_seq, step_idx,
                               collect_metrics=collect_metrics)


__all__ = ["OperatorGatedStreamCollideNCA"]
