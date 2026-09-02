"""H3T-WarpResidual NCA 模組：multirate residual 與 learned warp transport 的 gate 融合架構。

HD-4a H3T-WarpResidual NCA。

H3 multirate residual + explicit coarse-to-fine learned **transport** (warp):

每個 outer recurrent step：
  1. residual path：沿用 HMultirate（coarse 多 ratio + fine substeps）算出 residual_phys。
  2. transport path：從 step 起點 state 下採樣 → warp encoder → coarse displacement field，
     上採樣後 grid_sample 把 **前一步 physics** 搬移（advection/transport），再加 zero-init
     residual correction。
  3. gate（sigmoid，初始 bias 偏 residual）混合：
        next_phys = (1-gate) * residual_phys + gate * (warped + corr)

H3T-WarpResidual：殘差路徑 + 半拉格朗日平流平行路徑，sigmoid 混合。

設計遵守 HD_plan §6.2 / §14.1：
  - displacement / correction head zero-init → 初始 disp=0、corr=0 → warped=prev_phys。
  - gate_bias 偏負（預設 -4）→ gate≈0.018 → 初始輸出 ≈ residual baseline（identity-compatible）。
  - physics channels 才 warp；static / fluid_mask 不 warp，solid 區由 fluid_mask gate。
  - grid_sample padding_mode='border'、align_corners=False，避免邊界吸入無效值。
  - 不靠增加 rollout k 擴感受野；transport 在單次 forward 內完成。

forward contract 與 TESNCA / HMultirateNCA 完全一致。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....config import TESModelConfig
from ...h_multirate_nca import HMultirateNCA
from ...multiscale_ops import downsample_state, upsample_state


class H3TWarpResidualNCA(HMultirateNCA):
    """H3T-WarpResidual：殘差路徑 + 半拉格朗日平流平行路徑，sigmoid 混合。

    並行版本：residual path 與 transport path 都從步首 state 出發，
    最後以 sigmoid gate 做 convex blend。

    每個 outer step 的合成公式：
        next_phys = (1 - gate) * residual_phys + gate * (warped + corr)

    gate_bias 偏負（預設 -4）確保起步行為等同 HMultirateNCA baseline。
    """

    def __init__(self, cfg: TESModelConfig) -> None:
        """初始化 H3TWarpResidualNCA。

        Args:
            cfg: TES 模型設定，包含 transport 子設定（warp_ratio、disp_max_px 等）。
        """
        super().__init__(cfg)
        tc = cfg.transport
        self._warp_ratio = int(tc.warp_ratio)
        self._disp_max_px = float(tc.disp_max_px)
        self._disp_ctu_cap = float(tc.disp_ctu_cap)
        self._gate_bias = float(tc.gate_bias)

        enc_h = int(tc.enc_hidden)
        # warp encoder：對下採樣 state 抽特徵（含 static + physics + hidden）。
        self.warp_encoder = nn.Sequential(
            nn.Conv2d(cfg.state_dim, enc_h, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_h, enc_h, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # displacement head：輸出 2ch (dx, dy)；zero-init → 初始無位移。
        self.disp_head = nn.Conv2d(enc_h, 2, kernel_size=1)
        nn.init.zeros_(self.disp_head.weight)
        nn.init.zeros_(self.disp_head.bias)
        # residual correction head：輸出 physics_dim；zero-init → 初始無修正。
        self.corr_head = nn.Conv2d(enc_h, cfg.physics_dim, kernel_size=1)
        nn.init.zeros_(self.corr_head.weight)
        nn.init.zeros_(self.corr_head.bias)
        # gate head：輸出 1ch logit；weight zero-init、bias=gate_bias → gate≈sigmoid(bias)。
        self.gate_head = nn.Conv2d(enc_h, 1, kernel_size=1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, self._gate_bias)

    # ------------------------------------------------------------------
    def _warp_branch(
        self,
        state_in: torch.Tensor,
        prev_phys: torch.Tensor,
        fluid_mask: torch.Tensor,
        delta_ctu_seq: Optional[torch.Tensor],
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """計算 warp transport branch，回傳平流結果、修正量、gate 與診斷指標。

        所有 warp 計算在 float32（AMP 安全）。

        Args:
            state_in: 步首完整 state，形狀 [B, state_dim, H, W]。
            prev_phys: 步首 physics channels，形狀 [B, physics_dim, H, W]。
            fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，solid 區為 0。
            delta_ctu_seq: 每步 delta CTU 序列 [B, n_steps]，None 表示不使用時間條件。
            step_idx: 當前 outer step 索引，用於取 delta_ctu_seq[:, step_idx]。

        Returns:
            warped: 平流後的 physics，形狀 [B, physics_dim, H, W]。
            corr: 平流後修正量，形狀 [B, physics_dim, H, W]。
            gate: sigmoid gate，形狀 [B, 1, H, W]，值域 [0, 1]。
            metrics: 位移與 gate 診斷指標字典（detached）。
        """
        b, _, h, w = state_in.shape
        feat = self.warp_encoder(downsample_state(state_in, self._warp_ratio, mode="area"))
        disp_coarse = torch.tanh(self.disp_head(feat))           # [B,2,hr,wr] in [-1,1]
        corr_coarse = self.corr_head(feat)                       # [B,pd,hr,wr]
        gate_coarse = self.gate_head(feat)                       # [B,1,hr,wr] logit

        # delta_ctu 決定位移尺度（大 Δt → 大 transport span）。
        if delta_ctu_seq is not None:
            dt_i = delta_ctu_seq[:, step_idx].to(device=feat.device, dtype=feat.dtype)
            dt_scale = dt_i.clamp(0.0, self._disp_ctu_cap)[:, None, None, None]
        else:
            dt_scale = torch.ones(b, 1, 1, 1, device=feat.device, dtype=feat.dtype)
        disp_px_coarse = disp_coarse * (self._disp_max_px * dt_scale)  # px at coarse res

        disp_px = upsample_state(disp_px_coarse, (h, w), mode="bilinear")  # [B,2,H,W]
        corr = upsample_state(corr_coarse, (h, w), mode="bilinear")        # [B,pd,H,W]
        gate = torch.sigmoid(upsample_state(gate_coarse, (h, w), mode="bilinear"))  # [B,1,H,W]

        # grid_sample 在 float32 + autocast 關閉，確保數值穩定。
        with torch.autocast(device_type=state_in.device.type, enabled=False):
            prev_f = prev_phys.float()
            disp_f = disp_px.float()
            theta = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=state_in.device, dtype=torch.float32
            ).unsqueeze(0).expand(b, 2, 3)
            base_grid = F.affine_grid(theta, (b, prev_f.shape[1], h, w), align_corners=False)
            # px → normalized（align_corners=False）：dx_norm = dx * 2 / W。
            dx_norm = disp_f[:, 0] * (2.0 / w)
            dy_norm = disp_f[:, 1] * (2.0 / h)
            sample_grid = torch.stack(
                [base_grid[..., 0] - dx_norm, base_grid[..., 1] - dy_norm], dim=-1
            )
            warped = F.grid_sample(
                prev_f, sample_grid, mode="bilinear",
                padding_mode="border", align_corners=False,
            )
        warped = warped.to(prev_phys.dtype) * fluid_mask

        # 位移 / gate 統計（detached，供 trainer logging）。
        with torch.no_grad():
            disp_mag = torch.sqrt(disp_px[:, 0] ** 2 + disp_px[:, 1] ** 2).float()  # [B,H,W]
            fm = fluid_mask[:, 0].float()
            denom = fm.sum().clamp_min(1.0)
            disp_mean = (disp_mag * fm).sum() / denom
            flat = disp_mag[fm > 0.5]
            disp_p95 = torch.quantile(flat, 0.95) if flat.numel() > 0 else disp_mag.new_zeros(())
            disp_max = flat.max() if flat.numel() > 0 else disp_mag.new_zeros(())
            g = gate[:, 0].float()
            gate_mean = (g * fm).sum() / denom
            gflat = g[fm > 0.5]
            gate_p95 = torch.quantile(gflat, 0.95) if gflat.numel() > 0 else g.new_zeros(())
            # disp_smoothness：位移場空間梯度 L1（越大越不平滑），供 transport 正則監控。
            dpx = disp_px.float()
            d_dx = (dpx[:, :, :, 1:] - dpx[:, :, :, :-1]).abs().mean()
            d_dy = (dpx[:, :, 1:, :] - dpx[:, :, :-1, :]).abs().mean()
            disp_smoothness = d_dx + d_dy
        metrics: dict[str, torch.Tensor] = {
            "disp_mean_px": disp_mean, "disp_p95_px": disp_p95, "disp_max_px": disp_max,
            "disp_smoothness": disp_smoothness,
            "warp_gate_mean": gate_mean, "warp_gate_p95": gate_p95,
        }
        return warped, corr, gate, metrics

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
        """執行 n_steps outer recurrent step（並行 warp-residual 版）。

        每步：residual path（multirate coarse + fine substep）與 transport path
        （warp + corr）並行計算，最後由 sigmoid gate 做 convex blend。

        Args:
            state: 起始 state，形狀 [B, state_dim, H, W]。
            cond_norm: 正規化後的條件向量，形狀 [B, cond_dim]。
            fluid_mask: 流體遮罩，形狀 [B, 1, H, W]。
            n_steps: outer recurrent step 數。
            return_trajectory: 若 True，回傳每步 physics 的軌跡列表。
            delta_ctu_seq: 每步 delta CTU [B, n_steps]，None 表示固定 dt。
            g_vec_norm: 幾何向量 FiLM 條件 [B, g_dim]，None 表示不使用。
            tf_ratio: teacher forcing 比例，0.0 表示全自回歸。
            gt_traj: ground-truth 軌跡，teacher forcing 時使用。
            ss_detach: scheduled sampling 是否 detach。
            cumul_ctu_norm_seq: 累計 CTU 正規化序列，供 per-step cumul 條件使用。

        Returns:
            return_trajectory=False 時：最終 state [B, state_dim, H, W]。
            return_trajectory=True 時：(最終 state, trajectory list)，
                trajectory 中每個元素為 [B, physics_dim, H, W]。
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
        from ...multiscale_ops import downsample_mask
        coarse_masks = [downsample_mask(fluid_mask, ratio, threshold=0.5) for ratio in ratios]

        trajectory: list[torch.Tensor] = []
        warp_acc: dict[str, torch.Tensor] = {}
        warp_ratio_acc = state.new_zeros(())

        for i in range(n_steps):
            state_in = state
            prev_phys = state_in[:, sd:sd + pd]

            # ---- residual path: coarse (沿用 HMultirate) ----
            coarse_sum = state.new_zeros(state.shape[0], pd, h_full, w_full)
            for level_idx, (ratio, coarse_cell, coarse_mask) in enumerate(
                zip(ratios, self.coarse_cells, coarse_masks)
            ):
                gamma_c, beta_c, dt_c = self._encode_cond(
                    cond_norm, delta_ctu_seq, cumul_ctu_norm_seq, i,
                    time_conditioned, per_step_cumul,
                    level_id=float(level_idx), ratio=ratio, dt_scale=1.0,
                    gamma_g=gamma_g, beta_g=beta_g,
                )
                cs = downsample_state(state, ratio, mode="area")
                csn = coarse_cell(cs, gamma_c, beta_c, coarse_mask, dt_c)
                delta = csn[:, sd:sd + pd] - cs[:, sd:sd + pd]
                coarse_sum = coarse_sum + upsample_state(delta, (h_full, w_full), mode="bilinear")
            new_phys_res = (prev_phys + coarse_sum) * fluid_mask
            if hd > 0:
                state = torch.cat([state[:, :sd], new_phys_res, state[:, sd + pd:]], dim=1)
            else:
                state = torch.cat([state[:, :sd], new_phys_res], dim=1)

            # ---- residual path: fine substeps ----
            gamma_f, beta_f, dt_f = self._encode_cond(
                cond_norm, delta_ctu_seq, cumul_ctu_norm_seq, i,
                time_conditioned, per_step_cumul,
                level_id=float(len(ratios)), ratio=1, dt_scale=dt_fine,
                gamma_g=gamma_g, beta_g=beta_g,
            )
            for _ in range(sub):
                state = self.fine_cell(state, gamma_f, beta_f, fluid_mask, dt_f)
            residual_phys = state[:, sd:sd + pd]

            # ---- transport path: warp prev_phys + correction ----
            warped, corr, gate, wm = self._warp_branch(
                state_in, prev_phys, fluid_mask, delta_ctu_seq, i
            )
            blended_phys = (
                (1.0 - gate) * residual_phys + gate * (warped + corr)
            ) * fluid_mask

            if hd > 0:
                state = torch.cat([state[:, :sd], blended_phys, state[:, sd + pd:]], dim=1)
            else:
                state = torch.cat([state[:, :sd], blended_phys], dim=1)

            with torch.no_grad():
                for k, v in wm.items():
                    warp_acc[k] = warp_acc.get(k, state.new_zeros(())) + v
                res_mag = ((1.0 - gate) * residual_phys).abs().float().mean()
                warp_mag = (gate * (warped + corr)).abs().float().mean()
                warp_ratio_acc = warp_ratio_acc + warp_mag / res_mag.clamp_min(1e-12)

            if return_trajectory:
                trajectory.append(state[:, sd:sd + pd])
            state = self._apply_tfc(state, gt_traj, i, tf_ratio, fluid_mask, ss_detach)

        n_outer = max(1, n_steps)
        self._aux_metrics = {
            "fine_substeps": float(sub),
            "ratio": float(ratios[0]),
            "max_ratio": float(max(ratios)),
            "num_coarse_scales": float(len(ratios)),
            "dcs_enabled": float(delta_ctu_seq is not None),
            "warp_ratio": float(self._warp_ratio),
            "warp_residual_ratio": warp_ratio_acc / n_outer,
            "single_step_rf_diameter_px_est": float(4 * max(ratios)),
        }
        for k, v in warp_acc.items():
            self._aux_metrics[k] = v / n_outer

        if return_trajectory:
            return state, trajectory
        return state


class H3TWarpResidualSerialNCA(H3TWarpResidualNCA):
    """H3T-WarpResidual 的**串行**消融版本（LBM stream → collide 結構）。

    與 H3TWarpResidualNCA 的唯一差別是 transport 與 residual 的組合方式：

      - 並行（父類）：兩條 path 都讀步首 state，最後 gate convex blend
            next = (1-gate) * residual_phys + gate * (warped + corr)
        物理缺陷：blend ≠ composition；collide 作用在「平流前」位置；transport path
        的 corr 與 residual path 形成兩個重複的 collision。

      - 串行（本類）：先 STREAM 再 COLLIDE，符合 LBM 與 operator splitting
            streamed = (1-gate) * prev_phys + gate * (warped + corr)   # STREAM（gated 平流量）
            next     = collide(streamed)                               # COLLIDE 作用在平流後的場
        collide 吃平流後的場（正確位置）；不再對兩個 collision 做 blend。

    其餘全部繼承父類、保持一致（__init__ / warp_encoder / disp_head / corr_head /
    gate_head / _warp_branch / 所有 disp & gate metrics），故參數量與初始化相同，
    可作為「並行 vs 串行」的乾淨消融對照。

    gate 的角色從「path 混合」變為「平流量」：gate=0 → 不平流（streamed=prev_phys）、
    gate=1 → 全 warp+corr。identity-init 仍成立：gate≈sigmoid(gate_bias)≈0.018、
    disp=0 → warped=prev_phys、corr=0 → streamed≈prev_phys → collide(prev)=baseline。

    forward contract 與父類完全一致。
    """

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
        """執行 n_steps outer recurrent step（串行 stream → collide 版）。

        每步先執行 STREAM（gated 半 Lagrangian 平流），再執行 COLLIDE
        （multirate residual 作用在平流後的場），符合 LBM operator splitting。

        Args:
            state: 起始 state，形狀 [B, state_dim, H, W]。
            cond_norm: 正規化後的條件向量，形狀 [B, cond_dim]。
            fluid_mask: 流體遮罩，形狀 [B, 1, H, W]。
            n_steps: outer recurrent step 數。
            return_trajectory: 若 True，回傳每步 physics 的軌跡列表。
            delta_ctu_seq: 每步 delta CTU [B, n_steps]，None 表示固定 dt。
            g_vec_norm: 幾何向量 FiLM 條件 [B, g_dim]，None 表示不使用。
            tf_ratio: teacher forcing 比例，0.0 表示全自回歸。
            gt_traj: ground-truth 軌跡，teacher forcing 時使用。
            ss_detach: scheduled sampling 是否 detach。
            cumul_ctu_norm_seq: 累計 CTU 正規化序列，供 per-step cumul 條件使用。

        Returns:
            return_trajectory=False 時：最終 state [B, state_dim, H, W]。
            return_trajectory=True 時：(最終 state, trajectory list)，
                trajectory 中每個元素為 [B, physics_dim, H, W]。
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
        from ...multiscale_ops import downsample_mask
        coarse_masks = [downsample_mask(fluid_mask, ratio, threshold=0.5) for ratio in ratios]

        trajectory: list[torch.Tensor] = []
        warp_acc: dict[str, torch.Tensor] = {}
        warp_ratio_acc = state.new_zeros(())

        for i in range(n_steps):
            state_in = state
            prev_phys = state_in[:, sd:sd + pd]

            # ---- STREAM: gated semi-Lagrangian warp（產生平流後 physics）----
            warped, corr, gate, wm = self._warp_branch(
                state_in, prev_phys, fluid_mask, delta_ctu_seq, i
            )
            streamed_phys = (
                (1.0 - gate) * prev_phys + gate * (warped + corr)
            ) * fluid_mask
            if hd > 0:
                state = torch.cat([state_in[:, :sd], streamed_phys, state_in[:, sd + pd:]], dim=1)
            else:
                state = torch.cat([state_in[:, :sd], streamed_phys], dim=1)

            # ---- COLLIDE: residual path 作用在平流後的 state ----
            # coarse 多 ratio（沿用 HMultirate）
            coarse_sum = state.new_zeros(state.shape[0], pd, h_full, w_full)
            for level_idx, (ratio, coarse_cell, coarse_mask) in enumerate(
                zip(ratios, self.coarse_cells, coarse_masks)
            ):
                gamma_c, beta_c, dt_c = self._encode_cond(
                    cond_norm, delta_ctu_seq, cumul_ctu_norm_seq, i,
                    time_conditioned, per_step_cumul,
                    level_id=float(level_idx), ratio=ratio, dt_scale=1.0,
                    gamma_g=gamma_g, beta_g=beta_g,
                )
                cs = downsample_state(state, ratio, mode="area")
                csn = coarse_cell(cs, gamma_c, beta_c, coarse_mask, dt_c)
                delta = csn[:, sd:sd + pd] - cs[:, sd:sd + pd]
                coarse_sum = coarse_sum + upsample_state(delta, (h_full, w_full), mode="bilinear")
            # collide 把 residual 加到「平流後」的 physics（streamed_phys），非步首 prev_phys。
            new_phys_res = (streamed_phys + coarse_sum) * fluid_mask
            if hd > 0:
                state = torch.cat([state[:, :sd], new_phys_res, state[:, sd + pd:]], dim=1)
            else:
                state = torch.cat([state[:, :sd], new_phys_res], dim=1)

            # fine substeps（亦作用在平流後的 state）
            gamma_f, beta_f, dt_f = self._encode_cond(
                cond_norm, delta_ctu_seq, cumul_ctu_norm_seq, i,
                time_conditioned, per_step_cumul,
                level_id=float(len(ratios)), ratio=1, dt_scale=dt_fine,
                gamma_g=gamma_g, beta_g=beta_g,
            )
            for _ in range(sub):
                state = self.fine_cell(state, gamma_f, beta_f, fluid_mask, dt_f)

            # COLLIDE 輸出即為最終 physics（已含平流後修正），不再做 gate blend。
            next_phys = state[:, sd:sd + pd] * fluid_mask
            if hd > 0:
                state = torch.cat([state[:, :sd], next_phys, state[:, sd + pd:]], dim=1)
            else:
                state = torch.cat([state[:, :sd], next_phys], dim=1)

            with torch.no_grad():
                for k, v in wm.items():
                    warp_acc[k] = warp_acc.get(k, state.new_zeros(())) + v
                # 串行版 warp_residual_ratio：transport 改變量 vs collide 改變量。
                warp_mag = (streamed_phys - prev_phys).abs().float().mean()
                res_mag = (next_phys - streamed_phys).abs().float().mean()
                warp_ratio_acc = warp_ratio_acc + warp_mag / res_mag.clamp_min(1e-12)

            if return_trajectory:
                trajectory.append(state[:, sd:sd + pd])
            state = self._apply_tfc(state, gt_traj, i, tf_ratio, fluid_mask, ss_detach)

        n_outer = max(1, n_steps)
        self._aux_metrics = {
            "fine_substeps": float(sub),
            "ratio": float(ratios[0]),
            "max_ratio": float(max(ratios)),
            "num_coarse_scales": float(len(ratios)),
            "dcs_enabled": float(delta_ctu_seq is not None),
            "warp_ratio": float(self._warp_ratio),
            "warp_residual_ratio": warp_ratio_acc / n_outer,
            "serial_transport": 1.0,
            "single_step_rf_diameter_px_est": float(4 * max(ratios)),
        }
        for k, v in warp_acc.items():
            self._aux_metrics[k] = v / n_outer

        if return_trajectory:
            return state, trajectory
        return state


__all__ = ["H3TWarpResidualNCA", "H3TWarpResidualSerialNCA"]
