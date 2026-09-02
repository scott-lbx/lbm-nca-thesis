"""H3T-StreamCollide NCA（串行 transport → residual，LBM stream-collide 結構）。

H3T-StreamCollide：串行 stream→collide（LBM operator splitting），
支援 KE rescaling + 平均流調製。

H3TWarpResidualNCA 的**串行**對照版本。差別只在 transport 與 residual 的組合方式：

  - WarpResidual（並行）：兩條 path 都讀步首 state，gate convex blend
        next = (1-gate) * residual_phys + gate * (warped + corr)
    物理缺陷：blend ≠ composition；collide 作用在「平流前」位置；corr 與 residual
    形成兩個重複的 collision。

  - StreamCollide（本檔，串行）：先 stream 再 collide，符合 LBM 與 operator splitting
        warped   = Advect(prev_phys)            # STREAM：semi-Lagrangian warp
        next     = warped + NCA_residual(warped) # COLLIDE：residual 作用在平流後的場
    collide 吃的是平流後的場（正確位置）；不需要 corr（residual 本身就是平流後修正）；
    不需要 gate（identity-init 已保證起步 = baseline）。

每個 outer recurrent step：
  1. STREAM：從 step 起點 state 下採樣 → warp encoder → coarse displacement field，
     上採樣後 grid_sample 把 **前一步 physics** 後向搬移（backward gather / advection）。
  2. COLLIDE：把 physics 換成 warped 後的場，沿用 HMultirate 的 coarse 多 ratio +
     fine substeps residual path，作用在 **平流後** 的 state 上。
  3. next_phys = collide 輸出（physics channels）* fluid_mask。

identity-init 安全性（與並行版同等）：
  - disp_head zero-init → 初始 disp=0 → warped = prev_phys（grid_sample 零位移 = 恆等）。
  - 故初始 streamed state = step 起點 state → collide(state_in) = HMultirate baseline。
  - warp 在訓練中才慢慢學出非零位移；起步不擾動已驗證的 residual baseline。

設計細節：
  - physics channels 才 warp；static / hidden 不 warp，solid 區由 fluid_mask gate。
  - grid_sample padding_mode='border'、align_corners=False，避免邊界吸入無效值。
  - warp 計算在 float32（AMP 安全）。
  - 不靠增加 rollout k 擴感受野；transport 在單次 forward 內完成。

forward contract 與 TESNCA / HMultirateNCA / H3TWarpResidualNCA 完全一致。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....config import TESModelConfig
from ...h_multirate_nca import HMultirateNCA
from ...multiscale_ops import downsample_mask, downsample_state, upsample_state


class H3TStreamCollideNCA(HMultirateNCA):
    """H3T-StreamCollide：串行 semi-Lagrangian warp（stream）→ multirate residual（collide）。

    實作 LBM operator splitting 的神經網路類比：先平流（stream）再碰撞（collide），
    確保 collide（residual correction）作用在正確的平流後位置。

    支援功能：
      - KE rescale（V2-C/V2-C2）：每步動能守恆補正，含 warmup gate（LC-1 [A17]）。
      - mean_film（LC decompose）：平均流場 FiLM 條件，per-pixel 空間調製。
      - gradient checkpointing：訓練時省 activation 記憶體。
    """

    def __init__(self, cfg: TESModelConfig) -> None:
        """初始化 H3TStreamCollideNCA。

        Args:
            cfg: TES 模型設定，包含 transport 子設定（warp_ratio、disp_max_px、
                 ke_rescale 等）以及 mean_film 旗標。
        """
        super().__init__(cfg)
        # gradient checkpointing：訓練時對 collide cell 反向重算、不存 activation（省全解析記憶體）。
        self._grad_ckpt = bool(cfg.grad_checkpoint)
        self._collect_aux_metrics = False
        tc = cfg.transport
        self._skip_transport = bool(tc.skip)
        # coarse residual 上採樣模式（bilinear C0→渦度顯 ratio-px 格紋；bicubic C1→平滑無格紋）
        self._coarse_upsample_mode = str(cfg.multiscale.upsample_mode)
        # 粗層 delta 是否乘 coarse_mask（切斷固體粗格的洩漏）
        self._coarse_delta_mask = bool(cfg.multiscale.coarse_delta_mask)
        # D1-surgical：state 下採 BlurPool + ψ 上採 blur-upsample（除取樣混疊殘 grain）
        self._d1_surgical = bool(cfg.d1_surgical)
        self._ds_mode = "blur" if self._d1_surgical else "area"
        # config: multiscale.psi_upsample_follows_config。
        #   本欄位原本硬編 bilinear，使 `multiscale.upsample_mode`（支援 bicubic）
        #   接不到 ψ 路徑——粗層 ψ 以 bilinear 上採後取 curl，是 column-banding 的
        #   結構性來源（bilinear 的導數在 cell 邊界不連續 → curl 顯 ratio-px 格紋）。
        #   d1_surgical 的 blur 路徑優先，不被本開關覆蓋。
        if self._d1_surgical:
            self._psi_up_mode = "blur"
        elif bool(cfg.multiscale.psi_upsample_follows_config):
            self._psi_up_mode = str(cfg.multiscale.upsample_mode)
        else:
            self._psi_up_mode = "bilinear"
        # P1 free-stream fluct 衰減：λ + mean|curl| 百分位門檻（治 free-stream grain；inference 每步作用）
        self._fs_decay = float(tc.fs_decay)
        self._fs_decay_thresh = float(tc.fs_decay_thresh)
        # #1 free-stream high-k 選擇性衰減（α + Gaussian 低通核 buffer）
        self._fs_hf_damp = float(tc.fs_hf_damp)
        self._fs_hf_damp_global = bool(tc.fs_hf_damp_global)
        if self._fs_hf_damp > 0.0:
            _k, _sig = int(tc.fs_hf_damp_ksize), float(tc.fs_hf_damp_sigma)
            _ax = torch.arange(_k, dtype=torch.float32) - _k // 2
            _g = torch.exp(-(_ax ** 2) / (2 * _sig ** 2))
            _g = _g / _g.sum()
            if str(tc.fs_hf_damp_aniso) == "x":          # 僅沿 x（W 軸）低通：專殺垂直欄向 band、保 wake y-結構
                self.register_buffer("_gauss_ker", _g.view(1, 1, 1, _k))
                self._hf_pad = (0, _k // 2)
            else:                                        # iso：2D Gaussian
                self.register_buffer("_gauss_ker", torch.outer(_g, _g).view(1, 1, _k, _k))
                self._hf_pad = (_k // 2, _k // 2)
        # coarse-residual anti-image 低通（default 0=零行為變更）
        self._coarse_aa_sigma = float(cfg.multiscale.upsample_antialias_sigma)
        if self._coarse_aa_sigma > 0.0:
            _ks = int(2 * math.ceil(3.0 * self._coarse_aa_sigma) + 1)  # ~±3σ 核
            _ax = torch.arange(_ks, dtype=torch.float32) - _ks // 2
            _g = torch.exp(-(_ax ** 2) / (2.0 * self._coarse_aa_sigma ** 2))
            _g = _g / _g.sum()
            self.register_buffer("_coarse_aa_ker", torch.outer(_g, _g).view(1, 1, _ks, _ks))
        # §9.1(b) 生成式 noise + (c) production forcing（default 0 = 零行為變更）
        self._noise_sigma = float(tc.noise_sigma)
        self._production_weight = float(tc.production_weight)
        self._production_shear_q = float(tc.production_shear_q)
        self._warp_ratio = int(tc.warp_ratio)
        self._disp_max_px = float(tc.disp_max_px)
        self._disp_ctu_cap = float(tc.disp_ctu_cap)

        # ψ-curl 動量頭：jx/jy 改由 curl(ψ) 重建。
        # 保證範圍：只保證「這一步新增的動量增量」在流體內部
        # 無散度，**不是** div(j)=0。它不移除既有散度（div(j_{n+1})=div(j_n)），且寫回時
        # 逐點乘 fluid_mask 會在固/流界面注入散度。完整 (a)(b)(c) 三段限定見
        # `config/_legacy.py::psi_curl_momentum`。
        # **不得回退為「div-free by construction」。**
        # 固定（不可學）3×1/1×3 中央差分 kernel；curl(ψ)=(∂ψ/∂y, -∂ψ/∂x)。邊界 replicate（非週期，
        # 禁 roll——roll 對有 inlet/outlet 的非週期流會把 outlet 端繞接到 inlet 端，混入虛假訊號）。
        self._psi_curl_momentum = bool(cfg.psi_curl_momentum)
        if self._psi_curl_momentum:
            _dax = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)
            self.register_buffer("_diff_ky", _dax.view(1, 1, 3, 1))  # ∂/∂y（H 軸，dim=-2）
            self.register_buffer("_diff_kx", _dax.view(1, 1, 1, 3))  # ∂/∂x（W 軸，dim=-1）

        # ψ low-pass：curl 前平滑 ψ → 每步壓中波長 → warm band 累積被 n×平滑。
        # 固定 Gaussian（不可學）；sigma=0 停用（identity，向後相容）。_curl_psi 開頭套用（粗細兩路共用）。
        self._psi_lowpass_sigma = float(cfg.psi_lowpass_sigma)
        if self._psi_lowpass_sigma > 0.0:
            _sig = self._psi_lowpass_sigma
            _r = max(1, int(round(3.0 * _sig)))
            _xs = torch.arange(-_r, _r + 1, dtype=torch.float32)
            _g = torch.exp(-(_xs ** 2) / (2.0 * _sig * _sig))
            _g = _g / _g.sum()
            _k2 = torch.outer(_g, _g)                        # [K,K] 可分離 2D Gaussian
            self.register_buffer("_psi_lp_kernel", _k2.view(1, 1, 2 * _r + 1, 2 * _r + 1))
            self._psi_lp_r = _r

        enc_h = int(tc.enc_hidden)
        if not self._skip_transport:
            # warp encoder：對下採樣 state 抽特徵（含 static + physics + hidden）。
            self.warp_encoder = nn.Sequential(
                nn.Conv2d(cfg.state_dim, enc_h, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(enc_h, enc_h, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            # displacement head：輸出 2ch (dx, dy)；zero-init → 初始無位移 → warped=prev_phys。
            self.disp_head = nn.Conv2d(enc_h, 2, kernel_size=1)
            nn.init.zeros_(self.disp_head.weight)
            nn.init.zeros_(self.disp_head.bias)
        # NOTE: 串行版無 corr_head / gate_head。residual(collide) path 本身即為平流後修正。

        # ── WFC v2 R3 算子耗散攻擊 ──
        self._warp_mode = str(tc.warp_mode)   # V2-D
        self._ke_rescale = bool(tc.ke_rescale)     # V2-C
        self._ke_rescale_gmax = float(tc.ke_rescale_gmax)
        # V2-C2 對稱 clamp：gmin<1 → 既補消散又壓過幅（移除 1.15^步 單向爆幅）。
        # default 1.0 = 非對稱 V2-C 原行為（只補不壓），零行為變更。
        self._ke_rescale_gmin = float(tc.ke_rescale_gmin)
        # LC-1 [A17]：spin-up warmup——cumulative CTU < start_ctu 時 bypass rescale（0=永遠啟用）。
        self._ke_rescale_start_ctu = float(tc.ke_rescale_start_ctu)
        if self._ke_rescale:
            # LC-1 [A12]：α fixed=1.0（buffer，非 learnable）→ s=g → 精確每步 KE 守恆，不漂回 0。
            self.register_buffer('ke_rescale_alpha', torch.ones(1))
        # ★iter7：IC-anchored KE ceiling（治 re-inject 能量棘輪；root-cause 分析背書）。
        #   只壓不補（g≤1）：ke_out>ref*margin 才動，向 ref 壓；低於 ref 完全自由（spin-up/耗散不限）。
        #   anchor 由 caller 傳入；mode='off'=逐 bit identity。語意由
        #   `tes.inference.protocol.resolve_ke_anchor` 單一定義：anchor = **本次 rollout
        #   起始 state** 的 fluid-masked KE，訓練／評估／部署同式。
        self._ke_anchor_mode = str(tc.ke_anchor_mode)
        self._ke_anchor_gmin = float(tc.ke_anchor_gmin)
        self._ke_anchor_margin = float(tc.ke_anchor_margin)

        # ── LC decompose turb_module：mean conditioning（PLAN §4.2，cfg.mean_film）──
        # MeanFlowEncoder：mean_flow_norm [B,9,H,W] → per-pixel (γ_m, β_m) 空間 FiLM，
        # 與向量 pathway 三路相加（γ = γ_p + γ_g + γ_m）。zero-init → 未傳 mean /
        # 剛啟用時行為與 baseline 逐元素相等。per-rollout 對每個 cell 解析度算一次。
        self._mean_film = bool(cfg.mean_film)
        self.mean_encoder: Optional[nn.Module] = None
        if self._mean_film:
            from ...conditioning.mean_flow_encoder import MeanFlowEncoder
            self.mean_encoder = MeanFlowEncoder(
                in_channels=cfg.physics_dim,
                hidden_dim=enc_h,
                out_dim=cfg.nca_channels,
            )
        # forward 期間的 mean runtime（M7-2：_disp_coarse 合成流場 momentum 用）
        self._mean_runtime: Optional[dict] = None

    # ------------------------------------------------------------------
    def _ckpt_cell(
        self,
        cell: nn.Module,
        *args: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """訓練時對 cell forward 套 gradient checkpointing（反向重算、不存 activation）。

        ``grad_checkpoint=False`` 或 eval 時直接呼叫（行為/數值不變）。recompute 為
        deterministic（cell 無 dropout/隨機）→ 數值與不 checkpoint 等價。
        ``psi_curl_momentum=True`` 時 cell 回傳 ``(new_state, psi_delta)`` tuple；
        ``torch.utils.checkpoint.checkpoint(use_reentrant=False)`` 原生支援 tuple-of-tensors
        回傳，passthrough 無需額外處理。

        Args:
            cell: 要執行的 NCA cell 模組。
            *args: 傳遞給 cell 的位置引數（tensors）。

        Returns:
            cell 的 forward 輸出 tensor。
        """
        if self._grad_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(cell, *args, use_reentrant=False)
        return cell(*args)

    # ------------------------------------------------------------------
    @staticmethod
    def _add_mean_film(
        gamma: torch.Tensor,
        beta: torch.Tensor,
        mf: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """三路 FiLM 合併：向量 γ/β [B,C] broadcast 後加空間 (γ_m, β_m) [B,C,h,w]。

        Args:
            gamma: 向量 FiLM scale 參數，形狀 [B, C]。
            beta: 向量 FiLM shift 參數，形狀 [B, C]。
            mf: 空間 FiLM tuple (γ_m, β_m)，各形狀 [B, C, h, w]；
                None 表示 mean_film 未啟用或未傳入 mean。

        Returns:
            合併後的 (gamma, beta)：若 mf 為 None 則原樣回傳（零行為變更）。
        """
        if mf is None:
            return gamma, beta
        return gamma[:, :, None, None] + mf[0], beta[:, :, None, None] + mf[1]

    # ------------------------------------------------------------------
    def encode_mean_films(
        self,
        mean_flow_norm: Optional[torch.Tensor],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> dict:
        """預算 per-scale mean FiLM (γ_m, β_m)（turb_step 跨多次 forward 共用，省冗餘 encode）。

        mean 在整個 training step 為常數 → encode 一次即可；self-preroll / detached-preroll /
        grad rollout 三段 forward 共用同一份。需在 **grad-enabled** context 呼叫（梯度經
        grad rollout 段回流至 mean_encoder；no_grad preroll 段僅消費、不貢獻梯度）。

        Args:
            mean_flow_norm: [B, physics_dim, H, W] norm 空間 mean_flow；None 回 {}。
            dtype: 可選，cast mean 至此 dtype（對齊 state.dtype）。
            device: 可選，搬 mean 至此 device。

        Returns:
            dict：key=ratio（1=full-res fine cell），value=(γ_m, β_m)；
            mean_film 未啟用或 mean 為 None 時回 {}。
        """
        if self.mean_encoder is None or mean_flow_norm is None:
            return {}
        mean_full = mean_flow_norm
        if dtype is not None or device is not None:
            mean_full = mean_full.to(dtype=dtype, device=device)
        films: dict = {1: self.mean_encoder(mean_full)}
        for ratio in self._ratios:
            films[ratio] = self.mean_encoder(
                downsample_state(mean_full, ratio, mode="area")
            )
        return films

    # ------------------------------------------------------------------
    def _disp_coarse(
        self,
        feat: torch.Tensor,
        state_coarse: torch.Tensor,
    ) -> torch.Tensor:
        """計算 coarse displacement field [B,2,hr,wr]。

        base 實作：自由學 tanh(disp_head(feat))。
        子類（OperatorGatedStreamCollideNCA，HF-1e）可覆寫以加入 velocity-conditioned 項。

        Args:
            feat: warp_encoder 輸出特徵，形狀 [B, enc_h, hr, wr]。
            state_coarse: 下採樣後 state（含 physics jx/jy），供子類 velocity 條件使用，
                形狀 [B, state_dim, hr, wr]。

        Returns:
            coarse 位移場，形狀 [B, 2, hr, wr]，值域 [-1, 1]（tanh 後）。
        """
        return torch.tanh(self.disp_head(feat))

    # ------------------------------------------------------------------
    def _compute_fs_mask(
        self, mean_full: torch.Tensor, fluid_mask: torch.Tensor
    ) -> torch.Tensor:
        """free-stream mask [B,1,H,W]：mean-flow 低 |curl|（自由流）→1、wake（高剪切）→0。

        供 P1 free-stream fluct 衰減：自由流區（GT fluct≈0）的 grain 每步衰減、wake 保留，
        治 hf_growth（grain 累積）同時不鈍化 wake 渦（ens）。mean 在 rollout 內常數→算一次。
        """
        mjx = mean_full[:, 3:4].float()
        mjy = mean_full[:, 5:6].float()
        # 與 `_curl_psi` 同一組 replicate-pad 中央差分核（禁用 torch.roll——roll 對
        # 非週期流的 inlet/outlet 會產生虛假 wrap-around 訊號，與同檔 `_curl_psi`
        # 的慣例一致）。
        curl = self._ddx_replicate(mjy) - self._ddy_replicate(mjx)
        s = curl.abs() * fluid_mask
        flat = s[fluid_mask > 0.5]
        if flat.numel() == 0:
            return torch.zeros_like(s)
        q = torch.quantile(flat, self._fs_decay_thresh)
        # s<q（自由流）→ 正值(趨1)，wake(s>q)→0；clamp [0,1]，再乘 fluid_mask。
        return ((q - s) / (q + 1e-6)).clamp(0.0, 1.0) * fluid_mask

    def _compute_prod_field(
        self, mean_full: torch.Tensor, fluid_mask: torch.Tensor
    ) -> torch.Tensor:
        """§9.1(c) production forcing 場 [B,1,H,W]：λ·shear_norm·wake_gate（mean 常數→算一次）。

        補回 RANS production −⟨u'u'⟩:∂ū 的 fluct 側：高 mean 剪切(wake)區給 fluct 持續能量源，
        治 ens↓ 根（渦不餓死）。shear=|∇ mean speed|，normalize 到 [0,1]，gate 到 >production_shear_q
        百分位（自由流不施）。每步 fluct ×(1+prod_field)，arch op 作用 → 繞 F19。
        """
        mjx = mean_full[:, 3:4].float()
        mjy = mean_full[:, 5:6].float()
        spd = torch.sqrt(mjx ** 2 + mjy ** 2)
        # 與 `_compute_fs_mask` 同用 replicate-pad 中央差分（禁 roll）。
        # 本函式由**獨立**的 `production_weight > 0` 閘門觸發，不受 `_need_fs` 保護；
        # canonical 的 production_weight=0 故同樣不執行。
        gx = self._ddx_replicate(spd)
        gy = self._ddy_replicate(spd)
        shear = torch.sqrt(gx ** 2 + gy ** 2) * fluid_mask
        flat = shear[fluid_mask > 0.5]
        if flat.numel() == 0:
            return torch.zeros_like(shear)
        thr = torch.quantile(flat, self._production_shear_q)
        smax = flat.max().clamp_min(1e-6)
        shear_norm = (shear / smax).clamp(0.0, 1.0)         # [0,1]
        gate = (shear > thr).to(shear.dtype)                # 1 在高剪切 wake
        return (self._production_weight * shear_norm * gate * fluid_mask).to(mean_full.dtype)

    # ------------------------------------------------------------------
    @staticmethod
    def _ddx_replicate(f: torch.Tensor) -> torch.Tensor:
        """∂f/∂x：與 ``_curl_psi`` 同一組固定中央差分係數 ＋ replicate padding。

        全檔慣例：本類別內所有空間微分一律走 replicate-pad conv，
        **禁用 torch.roll**——roll 對有 inlet/outlet 的非週期流會把 outlet 端繞接回
        inlet 端，混入虛假訊號。

        核就地建構而非取 ``self._diff_kx``：那兩個 buffer 只在
        ``psi_curl_momentum=True`` 時註冊，而本函式的呼叫端（fs_decay / production）
        與該旗標無關；就地建構同時避免動到 state_dict 的鍵集合。呼叫端每個
        rollout 只算一次（mean 在 rollout 內為常數），成本可忽略。

        Args:
            f: ``[B, 1, H, W]`` 純量場。

        Returns:
            ``[B, 1, H, W]`` x 方向中央差分。
        """
        k = torch.tensor([-0.5, 0.0, 0.5], dtype=f.dtype, device=f.device).view(1, 1, 1, 3)
        return F.conv2d(F.pad(f, (1, 1, 0, 0), mode="replicate"), k)

    @staticmethod
    def _ddy_replicate(f: torch.Tensor) -> torch.Tensor:
        """∂f/∂y：與 ``_ddx_replicate`` 同慣例（核就地建構，理由見該函式）。

        Args:
            f: ``[B, 1, H, W]`` 純量場。

        Returns:
            ``[B, 1, H, W]`` y 方向中央差分。
        """
        k = torch.tensor([-0.5, 0.0, 0.5], dtype=f.dtype, device=f.device).view(1, 1, 3, 1)
        return F.conv2d(F.pad(f, (0, 0, 1, 1), mode="replicate"), k)

    # ------------------------------------------------------------------
    def _curl_psi(self, psi: torch.Tensor) -> torch.Tensor:
        """離散 curl(ψ) = (∂ψ/∂y, -∂ψ/∂x)。回傳未乘 fluid_mask。

        保證範圍：本函式的輸出（**增量**）在流體內部對與 curl 共用的
        同一組中央差分核滿足 div=0（實測內部 absmean 達機器精度）。這**不**等於場的
        div(j)=0——curl 是 Hodge 投影的反面，只保證不新增散度、不移除既有散度。
        呼叫端乘 fluid_mask 後，固/流界面的 `u·∇m` 項使該處不再成立；域邊界另因
        replicate padding 失效。不得據本函式宣稱「div-free by construction」。

        固定中央差分 kernel、replicate padding（非週期邊界——
        禁用 torch.roll，roll 對非週期流的 inlet/outlet 會產生虛假 wrap-around 訊號）。

        Args:
            psi: [B, 1, H, W] 純量 ψ 場（已在呼叫端 upsample 至目標解析度）。

        Returns:
            [B, 2, H, W] = (djx, djy)，尚未乘 fluid_mask（呼叫端負責）。
        """
        if self._psi_lowpass_sigma > 0.0:                    # ★候選 A：curl 前 ψ low-pass（壓 warm band）
            _lp = self._psi_lp_kernel.to(dtype=psi.dtype, device=psi.device)
            psi = F.conv2d(F.pad(psi, (self._psi_lp_r,) * 4, mode="replicate"), _lp)
        ky = self._diff_ky.to(dtype=psi.dtype, device=psi.device)
        kx = self._diff_kx.to(dtype=psi.dtype, device=psi.device)
        psi_pad_y = F.pad(psi, (0, 0, 1, 1), mode="replicate")
        djx = F.conv2d(psi_pad_y, ky)          # ∂ψ/∂y
        psi_pad_x = F.pad(psi, (1, 1, 0, 0), mode="replicate")
        djy = F.conv2d(psi_pad_x, kx).neg()    # -∂ψ/∂x
        return torch.cat([djx, djy], dim=1)

    # ------------------------------------------------------------------
    def _stream(
        self,
        state_in: torch.Tensor,
        prev_phys: torch.Tensor,
        fluid_mask: torch.Tensor,
        delta_ctu_seq: Optional[torch.Tensor],
        step_idx: int,
        collect_metrics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """STREAM：semi-Lagrangian backward warp。回傳 (warped_phys, metrics)。

        所有 warp 計算在 float32（AMP 安全）。位移由資料預測、連續 sub-pixel、
        全 physics channel 共用同一張位移場（advection 語意），後向 gather。

        Args:
            state_in: 步首完整 state，形狀 [B, state_dim, H, W]。
            prev_phys: 步首 physics channels，形狀 [B, physics_dim, H, W]。
            fluid_mask: 流體遮罩，形狀 [B, 1, H, W]，solid 區為 0。
            delta_ctu_seq: 每步 delta CTU 序列 [B, n_steps]，None 表示不使用時間條件。
            step_idx: 當前 outer step 索引，用於取 delta_ctu_seq[:, step_idx]。
            collect_metrics: 是否收集位移診斷統計，False 時回傳空 dict（省略計算）。

        Returns:
            warped: 平流後 physics，形狀 [B, physics_dim, H, W]。
            metrics: 位移診斷指標字典（僅 collect_metrics=True 時非空）。
        """
        if self._skip_transport:
            return prev_phys, {}
        b, _, h, w = state_in.shape
        state_coarse = downsample_state(state_in, self._warp_ratio, mode="area")
        feat = self.warp_encoder(state_coarse)
        # disp hook：base = 自由學 tanh(disp_head)；子類（HF-1e）可加 velocity-conditioned 項。
        disp_coarse = self._disp_coarse(feat, state_coarse)      # [B,2,hr,wr]

        # delta_ctu 決定位移尺度（大 Δt → 大 transport span，對應 u·Δt 的 Δt）。
        if delta_ctu_seq is not None:
            dt_i = delta_ctu_seq[:, step_idx].to(device=feat.device, dtype=feat.dtype)
            dt_scale = dt_i.clamp(0.0, self._disp_ctu_cap)[:, None, None, None]
        else:
            dt_scale = torch.ones(b, 1, 1, 1, device=feat.device, dtype=feat.dtype)
        disp_px_coarse = disp_coarse * (self._disp_max_px * dt_scale)
        # 此處單位為 **full-resolution px**：下一行的 upsample_state 只改網格尺寸、
        # 不縮放數值，故與 coarse displacement 相差一個 warp_ratio（canonical 為 8 倍）。

        disp_px = upsample_state(disp_px_coarse, (h, w), mode="bilinear")  # [B,2,H,W]

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
            # base - disp = 後向 gather（往回看出發點），semi-Lagrangian。
            sample_grid = torch.stack(
                [base_grid[..., 0] - dx_norm, base_grid[..., 1] - dy_norm], dim=-1
            )
            warped = F.grid_sample(
                prev_f, sample_grid, mode=self._warp_mode,   # V2-D：bilinear|bicubic
                padding_mode="border", align_corners=False,
            )
        warped = warped.to(prev_phys.dtype) * fluid_mask

        if not collect_metrics:
            return warped, {}

        # 位移統計（detached，供 trainer logging）。
        with torch.no_grad():
            disp_mag = torch.sqrt(disp_px[:, 0] ** 2 + disp_px[:, 1] ** 2).float()  # [B,H,W]
            fm = fluid_mask[:, 0].float()
            denom = fm.sum().clamp_min(1.0)
            disp_mean = (disp_mag * fm).sum() / denom
            flat = disp_mag[fm > 0.5]
            disp_p95 = torch.quantile(flat, 0.95) if flat.numel() > 0 else disp_mag.new_zeros(())
            disp_max = flat.max() if flat.numel() > 0 else disp_mag.new_zeros(())
            # disp_smoothness：位移場空間梯度 L1（越大越不平滑），供 transport 正則監控。
            dpx = disp_px.float()
            d_dx = (dpx[:, :, :, 1:] - dpx[:, :, :, :-1]).abs().mean()
            d_dy = (dpx[:, :, 1:, :] - dpx[:, :, :-1, :]).abs().mean()
            disp_smoothness = d_dx + d_dy
        metrics: dict[str, torch.Tensor] = {
            "disp_mean_px": disp_mean, "disp_p95_px": disp_p95, "disp_max_px": disp_max,
            "disp_smoothness": disp_smoothness,
        }
        return warped, metrics

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
        mean_flow_norm: torch.Tensor | None = None,
        mean_films: dict | None = None,
        ke_anchor: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """執行 n_steps outer recurrent step（串行 stream → collide 版）。

        每步先 STREAM（semi-Lagrangian backward warp 平流 physics），再 COLLIDE
        （multirate residual 作用在平流後的 state），最後套 KE rescale（若啟用）。

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
            mean_flow_norm: 平均流場正規化張量 [B, physics_dim, H, W]，
                mean_film 啟用時使用；None 表示不使用 mean conditioning。

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
        coarse_masks = [downsample_mask(fluid_mask, ratio, threshold=0.5) for ratio in ratios]

        # ── mean_film（turb_module）：per-rollout 一次，per-scale (γ_m, β_m) 快取 ──
        # key=ratio（1=full-res fine cell）。mean 在 rollout 內為常數，不逐步重算。
        # caller（turb_step）可預傳 mean_films（跨 self-preroll/preroll/grad 三次 forward
        # 共用一次 encode）；未傳則內部算（向後相容，eval/test 路徑）。
        fs_mask = None  # P1 free-stream 衰減 mask（mean 常數→算一次）
        prod_field = None  # §9.1(c) production forcing 場 λ·shear_norm·wake_gate（mean 常數→算一次）
        if self.mean_encoder is not None and mean_flow_norm is not None:
            mean_full = mean_flow_norm.to(dtype=state.dtype, device=state.device)
            self._mean_runtime = {"mean_full": mean_full}
            if mean_films is None:
                mean_films = self.encode_mean_films(mean_full)
            # fs_mask 觸發條件擴充：fs_decay / region hf_damp / §9.1(b) wake-gated noise 皆需
            _need_fs = (self._fs_decay > 0.0
                        or (self._fs_hf_damp > 0.0 and not self._fs_hf_damp_global)
                        or self._noise_sigma > 0.0)
            if _need_fs:
                fs_mask = self._compute_fs_mask(mean_full, fluid_mask)
            if self._production_weight > 0.0:
                prod_field = self._compute_prod_field(mean_full, fluid_mask)
        else:
            self._mean_runtime = None
            mean_films = {}

        # ceiling 模式下 caller 未傳 anchor 會使同一顆 ckpt 在訓練（GT 錨）、論文評估
        # （無錨）、部署（running-min 錨）走三套不同算子，故直接報錯：錨點一律由
        # `tes.inference.protocol.resolve_ke_anchor` 產生，呼叫端漏傳即中止，
        # 不得有「看起來有修正器、實際沒有」的行為。
        if self._ke_anchor_mode == "ceiling" and ke_anchor is None:
            raise RuntimeError(
                "ke_anchor_mode='ceiling' 但 forward() 未收到 ke_anchor。"
                "三態協定要求錨點由 tes.inference.protocol.resolve_ke_anchor(cfg, state, "
                "fluid_mask) 產生後顯式傳入；若本路徑不應套 ceiling，請把 config 的 "
                "transport.ke_anchor_mode 設為 'off'。"
            )

        trajectory: list[torch.Tensor] = []
        stream_acc: dict[str, torch.Tensor] = {}
        collide_residual_ratio_acc = state.new_zeros(())
        collect_aux = bool(getattr(self, "_collect_aux_metrics", False))
        if not collect_aux:
            self._aux_metrics = {}
        _cumul_ctu = 0.0  # LC-1 [A17]：追蹤累計 CTU，用於 ke_rescale warmup gate

        for i in range(n_steps):
            state_in = state
            prev_phys = state_in[:, sd:sd + pd]

            # ---- STREAM: semi-Lagrangian warp prev physics ----
            warped, sm = self._stream(
                state_in, prev_phys, fluid_mask, delta_ctu_seq, i,
                collect_metrics=collect_aux,
            )
            # 把 physics 換成平流後的場（static / hidden 不動）→ streamed state。
            if hd > 0:
                state = torch.cat([state_in[:, :sd], warped, state_in[:, sd + pd:]], dim=1)
            else:
                state = torch.cat([state_in[:, :sd], warped], dim=1)

            # ---- COLLIDE: residual path 作用在平流後的 state ----
            # coarse (沿用 HMultirate 多 ratio residual)
            coarse_sum = state.new_zeros(state.shape[0], pd, h_full, w_full)
            if self._psi_curl_momentum:
                psi_full = state.new_zeros(state.shape[0], 1, h_full, w_full)
            for level_idx, (ratio, coarse_cell, coarse_mask) in enumerate(
                zip(ratios, self.coarse_cells, coarse_masks)
            ):
                gamma_c, beta_c, dt_c = self._encode_cond(
                    cond_norm, delta_ctu_seq, cumul_ctu_norm_seq, i,
                    time_conditioned, per_step_cumul,
                    level_id=float(level_idx), ratio=ratio, dt_scale=1.0,
                    gamma_g=gamma_g, beta_g=beta_g,
                )
                gamma_c, beta_c = self._add_mean_film(
                    gamma_c, beta_c, mean_films.get(ratio)
                )
                cs = downsample_state(state, ratio, mode=self._ds_mode)   # D1-surgical: area→blur(BlurPool)
                if self._psi_curl_momentum:
                    csn, psi_c = self._ckpt_cell(coarse_cell, cs, gamma_c, beta_c, coarse_mask, dt_c)
                    # ★psi 上採至 full-res 才取 curl（見 _curl_psi 呼叫處），非在 coarse 解析度
                    psi_full = psi_full + upsample_state(psi_c, (h_full, w_full), mode=self._psi_up_mode)  # D1: bilinear→blur
                else:
                    csn = self._ckpt_cell(coarse_cell, cs, gamma_c, beta_c, coarse_mask, dt_c)
                delta = csn[:, sd:sd + pd] - cs[:, sd:sd + pd]
                # config: multiscale.coarse_delta_mask。`coarse_mask` 由
                #   `downsample_mask(..., threshold=0.5)` 產生，判的是「過半為固體」
                #   而非「整格固體」，故本開關影響的粗格含部分為流體的邊界格。
                #   粗格若過半為固體，`cs[phys]` 是 pooled 的固體值、`csn[phys]` 是
                #   cell 對它的輸出，兩者相減得到一個**不可學**的 `−pooled_physics` 殘差；
                #   它接著被 bilinear 上採，把值灌進建物周圍的流體格。
                #   乘上 coarse_mask 即切斷。
                #   預設 False＝沿用舊行為（既有 ckpt 零變更）。
                if self._coarse_delta_mask:
                    delta = delta * coarse_mask
                coarse_sum = coarse_sum + upsample_state(delta, (h_full, w_full), mode=self._coarse_upsample_mode)
            # ---- R1：coarse 殘差 anti-image 低通（削 upsample 折疊鏡像 grain，不傷低-k 真渦）----
            if self._coarse_aa_sigma > 0.0:
                ker = self._coarse_aa_ker.to(coarse_sum.dtype)
                pad = ker.shape[-1] // 2
                cs_ch = [
                    F.conv2d(coarse_sum[:, c:c + 1], ker, padding=pad) if c in (3, 5)
                    else coarse_sum[:, c:c + 1]
                    for c in range(pd)
                ]
                coarse_sum = torch.cat(cs_ch, dim=1)
            # collide 把 residual 加到「平流後」的 physics（warped），非步首 prev_phys。
            new_phys = (warped + coarse_sum) * fluid_mask
            if self._psi_curl_momentum:
                # ★coarse 對 jx/jy 的貢獻：full-res ψ 取 curl 後疊加（增量無散度，限流體內部；
                # 保證範圍見 `_curl_psi` docstring，不得寫成 div-free by construction）。
                # coarse_sum 的 jx/jy 分量此時 ≈0（pass-through相減），new_phys[3]/[5] 仍是 warped
                # 的 pass-through 值（=prev_phys，因 STREAM 通常 skip），curl 疊加才是真正更新。
                curl_c = self._curl_psi(psi_full)
                new_phys = new_phys.clone()
                new_phys[:, 3:4] = (new_phys[:, 3:4] + curl_c[:, 0:1]) * fluid_mask
                new_phys[:, 5:6] = (new_phys[:, 5:6] + curl_c[:, 1:2]) * fluid_mask

            if hd > 0:
                state = torch.cat([state[:, :sd], new_phys, state[:, sd + pd:]], dim=1)
            else:
                state = torch.cat([state[:, :sd], new_phys], dim=1)

            # fine substeps（亦作用在平流後的 state）
            gamma_f, beta_f, dt_f = self._encode_cond(
                cond_norm, delta_ctu_seq, cumul_ctu_norm_seq, i,
                time_conditioned, per_step_cumul,
                level_id=float(len(ratios)), ratio=1, dt_scale=dt_fine,
                gamma_g=gamma_g, beta_g=beta_g,
            )
            gamma_f, beta_f = self._add_mean_film(gamma_f, beta_f, mean_films.get(1))
            for _ in range(sub):
                if self._psi_curl_momentum:
                    # fine 已在 full-res，psi_f 免上採，直接 curl：coarse 與 fine 的 curl
                    # 分開算，兩者的**增量**各自在流體內部無散度，線性相加後保持該性質
                    # ——保持的仍只是增量性質，不是 div(j)=0。
                    new_rest, psi_f = self._ckpt_cell(self.fine_cell, state, gamma_f, beta_f, fluid_mask, dt_f)
                    curl_f = self._curl_psi(psi_f)
                    new_rest = new_rest.clone()
                    jx_abs, jy_abs = sd + 3, sd + 5
                    new_rest[:, jx_abs:jx_abs + 1] = (new_rest[:, jx_abs:jx_abs + 1] + curl_f[:, 0:1]) * fluid_mask
                    new_rest[:, jy_abs:jy_abs + 1] = (new_rest[:, jy_abs:jy_abs + 1] + curl_f[:, 1:2]) * fluid_mask
                    state = new_rest
                else:
                    state = self._ckpt_cell(self.fine_cell, state, gamma_f, beta_f, fluid_mask, dt_f)

            collided_phys = state[:, sd:sd + pd] * fluid_mask
            if hd > 0:
                # tanh-bound hidden 每 outer step → 防無界漂移。
                #   hidden 由 fine_cell 逐步寫入，若無值域約束，長自迴歸（41-100 CTU）
                #   累積漂移會導致 speckle 發散。
                #   tanh 近 0 ≈ identity（小 hidden 不擾動）、極端才壓回 (-1,1)，
                #   保留 hidden 高頻結構記憶同時避免無界爆炸（DyNCA §2.2）。
                hidden_bounded = torch.tanh(state[:, sd + pd:])
                state = torch.cat([state[:, :sd], collided_phys, hidden_bounded], dim=1)
            else:
                state = torch.cat([state[:, :sd], collided_phys], dim=1)

            # ---- LC-1 [A15]：ke_rescale 移到 fine substeps 後——完整 step KE 守恆 ----
            # LC-1 [A17]：warmup gate——cumul_ctu < ke_rescale_start_ctu 時 bypass（spin-up 不限 KE）。
            # prev_phys(步首) vs fine substeps 後完整輸出；detach+gmin=0+gmax=1.15+α=1。
            if self._ke_rescale and _cumul_ctu >= self._ke_rescale_start_ctu:
                bsz = state.shape[0]
                full_out = state[:, sd:sd + pd]
                with torch.no_grad():
                    with torch.autocast(device_type=state_in.device.type, enabled=False):
                        jp = prev_phys.float()                       # 步首幅值參考（STREAM 前）
                        jn = full_out.float()                        # fine substeps 後完整輸出
                        fm = fluid_mask.float()
                        denom = fm.sum(dim=(1, 2, 3)).clamp_min(1.0)  # [B]
                        ke_in = ((jp[:, 3:4] ** 2 + jp[:, 5:6] ** 2) * fm).sum(dim=(1, 2, 3)) / denom
                        ke_out = ((jn[:, 3:4] ** 2 + jn[:, 5:6] ** 2) * fm).sum(dim=(1, 2, 3)) / denom
                        g = (ke_in / (ke_out + 1e-8)).sqrt().clamp(self._ke_rescale_gmin, self._ke_rescale_gmax)  # [B]
                a = self.ke_rescale_alpha
                s = (1.0 - a + a * g.view(bsz, 1, 1, 1)).to(full_out.dtype)  # [B,1,1,1]，α=1→s=g（detached）
                corrected = torch.cat(
                    [full_out[:, c:c + 1] * s if c in (3, 5) else full_out[:, c:c + 1]
                     for c in range(pd)], dim=1,
                ) * fluid_mask
                if hd > 0:
                    state = torch.cat([state[:, :sd], corrected, state[:, sd + pd:]], dim=1)
                else:
                    state = torch.cat([state[:, :sd], corrected], dim=1)

            # ---- ★iter7：IC-anchored KE ceiling（只壓不補；治 re-inject 能量棘輪）----
            #   ke_out > ref*margin（ref=caller 傳入 anchor KE）才動：g=√(ref/ke_out)∈[gmin,1]，僅乘 jx/jy。
            #   低於 ref → g=1 不動（spin-up/damage recovery/耗散完全自由）。detached scalar、AMP-safe，
            #   仿 ke_rescale 數值慣例。mode='off' 或 anchor=None → bypass（逐 bit identity）。
            if self._ke_anchor_mode == "ceiling" and ke_anchor is not None:
                full_out = state[:, sd:sd + pd]
                with torch.no_grad():
                    with torch.autocast(device_type=state_in.device.type, enabled=False):
                        jn = full_out.float(); fm = fluid_mask.float()
                        denom = fm.sum(dim=(1, 2, 3)).clamp_min(1.0)
                        ke_out = ((jn[:, 3:4] ** 2 + jn[:, 5:6] ** 2) * fm).sum(dim=(1, 2, 3)) / denom
                        ref = ke_anchor.float().to(ke_out.device) * self._ke_anchor_margin  # [B]
                        g = (ref / (ke_out + 1e-12)).sqrt().clamp(self._ke_anchor_gmin, 1.0)  # ≤1 只壓
                s = g.view(-1, 1, 1, 1).to(full_out.dtype)
                capped = torch.cat(
                    [full_out[:, c:c + 1] * s if c in (3, 5) else full_out[:, c:c + 1]
                     for c in range(pd)], dim=1,
                ) * fluid_mask
                if hd > 0:
                    state = torch.cat([state[:, :sd], capped, state[:, sd + pd:]], dim=1)
                else:
                    state = torch.cat([state[:, :sd], capped], dim=1)

            # ---- P1：free-stream fluct blanket 衰減（鈍；fs_decay>0 時）----
            if fs_mask is not None and self._fs_decay > 0.0:
                decay = (1.0 - self._fs_decay * fs_mask).to(state.dtype)  # [B,1,H,W]
                phys = state[:, sd:sd + pd]
                dphys = torch.cat(
                    [phys[:, c:c + 1] * decay if c in (3, 5) else phys[:, c:c + 1]
                     for c in range(pd)], dim=1,
                ) * fluid_mask
                if hd > 0:
                    state = torch.cat([state[:, :sd], dphys, state[:, sd + pd:]], dim=1)
                else:
                    state = torch.cat([state[:, :sd], dphys], dim=1)

            # ---- #1：high-k 選擇性衰減（移除 top-octave grain、保 mid-k 渦）----
            # fluct −= a·(fluct − blur(fluct))；a=α(全域 scale-gated) 或 α·fs(region-gated)。
            # 全域：治 region 把對流進自由流的真渦誤殺（ens↓）；小核→(fluct−blur)≈per-pixel grain。
            if self._fs_hf_damp > 0.0 and (self._fs_hf_damp_global or fs_mask is not None):
                phys = state[:, sd:sd + pd]
                ker = self._gauss_ker.to(phys.dtype)
                a = (self._fs_hf_damp if self._fs_hf_damp_global
                     else self._fs_hf_damp * fs_mask)  # scalar 或 [B,1,H,W]
                chans = []
                for c in range(pd):
                    xc = phys[:, c:c + 1]
                    if c in (3, 5):
                        blur = F.conv2d(xc, ker, padding=self._hf_pad)
                        xc = xc - a * (xc - blur)  # 移除 α·fs 比例的 high-freq
                    chans.append(xc)
                dphys = torch.cat(chans, dim=1) * fluid_mask
                if hd > 0:
                    state = torch.cat([state[:, :sd], dphys, state[:, sd + pd:]], dim=1)
                else:
                    state = torch.cat([state[:, :sd], dphys], dim=1)

            # ---- §9.1(c) production forcing：fluct ×(1+λ·shear_norm·wake_gate)（給渦能量源，治 ens↓）----
            if prod_field is not None and self._production_weight > 0.0:
                factor = (1.0 + prod_field).to(state.dtype)  # [B,1,H,W]
                phys = state[:, sd:sd + pd]
                pphys = torch.cat(
                    [phys[:, c:c + 1] * factor if c in (3, 5) else phys[:, c:c + 1]
                     for c in range(pd)], dim=1,
                ) * fluid_mask
                if hd > 0:
                    state = torch.cat([state[:, :sd], pphys, state[:, sd + pd:]], dim=1)
                else:
                    state = torch.cat([state[:, :sd], pphys], dim=1)

            # ---- §9.1(b) 生成式 noise：fluct(jx/jy) += σ·randn·wake（wake=1−fs_mask，給渦自發變化）----
            if self._noise_sigma > 0.0 and fs_mask is not None:
                wake = (1.0 - fs_mask).to(state.dtype)  # [B,1,H,W]，wake→1、自由流→0
                phys = state[:, sd:sd + pd]
                nphys = torch.cat(
                    [phys[:, c:c + 1] + self._noise_sigma * torch.randn_like(phys[:, c:c + 1]) * wake
                     if c in (3, 5) else phys[:, c:c + 1]
                     for c in range(pd)], dim=1,
                ) * fluid_mask
                if hd > 0:
                    state = torch.cat([state[:, :sd], nphys, state[:, sd + pd:]], dim=1)
                else:
                    state = torch.cat([state[:, :sd], nphys], dim=1)

            # ---- 狀態空間閉包（norm_frozen 的 clamp_state）----
            # ★ 正規化的像是閉盒 [-CLIP, CLIP]^9；每步夾回去，「更新算子把盒映進盒」
            #   才是構造性事實而非經驗觀察。實測不夾時隨機權重 500 步後 physics
            #   通道會衝到 CLIP 的 1500 倍 —— 模型自此在完全沒見過的量級上運作。
            #   （反正規化本身另有 clip，故物理讀數在任何情況下都合法；
            #     這裡夾的是**狀態**，兩者角色不同、不可互相取代。）
            #   ★ straight-through：夾的動作在 `no_grad` 下算，再以恆等反向併回。
            #     兩個理由，與 phys_adm 的 straight-through 同源：
            #     (a) **記憶體**。`clamp` 進計算圖會逐 outer step 存下遮罩，k=24 時
            #         累積數 GB；實測直接夾使 peak 從 8.3 GB 漲到 97 % 顯存、
            #         每步由 1.35 s 劣化到 5.4 s（allocator 抖動）。
            #     (b) **語義**。硬夾的真導數在界外為 0 → 越界的像素永遠收不到梯度、
            #         也就永遠學不會不越界。straight-through 讓梯度原樣傳回夾之前的值。
            if self.state_clip > 0.0:
                _c = self.state_clip
                with torch.no_grad():
                    _d_phys = state[:, sd:sd + pd].clamp(-_c, _c) - state[:, sd:sd + pd]
                    _d_hid = (state[:, sd + pd:].clamp(-1.0, 1.0) - state[:, sd + pd:]
                              if hd > 0 else None)
                _phys = state[:, sd:sd + pd] + _d_phys
                if hd > 0:
                    state = torch.cat(
                        [state[:, :sd], _phys, state[:, sd + pd:] + _d_hid], dim=1)
                else:
                    state = torch.cat([state[:, :sd], _phys], dim=1)

            # ---- 物理可容許性投影（本步的最後一個算子）----
            # ★ 角色已降級：密度恆正與次音速由 norm_frozen 的盒**構造性**保證
            #   （見 `admissible_box()`），投影只剩處理盒表達不了的其餘耦合約束。
            # 位置必須是最後：軌跡記錄與下一步的輸入都取這裡的 state，投影放在任何
            # 其他算子之前都會被後續算子再次推出可容許集合。停用時逐位恆等。
            if self.phys_adm.enabled:
                adm_phys = self.phys_adm(state[:, sd:sd + pd], fluid_mask,
                                         collect_counts=collect_aux)
                if hd > 0:
                    state = torch.cat([state[:, :sd], adm_phys, state[:, sd + pd:]], dim=1)
                else:
                    state = torch.cat([state[:, :sd], adm_phys], dim=1)

            # 更新累計 CTU（用於 ke_rescale warmup gate）
            if delta_ctu_seq is not None:
                _cumul_ctu += float(delta_ctu_seq[:, i].mean().item())
            else:
                _cumul_ctu += 1.0

            if collect_aux:
                with torch.no_grad():
                    for k, v in sm.items():
                        stream_acc[k] = stream_acc.get(k, state.new_zeros(())) + v
                    # collide 修正量 vs 平流量：診斷 collide 在平流後改了多少。
                    collide_mag = (collided_phys - warped).abs().float().mean()
                    stream_mag = warped.abs().float().mean()
                    collide_residual_ratio_acc = (
                        collide_residual_ratio_acc + collide_mag / stream_mag.clamp_min(1e-12)
                    )

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
            "collide_residual_ratio": collide_residual_ratio_acc / n_outer,
            "single_step_rf_diameter_px_est": float(4 * max(ratios)),
        }
        for k, v in stream_acc.items():
            self._aux_metrics[k] = v / n_outer
        # 觸界計數（最後一步）：投影是保險絲，不是修正器。長期非零代表模型在對抗
        # 約束、該回頭看訓練，故一律上報而非藏起來。
        for k, v in self.phys_adm.last_counts.items():
            self._aux_metrics[f"adm_{k}"] = float(v)

        self._mean_runtime = None  # 清 runtime，避免跨 forward 殘留

        if return_trajectory:
            return state, trajectory
        return state


__all__ = ["H3TStreamCollideNCA"]
