"""單步 Markov NCA 更新核心：感知鄰域 → FiLM 調製 → 投影 → 加法更新。

靜態通道不更新；固體區域強制歸零；支援 LP 模式（3×3 Conv）和 MRP 模式。

感知層模式：
  LP  (預設): 單一可學習 3×3 Conv2d，直接對完整 state 做鄰域感知。
  MRP (選配): Multi-Receptor Perception，嵌入 LBM 物理先驗的固定 kernel 感知器，
              對不同物理意義的受體分別感知後合併（見 TES_MRP）。

更新策略：物理通道加法更新（identity + delta），static 通道保持不變，
固體區（fluid_mask=0）強制歸零，確保邊界條件一致性。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lbmnca.nca.tes.models.multiscale.ops import downsample_state, upsample_state


def _box1d(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    """沿 `dim` 的 k 寬滑動**和**，用前綴和實作 → **O(1)／像素，與 k 無關**。"""
    pad = k // 2
    pads = [0, 0] * (x.dim() - 1 - (dim % x.dim()))
    pads += [pad, pad]
    xp = F.pad(x, pads)
    c = xp.cumsum(dim)
    zero = torch.zeros_like(c.narrow(dim, 0, 1))
    c = torch.cat([zero, c], dim=dim)          # 前綴和，長度 L+2*pad+1
    L = x.shape[dim]
    return c.narrow(dim, k, L) - c.narrow(dim, 0, L)


def _separable_box(x: torch.Tensor, k: int) -> torch.Tensor:
    """k×k stride-1 box 平均。**前綴和實作，成本與 k 無關**。

    ★三版演進，兩次都是被實測打臉才改對：
      v1 `F.avg_pool2d(x, k, stride=1)` —— O(k²)／像素，k=31 時 961 ops。
      v2 拆成兩次 1D `avg_pool2d` —— O(k)，CPU 單步 188ms→74ms，看起來夠了。
      v3（本版）前綴和 —— **O(1)**。v2 在 GPU 上實測是 **8.73 s/step**
         （其他臂 1.2–1.9），一臂就要 18.7 小時、吃光整個 24H 預算。
         CPU 的 3.7× 差距在 GPU 上放大成 5–7×：大 kernel 的 stride-1 池化
         是記憶體頻寬受限，PyTorch 沒有對它最佳化。

    ★邊界：分母用**同樣管線跑一個全 1 張量**得到，等價於
    `count_include_pad=False` —— 只對實際存在的像素取平均。
    預設把零填充算進分母會讓場邊緣被系統性拉暗，而流體入出口正在邊界上。

    Args:
        x: [B, C, H, W]。
        k: 奇數 kernel 大小。

    Returns:
        同形狀的局部平均場。
    """
    num = _box1d(_box1d(x, k, -1), k, -2)
    ones = torch.ones_like(x[:, :1])
    den = _box1d(_box1d(ones, k, -1), k, -2)
    return num / den


from ...config import MRPConfig
from ..baselines.ar_fno import SpectralConv2d
from ..mrp_cell import TES_MRP


class MarkovNCACell(nn.Module):
    """單步 NCA 更新：感知鄰域 → FiLM 條件調制 → 物理通道加法更新。

    架構（單步，LP 模式）：
        state [B, state_dim, H, W]
        └→ Conv2d(state_dim, nca_ch, 3, padding=1)  # LP 感知
        └→ SiLU
        └→ FiLM: h = γ[:,:,None,None] * h + β[:,:,None,None]
        └→ SiLU
        └→ Conv2d(nca_ch, dynamic_dim, 1)
        └→ additive update + IC masking

    架構（單步，MRP 模式）：
        state [B, state_dim, H, W]
        └→ TES_MRP(state) → [B, combined_ch, H, W]  # 5 受體感知
        └→ Conv2d(combined_ch, nca_ch, 1)            # fusion
        └→ SiLU → FiLM → SiLU → project → update

    Args:
        state_dim: 狀態向量通道數，預設 13（4 static + 9 physics）。
        nca_channels: 內部特徵維度，同時為 FiLM γ/β 的維度，預設 64。
        physics_dim: 物理通道數（加法更新目標），預設 9。
        hidden_dim: hidden 通道數；0 代表無 hidden 狀態（baseline 設定）。
        mrp_config: MRPConfig 實例或 None。None 或 enabled=False 時使用 LP 模式。
    """

    # physics 通道相對 dynamic 區塊起點的索引（jx=3, jy=5；ARCHITECTURE §5.1，與 operator_gated.py 一致）
    _JX_REL: int = 3
    _JY_REL: int = 5

    def __init__(
        self,
        state_dim: int = 13,
        nca_channels: int = 64,
        physics_dim: int = 9,
        hidden_dim: int = 0,
        mrp_config: MRPConfig | None = None,
        spectral_update: bool = False,
        spectral_modes: int = 24,
        msp_scales: list[int] | None = None,
        msp_mode: str = "resample",
        msp_down_mode: str = "blur",
        msp_up_mode: str = "blur",
        psi_curl_momentum: bool = False,
    ) -> None:
        super().__init__()
        self._physics_dim: int = physics_dim
        self._hidden_dim: int = hidden_dim
        self._dynamic_dim: int = physics_dim + hidden_dim
        self._static_dim: int = state_dim - self._dynamic_dim  # 應為 4
        self._use_mrp: bool = mrp_config is not None and mrp_config.enabled
        # MSP 下採因子列表（[1]=退化單尺度，[1,2,4]=P2 MSP）
        self._msp_scales: list[int] = [int(s) for s in (msp_scales or [1])]
        # ★MSP 取樣模式。舊實作寫死 avg_pool2d + bilinear，兩端都會產生格紋：
        #   降採 avg_pool 直接 subsample → 高頻 folding；
        #   升採 bilinear 的二階導在 coarse cell 邊界不連續 → imaging replica，
        #   而 delta 每步累加、curl 又放大導數，最後就是**棋盤狀大像素格紋**。
        #   `multiscale/ops.py` 的 "blur" 模式本來就是為此寫的，這裡改為沿用。
        self._msp_mode: str = str(msp_mode)
        self._msp_down_mode: str = str(msp_down_mode)
        self._msp_up_mode: str = str(msp_up_mode)
        # ψ-curl 動量頭：jx/jy 移出 project，改由 curl(ψ) 重建。
        # ★保證僅及於「每步**增量**無散度、流體內部」；不移除既有散度、界面不保證
        # （詳見 config/_legacy.py::psi_curl_momentum 的 (a)(b)(c)）。
        self._psi_curl_momentum: bool = bool(psi_curl_momentum)
        if self._psi_curl_momentum:
            self._direct_idx: list[int] = [
                c for c in range(self._dynamic_dim) if c not in (self._JX_REL, self._JY_REL)
            ]

        if self._use_mrp:
            self.mrp = TES_MRP(state_dim, physics_dim, mrp_config)
            # concat-fuse：in_ch = combined_ch × n_scales（[1]=退化單尺度，向後相容）
            self.fusion = nn.Conv2d(
                self.mrp.combined_ch * len(self._msp_scales), nca_channels, kernel_size=1, bias=True
            )
        else:
            self.perceive = nn.Conv2d(
                state_dim, nca_channels, kernel_size=3, padding=1, bias=True
            )

        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()
        # psi_curl_momentum 啟用時，project 輸出排除 jx/jy 兩個 slot（改由 curl(ψ) 重建）
        _proj_out = self._dynamic_dim - 2 if self._psi_curl_momentum else self._dynamic_dim
        self.project = nn.Conv2d(nca_channels, _proj_out, kernel_size=1, bias=True)

        # 物理更新量初始化為零（讓模型從 identity 開始學習，避免訓練初期 loss 爆炸）
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

        if self._psi_curl_momentum:
            # ψ delta head：純量場，curl(ψ)=(∂ψ/∂y,-∂ψ/∂x)；div(curl ψ)=0 對共用的中央差分核、
            # 在流體內部成立（機器精度）。zero-init → 起步 ψ=0 → identity。
            self.project_psi = nn.Conv2d(nca_channels, 1, kernel_size=1, bias=True)
            nn.init.zeros_(self.project_psi.weight)
            nn.init.zeros_(self.project_psi.bias)

        # 頻域 update 分支（FNO 風）：spectral 低模 → 全域低/中頻 delta（補局部 conv 缺的全域協調，
        # 攻過平滑/塌均勻）；local project（上方 1×1）保高頻。zero-init → 起步=純 local（identity）。
        self._spectral_update: bool = bool(spectral_update)
        if self._spectral_update:
            self.project_spectral = SpectralConv2d(
                nca_channels, _proj_out, int(spectral_modes), int(spectral_modes)
            )
            for _w in (
                self.project_spectral.weight1_real, self.project_spectral.weight1_imag,
                self.project_spectral.weight2_real, self.project_spectral.weight2_imag,
            ):
                nn.init.zeros_(_w)

        # Dy-C：per-CTU rate 軟夾（c·tanh(rate/c)）。0=停用。由 model_factory 統一設定。
        # 限 collide rate 幅度 → 限每步物理變化 → 壓 Jacobian spectral radius（防 long-roll 發散）。
        # tanh(0)=0 → 保 zero-init identity。clip 在 ×delta_ctu 之前（夾 rate 而非 actual change，步長一致）。
        self.update_clip: float = 0.0
        # ★stochastic update（Growing NCA，Mordvintsev）：train-only per-cell Bernoulli(rate) gate delta，
        # 打破同步 lock-step 相位相干（攻 band=相位鎖定中波長吸引子）；deploy(eval) full-sync（rate 不作用）。
        # 1.0=全同步（現行，identity 向後相容）。由 factory 依 cfg.stoch_update_rate 注入（比照 update_clip）。
        self.stoch_update_rate: float = 1.0

    def forward(
        self,
        state: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        fluid_mask: torch.Tensor,
        delta_ctu: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """執行一步 NCA 更新（加法更新：new_state = state + delta，static 通道不參與更新）。

        加法更新邏輯：
          1. 感知鄰域（LP 3×3 Conv 或 MRP 多受體），取得局部特徵 h。
          2. FiLM 條件調制：h = γ * h + β（γ/β 來自 TESConditionEncoder）。
          3. project 投影：delta = Conv1×1(h) → [B, dynamic_dim, H, W]，
             dynamic_dim = physics_dim + hidden_dim。
          4. update_clip 軟夾（Dy-C）：若 update_clip > 0，對 delta 套用
             c·tanh(delta/c)，限制每步最大 rate，壓制 Jacobian spectral radius。
          5. delta_ctu 縮放（E1i 模式）：rate × Δt → 實際物理變化量。
          6. 加法更新 physics + hidden 通道；固體區乘以 fluid_mask 強制歸零。
          7. 拼回 static 通道（SDF、mask 等），輸出完整 new_state。

        ``psi_curl_momentum=True`` 時：jx/jy 兩個 slot 移出 project，
        改由額外的純量 ψ delta 頭（``project_psi``）產生；本函式回傳 ``(new_state, psi_delta)``，
        其中 ``new_state`` 的 jx/jy 通道是 **pass-through**（未加 curl 分量，caller 負責在 coarse
        上採至 full-res 後才取 curl 疊加，見 ``h3t_streamcollide.py::_curl_psi``）。此設計避免
        coarse pathway 在 coarse 解析度取 curl 後再 bilinear 上採，導致 full-res 的**增量**
        不再無散度（bilinear 插值不與散度算子交換）。

        ★保證範圍：curl 構造只保證「這一步新增的動量變化」在
        流體內部無散度，**不是** div(j)=0。它不移除 j_n 既有的散度（div(j_{n+1})=div(j_n)），
        且寫回時逐點乘 fluid_mask 會在固/流界面注入散度。完整 (a)(b)(c) 三段限定見
        ``config/_legacy.py::psi_curl_momentum``。

        Args:
            state: [B, state_dim, H, W] 目前狀態（static + physics + hidden）。
            gamma: [B, nca_channels] FiLM 縮放參數（來自 TESConditionEncoder），
                   或 [B, nca_channels, H, W] 空間 FiLM（mean_film 三路相加後，
                   LC decompose turb_module mean conditioning）。
            beta:  [B, nca_channels] 或 [B, nca_channels, H, W] FiLM 平移參數。
            fluid_mask: [B, 1, H, W] float，流體區=1，固體區=0。
            delta_ctu: [B, 1, 1, 1] 或 None。E1i 模式下，將 project 輸出（per-CTU rate）
                       乘以 delta_ctu，使模型學習單位 CTU 的場變化率而非絕對變化量。

        Returns:
            psi_curl_momentum=False：new_state [B, state_dim, H, W]，固體區 physics/hidden=0。
            psi_curl_momentum=True：(new_state, psi_delta)，psi_delta 形狀 [B,1,H,W]，
                尚未取 curl、尚未乘 fluid_mask（原始 per-CTU rate，已套 clip/delta_ctu）。
        """
        # 感知鄰域（MRP 或 LP）
        if self._use_mrp:
            # MSP concat-fuse（P2 DyNCA）：
            #   各尺度下採 state → 同一 MRP → 上採回原解析度 → cat → fusion
            #   fusion in_ch = combined_ch * n_scales（可學 per-scale 權重，無閹割）。
            #   空間不足的尺度補零（保 concat 維度固定=combined_ch*n_scales）。
            #   warm-load：scale-0 block 承接舊單尺度 checkpoint 的 fusion.weight，
            #   其餘 block 補零 → t=0 時 concat-MSP forward 與舊單尺度 forward
            #   數值相等。
            _h, _w = state.shape[-2], state.shape[-1]
            _combined_ch = self.mrp.combined_ch
            parts = []
            for s in self._msp_scales:
                if s > 1 and self._msp_mode != "pool" and (_h < s or _w < s):
                    # 空間不足：補零（保 concat 維度一致；warm-load 此 block weight=0 → 無貢獻）
                    parts.append(torch.zeros(
                        state.shape[0], _combined_ch, _h, _w,
                        device=state.device, dtype=state.dtype,
                    ))
                else:
                    if self._msp_mode == "pool":
                        # ★stride-1 池化脈絡：不降採也不升採 → **零重採樣、零接縫**。
                        #   s 在此解讀為 kernel 大小（奇數）。用 avg 而非 max：
                        #   max 對有號動量場會把均值推偏 23%、並造成 75.5% 的平坦
                        #   平台（塊狀色階斷層），等於用一個視覺缺陷換另一個。
                        #   avg ＝局部平均＝粗粒化 CA cell 的物理意義。
                        st = _separable_box(state, s) if s > 1 else state
                        p = self.mrp(st)
                    else:
                        st = (downsample_state(state, s, self._msp_down_mode)
                              if s > 1 else state)
                        p = self.mrp(st)
                        if s > 1:
                            p = upsample_state(p, (_h, _w), self._msp_up_mode)
                    parts.append(p)
            fused = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
            h = self.fusion(fused)  # [B, combined_ch*n_scales, H, W] → [B, nca_ch, H, W]
        else:
            h = self.perceive(state)           # LP → nca_ch
        h = self.act1(h)

        # FiLM 條件調制：向量 [B, C] 廣播 → [B, C, 1, 1]；空間 [B, C, H, W] 直接逐像素
        if gamma.dim() == 2:
            h = gamma[:, :, None, None] * h + beta[:, :, None, None]
        else:
            h = gamma * h + beta
        h = self.act2(h)

        # 物理更新量（per-CTU rate；若 delta_ctu 提供則縮放至實際物理時間）
        delta = self.project(h)    # [B, proj_out, H, W]  local 高頻分支（psi 模式下 proj_out=dynamic_dim-2）
        if self._spectral_update:  # FNO 風 spectral 低模分支（全域低/中頻；zero-init 起步無貢獻）
            # SpectralConv2d 內部 rfft 強制 fp32 → 外層先 cast，輸出 cast 回 delta dtype
            delta = delta + self.project_spectral(h.float()).to(delta.dtype)

        if self._psi_curl_momentum:
            psi_delta = self.project_psi(h)   # [B,1,H,W]，zero-init → 起步全 0
            if self.update_clip > 0.0:
                c = self.update_clip
                delta = c * torch.tanh(delta / c)
                # ★clip 作用在 ψ 本身 → curl(tanh(ψ)) 的增量仍精確無散度
                #   （限流體內部；限定範圍見本函式 docstring 的「★保證範圍」段）
                psi_delta = c * torch.tanh(psi_delta / c)
            if delta_ctu is not None:
                delta = delta * delta_ctu
                psi_delta = psi_delta * delta_ctu
            if self.training and self.stoch_update_rate < 1.0:   # ★stochastic update：train-only 打破相位相干
                _um = torch.bernoulli(torch.full(
                    (delta.shape[0], 1, delta.shape[-2], delta.shape[-1]),
                    self.stoch_update_rate, device=delta.device, dtype=delta.dtype))
                delta = delta * _um
                psi_delta = psi_delta * _um   # 同 mask gate ψ（訓練期輕破增量無散度性；deploy full-sync 恢復）
            # 重組完整 dynamic_dim 通道：非 jx/jy 位置填 direct delta，jx/jy 位置暫填 0
            # （=pass-through，caller 用 curl(psi) 疊加後才是最終值，見 h3t_streamcollide.py）。
            # ★dtype 須跟 delta 一致（非 state）：AMP autocast 下 delta 是 fp16、state 是 fp32，
            # index_put_（fancy indexing 賦值）嚴格要求來源/目標 dtype 相同，不像一般運算會自動 upcast。
            delta_full = delta.new_zeros(state.shape[0], self._dynamic_dim, state.shape[-2], state.shape[-1])
            delta_full[:, self._direct_idx] = delta
            dynamic = state[:, self._static_dim:] + delta_full
            dynamic = dynamic * fluid_mask
            new_state = torch.cat([state[:, :self._static_dim], dynamic], dim=1)
            return new_state, psi_delta

        if self.update_clip > 0.0:  # Dy-C：軟夾 rate 幅度，壓 ρ(J)；tanh(0)=0 保 identity-init
            c = self.update_clip
            delta = c * torch.tanh(delta / c)
        if delta_ctu is not None:
            delta = delta * delta_ctu  # E1i：rate × Δt → 實際變化量
        if self.training and self.stoch_update_rate < 1.0:   # ★stochastic update：train-only per-cell Bernoulli gate
            _um = torch.bernoulli(torch.full(
                (delta.shape[0], 1, delta.shape[-2], delta.shape[-1]),
                self.stoch_update_rate, device=delta.device, dtype=delta.dtype))
            delta = delta * _um

        # 更新 physics + hidden 通道（static 不變）
        dynamic = state[:, self._static_dim:] + delta
        dynamic = dynamic * fluid_mask   # 固體區歸零（physics + hidden 均歸零）

        new_state = torch.cat([state[:, :self._static_dim], dynamic], dim=1)
        return new_state
