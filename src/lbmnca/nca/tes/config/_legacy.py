"""TES-NCA 訓練設定模組，定義所有訓練超參數的 dataclass 與 YAML 載入介面。"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Optional

import yaml

from ....paths import resolve_placeholders

_logger = logging.getLogger(__name__)


def _check(cond: bool, msg: str) -> None:
    """簡易斷言，違反 dataclass 不變量時拋 ValueError。

    Args:
        cond: 需為 True 的條件。
        msg: 條件不成立時的錯誤訊息。

    Raises:
        ValueError: 當 cond 為 False 時拋出。
    """
    if not cond:
        raise ValueError(msg)


@dataclass
class MRPConfig:
    """Multi-Receptor Perception 感知層設定。

    enabled=False 時 MarkovNCACell 退回單一 LP（向後相容，行為完全相同）。
    各受體可獨立啟用/停用，供消融實驗使用。
    """

    enabled: bool = False
    use_topo: bool = True  # TopoR：Sobel_x + Sobel_y + Laplacian
    use_stream: bool = True  # StreamK：D2Q9 方向平移 kernel
    use_trans: bool = True  # MTransK：MRT D2Q9 M-matrix（9×9）
    num_learnable: int = 32  # AdapR 輸出通道數
    hormone_dim: int = 8  # HormB 輸出通道數
    use_groupnorm: bool = True  # 各受體輸出是否加 GroupNorm(1, C)

    def __post_init__(self) -> None:
        _check(
            self.num_learnable >= 0,
            f"MRPConfig.num_learnable 必須 >=0，得到 {self.num_learnable}",
        )
        _check(
            self.hormone_dim >= 0,
            f"MRPConfig.hormone_dim 必須 >=0，得到 {self.hormone_dim}",
        )


@dataclass
class TESEarlyStopConfig:
    """Plateau-based early stopping 設定（NaN 守衛另見 TESTrainConfig.nan_patience）。

    監控 validation total loss 連續無改善次數（單位：val_interval 步）。
    同時偵測 val_loss 相對最佳值大幅發散（訓練崩潰）。
    """

    enabled: bool = True
    monitor: str = "total"  # val event 中 key 名稱
    patience: int = 10  # 連續 N 次 val 無改善 → 停止
    min_delta: float = 1e-4  # 小於此改善量不計入
    mode: str = "min"  # "min" 或 "max"
    restore_best: bool = True  # 停止時載入最佳權重
    start_step: int = 5_000  # 前 N 步不觸發（warmup 期保護）
    diverge_threshold: float = 5.0  # val_loss / best_val > 此值 → 崩潰停止

    def __post_init__(self) -> None:
        _check(
            self.patience > 0,
            f"TESEarlyStopConfig.patience 必須 >0，得到 {self.patience}",
        )
        _check(
            self.min_delta >= 0,
            f"TESEarlyStopConfig.min_delta 必須 >=0，得到 {self.min_delta}",
        )
        _check(
            self.mode in ("min", "max"),
            f"TESEarlyStopConfig.mode 必須是 'min' 或 'max'，得到 '{self.mode}'",
        )
        _check(
            self.start_step >= 0,
            f"TESEarlyStopConfig.start_step 必須 >=0，得到 {self.start_step}",
        )
        _check(
            self.diverge_threshold > 1.0,
            f"TESEarlyStopConfig.diverge_threshold 必須 >1，得到 {self.diverge_threshold}",
        )


@dataclass
class TESTFCConfig:
    """Teacher Forcing Curriculum 設定。

    Phase1 (0→phase1_end): tf_ratio = tf_ratio_init（全 teacher forcing）
    Phase2 (phase1_end→phase2_end): tf_ratio 線性衰減
    Phase3 (phase2_end→end): tf_ratio = tf_ratio_final（純自迴歸）
    """

    enabled: bool = False
    phase1_end: int = 15_000
    phase2_end: int = 35_000
    tf_ratio_init: float = 1.0  # Phase1 tf_ratio
    tf_ratio_final: float = 0.0  # Phase3 tf_ratio

    def __post_init__(self) -> None:
        _check(
            self.phase1_end >= 0,
            f"TESTFCConfig.phase1_end 必須 >=0，得到 {self.phase1_end}",
        )
        _check(
            self.phase2_end >= self.phase1_end,
            f"TESTFCConfig.phase2_end({self.phase2_end}) 必須 >= phase1_end({self.phase1_end})",
        )
        _check(
            0.0 <= self.tf_ratio_init <= 1.0,
            f"TESTFCConfig.tf_ratio_init 必須在 [0,1]，得到 {self.tf_ratio_init}",
        )
        _check(
            0.0 <= self.tf_ratio_final <= 1.0,
            f"TESTFCConfig.tf_ratio_final 必須在 [0,1]，得到 {self.tf_ratio_final}",
        )


@dataclass
class TESSScheduleConfig:
    """Scheduled Sampling（DAgger 類）設定。

    取代 TFC 的 covariate shift 修正方案。
    per-step、per-sample 獨立 Bernoulli 決定用 GT 或 detach(model output)。

    Phase 1 (0→phase1_end): p_teacher = p_teacher_init（全 GT，穩定初期）
    Phase 2 (phase1_end→phase2_end): p_teacher 線性衰減
    Phase 3 (phase2_end→end): p_teacher = p_teacher_final（純自迴歸）
    """

    enabled: bool = False
    phase1_end: int = 5_000
    phase2_end: int = 35_000
    p_teacher_init: float = 1.0
    p_teacher_final: float = 0.0

    def __post_init__(self) -> None:
        _check(
            self.phase1_end >= 0,
            f"TESSScheduleConfig.phase1_end 必須 >=0，得到 {self.phase1_end}",
        )
        _check(
            self.phase2_end >= self.phase1_end,
            f"TESSScheduleConfig.phase2_end({self.phase2_end}) 必須 >= phase1_end({self.phase1_end})",
        )
        _check(
            0.0 <= self.p_teacher_init <= 1.0,
            f"TESSScheduleConfig.p_teacher_init 必須在 [0,1]，得到 {self.p_teacher_init}",
        )
        _check(
            0.0 <= self.p_teacher_final <= 1.0,
            f"TESSScheduleConfig.p_teacher_final 必須在 [0,1]，得到 {self.p_teacher_final}",
        )


@dataclass
class TESRPConfig:
    """Detached Self-Rollout（Rolling Pool 簡化版）設定。

    pool_prob 機率觸發：先 torch.no_grad() rollout r_steps 步，
    再以 detached state 做 k_rp_steps 步監督訓練。
    """

    enabled: bool = False
    pool_prob: float = 0.20  # 每 step 觸發 RP 的機率
    r_steps: int = 4  # detached pre-rollout 步數
    k_rp_steps: int = 4  # 監督 rollout 步數

    def __post_init__(self) -> None:
        _check(
            0.0 <= self.pool_prob <= 1.0,
            f"TESRPConfig.pool_prob 必須在 [0,1]，得到 {self.pool_prob}",
        )
        _check(self.r_steps >= 0, f"TESRPConfig.r_steps 必須 >=0，得到 {self.r_steps}")
        _check(
            self.k_rp_steps >= 1,
            f"TESRPConfig.k_rp_steps 必須 >=1，得到 {self.k_rp_steps}",
        )


@dataclass
class TESTemporalConfig:
    """Dy-C/E：L_temporal（solver 2CTU two-window KE 統計穩態 loss）設定。

    每 step 額外做一段長 rollout（涵蓋 ≥2×window_ctu）：preroll 步 ``torch.no_grad``
    （detached）+ grad_tail 步帶梯度，收集每步 domain-mean KE → two-window relu
    （ratio=|m2-m1|/mean，loss=relu(ratio-eps)，只在比 solver 容忍更不穩時罰）。
    grad 只經尾段 KE 回流（plan §2.4「長段算曲線、grad 回 short tail」）。
    enabled=False 時不額外 rollout，行為與既有訓練完全一致。
    """

    enabled: bool = False
    r_total: int = 50  # 長 rollout 總步數（Δ=0.2 → 50 步 ≈ 10 CTU = 2×window）
    grad_tail: int = 2  # 帶梯度尾段步數（其餘 preroll 走 no_grad）
    window_ctu: float = 5.0  # 半窗大小（CTU），與 solver convergence_monitor 同
    eps: float = 0.05  # |Δmean|/mean 容忍門檻，與 solver 同
    delta: float = 0.2  # 長 rollout 每步 Δt（CTU）；外推區固定步長（V3）
    weight: float = 1.0  # L_temporal 在 total loss 的加權
    interval: int = (
        1  # 每 N 步才算一次 temporal loss（攤平長 rollout 成本；統計穩態壓力不需每步）
    )

    def __post_init__(self) -> None:
        _check(
            self.r_total >= 4,
            f"TESTemporalConfig.r_total 必須 >=4，得到 {self.r_total}",
        )
        _check(
            1 <= self.grad_tail <= self.r_total,
            f"TESTemporalConfig.grad_tail 必須在 [1, r_total]，得到 {self.grad_tail}",
        )
        _check(
            self.window_ctu > 0,
            f"TESTemporalConfig.window_ctu 必須 >0，得到 {self.window_ctu}",
        )
        _check(self.eps >= 0, f"TESTemporalConfig.eps 必須 >=0，得到 {self.eps}")
        _check(self.delta > 0, f"TESTemporalConfig.delta 必須 >0，得到 {self.delta}")
        _check(
            self.weight >= 0, f"TESTemporalConfig.weight 必須 >=0，得到 {self.weight}"
        )
        _check(
            self.interval >= 1,
            f"TESTemporalConfig.interval 必須 >=1，得到 {self.interval}",
        )


@dataclass
class TESPoolConfig:
    """RollingPool 設定（state-continuous warm-state memory）。

    沿單一 case 的 GT 軌跡做 state-continuous walk：每次從 pool slot 的當前態
    rollout k 步、以模型自身預測 writeback 回 pool（frame_idx += k），target 仍為
    GT@matching-frame（**維持 TES 原本 per-CTU 目標**，不改 loss）。long-horizon
    exposure 由 frame_idx 跨 iteration 累積產生，無需單次長 rollout / 長 BPTT。

    mode:
      ``off``                  — 不用 pool（現行 fresh-GT 每步，行為不變）。
      ``detached_selfrollout`` — 現行 detached self-rollout（相容舊 rp 語義）。
      ``rolling``              — 持久 pool（本類核心）。

    store:
      ``state_hidden`` — writeback 保留 hidden（跨步記憶延續）。
      ``state_only``   — writeback 僅存 physical，hidden 每次 reseed 歸零（無 hidden leakage）。

    damage / priority / cross_bucket 為後續 RP-V6/V8 實驗子開關，目前預設 none（未串接）。
    """

    mode: str = "off"
    pool_size: int = 256
    pool_device: str = "cpu"  # cpu | cuda（大網格建議 cpu，sample 時搬 GPU）
    reset_max_age: int = 32  # frame_idx 上限（GT frame 單位），超過 reseed
    reset_prob: float = 0.05
    seed_zero_ratio: float = 0.20  # reseed 時 cold（physics=0）比例
    seed_gt_ratio: float = 0.60  # reseed 時從 GT frame0 起的比例（其餘 = 保留自生）
    seed_noise_scale: float = 0.0  # seed physics 加噪幅度（0=不加）
    store: str = "state_hidden"  # state_only | state_hidden
    horizon_mode: str = "fixed"  # fixed | uniform
    k_min: int = 1
    k_max: int = 4
    bptt_tail: int = 0  # 0=全 grad；>0=僅末 G 步 grad（K−G 步 no_grad）
    writeback_require_finite: bool = True
    writeback_improve_only: bool = False
    # writeback 品質閘：候選 state 的 physics mean(|·|) 超過此值 → 視為壞態、reseed
    # （防 self-poisoning runaway：模型放大自身誤差使 pool physics 爆量但仍 finite）。
    # 0 = 停用。GT physics mean|·|~0.1-0.2，建議 cap ~0.8-1.2（容模型 overshoot、擋發散）。
    writeback_max_phys: float = 0.0
    # 後續實驗子開關（stub，預設 none，尚未串接 trainer）
    damage_mode: str = "none"  # none | gaussian | mask_respecting
    damage_scale: float = 0.05
    damage_prob: float = 0.10
    priority_mode: str = "none"  # none | high_error
    cross_bucket_mode: str = "none"  # none | transition

    def __post_init__(self) -> None:
        _check(
            self.mode in ("off", "detached_selfrollout", "rolling"),
            f"TESPoolConfig.mode 必須 off|detached_selfrollout|rolling，得到 {self.mode!r}",
        )
        _check(
            self.pool_size > 0,
            f"TESPoolConfig.pool_size 必須 >0，得到 {self.pool_size}",
        )
        _check(
            self.pool_device in ("cpu", "cuda"),
            f"TESPoolConfig.pool_device 必須 cpu|cuda，得到 {self.pool_device!r}",
        )
        _check(
            self.reset_max_age >= 1,
            f"TESPoolConfig.reset_max_age 必須 >=1，得到 {self.reset_max_age}",
        )
        _check(
            0.0 <= self.reset_prob <= 1.0,
            f"TESPoolConfig.reset_prob 必須在 [0,1]，得到 {self.reset_prob}",
        )
        _check(
            0.0 <= self.seed_zero_ratio <= 1.0,
            f"TESPoolConfig.seed_zero_ratio 必須在 [0,1]，得到 {self.seed_zero_ratio}",
        )
        _check(
            0.0 <= self.seed_gt_ratio <= 1.0,
            f"TESPoolConfig.seed_gt_ratio 必須在 [0,1]，得到 {self.seed_gt_ratio}",
        )
        _check(
            self.seed_zero_ratio + self.seed_gt_ratio <= 1.0 + 1e-6,
            f"TESPoolConfig.seed_zero_ratio+seed_gt_ratio 必須 <=1，得到 "
            f"{self.seed_zero_ratio + self.seed_gt_ratio}",
        )
        _check(
            self.seed_noise_scale >= 0.0,
            f"TESPoolConfig.seed_noise_scale 必須 >=0，得到 {self.seed_noise_scale}",
        )
        _check(
            self.store in ("state_only", "state_hidden"),
            f"TESPoolConfig.store 必須 state_only|state_hidden，得到 {self.store!r}",
        )
        _check(
            self.horizon_mode in ("fixed", "uniform"),
            f"TESPoolConfig.horizon_mode 必須 fixed|uniform，得到 {self.horizon_mode!r}",
        )
        _check(
            1 <= self.k_min <= self.k_max,
            f"TESPoolConfig 必須 1<=k_min<=k_max，得到 k_min={self.k_min} k_max={self.k_max}",
        )
        _check(
            self.bptt_tail >= 0,
            f"TESPoolConfig.bptt_tail 必須 >=0，得到 {self.bptt_tail}",
        )
        _check(
            self.writeback_max_phys >= 0.0,
            f"TESPoolConfig.writeback_max_phys 必須 >=0，得到 {self.writeback_max_phys}",
        )
        _check(
            self.damage_mode in ("none", "gaussian", "mask_respecting"),
            f"TESPoolConfig.damage_mode 必須 none|gaussian|mask_respecting，得到 {self.damage_mode!r}",
        )
        _check(
            self.priority_mode in ("none", "high_error"),
            f"TESPoolConfig.priority_mode 必須 none|high_error，得到 {self.priority_mode!r}",
        )
        _check(
            self.cross_bucket_mode in ("none", "transition"),
            f"TESPoolConfig.cross_bucket_mode 必須 none|transition，得到 {self.cross_bucket_mode!r}",
        )


@dataclass
class TESMultiScaleConfig:
    """階層式 NCA 多尺度架構選項設定。

    levels 決定尺度層數，ratios 指定各層下採樣倍率；
    share_core 啟用時不同尺度共用同一 NCA cell 權重。
    pod_mode_groups 供 H5 POD modal 架構使用。
    """

    # levels 不一致警告的一次性旗標（避免每次建 config 洗版）。
    _levels_warned: ClassVar[bool] = False

    levels: int = 2
    ratios: list[int] = field(default_factory=lambda: [4])
    share_core: bool = True
    scale_condition: bool = True
    topdown: bool = True
    bottomup: bool = False
    parent_channels: int = 16
    mid_channels: int = 16
    block_size: int = 8
    active_ratio: float = 0.25
    fine_substeps: int = 1
    pod_mode_groups: list[int] = field(default_factory=lambda: [4, 4, 4, 4])
    # coarse residual 上採樣模式：bilinear(C0,導數在 cell 邊界不連續→渦度顯 ratio-px 格紋)
    # vs bicubic(C1,導數平滑→無格紋,但 overshoot→方向 banding)。
    upsample_mode: str = "bilinear"
    # anti-image 低通（StyleGAN3 配方）：upsample 後對 coarse 殘差(jx/jy)作可分離 Gaussian，
    # σ≈ratio/π 削粗-Nyquist 之上的折疊鏡像(grain)；真渦波長 28px≫8px 粗格→遠低於 cutoff 不傷渦。
    # 0=停用(零行為變更)。治 bilinear 導數不連續產的 grain，且無 bicubic overshoot 的方向 banding。
    upsample_antialias_sigma: float = 0.0
    # 架構修正批次：三項預設 False/沿用舊值 → 既有 run 與 ckpt 零行為變更；由 config 明示開啟。
    #
    # (1) coarse_delta_mask：粗層 delta 未乘 coarse_mask，**過半為固體**的粗格
    #     （`downsample_mask(threshold=0.5)`，非「整格固體」）產生不可學的
    #     `−pooled_physics`，經 bilinear 上採灌進建物周圍流體。
    coarse_delta_mask: bool = False
    # (2) psi_upsample_follows_config：`_psi_up_mode` 硬編 bilinear，本區段的
    #     `upsample_mode`（支援 bicubic）接不到 ψ 路徑。True → ψ 上採改讀本欄位。
    #     注意：d1_surgical=True 時 ψ 走 blur，本開關不覆蓋該路徑。
    psi_upsample_follows_config: bool = False

    def __post_init__(self) -> None:
        _check(
            self.upsample_antialias_sigma >= 0.0,
            f"TESMultiScaleConfig.upsample_antialias_sigma must be >=0, got {self.upsample_antialias_sigma}",
        )
        # levels 與 ratios 不符時的告警：canonical 的多 ratio cell
        # （`operator_gated_streamcollide` 等）不讀 `levels`，其層數由 `len(ratios)+1`
        # 決定；但 h_variants 系列會讀且硬性要求 levels 恆為 2，不符即 raise：
        #   - `models/assembly/h_variants.py::HSharedScaleNCA.__init__`：`if ms.levels != 2` → ValueError
        #   - `models/assembly/h_variants.py::HLapResNCA.__init__`     ：同上
        # 對應 `arch: h_shared_scale` / `h_lapres`（註冊於 `models/factory.py::_build_arch`）。
        #
        # 因此 levels 與 ratios 不符是「可能誤導」而非「必然無害」，且正確的對齊
        # 方式**取決於 arch**：多 ratio 架構應對齊 `len(ratios)+1`，但 h_variants
        # 系列恆要求 2，若對齊 len(ratios)+1 反而會使其建構失敗——故警告只陳述
        # 事實，不給跨架構的統一建議。
        #
        # 用 warning 而非 _check：硬擋會對大量帶 multiscale 區段的 config 與測試
        # fixture 強加簿記，而真正會出事的兩個架構自己就會 raise，已有把關。
        if (
            self.levels != len(self.ratios) + 1
            and not TESMultiScaleConfig._levels_warned
        ):
            TESMultiScaleConfig._levels_warned = True
            warnings.warn(
                f"TESMultiScaleConfig.levels={self.levels} 與 len(ratios)+1="
                f"{len(self.ratios) + 1} 不符（ratios={self.ratios}）。"
                " 多 ratio 架構（canonical）的實際層數由 ratios 決定，不讀 levels，"
                "此宣告只會誤導讀者；但 arch=h_shared_scale／h_lapres **會**讀 levels"
                " 且硬性要求 levels==2（h_variants.py::HSharedScaleNCA/HLapResNCA，不符即 ValueError）。"
                " 正確值取決於 arch，請依實際使用的架構決定，不要一律對齊 len(ratios)+1。",
                RuntimeWarning,
                stacklevel=2,
            )
        # `upsample_mode` 若無值域驗證，拼錯要到 forward 才在 `upsample_state`
        # 炸。合法值以該函式的分支為準。
        _UP_MODES = ("bilinear", "nearest", "bicubic", "blur")
        _check(
            self.upsample_mode in _UP_MODES,
            f"TESMultiScaleConfig.upsample_mode 必須是 {_UP_MODES} 之一，"
            f"得到 '{self.upsample_mode}'",
        )
        # ψ 路徑另有物理限制：curl 取的是 ψ 的一階導。`nearest` 是分段常數，
        # 其導數是 cell 邊界上的 delta——curl(ψ_up) 會退化成邊界脈衝而非平滑動量場，
        # 比它要治的 bilinear 格紋更糟。開了 psi_upsample_follows_config 就不許選它。
        _check(
            not (self.psi_upsample_follows_config and self.upsample_mode == "nearest"),
            "TESMultiScaleConfig: psi_upsample_follows_config=True 時 "
            "upsample_mode 不得為 'nearest'——ψ 是分段常數時 curl(ψ) 退化為 "
            "cell 邊界上的脈衝。請用 bilinear／bicubic／blur。",
        )
        _check(
            self.levels >= 1,
            f"TESMultiScaleConfig.levels must be >=1, got {self.levels}",
        )
        _check(len(self.ratios) >= 1, "TESMultiScaleConfig.ratios must not be empty")
        self.ratios = [int(r) for r in self.ratios]
        _check(
            all(r >= 1 for r in self.ratios),
            f"TESMultiScaleConfig.ratios must be >=1, got {self.ratios}",
        )
        _check(
            self.parent_channels > 0,
            f"TESMultiScaleConfig.parent_channels must be >0, got {self.parent_channels}",
        )
        _check(
            self.mid_channels > 0,
            f"TESMultiScaleConfig.mid_channels must be >0, got {self.mid_channels}",
        )
        _check(
            self.block_size > 0,
            f"TESMultiScaleConfig.block_size must be >0, got {self.block_size}",
        )
        _check(
            0.0 < self.active_ratio <= 1.0,
            f"TESMultiScaleConfig.active_ratio must be in (0,1], got {self.active_ratio}",
        )
        _check(
            self.fine_substeps >= 1,
            f"TESMultiScaleConfig.fine_substeps must be >=1, got {self.fine_substeps}",
        )
        _check(
            len(self.pod_mode_groups) >= 1,
            "TESMultiScaleConfig.pod_mode_groups must not be empty",
        )
        self.pod_mode_groups = [int(v) for v in self.pod_mode_groups]
        _check(
            all(v > 0 for v in self.pod_mode_groups),
            f"TESMultiScaleConfig.pod_mode_groups must be >0, got {self.pod_mode_groups}",
        )


@dataclass
class TESTransportConfig:
    """HD/HE H3T transport branch 超參數。

    arch='h3t_warpresidual' 使用 learned coarse displacement + grid_sample warp
    + residual correction + gate；arch='h3t_streamcollide' 使用 displacement-only
    STREAM，再由 multirate residual 做 COLLIDE。zero-init displacement 讓兩者初始
    輸出接近 residual baseline。
    """

    skip: bool = (
        False  # Phase 0 速度：True→跳過 grid_sample warp（fixed-point 不需平流）
    )
    warp_ratio: int = 8  # displacement field 計算的 coarse 下採樣倍率
    enc_hidden: int = 32  # warp encoder 隱藏通道
    disp_max_px: float = 20.0  # tanh 後位移上限(px)，再乘 delta_ctu scale
    # 現行 production 資料集的 max Δt=2.0 < 2.5，故此 clamp 在全部既有資料上
    # **恆不生效**。保留以防未來大 Δt 資料，但引用時不可把它當成「已生效的位移上限」。
    disp_ctu_cap: float = 2.5  # delta_ctu 用於 disp scaling 的上限
    gate_bias: float = -4.0  # gate sigmoid 初始偏置（偏向 residual path）

    # ── HF operator-gated streamcollide flags（default = 行為不變 = HE-1d streamcollide）──
    prior_mode: str = (
        "off"  # HF-1d：off | composition（deterministic inflow prior 注入 state_create）
    )
    prior_w_front: float = 0.05  # inflow front sigmoid 寬（X_01 單位，HPO-1 調）
    use_velocity_disp: bool = (
        False  # HF-1e：velocity-conditioned displacement（alpha zero-init）
    )
    gate_mode: str = "off"  # HF-1f：off | g0 | g1 | g2（operator gate）
    use_create: bool = False  # HF-1g：create / source branch（zero-init）
    warp_substeps: int = 1  # HF-3f：STREAM 內 multi-substep warp（1 = 現行單次）

    # ── WFC v2 R3 算子耗散攻擊（default 行為不變）──
    warp_mode: str = (
        "bilinear"  # V2-D：bilinear | bicubic（bicubic 對極值壓縮小=少耗散，但 ringing 風險）
    )
    ke_rescale: bool = (
        False  # V2-C：per-step KE 重正規化層（抵消 bilinear 凸組合壓幅；zero-init α 起步=identity）
    )
    ke_rescale_gmax: float = 1.15  # V2-C：每步幅值補打上限（防 over-amp 複利）
    ke_rescale_gmin: float = (
        1.0  # V2-C2：對稱 clamp 下限（<1 既補消散又壓過幅；1.0=非對稱 V2-C 原行為）
    )
    ke_rescale_start_ctu: float = (
        0.0  # spin-up warmup（累計 CTU < 此值 bypass rescale，0=永遠啟用）
    )
    # ★iter7：IC-anchored KE ceiling（只壓不補，治 re-inject 能量棘輪；caller 傳 anchor KE）
    ke_anchor_mode: str = "off"  # off（逐 bit identity）| ceiling
    ke_anchor_gmin: float = 0.90  # 每步最大壓幅（g≥gmin；KE 最多 −19%/step）
    ke_anchor_margin: float = (
        1.0  # ceiling = anchor*margin（chain 頭寬備用，iter7=1.0）
    )
    # ── P1 free-stream fluct 主動衰減（治 free-stream grain；arch op 作用在 inference 每步，繞過 F19）──
    # 每 outer step 後 fluct 動量 ×(1−λ·fs_mask)，fs_mask=mean-flow 低剪切(自由流)區。0=停用(零行為變更)。
    fs_decay: float = 0.0  # λ：free-stream 每步衰減率（0.1~0.2 典型）— P1 blanket（鈍）
    fs_decay_thresh: float = (
        0.6  # mean |curl| 百分位門檻；低於此(自由流)才衰減/濾，高於(wake)保留
    )
    # ── #1 free-stream high-k 選擇性衰減（spectral viscosity）：只殺自由流 grain、不動低k(保 ens)──
    # 每步 fluct -= α·fs_mask·(fluct − blur(fluct))，blur=Gaussian 低通。α=0 停用。比 fs_decay(整體幅值)sharp。
    fs_hf_damp: float = (
        0.0  # α：high-freq 移除強度（0.5~1.0）；小核→只移 top-octave per-pixel grain
    )
    fs_hf_damp_global: bool = (
        False  # True=全域 scale-gated(不用 region mask；治 wake/下游渦被 region 誤殺)
    )
    fs_hf_damp_ksize: int = (
        5  # Gaussian 低通核尺寸（奇數）；預設 5→只觸 ~2px grain。大核(9~13)+大σ 觸及欄向 band 波長 13-40px
    )
    fs_hf_damp_sigma: float = (
        1.0  # Gaussian σ；σ≈4 才有效低通 13-30px 欄向 band（小 σ 僅殺 per-pixel grain）
    )
    fs_hf_damp_aniso: str = (
        "iso"  # iso（2D Gaussian，殺各向同性 grain）| x（僅沿 x 低通，專殺垂直欄向 band、保 wake y-結構不糊）
    )
    # ── §9.1(b) 生成式變化：collide 後對 fluct 速度(jx/jy)加 σ·randn，gated 到 wake(1−fs_mask)──
    # 給渦「自發變化」來源 → 不塌成單一糊解；配能譜 loss = 學「乾淨且有變化」=生成式。0=停用(零行為變更)。
    noise_sigma: float = (
        0.0  # σ：每步 wake 區隨機擾動幅值（norm 動量單位，小→中掃 0.005~0.03）
    )
    # ── §9.1(c) 能量回饋(production forcing)：fluct ×(1+λ·mean_shear_norm)，gated 到高 mean 剪切(wake)──
    # 補回 RANS production −⟨u'u'⟩:∂ū 的 fluct 側 → 給渦持續能量源、治 ens↓根(渦不餓死)。arch op 每步作用→繞 F19。
    production_weight: float = (
        0.0  # λ：production 強度（小→中掃 0.01~0.05；過大→渦爆，靠 ke_rescale/overflow guard）
    )
    production_shear_q: float = (
        0.7  # mean shear 百分位門檻：高於此(wake)才施 production；低於(自由流)不動
    )

    def __post_init__(self) -> None:
        _check(
            self.warp_ratio >= 1,
            f"TESTransportConfig.warp_ratio 必須 >=1，得到 {self.warp_ratio}",
        )
        _check(
            self.enc_hidden > 0,
            f"TESTransportConfig.enc_hidden 必須 >0，得到 {self.enc_hidden}",
        )
        _check(
            self.disp_max_px > 0,
            f"TESTransportConfig.disp_max_px 必須 >0，得到 {self.disp_max_px}",
        )
        _check(
            self.disp_ctu_cap > 0,
            f"TESTransportConfig.disp_ctu_cap 必須 >0，得到 {self.disp_ctu_cap}",
        )
        _check(
            self.prior_mode in ("off", "composition"),
            f"TESTransportConfig.prior_mode 必須 off|composition，得到 {self.prior_mode!r}",
        )
        _check(
            self.gate_mode in ("off", "g0", "g1", "g2"),
            f"TESTransportConfig.gate_mode 必須 off|g0|g1|g2，得到 {self.gate_mode!r}",
        )
        _check(
            self.warp_substeps >= 1,
            f"TESTransportConfig.warp_substeps 必須 >=1，得到 {self.warp_substeps}",
        )
        _check(
            self.noise_sigma >= 0.0,
            f"TESTransportConfig.noise_sigma 必須 >=0，得到 {self.noise_sigma}",
        )
        _check(
            self.production_weight >= 0.0,
            f"TESTransportConfig.production_weight 必須 >=0，得到 {self.production_weight}",
        )
        _check(
            0.0 <= self.production_shear_q <= 1.0,
            f"TESTransportConfig.production_shear_q 必須在 [0,1]，得到 {self.production_shear_q}",
        )


@dataclass
class BaselineConfig:
    """Baseline 對照模型超參（AR-FNO / modern U-Net / Dilated-ResNet / 穩態 U-Net）。

    僅 arch ∈ {baseline_ar_fno, baseline_unet_modern, baseline_dilated_resnet,
    baseline_unet_steady} 時生效。協定：hidden=0、physics_dim=9 輸出、
    param budget 對齊 T-NCA。
    """

    backbone: str = "dummy"  # dummy | fno | unet | drn | unet_steady
    out_residual: bool = True  # True: 殘差（末層 zero-init）；False: 直接預測
    cond_features: str = "re_lchar_delta"  # broadcast 條件 channel 組合

    # FNO（spectral conv；AMP 下 backbone 內自動轉 float32）
    fno_modes: int = 24
    fno_width: int = 48
    fno_layers: int = 4

    # modern U-Net / 穩態 U-Net
    unet_base: int = 48
    unet_depth: int = 3  # 下採樣層數（H=360：depth<=3 安全或內部 pad）

    # Dilated-ResNet
    drn_width: int = 64
    drn_blocks: int = 8

    def __post_init__(self) -> None:
        _check(
            self.backbone in ("dummy", "fno", "unet", "drn", "unet_steady"),
            f"BaselineConfig.backbone 必須 dummy|fno|unet|drn|unet_steady，得到 {self.backbone!r}",
        )
        _check(
            self.cond_features in ("re_lchar_delta", "re_lchar", "none"),
            f"BaselineConfig.cond_features 不支援：{self.cond_features!r}",
        )
        _check(
            self.fno_modes > 0 and self.fno_width > 0 and self.fno_layers > 0,
            "BaselineConfig FNO 超參須 >0",
        )
        _check(
            self.unet_base > 0 and self.unet_depth >= 1, "BaselineConfig U-Net 超參非法"
        )
        _check(
            self.drn_width > 0 and self.drn_blocks >= 1, "BaselineConfig DRN 超參非法"
        )


@dataclass
class TESModelConfig:
    """模型架構超參數。

    state_dim = static_dim + physics_dim（+ hidden_dim 若啟用）。
    cond_dim = G_vec_B (14D) + P_vec_B (3D) = 17D（標準設定）。
    gvec_mode 控制 G_vec 注入策略；rate_mode 控制 collide cell 輸出形式。
    grad_checkpoint 啟用後以重算換記憶體，僅作用於訓練階段。
    mean_film 啟用後 H3TStreamCollide 系接受 mean_flow_norm 做空間 FiLM。
    """

    # 狀態向量維度
    arch: str = "flat"
    state_dim: int = 13  # 4 static + 9 physics（無 hidden）
    static_dim: int = 4
    physics_dim: int = 9

    # 條件向量維度
    cond_dim: int = 17  # G_vec_B (14D) + P_vec_B (3D)

    # ConditionEncoder MLP 隱藏維度
    encoder_hidden: int = 128

    # MarkovNCACell 內部特徵維度（同時為 FiLM γ/β 的維度）
    nca_channels: int = 64

    # 頻域 update 分支（FNO 風 spec+local 雙分支 project）：spectral 低模補全域低/中頻結構
    # （破均勻，攻局部 conv 缺的全域協調），local conv 保高頻細節。zero-init → 起步=純 local（identity）。
    nca_spectral_update: bool = False  # True=MarkovNCACell project 加 spectral 低模分支
    nca_spectral_modes: int = 24  # spectral 保留低頻 mode 數（兩空間維，比照 FNO）
    nca_msp_scales: list[int] = field(
        default_factory=lambda: [1]
    )  # MSP 下採因子列表；[1]=單尺度（向後相容）
    # ★MSP 的降採／升採模式。預設 "blur"＝BlurPool／anti-image Gaussian。
    #   舊實作用裸 avg_pool2d + bilinear interpolate，會在 coarse cell 邊界留下
    #   **棋盤狀大像素格紋**（降採 folding ＋ 升採 imaging replica，兩者都被 curl 放大）。
    #   `models/multiscale/ops.py` 早就為此寫了 "blur" 模式，但 MSP 分支沒用到它。
    #   ⚠ 改成 "bilinear"/"area" 可復現舊行為（僅供對照，勿用於正式實驗）。
    # ★MSP 的取樣典範：
    #   "resample" = 舊法（降採 → MRP → 升採）。即使用 blur 抗鋸齒，重採樣本身
    #                仍是格紋的來源，s=2 加雜訊時接縫比 1.446（blur 後 1.011）。
    #   "pool"     = **stride-1 池化脈絡**：不降採也不升採，直接在原解析度上取
    #                k×k 鄰域的平均當作「該尺度的粗粒化視野」。
    #                零重採樣 → 接縫比 1.0001（實測）。`nca_msp_scales` 此時
    #                改解讀為**奇數 kernel 大小**（如 [1, 7, 15, 31]）。
    #                ★仍是有限鄰域算子，完全符合 CA 局部性；k=31 在 24 步後
    #                達 ±360 px，超過場寬一半 → 局部算子也能全域耦合。
    nca_msp_mode: str = "resample"  # resample / pool
    nca_msp_down_mode: str = "blur"  # area / blur / bilinear（僅 resample 模式）
    nca_msp_up_mode: str = "blur"  # bilinear / bicubic / blur / nearest

    # ψ-curl 動量頭（FLUXSTREAM_DESIGN.md v2）：jx/jy 改由純量 ψ 場的離散 curl 重建
    # （curl(ψ)=(∂ψ/∂y,-∂ψ/∂x)）。zero-init → 起步=identity。
    # 穩態 mean（∂ρ/∂t≈0）連續方程精確退化成 ∇·j=0，物理上僅適用於此類穩態任務。
    psi_curl_momentum: bool = (
        False  # True=MarkovNCACell project 排除 jx/jy，改由 curl(ψ) 重建
    )
    # ★保證範圍：
    #   (a) **增量**無散度：div(curl ψ)=0 對與 curl 共用的中央差分核、在流體內部成立
    #       （機器精度：內部（距邊界 ≥3 px）absmean 量級 ~1e-20；全域 absmean 略高，
    #       差額全來自邊界 1–2 px 環）。
    #   (b) **不移除既有散度**：curl 是 Hodge 投影的反面——投影移除既有散度，curl 只保證
    #       不新增。故 div(j_{n+1}) = div(j_n)，暖啟動 IC 攜帶的散度在架構上**不可移除**。
    #       實測暖啟動 IC 對 GT 的 divj_norm 中位比約 3×（跨 case 區間約 2.4–4.8 倍），
    #       單靠加損失無法突破此下限（理論天花板約 GT 的 ~2.9×）。
    #   (c) **界面不保證**：寫回為 mask∘curl（h3t_streamcollide.py 對 jx/jy 逐點乘 fluid_mask），
    #       div(m·u)=m·div(u)+u·∇m，固/流界面第二項非零（以中央差分量測，壁面 2 px 內達
    #       0.444×|j|；緊緻前向差分下內部散度僅 2.50e-03×|j|）；域邊界因 replicate
    #       padding 亦失效。

    # band 攻源：curl 前對 ψ 套固定 Gaussian low-pass（sigma px）。
    # 機制：fresh 乾淨=ψ 單發平滑；band=ψ_delta warm 累積中波長。每步平滑 → band 累積 n 步得 n×平滑、
    # fresh 單發僅 1× → 不對稱壓 warm band 輕傷 fresh。curl(平滑ψ) 的**增量**仍無散度（限流體內部；
    # 保證範圍見 psi_curl_momentum 欄位下方的 (a)(b)(c) 段）。0=停用（identity，向後相容）。
    psi_lowpass_sigma: float = 0.0

    # D1-surgical：True → h3t state 下採 area→BlurPool、ψ 上採 bilinear→blur-upsample
    # （精準除 multiscale 取樣混疊/imaging replica → curl 不放大殘 grain；比鈍 ψlp 全場平滑更保銳利）。
    # False=現行（canonical 架構用 False，只靠 psi_lowpass_sigma）。向後相容。
    d1_surgical: bool = False

    # stochastic update（Growing NCA，Mordvintsev）：train-only 每 cell Bernoulli(rate)
    # gate delta，打破同步 lock-step 相位相干（攻 band=相位鎖定中波長吸引子）；deploy(eval) full-sync（rate 不作用）。
    # 1.0=全同步（現行，向後相容）。<1.0（如 0.5）啟用。由 factory 注入 MarkovNCACell（比照 update_clip）。
    stoch_update_rate: float = 1.0

    # Hidden channels (消融 E1d): 0 = baseline（無 hidden）；16 = ablation
    # state_dim when hidden_dim>0 should be static_dim + physics_dim + hidden_dim
    hidden_dim: int = 0

    # Dual-pathway G_vec conditioning: 0=disabled，14=啟用獨立 GVecEncoder
    # 啟用後 TESNCA.forward 需額外接受 g_vec_norm 並加法合併 FiLM
    g_vec_dim: int = 0

    # G_vec 注入模式：
    #   "none"    → 不使用 G_vec（M0 baseline）
    #   "additive"→ GVecEncoder 加法 FiLM（舊版行為，向後相容）
    #   "gated"   → GVecEncoder + alpha_g≈0 殘差門控（M1_full）
    gvec_mode: str = "additive"

    # MRP 感知層（預設 disabled，向後相容）
    mrp: MRPConfig = field(default_factory=MRPConfig)
    multiscale: TESMultiScaleConfig = field(default_factory=TESMultiScaleConfig)
    # HD/HE transport branch（arch='h3t_warpresidual' / 'h3t_streamcollide'）
    transport: TESTransportConfig = field(default_factory=TESTransportConfig)
    # Baseline 對照模型超參（arch='baseline_*'）
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    # POD modal architecture (H5 only). Populated automatically from
    # TESTrainConfig.pod_basis_path / pod_modes by load_tes_config when the
    # YAML does not override the values inside ``model``.
    pod_basis_path: str = ""
    pod_modes: int = 16

    # Dy-C：per-CTU rate 軟夾 c·tanh(rate/c)（套用所有 MarkovNCACell，由 model_factory 設定）。
    # 0 = 停用。>0 限每步 collide rate 幅度 → 壓 Jacobian spectral radius → 抑 long-roll 發散。
    # GT physics ~[-0.88,0.88]、單 CTU rate 量級小，建議掃 0.1~0.5。
    # `update_clip` 作用在 **tanh 正規化態**上，
    # 故 c=1.0 等於允許單步變化半個動態範圍（tanh 值域 [−1,1]）——不構成
    # 「限制每步物理變化、壓 Jacobian 譜半徑」的有效約束。實測典型工作點下
    # c=1.0 只衰減 0.1–0.3%，數值上等同恆等映射。
    # c 的取值目前無物理依據（非由 CFL 或任何物理尺度導出）；若要讓它有作用，
    # 須先量測 delta 的實際分布再定 c。
    update_clip: float = 0.0

    # Dy2 §2.4：rate 形式 arm-F / arm-FΔ 消融軸（TESNCA.forward 已以 getattr 讀取）。
    #   "fdelta"（default，arm-FΔ／現況）：time_conditioned 時 cond 附加 per-step ΔCTU，
    #            且 NCA cell 輸出 rate × Δt（explicit physical scaling）。行為與既往完全一致。
    #   "f"（arm-F）：rate=f(x)，步長不變（不乘 Δt、cond 不附加 ΔCTU）→ 留正式 ablation。
    # ⚠ 僅在 flat TESNCA.forward 內生效；H 系列不受此 flag 影響。
    rate_mode: str = "fdelta"

    # grad_checkpoint：訓練時對 NCA collide cell forward 用 gradient checkpointing（反向重算、
    # 不存中間 activation）→ 大砍 full-res rollout 記憶體（fine_substeps×lk 的 85ch/64ch 全解析張量）。
    # 數值等價（recompute deterministic）；代價 ~25-35% 慢。eval 不啟用。預設 False 行為不變。
    # 目前接於 H3TStreamCollideNCA（含 operator_gated）的 coarse/fine cell。
    grad_checkpoint: bool = False

    # ── 狀態空間閉包（norm_frozen 的 clamp_state）──
    # state 存在 tanh 正規化的像裡，而該像是閉盒 [-CLIP, CLIP]^9。MarkovNCACell 的
    # 更新是無界加法，不夾的話 state 會漂出像外 —— 實測隨機權重 500 步後 physics
    # 通道衝到 CLIP 的 1500 倍，模型自此在完全沒見過的量級上運作。
    # >0 時每個 outer step 尾端把 physics 夾回 ±state_clip、hidden 夾回 ±1。
    # 建議值 = `norm_frozen.CLIP`。0 = 停用（既有 ckpt 零行為變更）。
    # ⚠ 這與 phys_adm 角色不同、不可互相取代：本項夾的是**狀態**（保模型待在受訓
    #   定義域內）；物理讀數的合法性另由反正規化內建的 clip 保證。
    state_clip: float = 0.0

    # ── 物理可容許性投影（models/transport/admissibility.py）──
    # state 存在 tanh 正規化的像裡（合法值域 (-1,1)），但 MarkovNCACell 的更新是無界加法，
    # 且 ψ-curl／KE rescale／hidden tanh 三道護欄都只覆蓋 jx/jy 與 hidden——密度通道無界。
    # 實測長推進段 ρ 漂出像外、反正規化讀出 clip 底 −3.9517，u=j/ρ 隨之發散。
    # 啟用後每個 outer step 尾端把 physics 投影回 ρ∈[rho_min, rho_max] 與 |u|≤mach_max·c_s，
    # 兩者在區間內部逐位恆等（未違反的像素連浮點來回都不做）。
    # m_ref / s_combined 為 tanh 正規化常數，由 config 產生腳本自 dataset stats 寫入，
    # 註冊為 buffer 後隨 ckpt 序列化 → 訓練／評估／部署三態拿到同一組常數。
    phys_adm_enabled: bool = False
    phys_adm_rho_min: float = 0.0
    phys_adm_rho_max: float = 0.0
    phys_adm_mach_max: float = 0.0
    phys_adm_m_ref: list[float] = field(default_factory=list)
    phys_adm_s_combined: list[float] = field(default_factory=list)

    # LC decompose turb_module（PLAN §4.2 mean conditioning）：True → H3TStreamCollide
    # 系建 MeanFlowEncoder，forward 接受 mean_flow_norm，per-pixel (γ_m, β_m) 空間 FiLM
    # 三路相加（γ = γ_p + γ_g + γ_m）。zero-init → 未傳 mean 或剛啟用時行為不變。
    mean_film: bool = False

    # ── TURB-NCA 重新設計（arch='turb_renca'，clean-slate, spec §5/§9/§10）──
    # Reynolds-decomposed fluct NCA：state=9ch m′（no hidden），SpatialFiLM(m̄+∇m̄) +
    # VectorFiLM(g+p) + MarkovCellTurb(PerceptionA 三分支)。下列欄位僅 turb_renca 用，
    # 其他 arch 不讀（新增分支不破壞既有 arch）。
    # ── Stage-3：forward-time fluctuation KE 硬上界（治 long-rollout blow-up）──
    # True → MarkovCellTurb 建 TurbKErescaler，rollout 每步把 jx/jy fluct KE 壓回 IC
    # 上界（gmax=1.0 → 只壓不補，hard cap）。False = 零行為變更（既有 v1/v2 不設）。

    def __post_init__(self) -> None:
        valid_arch = {
            "flat",
            "h_shared_scale",
            "h_lapres",
            "h_parent_leaf",
            "h_multirate",
            "h_block_refine",
            "h_pod_modal",
            "h3t_warpresidual",
            "h3t_streamcollide",
            "h3t_warpresidual_serial",
            "operator_gated_streamcollide",
            "baseline_ar_fno",
            "baseline_unet_modern",
            "baseline_dilated_resnet",
            "baseline_unet_steady",
            "baseline_fno_steady",
        }
        _check(
            self.arch in valid_arch,
            f"TESModelConfig.arch must be one of {sorted(valid_arch)}, got '{self.arch}'",
        )
        _check(
            self.state_dim > 0,
            f"TESModelConfig.state_dim 必須 >0，得到 {self.state_dim}",
        )
        _check(
            self.static_dim >= 0,
            f"TESModelConfig.static_dim 必須 >=0，得到 {self.static_dim}",
        )
        _check(
            self.physics_dim > 0,
            f"TESModelConfig.physics_dim 必須 >0，得到 {self.physics_dim}",
        )
        _check(
            self.cond_dim >= 0,
            f"TESModelConfig.cond_dim 必須 >=0，得到 {self.cond_dim}",
        )
        _check(
            self.encoder_hidden > 0,
            f"TESModelConfig.encoder_hidden 必須 >0，得到 {self.encoder_hidden}",
        )
        _check(
            self.nca_channels > 0,
            f"TESModelConfig.nca_channels 必須 >0，得到 {self.nca_channels}",
        )
        _check(
            self.hidden_dim >= 0,
            f"TESModelConfig.hidden_dim 必須 >=0，得到 {self.hidden_dim}",
        )

        # ψ-curl 與 STREAM 不得靜默共存。
        #
        # 動量更新鏈為 `j_{n+1} = warp(j_n) + curl(psi)`
        # （`H3TStreamCollideNCA.forward` 內 `new_phys = (warped + coarse_sum) * fluid_mask`
        #   一行，與其後 `curl_c = self._curl_psi(psi_full)` 對 `new_phys[:, 3:4]` /
        #   `new_phys[:, 5:6]` 的兩行寫回）。`curl(psi)` 的**增量**在流體內部無散度
        # （限定範圍見上方 `psi_curl_momentum` 欄位下方的 (a)(b)(c)），但 `warp(j_n)`
        # 不與散度算子交換 → **連該增量性質也不再成立**。實測：解析 div-free 場的
        # mean|div| 由 1.641e-07 增至 20px 位移後的 2.084e-03（×12697，約場 RMS 的 9%）。
        #
        # 這會使「psi-curl 增量無散度性」的假設失效，而現況**無任何守衛**：
        # 掃全部 yaml，兩者從未共存 → 屬全新未測配置，一旦有人同時設定，
        # 得到的是**靜默錯誤的物理**，不是錯誤訊息。
        #
        # 若日後要啟用 STREAM，須先走 flux-form（對 psi 平流再取 curl）或重新驗證此
        # 假設，並連同本檢查一起修改 —— ⛔ 不得只把這幾行刪掉。
        _check(
            not (self.psi_curl_momentum and not self.transport.skip),
            "TESModelConfig: psi_curl_momentum=True 與 transport.skip=False 不得共存。"
            " 動量增量 (warp(j)-j)+curl(psi) 不再滿足『增量在流體內部無散度』（實測 ×12697），"
            " 會使 psi-curl 僅存的那條增量保證失效"
            "（保證範圍見 psi_curl_momentum 欄位下方的 (a)(b)(c)）。"
            " 需啟用 STREAM 請先改 flux-form 或重裁 D3。",
        )
        # ψ-curl 的「增量在流體內部無散度」保證，會被任何**在 curl 之後**直接改動
        # jx/jy 的代數算子作廢——fs_decay（自由流衰減）、區域式 fs_hf_damp（非 global
        # 時逐區乘不同係數）、noise_sigma（加隨機擾動）、production_weight（乘
        # 1+λ·shear）四者皆非無散度算子。canonical 四項皆為 0，故此守衛不觸發、
        # 零行為變更；但若日後有人開了其中之一，保證會**無聲**失效。
        #
        # 用 `_check` 硬擋而非 warning：這條保證是論文引用的物理性質，靜默失效的
        # 代價遠高於擋下一個原本就不該共存的組合。
        # ⛔ 日後若真要啟用其中某個 lever，請連同論文對該保證的引用一起改，
        #    不得只把這個 `_check` 刪掉。
        _psi_breaking = [
            name
            for name, val in (
                ("transport.fs_decay", self.transport.fs_decay),
                ("transport.noise_sigma", self.transport.noise_sigma),
                ("transport.production_weight", self.transport.production_weight),
            )
            if float(val) > 0.0
        ]
        if self.transport.fs_hf_damp > 0.0 and not self.transport.fs_hf_damp_global:
            _psi_breaking.append("transport.fs_hf_damp（region-gated）")
        _check(
            not (self.psi_curl_momentum and _psi_breaking),
            "TESModelConfig: psi_curl_momentum=True 但同時啟用了會直接改動 jx/jy 的"
            f" 代數修正算子 {_psi_breaking}。這些算子非無散度，會作廢 ψ-curl"
            "『增量在流體內部無散度』的唯一結構性保證（保證範圍見 psi_curl_momentum"
            " 欄位下方的 (a)(b)(c)），而該保證是論文引用的物理性質。"
            " 實測全 419 支 yaml 中此組合為 0 支，本檢查不影響任何既有 config。"
            " 若確定要啟用，請同時修改論文／SSOT 對該保證的引用，再放寬本檢查。",
        )
        _check(
            self.g_vec_dim >= 0,
            f"TESModelConfig.g_vec_dim 必須 >=0，得到 {self.g_vec_dim}",
        )
        _check(
            self.update_clip >= 0.0,
            f"TESModelConfig.update_clip 必須 >=0，得到 {self.update_clip}",
        )
        _check(
            self.rate_mode in ("fdelta", "f"),
            f"TESModelConfig.rate_mode 必須是 'fdelta' | 'f'，得到 '{self.rate_mode}'",
        )
        _check(
            self.gvec_mode in ("none", "additive", "gated"),
            f"TESModelConfig.gvec_mode 必須是 'none' | 'additive' | 'gated'，得到 '{self.gvec_mode}'",
        )
        if self.gvec_mode == "gated":
            _check(
                self.g_vec_dim > 0,
                "TESModelConfig.gvec_mode='gated' 需要 g_vec_dim > 0",
            )
        if self.hidden_dim > 0:
            expected = self.static_dim + self.physics_dim + self.hidden_dim
            _check(
                self.state_dim == expected,
                f"TESModelConfig.state_dim({self.state_dim}) 與 static+physics+hidden({expected}) 不一致",
            )
        _check(
            self.pod_modes > 0,
            f"TESModelConfig.pod_modes must be >0, got {self.pod_modes}",
        )
        _check(
            len(self.nca_msp_scales) >= 1
            and all(int(s) >= 1 for s in self.nca_msp_scales),
            f"TESModelConfig.nca_msp_scales 各元素須 >=1，得到 {self.nca_msp_scales}",
        )
        self.nca_msp_scales = [int(s) for s in self.nca_msp_scales]
        if self.phys_adm_enabled:
            _check(
                0.0 < self.phys_adm_rho_min < self.phys_adm_rho_max,
                "TESModelConfig: phys_adm 需 0 < rho_min < rho_max，得到 "
                f"({self.phys_adm_rho_min}, {self.phys_adm_rho_max})",
            )
            _check(
                0.0 < self.phys_adm_mach_max <= 1.0,
                f"TESModelConfig.phys_adm_mach_max 須落在 (0,1]，得到 {self.phys_adm_mach_max}",
            )
            _check(
                len(self.phys_adm_m_ref) == self.physics_dim
                and len(self.phys_adm_s_combined) == self.physics_dim,
                "TESModelConfig: phys_adm_m_ref／phys_adm_s_combined 長度須等於 "
                f"physics_dim({self.physics_dim})，得到 "
                f"({len(self.phys_adm_m_ref)}, {len(self.phys_adm_s_combined)})",
            )
            _check(
                all(float(s) > 0.0 for s in self.phys_adm_s_combined),
                "TESModelConfig.phys_adm_s_combined 各元素須 >0",
            )


@dataclass
class TESCurriculumConfig:
    """k rollout 步數漸進課程設定 + Loss 版本切換閾值。

    k curriculum（三階段）：
        Phase 1: k=1              步數 0 → phase1_end
        Phase 2: k ∈ [2,4]        步數 phase1_end → phase2_end
        Phase 3: k ∈ [4,8]        步數 > phase2_end

    loss_version 目前只保留為 trainer forward 相容欄位；正式 loss term
    由 loss_terms 的 semantic names 決定。
    """

    # k 階段切換步數
    phase1_end: int = 5_000
    phase2_end: int = 15_000

    # 各階段 k 範圍 [k_min, k_max]（含邊界，均勻隨機取樣）
    phase1_k_min: int = 1
    phase1_k_max: int = 1
    phase2_k_min: int = 2
    phase2_k_max: int = 4
    phase3_k_min: int = 4
    phase3_k_max: int = 8

    # Loss 版本切換步數（與 k 相位對齊）
    loss_v2_start: int = 5_000
    loss_v3_start: int = 15_000

    def __post_init__(self) -> None:
        _check(
            self.phase1_end >= 0,
            f"curriculum.phase1_end 必須 >=0，得到 {self.phase1_end}",
        )
        _check(
            self.phase2_end >= self.phase1_end,
            f"curriculum.phase2_end({self.phase2_end}) 必須 >= phase1_end({self.phase1_end})",
        )
        for tag, lo, hi in (
            ("phase1", self.phase1_k_min, self.phase1_k_max),
            ("phase2", self.phase2_k_min, self.phase2_k_max),
            ("phase3", self.phase3_k_min, self.phase3_k_max),
        ):
            _check(lo >= 1, f"curriculum.{tag}_k_min 必須 >=1，得到 {lo}")
            _check(hi >= lo, f"curriculum.{tag}_k_max({hi}) 必須 >= {tag}_k_min({lo})")


@dataclass
class TESTrainConfig:
    """完整訓練設定，整合所有子 config。

    整合資料路徑、Optimizer、LR schedule、DataLoader、混合精度、
    存檔間隔、資料集分割、noise/Jacobian reg、Dy2 整合路徑、
    mean/turb module 設定，以及所有子設定 dataclass 實例。
    """

    # 資料路徑（zarr_path 指向 tes_unified.zarr；zarr_paths 為多 zarr 模式，向後相容）
    zarr_paths: list[str] = field(default_factory=list)
    # ★A 方案（tes_unified_v1）：角色由 config 決定，不存在資料裡。
    #   `zarr_paths` = train 用，`val_zarr_paths` = val 用（DE 系列）。
    #   test（DT）刻意**不設欄位**——它只在論文最終實驗使用，且 train.py 的
    #   `_assert_no_holdout` 會擋掉任何指向 holdout 的路徑。
    #   舊 v4（graph.json 帶 split 欄）不需此欄，維持單一 zarr_paths 的行為。
    val_zarr_paths: list[str] = field(default_factory=list)
    stats_paths: list[str] = field(default_factory=list)
    stats_policy: str = "single"  # single | dataset_global_runtime
    # dataset_names: 過濾 tes_unified.zarr 中的子 dataset（空列表 = 全部）
    dataset_names: list[str] = field(default_factory=list)
    output_dir: str = "runs/misc/default"

    # Optimizer
    lr: float = 1e-3
    lr_min: float = 1e-5
    max_steps: int = 100_000
    warmup_steps: int = 500

    # LR schedule：
    #   "cosine"       — warmup + 單段 cosine decay→lr_min（既有預設，向後相容）。
    #   "warm_restart" — per-phase SGDR：每個 boundary 起點 re-warmup→phase 內 cosine→lr_min。
    #                    配 staggered schedule（每加一難度軸＝landscape shift＝給新鮮 lr）。
    #                    peak_k = lr × peak_decay^k（後期 phase peak 漸降，避 jac/temporal restart 炸穩定）。
    lr_schedule: str = "cosine"
    lr_restart_boundaries: list[int] = field(
        default_factory=list
    )  # phase 起點步數 e.g. [0,1500,2500,3500,4500]
    lr_restart_peak_decay: float = 0.85  # 每次 restart peak 乘數（base × decay^k）
    lr_restart_warmup: int = 100  # 每 phase 內 re-warmup 步數（軟化 restart 跳升）

    # DataLoader
    batch_size: int = 4
    num_workers: int = 4  # 平行預取與主訓練重疊（preload_ram=True 時建議 ≤2，否則 4-8）
    # `num_workers` 被 **4 個 loader** 共用（train／val_cold／val_warm／chain，見
    # train.py 的建構段），num_workers=4 → 實際 spawn **16 個** worker 進程，合計
    # 占用可觀的常駐 RAM（隨 zarr 讀取快取單調增長）。
    # val/chain loader 每 val_interval 才用一次，常駐 12 個 worker 不划算 →
    # 獨立設定，預設 0（不 spawn）。要恢復舊行為就設成與 num_workers 相同。
    val_num_workers: int = 0

    window_size: int = 12  # 每 __getitem__ 從 zarr 載入的最大幀數（≥ k_max+2）
    time_conditioned: bool = (
        False  # E1i：per-step Δt conditioning + output delta scaling
    )
    per_step_delta_ctu: bool = (
        False  # validator：cond_dim∈(5,19) 時逐步 delta_ctu（預設關；取代 getattr fallback）
    )

    # 混合精度
    use_amp: bool = True
    async_eval_generation: bool = (
        False  # True: generate training plots in a background worker
    )

    # per-case gate 判準閾值覆寫。`None` = 使用
    # `eval/case_metrics.py::DEFAULT_GATE_THRESHOLDS`。
    gate_thresholds: Optional[dict[str, float]] = None

    # ckpt 身分不符時的處置（單一真相）。`verify_identity(..., on_mismatch=...)` 的
    # 呼叫端（trainer、checkpointer、long_rollout、run_mean_gate_eval、
    # rq2_chain_eval）共用這個欄位，避免「拿 A config 評 B ckpt」的不一致只留一行
    # warning、淹沒在評測 log 中而未被注意到（psi_lowpass 0.8/0.9 即此類型事故）。
    #
    # `"warn"` ＝記 warning 後繼續；`"raise"` ＝不一致即 `ValueError` 中止。預設為
    # `"raise"`：`verify_identity` 只在 (a) curated 推論敏感超參數逐欄比對出差異，或
    # (b) denorm 常數 sha256 不符時才走 raise 分支；無 identity 區塊的 ckpt 走的是
    # 提早 return 的「無身分」分支，兩種模式都只記 warning。`verify_identity` 不比對
    # config 整體 sha256，新增 config 欄位不會造成假 mismatch。
    #
    # 逃生口：評測腳本的 `--identity-mismatch warn`，或在實驗 yaml 明寫本欄位。
    ckpt_identity_mismatch: str = "raise"

    # 續訓／warm-restart 載入路徑（`Checkpointer._check_identity`、
    # `TESTrainer._check_ckpt_identity`）專用的處置，與上面的載入端刻意分開。
    # 預設 `"warn"`：identity 比對本身出錯不得讓一個跑到一半的訓練無法續跑；
    # 不一致與「無身分」都仍會寫進 training.log 與 metrics.json 的 `ckpt_identity` 事件。
    # 想讓續訓也硬擋，在實驗 yaml 把本欄位設為 `"raise"`。
    ckpt_identity_mismatch_resume: str = "warn"

    # 存檔與可視化間隔（步數）
    val_interval: int = 1_000
    ckpt_interval: int = 2_000
    vis_interval: int = 1_000

    # 資料集分割

    # 全域 RNG seed（torch/cuda/numpy/python + RollingPool）。RP 多 seed 實驗變動此值，
    # split_seed 固定以保 val 集可比。0 = 預設。
    seed: int = 0

    # ── R7 決定性／重測 ──
    # determinism_mode／cudnn_benchmark／rng_stream_isolation／loader_seed_offset 四個
    # 行為欄位若改變設定，會改變抽樣序列與資料順序，使**既有 run 不可比**。
    # determinism_mode：
    #   "off"        ＝完全不動任何旗標（舊版預設）；
    #   "warn_only"  ＝只把 cudnn.deterministic/benchmark/TF32/torch 版本等執行期旗標
    #                  記進 training.log 並提醒 determinism 未強制（不改 kernel 選擇）；
    #   "strict"（現預設）＝啟用 torch.use_deterministic_algorithms(True)。
    #                  已實測可行：canonical mean 架構在 360x640 下 forward+backward
    #                  正常。含 grid_sample 的 transport 分支在
    #                  canonical 為 skip=true 未建構；若日後啟用 transport 而遇到
    #                  無 deterministic kernel 的算子，降到 "warn_only" 並記錄理由。
    #   實作見 training/rng_streams.py::apply_determinism_mode。
    determinism_mode: str = "strict"
    # cuDNN autotune：對固定輸入尺寸（本專案 360×640 恆定）挑最快 kernel。
    # 與 determinism_mode="strict" 互斥（autotune 結果不保證 deterministic），
    #   strict 下會被強制關閉並在 log 明示——不讓 config 值靜默失效。
    cudnn_benchmark: bool = True
    # repeat_tolerance：metric 名 → 可接受的 run-to-run 絕對抖動上限。空 dict（預設）
    #   ＝尚未量化。量測工具：repeatability probe（本欄由其 --emit-tolerance 產出）。
    #   **本欄不是訓練期門檻**：除了下方 __post_init__ 的型別驗證（只檢查各值是
    #   非負數值，不看內容）之外，訓練／評估流程沒有任何一行讀它，設了不會擋任何事。
    #   會讀取其值並據以判定的只有離線比對 `repeatability_probe.py --against-config
    #   <本 yaml>`，該模式才會把實測 mdd 與本欄比對並以非零 exit code 表示超標。
    #   寫法：必須放在 yaml **頂層**（三支載入器都只取頂層鍵，巢狀節一律讀不到且
    #   不可靠 warning 察覺）。repeatability_probe 的 load_config_tolerance() 會為此報
    #   ValueError；守衛見 tests/nca/tes/test_r7_countable_claims.py 的兩條 yaml 斷言。
    repeat_tolerance: dict[str, float] = field(default_factory=dict)
    # rng_stream_isolation：True＝訓練期各離散抽樣用途改用固定 offset 的獨立 RNG 串流
    #   （見 training/rng_streams.py）。會改變抽樣序列 → 與既有 run 不可比；預設為
    #   True，因為不隔離串流時，同 config 同 seed 兩次訓練的終點可能有顯著落差。
    rng_stream_isolation: bool = True
    # loader_seed_offset：True＝train/val_cold/val_warm/chain 四個 DataLoader 改用
    #   cfg.seed + 固定 per-loader 偏移（見 train.py::LOADER_SEED_OFFSET_*），
    #   使 train 與 chain（皆 shuffle=True）不再共用同一 permutation 與 worker base_seed。
    #   會改變資料順序 → 與既有 run 不可比；預設為 True，因為兩個 loader 共用
    #   permutation 會讓 chain 的取樣與 train 完全相關，等於少一個獨立觀測。
    loader_seed_offset: bool = True

    # Dy-A：training noise injection。每 step IC physics 加 N(0,σ²)（fluid 區），target 仍 clean
    # → 教模型「偏離→拉回」→ GT 軌跡 manifold 變 attractor（limit-cycle 無定點靠此拘束）。
    # 0 = 停用。norm space 建議掃 0.01~0.05。
    train_noise_sigma: float = 0.0

    # Dy-V8：Jacobian regularization（真收縮；plan §3.5）。直接懲罰單步局部放大率 ‖J·ε‖/‖ε‖>γ
    # → 訓練成收縮 map（‖J‖<1）→ 短訓練即壓長 rollout 發散。clip 限幅≠收縮，此為正解。
    # jac_reg_weight=0 停用。每 step 多 2 次單步 forward（state, state+ε）。
    jac_reg_weight: float = 0.0  # 0=off，建議掃 0.01~0.1
    spectral_temporal_weight: float = (
        0.0  # LC-1：peak-sharpening spectral loss（0=off）
    )
    mean_jac_interval: int = (
        1  # mean_step jac_reg 計算間隔（每 N 步算一次；收縮慢變，>1 省 2×forward/step）
    )
    mean_jac_start_frac: float = (
        0.0  # mean jac_reg 起始 step 比例（0=從頭；0.5=後半才進，前半先學結構）
    )
    mean_jac_ramp_frac: float = (
        0.1  # 從 start_frac 線性 ramp 到滿權重的寬度（避硬切致抖）
    )
    jac_eps: float = 1e-2  # 擾動幅度（norm space）
    jac_gamma: float = 1.0  # 收縮門檻 γ：罰 ratio>γ；<1 強制真收縮（掃 0.9/0.95/1.0）
    # (3) 架構修正：SSIM 值域改讀 stats.json 的 vis_range（與 eval 同一套來源）。
    #     `perceptual.py` 硬編 _VEL_MAX=0.20／_VOR_RANGE=0.04，而 eval 側 vis_range
    #     是 0.14／0.016 → 訓練目標與 gate 是兩個不同泛函，對條紋缺陷的敏感度差 5.43 倍。
    #     False=沿用硬編常數（既有 run 零行為變更）。
    ssim_range_from_stats: bool = False
    jac_gamma_low: float = (
        0.0  # W0-1 bug-B：two-sided 下界，罰 ratio<γ_low（死場/凍結）；0=單側相容
    )
    # band-shell 定向 jac probe。各向同性隨機 eps 漏掉 band(12-40px) 子空間
    # → jac_reg 盲於 band gain（解釋收縮 backfire）。True=對 jac probe eps 做 DoG band-pass → 探測中波長增益。
    jac_band_probe: bool = False
    # Dy-V9：JAWS 空間自適應 Jacobian reg（plan §3.6）。逐 pixel 加權收縮：
    # 平滑區強收縮（殺噪聲）、渦核區放鬆（保結構）→ 收縮但不 smooth-blob。
    jaws_enabled: bool = (
        False  # True=逐 pixel 加權（需 jac_reg_weight>0；V9 per-pixel Jacobian）
    )
    jaws_lambda_c: float = 1.0  # 複雜度→權重斜率 w=1/(1+λ_c·complexity)
    jaws_n_probes: int = 1  # Hutchinson 探針數（turb JAWS MAP loss 用）
    jaws_map_enabled: bool = (
        False  # True=turb_step Hutchinson+VJP MAP loss short-circuit（Eq.9）
    )

    # ── Dy2：全開整合訓練路徑（dy2_enabled=False 時走原 _train_step，行為完全不變）──
    # Dy2 §3-§5：detached self-exposure + batch 多態 + Δ法A landing-pixel + motion/temporal
    # + Jacobian/JAWS + 錯開 schedule。所有欄位 default OFF/identity。
    dy2_enabled: bool = False

    # §3.2 batch 多態暴露比例（對映 [t0_GT, tn_GT, detached_tn, noise_tn]，預設 4×25%）。
    # 和不必恰為 1（_train_step_dy2 依比例切 batch；最後一臂吸收餘數）。
    dy2_batch_ratios: list[float] = field(
        default_factory=lambda: [0.25, 0.25, 0.25, 0.25]
    )

    # §3.1 detached/grad rollout 步數（schedule 未串接時的固定 fallback；schedule.get_dk/get_lk
    # 存在時由 schedule 取代）。dk=detached 暴露深度（可大）、lk=grad rollout（鐵律短，封頂建議 ~8）。
    dy2_dk: int = 0  # 固定 detached 暴露步數（schedule 未啟用時）
    dy2_lk: int = 4  # 固定 grad rollout 步數
    dy2_gt_delta_ctu: float = 0.2  # Δ法A：細步 rollout 每步 ΔCTU（外推區固定步長）
    dy2_gt_spacing_ctu: float = 0.2  # GT CTU 間距（landing step = 細步累積整除此間距）

    # ── WFC cross-all-CTU（cross_ctu_enabled=True → _train_step_cross_ctu）──────────
    # WFCrossCTU_Warm：單一 warm conditioned 模型，固定 0.2 細步跨整條 staircase，
    # landing 步 pixel（對 _cross_ctu_landing_map 算出的正確 GT frame）、no-GT 步自監督
    # 物理（fneq keep-alive deficit-only + DivJ）撐合法不死。dy2_enabled 須 False。
    cross_ctu_enabled: bool = False
    cross_ctu_dk_preroll: int = (
        0  # detached pushforward no_grad 前置步數（0=全 grad；>0=滾到 drift 態再 grad）
    )
    cross_ctu_nogt_fneq_weight: float = (
        0.1  # no-GT 步 fneq keep-alive 權重（deficit-only）
    )
    cross_ctu_nogt_enstrophy_weight: float = (
        0.0  # no-GT 步 enstrophy(渦量能量)-deficit keep-alive
    )
    # （fneq 量非均衡應力⊥幅值，全程0=沒擋幅值流失；
    #   改用渦量能量 deficit vs detached 上一 landing GT 守幅值）
    cross_ctu_nogt_velfield_weight: float = (
        0.0  # no-GT 步 DyNCA-style 方向+強度解耦 velocity keep-alive
    )
    # （vel_field_decoupled_loss，
    #   方向 1−cos + 強度 deficit 補幅；與 enstrophy keep-alive 並列比較）
    cross_ctu_nogt_velfield_overamp_w: float = (
        0.0  # velfield 強度 over-amp clamp 權重
    )
    # （mp>1.3·mg 輕罰，防更長 rollout 複利過衝；deficit 端不變）
    cross_ctu_nogt_self_anchor: bool = (
        False  # no-GT velfield 強度項參考改「自身上一步
    )
    # 不准更弱」(self-non-decreasing 補幅，教漂移態 re-amplify)；
    # 方向項仍 vs 乾淨 GT 守結構。top-level 直讀(避 dead-knob)。
    cross_ctu_nogt_divj_weight: float = (
        0.0  # no-GT 步 DivJ 自監督權重（0=關，先靠 fneq）
    )
    cross_ctu_temporal_weight: float = (
        0.0  # 整段 temporal-KE 權重（0=關；複用 _temporal_ke_loss）
    )
    cross_ctu_grad_checkpoint: bool = (
        False  # grad 段改 per-step gradient
    )
    # checkpointing → BPTT 鋪滿全 rollout 窗（記憶體換計算繞 OOM）。
    # 配 dk_preroll=0 → grad 穿全窗，教自持極限環。
    # top-level 直讀(避 dead-knob)。
    cross_ctu_ssim_last_n: int = (
        0  # SSIM 項只在末 N 個 landing 算
    )
    # （0=全 landing 照舊）。砍 SSIM graph 記憶體 hog，L1/vort/vel
    # 仍全 landing（每 gap 兩側振幅錨）。top-level 直讀(避 dead-knob)。
    cross_ctu_nogt_spectral_weight: float = (
        0.0  # no-GT 步 2D radial power spectrum
    )
    # match（spectral_power_match_loss）權重。取代 scalar 振幅錨
    # 二分→錨能量結構(per-scale 絕對能量)多自由度破漂/鎖。
    # top-level 直讀(避 dead-knob)。
    cross_ctu_nogt_spectral_khi: float = (
        1.0  # spectral band-limit 上限（k_hi_frac，
    )
    # <1.0 只錨低/中頻能量降 HF；高頻結構交 landing/velf-dir）。
    cross_ctu_nogt_hybrid_mag: bool = (
        False  # spectral active 時仍保 scalar-對稱-mag
    )
    # （dir+scalar-mag+spectral 三合，互補；default false 保原行為）。
    # top-level 直讀(避 dead-knob)。
    cross_ctu_nogt_mag_symmetric: bool = (
        False  # no-GT self-anchor 強度項改兩側對稱
    )
    # （罰偏離上一步雙向 |Δ‖v‖|，非只罰變弱）→ 殺「只推上」棘輪
    # overshoot。保 velfield 方向項(1-cos)+landing 兩側 GT 錨。
    # top-level 直讀(避 dead-knob)。
    cross_ctu_tbptt: bool = (
        True  # per-landing truncated BPTT：landing 邊界
    )
    # detach 切段，每段獨立 scale-backward 立即釋放圖（記憶體~段長~7GB、
    # 免 checkpoint=快、grad 覆蓋全窗每 gap）。tiled 失跨-landing
    # 耦合(limit-cycle 穩定性無損)。配 dk_preroll=0+grad_checkpoint=false。
    # top-level 直讀(避 dead-knob)。
    # GOAL 達後全域 flip False→True=cross-CTU 標準訓練
    # 基建（升格 thesis 方法論；長 rollout 多步 BPTT 預設，
    # 單步 bucketing 不受影響=cross_ctu_enabled=false 不入此路）。

    # ── Phase 0 mean_module 訓練路徑（LC decompose）──────────────
    mean_module_enabled: bool = False  # True → dispatch to _mean_module_train_step
    mean_rollout_k: int = (
        24  # rollout 步數（24×32 px/step=768≥對角 734；傳播 FULL H×W）
    )
    mean_grad_tail: int = 8  # TBPTT grad tail（preroll=k−tail = self-rollout 暴露）
    mean_ssim_weight: float = (
        0.3  # SSIM loss 權重（total = MSE + w×(1−SSIM)；作用 jx+jy raw channel）
    )
    mean_vel_ssim_weight: float = (
        0.0  # vel_mag fixed-range（視覺面板值域）SSIM 權重；0=關（抗尾流過平滑）
    )
    mean_vort_ssim_weight: float = (
        0.0  # vorticity fixed-range SSIM 權重；0=關（直接督渦街結構，最抗平滑）
    )

    # ── 不會飽和的銳利度／渦度目標 ──────────────────────────────────────
    # 背景：渦度經 `(ω+R)/(2R)` 映射後全場擠在 0.5 附近，SSIM 的局部變異數遠小於
    #   穩定常數 C2=9e-4 → 比值恆為 1、與內容無關，使 `mean_vort_ssim_weight` 對
    #   渦度結構幾乎不起作用；只調小 R 只是延後塌陷點，沒有根除問題。
    # 實測指出真正的缺陷：渦量總能量與高渦度面積比皆接近 GT，但速度梯度能量比明顯
    #   偏低——模型輸出比 GT 平滑，這才是視覺缺陷所在。而 SSIM 在**任何值域下**對
    #   模糊都幾乎全盲，需要獨立於 SSIM 的目標函數。
    # 兩項都復用既有實作（`loss/terms/struct_velocity.py`），不新寫 loss；建議權重
    # 約 0.4–0.5，使各項貢獻與其餘 loss 同量級（訓練後期總 loss 約 0.5–1.0）。
    # 兩項都是 max(E_t, E_p) 正規化的**有界**比值（∈[0,1]），不是逐點 MAE——逐點的
    # 無界版本配大權重會在訓練初期就發散，有界化是這兩項可用的前提。
    mean_enstrophy_weight: float = (
        0.0  # StructVort_EnstrophyMatch；罰渦量總能量低於 GT（非逐點，不飽和）
    )
    mean_velgrad_weight: float = (
        0.0  # StructVel_GradEnergyRatio；有界的 ⟨|∇u|²⟩ 相對匹配＝直接對準「比 GT 更平滑」的視覺缺陷
    )
    # `HorizontalBand_FFT` 是 eval 端量測的軸之一，但 mean 訓練路徑原本沒有對應的
    #   訓練項（`mean_column_banding_weight` 罰的是**垂直**條紋，`mean_dirband_weight`
    #   罰空間局域方向性，皆非此軸）。`horiz_band_fft_loss` 是該 eval 軸的可微版本，
    #   早就實作好、只是沒接線。建議權重約 2.0，使貢獻與其餘 loss 同量級。
    mean_hband_weight: float = (
        0.0  # StructHBandFFT_Loss；罰 pred 相對 GT 的 kx=0 超額能量
    )
    mean_column_banding_weight: float = (
        0.0  # ★抗垂直條紋：罰 ρ/jx/jy column 高頻超 GT；0=關
    )
    mean_dirband_weight: float = (
        0.0  # ★空間局域方向性 band loss；自由流區罰 x-梯度主導超GT（頻域不可分→換空間軸）；0=關
    )
    mean_delta_focus_weight: float = 0.0  # child Δ（尾流偏轉）聚焦加權 MSE 權重；0=關
    mean_delta_focus_gamma: float = (
        1.0  # Δ 權重指數：w=(|target−parent|/mean)^γ（γ>1 更聚焦高 Δ 區）
    )
    mean_colored_huber_weight: float = (
        0.0  # colored vel/vort RGB 空間 wake-weighted Huber（視覺優先主項）；0=關
    )
    mean_colored_huber_beta: float = 0.1  # smooth_l1 轉折點（colored∈[0,1]）
    mean_moment_weight: float = 1.0  # mphy moment MSE 次要層級權重（保留矩監督）
    mean_rel_l2_weight: float = (
        0.0  # 相對 L2 ‖pred−tgt‖/‖tgt‖ 輔助（sqrt 非模糊 + 跨 case 正規化）；0=關
    )
    # ── VGG 感知 loss（DyNCA 機制：深層特徵統計，破解 MSE→均勻場模糊；v15）──
    mean_vgg_weight: float = (
        0.0  # structure-matching 權重（VGG 特徵 spatial L1，保尾流位置）；0=關
    )
    mean_vgg_moment_weight: float = (
        0.0  # moment-matching 權重（VGG 特徵 channel-wise mean+std 差，抗模糊核心）；0=關
    )
    mean_vgg_ot_weight: float = (
        0.0  # Sliced OT（Sliced Wasserstein）權重，DyNCA SOTA style；比 moment/Gram 佳；0=關
    )
    mean_vgg_max_size: int = (
        256  # VGG 前降採樣上限（360×640 全解析太慢；統計不需全解析）
    )
    mean_vgg_layers: str = (
        "3,8,15"  # 擷取特徵的 VGG relu 層索引（逗號分隔；relu1_2=3 / relu2_2=8 / relu3_3=15）
    )
    # ── 複數低-mode 頻譜 loss（FNO 風：低頻複數係數匹配，能量加權優先重要 mode；v18）──
    mean_spectral_weight: float = (
        0.0  # 複數低-mode 頻譜匹配權重（含相位，破均勻+釘大尺度結構）；0=關
    )
    mean_spectral_modes: int = (
        24  # 低頻保留 mode 數（兩空間維，比照 FNO fno_modes=24）；高頻不錨（mean 場低頻主導）
    )
    # ── SFA2-02：全譜 radial power match（spectral_power_match_loss）——罰高k尾 excess（grain/band/tv 譜根）──
    mean_specmatch_weight: float = (
        0.0  # ★全譜功率匹配權重（k_hi_frac 全頻）；補 lowmode 未罰的高k尾；0=關
    )
    mean_specmatch_khi: float = 1.0  # k_hi_frac：1.0=全頻段（含高k尾）；<1=只錨低/中頻
    mean_grad_normalize: bool = (
        False  # Mordvintsev NCA：per-param 梯度正規化（grad/=‖grad‖）→ 每步等幅小更新抗平滑
    )
    mean_stay_weight: float = (
        0.0  # 兩視野「保持」監督：繼續 rollout 後仍須 match target（學穩定固定點，非固定步收斂）；0=關
    )
    mean_stay_extra_k_min: int = (
        2  # k1 收斂後繼續的額外步數下限（隨機，學任意時長保持）
    )
    mean_stay_extra_k_max: int = 6  # 額外步數上限
    # ── RolloutSampler（常態分布漸進 K；取代 uniform randint）+ Experience Pool（§14）──
    mean_rollout_sampler: bool = (
        False  # True → K~N(μ(t),σ(t))，μ 由 sigmoid 從 K_min 滑向 K_max
    )
    mean_sampler_mu_schedule: str = "sigmoid"
    mean_sampler_sigma_schedule: str = "decay"
    mean_pool_enabled: bool = (
        False  # True → Experience Pool（cross-iteration 累積偏離，pool 即 pushforward 機制）
    )
    mean_pool_size: int = 64
    mean_pool_seed_frac: float = 0.15  # 每 batch 重置為 seed（從頭收斂）比例
    mean_pool_corrupt_frac: float = 0.10  # 加噪 state（自修復壓力）比例
    mean_pool_corrupt_sigma: float = 0.2  # corrupt 噪聲量級（norm 空間）
    mean_pool_warmup_frac: float = 0.05  # 前 frac 訓練走正常路徑灌 pool，之後啟用 pool
    mean_pool_damage_frac: float = (
        0.0  # Growing-NCA local damage：pool slot 中挖 rect 區域比例（0=停用）
    )
    mean_pool_damage_mode: str = "rect"  # 破壞幾何（目前僅 rect）
    mean_pool_damage_size_frac: float = 0.2  # 破壞區域邊長 / 場邊長
    mean_pool_refill_interval: int = (
        1  # ★深池：>1=warmup 後每 N 步才 refill（rolling slots 深層累積）
    )
    mean_pool_damage_fill: str = "zero"  # zero=硬挖（Growing NCA 經典）/ noise=軟破壞
    # 跨案例鏈式接力機率。parent case
    # writeback 後，以此機率查其 children（build_same_zarr_child_map），若 child 當下
    # 也在池中，把 parent 的 model 預測輸出接成 child 的新 IC。補訓練期「IC 恆為 GT
    # 父案例」與評測期「IC 為 model 自身鏈式輸出」的系統性落差。0=停用，零行為變更。
    mean_pool_chain_frac: float = 0.0
    mean_per_channel_norm: bool = (
        False  # True → MSE 各 ch 除 target 變異數（等貢獻，防 velocity 被 rho 淹沒；pilot 揭露不平衡）
    )
    mean_fp_reg_weight: float = (
        0.0  # GT-anchored fixed-point reg：‖f(init(static,x̄_GT))−x̄_GT‖²_fluid（visible 13ch 用，0=關）
    )
    mean_fp_rollout_k: int = (
        1  # ★fp-reg 多步 identity rollout 步數（1=舊 1 步；>1 罰 K 步漂移+rollout banding）
    )
    mean_inv_reg_weight: float = (
        0.0  # ★self-recurse invariance：re-inject 自身輸出 roll K 罰漂回 GT
    )
    mean_inv_depth_max: int = 0  # 額外 no_grad re-inject 循環上限（造深層複利態）
    mean_divj_weight: float = (
        0.0  # 散度軟懲罰 λ_divj：phys_mass_divj_absmean（▶FIX-C7，0=關）
    )
    mean_overflow_weight: float = (
        0.0  # DyNCA Overflow loss：(state-clamp(-1,1)).abs().mean()（全通道含hidden，0=關）
    )
    mean_rollout_k_min: int = (
        0  # 隨機 k 下限（0=停用隨機 k，使用 mean_rollout_k 固定值）
    )
    mean_rollout_k_max: int = (
        0  # 隨機 k 上限（0=同 k_min；>0 → k~U[k_min,k_max] 相位不變訓練）
    )
    mean_cold_ic: bool = (
        False  # True → cold IC（trajectory[:,0]，無 parent_mean_renorm 時使用；P3 cold ablation）
    )
    mean_vflip_aug_prob: float = (
        0.0  # 垂直翻轉 aug 機率（CFD 左進右出僅 y 鏡像合法；只套 train）；物理變換見 LC_Mean_VFlip_Aug_Analysis.md
    )

    # ── 起點屬性的受控析因（論文 §5.2「暖啟動能力的來源」）──────────────
    # 各臂僅改變訓練時的邊選取與初始條件來源，資料集與模型架構完全相同。
    # 兩個正交屬性：A 流形membership（起點是否為某幾何的真實收斂時均流場）、
    # B 目標相關性（起點與監督目標是否有設計上的鄰近關係）。
    #
    #   臂          A   B   IC 來源
    #   use_edge    ✓   ✓   資料集既有暖啟動邊的父節點收斂時均場（現狀）
    #   far_edge    ✓   ✗   同資料集中距離最遠之節點時均場
    #   rand_edge   ✓   ✗   同資料集中隨機抽取之節點時均場（far 的乾淨版）
    #   noise_edge  ✗   ✓   父節點時均場加大幅高斯雜訊（**唯一只移除 A 的操作**）
    #   no_edge     ✗   ✗   零場
    #
    # far_edge／rand_edge 於 dataset 層以 src 覆寫實作（graph_split）；
    # no_edge／noise_edge 於 mean_step 送入 init_state 前覆寫 IC
    # （該處為 run() 與 run_chain() 的共同匯合點）。
    mean_ablation_arm: str = "use_edge"
    # noise_edge 的雜訊標準差，取該案流體區均方根的比例。事前登記值為 0.5，
    # 遠大於 train_noise_sigma（0.02）以確保起點確實離開流形。
    mean_ablation_noise_sigma: float = 0.5

    # ── multi-edge pushforward（缺陷 b：跨幾何 warm 累積；mean_config 同名同型同預設）──
    mean_chain_enabled: bool = False  # 啟用 multi-edge pushforward 訓練分支
    mean_chain_len: int = 2  # edge 數（stage 數）；2=攻 hand-off 本體
    mean_chain_prob: float = (
        0.5  # 每步抽 chain 分支機率（其餘 normal run 保 fp-reg/jac）
    )
    mean_chain_stage_weight: float = 1.0  # stage j>1 hand-off loss 權重
    mean_chain_batch_size: int = 2  # chain batch
    mean_chain_vgg: bool = False  # stage j>1 是否算 VGG（預設 lean）
    mean_chain_ke_weight: float = 0.0  # ★energy-anchor（校準證訓練尺度 no-op，預設關）
    mean_chain_converge_k: int = 16  # chain 每 stage rollout 步數
    mean_chain_contract: bool = (
        False  # ★chain 每 stage 加 fp-reg+jac（鏡射 run() 收縮包）
    )
    mean_chain_fp_rollout_k: int = 0  # 0=沿用 mean_fp_rollout_k
    # ghost-IC handoff 修復：stage j>1 hand-off 時，IC 重置區從「僅新幾何固體」擴大
    # 為「新固體∪舊固體∪變動邊界膨脹 dilate_px」，把該區交還模型既有 damage-regrow
    # 能力重新長出。0=關（原行為，零風險）。
    mean_chain_handoff_dilate_px: int = 0

    # ── Phase 1 turb_module 訓練路徑（M3 預建，`_turb_module_train_step` 接線，
    #    欄位先行避免 config 載入失敗）──
    # grain 指紋鑑別 loss（redesign grain_fingerprint_loss 移植進 operator_gated 線）：
    # 鑑別 loss（非相似度）：對 rollout 後段 denorm vorticity 算斜向各向異性 P(θ)+Nyquist hinge，
    # 只罰 grain 指紋不誤殺合法細渦。w_grain 須校到與 loss_pod(O(24-88)) 同量級（grain_fingerprint
    # 內部 O(0.01-0.1)→w_grain O(100-1000)），否則被 POD 壓死=從未施加。
    # grain 權重建議搭配 warmup ramp：從已收斂的 checkpoint 續訓時，若直接吃滿 w_grain
    # 梯度容易使權重被推爆 NaN。grain 有效權重從 0 線性 ramp 到 turb_rd_w_grain
    # （前 turb_rd_grain_ramp_steps 步），讓模型漸進適應 grain 梯度，不突然爆。0=不 ramp（向後相容）。

    # Mean/turb GT injection schedule（scheduled sampling）
    # progress = step/max_steps；[0, p1]: p_gt=1.0；[p1,p2]: linear；[p2,1]: p_gt=0.0
    mean_gt_phase1_end: float = 0.20  # fraction：此前 p_gt 保持 1.0（全 GT injection）
    mean_gt_phase2_end: float = (
        0.80  # fraction：此後 p_gt 降至 0.0（全 TBPTT self-rollout）
    )

    # Turb module GT supervision mode
    # Turb module IC source（warm-transfer）

    # ── condition concat ──
    # True → state 由 [static4, m′9] 擴成 [static4, m′9, m̄9, ∇m̄4]=26ch，m̄+∇m̄ 接 m′ 後當
    # 唯讀 input condition（取代 mean FiLM）。需配 model.state_dim=26 / mean_film=false /
    # arch='turb_perception_a'。False（預設）= 原 13ch。

    # ── ▶GRAFT-1 統計監督與課程學習（M3 預建欄位對齊） ──
    mean_mix_ckpt: str | None = None
    rij_loss_weight: float = 0.0
    tke_loss_weight: float = 0.0
    spectral_loss_weight: float = 0.0
    ke_ratio_penalty_weight: float = (
        0.0  # KE ratio ≠1 懲罰（治 IC_Copy > ic_max / 長時能量膨脹）
    )
    ke_ratio_full_rollout: bool = (
        False  # True → KE ratio 罰逐幀雙向（全 rollout 段平均，非僅末幀；治長時 speckle 發散）
    )
    ic_copy_penalty_weight: float = (
        0.0  # 末幀 IC_Copy ratio > 1 直接懲罰（all-channel L2）
    )
    w_f_start: float = 1.0
    w_f_end: float = 1.0

    # §5.3 錯開 schedule 各軸 phase 窗（互不重疊；欄位名與 SCHED agent schedule.py 的 getattr 對齊：
    # ``<axis>_phase_start/end`` + ``<axis>_init/final``）。各軸 init→final 線性 ramp：
    # step<start→init、start<=step<end→線性、step>=end→final；退化窗（end<=start）→恆 init。
    # dk 暴露深度（§5.3 Phase C）
    dk_phase_start: int = 0
    dk_phase_end: int = 0
    dk_init: float = 0.0
    dk_final: float = 0.0
    # lk grad rollout（§5.3，通常固定短，封頂 ~8）
    lk_phase_start: int = 0
    lk_phase_end: int = 0
    lk_init: float = 4.0
    lk_final: float = 4.0
    # noise σ（§5.3 Phase D）
    noise_phase_start: int = 0
    noise_phase_end: int = 0
    noise_sigma_init: float = 0.0
    noise_sigma_final: float = 0.0
    # Jacobian γ（§5.3 Phase E；γ 1.0→0.95 收縮漸強，方向由 schedule helper 解讀，此處存兩端點）
    jac_phase_start: int = 0
    jac_phase_end: int = 0
    jac_gamma_init: float = 1.0
    jac_gamma_final: float = 1.0

    # §10 Dy2-S1 weight ramps：temporal/jac loss **權重**漸進（與 §5.3 各軸 phase 錯開）。
    # init/final = NaN（未設）→ schedule helper fallback 至固定 cfg.temporal.weight /
    # cfg.jac_reg_weight（向後相容無 schedule）。設值後線性 ramp（get_temporal_weight /
    # get_jac_weight）。tempw=Phase E（0→0.2）、jacw=Phase D（0→0.05，與 jac_gamma 同窗）。
    tempw_phase_start: int = 0
    tempw_phase_end: int = 0
    tempw_init: float = float("nan")
    tempw_final: float = float("nan")
    jacw_phase_start: int = 0
    jacw_phase_end: int = 0
    jacw_init: float = float("nan")
    jacw_final: float = float("nan")

    # §8.1 週期 long-roll eval：每 N 步跑一次 evaluate_long_rollout 並寫 event=longroll。0=off。
    periodic_longroll_interval: int = 0

    # §4 motion loss（velocity-field 兩幀運動統計 moment matching）在 Dy2 total loss 的加權。
    # 0 = 停用（與既有行為一致）。motion 對「所有 step」算（相位不變防 smooth-blob）。
    motion_loss_weight: float = 0.0

    # IC masking：warm-start IC 中的固體區域歸零
    apply_ic_mask: bool = True

    # Optimizer
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # NaN early stop：連續 NaN/spike 步數上限（超過則儲存 emergency ckpt 並停止）
    nan_patience: int = 20
    # loss spike 閾值：loss > 此值視為不穩定步（同樣計入 nan_patience 計數）
    loss_spike_threshold: float = 500.0

    # 靜態通道歸零（舊 X2 方案，已棄用）

    # X2 static transform: C0 SDF_norm → binary fluid mask; C1 upstream_norm → SDF_norm
    # 用途：消除 C1 upstream_norm 的 kx=0 貢獻；C0 改為 {0,1} mask
    # sdf_fluid_threshold：norm space 中 raw_sdf=0 的對應值 = -sdf_mean/sdf_std
    # S1mix 三個 dataset 一致：≈ -1.203（由 train.py 從 stats 自動計算後填入）

    # X3 static transform: C0 SDF_norm → soft mask sigmoid(sdf_norm * scale); C1 ← SDF_norm
    # 用途：取代 X2 binary mask 的硬邊界，全可微邊界過渡；避免 warm-start semantic mismatch
    # 建議搭配 from-scratch 訓練（不 finetune），mask_scale 控制過渡寬度

    # 子設定
    model: TESModelConfig = field(default_factory=TESModelConfig)
    # Semantic loss mode. New configs must use names such as:
    # {"ObsMphy_Traj_Mae": 1.0, "ObsMphy_Delta_Mae": 0.5, ...}
    loss_terms: dict[str, float] = field(default_factory=dict)
    pod_basis_path: str = ""
    pod_modes: int = 16
    curriculum: TESCurriculumConfig = field(default_factory=TESCurriculumConfig)
    early_stop: TESEarlyStopConfig = field(default_factory=TESEarlyStopConfig)
    # Teacher Forcing Curriculum
    tfc: TESTFCConfig = field(default_factory=TESTFCConfig)
    # Scheduled Sampling（取代 TFC 的 DAgger-style 方案）
    ss: TESSScheduleConfig = field(default_factory=TESSScheduleConfig)
    # Detached Self-Rollout
    rp: TESRPConfig = field(default_factory=TESRPConfig)
    # RollingPool（state-continuous warm-state memory；mode=off 時行為不變）
    pool: TESPoolConfig = field(default_factory=TESPoolConfig)
    # Dy L_temporal（solver 2CTU KE 統計穩態；enabled=False 時行為不變）
    temporal: TESTemporalConfig = field(default_factory=TESTemporalConfig)

    def __post_init__(self) -> None:
        _check(self.lr > 0, f"TESTrainConfig.lr 必須 >0，得到 {self.lr}")
        _check(self.lr_min >= 0, f"TESTrainConfig.lr_min 必須 >=0，得到 {self.lr_min}")
        _check(
            self.max_steps > 0,
            f"TESTrainConfig.max_steps 必須 >0，得到 {self.max_steps}",
        )
        _check(
            self.warmup_steps >= 0,
            f"TESTrainConfig.warmup_steps 必須 >=0，得到 {self.warmup_steps}",
        )
        _check(
            self.lr_schedule in ("cosine", "warm_restart"),
            f"TESTrainConfig.lr_schedule 必須是 'cosine' 或 'warm_restart'，得到 '{self.lr_schedule}'",
        )
        _check(
            0.0 < self.lr_restart_peak_decay <= 1.0,
            f"TESTrainConfig.lr_restart_peak_decay 必須在 (0,1]，得到 {self.lr_restart_peak_decay}",
        )
        _check(
            self.lr_restart_warmup >= 0,
            f"TESTrainConfig.lr_restart_warmup 必須 >=0，得到 {self.lr_restart_warmup}",
        )
        if self.lr_schedule == "warm_restart":
            b = self.lr_restart_boundaries
            _check(
                len(b) >= 1
                and list(b) == sorted(b)
                and b[0] >= 0
                and b[-1] < self.max_steps,
                f"TESTrainConfig.lr_restart_boundaries 須遞增、起點>=0、末項<max_steps，得到 {b}",
            )
        _check(
            self.batch_size > 0,
            f"TESTrainConfig.batch_size 必須 >0，得到 {self.batch_size}",
        )
        _check(
            self.num_workers >= 0,
            f"TESTrainConfig.num_workers 必須 >=0，得到 {self.num_workers}",
        )
        _check(
            self.window_size > 0,
            f"TESTrainConfig.window_size 必須 >0，得到 {self.window_size}",
        )
        _check(
            self.val_interval > 0,
            f"TESTrainConfig.val_interval 必須 >0，得到 {self.val_interval}",
        )
        _check(
            self.ckpt_interval > 0,
            f"TESTrainConfig.ckpt_interval 必須 >0，得到 {self.ckpt_interval}",
        )
        _check(
            self.vis_interval > 0,
            f"TESTrainConfig.vis_interval 必須 >0，得到 {self.vis_interval}",
        )
        _check(
            self.weight_decay >= 0,
            f"TESTrainConfig.weight_decay 必須 >=0，得到 {self.weight_decay}",
        )
        _check(
            self.grad_clip_norm > 0,
            f"TESTrainConfig.grad_clip_norm 必須 >0，得到 {self.grad_clip_norm}",
        )
        _check(
            self.nan_patience > 0,
            f"TESTrainConfig.nan_patience 必須 >0，得到 {self.nan_patience}",
        )
        _check(
            self.loss_spike_threshold > 0,
            f"TESTrainConfig.loss_spike_threshold 必須 >0，得到 {self.loss_spike_threshold}",
        )
        _check(
            self.stats_policy in ("single", "dataset_global_runtime"),
            f"TESTrainConfig.stats_policy 必須是 'single' 或 'dataset_global_runtime'，得到 '{self.stats_policy}'",
        )
        _check(
            self.ckpt_identity_mismatch in ("warn", "raise"),
            "TESTrainConfig.ckpt_identity_mismatch 必須是 'warn' 或 'raise'，"
            f"得到 '{self.ckpt_identity_mismatch}'",
        )
        _check(
            self.ckpt_identity_mismatch_resume in ("warn", "raise"),
            "TESTrainConfig.ckpt_identity_mismatch_resume 必須是 'warn' 或 'raise'，"
            f"得到 '{self.ckpt_identity_mismatch_resume}'",
        )
        _check(
            self.mean_ablation_arm
            in (
                "use_edge",
                "far_edge",
                "rand_edge",
                "noise_edge",
                "no_edge",
                # farnoise_edge：far 的 src 覆寫（B✗）＋ 加噪（A✗）。
                # 它才是真正的 A✗B✗ 格；no_edge（零場）退回「原點」參考點角色。
                "farnoise_edge",
            ),
            "TESTrainConfig.mean_ablation_arm 必須是 'use_edge'／'far_edge'／"
            "'rand_edge'／'noise_edge'／'no_edge'／'farnoise_edge'，"
            f"得到 '{self.mean_ablation_arm}'",
        )
        _check(
            self.mean_ablation_noise_sigma > 0,
            "TESTrainConfig.mean_ablation_noise_sigma 必須為正，"
            f"得到 {self.mean_ablation_noise_sigma}",
        )
        _check(
            self.determinism_mode in ("off", "warn_only", "strict"),
            "TESTrainConfig.determinism_mode 必須是 'off'／'warn_only'／'strict'，"
            f"得到 '{self.determinism_mode}'",
        )
        # 先判型別再比大小：直接 float(v) 會在 _check 參數求值階段丟出原生
        # ValueError/TypeError（yaml 寫成字串／null 時），使用者看不到本訊息。
        _check(
            all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and float(v) >= 0.0
                for v in self.repeat_tolerance.values()
            ),
            "TESTrainConfig.repeat_tolerance 各值必須為非負數值（int/float，不接受 bool／"
            f"字串／null），得到 {self.repeat_tolerance}",
        )

        _check(
            self.pod_modes > 0,
            f"TESTrainConfig.pod_modes must be >0, got {self.pod_modes}",
        )

        # Dy-V8/V9：Jacobian reg / JAWS
        _check(
            self.jac_reg_weight >= 0.0,
            f"TESTrainConfig.jac_reg_weight 必須 >=0，得到 {self.jac_reg_weight}",
        )
        _check(
            self.jac_eps > 0.0, f"TESTrainConfig.jac_eps 必須 >0，得到 {self.jac_eps}"
        )
        _check(
            self.jac_gamma > 0.0,
            f"TESTrainConfig.jac_gamma 必須 >0，得到 {self.jac_gamma}",
        )
        _check(
            self.jaws_lambda_c >= 0.0,
            f"TESTrainConfig.jaws_lambda_c 必須 >=0，得到 {self.jaws_lambda_c}",
        )
        _check(
            self.jaws_n_probes >= 1,
            f"TESTrainConfig.jaws_n_probes 必須 >=1，得到 {self.jaws_n_probes}",
        )

        # Dy2：全開整合路徑欄位
        _check(
            len(self.dy2_batch_ratios) == 4,
            f"TESTrainConfig.dy2_batch_ratios 必須是 4 個 float，得到 {self.dy2_batch_ratios}",
        )
        self.dy2_batch_ratios = [float(r) for r in self.dy2_batch_ratios]
        _check(
            all(r >= 0.0 for r in self.dy2_batch_ratios),
            f"TESTrainConfig.dy2_batch_ratios 各項必須 >=0，得到 {self.dy2_batch_ratios}",
        )
        _check(
            sum(self.dy2_batch_ratios) > 0.0,
            "TESTrainConfig.dy2_batch_ratios 總和必須 >0",
        )
        _check(self.dy2_dk >= 0, f"TESTrainConfig.dy2_dk 必須 >=0，得到 {self.dy2_dk}")
        _check(self.dy2_lk >= 1, f"TESTrainConfig.dy2_lk 必須 >=1，得到 {self.dy2_lk}")
        _check(
            self.dy2_gt_delta_ctu > 0.0,
            f"TESTrainConfig.dy2_gt_delta_ctu 必須 >0，得到 {self.dy2_gt_delta_ctu}",
        )
        _check(
            self.dy2_gt_spacing_ctu > 0.0,
            f"TESTrainConfig.dy2_gt_spacing_ctu 必須 >0，得到 {self.dy2_gt_spacing_ctu}",
        )
        # 各 schedule 軸 phase 窗：end >= start（lo/hi 方向不限制，留 schedule helper 解讀）
        for _tag, _s, _e in (
            ("dk", self.dk_phase_start, self.dk_phase_end),
            ("lk", self.lk_phase_start, self.lk_phase_end),
            ("noise", self.noise_phase_start, self.noise_phase_end),
            ("jac", self.jac_phase_start, self.jac_phase_end),
            ("tempw", self.tempw_phase_start, self.tempw_phase_end),
            ("jacw", self.jacw_phase_start, self.jacw_phase_end),
        ):
            _check(_s >= 0, f"TESTrainConfig.{_tag}_phase_start 必須 >=0，得到 {_s}")
            _check(
                _e >= _s,
                f"TESTrainConfig.{_tag}_phase_end({_e}) 必須 >= {_tag}_phase_start({_s})",
            )
        _check(
            self.dk_init >= 0 and self.dk_final >= 0,
            "TESTrainConfig.dk_init/dk_final 必須 >=0",
        )
        _check(
            self.lk_init >= 1 and self.lk_final >= 1,
            "TESTrainConfig.lk_init/lk_final 必須 >=1",
        )
        _check(
            self.noise_sigma_init >= 0.0 and self.noise_sigma_final >= 0.0,
            "TESTrainConfig.noise_sigma_init/final 必須 >=0",
        )
        _check(
            self.jac_gamma_init > 0.0 and self.jac_gamma_final > 0.0,
            "TESTrainConfig.jac_gamma_init/final 必須 >0",
        )
        _check(
            self.periodic_longroll_interval >= 0,
            f"TESTrainConfig.periodic_longroll_interval 必須 >=0，得到 {self.periodic_longroll_interval}",
        )
        _check(
            self.motion_loss_weight >= 0.0,
            f"TESTrainConfig.motion_loss_weight 必須 >=0，得到 {self.motion_loss_weight}",
        )
        _check(
            0.0 <= self.mean_gt_phase1_end <= 1.0,
            f"mean_gt_phase1_end 必須在 [0,1]，得到 {self.mean_gt_phase1_end}",
        )
        _check(
            self.mean_gt_phase1_end <= self.mean_gt_phase2_end <= 1.0,
            f"mean_gt_phase2_end 必須 >= phase1_end 且 <=1，得到 {self.mean_gt_phase2_end}",
        )


_DATACLASS_FIELD_NAMES = {
    "TESTrainConfig": {f.name for f in fields(TESTrainConfig)},
    "TESModelConfig": {f.name for f in fields(TESModelConfig)},
    "TESMultiScaleConfig": {f.name for f in fields(TESMultiScaleConfig)},
    "TESTransportConfig": {f.name for f in fields(TESTransportConfig)},
    "BaselineConfig": {f.name for f in fields(BaselineConfig)},
    "MRPConfig": {f.name for f in fields(MRPConfig)},
    "TESCurriculumConfig": {f.name for f in fields(TESCurriculumConfig)},
    "TESEarlyStopConfig": {f.name for f in fields(TESEarlyStopConfig)},
    "TESTFCConfig": {f.name for f in fields(TESTFCConfig)},
    "TESSScheduleConfig": {f.name for f in fields(TESSScheduleConfig)},
    "TESRPConfig": {f.name for f in fields(TESRPConfig)},
    "TESPoolConfig": {f.name for f in fields(TESPoolConfig)},
    "TESTemporalConfig": {f.name for f in fields(TESTemporalConfig)},
}


def _warn_unknown_keys(
    yaml_path: Path, section: str, raw: dict[str, Any], known: set[str]
) -> None:
    """對未知 YAML 欄位發出 warning（不中斷載入）。

    Args:
        yaml_path: 設定檔路徑（用於 warning 訊息）。
        section: YAML 節名稱（用於 warning 訊息）。
        raw: 待驗證的原始 dict。
        known: 已知合法欄位名稱集合。
    """
    unknown = set(raw.keys()) - known
    if unknown:
        _logger.warning(
            "%s [%s]: 未知欄位 %s（可能為手誤，將被忽略）",
            yaml_path,
            section,
            sorted(unknown),
        )


def expand_config_placeholders(node: Any) -> Any:
    """遞迴展開設定樹中所有字串的 ``${...}`` 路徑佔位符。

    yaml 內的資料路徑一律寫成 ``${ZARR_ROOT}/xxx.zarr`` 這類與機器無關的形式，
    實際位置由 ``lbmnca.paths`` 依環境變數決定。載入 yaml 後立即整棵展開，
    下游便不需要知道佔位符存在。

    Args:
        node: yaml 解析結果的任一節點（dict / list / str / 其他純量）。

    Returns:
        同結構的節點，其中所有字串已展開佔位符；非字串純量原樣回傳。
    """
    if isinstance(node, str):
        return resolve_placeholders(node)
    if isinstance(node, dict):
        return {k: expand_config_placeholders(v) for k, v in node.items()}
    if isinstance(node, list):
        return [expand_config_placeholders(v) for v in node]
    return node


def load_tes_config(yaml_path: str | Path) -> TESTrainConfig:
    """從 YAML 載入 TES 訓練設定，未指定欄位採用 dataclass 預設值。

    所有子設定均經由 ``__init__`` 重新建構，因此 ``__post_init__`` 驗證會在
    YAML 提供的值上觸發（不再是 silent setattr）。未知欄位會發出 warning。

    Args:
        yaml_path: YAML 設定檔路徑。

    Returns:
        TESTrainConfig 實例。

    Raises:
        FileNotFoundError: YAML 檔不存在。
        ValueError: 遇到 legacy ``loss_weights`` 或欄位值違反 dataclass 不變量。
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"設定檔不存在：{yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        raw: dict[str, Any] = expand_config_placeholders(yaml.safe_load(f) or {})

    if "loss_weights" in raw:
        raise ValueError(
            f"{yaml_path}: legacy numeric loss configs are no longer supported. "
            "Use semantic loss_terms instead."
        )

    top_known = _DATACLASS_FIELD_NAMES["TESTrainConfig"]
    _warn_unknown_keys(yaml_path, "root", raw, top_known)

    def _filter(cls_name: str, sub_raw: dict[str, Any], section: str) -> dict[str, Any]:
        """過濾 sub_raw，只保留 cls_name 的已知欄位，並對未知欄位發出 warning。

        Args:
            cls_name: 目標 dataclass 名稱。
            sub_raw: 待過濾的原始 dict。
            section: YAML 節名稱（用於 warning）。

        Returns:
            只含合法欄位的 dict。
        """
        known = _DATACLASS_FIELD_NAMES[cls_name]
        _warn_unknown_keys(yaml_path, section, sub_raw, known)
        return {k: v for k, v in sub_raw.items() if k in known}

    # 子設定：MRP + Model
    model_raw = dict(raw.get("model", {}) or {})
    mrp_raw = dict(model_raw.pop("mrp", {}) or {})
    multiscale_raw = dict(model_raw.pop("multiscale", {}) or {})
    transport_raw = dict(model_raw.pop("transport", {}) or {})
    mrp_cfg = MRPConfig(**_filter("MRPConfig", mrp_raw, "model.mrp"))
    multiscale_cfg = TESMultiScaleConfig(
        **_filter("TESMultiScaleConfig", multiscale_raw, "model.multiscale")
    )
    transport_cfg = TESTransportConfig(
        **_filter("TESTransportConfig", transport_raw, "model.transport")
    )
    baseline_raw = dict(model_raw.pop("baseline", {}) or {})
    baseline_cfg = BaselineConfig(
        **_filter("BaselineConfig", baseline_raw, "model.baseline")
    )
    model_cfg = TESModelConfig(
        mrp=mrp_cfg,
        multiscale=multiscale_cfg,
        transport=transport_cfg,
        baseline=baseline_cfg,
        **_filter("TESModelConfig", model_raw, "model"),
    )

    def _build_sub(name: str, cls: type[Any]) -> Any:
        """從 raw 中取出 name 節並建構對應的 dataclass 實例。

        Args:
            name: YAML 節名稱，同時作為 raw 的 key。
            cls: 目標 dataclass 類別。

        Returns:
            cls 的實例，以 YAML 節內容初始化。
        """
        sub_raw = dict(raw.get(name, {}) or {})
        return cls(**_filter(cls.__name__, sub_raw, name))

    curriculum_cfg = _build_sub("curriculum", TESCurriculumConfig)
    early_stop_cfg = _build_sub("early_stop", TESEarlyStopConfig)
    tfc_cfg = _build_sub("tfc", TESTFCConfig)
    ss_cfg = _build_sub("ss", TESSScheduleConfig)
    rp_cfg = _build_sub("rp", TESRPConfig)
    pool_cfg = _build_sub("pool", TESPoolConfig)
    temporal_cfg = _build_sub("temporal", TESTemporalConfig)

    # Top-level kwargs（排除子設定的 key，避免重覆傳入）
    _NESTED_KEYS = {
        "model",
        "curriculum",
        "early_stop",
        "tfc",
        "ss",
        "rp",
        "pool",
        "temporal",
    }
    top_kwargs: dict[str, Any] = {}
    for key, val in raw.items():
        if key in _NESTED_KEYS or key not in top_known:
            continue
        if key == "zarr_paths":
            top_kwargs[key] = list(val)
        elif key == "stats_paths":
            top_kwargs[key] = list(val)
        elif key == "dataset_names":
            top_kwargs[key] = list(val)
        elif key == "dy2_batch_ratios":
            top_kwargs[key] = [float(x) for x in val]
        elif key == "zero_static_channels":
            top_kwargs[key] = [int(c) for c in val]
        elif key == "loss_terms":
            top_kwargs[key] = {str(k): float(v) for k, v in val.items()}
        elif key == "pod_basis_path":
            top_kwargs[key] = str(val)
        elif key == "pod_modes":
            top_kwargs[key] = int(val)
        else:
            top_kwargs[key] = val

    train_cfg = TESTrainConfig(
        model=model_cfg,
        curriculum=curriculum_cfg,
        early_stop=early_stop_cfg,
        tfc=tfc_cfg,
        ss=ss_cfg,
        rp=rp_cfg,
        pool=pool_cfg,
        temporal=temporal_cfg,
        **top_kwargs,
    )
    # Propagate top-level pod_basis_path / pod_modes into model config so
    # H5 architectures (factory only sees cfg.model) can locate the basis.
    if not train_cfg.model.pod_basis_path and train_cfg.pod_basis_path:
        train_cfg.model.pod_basis_path = train_cfg.pod_basis_path
    if "pod_modes" not in (model_raw or {}):
        train_cfg.model.pod_modes = int(train_cfg.pod_modes)
    # ★來源 yaml 記錄（非 dataclass 欄位 → 不進 asdict、不影響 config sha256）。
    # 供 checkpoint identity 記下「訓練此 ckpt 的 config 檔是哪一份」，
    # 讓事後 re-gate 拿錯 yaml 時能被指名（見 training/checkpoint/identity.py）。
    # 條件向量只餘 cond_17D = [G_vec_norm(14), P_vec_norm(3)]。
    #   判準放在**載入端**而非 dataclass：`cond_dim` 對 dataclass 而言只是
    #   「condition encoder 的輸入寬度」這個架構參數，架構測試會用各種值建模型；
    #   真正的語義約束是「本專案的資料只組得出 17D」，那屬於 config 檔這一層。
    #   放錯層級會讓一堆與資料無關的模型測試陪葬（實測炸掉 120 支）。
    if train_cfg.model.cond_dim != 17:
        raise ValueError(
            f"{yaml_path}: model.cond_dim 必須為 17（G_vec 14 + P_vec 3），"
            f"得到 {train_cfg.model.cond_dim}。含時間維的 19/5/4 已於 2026-08-05 移除"
            "——mean flow 是時均場，delta_ctu/cumulative_ctu 兩維過去恆為常數填補。"
        )
    train_cfg._source_yaml = str(yaml_path.resolve())
    return train_cfg
