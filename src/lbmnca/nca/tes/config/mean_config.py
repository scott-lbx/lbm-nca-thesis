"""LC decompose Phase 0 mean module 訓練設定（MeanTrainConfig + load_mean_config）。

``MeanTrainConfig`` 組合
``BaseArchConfig`` + ``DataConfig`` + ``OptimConfig``（共用底層）＋ mean-only 欄位
（rollout_k/grad_tail、各式 vel/vort/spectral SSIM、fp_reg、stay、delta_focus、
moment、rel_l2、overflow、pool、rollout_sampler、vflip_aug、jacobian reg…）。

``load_mean_config`` 自 yaml 載入並各自 ``__post_init__`` 驗證**自己**的不變量。
廢除 ``mean_module_enabled``（入口即決定 module，flag 無意義）。

★欄位與 legacy ``TESTrainConfig`` 的 ``mean_*`` / jac / 巢狀子設定**同名同型同預設**，
保證 mean yaml 在新舊 schema 皆可載入。
★禁 getattr fallback：所有存取一律 ``cfg.key`` 直存。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from ._legacy import (
    BaselineConfig,
    MRPConfig,
    TESModelConfig,
    TESMultiScaleConfig,
    TESTemporalConfig,
    TESTransportConfig,
    _check,
    expand_config_placeholders,
)
from .base_config import BaseArchConfig, DataConfig, OptimConfig

_logger = logging.getLogger(__name__)

__all__ = ["MeanTrainConfig", "load_mean_config"]


@dataclass
class MeanTrainConfig:
    """mean module 完整訓練設定：共用底層（arch/data/optim）＋ mean-only 欄位。

    巢狀組合而非繼承：``arch`` / ``data`` / ``optim`` 各為獨立 dataclass，
    透過 ``load_mean_config`` 由 yaml 對應節（``model`` / 頂層 data / 頂層 optim）建構。
    """

    arch: BaseArchConfig = field(default_factory=BaseArchConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    # ── rollout / TBPTT ──
    mean_rollout_k: int = 24
    mean_grad_tail: int = 8
    mean_rollout_k_min: int = 0
    mean_rollout_k_max: int = 0

    # ── SSIM / 視覺結構 loss ──
    mean_ssim_weight: float = 0.3
    mean_vel_ssim_weight: float = 0.0
    mean_vort_ssim_weight: float = 0.0
    mean_column_banding_weight: float = 0.0   # ★抗垂直條紋：罰 ρ/jx/jy column 高頻超 GT
    mean_delta_focus_weight: float = 0.0
    mean_delta_focus_gamma: float = 1.0
    mean_colored_huber_weight: float = 0.0
    mean_colored_huber_beta: float = 0.1
    mean_moment_weight: float = 1.0
    mean_rel_l2_weight: float = 0.0

    # ── VGG 感知 loss ──
    mean_vgg_weight: float = 0.0
    mean_vgg_moment_weight: float = 0.0
    mean_vgg_ot_weight: float = 0.0
    mean_vgg_max_size: int = 256
    mean_vgg_layers: str = "3,8,15"

    # ── 複數低-mode 頻譜 loss ──
    mean_spectral_weight: float = 0.0
    mean_spectral_modes: int = 24
    mean_grad_normalize: bool = False

    # ── stay（兩視野保持監督）──
    mean_stay_weight: float = 0.0
    mean_stay_extra_k_min: int = 2
    mean_stay_extra_k_max: int = 6

    # ── RolloutSampler + Experience Pool ──
    mean_rollout_sampler: bool = False
    mean_sampler_mu_schedule: str = "sigmoid"
    mean_sampler_sigma_schedule: str = "decay"
    mean_pool_enabled: bool = False
    mean_pool_size: int = 64
    mean_pool_seed_frac: float = 0.15
    mean_pool_corrupt_frac: float = 0.10
    mean_pool_corrupt_sigma: float = 0.2
    mean_pool_warmup_frac: float = 0.05
    # Growing-NCA local damage：pool slot 中挖連續 rect 區域（physics 通道）→ 練習
    # 「局部破壞後靠鄰域自組織重建」。與全域高斯 corrupt 並存互補。frac=0 即停用。
    mean_pool_damage_frac: float = 0.0
    mean_pool_damage_mode: str = "rect"        # 破壞幾何（目前僅 rect）
    mean_pool_damage_size_frac: float = 0.2    # 破壞區域邊長 / 場邊長
    mean_pool_damage_fill: str = "zero"        # zero=硬挖（Growing NCA 經典）/ noise=軟破壞
    # ★深池：refill 每步灌 fresh state 把 rolling slots 深度重置（~1-2 層），漏掉自遞迴 i≥6 深層漂移。
    #   refill_interval>1 → warmup 後只每 N 步灌，rolling slots 存活久、累積深層 re-injection（治淺池）。
    mean_pool_refill_interval: int = 1         # 1=每步灌（現行）；>1=週期灌（深池）

    # ── 其他 mean-only ──
    mean_per_channel_norm: bool = False
    mean_fp_reg_weight: float = 0.0
    mean_fp_rollout_k: int = 1            # ★fp-reg 多步 identity：從 GT IC roll K 步仍須≈GT（1=舊 1 步）
    # ★self-recurse invariance（root-cause：算子吸引子≠GT，re-inject 自身輸出漂到髒高能態）：
    #   re-inject 模型自身收斂輸出當 IC → roll K → 罰漂回 GT。與 fp-reg 差別=IC 是模型輸出非 GT，
    #   覆蓋 warm-transfer 的 re-injection 流形。depth 隨機 0..max 額外 no_grad re-inject 造深層複利態。
    mean_inv_reg_weight: float = 0.0
    mean_inv_depth_max: int = 0           # 額外 no_grad re-inject 循環上限（造深層態，攻 i≥6 才顯著的複利）
    mean_divj_weight: float = 0.0
    mean_overflow_weight: float = 0.0
    mean_cold_ic: bool = False
    mean_vflip_aug_prob: float = 0.0

    # ── multi-edge pushforward（缺陷 b：跨幾何 warm 累積；單邊 loss 訓不到）──
    # 訓練期 unroll 連續 chain_len 個 warm edge（模型輸出 hand-off 當下一 IC），逐 stage 錨 GT。
    # 與 normal 步以 chain_prob 混編（normal 步保 fp-reg/jac/stay 收縮監督）。
    mean_chain_enabled: bool = False
    mean_chain_len: int = 2               # edge 數（stage 數）；2=攻 hand-off 本體
    mean_chain_prob: float = 0.5          # 每步抽 chain 分支機率（其餘走 normal run）
    mean_chain_stage_weight: float = 1.0  # stage j>1（hand-off）loss 權重
    mean_chain_batch_size: int = 2        # chain batch（L stages ×B ≈ normal B 一步量）
    mean_chain_vgg: bool = False          # stage j>1 是否算 VGG（預設 lean 省時）
    mean_chain_ke_weight: float = 0.0     # ★energy-anchor（momentum-KE 雙向 log-ratio）；校準證訓練尺度 no-op，預設關
    mean_chain_converge_k: int = 16       # chain 每 stage rollout 步數（no_grad preroll 便宜，grad tail=min(grad_tail,6)）
    mean_chain_contract: bool = False     # ★chain 每 stage 加 fp-reg+jac（鏡射 run() 收縮包，治泵浦=收縮稀釋假說）
    mean_chain_fp_rollout_k: int = 0      # 0=沿用 mean_fp_rollout_k；OOM 時減載（如 4）
    # ★ghost-IC handoff 修復：stage j>1 IC 重置區擴大為新固體∪舊固體∪變動邊界膨脹 dilate_px
    # px（見 spatial_ops.handoff_reset_mask）。0=關（原行為，零風險）。
    mean_chain_handoff_dilate_px: int = 0

    # ── GT injection schedule（scheduled sampling）──
    mean_gt_phase1_end: float = 0.20
    mean_gt_phase2_end: float = 0.80

    # ── Jacobian 收縮正則（mean 訓練穩定性）──
    jac_reg_weight: float = 0.0
    mean_jac_interval: int = 1
    mean_jac_start_frac: float = 0.0
    mean_jac_ramp_frac: float = 0.1
    jac_eps: float = 1e-2
    jac_gamma: float = 1.0
    jac_gamma_low: float = 0.0
    jaws_enabled: bool = False
    jaws_lambda_c: float = 1.0

    # ── training noise injection ──
    train_noise_sigma: float = 0.0

    # ── L_temporal（巢狀，複用 legacy 子設定）──
    temporal: TESTemporalConfig = field(default_factory=TESTemporalConfig)

    # ── semantic loss term 權重 ──
    loss_terms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check(self.mean_rollout_k >= 1, f"MeanTrainConfig.mean_rollout_k 必須 >=1，得到 {self.mean_rollout_k}")
        _check(self.mean_grad_tail >= 0, f"MeanTrainConfig.mean_grad_tail 必須 >=0，得到 {self.mean_grad_tail}")
        _check(self.mean_rollout_k_min >= 0, f"MeanTrainConfig.mean_rollout_k_min 必須 >=0，得到 {self.mean_rollout_k_min}")
        _check(self.mean_rollout_k_max >= 0, f"MeanTrainConfig.mean_rollout_k_max 必須 >=0，得到 {self.mean_rollout_k_max}")
        if self.mean_rollout_k_max > 0:
            _check(self.mean_rollout_k_min <= self.mean_rollout_k_max,
                   f"MeanTrainConfig.mean_rollout_k_min({self.mean_rollout_k_min}) 必須 <= "
                   f"mean_rollout_k_max({self.mean_rollout_k_max})")
        for _name in (
            "mean_ssim_weight", "mean_vel_ssim_weight", "mean_vort_ssim_weight",
            "mean_column_banding_weight",
            "mean_delta_focus_weight", "mean_colored_huber_weight", "mean_moment_weight",
            "mean_rel_l2_weight", "mean_vgg_weight", "mean_vgg_moment_weight",
            "mean_vgg_ot_weight", "mean_spectral_weight", "mean_stay_weight",
            "mean_fp_reg_weight", "mean_divj_weight", "mean_overflow_weight",
            "mean_inv_reg_weight",
        ):
            _v = getattr(self, _name)  # 自身欄位反射僅供統一驗證迴圈，非 config fallback
            _check(_v >= 0.0, f"MeanTrainConfig.{_name} 必須 >=0，得到 {_v}")
        _check(self.mean_inv_depth_max >= 0,
               f"MeanTrainConfig.mean_inv_depth_max 必須 >=0，得到 {self.mean_inv_depth_max}")
        _check(self.mean_pool_refill_interval >= 1,
               f"MeanTrainConfig.mean_pool_refill_interval 必須 >=1，得到 {self.mean_pool_refill_interval}")
        _check(self.mean_spectral_modes > 0,
               f"MeanTrainConfig.mean_spectral_modes 必須 >0，得到 {self.mean_spectral_modes}")
        _check(self.mean_vgg_max_size > 0,
               f"MeanTrainConfig.mean_vgg_max_size 必須 >0，得到 {self.mean_vgg_max_size}")
        _check(self.mean_stay_extra_k_min >= 0 and self.mean_stay_extra_k_max >= self.mean_stay_extra_k_min,
               f"MeanTrainConfig.mean_stay_extra_k_min/max 非法："
               f"{self.mean_stay_extra_k_min}/{self.mean_stay_extra_k_max}")
        _check(0.0 <= self.mean_vflip_aug_prob <= 1.0,
               f"MeanTrainConfig.mean_vflip_aug_prob 必須在 [0,1]，得到 {self.mean_vflip_aug_prob}")
        _check(self.mean_chain_len >= 2,
               f"MeanTrainConfig.mean_chain_len 必須 >=2，得到 {self.mean_chain_len}")
        _check(0.0 <= self.mean_chain_prob <= 1.0,
               f"MeanTrainConfig.mean_chain_prob 必須在 [0,1]，得到 {self.mean_chain_prob}")
        _check(self.mean_chain_stage_weight >= 0.0 and self.mean_chain_ke_weight >= 0.0,
               f"MeanTrainConfig.mean_chain_stage_weight/ke_weight 必須 >=0")
        _check(self.mean_chain_batch_size >= 1,
               f"MeanTrainConfig.mean_chain_batch_size 必須 >=1，得到 {self.mean_chain_batch_size}")
        _check(self.mean_chain_converge_k >= 1,
               f"MeanTrainConfig.mean_chain_converge_k 必須 >=1，得到 {self.mean_chain_converge_k}")
        _check(self.mean_chain_handoff_dilate_px >= 0,
               f"MeanTrainConfig.mean_chain_handoff_dilate_px 必須 >=0，得到 {self.mean_chain_handoff_dilate_px}")
        _check(self.mean_pool_size > 0, f"MeanTrainConfig.mean_pool_size 必須 >0，得到 {self.mean_pool_size}")
        for _name in ("mean_pool_seed_frac", "mean_pool_corrupt_frac", "mean_pool_warmup_frac"):
            _v = getattr(self, _name)
            _check(0.0 <= _v <= 1.0, f"MeanTrainConfig.{_name} 必須在 [0,1]，得到 {_v}")
        _check(self.mean_pool_corrupt_sigma >= 0.0,
               f"MeanTrainConfig.mean_pool_corrupt_sigma 必須 >=0，得到 {self.mean_pool_corrupt_sigma}")
        _check(0.0 <= self.mean_pool_damage_frac <= 1.0,
               f"MeanTrainConfig.mean_pool_damage_frac 必須在 [0,1]，得到 {self.mean_pool_damage_frac}")
        _check(self.mean_pool_seed_frac + self.mean_pool_corrupt_frac + self.mean_pool_damage_frac <= 1.0,
               f"MeanTrainConfig：seed+corrupt+damage frac 之和必須 <=1（其餘為 rolling），得到 "
               f"{self.mean_pool_seed_frac + self.mean_pool_corrupt_frac + self.mean_pool_damage_frac}")
        _check(self.mean_pool_damage_mode in ("rect",),
               f"MeanTrainConfig.mean_pool_damage_mode 目前僅支援 'rect'，得到 {self.mean_pool_damage_mode}")
        _check(0.0 < self.mean_pool_damage_size_frac <= 1.0,
               f"MeanTrainConfig.mean_pool_damage_size_frac 必須在 (0,1]，得到 {self.mean_pool_damage_size_frac}")
        _check(self.mean_pool_damage_fill in ("zero", "noise"),
               f"MeanTrainConfig.mean_pool_damage_fill 必須為 'zero'/'noise'，得到 {self.mean_pool_damage_fill}")
        _check(0.0 <= self.mean_gt_phase1_end <= 1.0,
               f"MeanTrainConfig.mean_gt_phase1_end 必須在 [0,1]，得到 {self.mean_gt_phase1_end}")
        _check(self.mean_gt_phase1_end <= self.mean_gt_phase2_end <= 1.0,
               f"MeanTrainConfig.mean_gt_phase2_end 必須 >= phase1_end 且 <=1，得到 {self.mean_gt_phase2_end}")
        # Jacobian reg
        _check(self.jac_reg_weight >= 0.0, f"MeanTrainConfig.jac_reg_weight 必須 >=0，得到 {self.jac_reg_weight}")
        _check(self.mean_jac_interval >= 1, f"MeanTrainConfig.mean_jac_interval 必須 >=1，得到 {self.mean_jac_interval}")
        _check(0.0 <= self.mean_jac_start_frac <= 1.0,
               f"MeanTrainConfig.mean_jac_start_frac 必須在 [0,1]，得到 {self.mean_jac_start_frac}")
        _check(self.mean_jac_ramp_frac >= 0.0,
               f"MeanTrainConfig.mean_jac_ramp_frac 必須 >=0，得到 {self.mean_jac_ramp_frac}")
        _check(self.jac_eps > 0.0, f"MeanTrainConfig.jac_eps 必須 >0，得到 {self.jac_eps}")
        _check(self.jac_gamma > 0.0, f"MeanTrainConfig.jac_gamma 必須 >0，得到 {self.jac_gamma}")
        _check(self.jaws_lambda_c >= 0.0, f"MeanTrainConfig.jaws_lambda_c 必須 >=0，得到 {self.jaws_lambda_c}")
        _check(self.train_noise_sigma >= 0.0,
               f"MeanTrainConfig.train_noise_sigma 必須 >=0，得到 {self.train_noise_sigma}")


# ── YAML 載入 ─────────────────────────────────────────────────────────────

_MEAN_FIELD_NAMES = {f.name for f in fields(MeanTrainConfig)}
_ARCH_FIELD_NAMES = {f.name for f in fields(BaseArchConfig)}
_DATA_FIELD_NAMES = {f.name for f in fields(DataConfig)}
_OPTIM_FIELD_NAMES = {f.name for f in fields(OptimConfig)}


# `build_mean_training` 對同一份 yaml 載入兩次 —— `load_mean_config`（本模組，只做驗證）
# 與 `load_tes_config`（legacy，**實際建模型的那條**）。model 區段的欄位集合兩邊不同：
# 例如 `psi_curl_momentum` / `psi_lowpass_sigma` 只在 `TESModelConfig`，不在 `BaseArchConfig`，
# 但 legacy 路徑仍會消費它們（且是 canonical 架構的定義參數），故不能被本模組當作
# 「未知欄位」發警告，否則容易被誤刪。
# 欄位只要被**任一**載入路徑認得就不算未知。以 dataclass 欄位動態求集合，
# 不硬編清單，避免日後兩邊漂移。
_LEGACY_MODEL_FIELD_NAMES = {f.name for f in fields(TESModelConfig)}


def _warn_unknown(yaml_path: Path, section: str, raw: dict[str, Any], known: set[str]) -> None:
    """對未知 YAML 欄位發出 warning（不中斷載入）。

    Args:
        yaml_path: 來源 YAML 路徑（僅供訊息顯示）。
        section: 區段名稱（``model`` / ``data`` / ``optim``）。
        raw: 該區段的原始 dict。
        known: 本載入路徑認得的欄位名集合。
    """
    unknown = set(raw.keys()) - known
    if section == "model":
        # legacy build 路徑消費得到的欄位不算未知（見上方 F1 註解）。
        unknown -= _LEGACY_MODEL_FIELD_NAMES
    if unknown:
        _logger.warning("%s [%s]: 未知欄位 %s（將被忽略）", yaml_path, section, sorted(unknown))


def _filter(raw: dict[str, Any], known: set[str]) -> dict[str, Any]:
    """只保留 known 欄位。"""
    return {k: v for k, v in raw.items() if k in known}


def _build_arch(yaml_path: Path, model_raw: dict[str, Any]) -> BaseArchConfig:
    """從 yaml ``model`` 節建 BaseArchConfig（巢狀子設定複用 legacy parser 慣例）。"""
    model_raw = dict(model_raw)
    mrp_raw = dict(model_raw.pop("mrp", {}) or {})
    multiscale_raw = dict(model_raw.pop("multiscale", {}) or {})
    transport_raw = dict(model_raw.pop("transport", {}) or {})
    baseline_raw = dict(model_raw.pop("baseline", {}) or {})
    mrp_cfg = MRPConfig(**_filter(mrp_raw, {f.name for f in fields(MRPConfig)}))
    multiscale_cfg = TESMultiScaleConfig(
        **_filter(multiscale_raw, {f.name for f in fields(TESMultiScaleConfig)}))
    transport_cfg = TESTransportConfig(
        **_filter(transport_raw, {f.name for f in fields(TESTransportConfig)}))
    baseline_cfg = BaselineConfig(
        **_filter(baseline_raw, {f.name for f in fields(BaselineConfig)}))
    _warn_unknown(yaml_path, "model", model_raw, _ARCH_FIELD_NAMES)
    return BaseArchConfig(
        mrp=mrp_cfg, multiscale=multiscale_cfg,
        transport=transport_cfg, baseline=baseline_cfg,
        **_filter(model_raw, _ARCH_FIELD_NAMES),
    )


def load_mean_config(yaml_path: str | Path) -> MeanTrainConfig:
    """從 YAML 載入 mean module 訓練設定。

    yaml 結構：``model`` 節 → BaseArchConfig；頂層 data/optim 欄位 → DataConfig/OptimConfig；
    頂層 ``mean_*`` / jac / temporal / loss_terms → MeanTrainConfig 自身欄位。
    各 dataclass ``__post_init__`` 在 yaml 值上觸發驗證；未知欄位發 warning。

    Args:
        yaml_path: YAML 設定檔路徑。

    Returns:
        MeanTrainConfig 實例。

    Raises:
        FileNotFoundError: YAML 檔不存在。
        ValueError: 欄位值違反 dataclass 不變量。
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"設定檔不存在：{yaml_path}")
    with open(yaml_path, encoding="utf-8") as f:
        raw: dict[str, Any] = expand_config_placeholders(yaml.safe_load(f) or {})

    arch_cfg = _build_arch(yaml_path, dict(raw.get("model", {}) or {}))
    data_cfg = DataConfig(**_filter(raw, _DATA_FIELD_NAMES))
    optim_cfg = OptimConfig(**_filter(raw, _OPTIM_FIELD_NAMES))

    temporal_raw = dict(raw.get("temporal", {}) or {})
    temporal_cfg = TESTemporalConfig(
        **_filter(temporal_raw, {f.name for f in fields(TESTemporalConfig)}))

    # mean-only 頂層欄位（排除 nested + data/optim 已吃的 key）
    _consumed = _DATA_FIELD_NAMES | _OPTIM_FIELD_NAMES | {"model", "temporal", "arch", "data", "optim"}
    mean_kwargs: dict[str, Any] = {}
    for key, val in raw.items():
        if key in _consumed:
            continue
        if key not in _MEAN_FIELD_NAMES:
            continue
        if key == "loss_terms":
            mean_kwargs[key] = {str(k): float(v) for k, v in (val or {}).items()}
        else:
            mean_kwargs[key] = val

    return MeanTrainConfig(
        arch=arch_cfg, data=data_cfg, optim=optim_cfg,
        temporal=temporal_cfg, **mean_kwargs,
    )
