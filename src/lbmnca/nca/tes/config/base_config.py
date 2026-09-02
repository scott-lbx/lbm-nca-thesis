"""mean/turb 共用設定基底：模型結構（BaseArchConfig）＋資料（DataConfig）＋優化器（OptimConfig）。

這三個 dataclass 為 mean / turb 兩條獨立 pipeline 共用的底層設定積木：

- ``BaseArchConfig``：模型結構（state/cond 維度、nca_channels、spectral、arch、以及巢狀的
  MRP / multiscale / transport / baseline 子設定）。mean yaml 設 hidden=8/mrp=true/spectral=48，
  turb yaml 設 hidden=0；結構欄位本身共用同一 schema。
- ``DataConfig``：資料來源與分割（zarr_paths/stats/window/split_seed/val_fraction…）。
- ``OptimConfig``：優化器與排程（lr/lr_schedule/max_steps/warmup/amp/grad_clip…）。

★欄位與既有 ``TESModelConfig`` / ``TESTrainConfig`` 的對應欄位**同名同型同預設**，
保證新舊 schema 一致、yaml 可互通；巢狀結構直接複用 legacy 的子 dataclass（單一真相）。

★禁 getattr fallback：所有存取一律 ``cfg.key`` 直存，缺欄位即 ``AttributeError``。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ._legacy import (
    BaselineConfig,
    MRPConfig,
    TESMultiScaleConfig,
    TESTransportConfig,
    _check,
)

__all__ = [
    "BaseArchConfig",
    "DataConfig",
    "OptimConfig",
    # 巢狀結構子設定（自 legacy re-export，單一真相）
    "MRPConfig",
    "TESMultiScaleConfig",
    "TESTransportConfig",
    "BaselineConfig",
]


@dataclass
class BaseArchConfig:
    """模型結構超參（mean / turb 共用 schema）。

    對應既有 ``TESModelConfig`` 的結構欄位子集（同名同型同預設）。巢狀子設定
    （mrp / multiscale / transport / baseline）直接複用 legacy dataclass。

    state_dim = static_dim + physics_dim（+ hidden_dim 若啟用）。
    mean 部署版 hidden_dim=8（state_dim 21）；turb 維持 hidden_dim=0（state_dim 13）。
    """

    arch: str = "operator_gated_streamcollide"

    # 狀態向量維度
    state_dim: int = 13
    static_dim: int = 4
    physics_dim: int = 9
    hidden_dim: int = 0

    # 條件向量維度（G_vec + P_vec [+ dt_log + ct_log]）
    cond_dim: int = 17

    # ConditionEncoder MLP 隱藏維度
    encoder_hidden: int = 128

    # MarkovNCACell 內部特徵維度
    nca_channels: int = 64

    # 頻域 update 分支
    nca_spectral_update: bool = False
    nca_spectral_modes: int = 24
    nca_msp_scales: list[int] = field(default_factory=lambda: [1])  # MSP 下採因子；[1]=單尺度

    # G_vec dual-pathway conditioning
    g_vec_dim: int = 0
    gvec_mode: str = "additive"

    # rate 形式（fdelta / f）
    rate_mode: str = "fdelta"

    # 軟夾 per-CTU rate
    update_clip: float = 0.0

    # 訓練時 gradient checkpointing
    grad_checkpoint: bool = False

    # LC decompose turb mean conditioning（spatial FiLM）
    mean_film: bool = False

    # 巢狀子設定（複用 legacy dataclass）
    mrp: MRPConfig = field(default_factory=MRPConfig)
    multiscale: TESMultiScaleConfig = field(default_factory=TESMultiScaleConfig)
    transport: TESTransportConfig = field(default_factory=TESTransportConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    def __post_init__(self) -> None:
        _check(self.state_dim > 0, f"BaseArchConfig.state_dim 必須 >0，得到 {self.state_dim}")
        _check(self.static_dim >= 0, f"BaseArchConfig.static_dim 必須 >=0，得到 {self.static_dim}")
        _check(self.physics_dim > 0, f"BaseArchConfig.physics_dim 必須 >0，得到 {self.physics_dim}")
        _check(self.hidden_dim >= 0, f"BaseArchConfig.hidden_dim 必須 >=0，得到 {self.hidden_dim}")
        _check(self.cond_dim >= 0, f"BaseArchConfig.cond_dim 必須 >=0，得到 {self.cond_dim}")
        _check(self.encoder_hidden > 0, f"BaseArchConfig.encoder_hidden 必須 >0，得到 {self.encoder_hidden}")
        _check(self.nca_channels > 0, f"BaseArchConfig.nca_channels 必須 >0，得到 {self.nca_channels}")
        _check(self.g_vec_dim >= 0, f"BaseArchConfig.g_vec_dim 必須 >=0，得到 {self.g_vec_dim}")
        _check(self.update_clip >= 0.0, f"BaseArchConfig.update_clip 必須 >=0，得到 {self.update_clip}")
        _check(self.rate_mode in ("fdelta", "f"),
               f"BaseArchConfig.rate_mode 必須是 'fdelta' | 'f'，得到 '{self.rate_mode}'")
        _check(self.gvec_mode in ("none", "additive", "gated"),
               f"BaseArchConfig.gvec_mode 必須是 'none' | 'additive' | 'gated'，得到 '{self.gvec_mode}'")
        if self.gvec_mode == "gated":
            _check(self.g_vec_dim > 0, "BaseArchConfig.gvec_mode='gated' 需要 g_vec_dim > 0")
        if self.hidden_dim > 0:
            expected = self.static_dim + self.physics_dim + self.hidden_dim
            _check(self.state_dim == expected,
                   f"BaseArchConfig.state_dim({self.state_dim}) 與 static+physics+hidden({expected}) 不一致")
        if self.arch == "turb_renca":
            _check(self.hidden_dim == 0,
                   f"turb_renca 為 no-hidden 設計，hidden_dim 必須 0，得到 {self.hidden_dim}")
            _check(self.g_vec_dim >= 1,
                   f"turb_renca VectorFiLM 需 g_vec_dim>=1，得到 {self.g_vec_dim}")


@dataclass
class DataConfig:
    """資料來源、正規化策略與訓練/驗證分割（mean / turb 共用）。

    對應既有 ``TESTrainConfig`` 的資料欄位子集（同名同型同預設）。
    """

    zarr_path: str = ""
    zarr_paths: list[str] = field(default_factory=list)
    stats_paths: list[str] = field(default_factory=list)
    stats_policy: str = "single"
    dataset_names: list[str] = field(default_factory=list)
    stats_path: str = ""
    # TURB redesign 錨定 Stage-1：mature_segment sidecar（1:1 對應 zarr_paths，
    # 提供成熟段錨定起點，避 t=0 transient，spec §12.3）。其他 pipeline 不讀。
    mature_sidecars: list[str] = field(default_factory=list)

    # DataLoader
    batch_size: int = 4
    num_workers: int = 4
    window_size: int = 12

    # 分割

    # IC mask（warm-start IC 固體區歸零）
    apply_ic_mask: bool = True

    def __post_init__(self) -> None:
        _check(self.stats_policy in ("single", "dataset_global_runtime"),
               f"DataConfig.stats_policy 必須是 'single' 或 'dataset_global_runtime'，得到 '{self.stats_policy}'")
        _check(self.batch_size > 0, f"DataConfig.batch_size 必須 >0，得到 {self.batch_size}")
        _check(self.num_workers >= 0, f"DataConfig.num_workers 必須 >=0，得到 {self.num_workers}")
        _check(self.window_size > 0, f"DataConfig.window_size 必須 >0，得到 {self.window_size}")


@dataclass
class OptimConfig:
    """優化器、LR 排程、混合精度與梯度裁剪（mean / turb 共用）。

    對應既有 ``TESTrainConfig`` 的優化器欄位子集（同名同型同預設）。
    """

    lr: float = 1e-3
    lr_min: float = 1e-5
    max_steps: int = 100_000
    warmup_steps: int = 500

    # LR schedule（cosine / warm_restart）
    lr_schedule: str = "cosine"
    lr_restart_boundaries: list[int] = field(default_factory=list)
    lr_restart_peak_decay: float = 0.85
    lr_restart_warmup: int = 100

    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # 混合精度
    use_amp: bool = True

    # 全域 seed
    seed: int = 0

    # ── R7 決定性／重測（與 legacy TESTrainConfig 同名同型同預設）──
    # 語義與實作見 `_legacy.py::TESTrainConfig` 對應欄位註解與
    # `training/rng_streams.py`。三個行為欄位（determinism_mode／
    # rng_stream_isolation／loader_seed_offset）預設為 strict／True／True
    # （與 legacy 同步，兩處必須一致）；repeat_tolerance 仍為空 dict 預設，
    # 實際值寫在各 config yaml 頂層。
    # `src/lbmnca/nca/tes/train.py` 走的是 legacy `TESTrainConfig` 的同名欄位，
    # 不經過本 dataclass；本 dataclass 內只有下方 `__post_init__` 對
    # repeat_tolerance 做型別驗證（不擋訓練）。
    # ★repeat_tolerance 的 yaml 寫法：必須放在**頂層**（本 dataclass 是以
    #   `OptimConfig(**_filter(raw, _OPTIM_FIELD_NAMES))` 從整份 yaml 的頂層建的，
    #   `optim:` 鍵反而在兩支載入器的 `_consumed` 集合內而完全不讀，且無 warning）。
    #   守衛：`tests/nca/tes/test_r7_countable_claims.py` 的 yaml 頂層／無 `optim:` 節兩條斷言。
    determinism_mode: str = "strict"
    repeat_tolerance: dict[str, float] = field(default_factory=dict)
    rng_stream_isolation: bool = True
    loader_seed_offset: bool = True

    # NaN / spike 守衛
    nan_patience: int = 20
    loss_spike_threshold: float = 500.0

    # 存檔與驗證間隔（步數）
    val_interval: int = 1_000
    ckpt_interval: int = 2_000
    vis_interval: int = 1_000

    # 輸出目錄
    output_dir: str = "runs/misc/default"

    def __post_init__(self) -> None:
        _check(self.lr > 0, f"OptimConfig.lr 必須 >0，得到 {self.lr}")
        _check(self.lr_min >= 0, f"OptimConfig.lr_min 必須 >=0，得到 {self.lr_min}")
        _check(self.max_steps > 0, f"OptimConfig.max_steps 必須 >0，得到 {self.max_steps}")
        _check(self.warmup_steps >= 0, f"OptimConfig.warmup_steps 必須 >=0，得到 {self.warmup_steps}")
        _check(self.lr_schedule in ("cosine", "warm_restart"),
               f"OptimConfig.lr_schedule 必須是 'cosine' 或 'warm_restart'，得到 '{self.lr_schedule}'")
        _check(0.0 < self.lr_restart_peak_decay <= 1.0,
               f"OptimConfig.lr_restart_peak_decay 必須在 (0,1]，得到 {self.lr_restart_peak_decay}")
        _check(self.lr_restart_warmup >= 0,
               f"OptimConfig.lr_restart_warmup 必須 >=0，得到 {self.lr_restart_warmup}")
        if self.lr_schedule == "warm_restart":
            b = self.lr_restart_boundaries
            _check(len(b) >= 1 and list(b) == sorted(b) and b[0] >= 0 and b[-1] < self.max_steps,
                   f"OptimConfig.lr_restart_boundaries 須遞增、起點>=0、末項<max_steps，得到 {b}")
        _check(self.weight_decay >= 0, f"OptimConfig.weight_decay 必須 >=0，得到 {self.weight_decay}")
        _check(self.grad_clip_norm > 0, f"OptimConfig.grad_clip_norm 必須 >0，得到 {self.grad_clip_norm}")
        _check(self.nan_patience > 0, f"OptimConfig.nan_patience 必須 >0，得到 {self.nan_patience}")
        _check(self.loss_spike_threshold > 0,
               f"OptimConfig.loss_spike_threshold 必須 >0，得到 {self.loss_spike_threshold}")
        _check(self.val_interval > 0, f"OptimConfig.val_interval 必須 >0，得到 {self.val_interval}")
        _check(self.ckpt_interval > 0, f"OptimConfig.ckpt_interval 必須 >0，得到 {self.ckpt_interval}")
        _check(self.vis_interval > 0, f"OptimConfig.vis_interval 必須 >0，得到 {self.vis_interval}")
        _check(self.determinism_mode in ("off", "warn_only", "strict"),
               "OptimConfig.determinism_mode 必須是 'off'／'warn_only'／'strict'，"
               f"得到 '{self.determinism_mode}'")
        # 先判型別再比大小：直接 float(v) 會在 _check 參數求值階段丟出原生
        # ValueError/TypeError（yaml 寫成字串／null 時），使用者看不到本訊息。
        _check(all(isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) >= 0.0
                   for v in self.repeat_tolerance.values()),
               "OptimConfig.repeat_tolerance 各值必須為非負數值（int/float，不接受 bool／"
               f"字串／null），得到 {self.repeat_tolerance}")
