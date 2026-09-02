"""LC decompose 兩階段推理 handoff 模組函式。

預建（可 import；供前端 DecomposeAdapter 與整合驗證腳本共用）：

    Phase 1  run_mean_phase   — mean_module rollout → KE 兩窗收斂判定 → mean
    Phase 2  run_turb_phase   — turb step 函式無限 rollout，yield mean + fluct
    協調     decompose_rollout — Phase1 → Phase2 串接（mean→turb handoff）

設計重點：
- Phase 1 收斂判準 = `_two_window_ke`（借用 LBM solver 的兩窗邏輯，經
  `mean_gate_eval.ke_curve_convergence` 同款 ratio）。
- Phase 2 的 turb 模型 API（FiLM mean conditioning）尚未定稿，故以
  ``turb_step_fn: (fluct, mean) → fluct`` callable 解耦——待其定案後包一層
  即可；前端在 turb ckpt 就緒前也能先單測 handoff 流程。
- fluct 起點 = `seed_fluct` 固定 seed 小擾動，**非全 0**——
  殘差累加 cell 無內生振盪器，fluct=0 為對稱平衡點，0→0 則永不起振。
- 本模組**介面上**進出的張量一律為 dataset-global norm 空間；組合場 = mean + fluct
  （Reynolds）。``compose_field`` 在拿到 ``normalizer`` 時會先 ``denorm_mphy`` 到 raw
  相加、再 norm 回來（一致性投影），該函式體內因此存在 raw 空間的中間張量。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import torch
from torch import Tensor

from ..eval.ke_convergence import two_window_ke as _two_window_ke
from ..eval.mean_gate_eval import ke_norm
from . import protocol

__all__ = [
    "DecomposeModels",
    "MeanPhaseResult",
    "build_decompose_models",
    "compose_field",
    "decompose_rollout",
    "make_turb_stepper",
    "run_decompose_pipeline",
    "run_mean_phase",
    "run_turb_phase",
    "seed_fluct",
    "seed_from_head",
]


@dataclass
class MeanPhaseResult:
    """Phase 1 輸出：收斂 mean 與診斷曲線。"""

    mean_phys: Tensor                 # [B, 9, H, W] 收斂均場（norm 空間）
    n_steps: int                      # 實際 rollout 步數
    converged: bool                   # KE 兩窗判準是否在 max_steps 內達成
    ke_ratio: float                   # 末次兩窗 |Δmean|/mean
    ke_curve: list[float] = field(default_factory=list)


def run_mean_phase(
    model,
    state: Tensor,
    cond_norm: Tensor,
    fluid_mask: Tensor,
    *,
    static_dim: int,
    physics_dim: int,
    g_vec_norm: Optional[Tensor] = None,
    max_steps: int = 200,
    min_steps: int = 8,
    ctu_per_step: float = protocol.NOMINAL_DELTA_CTU_PER_STEP,
    window_ctu: float = 2.0,
    eps: float = 0.05,
    on_step: Optional[Callable[[int, Tensor], None]] = None,
) -> MeanPhaseResult:
    """Phase 1：mean_module rollout 至 KE 兩窗收斂。

    每步推進一格並累積 norm 空間 KE；自 ``min_steps`` 起逐步檢查
    `_two_window_ke`（pseudo-CTU 軸 = step × ctu_per_step），converged
    即停。eps 初 0.05，過嚴可放寬 0.08。

    Args:
        model: TESNCA 同款 call 介面
            ``model(state, cond_norm, fluid_mask, n, return_trajectory=False,
            g_vec_norm=...) → state``。**額外要求 ``model._cfg``**
            （``TESModelConfig``）：下方 `protocol.forward_anchor_kwargs` 由它讀
            ``transport.ke_anchor_mode`` 決定要不要傳 ``ke_anchor``，禁 fallback。
            ``build_tes_model`` 建出的模型都帶此屬性（各 assembly 類別
            ``self._cfg = cfg``）；純 callable 的 duck-type stub 已不足以滿足本介面。
        state: `protocol.initial_state(model, cfg, static, ic_physics, fluid_mask)` 輸出。
        cond_norm / fluid_mask / g_vec_norm: 同 trainer 推理參數。
        static_dim / physics_dim: state 內 physics 切片位置
            （``state[:, sd:sd+pd]``）。
        max_steps: 收斂上限；未達 converged=False（呼叫端決定 fallback）。
        on_step: 每步回呼 ``(step, phys)``，供序列圖等需要中間幀的呼叫端取樣。
            回呼收到的 ``phys`` 是**當步 tensor 本身**，
            需要保留請自行 ``.clone()``。

    Returns:
        MeanPhaseResult。
    """
    ke_curve: list[float] = []
    ratio = float("nan")
    converged = False
    n_steps = 0

    # 錨點語意由協定定義為「本次 rollout 起始 state 的 KE」，此處的 `state` 正是該起點，
    # 故就地解析即可，不必要求每個呼叫端各自傳。mode='off' 時回 {}。
    anchor_kwargs = protocol.forward_anchor_kwargs(model, state, fluid_mask)

    with torch.inference_mode():
        for step in range(1, max_steps + 1):
            state = model(
                state, cond_norm, fluid_mask, 1,
                return_trajectory=False, g_vec_norm=g_vec_norm,
                **anchor_kwargs,
            )
            phys = state[:, static_dim : static_dim + physics_dim].float()
            ke_curve.append(ke_norm(phys, fluid_mask))
            n_steps = step
            if on_step is not None:
                on_step(step, phys)
            if step >= min_steps:
                ctu_axis = [i * ctu_per_step for i in range(step)]
                conv, ratio = _two_window_ke(
                    ke_curve, ctu_axis, window_ctu=window_ctu, eps=eps
                )
                if conv >= 1.0:
                    converged = True
                    break

    mean_phys = state[:, static_dim : static_dim + physics_dim].float()
    return MeanPhaseResult(
        mean_phys=mean_phys,
        n_steps=n_steps,
        converged=converged,
        ke_ratio=float(ratio),
        ke_curve=ke_curve,
    )


def seed_fluct(
    mean_phys: Tensor,
    *,
    amplitude: float = 1e-3,
    seed: int = 42,
) -> Tensor:
    """turb 起點 fluct：固定 seed 小擾動（非全 0，起振保證）。

    Args:
        mean_phys: [B, 9, H, W]，僅取 shape/device/dtype。
        amplitude: 擾動標準差（norm 空間；遠小於 fluct RMS 0.08-0.25）。

    Returns:
        fluct0: [B, 9, H, W] float32。
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(mean_phys.shape, generator=gen, dtype=torch.float32)
    return noise.to(mean_phys.device) * amplitude


def seed_from_head(seed_head, mean_phys: Tensor, static: Tensor, fluid_mask: Tensor,
                   cond_norm: Optional[Tensor] = None) -> Tensor:
    """u'_IC via SeedFluctHead（部署 handoff：取代 seed_fluct 噪聲）。

    cond 一致性（禁 silent fallback）：head.cond_dim>0 時**必須**傳 cond_norm，
    否則 forward 走 `cond is None` 分支靜默跳過 FiLM → 退化成 v1-identity、與 eval
    路徑（turb_long_rollout）行為不一致。故 cond_dim>0 而 cond_norm=None 直接報錯；
    並 assert cond 維度 == head.cond_dim。

    Args:
        seed_head: 載入權重的 SeedFluctHead（eval 模式）。
        mean_phys: [B, 9, H, W] 收斂 mean（norm）。
        static: [B, 4, H, W]。
        fluid_mask: [B, 1, H, W]。
        cond_norm: [B, cond_dim] 條件向量（head.cond_dim>0 時必填）。
    Returns:
        seed_fluct: [B, 9, H, W]（交 run_turb_phase 的 init_fluct）。
    """
    cd = int(getattr(seed_head, "cond_dim", 0))
    if cd > 0:
        if cond_norm is None:
            raise ValueError(
                f"SeedFluctHead cond_dim={cd} 需 cond_norm，禁傳 None（會靜默跳過 FiLM "
                "退化成 identity，與 eval 路徑不一致）。"
            )
        if cond_norm.shape[-1] != cd:
            raise ValueError(f"cond 維度不符：head.cond_dim={cd}, cond_norm={tuple(cond_norm.shape)}")
    with torch.inference_mode():
        return seed_head(mean_phys, static, fluid_mask, cond_norm)


def compose_field(mean_phys: Tensor, fluct: Tensor, normalizer: Optional[Any] = None) -> Tensor:
    """Reynolds 一致性投影組合：u = norm(denorm(ū) + denorm(u'))。

    在 raw 空間逐通道進行嚴格物理相加，再正規化回 norm 空間，
    避免在 tes_v3 tanh 非線性 norm 空間直接相加產生的不一致瑕疵。
    """
    if normalizer is None:
        return mean_phys + fluct
    # 1. 反正規化回 raw 真實物理空間
    mean_raw = normalizer.denorm_mphy(mean_phys.float())
    fluct_raw = normalizer.denorm_mphy(fluct.float())
    # 2. 進行物理雷諾相加
    composed_raw = mean_raw + fluct_raw
    # 3. 重新正規化回 norm 空間並轉回原裝置/精度
    composed_norm = normalizer.norm_mphy(composed_raw)
    return composed_norm.to(device=mean_phys.device, dtype=mean_phys.dtype)


def run_turb_phase(
    turb_step_fn: Callable[[Tensor, Tensor], Tensor],
    mean_phys: Tensor,
    *,
    init_fluct: Optional[Tensor] = None,
    n_steps: Optional[int] = None,
    fluid_mask: Optional[Tensor] = None,
    rms_ref: Optional[Tensor] = None,
    amp_governor_factor: float = 3.0,
    normalizer: Optional[Any] = None,
) -> Iterator[Tensor]:
    """Phase 2：turb 無限 rollout generator，逐步 yield 完整流場。

    Args:
        turb_step_fn: ``(fluct, mean) → next_fluct``。turb_module 定稿後
            包裝（內部做 FiLM mean conditioning）；測試可用解析函式。
        mean_phys: Phase 1 收斂 mean（推理用 mean_module 預測，非 GT）。
        init_fluct: 起點 fluct；None → `seed_fluct(mean_phys)`。
        n_steps: None = 無限（至呼叫端停止迭代）；int = 固定步數
            （供整合驗證 120+ 步用）。
        fluid_mask: [B, 1, H, W]；提供時對 fluct 套 solid 區歸零。
        rms_ref: 振幅參考 RMS（scalar tensor）。非 None 時啟用振幅 governor：
            RMS(fluct) > amp_governor_factor × rms_ref 時等比縮回
            （修補無限 rollout 缺乏振幅上界的問題）。
        amp_governor_factor: governor 倍率，預設 3.0（GT fluct RMS 分布 0.08-0.25，×3=上界）。
        normalizer: 全域 FieldNormalizer 實例，用於雷諾組合的一致性投影。

    Yields:
        composed: [B, 9, H, W] mean + fluct（每步）。
    """
    fluct = init_fluct if init_fluct is not None else seed_fluct(mean_phys)
    step = 0
    with torch.inference_mode():
        while n_steps is None or step < n_steps:
            fluct = turb_step_fn(fluct, mean_phys)
            if fluid_mask is not None:
                fluct = fluct * fluid_mask.float()
            # 振幅 governor（修補無限 rollout 缺乏振幅上界的問題）
            if rms_ref is not None:
                rms_now = fluct.pow(2).mean().sqrt()
                limit = amp_governor_factor * rms_ref
                if rms_now > limit:
                    fluct = fluct * (limit / rms_now.clamp_min(1e-8))
            yield compose_field(mean_phys, fluct, normalizer=normalizer)
            step += 1


@dataclass
class DecomposeModels:
    """三模組 decompose pipeline 載入結果（mean ⊕ uIC ⊕ turb）。"""

    mean_model: Any
    turb_model: Any
    seed_head: Any
    normalizer: Any                  # TESCondNormalizer（build_cond / cond 統計）
    field_norm: Any                  # FieldNormalizer（denorm/norm + vis_range）
    mean_cond_dim: int
    turb_cond_dim: int
    static_dim: int = 4
    physics_dim: int = 9


def build_decompose_models(
    *,
    mean_ckpt: str,
    mean_config: str,
    turb_ckpt: str,
    turb_config: str,
    seed_ckpt: str,
    device,
) -> DecomposeModels:
    """載入並串接 mean module + uIC SeedFluctHead + turb module 三模組（部署 handoff）。

    cond 一致性（禁 silent fallback）：seed_head.cond_dim 由 ckpt 自動偵測，必須等於
    turb config 的 cond_dim（seed 與 turb 推論共用同一 cond_norm）；不符直接報錯。

    Args:
        mean_ckpt / mean_config: mean module ckpt 與其 yaml。
        turb_ckpt / turb_config: turb module ckpt 與其 yaml。
        seed_ckpt: SeedFluctHead ckpt。
        device: torch device。
    Returns:
        DecomposeModels（皆 eval 模式、已移至 device）。
    """
    from ..config import load_tes_config
    from ..data.normalizer import TESCondNormalizer
    from ..data.runtime_global_stats import build_runtime_global_stats, resolve_stats_paths
    from ..models.conditioning.seed_fluct_head import SeedFluctHead
    from ..models.factory import build_tes_model
    from ...utils.FieldNormalizer import FieldNormalizer

    mean_cfg = load_tes_config(mean_config)
    turb_cfg = load_tes_config(turb_config)

    # normalizer/field_norm 由 turb cfg 的 dataset stats（mean 與 turb 使用同一份）
    zarr_paths = [str(p) for p in turb_cfg.zarr_paths]
    sp = resolve_stats_paths(zarr_paths, turb_cfg.stats_paths)
    gstats = build_runtime_global_stats(sp, zarr_paths=zarr_paths,
                                        dataset_names=turb_cfg.dataset_names or None)
    normalizer = TESCondNormalizer.from_stats_dict(gstats)
    field_norm = FieldNormalizer.from_stats_dict(gstats).to(device)

    def _load(model, ckpt):
        sd = torch.load(ckpt, map_location=device, weights_only=False)["model"]
        missing, _ = model.load_state_dict(sd, strict=False)
        if missing:
            import logging
            logging.getLogger(__name__).warning("missing keys (%d): %s", len(missing), missing[:4])
        return model.eval()

    mean_model = _load(build_tes_model(mean_cfg.model).to(device), mean_ckpt)
    turb_model = _load(build_tes_model(turb_cfg.model).to(device), turb_ckpt)

    _ssd = torch.load(seed_ckpt, map_location=device, weights_only=False)["model"]
    seed_cond_dim = _ssd["film.weight"].shape[1] if "film.weight" in _ssd else 0
    turb_cond_dim = int(turb_cfg.model.cond_dim)
    if seed_cond_dim != turb_cond_dim:
        raise ValueError(
            f"seed_head cond_dim={seed_cond_dim} ≠ turb cond_dim={turb_cond_dim}；"
            "seed 與 turb 推論共用 cond_norm，維度必須一致（禁靜默退化）。"
        )
    seed_head = SeedFluctHead(physics_dim=9, static_dim=4, cond_dim=seed_cond_dim).to(device)
    seed_head.load_state_dict(_ssd)
    seed_head.eval()

    return DecomposeModels(
        mean_model=mean_model, turb_model=turb_model, seed_head=seed_head,
        normalizer=normalizer, field_norm=field_norm,
        mean_cond_dim=int(mean_cfg.model.cond_dim), turb_cond_dim=turb_cond_dim,
    )


def make_turb_stepper(turb_model, *, cond_norm: Tensor, fluid_mask: Tensor,
                      static: Tensor, mean_phys: Tensor, g_vec_norm: Optional[Tensor],
                      static_dim: int, physics_dim: int) -> Callable[[Tensor, Tensor], Tensor]:
    """把 stateful NCA turb_model 包成 run_turb_phase 的 (fluct, mean)→fluct 介面。

    首呼叫用傳入 fluct（= seed）init NCA state，之後 model 自帶 state 演化（忽略傳入 fluct）。
    錨點於首呼叫（state 建立時）解析一次並沿用全程——錨點語意是「本次
    rollout 起始 state 的 KE」，不得逐步重算。現行 turb config 的 ``ke_anchor_mode``
    為 ``off``，``forward_anchor_kwargs`` 因此回 ``{}``，本段為行為不變的防漏接措施
    （turb arch ``turb_perception_a`` 的 forward 並不收 ``ke_anchor``，故不可硬傳）。
    """
    holder: dict = {"state": None, "anchor_kwargs": {}}

    def _step(fluct: Tensor, mean: Tensor) -> Tensor:
        if holder["state"] is None:
            holder["state"] = turb_model.init_state(static, fluct)
            holder["anchor_kwargs"] = protocol.forward_anchor_kwargs(
                turb_model, holder["state"], fluid_mask
            )
        holder["state"] = turb_model(
            holder["state"], cond_norm, fluid_mask, 1,
            return_trajectory=False, g_vec_norm=g_vec_norm, mean_flow_norm=mean_phys,
            **holder["anchor_kwargs"],
        )
        return holder["state"][:, static_dim:static_dim + physics_dim].float()

    return _step


def run_decompose_pipeline(
    models: DecomposeModels,
    static: Tensor,
    g_vec: Tensor,
    p_vec: Tensor,
    *,
    g_vec_norm: Optional[Tensor] = None,
    mean_max_steps: int = 200,
    turb_steps: Optional[int] = None,
    eps: float = 0.05,
) -> tuple[MeanPhaseResult, Iterator[Tensor]]:
    """完整三模組部署 rollout：mean 收斂 → uIC seed（cond）→ turb 演化。

    與 `decompose_rollout` 不同處：mean 來自 **mean_model 預測**（非 GT），seed 來自
    **SeedFluctHead（cond 一致）**，turb 為 stateful NCA（make_turb_stepper 包裝）。

    Args:
        static: [B, static_dim, H, W]。
        g_vec / p_vec: [B,14] / [B,3] 原始條件（Re/幾何）。
        g_vec_norm: 選用 g_vec norm（部分 arch 需要）。
    Returns:
        (mean_result, composed_frames generator)。
    """
    device = static.device
    fluid_mask = (static[:, 0:1] > 0.0).float()
    z = torch.zeros(1, device=device)
    mean_cond = models.normalizer.build_cond(models.mean_cond_dim, g_vec, p_vec).to(device)
    turb_cond = models.normalizer.build_cond(models.turb_cond_dim, g_vec, p_vec).to(device)

    # Phase 1：mean_model 收斂（cold IC physics=0）。
    # ic_physics=None 時 protocol.initial_state 與 model.init_state 逐位元相同
    # （apply_ic_mask 只作用於非 None 的 IC），故此處直接呼叫 init_state 不構成協定分歧。
    init = models.mean_model.init_state(static, None)
    mean_result = run_mean_phase(
        models.mean_model, init, mean_cond, fluid_mask,
        static_dim=models.static_dim, physics_dim=models.physics_dim,
        g_vec_norm=g_vec_norm, max_steps=mean_max_steps, eps=eps,
    )

    # uIC seed（cond 一致，strict）
    seed = seed_from_head(models.seed_head, mean_result.mean_phys, static, fluid_mask, turb_cond)

    # Phase 2：turb stateful rollout → composed frames（Reynolds 一致組合）
    stepper = make_turb_stepper(
        models.turb_model, cond_norm=turb_cond, fluid_mask=fluid_mask,
        static=static, mean_phys=mean_result.mean_phys, g_vec_norm=g_vec_norm,
        static_dim=models.static_dim, physics_dim=models.physics_dim,
    )
    frames = run_turb_phase(
        stepper, mean_result.mean_phys, init_fluct=seed,
        n_steps=turb_steps, fluid_mask=fluid_mask, normalizer=models.field_norm,
    )
    return mean_result, frames


def decompose_rollout(
    mean_model,
    turb_step_fn: Callable[[Tensor, Tensor], Tensor],
    state: Tensor,
    cond_norm: Tensor,
    fluid_mask: Tensor,
    *,
    static_dim: int,
    physics_dim: int,
    g_vec_norm: Optional[Tensor] = None,
    mean_max_steps: int = 200,
    turb_steps: Optional[int] = None,
    eps: float = 0.05,
    normalizer: Optional[Any] = None,
) -> tuple[MeanPhaseResult, Iterator[Tensor]]:
    """兩階段 handoff 協調：Phase 1 收斂 → Phase 2 generator。

    Returns:
        (mean_result, composed_frames)：mean_result.converged=False 時
        仍回傳 generator（以未收斂 mean 起步），由呼叫端決定是否中止
        ——前端 demo 與評估流程的容錯策略不同，不在此硬性 raise。
    """
    mean_result = run_mean_phase(
        mean_model, state, cond_norm, fluid_mask,
        static_dim=static_dim, physics_dim=physics_dim,
        g_vec_norm=g_vec_norm, max_steps=mean_max_steps, eps=eps,
    )
    frames = run_turb_phase(
        turb_step_fn, mean_result.mean_phys,
        n_steps=turb_steps, fluid_mask=fluid_mask,
        normalizer=normalizer,
    )
    return mean_result, frames
