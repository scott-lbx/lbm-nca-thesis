"""training/steps/mean_step.py — Phase 0 平均流訓練步驟（TBPTT，mean_gate 課程）。

從 TESTrainer._mean_module_train_step 提取：
  - IC = parent_mean_renorm（warm-transfer task）
  - target = child mean_flow_post_conv（固定點均值流）
  - p_gt schedule 控制 GT-injection rollout 與 TBPTT preroll(k−G no_grad) + grad tail G
  - loss = MSE + SSIM + 選配 DivJ 散度軟懲罰 + GT-anchored 固定點正則（FP-reg）

LC decompose：TBPTT preroll(k−G no_grad) + grad tail G。
"""

from __future__ import annotations

import zlib

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.nn as nn

from ..rng_streams import stream
from ...inference import protocol
from ...loss.loss_terms_v2 import mean_flow_loss, phys_mass_divj_absmean
from ...loss.terms.perceptual import (
    mean_spectral_lowmode_loss,
    spectral_power_match_loss,
)
from ...utils import spatial_ops

if TYPE_CHECKING:
    from ..context import TrainerContext

_LOG_INTERVAL: int = 100

# [perf] per-step dt_*_ms 細分計時開關。為量測各段耗時需在 hot loop 插
# torch.cuda.synchronize()（排空 pipeline → 序列化 GPU），即使僅 _need_log 步
# 觸發，每 100 步仍排空 5 次。穩態關閉（False）；要 profiling 時手動設 True。
# 關閉純移除計時量測點，零訓練數值影響。
PROFILE_STEP_TIMING: bool = False

logger = logging.getLogger(__name__)


def _apply_ic_noise(
    ic: torch.Tensor,
    fluid_mask: torch.Tensor,
    sigma_frac: float,
    case_keys: "list[str] | None" = None,
) -> torch.Tensor:
    """`noise_edge` 消融臂：對初始條件加大幅高斯雜訊，只作用於流體區。

    ## 這一臂在隔離什麼

    2×2 析因的兩個屬性為「起點是否為真實流場（流形membership）」與
    「起點與監督目標是否鄰近（目標相關性）」。`no_edge` 把兩者同時移除，
    `far_edge`／`rand_edge` 只移除後者，**本臂是唯一只移除前者的操作**——
    底下仍是那張 parent 場，故目標相關性完整保留，被破壞的只有物理合法性。
    沒有這一臂，`no_edge` 的劣化可同時被兩個屬性解釋，析因不具區辨力。

    ## 為什麼振幅取逐案均方根的比例而非固定值

    正規化後各案的場強度仍有差異，固定振幅對弱場是毀滅性的、對強場是搔癢。
    取該案流體區的均方根再乘 `sigma_frac`，各案受到的相對擾動才一致。

    Args:
        ic: 初始條件，`[B, C, H, W]`，已 renorm 且已過 train noise。
        fluid_mask: 流體遮罩，`[B, 1, H, W]`，大於 0 為流體。
        sigma_frac: 雜訊標準差相對於該案流體區均方根的比例。

    ## 為什麼雜訊必須逐案固定而非每步重抽

    原本用 `torch.randn_like` 每次呼叫重抽，帶來兩個本臂獨有、
    不屬於被操作屬性的變因：

    1. 重抽使 `E[IC]` 恰等於真實 parent，模型可用空間平滑把雜訊平均掉，
       parent 資訊在期望意義上完整可還原 —— 量到的偏向「模型的去噪能力」
       而非「起點離開流形的程度」。
    2. 它是五臂中唯一 IC 帶隨機擾動的臂，額外獲得資料增強式正則，
       這個第二變因直接進到交互作用 I。

    改為以 `case_key` 衍生種子產生，使「離開流形」成為該筆資料的固定屬性。

    Args:
        ic: 初始條件，`[B, C, H, W]`，已 renorm 且已過 train noise。
        fluid_mask: 流體遮罩，`[B, 1, H, W]`，大於 0 為流體。
        sigma_frac: 雜訊標準差相對於該案流體區均方根的比例。
        case_keys: 各筆的 case key；提供時逐案以其衍生種子產生固定雜訊。
            None 時退回每步重抽（僅供無 case_key 的呼叫端，會在 log 留痕）。

    Returns:
        加噪後的初始條件，固體區維持原值（雜訊乘上遮罩後為 0）。
    """
    n_fluid = fluid_mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
    # 均方根對通道維一併平均，故分母另乘通道數。
    rms = (ic.pow(2) * fluid_mask).sum(dim=(1, 2, 3), keepdim=True)
    rms = (rms / (n_fluid * ic.shape[1])).sqrt()

    if case_keys is not None and len(case_keys) == ic.shape[0]:
        noise = torch.empty_like(ic)
        for i, ck in enumerate(case_keys):
            # zlib.crc32 是決定性的；Python 字串 hash 逐進程隨機化，不可用。
            g = torch.Generator(device="cpu")
            g.manual_seed(zlib.crc32(str(ck).encode("utf-8")))
            noise[i] = torch.randn(ic.shape[1:], generator=g).to(ic.device, ic.dtype)
    else:
        noise = torch.randn_like(ic)
    return ic + noise * (sigma_frac * rms) * fluid_mask


class MeanStep:
    """Phase 0 平均流訓練步驟（TBPTT，mean_gate 課程）。

    封裝 TESTrainer._mean_module_train_step 的完整邏輯：
    source-local → global renorm、TBPTT preroll / GT-injection rollout、
    grad tail forward、MSE + SSIM loss、DivJ 軟懲罰、FP-reg 正則、
    NaN guard、AMP scaler、gradient clip。

    Args:
        ctx: TESTrainer 共享狀態容器。
        data_prep: 資料前處理回呼 dict，需含以下 callable：
            - "renorm_source_to_global": (static, ic, traj, batch) -> tuple
            - "get_fluid_mask": (static) -> Tensor
            - "build_cond_norm": (g_vec, p_vec, delta_ctu, cumul_ctu) -> Tensor
            - "build_g_vec_norm": (g_vec) -> Tensor
            - "maybe_train_noise": (ic, fluid_mask) -> Tensor
            - "get_mean_p_gt": () -> float
    """

    def __init__(
        self,
        ctx: "TrainerContext",
        data_prep: dict[str, Callable[..., Any]],
        regularizer: Optional[Any] = None,
    ) -> None:
        self.ctx = ctx
        self._data_prep = data_prep
        self._regularizer = regularizer  # 提供 jacobian_reg_loss（收縮正則）；None=停用
        self._nan_consecutive: int = 0
        self._meanpool = None  # MeanPool（lazy create）；None=未建/停用。trainer vis 讀此 dump 真實訓練 pool
        self._pool_case_ids_mean: Optional[list] = (
            None  # sample() 回傳的 case_ids，writeback 時鏈式接力用
        )
        self._chain_map: Optional[dict[str, list[str]]] = (
            None  # lazy-build，見 _get_chain_map
        )

    # ------------------------------------------------------------------
    # 私有 helpers
    # ------------------------------------------------------------------

    def _get_chain_map(self, cfg: Any) -> dict[str, list[str]]:
        """lazy-build 全部 zarr_paths 的 {parent_case_id: [child_case_id]} 合併表。

        只建一次（跨 training step 快取於 self._chain_map）。同 zarr only（v1 範圍）。

        Args:
            cfg: TESTrainConfig，需含 zarr_paths。

        Returns:
            合併後的 parent→children 表；zarr_paths 間 case_id 視為互不相碰（各 zarr
            命名空間獨立），故直接 dict.update 合併不需去重判斷。
        """
        if self._chain_map is None:
            # v4 的 parent→children 直接由 topology 的 edge 提供
            #   （src_node → dst_node），不需 v3 的 cases.json 樹掃描。
            from ...data.graph_index import GraphIndex

            merged: dict[str, list[str]] = {}
            for zp in cfg.zarr_paths:
                g = GraphIndex.for_zarr(str(zp))
                for e in g.edges.values():
                    if e["src_node"] is None:
                        continue
                    merged.setdefault(e["src_node"], []).append(e["dst_node"])
            self._chain_map = merged
            logger.info("[A2 chain_map] %d parent node 有 child", len(merged))
        return self._chain_map

    @staticmethod
    def _pool_prob(
        step: int, max_steps: int, phase1_end: float, phase2_end: float
    ) -> float:
        p = float(step) / float(max(max_steps, 1))
        if p <= phase1_end:
            return 0.0
        if p >= phase2_end:
            return 1.0
        return (p - phase1_end) / max(phase2_end - phase1_end, 1e-6)

    # ------------------------------------------------------------------
    # 公開介面
    # ------------------------------------------------------------------

    def run(self, batch: dict) -> dict[str, float]:
        """執行一步 Phase 0 均值流梯度更新。

        Args:
            batch: DataLoader 回傳的 batch dict（需含 static /
                parent_mean_renorm / mean_flow_post_conv / g_vec / p_vec）。

        Returns:
            metrics dict（total / mse / ssim_term + 訓練動態與計時指標）。

        Raises:
            RuntimeError: 連續不穩定步超過 cfg.nan_patience 上限。
        """
        ctx = self.ctx
        cfg = ctx.cfg
        _need_log: bool = ctx.step % _LOG_INTERVAL == 0 or ctx.step < 100
        _t0: float = time.perf_counter()

        # ── Batch unpack + GPU transfer ──────────────────────────────
        static: torch.Tensor = batch["static"].to(ctx.device, non_blocking=True)
        target_mean: torch.Tensor = (
            batch["mean_flow_post_conv"].to(ctx.device, non_blocking=True).float()
        )
        if "parent_mean_renorm" in batch:
            ic_physics: torch.Tensor = (
                batch["parent_mean_renorm"].to(ctx.device, non_blocking=True).float()
            )
        else:
            # cold IC（mean_cold_ic=True）：trajectory[:,0] = 冷啟動初始狀態
            ic_physics = (
                batch["trajectory"][:, 0].to(ctx.device, non_blocking=True).float()
            )
        g_vec: torch.Tensor = batch["g_vec"].to(ctx.device, non_blocking=True)
        p_vec: torch.Tensor = batch["p_vec"].to(ctx.device, non_blocking=True)
        case_keys: Optional[list[str]] = batch.get(
            "case_key"
        )  # pool 跨案例鏈式接力用

        # M2：source-local → global renorm（mean 以 unsqueeze 方式一起轉換）
        static, ic_physics, _tgt5 = self._data_prep["renorm_source_to_global"](
            static, ic_physics, target_mean.unsqueeze(1), batch
        )
        target_mean = _tgt5.squeeze(1)

        # parent baseline（renorm 後、mask/noise 前）：Δ-focus 與 wake 的空間權重來源。
        # 一律取**覆寫前**的真實 parent。若各臂改用自己覆寫後的 IC 建權重，
        #   等於五臂最佳化五個不同的目標函數，效應量混入「損失權重圖改變」這個
        #   與被操作屬性完全共線的變因，無從分離。權重跨臂一致後，臂間唯一差異
        #   才真的只有送進模型的 IC。
        # 缺鍵即報錯，不 fallback：若靜默退回用覆寫後的 IC 建權重，一旦 collate
        # 的 _FIELD_KEYS 漏掉這個鍵，問題會難以察覺。
        _true_parent = batch.get("true_parent_mean_renorm")
        if _true_parent is None and "parent_mean_renorm" in batch:
            raise KeyError(
                "batch 缺 true_parent_mean_renorm，但有 parent_mean_renorm。"
                "檢查 graph_dataset._FIELD_KEYS 是否含該鍵（collate 只 stack 清單內的鍵）。"
            )
        if _true_parent is not None:
            _tp = _true_parent.to(ic_physics.device).float()
            if _tp.dim() == ic_physics.dim() + 1:
                _tp = _tp.squeeze(1)
            _st_tp, _tp, _ = self._data_prep["renorm_source_to_global"](
                static, _tp, target_mean.unsqueeze(1), batch
            )
            parent_baseline: torch.Tensor = _tp.clone()
        else:
            parent_baseline = ic_physics.clone()

        fluid_mask: torch.Tensor = self._data_prep["get_fluid_mask"](static)
        cond_norm: torch.Tensor = self._data_prep["build_cond_norm"](g_vec, p_vec)
        g_vec_norm: torch.Tensor = self._data_prep["build_g_vec_norm"](g_vec)

        # IC masking 已移交三態協定唯一實作（見下方 protocol.initial_state）。
        # train noise 本身乘 fluid_mask（trainer._maybe_train_noise），固體區恆為 0，
        # 故「先 mask 再加噪」與「加噪後由協定統一 mask」逐位元等價。
        ic_physics = self._data_prep["maybe_train_noise"](ic_physics, fluid_mask)

        # 消融臂的 IC 覆寫。邊集、監督目標與條件向量皆不動，故各臂與 use_edge 的
        # 唯一差異即其所隔離的因子。置於 renorm 與 mask 之後，確保送入模型的確為
        # 覆寫後的場，而非覆寫值再經 renorm 後的結果。
        #   no_edge     IC 改為零場 → 隔離「有無暖起點」。
        #   noise_edge  IC 改為 parent 場加大幅高斯雜訊 → **保留 parent 資訊**而只
        #               移除其流形membership，是分離「流形」與「目標相關性」兩個
        #               屬性的唯一操作（no_edge 兩者同時移除，far/rand 只移除後者）。
        # 這裡**只動 ic_physics**。parent_baseline 保持為真實 parent，
        #   使各臂共用同一組損失空間權重（見上方註解）。
        if cfg.mean_ablation_arm == "no_edge":
            ic_physics = torch.zeros_like(ic_physics)
        elif cfg.mean_ablation_arm in ("noise_edge", "farnoise_edge"):
            # farnoise_edge：src 已在資料層被換成遠配對（B✗），
            #   此處再加同一組雜訊（A✗）。它才是真正的 A✗B✗ 格 ——
            #   零場（no_edge）與「加噪 parent」是兩種不同的
            #   A✗ 實作，I = use − far − noise + no 因此無法歸因於交互作用。
            #   no_edge 保留其作為參考原點的角色。
            ic_physics = _apply_ic_noise(
                ic_physics, fluid_mask, cfg.mean_ablation_noise_sigma, case_keys
            )

        # stage 0 起始 state 走三態協定唯一實作（apply_ic_mask 直接存取，禁 fallback）。
        state: torch.Tensor = protocol.initial_state(
            ctx.model, cfg, static, ic_physics, fluid_mask
        )

        # ── RolloutSampler + Experience Pool ──
        _sd0: int = ctx.model._cfg.static_dim
        _pd0: int = ctx.model._cfg.physics_dim
        self._pool_slot_ids_mean = None
        if cfg.mean_rollout_sampler and getattr(self, "_rsampler", None) is None:
            from ..rollout_sampler import RolloutSampler

            self._rsampler = RolloutSampler.build(
                K_min=max(1, int(cfg.mean_rollout_k_min or 1)),
                K_max=int(cfg.mean_rollout_k_max or cfg.mean_rollout_k),
                total_steps=int(cfg.max_steps),
                mu_schedule=cfg.mean_sampler_mu_schedule,
                sigma_schedule=cfg.mean_sampler_sigma_schedule,
            )
        if cfg.mean_pool_enabled:
            if self._meanpool is None:
                from ..mean_pool import MeanPool

                self._meanpool = MeanPool(
                    cfg.mean_pool_size,
                    _sd0,
                    _pd0,
                    ctx.device,
                    cfg.mean_pool_seed_frac,
                    cfg.mean_pool_corrupt_frac,
                    cfg.mean_pool_corrupt_sigma,
                    damage_frac=cfg.mean_pool_damage_frac,
                    damage_mode=cfg.mean_pool_damage_mode,
                    damage_size_frac=cfg.mean_pool_damage_size_frac,
                    damage_fill=cfg.mean_pool_damage_fill,
                )
            # 用當前 batch 的 case 灌入 pool（乾淨 seed state），引入新案例。
            # ★深池：refill_interval>1 → warmup 內每步灌滿，warmup 後只每 N 步灌 → rolling slots
            #   存活久、累積深層 re-injection（治「每步灌 fresh 把深度重置成 ~1-2 層」淺池，攻自遞迴 i≥6 漂移）。
            _refill_iv = max(1, int(cfg.mean_pool_refill_interval))
            _pool_warmup = ctx.step < cfg.mean_pool_warmup_frac * cfg.max_steps
            if _pool_warmup or (ctx.step % _refill_iv == 0):
                # parent_baseline＝真實 parent（wake 權重）；ic_physics＝該臂覆寫後的 IC
                # （seed 重置用）。兩者必須分開傳，否則 seed 重置會把真實 parent
                # 當 IC 灌進消融臂，等於在 pool 路徑上還原被移除的暖起點。
                self._meanpool.refill(
                    state,
                    target_mean,
                    parent_baseline,
                    ic_physics,
                    g_vec,
                    p_vec,
                    case_ids=case_keys,
                )
            # 過 warmup 後以 pool sample 作 IC（cross-iteration 累積偏離 = pushforward）
            self._pool_case_ids_mean: Optional[list] = (
                None  # writeback 後鏈式接力用
            )
            if (
                ctx.step >= cfg.mean_pool_warmup_frac * cfg.max_steps
                and self._meanpool.ready
            ):
                _ps = self._meanpool.sample(static.shape[0], ctx.model)
                self._pool_slot_ids_mean = _ps.slot_ids
                self._pool_case_ids_mean = _ps.case_ids
                state = _ps.state
                target_mean = _ps.target
                parent_baseline = _ps.parent
                static = state[:, :_sd0]
                fluid_mask = self._data_prep["get_fluid_mask"](static)
                cond_norm = self._data_prep["build_cond_norm"](_ps.g_vec, _ps.p_vec)
                g_vec_norm = self._data_prep["build_g_vec_norm"](_ps.g_vec)

        # ── Pool IC path（Growing NCA：pool state 作為 IC，跳過 init_state rollout）──
        _pool_slot_ids: Optional[torch.Tensor] = None
        _n_pool: int = 0
        if ctx.pool is not None:
            _pp = self._pool_prob(
                ctx.step, cfg.max_steps, cfg.mean_gt_phase1_end, cfg.mean_gt_phase2_end
            )
            if _pp > 0.0 and stream("schedule").random() < _pp:
                B: int = static.shape[0]
                _n_gt = min(2, B)
                _n_pool = B - _n_gt
                if _n_pool > 0:
                    _pb = ctx.pool.sample(_n_pool, k=1)
                    # Build pool IC: state from pool (sd+pd+hd channels)
                    _pool_states = _pb.state.to(ctx.device).float()  # [n_pool, C, H, W]
                    # Fetch mean_flow target + g/p_vec per slot
                    _target_pool = []
                    _gv_pool = []
                    _pv_pool = []
                    _pool_ok = True
                    sd: int = ctx.model._cfg.static_dim
                    pd: int = ctx.model._cfg.physics_dim
                    for _i in range(_n_pool):
                        _ci = int(_pb.case_ids[_i])
                        _case = ctx.pool._fetch(_ci)
                        # 原為 _case.get("mean_flow")（缺鍵靜默回 None）。
                        # pool 的 case 必然帶 mean_flow；缺鍵代表 pool 內容有誤，應直接報錯。
                        _mf = _case["mean_flow"]
                        if _mf is None:
                            _pool_ok = False
                            break
                        _target_pool.append(_mf.to(ctx.device))
                        _gv_pool.append(_case["g_vec"].float().to(ctx.device))
                        _pv_pool.append(_case["p_vec"].float().to(ctx.device))
                    if _pool_ok:
                        # GT anchor: last _n_gt slots use DL IC (init_state already computed)
                        _pool_slot_ids = _pb.slot_ids
                        _pool_states_full = torch.cat(
                            [
                                _pool_states,
                                state[B - _n_gt :],  # DL-init states for GT anchor
                            ],
                            dim=0,
                        )  # [B, C, H, W]
                        _gv_dl = g_vec[B - _n_gt :]
                        _pv_dl = p_vec[B - _n_gt :]
                        _gv_comb = torch.cat([torch.stack(_gv_pool), _gv_dl], dim=0)
                        _pv_comb = torch.cat([torch.stack(_pv_pool), _pv_dl], dim=0)
                        _tgt_pool = torch.stack(
                            _target_pool
                            + [target_mean[B - _n_gt + _j] for _j in range(_n_gt)]
                        )  # [B, pd, H, W]
                        # Override training tensors
                        state = _pool_states_full
                        target_mean = _tgt_pool
                        fluid_mask = self._data_prep["get_fluid_mask"](state[:, :sd])
                        cond_norm = self._data_prep["build_cond_norm"](
                            _gv_comb, _pv_comb
                        )
                        g_vec_norm = self._data_prep["build_g_vec_norm"](_gv_comb)

        if PROFILE_STEP_TIMING and _need_log and ctx.device.type == "cuda":
            torch.cuda.synchronize()
        _t_gpu: float = time.perf_counter()

        # rollout k：RolloutSampler（常態漸進）優先，否則 uniform randint
        k_min = int(cfg.mean_rollout_k_min or 0)
        k_max = int(cfg.mean_rollout_k_max or 0)
        if cfg.mean_rollout_sampler and getattr(self, "_rsampler", None) is not None:
            rollout_k = self._rsampler.get_k(ctx.step)
        elif k_min > 0 and k_max >= k_min:
            rollout_k = stream("schedule").randint(k_min, k_max)
        else:
            rollout_k = int(cfg.mean_rollout_k)
        # grad_tail 作為 cap：grad_steps = min(rollout_k, grad_tail)（preroll = K−grad_tail）
        grad_tail: int = min(int(cfg.mean_grad_tail), rollout_k)
        preroll: int = max(0, rollout_k - grad_tail)
        p_gt: float = self._data_prep["get_mean_p_gt"]()

        # p_gt=1.0：所有 intermediate states 被 GT 蓋掉 → 只跑 grad_tail 步即可
        # 避免 24 步中 23 步純浪費 compute
        effective_k: int = grad_tail if p_gt >= 1.0 else rollout_k

        # KE ceiling 錨點由三態協定唯一實作產生（本 rollout 起始
        #   state 的 KE），而非「監督目標 target_mean 的 KE」——推論期根本拿不到，
        #   是三態算子分歧的源頭之一，且 CHAIN_PASS 的 ke 判準分母正是 GT KE（評測循環）。
        #   mode='off' 時回 None → arch bypass。
        _ke_anchor: Optional[torch.Tensor] = protocol.resolve_ke_anchor(
            cfg, state, fluid_mask
        )

        # ── Preroll (no_grad, all paths) ─────────────────────────────
        # p_gt∈(0,1)：TF-injected preroll 生成 mixed trajectory state；
        # ss_detach 已截斷時序梯度，k 步全展 autograd 為無效開銷 → 改 no_grad preroll。
        # p_gt==0 or p_gt>=1.0：純 TBPTT preroll（與舊邏輯相同）。
        ctx.optimizer.zero_grad(set_to_none=True)
        final_state: Optional[torch.Tensor] = None
        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            if preroll > 0:
                with torch.no_grad():
                    if 0.0 < p_gt < 1.0:
                        gt_expand_pre = (
                            target_mean.unsqueeze(1)
                            .expand(-1, preroll, -1, -1, -1)
                            .contiguous()
                        )
                        state = ctx.model(
                            state,
                            cond_norm,
                            fluid_mask,
                            preroll,
                            return_trajectory=False,
                            g_vec_norm=g_vec_norm,
                            tf_ratio=p_gt,
                            gt_traj=gt_expand_pre,
                            ss_detach=True,
                            ke_anchor=_ke_anchor,
                        )
                    else:
                        state = ctx.model(
                            state,
                            cond_norm,
                            fluid_mask,
                            preroll,
                            return_trajectory=False,
                            g_vec_norm=g_vec_norm,
                            ke_anchor=_ke_anchor,
                        )
                state = state.detach()
        if PROFILE_STEP_TIMING and _need_log and ctx.device.type == "cuda":
            torch.cuda.synchronize()
        _t_preroll: float = time.perf_counter()

        # ── Grad forward + loss ──────────────────────────────────────
        sd: int = ctx.model._cfg.static_dim
        pd: int = ctx.model._cfg.physics_dim
        # 變動收斂步數：grad 段步數 = rollout_k − preroll（k≤grad_tail 時 = rollout_k，全程 grad）。
        # 用 rollout_k（隨機 U[k_min,k_max]）而非固定 grad_tail → 學「收斂動作」非「固定步收斂」。
        grad_steps: int = max(1, rollout_k - preroll)

        _moment_w: float = float(cfg.mean_moment_weight)
        _chuber_w: float = float(cfg.mean_colored_huber_weight)
        _vgg_w: float = float(cfg.mean_vgg_weight)
        _vgg_moment_w: float = float(cfg.mean_vgg_moment_weight)
        _vgg_ot_w: float = float(cfg.mean_vgg_ot_weight)
        _vgg_on: bool = _vgg_w > 0.0 or _vgg_moment_w > 0.0 or _vgg_ot_w > 0.0
        _vgg_layers: tuple[int, ...] = tuple(
            int(s) for s in str(cfg.mean_vgg_layers).split(",") if s.strip()
        )
        _fn_c = getattr(ctx, "field_normalizer", None)
        # colored Huber 或 VGG 感知 loss 皆需 FieldColorizer（lazy 共用）
        if (
            (_chuber_w > 0.0 or _vgg_on)
            and _fn_c is not None
            and getattr(self, "_colorizer", None) is None
        ):
            from lbmnca.nca.utils.FieldColorizer import FieldColorizer

            # 必用 from_normalizer 注入 vis_range（colormap SSOT，禁 fallback）；
            # 舊的裸 FieldColorizer() 在 strict vis_range guard 後會 raise（getVelColor）。
            self._colorizer = FieldColorizer.from_normalizer(_fn_c).to(state.device)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            # wake 加權：|target − parent| soft mask（聚焦 child 偏轉），colored Huber 與 stay 共用
            with torch.no_grad():
                _wmag = (target_mean - parent_baseline).abs().mean(dim=1, keepdim=True)
                _wm = fluid_mask
                _wmn = (_wmag * _wm).sum() / _wm.sum().clamp_min(1.0)
                _wake = ((_wmag / _wmn.clamp_min(1e-8)).clamp(0.0, 10.0) ** 1.5) * _wm

            # 下列張量先給 0 預設，未啟用該項時記 0；overflow／fp_reg／rel_l2／stay
            # 皆有對應的 per-term 欄位。
            loss_overflow: torch.Tensor = torch.zeros((), device=fluid_mask.device)
            loss_fp_reg: torch.Tensor = torch.zeros((), device=fluid_mask.device)
            # rel_l2（canonical 權重 1.5）在 `_primary_field_loss` 內累加，而該函式
            #   對 k1 與 stay 各呼叫一次，故此處記的是**兩次呼叫的總和**——與
            #   它對 total 的實際貢獻同尺度。
            loss_rel_l2: torch.Tensor = torch.zeros((), device=fluid_mask.device)
            _rel_w: float = float(cfg.mean_rel_l2_weight)

            def _primary_field_loss(
                pred_p: torch.Tensor,
                with_vgg: bool = True,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                """moment（mphy MSE）次要 + colored Huber 主項 + 選配 VGG 感知 + rel_l2；k1 與 stay 共用。

                VGG 項（DyNCA 機制）：denorm→vphy→colored RGB→凍結 VGG 深層特徵統計
                （structure spatial L1 + moment channel-wise mean/std），破解 pixel-MSE
                的均勻場模糊吸引子。全程 fp32（函式內 autocast disabled）。
                """
                l_m, _ = mean_flow_loss(
                    pred_p,
                    target_mean,
                    fluid_mask,
                    ssim_weight=0.0,
                    per_channel_norm=cfg.mean_per_channel_norm,
                )
                l_p = _moment_w * l_m
                l_ch = pred_p.new_zeros(())
                if _chuber_w > 0.0 and _fn_c is not None:
                    from ...loss.terms.perceptual import colored_residual_huber_loss

                    l_ch = _chuber_w * colored_residual_huber_loss(
                        pred_p,
                        target_mean,
                        fluid_mask,
                        _fn_c,
                        self._colorizer,
                        beta=float(cfg.mean_colored_huber_beta),
                        wake_weight=_wake,
                    )
                    l_p = l_p + l_ch
                l_vgg = pred_p.new_zeros(())
                if with_vgg and _vgg_on and _fn_c is not None:
                    from ...loss.terms.vgg_perceptual import vgg_perceptual_loss

                    l_vgg = vgg_perceptual_loss(
                        pred_p,
                        target_mean,
                        fluid_mask,
                        _fn_c,
                        self._colorizer,
                        structure_weight=_vgg_w,
                        moment_weight=_vgg_moment_w,
                        ot_weight=_vgg_ot_w,
                        layers=_vgg_layers,
                        max_size=int(cfg.mean_vgg_max_size),
                    )
                    l_p = l_p + l_vgg
                if _rel_w > 0.0:
                    nonlocal loss_rel_l2
                    _m = fluid_mask.expand_as(pred_p)
                    _diff = (pred_p.float() - target_mean) * _m
                    _tgt = target_mean * _m
                    _rel = (
                        _diff.pow(2).sum().clamp_min(1e-12).sqrt()
                        / _tgt.pow(2).sum().clamp_min(1e-12).sqrt()
                    )
                    _rel_term = _rel_w * _rel
                    loss_rel_l2 = loss_rel_l2 + _rel_term.detach()
                    l_p = l_p + _rel_term
                return l_p, l_ch, l_vgg

            # ── k1：變動步數收斂 ──
            final_state = ctx.model(
                state,
                cond_norm,
                fluid_mask,
                grad_steps,
                return_trajectory=False,
                g_vec_norm=g_vec_norm,
                ke_anchor=_ke_anchor,
            )
            pred_phys: torch.Tensor = final_state[:, sd : sd + pd]  # [B, 9, H, W]
            # SSIM metric 用：k1 forward 後立即存 detached fp32 clone（避 stay in-place /
            # backward graph free / autocast 殘留導致 nan）。[perf] 僅 _need_log 步需要
            # （下方 SSIM metric 區整段 if _need_log）→ 非 log 步免每步 [B,9,H,W] fp32 clone。
            _pred_metric: Optional[torch.Tensor] = None
            _tgt_metric: Optional[torch.Tensor] = None
            if _need_log:
                _pred_metric = pred_phys.detach().float().clone()
                _tgt_metric = target_mean.detach().float().clone()

            # [perf] 計時：primary loss（含 VGG）；_need_log 時 cuda.sync 量測
            def _sync():
                if _need_log and ctx.device.type == "cuda":
                    torch.cuda.synchronize()

            _sync()
            _tt_a = time.perf_counter()
            loss_primary, loss_chuber, loss_vgg = _primary_field_loss(pred_phys)
            _sync()
            _dt_primary_ms = round((time.perf_counter() - _tt_a) * 1000, 1)
            # raw jx/jy SSIM（mean_ssim_weight；v11=0）
            _, loss_ssim_term = mean_flow_loss(
                pred_phys,
                target_mean,
                fluid_mask,
                ssim_weight=cfg.mean_ssim_weight,
                per_channel_norm=cfg.mean_per_channel_norm,
            )
            loss_mse, _ = mean_flow_loss(
                pred_phys,
                target_mean,
                fluid_mask,
                ssim_weight=0.0,
                per_channel_norm=cfg.mean_per_channel_norm,
            )
            loss: torch.Tensor = loss_primary + loss_ssim_term

            # ── stay：繼續變動 extra 步，仍須 match target（學穩定固定點，非固定步收斂）──
            loss_stay: torch.Tensor = pred_phys.new_zeros(())
            _dt_stay_ms = 0.0
            _stay_w: float = float(cfg.mean_stay_weight)
            if _stay_w > 0.0:
                _sync()
                _tt_s = time.perf_counter()
                _ek_min = int(cfg.mean_stay_extra_k_min)
                _ek_max = int(cfg.mean_stay_extra_k_max)
                _extra = stream("rollout").randint(_ek_min, max(_ek_min, _ek_max))
                stay_state = ctx.model(
                    final_state,
                    cond_norm,
                    fluid_mask,
                    _extra,
                    return_trajectory=False,
                    g_vec_norm=g_vec_norm,
                    ke_anchor=_ke_anchor,
                )
                pred_stay = stay_state[:, sd : sd + pd]
                _l_stay, _, _ = _primary_field_loss(
                    pred_stay, with_vgg=False
                )  # stay 跳 VGG（省時）
                loss_stay = _stay_w * _l_stay
                loss = loss + loss_stay
                _sync()
                _dt_stay_ms = round((time.perf_counter() - _tt_s) * 1000, 1)

            # 視覺 fixed-range 結構 SSIM（vel_mag / vorticity；對齊視覺面板值域）
            # jx-raw SSIM 不足以督導尾流結構 → offline rollout 過平滑；
            # vorticity SSIM 直接督導渦街結構，是最強的抗平滑信號。
            loss_vel_ssim: torch.Tensor = pred_phys.new_zeros(())
            loss_vort_ssim: torch.Tensor = pred_phys.new_zeros(())
            _vel_ssim_w: float = float(cfg.mean_vel_ssim_weight)
            _vort_ssim_w: float = float(cfg.mean_vort_ssim_weight)
            _fn = getattr(ctx, "field_normalizer", None)
            if _fn is not None and (_vel_ssim_w > 0.0 or _vort_ssim_w > 0.0):
                from ...loss.terms.perceptual import (
                    struct_vel_ssim_loss,
                    struct_vort_ssim_loss,
                )

                _pk = pred_phys.unsqueeze(1)  # [B, 1, 9, H, W]
                _tk = target_mean.unsqueeze(1)
                if _vel_ssim_w > 0.0:
                    loss_vel_ssim = _vel_ssim_w * struct_vel_ssim_loss(
                        _pk,
                        _tk,
                        fluid_mask,
                        _fn,
                        range_from_stats=cfg.ssim_range_from_stats,
                    )
                    loss = loss + loss_vel_ssim
                if _vort_ssim_w > 0.0:
                    loss_vort_ssim = _vort_ssim_w * struct_vort_ssim_loss(
                        _pk,
                        _tk,
                        fluid_mask,
                        _fn,
                        range_from_stats=cfg.ssim_range_from_stats,
                    )
                    loss = loss + loss_vort_ssim

            # 不會飽和的銳利度／渦度目標（見 config/_legacy.py 的說明）
            #   SSIM 在任何值域下對「模糊」都幾乎全盲，而部署常見的缺陷正是
            #   模糊——速度梯度能量偏低。enstrophy 與 velgrad 這兩項直接罰
            #   同一個模糊，且皆非逐點飽和型。
            # 水平帶。eval 有 `HorizontalBand_FFT` 這一軸，但 mean 路徑
            #   從來沒有對應的訓練項（`mean_column_banding_weight` 罰的是垂直條紋）。
            loss_hband: torch.Tensor = pred_phys.new_zeros(())
            _hband_w: float = float(cfg.mean_hband_weight)
            if _fn is not None and _hband_w > 0.0:
                from ...loss.terms.perceptual import horiz_band_fft_loss

                loss_hband = _hband_w * horiz_band_fft_loss(
                    pred_phys.unsqueeze(1), target_mean.unsqueeze(1), fluid_mask, _fn
                )
                loss = loss + loss_hband

            loss_enstrophy: torch.Tensor = pred_phys.new_zeros(())
            loss_velgrad: torch.Tensor = pred_phys.new_zeros(())
            _enst_w: float = float(cfg.mean_enstrophy_weight)
            _vgrad_w: float = float(cfg.mean_velgrad_weight)
            if _fn is not None and (_enst_w > 0.0 or _vgrad_w > 0.0):
                from ...loss.terms.struct_velocity import (
                    enstrophy_match_loss,
                    grad_energy_ratio_loss,
                )

                _pk3 = pred_phys.unsqueeze(1)  # [B, 1, 9, H, W]
                _tk3 = target_mean.unsqueeze(1)
                if _enst_w > 0.0:
                    loss_enstrophy = _enst_w * enstrophy_match_loss(
                        _pk3, _tk3, fluid_mask, _fn
                    )
                    loss = loss + loss_enstrophy
                if _vgrad_w > 0.0:
                    # ★用有界的能量比，不用 obs_vel_gradient_mae：後者是逐點 MAE，
                    #   在 raw lattice 單位下數值極小、需配很大的權重才有作用，
                    #   但本身完全無界——一旦 pred 幅值過量，配上大權重會讓 loss 發散。
                    loss_velgrad = _vgrad_w * grad_energy_ratio_loss(
                        _pk3, _tk3, fluid_mask, _fn
                    )
                    loss = loss + loss_velgrad

            # ★Column-banding 抗條紋（部署稽核缺陷）：罰 pred 相對 GT 之垂直高頻(週期≤40px)
            #   column 條紋超出量（ρ/jx/jy）。moment MSE 因 ρ 主導遮蔽此缺陷 → 需專項督導。
            loss_band: torch.Tensor = pred_phys.new_zeros(())
            _band_w: float = float(cfg.mean_column_banding_weight)
            if _band_w > 0.0:
                from ...loss.terms.struct_banding import struct_column_banding_loss

                loss_band = _band_w * struct_column_banding_loss(
                    pred_phys.unsqueeze(1), target_mean.unsqueeze(1), fluid_mask
                )
                loss = loss + loss_band

            # 空間局域方向性 band loss（NCA 文獻啟發：自由流區罰 x-梯度主導超 GT）
            _dirb_w: float = float(cfg.mean_dirband_weight)
            if _dirb_w > 0.0:
                from ...loss.terms.struct_banding import struct_directional_band_loss

                loss = loss + _dirb_w * struct_directional_band_loss(
                    pred_phys.unsqueeze(1), target_mean.unsqueeze(1), fluid_mask
                )

            # Δ-focus 加權 MSE：用 |target − parent| 當 soft 權重，聚焦 child 偏轉（尾流）區
            # warm-transfer 下 parent≈child，整場 MSE 容易被共用背景主導，
            # child 的 Δ 小信號被淹沒；Δ-focus 直接放大尾流梯度區的權重。
            loss_delta: torch.Tensor = pred_phys.new_zeros(())
            _df_w: float = float(cfg.mean_delta_focus_weight)
            if _df_w > 0.0 and _n_pool == 0:
                _df_gamma: float = float(cfg.mean_delta_focus_gamma)
                with torch.no_grad():
                    _dmag = (
                        (target_mean - parent_baseline).abs().mean(dim=1, keepdim=True)
                    )  # [B, 1, H, W] child Δ 幅度
                    _m1 = fluid_mask
                    _dmean = (_dmag * _m1).sum() / _m1.sum().clamp_min(1.0)
                    _w = (_dmag / _dmean.clamp_min(1e-8)).clamp(0.0, 10.0) ** _df_gamma
                    _w = _w * _m1
                _diff_d = pred_phys.float() - target_mean
                # ÷ 固定尺度（N_fluid · C），非 ÷Σw —— 後者把絕對梯度稀釋成相對分布，
                # 對 systematic under-shoot 無糾正力
                _n_fc = _m1.sum().clamp_min(1.0) * float(pred_phys.shape[1])
                loss_delta = _df_w * ((_w * _diff_d.pow(2)).sum() / _n_fc)
                loss = loss + loss_delta

            # Overflow loss（DyNCA §2.2：全通道含 hidden，soft L1 超界量）
            overflow_w: float = float(cfg.mean_overflow_weight)
            if overflow_w > 0.0:
                # 存下加權後的值供下方 ret dict 記錄。
                loss_overflow = (
                    overflow_w
                    * (final_state - final_state.clamp(-1.0, 1.0)).abs().mean()
                )
                loss = loss + loss_overflow

            # DivJ 散度軟懲罰（λ_divj）
            # ★此處需要 ctx.field_normalizer（FieldNormalizer，有 denorm_mphy），
            #   非 ctx.normalizer（TESCondNormalizer，僅 cond z-score）；誤用後者
            #   會在 mean_divj_weight>0 時拋出 AttributeError。
            divj_w: float = float(cfg.mean_divj_weight)
            if divj_w > 0.0:
                loss = loss + divj_w * phys_mass_divj_absmean(
                    pred_phys.unsqueeze(1), fluid_mask, ctx.field_normalizer
                )

            # FP-reg：GT-anchored fixed-point regularizer（λ_fp）
            # ★mean_fp_rollout_k>1：多步 identity——從 GT IC roll K 步仍須≈GT，直罰
            #   inference-horizon 漂移 + rollout banding 累積（1 步版不足；gate identity_drift 用 ~30 步）。
            fp_reg_w: float = float(cfg.mean_fp_reg_weight)
            if fp_reg_w > 0.0:
                _static_det = final_state[:, :sd].detach()
                fp_ic = ctx.model.init_state(_static_det, target_mean)
                fp_out = ctx.model(
                    fp_ic,
                    cond_norm,
                    fluid_mask,
                    max(1, int(cfg.mean_fp_rollout_k)),
                    return_trajectory=False,
                    g_vec_norm=g_vec_norm,
                    ke_anchor=protocol.resolve_ke_anchor(cfg, fp_ic, fluid_mask),
                )
                fp_phys: torch.Tensor = fp_out[:, sd : sd + pd]
                n_fluid = fluid_mask.sum().clamp_min(1.0)
                l_fp = ((fp_phys - target_mean) ** 2 * fluid_mask).sum() / n_fluid
                # fp_reg 是名目權重最大的一項（canonical 5.0）。存下加權值供 ret dict。
                loss_fp_reg = fp_reg_w * l_fp
                loss = loss + loss_fp_reg

            # ★self-recurse invariance regularizer（λ_inv）：root-cause 分析證算子吸引子≠GT——
            #   re-inject 模型自身收斂輸出當 IC → 漂到 ~7× GT 髒吸引子（fp_res 卡 0.02 limit-cycle）。
            #   此項 IC=**模型自身輸出**（非 GT，覆蓋 warm-transfer re-injection 流形）→ roll K → 罰漂回 GT，
            #   令 GT 成穩定不動點。與 fp-reg 並存（GT-anchor 防遺忘 + self-anchor 攻復發）。
            loss_inv: torch.Tensor = pred_phys.new_zeros(())
            inv_reg_w: float = float(cfg.mean_inv_reg_weight)
            if inv_reg_w > 0.0:
                _static_inv = final_state[:, :sd].detach()
                _inv_k = max(1, int(cfg.mean_fp_rollout_k))
                with torch.no_grad():
                    # out1 = 模型自身收斂輸出（masked，鏡射部署 apply_ic_mask）
                    _inv_phys = pred_phys.detach() * fluid_mask
                    # depth：隨機 0..max 額外 no_grad re-inject，造深層複利態（攻 i≥6 才顯著的複利）
                    for _ in range(
                        stream("rollout").randint(0, int(cfg.mean_inv_depth_max))
                    ):
                        _s = ctx.model.init_state(_static_inv, _inv_phys)
                        _s = ctx.model(
                            _s,
                            cond_norm,
                            fluid_mask,
                            _inv_k,
                            return_trajectory=False,
                            g_vec_norm=g_vec_norm,
                            ke_anchor=protocol.resolve_ke_anchor(cfg, _s, fluid_mask),
                        )
                        _inv_phys = _s[:, sd : sd + pd] * fluid_mask
                inv_ic = ctx.model.init_state(_static_inv, _inv_phys.detach())
                inv_out = ctx.model(
                    inv_ic,
                    cond_norm,
                    fluid_mask,
                    _inv_k,
                    return_trajectory=False,
                    g_vec_norm=g_vec_norm,
                    ke_anchor=protocol.resolve_ke_anchor(cfg, inv_ic, fluid_mask),
                )
                inv_phys: torch.Tensor = inv_out[:, sd : sd + pd]
                _n_fi = fluid_mask.sum().clamp_min(1.0)
                loss_inv = inv_reg_w * (
                    ((inv_phys - target_mean) ** 2 * fluid_mask).sum() / _n_fi
                )
                loss = loss + loss_inv

            # Spectral 低-mode loss（FNO 風：複數低頻匹配 + 能量加權優先重要 mode）。
            # 破均勻吸引子（均勻場低頻係數≈0→loss 高）、釘大尺度結構含相位（攻病灶1）。
            loss_spec: torch.Tensor = pred_phys.new_zeros(())
            _spec_w: float = float(cfg.mean_spectral_weight)
            if _spec_w > 0.0:
                loss_spec = _spec_w * mean_spectral_lowmode_loss(
                    pred_phys,
                    target_mean,
                    fluid_mask,
                    modes=int(cfg.mean_spectral_modes),
                )
                loss = loss + loss_spec

            # ── 全譜 radial power match（罰高k尾 excess；補 lowmode 未罰的 grain/band/tv 譜根）──
            if float(cfg.mean_specmatch_weight) > 0.0:
                loss = loss + float(
                    cfg.mean_specmatch_weight
                ) * spectral_power_match_loss(
                    pred_phys.unsqueeze(1),
                    target_mean.unsqueeze(1),
                    fluid_mask,
                    ctx.field_normalizer,
                    k_hi_frac=float(cfg.mean_specmatch_khi),
                )

            # Jacobian 收縮正則（SSOT §4.1）：在固定點 final_state 上罰
            # ‖F(x*+ε)−F(x*)‖/‖ε‖ > γ → 算子收縮（譜半徑<1）→ 數學壓制長 rollout 發散（攻病灶2）。
            # detach final_state：收縮為 F 在 x* 的局部性質，梯度只塑 F（不回傳 rollout 歷史，省顯存）。
            loss_jac: torch.Tensor = pred_phys.new_zeros(())
            # 起始比例 + ramp：前半先學結構（破均勻、長尖銳尾流），後半才漸入 jac_reg 穩定收斂軌跡。
            # 早期硬壓收縮會把尖銳結構推平（朝均勻）→ 故延後 + 漸入。
            _jac_w0: float = float(cfg.jac_reg_weight)
            _frac: float = float(ctx.step) / float(max(1, cfg.max_steps))
            _start: float = float(cfg.mean_jac_start_frac)
            _ramp: float = max(1e-6, float(cfg.mean_jac_ramp_frac))
            _jac_scale: float = max(0.0, min(1.0, (_frac - _start) / _ramp))
            _jac_w: float = _jac_w0 * _jac_scale
            _jac_iv: int = int(cfg.mean_jac_interval)
            # 每 N 步才算 jac_reg（+2 forward/step 為主要開銷；收縮慢變，間隔算即可省 2×forward）
            if (
                _jac_w > 0.0
                and self._regularizer is not None
                and (ctx.step % max(1, _jac_iv) == 0)
            ):
                l_jac = self._regularizer["jacobian_reg_loss"](
                    final_state.detach(), cond_norm, fluid_mask, g_vec_norm, None, None
                )
                loss_jac = _jac_w * l_jac
                loss = loss + loss_jac

        if PROFILE_STEP_TIMING and _need_log and ctx.device.type == "cuda":
            torch.cuda.synchronize()
        _t_grad_fwd: float = time.perf_counter()

        # ── NaN guard ────────────────────────────────────────────────
        loss_val: float = loss.detach().float().item()
        is_bad: bool = (not math.isfinite(loss_val)) or (
            loss_val > cfg.loss_spike_threshold
        )
        if is_bad:
            self._nan_consecutive += 1
            logger.warning(
                "[step %d] mean_module unstable loss=%.4g  consecutive=%d/%d",
                ctx.step,
                loss_val,
                self._nan_consecutive,
                cfg.nan_patience,
            )
            if self._nan_consecutive >= cfg.nan_patience:
                raise RuntimeError(
                    f"NaN early stop at step {ctx.step}: "
                    f"{self._nan_consecutive} consecutive unstable steps"
                )
            ctx.step += 1
            return {"total": float("nan"), "k": rollout_k, "loss_version": "mean"}

        # ── Backward + optimizer step ────────────────────────────────
        ctx.scaler.scale(loss).backward()
        ctx.scaler.unscale_(ctx.optimizer)
        # Mordvintsev NCA 招牌：per-param 梯度正規化（每參數 grad /= ‖grad‖）→ 每步等幅小更新、
        # 不激進、漸進形成結構（訓練動態層抗平滑，與 VGG 特徵 loss 互補）。在 Adam 前作用。
        if cfg.mean_grad_normalize:
            with torch.no_grad():
                for _p in ctx.model.parameters():
                    if _p.grad is not None:
                        _p.grad.div_(_p.grad.norm() + 1e-8)
        grad_norm_tensor = nn.utils.clip_grad_norm_(
            ctx.model.parameters(), max_norm=cfg.grad_clip_norm
        )
        grad_norm: Optional[float] = grad_norm_tensor.item() if _need_log else None
        if PROFILE_STEP_TIMING and _need_log and ctx.device.type == "cuda":
            torch.cuda.synchronize()
        _t_bwd: float = time.perf_counter()

        # AMP scaler 在初期 scale 校準（或 inf/nan grad）時會 skip optimizer.step；
        # 此時不可推進 scheduler（否則 PyTorch 警告「scheduler.step before optimizer.step」
        # 且會多跳一個 LR 值）。以 scale 是否下降判定 skip。
        _scale_before: float = ctx.scaler.get_scale() if cfg.use_amp else 0.0
        ctx.scaler.step(ctx.optimizer)
        ctx.scaler.update()
        _opt_skipped: bool = cfg.use_amp and ctx.scaler.get_scale() < _scale_before
        if not _opt_skipped:
            ctx.scheduler.step()
        if PROFILE_STEP_TIMING and _need_log and ctx.device.type == "cuda":
            torch.cuda.synchronize()
        _t_opt: float = time.perf_counter()

        # Pool writeback（pool IC 路徑才寫回；只寫 pool slots，不含 GT anchor）
        if _pool_slot_ids is not None and _n_pool > 0:
            ctx.pool.writeback(
                _pool_slot_ids, final_state[:_n_pool].detach(), advanced_k=1
            )

        # MeanPool writeback（v13）：rollout 後最終 state 寫回，累積偏離分布
        if self._pool_slot_ids_mean is not None and self._meanpool is not None:
            self._meanpool.writeback(self._pool_slot_ids_mean, final_state.detach())
            # 跨案例鏈式接力。0=停用零行為變更。
            if cfg.mean_pool_chain_frac > 0.0 and self._pool_case_ids_mean is not None:
                self._meanpool.chain_writeback(
                    self._pool_case_ids_mean,
                    final_state,
                    ctx.model,
                    self._get_chain_map(cfg),
                    cfg.mean_pool_chain_frac,
                )

        self._nan_consecutive = 0

        # ── 原始 SSIM metric（無條件計算，與 loss 權重無關；供 loss_breakdown 監控）──
        # SSIM 是品質 metric 不是 loss term —— 不管 mean_ssim_weight 是否為 0 都要記錄。
        _jx_ssim_m: float = float("nan")
        _vel_ssim_m: float = float("nan")
        _vort_ssim_m: float = float("nan")
        if _need_log:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                from ...loss.terms.perceptual import (
                    _ssim_loss_2d,
                    struct_vel_ssim_loss,
                    struct_vort_ssim_loss,
                )

                _jxp = ((_pred_metric[:, 3:4] + 1.0) * 0.5).clamp(0.0, 1.0)
                _jxt = ((_tgt_metric[:, 3:4] + 1.0) * 0.5).clamp(0.0, 1.0)
                _jx_ssim_m = 1.0 - float(_ssim_loss_2d(_jxp, _jxt))
                _fn_m = getattr(ctx, "field_normalizer", None)
                if _fn_m is not None:
                    _pk_m = _pred_metric.unsqueeze(1)
                    _tk_m = _tgt_metric.unsqueeze(1)
                    _vel_ssim_m = 1.0 - float(
                        struct_vel_ssim_loss(
                            _pk_m,
                            _tk_m,
                            fluid_mask,
                            _fn_m,
                            range_from_stats=cfg.ssim_range_from_stats,
                        )
                    )
                    _vort_ssim_m = 1.0 - float(
                        struct_vort_ssim_loss(
                            _pk_m,
                            _tk_m,
                            fluid_mask,
                            _fn_m,
                            range_from_stats=cfg.ssim_range_from_stats,
                        )
                    )

        scaler_scale: Optional[float] = (
            ctx.scaler.get_scale() if (cfg.use_amp and _need_log) else None
        )
        gpu_mem_mb: Optional[float] = (
            torch.cuda.memory_allocated(ctx.device) / 1e6
            if (_need_log and ctx.device.type == "cuda")
            else None
        )
        ctx.step += 1
        # `total` 是**至少五個隨機因子的混合量**——rollout_k 隨機、chain 分支以
        # 0.5 機率抽樣、pool/damage 抽樣、stochastic update、AMP 尺度。
        # 因此它**不可**作為收斂判讀、早停監控或跨 run 比較的標量：兩個 run 的
        # total 差異可能完全來自抽樣組成而非模型品質。要比就比 per-term 欄位或
        # 事後的 per-case 評估（em2 等）。早停請改監控 `mean_rel_l2` 之類的
        # val 指標（見 `training/schedule.py::select_early_stop_monitor`）。
        ret: dict = {
            "total": loss_val,
            "mse": loss_mse.detach().item(),
            "ssim_term": loss_ssim_term.detach().item(),
            "vel_ssim": loss_vel_ssim.detach().item(),
            "vort_ssim": loss_vort_ssim.detach().item(),
            "enstrophy": loss_enstrophy.detach().item(),  # 渦量總能量缺額（不飽和）
            "velgrad": loss_velgrad.detach().item(),  # |∇u| 逐點偏差＝抗過度平滑
            "hband": loss_hband.detach().item(),  # kx=0 超額能量（水平帶）
            "delta_focus": loss_delta.detach().item(),
            "chuber": loss_chuber.detach().item(),
            "vgg": loss_vgg.detach().item(),
            "spec": loss_spec.detach().item(),
            "band": loss_band.detach().item(),  # ★per-term 診斷：column_banding 罰（band 是否被驅動下降）
            "inv": loss_inv.detach().item(),
            # `jac_reg` 只在計算閘門成立的 step 才是真值，而記錄點的
            # ctx.step 奇偶與該閘門互斥 → 此欄在整個訓練期恆為 0，對「該項有無作用」
            # 是零資訊。下方四個 `jac_*` 欄與 step 奇偶無關，才是可信的觀測面。
            "jac_reg": loss_jac.detach().item(),
            **(
                self._regularizer.jac_observability()
                if hasattr(self._regularizer, "jac_observability")
                else {}
            ),
            # `overflow`、`fp_reg`、`stay`、`rel_l2`（canonical 權重 1.5）皆為加權
            #   後的值、與 total 同尺度，已記錄為獨立的 per-term 欄位。
            #   `_chain_stage_loss` 那條路徑的 per-term 仍全數未記錄，是具名例外，
            #   白名單是 tests/nca/tes/test_loss_breakdown_ledger.py 的 LEDGER
            #   （現行名稱與路徑由該檔的
            #   test_ledger_name_referenced_in_source_actually_exists 把關）。
            #   既有 canonical run 的 metrics.json 無這幾欄，無法回溯拆解。
            "overflow": loss_overflow.detach().item(),
            "fp_reg": loss_fp_reg.detach().item(),
            "rel_l2": loss_rel_l2.detach().item(),
            "stay": loss_stay.detach().item(),
            "jx_ssim": _jx_ssim_m,
            "vel_ssim_metric": _vel_ssim_m,
            "vort_ssim_metric": _vort_ssim_m,
            "k": grad_steps,
            "loss_version": "mean",
            "grad_norm": grad_norm,
            "scaler_scale": scaler_scale,
            "gpu_mem_mb": gpu_mem_mb,
        }
        if _need_log:
            # 永遠量測（自帶 sync）：primary loss（含 VGG）+ stay 段耗時，定位效率瓶頸
            ret["dt_loss_primary_ms"] = _dt_primary_ms
            ret["dt_stay_ms"] = _dt_stay_ms
        if PROFILE_STEP_TIMING and _need_log:
            # 註：未開 PROFILE 時上面 synchronize 全略過 → 各 _t_* 時間戳非同步邊界，
            # dt 細分無意義，故一併略過發送（dt_*_ms 屬 profiling extras，非訓練指標）。
            ret["dt_transfer_ms"] = round((_t_gpu - _t0) * 1000, 1)
            ret["dt_preroll_ms"] = round((_t_preroll - _t_gpu) * 1000, 1)
            ret["dt_grad_fwd_ms"] = round((_t_grad_fwd - _t_preroll) * 1000, 1)
            ret["dt_backward_ms"] = round((_t_bwd - _t_grad_fwd) * 1000, 1)
            ret["dt_optim_ms"] = round((_t_opt - _t_bwd) * 1000, 1)
            ret["dt_step_ms"] = round((_t_opt - _t0) * 1000, 1)
        return ret

    # ------------------------------------------------------------------
    # multi-edge pushforward（缺陷 b：跨幾何 warm 累積）
    # ------------------------------------------------------------------

    def _chain_stage_loss(
        self,
        pred_phys: torch.Tensor,
        target_mean: torch.Tensor,
        fluid_mask: torch.Tensor,
        with_vgg: bool,
    ) -> torch.Tensor:
        """chain 單 stage 場精度+乾淨 loss（moment+chuber+vgg?+rel_l2+vel/vort ssim+spec+band+divj）。

        不含 fp-reg / jac / stay（收縮監督留給 normal `run()` 步）。複用 run() 同一批
        loss 函式與 cfg 權重，確保 chain stage 與單邊步的場精度目標一致。
        """
        ctx = self.ctx
        cfg = ctx.cfg
        _fn = getattr(ctx, "field_normalizer", None)
        loss = pred_phys.new_zeros(())

        # moment（mphy MSE）
        l_m, _ = mean_flow_loss(
            pred_phys,
            target_mean,
            fluid_mask,
            ssim_weight=0.0,
            per_channel_norm=cfg.mean_per_channel_norm,
        )
        loss = loss + float(cfg.mean_moment_weight) * l_m

        # colored Huber（感知殘差）
        if (
            float(cfg.mean_colored_huber_weight) > 0.0
            and _fn is not None
            and self._colorizer is not None
        ):
            from ...loss.terms.perceptual import colored_residual_huber_loss

            loss = loss + float(
                cfg.mean_colored_huber_weight
            ) * colored_residual_huber_loss(
                pred_phys,
                target_mean,
                fluid_mask,
                _fn,
                self._colorizer,
                beta=float(cfg.mean_colored_huber_beta),
            )
        # VGG 感知（stage2+ 預設 lean 跳過）
        _vgg_on = (
            float(cfg.mean_vgg_weight) > 0.0 or float(cfg.mean_vgg_ot_weight) > 0.0
        )
        if with_vgg and _vgg_on and _fn is not None and self._colorizer is not None:
            from ...loss.terms.vgg_perceptual import vgg_perceptual_loss

            _vgg_layers = tuple(
                int(s) for s in str(cfg.mean_vgg_layers).split(",") if s.strip()
            )
            loss = loss + vgg_perceptual_loss(
                pred_phys,
                target_mean,
                fluid_mask,
                _fn,
                self._colorizer,
                structure_weight=float(cfg.mean_vgg_weight),
                moment_weight=float(cfg.mean_vgg_moment_weight),
                ot_weight=float(cfg.mean_vgg_ot_weight),
                layers=_vgg_layers,
                max_size=int(cfg.mean_vgg_max_size),
            )
        # rel_l2
        if float(cfg.mean_rel_l2_weight) > 0.0:
            _m = fluid_mask.expand_as(pred_phys)
            _diff = (pred_phys.float() - target_mean) * _m
            _tgt = target_mean * _m
            _rel = (
                _diff.pow(2).sum().clamp_min(1e-12).sqrt()
                / _tgt.pow(2).sum().clamp_min(1e-12).sqrt()
            )
            loss = loss + float(cfg.mean_rel_l2_weight) * _rel
        # vel / vort 結構 SSIM
        if _fn is not None and (
            float(cfg.mean_vel_ssim_weight) > 0.0
            or float(cfg.mean_vort_ssim_weight) > 0.0
        ):
            from ...loss.terms.perceptual import (
                struct_vel_ssim_loss,
                struct_vort_ssim_loss,
            )

            _pk = pred_phys.unsqueeze(1)
            _tk = target_mean.unsqueeze(1)
            if float(cfg.mean_vel_ssim_weight) > 0.0:
                loss = loss + float(cfg.mean_vel_ssim_weight) * struct_vel_ssim_loss(
                    _pk,
                    _tk,
                    fluid_mask,
                    _fn,
                    range_from_stats=cfg.ssim_range_from_stats,
                )
            if float(cfg.mean_vort_ssim_weight) > 0.0:
                loss = loss + float(cfg.mean_vort_ssim_weight) * struct_vort_ssim_loss(
                    _pk,
                    _tk,
                    fluid_mask,
                    _fn,
                    range_from_stats=cfg.ssim_range_from_stats,
                )
        # spectral 低-mode
        if float(cfg.mean_spectral_weight) > 0.0:
            loss = loss + float(cfg.mean_spectral_weight) * mean_spectral_lowmode_loss(
                pred_phys,
                target_mean,
                fluid_mask,
                modes=int(cfg.mean_spectral_modes),
            )
        # column banding 抗條紋
        if float(cfg.mean_column_banding_weight) > 0.0:
            from ...loss.terms.struct_banding import struct_column_banding_loss

            loss = loss + float(
                cfg.mean_column_banding_weight
            ) * struct_column_banding_loss(
                pred_phys.unsqueeze(1), target_mean.unsqueeze(1), fluid_mask
            )
        # 全譜 radial power match（chain stage：攻累積的高k band/grain 譜根）
        if float(cfg.mean_specmatch_weight) > 0.0 and _fn is not None:
            loss = loss + float(cfg.mean_specmatch_weight) * spectral_power_match_loss(
                pred_phys.unsqueeze(1),
                target_mean.unsqueeze(1),
                fluid_mask,
                _fn,
                k_hi_frac=float(cfg.mean_specmatch_khi),
            )
        # 空間局域方向性 band loss（chain stage 上攻累積的欄向條紋）
        if float(cfg.mean_dirband_weight) > 0.0:
            from ...loss.terms.struct_banding import struct_directional_band_loss

            loss = loss + float(cfg.mean_dirband_weight) * struct_directional_band_loss(
                pred_phys.unsqueeze(1), target_mean.unsqueeze(1), fluid_mask
            )
        # divj 質量守恆（同 run，用 field_normalizer）
        if float(cfg.mean_divj_weight) > 0.0 and _fn is not None:
            loss = loss + float(cfg.mean_divj_weight) * phys_mass_divj_absmean(
                pred_phys.unsqueeze(1), fluid_mask, _fn
            )
        # ★energy-anchor：momentum-KE 雙向 log-ratio（治 chain 每-edge 能量泵浦/塌縮）
        if float(cfg.mean_chain_ke_weight) > 0.0 and _fn is not None:
            from ...loss.terms.phys_residual import mean_ke_log_anchor

            loss = loss + float(cfg.mean_chain_ke_weight) * mean_ke_log_anchor(
                pred_phys, target_mean, fluid_mask, _fn
            )
        return loss

    def _chain_rollout(
        self,
        state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        g_vec_norm: torch.Tensor,
        ke_anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """chain 單 stage rollout：no_grad preroll 收斂 + grad tail（bound 顯存）。回 final_state。

        k_total = mean_chain_converge_k（>16 roll 到收斂 attractor，KE 泵浦才可見 → anchor 才咬得動）；
        preroll no_grad 便宜、grad tail ≤6 bound 顯存。

        Args:
            state: 本 stage 起始 state。
            cond_norm: 正規化條件向量。
            fluid_mask: 流體遮罩。
            g_vec_norm: 正規化幾何向量或 None。
            ke_anchor: 三態協定產生的 KE ceiling 錨點；``ke_anchor_mode='off'`` 時為 None。

        Returns:
            本 stage 的 final_state。
        """
        cfg = self.ctx.cfg
        k_total = int(cfg.mean_chain_converge_k)
        grad_tail = min(
            int(cfg.mean_grad_tail), 6, k_total
        )  # chain grad tail 上限 6（2 stage 顯存）
        preroll = max(0, k_total - grad_tail)
        if preroll > 0:
            with torch.no_grad():
                state = self.ctx.model(
                    state,
                    cond_norm,
                    fluid_mask,
                    preroll,
                    return_trajectory=False,
                    g_vec_norm=g_vec_norm,
                    ke_anchor=ke_anchor,
                )
            state = state.detach()
        return self.ctx.model(
            state,
            cond_norm,
            fluid_mask,
            max(1, grad_tail),
            return_trajectory=False,
            g_vec_norm=g_vec_norm,
            ke_anchor=ke_anchor,
        )

    def _chain_contract_loss(
        self,
        static_jg: torch.Tensor,
        target_mean: torch.Tensor,
        final_state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        g_vec_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """chain 每 stage 收縮包（鏡射 run() 的 fp-reg + jac）：治 v11 泵浦=收縮稀釋假說。

        ★jac 施加在 chain 的 final_state（handoff-manifold，部署深鏈實際流形，normal 步採不到）。
        回 (loss, fp_val, jac_val)。
        """
        ctx = self.ctx
        cfg = ctx.cfg
        sd, pd = ctx.model._cfg.static_dim, ctx.model._cfg.physics_dim
        loss = final_state.new_zeros(())
        fp_val, jac_val = 0.0, 0.0
        # fp-reg：從 GT IC roll K 步仍須≈GT（GT-anchored identity；鏡射 run 512-527）
        fp_reg_w = float(cfg.mean_fp_reg_weight)
        if fp_reg_w > 0.0:
            fp_k = int(cfg.mean_chain_fp_rollout_k) or int(cfg.mean_fp_rollout_k)
            fp_ic = ctx.model.init_state(static_jg.detach(), target_mean)
            fp_out = ctx.model(
                fp_ic,
                cond_norm,
                fluid_mask,
                max(1, fp_k),
                return_trajectory=False,
                g_vec_norm=g_vec_norm,
                ke_anchor=protocol.resolve_ke_anchor(cfg, fp_ic, fluid_mask),
            )
            fp_phys = fp_out[:, sd : sd + pd]
            n_fluid = fluid_mask.sum().clamp_min(1.0)
            l_fp = ((fp_phys - target_mean) ** 2 * fluid_mask).sum() / n_fluid
            loss = loss + fp_reg_w * l_fp
            fp_val = float(l_fp.detach())
        # jac：收縮正則（同 run 543-556 的 ramp + interval，對齊 ctx.step）
        _jac_w0 = float(cfg.jac_reg_weight)
        _frac = float(ctx.step) / float(max(1, cfg.max_steps))
        _start = float(cfg.mean_jac_start_frac)
        _ramp = max(1e-6, float(cfg.mean_jac_ramp_frac))
        _jac_w = _jac_w0 * max(0.0, min(1.0, (_frac - _start) / _ramp))
        if (
            _jac_w > 0.0
            and self._regularizer is not None
            and (ctx.step % max(1, int(cfg.mean_jac_interval)) == 0)
        ):
            l_jac = self._regularizer["jacobian_reg_loss"](
                final_state.detach(), cond_norm, fluid_mask, g_vec_norm, None, None
            )
            loss = loss + _jac_w * l_jac
            jac_val = float(l_jac.detach())
        return loss, fp_val, jac_val

    def run_chain(self, batch: dict) -> dict[str, float]:
        """multi-edge pushforward 訓練步：連續 L 個 warm edge，逐 stage hand-off 並錨 GT。

        stage1：N0 mean_flow 注入 IC → rollout → 對 N1；stage j>1：上一 stage 模型輸出
        hand-off（``apply_ic_mask`` + static 置換，鏡射部署）→ rollout → 對 N_j。stage 間
        ``.detach()`` 截時序梯度（TBPTT 界，bound 顯存）。總 loss = Σ stage_weight·stage_loss，
        backward 一次。不含 fp-reg/jac/stay（留給 normal 步）。

        Args:
            batch: ``collate_chain_batch`` 產（parent_mean_renorm / chain_static /
                chain_target / chain_g_vec / chain_p_vec / source_stats_idx）。

        Returns:
            metrics dict（total / chain_stage0 / chain_stage_last / k）。
        """
        ctx = self.ctx
        cfg = ctx.cfg
        dev = ctx.device
        _need_log: bool = ctx.step % _LOG_INTERVAL == 0 or ctx.step < 100

        parent_mean = (
            batch["parent_mean_renorm"].to(dev, non_blocking=True).float()
        )  # [B,9,H,W]
        chain_static = (
            batch["chain_static"].to(dev, non_blocking=True).float()
        )  # [B,L,4,H,W]
        chain_target = (
            batch["chain_target"].to(dev, non_blocking=True).float()
        )  # [B,L,9,H,W]
        chain_g = batch["chain_g_vec"].to(dev, non_blocking=True)  # [B,L,14]
        chain_p = batch["chain_p_vec"].to(dev, non_blocking=True)  # [B,L,3]
        B, L = chain_static.shape[0], chain_static.shape[1]
        sd, pd = ctx.model._cfg.static_dim, ctx.model._cfg.physics_dim

        # lazy colorizer（chain 為首步時 run() 尚未建）
        _fn_c = getattr(ctx, "field_normalizer", None)
        _need_col = (
            float(cfg.mean_colored_huber_weight) > 0.0
            or float(cfg.mean_vgg_weight) > 0.0
            or float(cfg.mean_vgg_ot_weight) > 0.0
        )
        if (
            _need_col
            and _fn_c is not None
            and getattr(self, "_colorizer", None) is None
        ):
            from lbmnca.nca.utils.FieldColorizer import FieldColorizer

            self._colorizer = FieldColorizer.from_normalizer(_fn_c).to(dev)

        ctx.optimizer.zero_grad(set_to_none=True)
        stage_losses: list[torch.Tensor] = []
        _ke_ratios: list[float] = (
            []
        )  # 監控：per-stage KE_pred/KE_gt（anchor 是否壓向 1.0）
        _fp0, _jac0 = 0.0, 0.0  # 監控：chain 收縮包 loss（確認非零）
        final_state: Optional[torch.Tensor] = None
        fluid_prev: Optional[torch.Tensor] = (
            None  # 上一 stage 的 fluid_mask（ghost-IC handoff 用）
        )

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            for j in range(L):
                static_j = chain_static[:, j]  # [B,4,H,W]
                target_j = chain_target[:, j]  # [B,9,H,W]
                g_j, p_j = chain_g[:, j], chain_p[:, j]
                # source-local → global renorm（單 zarr 內同 source_idx）
                static_jg, ic_g, tgt_j5 = self._data_prep["renorm_source_to_global"](
                    static_j, parent_mean, target_j.unsqueeze(1), batch
                )
                target_jg = tgt_j5.squeeze(1)
                fluid_j = self._data_prep["get_fluid_mask"](static_jg)
                cond_j = self._data_prep["build_cond_norm"](g_j, p_j)
                gvn_j = self._data_prep["build_g_vec_norm"](g_j)

                if j == 0:
                    ic = ic_g  # renorm 後的 parent_mean
                    # 消融臂：鏈的 stage1 起點與 run() 作同樣的覆寫。
                    # j>0 為模型自身 hand-off，本就不取資料層 IC，不需處理。
                    if cfg.mean_ablation_arm == "no_edge":
                        ic = torch.zeros_like(ic)
                    elif cfg.mean_ablation_arm in ("noise_edge", "farnoise_edge"):
                        ic = _apply_ic_noise(
                            ic,
                            fluid_j,
                            cfg.mean_ablation_noise_sigma,
                            batch.get("case_key"),
                        )
                    # apply_ic_mask 由三態協定唯一實作決定（禁 fallback）
                    state = protocol.initial_state(
                        ctx.model, cfg, static_jg, ic, fluid_j
                    )
                else:
                    # hand-off 由三態協定唯一實作（hidden 保留 +
                    # handoff_reset_mask + static 置換）；評估腳本改走同一函式。
                    state = protocol.chain_handoff(
                        ctx.model,
                        cfg,
                        final_state.detach(),
                        static_jg,
                        fluid_j,
                        fluid_prev,
                    )

                # chain 分支的 anchor 由三態協定的 resolve_ke_anchor 統一產生，
                # 與 run() 的 GT 錨、部署的 running-min 錨走同一套邏輯。
                ke_anchor_j = protocol.resolve_ke_anchor(cfg, state, fluid_j)
                final_state = self._chain_rollout(
                    state, cond_j, fluid_j, gvn_j, ke_anchor_j
                )
                fluid_prev = fluid_j
                pred_j = final_state[:, sd : sd + pd]
                _w = 1.0 if j == 0 else float(cfg.mean_chain_stage_weight)
                _with_vgg = (j == 0) or bool(cfg.mean_chain_vgg)
                _sl = self._chain_stage_loss(pred_j, target_jg, fluid_j, _with_vgg)
                if cfg.mean_chain_contract:  # ★收縮包 on chain（fp-reg+jac）
                    _cl, _fpv, _jacv = self._chain_contract_loss(
                        static_jg, target_jg, final_state, cond_j, fluid_j, gvn_j
                    )
                    _sl = _sl + _cl
                    if j == 0:
                        _fp0, _jac0 = _fpv, _jacv
                stage_losses.append(_w * _sl)
                if (
                    _need_log and _fn_c is not None
                ):  # 監控 KE ratio（detached，僅 log 步）
                    with torch.no_grad():
                        _rp = _fn_c.denorm_mphy(pred_j.detach().float())
                        _rt = _fn_c.denorm_mphy(target_jg.float())
                        _mm = fluid_j.squeeze(1)
                        _dn = _mm.sum().clamp_min(1.0)
                        _kp = (
                            0.5 * (_rp[:, 3].square() + _rp[:, 5].square()) * _mm
                        ).sum() / _dn
                        _kt = (
                            0.5 * (_rt[:, 3].square() + _rt[:, 5].square()) * _mm
                        ).sum() / _dn
                        _ke_ratios.append(float(_kp / _kt.clamp_min(1e-8)))

            loss = sum(stage_losses)

        # ── NaN guard（比照 run）──
        loss_val = float(loss.detach().float().item())
        if (not math.isfinite(loss_val)) or (loss_val > cfg.loss_spike_threshold):
            self._nan_consecutive += 1
            logger.warning(
                "[step %d] chain unstable loss=%.4g consecutive=%d/%d",
                ctx.step,
                loss_val,
                self._nan_consecutive,
                cfg.nan_patience,
            )
            if self._nan_consecutive >= cfg.nan_patience:
                raise RuntimeError(f"NaN early stop (chain) at step {ctx.step}")
            ctx.step += 1
            return {"total": float("nan"), "k": L, "loss_version": "mean_chain"}

        # ── backward + opt（比照 run 尾段）──
        ctx.scaler.scale(loss).backward()
        ctx.scaler.unscale_(ctx.optimizer)
        if cfg.mean_grad_normalize:
            with torch.no_grad():
                for _p in ctx.model.parameters():
                    if _p.grad is not None:
                        _p.grad.div_(_p.grad.norm() + 1e-8)
        grad_norm_t = nn.utils.clip_grad_norm_(
            ctx.model.parameters(), max_norm=cfg.grad_clip_norm
        )
        _scale_before = ctx.scaler.get_scale() if cfg.use_amp else 0.0
        ctx.scaler.step(ctx.optimizer)
        ctx.scaler.update()
        if not (cfg.use_amp and ctx.scaler.get_scale() < _scale_before):
            ctx.scheduler.step()
        self._nan_consecutive = 0
        ctx.step += 1
        return {
            "total": loss_val,
            "chain_stage0": float(stage_losses[0].detach().item()),
            "chain_stage_last": float(stage_losses[-1].detach().item()),
            "chain_ke_ratio_s0": (_ke_ratios[0] if _ke_ratios else None),
            "chain_ke_ratio_slast": (_ke_ratios[-1] if _ke_ratios else None),
            "chain_fp": _fp0,
            "chain_jac": _jac0,
            "k": L,
            "loss_version": "mean_chain",
            "grad_norm": grad_norm_t.item() if _need_log else None,
        }
