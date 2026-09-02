"""TESTrainer — TES-NCA 訓練迴圈，k curriculum + Loss curriculum + IC masking。"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import IO, Any, Callable, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ....paths import REPO_ROOT
from ..config import TESTrainConfig
from ..data.normalizer import TESCondNormalizer
from ..inference import protocol
from ..loss.loss_terms_v2 import (
    domain_mean_ke,  # cross-CTU 整段 KE 衰罰
    enstrophy_match_loss,
    fneq_match_loss,
    jacobian_amp_loss,
    mean_flow_loss,  # mean_module loss（MSE + SSIM vs mean_flow_post_conv）
    phys_mass_divj_absmean,
    spectral_power_match_loss,  # 2D radial power spectrum match（破 scalar 錨二分）
    two_window_ke_loss,
    vel_field_decoupled_loss,
)
from ..eval.mean_gate_eval import (  # mean gate 定量評估
    ke_curve_convergence,
    ke_norm,
    mean_gate_metrics,
    mean_gate_pool_terms,
    self_rollout_drift,
)
from ..models.tes_nca import TESNCA
from . import rng_streams  # 具名 RNG 串流（isolation 預設關閉＝全域 random）
from . import schedule as _sched
from .checkpoint import identity as ckpt_identity  # ckpt 身分欄位（build/verify）
from .checkpoint.checkpointer import (
    save_checkpoint,
)  # ★存檔單一實作（含落盤後身分斷言）
from .early_stopping import TESEarlyStopping
from .validation.aggregate import POOL_KEY_2K, aggregate_val_metrics
from .early_stop_dispatch import dispatch_early_stop
from .jsonl import dumps_record

from .context import TrainerContext
from .data_prep import ConditionBuilder, DataRenormalizer, ICOps
from .regularization.regularizer import Regularizer as _RegularizerCore
from .steps import BaseStep, MeanStep

logger = logging.getLogger(__name__)

_LOG_INTERVAL: int = 100  # train_step record write/flush 頻率（每 N 步寫一次）


class TESTrainer:
    """TES-NCA 訓練器。

    訓練協議：
        1. k curriculum：phase1(k=1) → phase2(k=2-4) → phase3(k=4-8)
        2. Loss curriculum：v1 → v2 → v3（與 k phase 對齊）
        3. IC masking：t_start=0 時對整個 batch 執行（cold-start 安全，warm-start 必要）
        4. 單階段訓練，無 stop_gradient

    Args:
        cfg: TESTrainConfig 完整設定。
        model: TESNCA 模型實例。
        loss_fn: semantic loss module.
        normalizer: TESCondNormalizer 條件正規化器。
        train_loader: 訓練 DataLoader（cold + warm 合併）。
        val_cold_loader: cold val DataLoader（可選，無 cold val cases 時為 None）。
        val_warm_loader: warm val DataLoader（可選，無 warm val cases 時為 None）。
        device: 訓練裝置。
        create_run_dir: 是否在 ``cfg.output_dir`` 建立 run 目錄並接上檔案 logging。
            訓練一律為 True（預設）。純推論（製圖 / 評估 harness）須傳 False，
            否則每建一次 trainer 就在 ``runs/`` 留下一個沒有 ckpt 的空 run 目錄。
    """

    def __init__(
        self,
        cfg: TESTrainConfig,
        model: TESNCA,
        loss_fn: nn.Module,
        normalizer: TESCondNormalizer,
        train_loader: DataLoader,
        val_cold_loader: Optional[DataLoader] = None,
        val_warm_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        field_normalizer: Optional[object] = None,
        source_field_normalizers: Optional[list[object]] = None,
        chain_loader: Optional[DataLoader] = None,
        create_run_dir: bool = True,
    ) -> None:
        self._cfg = cfg
        self._model = model
        self._loss_fn = loss_fn
        self._normalizer = normalizer
        self._field_normalizer: Optional[object] = field_normalizer
        self._source_field_normalizers = source_field_normalizers or []
        self._train_loader = train_loader
        self._chain_loader = chain_loader  # multi-edge pushforward（None=停用）
        self._val_cold_loader = val_cold_loader
        self._val_warm_loader = val_warm_loader
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self._field_normalizer is not None and hasattr(self._field_normalizer, "to"):
            self._field_normalizer = self._field_normalizer.to(self._device)
        self._source_field_normalizers = [
            n.to(self._device) if hasattr(n, "to") else n
            for n in self._source_field_normalizers
        ]
        if hasattr(self._normalizer, "to"):
            self._normalizer = self._normalizer.to(self._device)
        self._step = 0
        self._nan_consecutive = 0  # 連續不穩定步計數

        self._model.to(self._device)
        self._loss_fn.to(self._device)

        self._optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self._scheduler = self._build_scheduler()
        self._scaler = GradScaler("cuda", enabled=cfg.use_amp)
        self._early_stop = TESEarlyStopping(cfg.early_stop)
        self._should_stop: bool = False

        # 推論模式（create_run_dir=False）只保留路徑供參考，不落地任何檔案。
        self._create_run_dir: bool = create_run_dir
        output_dir = Path(cfg.output_dir)
        if create_run_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir

        self._metrics_fp: Optional[IO[str]] = None
        self._bg_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="tes-eval")
            if cfg.async_eval_generation
            else None
        )
        self._bg_jobs: list[Future] = []
        self._async_eval_proc: subprocess.Popen | None = None
        self._async_vis_proc: subprocess.Popen | None = None
        self._sdf_norm_threshold: float = self._load_sdf_threshold()
        # _setup_logging 會在 output_dir 內開 training.log / metrics.json；推論模式
        # 沒有 run 目錄，一併略過（metrics_fp 維持 None，寫入端皆已做 None 檢查）。
        if create_run_dir:
            self._setup_logging()
        # `pool.mode == "rolling"`（TESRollingPool）依賴 v3 的 `pool_fetch`／
        # `rolling_pool`（沿 case 取完整 GT 軌跡），v4 unified 沒有軌跡可取，
        # 整條已歸檔；`self._pool` 恆為 None，只保留給下游 None 檢查。
        self._pool = None

        # ── 組合模式（Option B）：建構共享 context + 子元件 ──────────────
        self._build_components()

    def _build_components(self) -> None:
        """建構 TrainerContext 與所有訓練子元件（資料前處理 / 正則化 / step）。

        子元件透過共享的 TrainerContext 存取模型、設定與可變訓練狀態；
        部分子元件（Dy2Step）以 ``ctx._trainer`` 反向委派至本 trainer 的
        權威方法（_build_cond_norm / _renorm_* / _temporal_ke_loss 等），
        確保行為與單體版本完全一致。
        """
        ctx = TrainerContext(
            model=self._model,
            loss_fn=self._loss_fn,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            scaler=self._scaler,
            early_stopping=self._early_stop,
            train_loader=self._train_loader,
            val_cold_loader=self._val_cold_loader,
            val_warm_loader=self._val_warm_loader,
            normalizer=self._normalizer,
            field_normalizer=self._field_normalizer,
            source_field_normalizers=self._source_field_normalizers,
            pool=self._pool,
            cfg=self._cfg,
            device=self._device,
            output_dir=self._output_dir,
            sdf_threshold=self._sdf_norm_threshold,
            step=self._step,
            metrics_file=self._metrics_fp,
            logger=logger,
            executor=self._bg_executor,
        )
        # 反向委派掛載：子元件（Dy2Step / Validator 等）以 getattr 查找這些
        # 屬性委派回 trainer 的權威實作；缺失時退回各自 fallback。
        ctx._trainer = self
        ctx._save_checkpoint = self._save_checkpoint
        ctx._bg_jobs = self._bg_jobs
        ctx._renorm_source_to_global = self._renorm_source_to_global
        ctx._renorm_ic_mphy = self._renorm_ic_mphy
        ctx._renorm_ic_mphy_traj = self._renorm_ic_mphy_traj
        self._ctx = ctx

        # 資料前處理 / 正則化 facade（同時支援 dict-style 與 attribute-style 存取）
        self._data_prep = _DataPrepFacade(ctx)
        self._regularizer = _RegularizerFacade(ctx)

        # step 子元件（依 cfg 對應訓練模式）
        self._step_base = BaseStep(ctx, self._data_prep, self._regularizer)
        self._step_mean = MeanStep(ctx, self._data_prep, self._regularizer)

    def _select_step(self):
        """回傳訓練 step 子元件。

        只餘 mean 一條路徑（turb / cross_ctu / dy2 / pool 已歸檔），故此處無分支。
        ``_step_base`` 保留供 baseline 模型比較用。

        Returns:
            ``MeanStep`` 實例。
        """
        return self._step_mean

    def _load_sdf_threshold(self) -> float:
        """載入 SDF 流體判定門檻。

        v3 tanh_lchar 正規化下門檻恆為 0.0（tanh(0)=0 = 障礙物表面）；提供 runtime
        normalizer 時改由其 norm_sdf(0) 推導。

        ★**canonical 走的是 runtime 分支**
        （``configs/nca/main.yaml`` 的 ``stats_policy:
        dataset_global_runtime``），而該分支原本對 ``norm_sdf(0)`` 的回傳值
        完全不做檢查。``SDF_norm`` 經 ``tanh`` 壓在 ``(-1, 1)``，所以門檻一旦
        ``≤ -1`` 就等於「遮罩全開」（``SDF_norm > thr`` 恆真）、``≥ 1`` 等於
        「遮罩全關」——兩者都會靜默地讓整個流體遮罩失效。這正是
        ``eval_warmzeroshot`` 那個 ``-1.0`` 回退的同一個缺陷類別，只是發生在
        canonical 路徑上、而且先前連 stats-file 分支那種檢查都沒有。現補上值域守衛。

        Returns:
            SDF_norm 流體門檻值。

        Raises:
            ValueError: runtime normalizer 給出的門檻落在 ``(-1, 1)`` 之外
                （遮罩會全開或全關），或 stats 檔的 ``static_norm.sdf.method``
                非 ``tanh_lchar``（v3）。
        """
        if self._field_normalizer is not None:
            zero = torch.tensor(0.0)
            thr = float(self._field_normalizer.norm_sdf(zero).item())
            if not (-1.0 < thr < 1.0):
                raise ValueError(
                    f"runtime normalizer 給出的 SDF fluid threshold={thr} 落在 "
                    "tanh 值域 (-1, 1) 之外——門檻 ≤ -1 使流體遮罩全開、≥ 1 使其全關，"
                    "兩者都會讓遮罩靜默失效。請檢查 normalizer 的 SDF 統計量"
                    "（tanh_lchar 佈局下正確值應為 0.0）。"
                )
            logger.info("SDF fluid threshold from runtime normalizer: %.4f", thr)
            return thr
        # runtime normalizer 是唯一的正規化空間來源（train.py 強制
        # stats_policy=dataset_global_runtime）。normalizer 不存在就是組裝順序錯了，
        # 不該退回一條會讀到「某一本」stats 的路徑。
        raise RuntimeError(
            "_load_sdf_threshold：field_normalizer 尚未建立。SDF 門檻只能由 runtime "
            "dataset-global normalizer 推導。"
        )

    def _get_field_norm(self) -> object:
        """延遲載入 FieldNormalizer（將 MRT physics 反正規化回 lattice 單位）。

        Returns:
            FieldNormalizer 實例。由 ``build_tes_trainer`` 於建構時注入
            （runtime dataset-global stats），本函式不再自行從檔案建立——
            那會拿到「某一本」dataset 的 stats 而非合併後的全域空間。

        Raises:
            RuntimeError: normalizer 未被注入（組裝順序錯誤）。
        """
        if self._field_normalizer is None:
            raise RuntimeError(
                "_get_field_norm：field_normalizer 未注入。它應由 build_tes_trainer "
                "以 FieldNormalizer.from_stats_dict(global_stats) 建立後傳入。"
            )
        return self._field_normalizer

    def _renorm_source_to_global(
        self,
        static: torch.Tensor,
        ic_physics: torch.Tensor,
        target_traj: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """將 source-local 正規化張量轉至 runtime dataset-global 正規化空間。

        多 zarr local stats 不一致時必須（逐 source_stats_idx 反正規化回 raw 再以
        global normalizer 重新正規化）；單 dataset（無 source_field_normalizers 或
        無 source_stats_idx）為 no-op 直接回傳原張量。

        Args:
            static: 靜態通道張量 [B, 4, H, W]。
            ic_physics: IC physics 張量 [B, 9, H, W]。
            target_traj: 目標軌跡張量 [B, K, 9, H, W]。
            batch: DataLoader batch dict（讀取 source_stats_idx）。

        Returns:
            (static, ic_physics, target_traj) 轉至 global norm 空間後的版本。
        """
        if not self._source_field_normalizers:
            return static, ic_physics, target_traj
        source_idx = batch.get("source_stats_idx")
        if source_idx is None:
            return static, ic_physics, target_traj

        global_norm = self._get_field_norm()
        source_idx = source_idx.to(self._device, non_blocking=True)
        B, K, C, H, W = target_traj.shape
        static_out = static.clone()
        ic_out = ic_physics.clone()
        target_out = target_traj.clone()

        for idx in torch.unique(source_idx).tolist():
            idx_int = int(idx)
            src_norm = self._source_field_normalizers[idx_int]
            mask = source_idx == idx_int
            if not bool(mask.any()):
                continue

            ic_out[mask] = global_norm.norm_mphy(src_norm.denorm_mphy(ic_physics[mask]))

            tgt_src = target_traj[mask].reshape(-1, C, H, W)
            tgt_raw = src_norm.denorm_mphy(tgt_src)
            target_out[mask] = global_norm.norm_mphy(tgt_raw).reshape(-1, K, C, H, W)

            st = static[mask]
            st_out = st.clone()
            st_out[:, 0] = global_norm.norm_sdf(src_norm.denorm_sdf(st[:, 0]))
            st_out[:, 1] = global_norm.norm_upstream(src_norm.denorm_upstream(st[:, 1]))
            st_out[:, 2] = global_norm.norm_pos_x(src_norm.denorm_pos_x(st[:, 2]))
            st_out[:, 3] = global_norm.norm_pos_y(src_norm.denorm_pos_y(st[:, 3]))
            static_out[mask] = st_out

        return static_out, ic_out, target_out

    def _renorm_ic_mphy(self, ic_raw: torch.Tensor, batch: dict) -> torch.Tensor:
        """將單一 Mphy IC 幀 renorm 至 dataset-global 正規化空間。

        供 RP delta-loss 參考幀（``t_start + r_rp``）使用——該幀不屬於
        ``target_traj`` 但須共用 global norm 策略。

        Args:
            ic_raw: 單幀 Mphy 張量 [B, 9, H, W]（source-local norm）。
            batch: DataLoader batch dict（讀取 source_stats_idx）。

        Returns:
            轉至 global norm 空間後的單幀張量。
        """
        if not self._source_field_normalizers:
            return ic_raw
        source_idx = batch.get("source_stats_idx")
        if source_idx is None:
            return ic_raw
        global_norm = self._get_field_norm()
        source_idx = source_idx.to(self._device, non_blocking=True)
        ic_out = ic_raw.clone()
        for idx in torch.unique(source_idx).tolist():
            idx_int = int(idx)
            src_norm = self._source_field_normalizers[idx_int]
            mask = source_idx == idx_int
            if not bool(mask.any()):
                continue
            ic_out[mask] = global_norm.norm_mphy(src_norm.denorm_mphy(ic_raw[mask]))
        return ic_out

    _add_vg_composite = staticmethod(_sched.add_vg_composite)
    _select_early_stop_monitor = staticmethod(_sched.select_early_stop_monitor)

    def _get_fluid_mask(self, static: torch.Tensor) -> torch.Tensor:
        """由 SDF_norm 依 dataset SDF 統計推導正確的二值流體遮罩。

        ``SDF_norm > threshold`` → 流體（1.0），否則固體（0.0）。

        Args:
            static: 靜態通道張量 [B, 4, H, W]（channel 0 為 SDF_norm）。

        Returns:
            流體遮罩 [B, 1, H, W]（float，1.0=流體 / 0.0=固體）。
        """
        return (static[:, 0:1] > self._sdf_norm_threshold).float()

    def _get_prior_vel_norm(self) -> torch.Tensor:
        """均勻入口 momentum 的 normalized 值 [jx_inlet_norm, jy_inlet_norm]（cache）。

        raw jx_inlet = rho(≈1)·U_inlet(0.05)、jy_inlet = 0；以 physics normalizer
        tanh((raw−M_REF)/S_combined) 轉至 model state 空間。deterministic prior 的目標速度。
        """
        cached = getattr(self, "_prior_vel_norm_cache", None)
        if cached is None:
            nz = self._loss_fn._get_normalizer()
            mref, scomb = nz._mf_mean, nz._mf_std  # [9] M_REF / S_combined
            u_inlet = 0.05  # P_vec[1] 固定（lattice）
            jx_n = torch.tanh((u_inlet - mref[3]) / scomb[3])
            jy_n = torch.tanh((0.0 - mref[5]) / scomb[5])
            cached = torch.stack([jx_n, jy_n]).to(self._device).float()
            self._prior_vel_norm_cache = cached
        return cached

    def _get_mean_p_gt(self) -> float:
        """mean/turb scheduled sampling 的 GT injection 機率。

        排程（progress = step / max_steps）：
            progress ∈ [0, p1_end]      → p_gt=1.0（全 GT）
            progress ∈ [p1_end, p2_end] → 線性 1→0
            progress ∈ [p2_end, 1]      → p_gt=0.0（全 TBPTT）

        Returns:
            當前步的 GT injection 機率 p_gt。
        """
        p1 = float(self._cfg.mean_gt_phase1_end)
        p2 = float(self._cfg.mean_gt_phase2_end)
        progress = self._step / max(1, self._cfg.max_steps)
        if progress <= p1:
            return 1.0
        if progress >= p2:
            return 0.0
        return 1.0 - (progress - p1) / max(1e-6, p2 - p1)

    def _write_jsonl(self, record: dict, force_flush: bool = False) -> None:
        """寫一筆 JSON record 至 metrics.json（line-delimited 格式）。

        Args:
            record: 待寫入的 dict。
            force_flush: True 時立即 flush（event/error/warning 用）；
                         False 時僅 write buffer，由 100 步 train_step 統一 flush。
        """
        if self._metrics_fp is not None:
            self._metrics_fp.write(dumps_record(record) + "\n")
            if force_flush:
                self._metrics_fp.flush()

    def _submit_background_job(
        self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """在背景執行緒跑 CPU 端生成工作，避免阻塞訓練迴圈。

        Args:
            name: 工作名稱（log 用）。
            fn: 待執行的 callable。
            *args: 傳給 fn 的位置參數。
            **kwargs: 傳給 fn 的關鍵字參數。
        """
        if self._bg_executor is None:
            fn(*args, **kwargs)
            return

        def _wrapped() -> None:
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                logger.debug(
                    "Background %s skipped at step %d: %s", name, self._step, exc
                )

        self._bg_jobs = [job for job in self._bg_jobs if not job.done()]
        if self._bg_jobs:
            logger.debug(
                "Background %s skipped at step %d: previous job still running",
                name,
                self._step,
            )
            return
        self._bg_jobs.append(self._bg_executor.submit(_wrapped))

    def train(self) -> None:
        """執行完整訓練迴圈直至 max_steps。"""
        # ── 啟動 header ────────────────────────────────────────────────
        n_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        cfg_dict = dataclasses.asdict(self._cfg)
        val_cold_n = (
            len(self._val_cold_loader.dataset)
            if self._val_cold_loader is not None
            else 0
        )
        val_warm_n = (
            len(self._val_warm_loader.dataset)
            if self._val_warm_loader is not None
            else 0
        )
        logger.info(
            "TESTrainer 開始訓練，device=%s, max_steps=%d, n_params=%s (%.2fM)",
            self._device,
            self._cfg.max_steps,
            f"{n_params:,}",
            n_params / 1e6,
        )
        logger.info(
            "val split — cold: %d cases, warm: %d cases",
            val_cold_n,
            val_warm_n,
        )
        logger.info(
            "=== TES Training Config ===\n%s",
            json.dumps(cfg_dict, indent=2, default=str),
        )
        self._write_jsonl(
            {
                "event": "startup",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "device": str(self._device),
                "n_params": n_params,
                "val_cold_cases": val_cold_n,
                "val_warm_cases": val_warm_n,
                "config": cfg_dict,
            },
            force_flush=True,
        )

        self._model.train()
        data_iter = iter(self._train_loader)
        # multi-edge pushforward：獨立 chain loader（None=停用），以 chain_prob 混編
        _chain_iter = (
            iter(self._chain_loader) if self._chain_loader is not None else None
        )
        _chain_prob = (
            float(self._cfg.mean_chain_prob) if self._chain_loader is not None else 0.0
        )
        t0 = time.time()
        _window_t = t0  # 100-step 視窗計時器

        _log_ctx = logging_redirect_tqdm()
        _log_ctx.__enter__()

        pbar = tqdm(
            total=self._cfg.max_steps,
            initial=self._step,
            desc="TES-NCA",
            unit="step",
            dynamic_ncols=True,
            colour="white",
        )

        t_train_start: float = time.time()
        try:
            while self._step < self._cfg.max_steps and not self._should_stop:
                # ctx.step ↔ self._step 橋接：step 子元件以 ctx.step 排程並遞增，
                # train loop / val / checkpoint 仍以 self._step 為主視角。
                self._ctx.step = self._step
                # chain 分支機率走具名串流（isolation 關閉時＝全域 random）
                if (
                    _chain_iter is not None
                    and rng_streams.stream("chain").random() < _chain_prob
                ):
                    # multi-edge pushforward 分支（缺陷 b）：抽 chain batch → run_chain
                    try:
                        _cb = next(_chain_iter)
                    except StopIteration:
                        _chain_iter = iter(self._chain_loader)
                        _cb = next(_chain_iter)
                    metrics = self._step_mean.run_chain(_cb)
                else:
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self._train_loader)
                        batch = next(data_iter)
                    # 委派至對應 step 子元件（行為與原 _train_step* 完全一致）
                    metrics = self._select_step().run(batch)
                self._step = self._ctx.step
                elapsed = time.time() - t0

                # ── 計算吞吐量（每 100 步更新一次；前 100 步每步更新）──
                steps_per_sec: Optional[float] = None
                _early_step = self._step < 100
                if (self._step % 100 == 0 and self._step > 0) or _early_step:
                    window_elapsed = time.time() - _window_t
                    n_window = 1 if _early_step else 100
                    steps_per_sec = n_window / max(window_elapsed, 1e-6)
                    _window_t = time.time()

                # ── train_step JSON record：每 _LOG_INTERVAL 步 or 前 100 步每步 ──
                lr = self._scheduler.get_last_lr()[0]
                if self._step % _LOG_INTERVAL == 0 or _early_step:
                    json_record: dict = {
                        "event": "train_step",
                        "step": self._step,
                        "loss": metrics.get("total"),
                        "k": metrics.get("k"),
                        "v": metrics.get("loss_version"),
                        "lr": lr,
                        "t": elapsed,
                        "grad_norm": metrics.get("grad_norm"),
                        "scaler_scale": metrics.get("scaler_scale"),
                        "gpu_mem_mb": metrics.get("gpu_mem_mb"),
                        "t_start": metrics.get("t_start"),
                        "global_t_start": metrics.get("global_t_start"),
                        "nan_consec": self._nan_consecutive,
                    }
                    for ckey in (
                        "rho_mae",
                        "jx_mae",
                        "jy_mae",
                        "ch0_mae",
                        "ch1_mae",
                        "ch2_mae",
                        "ch3_mae",
                        "ch4_mae",
                        "ch5_mae",
                        "ch6_mae",
                        "ch7_mae",
                        "ch8_mae",
                        "vel_mag_mae",
                    ):
                        if ckey in metrics:
                            json_record[ckey] = metrics[ckey]
                    # 語義化 loss 會輸出 obsmphy_*/obsvel_*/physmass_*/bndwall_* 等 key。
                    # 保留所有 scalar metric，避免新 loss ablation 在記錄檔中只剩 total。
                    for key, value in metrics.items():
                        if key in json_record or key in {"nan_keys"}:
                            continue
                        if isinstance(value, (int, float, str)) or value is None:
                            json_record[key] = value
                    if steps_per_sec is not None:
                        json_record["steps_per_sec"] = steps_per_sec
                    self._write_jsonl(json_record, force_flush=True)

                # ── progress bar 更新 ─────────────────────────────────
                pbar.set_postfix(
                    loss=f"{metrics.get('total', float('nan')):.4f}",
                    k=metrics.get("k", 0),
                    v=metrics.get("loss_version", "?"),
                    lr=f"{lr:.2e}",
                    refresh=False,
                )
                pbar.update(1)

                # ── training.log：每 50 步詳細輸出（★加密：band 診斷需看 per-term 收斂）──
                if self._step % 50 == 0:
                    gn = metrics.get("grad_norm")
                    amp = metrics.get("scaler_scale")
                    mem = metrics.get("gpu_mem_mb")
                    pt = metrics.get("p_teacher")
                    gn_str = f"  gn={gn:.3f}" if gn is not None else ""
                    amp_str = f"  amp={amp:.0f}" if amp is not None else ""
                    mem_str = f"  mem={mem:.0f}MB" if mem is not None else ""
                    pt_str = f"  pt={pt:.3f}" if pt is not None else ""
                    logger.info(
                        "[step %6d] loss=%.4f  k=%d  v=%s  lr=%.2e  t=%.1fs%s%s%s%s",
                        self._step,
                        metrics.get("total", float("nan")),
                        metrics.get("k", 0),
                        metrics.get("loss_version", "?"),
                        lr,
                        elapsed,
                        gn_str,
                        amp_str,
                        mem_str,
                        pt_str,
                    )
                    # ★mean-flow per-term 診斷（band 收斂觀察）：只在 mean/mean_chain step 有這些 key
                    _ml = [
                        f"{_k}={metrics[_k]:.4f}"
                        for _k in (
                            "mse",
                            "spec",
                            "band",
                            "vgg",
                            "vel_ssim",
                            "vort_ssim",
                            "enstrophy",
                            "velgrad",
                            "hband",
                            "jac_reg",
                            "stay",
                            "delta_focus",
                            "chuber",
                        )
                        if isinstance(metrics.get(_k), (int, float))
                    ]
                    if _ml:
                        logger.info("  └─ mean loss: %s", "  ".join(_ml))
                    loss_parts = []
                    for key in sorted(metrics.keys()):
                        if key.startswith("w_") or key in {
                            "total",
                            "k",
                            "loss_version",
                            "grad_norm",
                            "scaler_scale",
                            "gpu_mem_mb",
                            "t_start",
                            "global_t_start",
                            "nan_keys",
                            "p_teacher",
                        }:
                            continue
                        if key.startswith(
                            (
                                "obsmphy_",
                                "obsvel_",
                                "obsvphy_",
                                "obsstate_",
                                "physmass_",
                                "bndwall_",
                            )
                        ):
                            wkey = f"w_{key}"
                            if wkey in metrics:
                                loss_parts.append(
                                    f"{key}={metrics[key]:.4f}(w={metrics[wkey]:.4f})"
                                )
                            else:
                                loss_parts.append(f"{key}={metrics[key]:.5f}")
                    if loss_parts:
                        logger.info("  └─ semantic loss: %s", "  ".join(loss_parts))
                    ch_parts = []
                    for ch_idx in range(9):
                        ckey = f"ch{ch_idx}_mae"
                        if ckey in metrics:
                            ch_parts.append(f"ch{ch_idx}={metrics[ckey]:.4f}")
                    if "vel_mag_mae" in metrics:
                        ch_parts.append(f"vmag={metrics['vel_mag_mae']:.4f}")
                    if ch_parts:
                        logger.info("  └─ ch: %s", "  ".join(ch_parts))

                # ── 驗證（cold / warm 分開記錄）────────────────────
                if self._step % self._cfg.val_interval == 0:
                    cold_metrics: dict[str, float] = {}
                    warm_metrics: dict[str, float] = {}
                    monitor = self._cfg.early_stop.monitor
                    do_visual = self._step % self._cfg.vis_interval == 0 or str(
                        monitor
                    ).startswith("vg_")
                    # 按需 aux 診斷：只在 val 區開（train hot path 保持關，省每步 quantile/max/smoothness）
                    if hasattr(self._model, "set_aux_metrics_enabled"):
                        self._model.set_aux_metrics_enabled(True)
                    for split_name, loader in (
                        ("cold", self._val_cold_loader),
                        ("warm", self._val_warm_loader),
                    ):
                        if loader is None:
                            continue
                        # mean gate 定量評估（唯一 val 路徑；monolithic 與
                        # turb 兩條分支已移除）
                        split_metrics = self._validate_split_mean(loader)
                        val_parts = "  ".join(
                            f"{k}={v:.4f}" for k, v in sorted(split_metrics.items())
                        )
                        logger.info(
                            "[step %6d] VAL %s: %s", self._step, split_name, val_parts
                        )
                        self._write_jsonl(
                            {
                                "event": "val",
                                "split": split_name,
                                "step": self._step,
                                **split_metrics,
                            },
                            force_flush=True,
                        )
                        if split_name == "cold":
                            cold_metrics = split_metrics
                        else:
                            warm_metrics = split_metrics

                    # ── Val field snapshot (for field_compare plot) ──
                    if do_visual:
                        try:
                            self._save_val_fields(
                                self._val_cold_loader, self._val_warm_loader
                            )
                        except Exception as _vis_err:
                            logger.warning("Val field snapshot skipped: %s", _vis_err)
                        # rollout/序列 vis 走 async subprocess（_launch_async_vis，不阻塞）。
                        # 但 pool_vel 須畫「當下 live 訓練 pool」→ 必在主線 inline（async subprocess
                        # 看不到訓練 process 的 pool；rebuild 會是假象）。每 vis_interval 才畫、阻塞 ~1s 可忽略。
                        try:
                            self._save_pool_vis()
                        except Exception as _pv_err:
                            logger.warning("Pool vis skipped: %s", _pv_err)

                    # ── Visual gate metrics（提前到 early stop 之前）──
                    _vg: dict[str, float] = {}
                    if do_visual:
                        try:
                            _vg = self._compute_val_visual_gate(
                                self._val_cold_loader, self._val_warm_loader
                            )
                            if _vg:
                                _vg = self._add_vg_composite(_vg)
                                _vg_str = "  ".join(
                                    f"{k.replace('vg_', '')}={v:.4f}"
                                    for k, v in sorted(_vg.items())
                                )
                                logger.info(
                                    "[step %6d] VIS-GATE: %s", self._step, _vg_str
                                )
                                self._write_jsonl(
                                    {"event": "visual_gate", "step": self._step, **_vg},
                                    force_flush=True,
                                )
                            # flush model aux scale metrics (H-series only)
                            try:
                                if hasattr(self._model, "get_aux_metrics"):
                                    _aux = self._model.get_aux_metrics()
                                    if _aux:
                                        self._write_jsonl(
                                            {
                                                "event": "scale_metrics",
                                                "step": self._step,
                                                **_aux,
                                            },
                                            force_flush=True,
                                        )
                            except Exception as _aux_err:
                                logger.debug(
                                    "Aux scale_metrics skipped at step %d: %s",
                                    self._step,
                                    _aux_err,
                                )
                        except Exception as _vg_err:
                            logger.debug(
                                "Visual gate skipped at step %d: %s",
                                self._step,
                                _vg_err,
                            )

                    # 關閉 aux 診斷（恢復 train hot path 快路徑；硬設 False 避洩漏）
                    if hasattr(self._model, "set_aux_metrics_enabled"):
                        self._model.set_aux_metrics_enabled(False)

                    # 記憶體修復：val + visual gate（VGG / 多 case rollout）峰值會把 caching
                    # allocator 撐到近滿，殘留 reserved cache 使後續 train step 近-OOM，
                    # cudaMalloc 反覆 retry/sync 造成 step time 大幅拉長。清 cache 讓 train
                    # 以正常工作集重啟。
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # ── Early stopping monitor 提取（支援 vg_*）─────
                    # 跑就必須解得出 monitor（解不出直接 KeyError 並列可用 key），
                    # 避免組態錯誤被靜默跳過。整段分派抽成 `early_stop_dispatch.py`
                    # 的純函式，與 TrainerBase 共用同一份，可在不載 GPU/dataset 的
                    # 情況下測試。
                    monitor = self._cfg.early_stop.monitor
                    if dispatch_early_stop(
                        es_cfg=self._cfg.early_stop,
                        step=self._step,
                        cold_metrics=cold_metrics,
                        warm_metrics=warm_metrics,
                        vg_metrics=_vg,
                        early_stop=self._early_stop,
                        state_dict_fn=self._model.state_dict,
                        write_jsonl=self._write_jsonl,
                    ):
                        self._should_stop = True
                        if (
                            self._cfg.early_stop.restore_best
                            and self._early_stop.best_state is not None
                        ):
                            self._model.load_state_dict(self._early_stop.best_state)
                            logger.info(
                                "[EarlyStopping] 已載入最佳權重（step=%d, %s=%.6f）",
                                self._early_stop.best_step,
                                monitor,
                                self._early_stop.best_val,
                            )
                        self._save_checkpoint(tag="early_stop_best")

                    # ── Training plots ────────────────────────────────
                    if do_visual:
                        try:
                            from ...plots.train import plot_all as _plot_all

                            self._submit_background_job(
                                "plot_all", _plot_all, str(self._output_dir), self._step
                            )
                        except Exception as _plot_err:
                            logger.debug(
                                "Plotting skipped at step %d: %s", self._step, _plot_err
                            )

                # ── Checkpoint ───────────────────────────────────────
                if self._step % self._cfg.ckpt_interval == 0:
                    ckpt_path = self._save_checkpoint()
                    # ckpt 評估圖一律 in-process；`async_eval_generation` 只控制
                    # 背景 I/O executor（npz 壓縮寫盤），不切換評估路徑。
                    self._generate_checkpoint_images()
                    # rollout/序列 vis 走 in-process `_rollout_vis`：視覺是
                    # 一票否決判準，不能靠子行程 log 決定有沒有輸出。
                    if do_visual:
                        self._save_rollout_vis(
                            self._val_cold_loader, self._val_warm_loader
                        )

            # 正常完成或 early stop：儲存 final checkpoint
            if self._should_stop:
                logger.info(
                    "訓練提前結束（EarlyStopping）— %s，step=%d，best val=%.6f @ step=%d",
                    self._early_stop.trigger_reason,
                    self._step,
                    self._early_stop.best_val,
                    self._early_stop.best_step,
                )
            else:
                self._save_checkpoint(tag="final")
                logger.info("訓練完成，共 %d steps", self._step)
            self._write_run_meta(t_train_start)

        finally:
            pbar.close()
            _log_ctx.__exit__(None, None, None)
            if self._metrics_fp is not None:
                self._metrics_fp.flush()  # flush 殘留 buffer
                self._metrics_fp.close()
                self._metrics_fp = None
            if self._bg_executor is not None:
                self._bg_executor.shutdown(wait=True)
                self._bg_executor = None

    def _write_run_meta(self, t_start: float) -> None:
        """訓練結束時寫 ``run_meta.json``（實驗對齊摘要）。

        獨立於 metrics.json 的理由：`metrics.json` 是 line-delimited 事件流，
        「這次訓練實際跑了幾步／早停於何處／最佳 val 在哪／花了多久／什麼硬體」
        散在數千行裡，事後要重建就得重寫一支 parser；本函式提供一個**單一、
        扁平、一眼可讀**的落點。

        參數量記兩個數字：一個是 ``requires_grad`` 的 trainable 參數量，另一個是
        ``state_dict`` 全部 entry 的 ``numel`` 總和（含 ``register_buffer`` 的固定
        張量），兩者不同，鍵名分別標明。

        Args:
            t_start: 訓練迴圈起始的 ``time.time()``。
        """
        import json as _json
        import platform
        import subprocess
        import time as _time

        cfg = self._cfg
        m = self._model
        try:
            _git = (
                subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=str(REPO_ROOT),
                    timeout=10,
                ).stdout.strip()
                or None
            )
        except Exception:
            _git = None

        meta = {
            "run_id": Path(cfg.output_dir).name,
            "config_path": getattr(cfg, "_source_yaml", None),
            "git_commit": _git,
            "seed": int(cfg.seed),
            # ── §三 對齊表欄位 ──
            "batch_size": int(cfg.batch_size),
            "mean_chain_batch_size": int(cfg.mean_chain_batch_size),
            "rollout_k": {
                "mean_rollout_k": int(cfg.mean_rollout_k),
                "k_min": int(cfg.mean_rollout_k_min),
                "k_max": int(cfg.mean_rollout_k_max),
                "grad_tail": int(cfg.mean_grad_tail),
            },
            "grad_checkpoint": bool(cfg.model.grad_checkpoint),
            "max_steps": int(cfg.max_steps),
            "steps_completed": int(self._step),
            "lr_schedule": {
                "schedule": str(cfg.lr_schedule),
                "lr": float(cfg.lr),
                "lr_min": float(cfg.lr_min),
                "warmup_steps": int(cfg.warmup_steps),
            },
            "optimizer": {
                "name": "AdamW",
                "weight_decay": float(cfg.weight_decay),
                "grad_clip_norm": float(cfg.grad_clip_norm),
            },
            "staged_curriculum": {
                "phase1_end": cfg.curriculum.phase1_end,
                "phase2_end": cfg.curriculum.phase2_end,
                "gt_phase1_end": float(cfg.mean_gt_phase1_end),
                "gt_phase2_end": float(cfg.mean_gt_phase2_end),
            },
            "damage_training": {
                "enabled": bool(cfg.mean_pool_enabled)
                and float(cfg.mean_pool_damage_frac) > 0,
                "damage_frac": float(cfg.mean_pool_damage_frac),
                "damage_mode": str(cfg.mean_pool_damage_mode),
                "damage_size_frac": float(cfg.mean_pool_damage_size_frac),
            },
            "experience_pool": {
                "enabled": bool(cfg.mean_pool_enabled),
                "size": int(cfg.mean_pool_size),
            },
            "tbptt_truncation": int(cfg.mean_grad_tail),
            "mean_ablation_arm": str(cfg.mean_ablation_arm),
            "mrp": (
                {
                    "enabled": bool(cfg.model.mrp.enabled),
                    "hormone_dim": int(cfg.model.mrp.hormone_dim),
                    "num_learnable": int(cfg.model.mrp.num_learnable),
                    "use_topo": bool(cfg.model.mrp.use_topo),
                    "use_stream": bool(cfg.model.mrp.use_stream),
                    "use_trans": bool(cfg.model.mrp.use_trans),
                }
                if getattr(cfg.model, "mrp", None) is not None
                else None
            ),
            # ── 參數量：兩個數字，鍵名標明語義 ──
            "params_trainable": sum(
                p.numel() for p in m.parameters() if p.requires_grad
            ),
            "params_state_dict_total": sum(v.numel() for v in m.state_dict().values()),
            # ── 訓練結果 ──
            "early_stop": {
                "enabled": bool(cfg.early_stop.enabled),
                "triggered": bool(self._should_stop),
                "trigger_reason": self._early_stop.trigger_reason,
                "best_val": self._early_stop.best_val,
                "best_step": self._early_stop.best_step,
            },
            "wall_clock_sec": round(_time.time() - t_start, 1),
            # ── 環境 ──
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "determinism_mode": str(cfg.determinism_mode),
            "cudnn_benchmark": bool(cfg.cudnn_benchmark),
            # ── 軸序聲明（清單 §三 陷阱 2）──
            "axis_order_note": (
                "本檔與 zarr 的空間軸序為 (C, H, W) = (9, 360, 640)；"
                "論文正文一律寫「寬 × 高」= 640 × 360，兩者相反，引用時請標明。"
            ),
            "dataset": {
                "train_zarr": [str(z) for z in cfg.zarr_paths],
                "val_zarr": [str(z) for z in cfg.val_zarr_paths],
                "stats": [str(z) for z in cfg.stats_paths],
            },
        }
        out = self._output_dir / "run_meta.json"
        out.write_text(
            _json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("run_meta.json 已寫出：%s", out)

    def _renorm_ic_mphy_traj(self, traj_raw: torch.Tensor, batch: dict) -> torch.Tensor:
        """對多幀 trajectory [B, K, 9, H, W] 套用 source→global renorm（複用單幀路徑）。

        多 source 時逐幀 renorm；單 source（無 source_field_normalizers）為 identity。
        """
        if not self._source_field_normalizers:
            return traj_raw
        B, K, C, H, W = traj_raw.shape
        out = traj_raw.clone()
        for kk in range(K):
            out[:, kk] = self._renorm_ic_mphy(traj_raw[:, kk], batch)
        return out

    # ------------------------------------------------------------------
    # RP RollingPool 訓練步（state-continuous walk + short-tail BPTT）
    # ------------------------------------------------------------------

    def _maybe_train_noise(
        self, physics: torch.Tensor, fluid_mask: torch.Tensor
    ) -> torch.Tensor:
        """訓練時對 IC physics 加 N(0,σ²)（fluid 區）。eval / σ=0 時不動。

        教模型「偏離→拉回」→ GT 軌跡 manifold 變 attractor（limit-cycle 無定點靠此拘束）。
        """
        sigma = float(self._cfg.train_noise_sigma or 0.0)
        if sigma > 0.0 and self._model.training:
            physics = physics + sigma * torch.randn_like(physics) * fluid_mask
        return physics

    def _temporal_ke_loss(
        self,
        state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        g_vec_norm: Optional[torch.Tensor],
        cumul_start_ctu: Optional[torch.Tensor],
        l_char_px: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """L_temporal（solver 2CTU KE 統計穩態）。

        從監督起點 ``state`` 做長 rollout，**逐步單步推進並增量算 domain-mean KE**（不 stack
        整段 [B,R,9,H,W]，省顯存 → 16GB 卡可行）：preroll ``R-G`` 步 ``no_grad``（detached）
        + grad_tail ``G`` 步帶梯度，收集 ke_t [B,R] → ``two_window_ke_loss``。外推區固定步長
        ``temporal.delta``（uniform），ctu_seq = cumul_start + step·Δ。grad 僅經尾段 KE 回流
        （preroll detached）。

        Returns:
            scalar Tensor（temporal.enabled=False 時不呼叫此函式）。
        """
        tc = self._cfg.temporal
        R = int(tc.r_total)
        G = min(int(tc.grad_tail), R)
        pre = R - G
        delta = float(tc.delta)
        B = state.shape[0]
        dev = state.device
        uses_dctu = self._uses_per_step_delta_ctu()
        wants_prior = getattr(self._model, "wants_prior_ctx", False)
        fnorm = self._get_field_norm()
        prior_vel = self._get_prior_vel_norm() if wants_prior else None

        def _one_step(st: torch.Tensor, j: int) -> tuple[torch.Tensor, torch.Tensor]:
            """單步推進；回傳 (new_state, phys[B,9,H,W])。j = step index（0-based）。"""
            dt = (
                torch.full((B, 1), delta, device=dev, dtype=torch.float32)
                if uses_dctu
                else None
            )
            kw = {"delta_ctu_seq": dt, "g_vec_norm": g_vec_norm}
            if wants_prior and cumul_start_ctu is not None and l_char_px is not None:
                cc = (
                    cumul_start_ctu.view(B, 1) + float(j) * delta
                )  # [B,1] step-start ctu
                kw["prior_ctx"] = {
                    "current_ctu_raw_seq": cc,
                    "l_char_px": l_char_px,
                    "prior_vel_norm": prior_vel,
                }
            new_state, traj = self._model(
                st, cond_norm, fluid_mask, 1, return_trajectory=True, **kw
            )
            return new_state, traj[-1]

        ke_list: list[torch.Tensor] = []  # 每元素 [B]
        # ── preroll（no_grad，逐步丟棄 frame，只留 KE）──────────────────
        with torch.no_grad():
            for j in range(pre):
                state, phys = _one_step(state, j)
                ke_list.append(domain_mean_ke(phys, fluid_mask, fnorm).detach())
            state = state.detach()
        # ── grad tail（梯度回流；僅 G 步圖，記憶體 ≈ 監督 k）──────────────
        for jj in range(G):
            state, phys = _one_step(state, pre + jj)
            ke_list.append(domain_mean_ke(phys, fluid_mask, fnorm))

        ke_t = torch.stack(ke_list, dim=1)  # [B, R]
        # ── ctu_seq：每步 END ctu（raw）= cumul_start + (i+1)·Δ ──────────
        i_end = torch.arange(1, R + 1, device=dev, dtype=torch.float32)  # [R]
        if cumul_start_ctu is not None:
            ctu_seq = cumul_start_ctu.view(B, 1) + i_end.view(1, R) * delta
        else:
            ctu_seq = i_end.view(1, R).expand(B, R) * delta
        return two_window_ke_loss(ke_t, ctu_seq, window_ctu=tc.window_ctu, eps=tc.eps)

    def _jacobian_reg_loss(
        self,
        state: torch.Tensor,
        cond_norm: torch.Tensor,
        fluid_mask: torch.Tensor,
        g_vec_norm: Optional[torch.Tensor],
        dt_first: Optional[torch.Tensor],
        prior_ctx: Optional[dict],
        fx_ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """單步 Jacobian 放大率懲罰（真收縮）。

        對 ``state`` 的 physics 加擾動 ε，比 ``f(x+ε)`` 與 ``f(x)`` 的放大率 →
        ``jacobian_amp_loss``（全域 relu(ratio−γ) 或 JAWS per-pixel）。jac_reg_weight=0 時不呼叫。

        記憶體優化：
            - ``fx_ref`` 給定時 **重用既算好的 f(x)**（如主監督 rollout 第一步輸出，已 detach）
              → **只算 1 次 grad forward（f(x+ε)）** 而非 2 次。fx_ref 為 reference（不需 grad，
              對它無梯度；梯度經 f(x+ε) 回參數，仍是有效的收縮懲罰）。
            - ``fx_ref=None`` 時退回原行為（自算 f(x) + f(x+ε) 兩次 forward）。

        Returns:
            scalar Tensor。
        """
        cfg = self._cfg
        sd, pd = cfg.model.static_dim, cfg.model.physics_dim
        phys = state[:, sd : sd + pd]
        eps = float(cfg.jac_eps) * torch.randn_like(phys) * fluid_mask
        if cfg.model.hidden_dim > 0:
            state_eps = torch.cat(
                [state[:, :sd], phys + eps, state[:, sd + pd :]], dim=1
            )
        else:
            state_eps = torch.cat([state[:, :sd], phys + eps], dim=1)
        kw = {"delta_ctu_seq": dt_first, "g_vec_norm": g_vec_norm}
        if prior_ctx is not None:
            kw["prior_ctx"] = prior_ctx
        if fx_ref is None:
            _, tx = self._model(
                state, cond_norm, fluid_mask, 1, return_trajectory=True, **kw
            )
            fx = tx[-1]
        else:
            fx = fx_ref  # 重用主 rollout 的 f(x)（detached reference，省一次 forward）
        _, txe = self._model(
            state_eps, cond_norm, fluid_mask, 1, return_trajectory=True, **kw
        )
        return jacobian_amp_loss(
            fx,
            txe[-1],
            eps,
            fluid_mask,
            gamma=float(cfg.jac_gamma),
            gamma_low=float(cfg.jac_gamma_low),  # two-sided 死場下界
            jaws_enabled=bool(cfg.jaws_enabled),
            jaws_lambda_c=float(cfg.jaws_lambda_c),
        )

    def _validate_split_mean(self, loader: DataLoader) -> dict[str, float]:
        """mean gate val：mean_module 專用定量評估（取代 monolithic 判準）。

        四項（指標實作於 src/lbmnca/nca/tes/eval/mean_gate_eval.py）：
          1. mean_mse / mean_rel_l2：fluid 區 vs mean_flow_post_conv
          2. mean_jx_ssim：SSIM（target=mean，jx 通道）
          3. ke_converged / ke_ratio：2k 步 rollout KE 兩窗收斂
          4. mean_mse_2k / mean_drift_2k：k→2k self-rollout recovery

        IC=parent_mean_renorm、rollout=2×mean_rollout_k（逐步推進省 VRAM）；
        與訓練 step 對齊：source local→global renorm 後才評。

        Returns:
            metrics dict（跨 batch 平均；loader 無 pair 欄位時回空 dict）。
        """
        self._model.eval()
        # ★逐 batch 收集 (batch_size, metrics, pool_terms)，聚合交純函式。
        records: list[tuple[int, dict[str, float], dict[str, float]]] = []
        n_batches = 0
        rollout_k = self._cfg.mean_rollout_k

        with torch.inference_mode():
            for batch in loader:
                if "mean_flow_post_conv" not in batch:
                    continue
                if "parent_mean_renorm" not in batch:
                    continue
                static = batch["static"].to(self._device)
                target_mean = batch["mean_flow_post_conv"].to(self._device).float()
                ic_physics = batch["parent_mean_renorm"].to(self._device).float()
                g_vec = batch["g_vec"].to(self._device)
                p_vec = batch["p_vec"].to(self._device)

                # source(local)→global norm，與 _mean_module_train_step 對齊
                static, ic_physics, _tgt5 = self._renorm_source_to_global(
                    static, ic_physics, target_mean.unsqueeze(1), batch
                )
                target_mean = _tgt5.squeeze(1)

                fluid_mask = self._get_fluid_mask(static)
                cond_norm = self._build_cond_norm(g_vec, p_vec)
                g_vec_norm = self._build_g_vec_norm(g_vec)
                # val 的 IC 協定與 ke_anchor 需與訓練 step 一致，故改走三態協定
                # （apply_ic_mask + ke_anchor）唯一實作，而非裸 init_state。
                state = protocol.initial_state(
                    self._model, self._cfg, static, ic_physics, fluid_mask
                )
                _val_ke_anchor = protocol.resolve_ke_anchor(
                    self._cfg, state, fluid_mask
                )

                sd = self._model._cfg.static_dim
                pd = self._model._cfg.physics_dim
                ke_curve: list[float] = []
                pred_at_k: Optional[torch.Tensor] = None
                with torch.amp.autocast("cuda", enabled=self._cfg.use_amp):
                    for step in range(1, 2 * rollout_k + 1):
                        state = self._model(
                            state,
                            cond_norm,
                            fluid_mask,
                            1,
                            return_trajectory=False,
                            g_vec_norm=g_vec_norm,
                            ke_anchor=_val_ke_anchor,
                        )
                        phys = state[:, sd : sd + pd].float()
                        ke_curve.append(ke_norm(phys, fluid_mask))
                        if step == rollout_k:
                            pred_at_k = phys.clone()
                pred_at_2k = state[:, sd : sd + pd].float()

                m = mean_gate_metrics(pred_at_k, target_mean, fluid_mask)
                m.update(ke_curve_convergence(ke_curve))
                mse_2k = mean_gate_metrics(pred_at_2k, target_mean, fluid_mask)[
                    "mean_mse"
                ]
                m.update(self_rollout_drift(m["mean_mse"], mse_2k))

                # 本函式是 canonical mean 驗證路徑（分派見 `_run_validation` 的
                # mean_module_enabled 分支）。
                # 加權規則：
                #   - 一般指標：以 batch size 加權（未加權批次平均會使尾批被高估）
                #   - mean_mse / mean_rel_l2：改用 `_pool_*` 原始累加量精確池化
                #     （前者分母是 n_fluid，後者是比值，兩者乘 B 都不對）
                # 聚合語義全部委派給 `validation/aggregate.py` 的純函式，可用 dict
                # fixture 完整測試（tests/nca/tes/test_val_aggregation.py），不必載 ckpt。
                # 2k 的平方誤差一併交給聚合器，使 mean_mse_2k 與 mean_mse 走同一種
                # 估計量（精確池化）、且 mean_drift_2k 於聚合後才相減——否則 split
                # 層級的恆等式 `drift == mse_2k - mse` 不成立，讀者自行相減會得到
                # 不同的數。
                _pool = mean_gate_pool_terms(pred_at_k, target_mean, fluid_mask)
                _pool[POOL_KEY_2K] = mean_gate_pool_terms(
                    pred_at_2k, target_mean, fluid_mask
                )["sq_err"]
                records.append((int(fluid_mask.shape[0]), dict(m), _pool))
                n_batches += 1

        self._model.train()
        if n_batches == 0:
            logger.warning(
                "[val-mean] loader 批次皆缺 pair 欄位"
                "（mean_flow_post_conv/parent_mean_renorm），無法評估"
            )
            return {}
        return aggregate_val_metrics(records)

    # ------------------------------------------------------------------
    # rollout 序列 + 誤差曲線視覺化（vis_interval）
    # ------------------------------------------------------------------

    def _save_rollout_vis(
        self,
        cold_loader: Optional[DataLoader],
        warm_loader: Optional[DataLoader],
        n_err_samples: int = 4,
        n_seq_cols: int = 7,
    ) -> None:
        """vis_interval rollout 視覺化：mean_eval_sequence + rollout_error_lines。

        mean rollout：IC=parent node 的 mean_flow → 收斂 → child node 的 mean_flow。

        輸出至 ``{output_dir}/plots/``（覆蓋 latest + step 標記）。

        Args:
            cold_loader / warm_loader: val DataLoader。
            n_err_samples: 誤差曲線取樣 case 數（每條線一個 case）。
            n_seq_cols: 序列圖中間 rollout 取樣欄數。
        """
        import numpy as np

        from ..eval.visual_metrics import struct_vel_rgb_ssim
        from ..visualization.rollout_vis import (
            plot_eval_sequence,
            plot_rollout_error_lines,
        )
        from lbmnca.nca.utils.FieldConverter import FieldConverter

        fn = self._get_field_norm()
        if fn is None:
            return
        converter = FieldConverter(fn)
        sd = self._model._cfg.static_dim
        pd = self._model._cfg.physics_dim
        plots_dir = self._output_dir / "plots"

        def _vel_vort(phys: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
            """phys [B,9,H,W](norm) → (vel_mag, vorticity) numpy（取 b=0）。"""
            vphy = converter.toVphy(phys.float(), is_norm=True)
            return (
                vphy["vel_mag"][0].cpu().numpy(),
                vphy["vorticity"][0].cpu().numpy(),
            )

        def _mae_mse(
            pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor
        ) -> tuple[float, float]:
            m = mask.float().expand_as(pred)
            diff = (pred - tgt) * m
            denom = m.sum().clamp_min(1.0)
            mae = float(diff.abs().sum().item() / denom.item())
            mse = float((diff * diff).sum().item() / denom.item())
            return mae, mse

        loader = self._val_warm_loader or self._val_cold_loader
        if loader is None:
            return

        self._model.eval()
        try:
            self._rollout_vis_mean(
                loader,
                converter,
                fn,
                sd,
                pd,
                plots_dir,
                n_err_samples,
                n_seq_cols,
                _vel_vort,
                _mae_mse,
                struct_vel_rgb_ssim,
                plot_eval_sequence,
                plot_rollout_error_lines,
                np,
            )
        finally:
            self._model.train()

    @torch.inference_mode()
    def _rollout_vis_mean(
        self,
        loader,
        converter,
        fn,
        sd,
        pd,
        plots_dir,
        n_err_samples,
        n_seq_cols,
        _vel_vort,
        _mae_mse,
        struct_vel_rgb_ssim,
        plot_eval_sequence,
        plot_rollout_error_lines,
        np,
    ) -> None:
        # ★共用核心：roll + masked MAE/SSIM + 收斂判定 + 組 cols。
        # 這是唯一路徑（外部 async 腳本已移除，見 train() 內註解）。
        from lbmnca.nca.tes.eval.mean_rollout_vis import build_mean_seq_cols
        from lbmnca.nca.plots.train.mean_eval_sequence import plot_eval_sequence_multi

        rollout_k = int(self._cfg.mean_rollout_k)
        n_steps = 100
        n_seq_samples = min(
            int(n_err_samples), 4
        )  # 序列圖只收前 N case，誤差曲線收全部
        curves: list[dict] = []
        seq_samples: list[dict] = []
        n_done = 0

        for batch in loader:
            if n_done >= n_err_samples:
                break
            if "mean_flow_post_conv" not in batch:
                continue
            if "parent_mean_renorm" not in batch:
                continue
            static = batch["static"].to(self._device)
            target_mean = batch["mean_flow_post_conv"].to(self._device).float()
            ic_physics = batch["parent_mean_renorm"].to(self._device).float()
            g_vec = batch["g_vec"].to(self._device)
            p_vec = batch["p_vec"].to(self._device)

            static, ic_physics, _tgt5 = self._renorm_source_to_global(
                static, ic_physics, target_mean.unsqueeze(1), batch
            )
            target_mean = _tgt5.squeeze(1)
            fluid_mask = self._get_fluid_mask(static)  # 二值 mask（正確 SDF threshold）
            cond_norm = self._build_cond_norm(g_vec, p_vec)
            g_vec_norm = self._build_g_vec_norm(g_vec)
            # vis 序列亦走三態協定（apply_ic_mask + ke_anchor），否則面板畫的是
            # 與訓練/部署不同的算子。
            state0 = protocol.initial_state(
                self._model, self._cfg, static, ic_physics, fluid_mask
            )
            _vis_anchor = protocol.resolve_ke_anchor(self._cfg, state0, fluid_mask)

            cols, curve, solid = build_mean_seq_cols(
                roll_step=lambda s: self._model(
                    s,
                    cond_norm,
                    fluid_mask,
                    1,
                    return_trajectory=False,
                    g_vec_norm=g_vec_norm,
                    ke_anchor=_vis_anchor,
                ),
                state0=state0,
                target_mean=target_mean,
                ic_physics=ic_physics,
                fluid_mask=fluid_mask,
                sd=sd,
                pd=pd,
                field_norm=fn,
                converter=converter,
                n_steps=n_steps,
                n_seq_cols=n_seq_cols,
                ctu_per_step=protocol.NOMINAL_DELTA_CTU_PER_STEP,
            )
            curves.append(curve)
            if n_done < n_seq_samples:
                seq_samples.append(
                    {"re": float(p_vec[0, 0].item()), "cols": cols, "solid": solid}
                )
            n_done += 1

        if not curves:
            return
        step = self._step
        if seq_samples:
            plot_eval_sequence_multi(
                seq_samples,
                f"mean_eval_sequence (multi-Re)  |  step {step}",
                plots_dir / "mean_eval_sequence_latest.png",
            )
        plot_rollout_error_lines(
            curves,
            plots_dir / "rollout_error_latest.png",
            f"mean rollout error  |  step {step}",
            transition_ctu=float(rollout_k),
        )

    @torch.inference_mode()
    @torch.inference_mode()
    def _save_pool_vis(self) -> None:
        """每 vis_interval：plot pool 首 8 / 末 8 slot 的 vel_mag（col8×row2，固定軸 [0,0.2]）。

        監控 Experience Pool 偏離分布演化（§18.1 pool 健康度）。僅 mean_pool_enabled 時作用。
        """
        if not self._cfg.mean_pool_enabled:
            return
        pool = self._step_mean._meanpool
        if pool is None or not pool.ready:
            return
        peek = pool.peek(8, 8)
        if peek is None:
            return
        from lbmnca.nca.utils.FieldConverter import FieldConverter
        from ..visualization.rollout_vis import plot_pool_vel_grid

        fn = self._get_field_norm()
        if fn is None:
            return
        conv = FieldConverter(fn)
        sd = self._model._cfg.static_dim
        pd = self._model._cfg.physics_dim

        def _vels(states):
            out = []
            for i in range(states.shape[0]):
                phys = states[i : i + 1, sd : sd + pd].float()
                vphy = conv.toVphy(phys, is_norm=True)
                out.append(vphy["vel_mag"][0].cpu().numpy())
            return out

        plot_pool_vel_grid(
            _vels(peek["first"]),
            _vels(peek["last"]),
            None,
            self._output_dir / "plots" / "pool_vel_latest.png",
            f"pool vel (first8 / last8)  |  step {self._step}",
            vmax=0.20,
        )

    def _save_val_fields(
        self,
        cold_loader: Optional[DataLoader],
        warm_loader: Optional[DataLoader],
        n_cold: int = 4,
        n_warm: int = 4,
    ) -> None:
        """蒐集 cold / warm split 的 val field 快照並存成 npz（供 field_compare 圖）。

        每個 case：隨機 t0 + 隨機 n 步 rollout（n ∈ [1, k_max]）。
        輸出至 ``{output_dir}/val_fields_latest.npz``：
            {cold,warm}_{target,ic,pred}_vor: [N, H, W] float32 — vorticity（lattice）
            {cold,warm}_fluid_mask:           [N, H, W] float32
            {cold,warm}_t0:                   [N] int32
            {cold,warm}_n_steps:              [N] int32
            step:                             scalar int

        Args:
            cold_loader: cold val DataLoader（None 時略過該 split）。
            warm_loader: warm val DataLoader（None 時略過該 split）。
            n_cold: cold split 蒐集 case 數。
            n_warm: warm split 蒐集 case 數。
        """
        import numpy as np

        def _vor(raw_np: np.ndarray) -> np.ndarray:
            # velocity vorticity = ∂(jy/ρ)/∂x − ∂(jx/ρ)/∂y  [B, H, W]
            rho = np.maximum(raw_np[:, 0], 1e-6)
            vel_x = raw_np[:, 3] / rho  # jx/rho = ux
            vel_y = raw_np[:, 5] / rho  # jy/rho = uy
            return np.gradient(vel_y, axis=2) - np.gradient(vel_x, axis=1)

        k_max = self._get_val_k()
        fn = self._get_field_norm()
        # 視覺取樣 (t0, n) 專用 RNG，每次呼叫重建 → 啟用
        # cfg.rng_stream_isolation 時各 ckpt 看到同一組 (t0, n)，便於跨 step 比對；
        # 停用（預設）時回傳全域 random 模組，抽樣序列與舊版逐字相同。
        _vis_rng = rng_streams.fresh_stream("vis")

        def _collect(loader: Optional[DataLoader], n_cases: int) -> list[dict]:
            rows: list[dict] = []
            if loader is None:
                return rows
            with torch.inference_mode():
                for batch in loader:
                    if len(rows) >= n_cases:
                        break
                    static = batch["static"].to(self._device)
                    g_vec = batch["g_vec"].to(self._device)
                    p_vec = batch["p_vec"].to(self._device)

                    # ★v4 mean-only：batch 無 `trajectory`（v4 給的是單幀 IC ＋ 單幀
                    #   target，不是時間序列）。原本無條件讀 batch["trajectory"] 會
                    #   KeyError，使整條視覺輸出靜默消失——視覺是本專案的一票否決
                    #   判準，不能靠一個被 except 吞掉的 KeyError 決定有沒有圖。
                    if (
                        "mean_flow_post_conv" not in batch
                        or "parent_mean_renorm" not in batch
                    ):
                        continue
                    rollout_k = self._cfg.mean_rollout_k
                    n = rollout_k
                    t0 = 0  # warm-transfer 無 t0 語義；固定 0 以維持下游欄位形狀
                    mean_fp = batch["mean_flow_post_conv"].to(self._device).float()
                    ic_raw = batch["parent_mean_renorm"].to(self._device).float()
                    static, ic_phys, _mf_traj = self._renorm_source_to_global(
                        static, ic_raw, mean_fp.unsqueeze(1), batch
                    )
                    B = static.shape[0]
                    tgt_phys = _mf_traj[:, 0]
                    fluid_mask = self._get_fluid_mask(static)
                    # ★三態協定：IC mask 與 ke_anchor 皆由 protocol 唯一實作產生。
                    #   裸 init_state + 不傳 ke_anchor 會讓 ke_anchor_mode='ceiling'
                    #   的 forward 直接 raise，整張圖被吞掉。
                    state = protocol.initial_state(
                        self._model, self._cfg, static, ic_phys, fluid_mask
                    )
                    _vis_anchor = protocol.resolve_ke_anchor(
                        self._cfg, state, fluid_mask
                    )
                    cond_norm = self._build_cond_norm(g_vec, p_vec)
                    g_vec_norm = self._build_g_vec_norm(g_vec)
                    _delta_rep_np = None
                    with torch.amp.autocast("cuda", enabled=self._cfg.use_amp):
                        final_state = self._model(
                            state,
                            cond_norm,
                            fluid_mask,
                            rollout_k,
                            return_trajectory=False,
                            g_vec_norm=g_vec_norm,
                            ke_anchor=_vis_anchor,
                        )
                    sd = self._model._cfg.static_dim
                    pd = self._model._cfg.physics_dim
                    pred_phys = final_state[:, sd : sd + pd]  # [B, 9, H, W]

                    tgt_raw_np = fn.denorm_mphy(tgt_phys).cpu().float().numpy()
                    ic_raw_np = fn.denorm_mphy(ic_phys).cpu().float().numpy()
                    pred_raw_np = fn.denorm_mphy(pred_phys).cpu().float().numpy()
                    fm = fluid_mask.squeeze(1).cpu().float().numpy()

                    t_vor = _vor(tgt_raw_np)
                    i_vor = _vor(ic_raw_np)
                    p_vor = _vor(pred_raw_np)

                    # vel_mag from raw lattice units: ux = jx/rho, uy = jy/rho
                    _vm_eps = 1e-6
                    t_vm = np.sqrt(
                        (tgt_raw_np[:, 3] / np.maximum(tgt_raw_np[:, 0], _vm_eps)) ** 2
                        + (tgt_raw_np[:, 5] / np.maximum(tgt_raw_np[:, 0], _vm_eps))
                        ** 2
                    )
                    i_vm = np.sqrt(
                        (ic_raw_np[:, 3] / np.maximum(ic_raw_np[:, 0], _vm_eps)) ** 2
                        + (ic_raw_np[:, 5] / np.maximum(ic_raw_np[:, 0], _vm_eps)) ** 2
                    )
                    p_vm = np.sqrt(
                        (pred_raw_np[:, 3] / np.maximum(pred_raw_np[:, 0], _vm_eps))
                        ** 2
                        + (pred_raw_np[:, 5] / np.maximum(pred_raw_np[:, 0], _vm_eps))
                        ** 2
                    )

                    for b in range(B):
                        if len(rows) >= n_cases:
                            break
                        rows.append(
                            {
                                "target_phys": tgt_raw_np[b],
                                "ic_phys": ic_raw_np[b],
                                "pred_phys": pred_raw_np[b],
                                "target_vor": t_vor[b],
                                "ic_vor": i_vor[b],
                                "pred_vor": p_vor[b],
                                "mask": fm[b],
                                "t0": t0,
                                "n_steps": n,
                                "delta_ctu": (
                                    float(_delta_rep_np[b])
                                    if _delta_rep_np is not None
                                    else float("nan")
                                ),
                                "target_vel": t_vm[b],
                                "ic_vel": i_vm[b],
                                "pred_vel": p_vm[b],
                            }
                        )
            return rows

        self._model.eval()
        cold_rows = _collect(cold_loader, n_cold)
        warm_rows = _collect(warm_loader, n_warm)
        self._model.train()

        if not cold_rows and not warm_rows:
            return

        import numpy as np

        def _stack(rows: list[dict], key: str, dtype=np.float32) -> np.ndarray:
            return (
                np.stack([r[key] for r in rows]).astype(dtype)
                if rows
                else np.empty((0,), dtype=dtype)
            )

        def _cat(rows_a, rows_b, key, dtype=np.float32):
            parts = [_stack(r, key, dtype) for r in (rows_a, rows_b) if r]
            return np.concatenate(parts) if parts else np.empty((0,), dtype=dtype)

        payload = dict(
            # cold split — vorticity
            cold_target_vor=_stack(cold_rows, "target_vor"),
            cold_ic_vor=_stack(cold_rows, "ic_vor"),
            cold_pred_vor=_stack(cold_rows, "pred_vor"),
            cold_fluid_mask=_stack(cold_rows, "mask"),
            cold_t0=_stack(cold_rows, "t0", np.int32),
            cold_n_steps=_stack(cold_rows, "n_steps", np.int32),
            cold_delta_ctu=_stack(cold_rows, "delta_ctu"),
            # cold split — velocity (raw lattice)
            cold_target_vel=_stack(cold_rows, "target_vel"),
            cold_ic_vel=_stack(cold_rows, "ic_vel"),
            cold_pred_vel=_stack(cold_rows, "pred_vel"),
            # warm split — vorticity
            warm_target_vor=_stack(warm_rows, "target_vor"),
            warm_ic_vor=_stack(warm_rows, "ic_vor"),
            warm_pred_vor=_stack(warm_rows, "pred_vor"),
            warm_fluid_mask=_stack(warm_rows, "mask"),
            warm_t0=_stack(warm_rows, "t0", np.int32),
            warm_n_steps=_stack(warm_rows, "n_steps", np.int32),
            warm_delta_ctu=_stack(warm_rows, "delta_ctu"),
            # warm split — velocity (raw lattice)
            warm_target_vel=_stack(warm_rows, "target_vel"),
            warm_ic_vel=_stack(warm_rows, "ic_vel"),
            warm_pred_vel=_stack(warm_rows, "pred_vel"),
            target_phys=_cat(cold_rows, warm_rows, "target_phys"),
            ic_phys=_cat(cold_rows, warm_rows, "ic_phys"),
            pred_phys=_cat(cold_rows, warm_rows, "pred_phys"),
            # backward-compat keys (union of cold+warm)
            target_vel=_cat(cold_rows, warm_rows, "target_vel"),
            ic_vel=_cat(cold_rows, warm_rows, "ic_vel"),
            pred_vel=_cat(cold_rows, warm_rows, "pred_vel"),
            fluid_mask=_cat(cold_rows, warm_rows, "mask"),
            target_vor=_cat(cold_rows, warm_rows, "target_vor"),
            ic_vor=_cat(cold_rows, warm_rows, "ic_vor"),
            pred_vor=_cat(cold_rows, warm_rows, "pred_vor"),
            step=np.array(self._step, dtype=np.int64),
            mean_module_mode=np.array(self._cfg.mean_module_enabled, dtype=bool),
        )
        # [perf] 111MB savez_compressed（zlib + 寫盤）卸到背景 executor：
        # payload 全為已 materialize 的 CPU numpy（_stack/_cat 產生新陣列，無 GPU 別名），
        # 純 I/O → 不阻塞訓練。executor 不存在（async_eval_generation=false）時退回同步。
        _npz_path = str(self._output_dir / "val_fields_latest.npz")
        if self._bg_executor is not None:
            self._bg_executor.submit(np.savez_compressed, _npz_path, **payload)
        else:
            np.savez_compressed(_npz_path, **payload)

    # ------------------------------------------------------------------
    # Visual gate metrics
    # ------------------------------------------------------------------

    def _compute_val_visual_gate(
        self,
        cold_loader: Optional[DataLoader],
        warm_loader: Optional[DataLoader],
        n_cases: int = 8,
    ) -> dict[str, float]:
        """對抽樣的 val cases 計算 visual gate 指標。

        Args:
            cold_loader: cold val DataLoader（None 時略過）。
            warm_loader: warm val DataLoader（None 時略過）。
            n_cases: 取樣 case 總數。

        Returns:
            metrics dict（key 以 ``vg_`` 為前綴）；visual_metrics 不可用或無 val
            資料時回空 dict。
        """
        try:
            from ..eval.visual_metrics import compute_visual_gate_metrics
        except Exception:
            return {}

        fn = self._get_field_norm()
        k = self._get_val_k()

        def _get_pod_basis():
            basis_path = self._cfg.pod_basis_path
            if not basis_path:
                return None
            if not hasattr(self, "_pod_basis_cache"):
                from ..eval.pod_metrics import load_pod_basis

                self._pod_basis_cache = load_pod_basis(basis_path, self._device)
            return self._pod_basis_cache

        pred_list: list[torch.Tensor] = []
        tgt_list: list[torch.Tensor] = []
        ic_list: list[torch.Tensor] = []
        fm_list: list[torch.Tensor] = []
        meta_list_data: list[dict] = []

        def _gather(loader: Optional[DataLoader], limit: int) -> None:
            if loader is None:
                return
            collected = 0
            with torch.inference_mode():
                for batch in loader:
                    if collected >= limit:
                        break
                    if (
                        "mean_flow_post_conv" not in batch
                        or "parent_mean_renorm" not in batch
                    ):
                        continue
                    static = batch["static"].to(self._device)
                    g_vec = batch["g_vec"].to(self._device)
                    p_vec = batch["p_vec"].to(self._device)

                    B = static.shape[0]
                    take = min(B, limit - collected)
                    if take <= 0:
                        break
                    static = static[:take]
                    g_vec = g_vec[:take]
                    p_vec = p_vec[:take]
                    B = take
                    n_steps = k

                    # ★v4 mean-only：IC = parent node 的 mean_flow，target = child node
                    #   的 mean_flow（皆單幀）。v3 的 trajectory 時間序列不存在，
                    #   turb 的 Reynolds 分解分支已移除。
                    _mean_fp = (
                        batch["mean_flow_post_conv"][:take].to(self._device).float()
                    )
                    ic_phys = (
                        batch["parent_mean_renorm"][:take].to(self._device).float()
                    )
                    static, ic_phys, _tgt = self._renorm_source_to_global(
                        static, ic_phys, _mean_fp.unsqueeze(1), batch
                    )
                    tgt_final = _tgt[:, 0].float()
                    _ic_gate = ic_phys
                    _model_ic = ic_phys

                    fluid_mask = self._get_fluid_mask(static)
                    state = protocol.initial_state(
                        self._model, self._cfg, static, _model_ic, fluid_mask
                    )
                    _vg_anchor = protocol.resolve_ke_anchor(
                        self._cfg, state, fluid_mask
                    )
                    cond_norm = self._build_cond_norm(g_vec, p_vec)
                    g_vec_norm = self._build_g_vec_norm(g_vec)

                    delta_ctu_seq: torch.Tensor | None = None
                    if self._uses_per_step_delta_ctu() and "delta_ctu" in batch:
                        delta_ctu_seq = batch["delta_ctu"][:take, 0:n_steps].to(
                            self._device
                        )

                    # ★cond 只餘 17D（無時間維），逐步 cumul-CTU 條件已移除。
                    _vg_cumul_seq_norm: torch.Tensor | None = None

                    with torch.amp.autocast("cuda", enabled=self._cfg.use_amp):
                        _, pred_traj_list = self._model(
                            state,
                            cond_norm,
                            fluid_mask,
                            n_steps,
                            True,
                            delta_ctu_seq=delta_ctu_seq,
                            g_vec_norm=g_vec_norm,
                            cumul_ctu_norm_seq=_vg_cumul_seq_norm,
                            ke_anchor=_vg_anchor,
                        )
                    pred_final = pred_traj_list[-1].float()

                    # Collect per-sample metadata for stratified analysis
                    B_actual = pred_final.shape[0]
                    case_keys_b = batch.get("case_key", [""] * B_actual)
                    ic_types_b = batch.get("ic_type", ["unknown"] * B_actual)
                    mask_ids_b = batch.get("mask_id", [""] * B_actual)
                    p_cpu = p_vec.cpu()
                    g_cpu = g_vec.cpu()
                    for bi in range(B_actual):
                        meta_list_data.append(
                            {
                                "case_key": (
                                    case_keys_b[bi]
                                    if isinstance(case_keys_b, list)
                                    else ""
                                ),
                                "re": float(p_cpu[bi, 0].item()),
                                "gvec_norm": float(g_cpu[bi].norm().item()),
                                "ic_type": (
                                    ic_types_b[bi]
                                    if isinstance(ic_types_b, list)
                                    else "unknown"
                                ),
                                "mask_id": (
                                    mask_ids_b[bi]
                                    if isinstance(mask_ids_b, list)
                                    else ""
                                ),
                            }
                        )
                    pred_list.append(pred_final)
                    tgt_list.append(tgt_final)
                    ic_list.append(_ic_gate.float())
                    fm_list.append(fluid_mask.float())
                    collected += B_actual

        self._model.eval()
        if cold_loader is not None and warm_loader is not None:
            cold_limit = max(1, n_cases // 2)
            warm_limit = max(1, n_cases - cold_limit)
            _gather(cold_loader, cold_limit)
            _gather(warm_loader, warm_limit)
        else:
            _gather(cold_loader, n_cases)
            _gather(warm_loader, n_cases)
        self._model.train()

        if not pred_list:
            return {}

        pred_cat = torch.cat(pred_list, dim=0)[:n_cases]
        tgt_cat = torch.cat(tgt_list, dim=0)[:n_cases]
        ic_cat = torch.cat(ic_list, dim=0)[:n_cases]
        fm_cat = torch.cat(fm_list, dim=0)[:n_cases]
        meta_cut = meta_list_data[:n_cases]

        pod_basis = _get_pod_basis()
        pod_modes = self._cfg.pod_modes
        gate = compute_visual_gate_metrics(
            pred_cat,
            tgt_cat,
            ic_cat,
            fm_cat,
            fn,
            pod_basis=pod_basis,
            pod_modes=pod_modes,
        )
        gate_named = {f"vg_{key}": val for key, val in gate.items()}

        # Per-case metrics + stratified vis_gate_detailed.json
        try:
            from ..eval.case_metrics import (
                compute_per_case_visual_gate,
                build_vis_gate_detailed,
                save_vis_gate_detailed,
            )

            per_case = compute_per_case_visual_gate(
                [pred_cat[i] for i in range(pred_cat.shape[0])],
                [tgt_cat[i] for i in range(tgt_cat.shape[0])],
                [ic_cat[i] for i in range(ic_cat.shape[0])],
                [fm_cat[i] for i in range(fm_cat.shape[0])],
                meta_cut,
                fn,
                pod_basis=pod_basis,
                pod_modes=pod_modes,
            )
            detailed = build_vis_gate_detailed(
                self._step,
                per_case,
                gate,
                gate_thresholds=self._cfg.gate_thresholds,
            )
            save_vis_gate_detailed(self._output_dir / "eval_output", detailed)
            self._write_eval_summary(detailed)
        except Exception as _pc_err:
            # ★原為 logger.debug → 證據檔寫不出來時完全無聲，使 gate 證據管線死掉
            #   而無人察覺。改 warning：保留 defensive except（eval 工具仍在改動），
            #   但失敗必須看得見。
            logger.warning(
                "[gate-evidence] per-case metrics/證據檔寫入失敗 @ step %d：%s: %s"
                "（eval_output/ 這一步不會產生）",
                self._step,
                type(_pc_err).__name__,
                _pc_err,
            )

        return gate_named

    def _write_eval_summary(self, detailed: dict) -> None:
        """HC-2a: 跨 run 可比的評估摘要（gate 布林 + 分項 + 分層）。

        ★實作已收斂至 `eval.case_metrics.save_eval_summary`（與 `Validator.write_summary`
        共用同一份），本方法只做委派。evidence 帶身分後綴 + `_provenance`。

        Args:
            detailed: `build_vis_gate_detailed` 回傳的 detailed dict。
        """
        from ..eval.case_metrics import save_eval_summary

        save_eval_summary(self._output_dir / "eval_output", detailed)

    # ------------------------------------------------------------------
    # TStart visualisation
    # ------------------------------------------------------------------

    def _generate_checkpoint_images(self) -> None:
        """於 checkpoint 產生評估影像：序列條、HBand 頻譜、分層報告。

        輸出至 ``eval_output/eval_ckpt_{step}/``，並自動清理舊目錄（保留最新 2 份）。
        全程 try/except 包覆：相依不可用或失敗時僅記 warning，不中斷訓練。
        """
        import json as _json
        import numpy as _np

        try:
            from ..eval.eval_cleanup import cleanup_eval_ckpts
            from ..eval.eval_plots import (
                plot_hband_spectrum,
                plot_icswap_demo,
                plot_sequence_strip,
                plot_stratified_report,
            )
            from ...plots.train.debug_hband_gt import plot_hband_debug
            from ...plots.train.report_hband_artifact import plot_hband_artifact

            eval_dir = self._output_dir / "eval_output"
            ckpt_dir = eval_dir / f"eval_ckpt_{self._step:07d}"
            seq_dir = ckpt_dir / "sequence"
            seq_dir.mkdir(parents=True, exist_ok=True)

            fn = self._get_field_norm()
            k_vis = min(20, max(self._get_val_k() * 2, 8))
            # ★ 顯示速度上限需與 SSOT（`stats.json` 的 `vis_range.vel_max`）一致，
            #   不可硬編：偏寬會把像素值壓向 0、使弱結構被系統性淡化，而訓練期
            #   比對圖是「一票否決」的審查核心——偏誤方向對「通過」有利。
            #   改由 normalizer 取（與 eval／deploy 同源），⛔ 不得再硬編。
            _VEL_MAX = float(fn.vel_vis_max)

            def _vm(t: torch.Tensor) -> "_np.ndarray":
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                raw = fn.denorm_mphy(t.to(self._device))
                rho = raw[:, 0].clamp_min(1e-6)
                vm = ((raw[:, 3] / rho).square() + (raw[:, 5] / rho).square()).sqrt()
                return _np.clip((vm / _VEL_MAX).squeeze(0).cpu().numpy(), 0.0, 1.0)

            cases: list[dict] = []

            def _gather_vis(loader: Optional[DataLoader], n_take: int) -> None:
                if loader is None:
                    return
                taken = 0
                with torch.inference_mode():
                    for batch in loader:
                        if taken >= n_take:
                            break
                        # ★v4 mean-only：batch 只有單幀 IC（parent node 的 mean_flow）
                        #   與單幀 target（child node 的 mean_flow）。原本無條件讀
                        #   batch["trajectory"] → KeyError → 每個 ckpt 的評估影像都
                        #   靜默失敗（只在 log 留一行 warning）。
                        if (
                            "parent_mean_renorm" not in batch
                            or "mean_flow_post_conv" not in batch
                        ):
                            continue
                        # 延長到 20 步展示固定點吸引子收斂行為（mean 的 target 與步數無關）
                        n_steps = (
                            max(k_vis, min(20, k_vis))
                            if k_vis >= 20
                            else min(20, max(k_vis, 1))
                        )
                        static = batch["static"].to(self._device)
                        ic_phys = (
                            batch["parent_mean_renorm"][:1].to(self._device).float()
                        )
                        mf = batch["mean_flow_post_conv"][:1].to(self._device).float()
                        # target 為固定點：沿步數軸重複 n_steps 次
                        tgt_slice = (
                            mf.unsqueeze(1).expand(-1, n_steps, -1, -1, -1).contiguous()
                        )
                        g_vec = batch["g_vec"][:1].to(self._device)
                        p_vec_b = batch["p_vec"][:1].to(self._device)
                        static_1 = static[:1]
                        batch_1 = {
                            k: v[:1] if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        static_1, ic_phys, tgt_slice = self._renorm_source_to_global(
                            static_1, ic_phys, tgt_slice, batch_1
                        )
                        fluid_mask = self._get_fluid_mask(static_1)
                        _vis_cumul_seq_norm: torch.Tensor | None = None
                        cond_norm = self._build_cond_norm(g_vec, p_vec_b)
                        g_vec_norm = self._build_g_vec_norm(g_vec)
                        delta_seq: torch.Tensor | None = None
                        if self._uses_per_step_delta_ctu() and "delta_ctu" in batch:
                            delta_seq = batch["delta_ctu"][:1, :n_steps].to(
                                self._device
                            )
                        state = protocol.initial_state(
                            self._model, self._cfg, static_1, ic_phys, fluid_mask
                        )
                        _ck_anchor = protocol.resolve_ke_anchor(
                            self._cfg, state, fluid_mask
                        )
                        with torch.amp.autocast("cuda", enabled=self._cfg.use_amp):
                            _, pred_traj = self._model(
                                state,
                                cond_norm,
                                fluid_mask,
                                n_steps,
                                True,
                                delta_ctu_seq=delta_seq,
                                g_vec_norm=g_vec_norm,
                                cumul_ctu_norm_seq=_vis_cumul_seq_norm,
                                ke_anchor=_ck_anchor,
                            )
                        ck_b = batch.get("case_key", [""])
                        ic_b = batch.get("ic_type", ["unknown"])
                        cases.append(
                            {
                                "pred_traj": [p[0].cpu().float() for p in pred_traj],
                                "tgt_traj": tgt_slice[0].cpu().float(),
                                "ic": ic_phys[0].cpu().float(),
                                "fm": fluid_mask[0].cpu().float(),
                                "case_key": ck_b[0] if isinstance(ck_b, list) else "",
                                "ic_type": (
                                    ic_b[0] if isinstance(ic_b, list) else "unknown"
                                ),
                                "re": float(p_vec_b[0, 0].item()),
                                "n_steps": n_steps,
                            }
                        )
                        taken += 1

            self._model.eval()
            _gather_vis(self._val_cold_loader, 2)
            _gather_vis(self._val_warm_loader, 2)
            self._model.train()

            for case in cases:
                n_st = case["n_steps"]
                preds = case["pred_traj"]
                tgts = case["tgt_traj"]
                ic = case["ic"]
                fm = case["fm"]
                ck = (
                    (case["case_key"] or "case")[:12]
                    .replace("/", "_")
                    .replace("\\", "_")
                )
                mode = case["ic_type"]
                title = f"{case['case_key']} | {mode} | Re={case['re']:.0f}"

                fidxs = _np.unique(
                    _np.round(_np.linspace(0, n_st - 1, 10)).astype(int)
                ).tolist()
                n_tgt_avail = tgts.shape[0]
                gt_f = [_vm(tgts[min(fi, n_tgt_avail - 1)]) for fi in fidxs]
                pred_f = [_vm(preds[min(fi, len(preds) - 1)]) for fi in fidxs]
                labels = [f"t={fi}" for fi in fidxs]
                fm_np = fm.squeeze(0).numpy()

                plot_sequence_strip(
                    gt_f,
                    pred_f,
                    labels,
                    title,
                    seq_dir / f"{mode}_{ck}.png",
                )
                # plot_hband_spectrum(
                #     _vm(preds[-1]), _vm(tgts[-1]), fm_np, title,
                #     ckpt_dir / f"hband_{mode}_{ck}.png",
                # )
                # plot_hband_debug(
                #     _vm(preds[-1]), _vm(tgts[-1]), fm_np, title,
                #     ckpt_dir / f"hband_{mode}_{ck}_debug.png",
                # )
                # plot_hband_artifact(
                #     _vm(preds[-1]), _vm(tgts[-1]), fm_np, title,
                #     ckpt_dir / f"hband_{mode}_{ck}_artifact.png",
                # )
                # if mode.startswith("warm"):
                #     di     = _np.unique(_np.round(_np.linspace(0, n_st - 1, 5)).astype(int)).tolist()
                #     d_gt   = [_vm(tgts[min(i, n_st - 1)])        for i in di]
                #     d_pred = [_vm(preds[min(i, len(preds) - 1)]) for i in di]
                #     plot_icswap_demo(
                #         _vm(ic), d_pred, d_gt,
                #         [f"t={i}" for i in di], title,
                #         ckpt_dir / f"icswap_{ck}.png",
                #     )

            vg_path = eval_dir / "vis_gate_detailed.json"
            if vg_path.exists():
                with open(vg_path, encoding="utf-8") as _f:
                    plot_stratified_report(
                        _json.load(_f), ckpt_dir / "stratified_report.png"
                    )

            cleanup_eval_ckpts(eval_dir, keep_n=2)
            logger.info("Checkpoint eval images → %s", ckpt_dir.name)

        except Exception as _img_err:
            logger.warning(
                "Checkpoint eval images failed at step %d: %s",
                self._step,
                _img_err,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: Optional[str] = None) -> Path:
        """儲存訓練 checkpoint（含 ``identity`` 身分區塊）。

        實作已收斂至 ``training/checkpoint/checkpointer.py::save_checkpoint``
        （與 ``Checkpointer.save`` 共用同一份），本方法只做屬性 → 參數的搬運。
        除五個原有 key（step/model/optimizer/scheduler/scaler）外另寫入
        ``identity``：git commit、config sha256、seed、推論敏感超參數
        （``psi_lowpass_sigma`` / ``ke_anchor_mode`` / ``apply_ic_mask`` …）、
        以及內嵌的反正規化常數。見 ``training/checkpoint/identity.py``。

        ★存檔後會 re-load 驗證 identity 確實落盤；缺則 raise（非 warning）。
        需要存無身分 ckpt 時，顯式設定 ``TES_ALLOW_UNIDENTIFIED_CKPT=1``。

        Args:
            tag: 可選標籤（如 "final"）；None 時用步數命名。

        Returns:
            儲存後的 checkpoint 檔案 Path。

        Raises:
            CheckpointIdentityError: 身分區塊建立/落盤驗證失敗且未顯式降級。
        """
        return save_checkpoint(
            self._output_dir,
            step=self._step,
            tag=tag,
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            scaler=self._scaler,
            cfg=self._cfg,
            field_normalizer=self._field_normalizer,
            cond_normalizer=self._normalizer,
            write_jsonl=self._write_jsonl,
        )

    def _check_ckpt_identity(self, ckpt: dict, ckpt_path: str | Path) -> None:
        """比對 ckpt 內 ``identity`` 與當前 config；處置由 ``cfg.ckpt_identity_mismatch_resume``。

        本方法只服務**續訓載入**（``load_checkpoint``），故讀 resume 專用欄位
        （預設 ``"warn"``），與評測／製圖／部署端的 ``cfg.ckpt_identity_mismatch``
        （預設 ``"raise"``）分開。取 ``"warn"`` 是為了不讓 identity 比對異常擋住續訓；
        「推論敏感超參數不符」與「此 ckpt 無身分資訊」都仍會留下明確 log ＋
        metrics.json 記錄。

        Args:
            ckpt: 已 ``torch.load`` 的 checkpoint dict。
            ckpt_path: checkpoint 路徑（訊息用）。

        Raises:
            ValueError: ``cfg.ckpt_identity_mismatch_resume == "raise"`` 且身分不一致。
        """
        # ⛔ 禁 fallback 存取：缺欄位即 AttributeError（config.py 是單一真相）。
        on_mismatch = self._cfg.ckpt_identity_mismatch_resume
        try:
            ckpt_identity.log_identity_on_load(
                ckpt, ckpt_path=ckpt_path, context="trainer"
            )
            report = ckpt_identity.verify_identity(
                ckpt,
                self._cfg,
                field_normalizer=self._field_normalizer,
                ckpt_path=ckpt_path,
                on_mismatch=on_mismatch,
                context="trainer",
            )
        except ValueError:
            # ★ 使用者顯式要求的中止不得被下面那個寬 except 吞成一行 warning。
            if on_mismatch == "raise":
                raise
            logger.warning("checkpoint identity 比對失敗（ValueError）", exc_info=True)
            return
        except Exception as exc:  # noqa: BLE001 — 比對失敗不得阻擋載入
            logger.warning("checkpoint identity 比對失敗：%s", exc, exc_info=True)
            return
        self._write_jsonl(
            {
                "event": "ckpt_identity",
                "ckpt": str(ckpt_path),
                "has_identity": report["has_identity"],
                "identity_source": report["identity_source"],
                "n_mismatch": len(report["mismatches"]),
                "mismatches": report["mismatches"],
                "denorm_status": report["denorm"].get("status"),
                "trained_at_commit": report["git"].get("trained_at_commit"),
            },
            force_flush=True,
        )

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        """從 checkpoint 恢復訓練狀態（先比對身分欄位）。

        Args:
            ckpt_path: checkpoint .pth 檔路徑。
        """
        ckpt = torch.load(ckpt_path, map_location=self._device)
        self._check_ckpt_identity(ckpt, ckpt_path)
        self._step = ckpt["step"]
        self._model.load_state_dict(ckpt["model"])
        self._optimizer.load_state_dict(ckpt["optimizer"])
        self._scheduler.load_state_dict(ckpt["scheduler"])
        self._scaler.load_state_dict(ckpt["scaler"])
        logger.info("已從 %s 恢復，step=%d", ckpt_path, self._step)

    def load_weights_only(self, ckpt_path: str | Path) -> None:
        """只載入 model weights；optimizer/scheduler/step 全部重置（warm restart 用）。

        Args:
            ckpt_path: checkpoint .pth 檔路徑。
        """
        ckpt = torch.load(ckpt_path, map_location=self._device)
        self._check_ckpt_identity(ckpt, ckpt_path)
        # strict=False 只略過 missing/unexpected key，但對「同名不同 shape」仍 raise。
        # 架構改動（如 mean_encoder lite 化：encoder.0 由 9→36 入通道）會撞此例外 →
        # 先濾掉 shape 不符的 ckpt key，讓該層 fresh init（與新模組同等 warm 行為）。
        cur_sd = self._model.state_dict()
        src_sd = ckpt["model"]
        shape_skipped = [
            k
            for k, v in src_sd.items()
            if k in cur_sd and tuple(cur_sd[k].shape) != tuple(v.shape)
        ]
        filtered_sd = {
            k: v
            for k, v in src_sd.items()
            if not (k in cur_sd and tuple(cur_sd[k].shape) != tuple(v.shape))
        }
        # concat-MSP fusion 擴展：若 fusion.weight 是 n_scales 倍數擴展（combined_ch → combined_ch*N），
        # 填 scale-0 block = src weights，其餘 block zero-init（而非 fresh random init）。
        # → t=0 時 concat-MSP forward ≡ 單尺度 forward（warm-load 保 init 行為不變）。
        _fusion_expanded: dict = {}
        for k in list(shape_skipped):
            src_v, cur_v = src_sd[k], cur_sd[k]
            if (
                k.endswith("fusion.weight")
                and src_v.ndim == 4
                and cur_v.ndim == 4
                and src_v.shape[0] == cur_v.shape[0]
                and src_v.shape[2:] == cur_v.shape[2:]
                and cur_v.shape[1] > src_v.shape[1]
                and cur_v.shape[1] % src_v.shape[1] == 0
            ):
                expanded = torch.zeros_like(cur_v)
                expanded[:, : src_v.shape[1], :, :] = src_v
                _fusion_expanded[k] = expanded
        if _fusion_expanded:
            logger.info(
                "Warm restart fusion concat-MSP scale-0 fill: %s",
                list(_fusion_expanded.keys()),
            )
            shape_skipped = [k for k in shape_skipped if k not in _fusion_expanded]
            filtered_sd.update(_fusion_expanded)
        if shape_skipped:
            logger.warning(
                "Warm restart shape-mismatch keys skipped (fresh init): %s",
                shape_skipped,
            )
        missing, unexpected = self._model.load_state_dict(filtered_sd, strict=False)
        if missing:
            logger.info(
                "Warm restart missing keys (expected for new modules): %s", missing
            )
        if unexpected:
            logger.warning("Warm restart unexpected keys: %s", unexpected)
        logger.info(
            "Warm restart：只載入 model weights from %s，step 從 0 重新計算", ckpt_path
        )

    # ------------------------------------------------------------------
    # Curriculum 工具
    # ------------------------------------------------------------------

    def _sample_k(self) -> int:
        """依目前步數從 k curriculum 取樣 rollout 步數（委派至 schedule.sample_k）。"""
        return _sched.sample_k(self._cfg.curriculum, self._step)

    def _get_val_k(self) -> int:
        """依目前步數回傳 val 評估使用的 k（與訓練同相位的 k_max）。"""
        return _sched.get_val_k(self._cfg.curriculum, self._step)

    def _get_loss_version(self) -> str:
        """依目前步數回傳 Loss curriculum 版本（``v1`` / ``v2`` / ``v3``）。"""
        return _sched.get_loss_version(self._cfg.curriculum, self._step)

    def _get_tf_ratio(self) -> float:
        """依目前步數計算 Teacher Forcing ratio（委派至 schedule.get_tf_ratio）。"""
        return _sched.get_tf_ratio(self._cfg.tfc, self._step)

    def _get_p_teacher(self) -> float:
        """Scheduled Sampling の p_teacher 退火值（委派至 schedule.get_p_teacher）。"""
        return _sched.get_p_teacher(self._cfg.ss, self._step)

    def _uses_per_step_delta_ctu(self) -> bool:
        """判斷 model.forward 是否應接收 per-step delta_ctu_seq。

        Returns:
            time_conditioned 或（arch=h_multirate 且 cond_dim=19）時為 True。
        """
        return bool(
            self._cfg.time_conditioned
            or (
                self._cfg.model.arch == "h_multirate" and self._cfg.model.cond_dim == 19
            )
        )

    # ------------------------------------------------------------------
    # 建構工具
    # ------------------------------------------------------------------

    def _build_cond_norm(
        self,
        g_vec: torch.Tensor,
        p_vec: torch.Tensor,
        delta_ctu: Optional[torch.Tensor] = None,
        cumulative_ctu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """依 cfg.model.cond_dim 組合條件向量（委派至 normalizer.build_cond）。

        cond_17D = [G_vec_norm(14), P_vec_norm(3)]。★這是唯一維度。
        """
        return self._normalizer.build_cond(self._cfg.model.cond_dim, g_vec, p_vec)

    def _build_g_vec_norm(self, g_vec: torch.Tensor) -> torch.Tensor | None:
        """Dual-pathway G_vec 正規化（gvec_mode=additive/gated 且 g_vec_dim>0 時回傳正規化 G_vec）。"""
        gvec_mode = self._cfg.model.gvec_mode
        if gvec_mode in ("additive", "gated") and self._cfg.model.g_vec_dim > 0:
            return self._normalizer.normalize_g(g_vec)
        return None

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """建立學習率排程器（"cosine" 單段 或 "warm_restart" per-phase SGDR）。

        LambdaLR 回傳 lr/base_lr 比值（optimizer base lr = cfg.lr）。

        Returns:
            LambdaLR scheduler（每步呼叫一次）。
        """
        cfg = self._cfg
        warmup = cfg.warmup_steps
        total = cfg.max_steps
        lr_ratio = cfg.lr_min / max(cfg.lr, 1e-12)

        if cfg.lr_schedule == "warm_restart":
            bounds = list(cfg.lr_restart_boundaries)
            decay = cfg.lr_restart_peak_decay
            rwarm = cfg.lr_restart_warmup
            ends = bounds[1:] + [total]  # 各 phase 終點：下一 boundary，末段到 total

            def lr_lambda(step: int) -> float:
                # 全域起始 warmup（第一 phase 前暖機，避 step0 直接 peak）
                if step < warmup:
                    return step / max(1, warmup)
                # 當前 phase k＝最大 boundary <= step
                k = 0
                for i, b in enumerate(bounds):
                    if step >= b:
                        k = i
                start, end = bounds[k], ends[k]
                peak = decay**k  # peak/base = decay^k
                local = step - start
                span = max(1, end - start)
                if local < rwarm:  # phase 內 re-warmup：lr_min→peak
                    frac = local / max(1, rwarm)
                    return lr_ratio + (peak - lr_ratio) * frac
                progress = (local - rwarm) / max(1, span - rwarm)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return lr_ratio + (peak - lr_ratio) * cosine

            return torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_ratio + (1.0 - lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda)

    def _setup_logging(self) -> None:
        """設定 logging 輸出至 stdout + 檔案，並開啟 metrics.json（line-delimited）。"""
        log_path = self._output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, encoding="utf-8"),
            ],
        )
        self._metrics_fp = (self._output_dir / "metrics.json").open(
            "a", encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# 子元件 facade：橋接 step 子元件對 data_prep / regularizer 的兩種存取慣例
# （部分 step 以 dict["key"] 取 callable，部分以 attribute method 呼叫）。
# 委派目標為已驗證正確的 ConditionBuilder / DataRenormalizer / ICOps /
# Regularizer 子元件，行為與單體版 trainer 對應方法一致。
# ---------------------------------------------------------------------------


class _DataPrepFacade:
    """資料前處理 facade：聚合 ConditionBuilder + DataRenormalizer + ICOps。

    同時提供 attribute-style 方法（cross_ctu / pool step 用）與 dict-style
    ``__getitem__``（base / mean / turb step 用），兩種存取均委派至同一組
    已驗證子元件，確保行為一致。

    Args:
        ctx: TrainerContext 共享狀態容器。
    """

    def __init__(self, ctx: TrainerContext) -> None:
        self.ctx = ctx
        self._cond = ConditionBuilder(ctx)
        self._renorm = DataRenormalizer(ctx)
        self._ic = ICOps(ctx)

    # ── attribute-style（cross_ctu / pool step）──────────────────────────
    def renorm_source_to_global(self, static, ic_physics, target_traj, batch):
        return self._renorm.renorm_source_to_global(
            static, ic_physics, target_traj, batch
        )

    def renorm_ic_mphy(self, ic_raw, batch):
        return self._renorm.renorm_ic_mphy(ic_raw, batch)

    def renorm_ic_mphy_traj(self, traj, batch):
        return self._renorm.renorm_ic_mphy_traj(traj, batch)

    def get_fluid_mask(self, static):
        return self._ic.get_fluid_mask(static)

    def maybe_train_noise(self, ic_physics, fluid_mask):
        return self._ic.apply_noise(ic_physics, fluid_mask)

    def get_prior_vel_norm(self):
        return self._ic.get_prior_vel_norm()

    def get_mean_p_gt(self):
        return self._ic.get_mean_p_gt()

    def build_cond_norm(self, g_vec, p_vec):
        return self._cond.build_cond_norm(g_vec, p_vec)

    def build_g_vec_norm(self, g_vec):
        return self._cond.build_g_vec_norm(g_vec)

    def uses_per_step_delta_ctu(self):
        return self._cond.uses_per_step_delta_ctu()

    # ── dict-style（base / mean / turb step）─────────────────────────────
    def __getitem__(self, key):
        _table = {
            "renorm_source_to_global": self.renorm_source_to_global,
            "renorm_ic_mphy": self.renorm_ic_mphy,
            "renorm_ic_mphy_traj": self.renorm_ic_mphy_traj,
            "get_fluid_mask": self.get_fluid_mask,
            "maybe_train_noise": self.maybe_train_noise,
            "get_prior_vel_norm": self.get_prior_vel_norm,
            "get_mean_p_gt": self.get_mean_p_gt,
            "build_cond_norm": self.build_cond_norm,
            "build_g_vec_norm": self.build_g_vec_norm,
            "uses_per_step_delta_ctu": self.uses_per_step_delta_ctu,
        }
        return _table[key]


class _RegularizerFacade:
    """正則化 facade：包裝 Regularizer 子元件並提供 dict-style + attribute 存取。

    base / turb step 以 ``regularizer["temporal_ke_loss"]`` /
    ``["jacobian_reg_loss"]`` 取 callable；其餘元件以 attribute 呼叫。兩者均
    委派至 Regularizer。額外提供 NaN-guard 介面（``nan_consecutive`` /
    ``increment_nan`` / ``reset_nan`` / ``check_nan_weights`` /
    ``save_emergency_checkpoint``）與 ``write_jsonl``，供 CrossCTUStep /
    pool step 之不穩定守衛使用；行為與單體版 trainer 對應段落一致。

    Args:
        ctx: TrainerContext 共享狀態容器。
    """

    def __init__(self, ctx: TrainerContext) -> None:
        self.ctx = ctx
        self._reg = _RegularizerCore(ctx)
        self.nan_consecutive: int = 0  # 連續不穩定步計數（CrossCTU NaN guard）

    def temporal_ke_loss(self, state, cond, mask, g_vec_norm, cumul_ctu, l_char):
        return self._reg.temporal_ke(state, cond, mask, g_vec_norm, cumul_ctu, l_char)

    def jacobian_reg_loss(
        self, state, cond, mask, g_vec_norm, dt_first, prior_ctx, fx_ref=None
    ):
        return self._reg.jacobian(
            state, cond, mask, g_vec_norm, dt_first, prior_ctx, fx_ref
        )

    def jac_observability(self) -> dict[str, float]:
        """jac_reg 的可觀測性欄位（與 log 取樣的 step 奇偶無關）。

        Returns:
            dict：``jac_gate_calls``（計算閘門觸發累計次數）、``jac_active_calls``
            （hinge 實際非零累計次數）、``jac_ratio_last``（最近一次量到的單步放大率）、
            ``jac_raw_last``（最近一次未加權的 hinge 值）。
        """
        return {
            "jac_gate_calls": float(self._reg.jac_gate_calls),
            "jac_active_calls": float(self._reg.jac_active_calls),
            "jac_ratio_last": float(self._reg.last_jac_ratio),
            "jac_raw_last": float(self._reg.last_jac_raw),
        }

    def increment_nan(self) -> None:
        """連續不穩定步計數 +1。"""
        self.nan_consecutive += 1

    def reset_nan(self) -> None:
        """正常步重置連續不穩定步計數。"""
        self.nan_consecutive = 0

    def check_nan_weights(self, step: int, model) -> None:
        """每 500 步檢查模型權重是否含 NaN/Inf；命中則寫 error event + 緊急存檔後 raise。

        Args:
            step: 當前訓練步數。
            model: 待檢查的 nn.Module。

        Raises:
            RuntimeError: 任意權重含 NaN/Inf 時。
        """
        if step % 500 != 0:
            return
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                logger.error(
                    "[step %d] NaN/Inf in param %s — emergency stop", step, name
                )
                self.write_jsonl(
                    {
                        "event": "error",
                        "step": step,
                        "reason": "nan_weight",
                        "param": name,
                    },
                    force_flush=True,
                )
                self.save_emergency_checkpoint()
                raise RuntimeError(f"NaN weight detected in {name} at step {step}")

    def save_emergency_checkpoint(self) -> None:
        """觸發緊急 checkpoint 儲存（委派至 ctx 上注入的 trainer._save_checkpoint）。"""
        save_fn = getattr(self.ctx, "_save_checkpoint", None)
        if callable(save_fn):
            save_fn(tag="emergency")

    def write_jsonl(self, record, force_flush=False):
        fp = getattr(self.ctx, "metrics_file", None)
        if fp is not None:
            fp.write(dumps_record(record) + "\n")
            if force_flush:
                fp.flush()

    def __getitem__(self, key):
        _table = {
            "temporal_ke_loss": self.temporal_ke_loss,
            "jacobian_reg_loss": self.jacobian_reg_loss,
        }
        return _table[key]
