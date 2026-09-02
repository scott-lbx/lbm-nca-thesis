"""training/trainer_base.py — module-agnostic 訓練主迴圈骨架。

本模組提供 ``TrainerBase``：一個與「訓練模組類型」(mean / turb / base) 解耦的
訓練主迴圈骨架，將 module-specific 行為抽象為三個 hook：

    - ``train_step(batch) -> dict``       單步梯度更新（回傳 metrics dict）。
    - ``validate(loader) -> dict``        在 val loader 上計算指標（不更新權重）。
    - ``save_monitor_vis(ckpt_path)``     存 ckpt 後的 rollout / pool 視覺化。

★定位（安全鐵則）：本檔為 **additive 並存** 結構，與既有 monolithic
``trainer.TESTrainer`` 完全獨立、**不改動任何舊路徑**。預設情況下 ``train.py``
仍走 ``TESTrainer``；``TrainerBase`` / ``MeanTrainer`` / ``TurbTrainer`` 為後續
切換 shim 時的目標骨架，目前可獨立 import 但未接入 live training path。

設計約束：
    - 子類別 **必須** override 三個抽象 hook；未 override 直接 ``NotImplementedError``。
    - 主迴圈骨架（step 計數、pbar、jsonl record、early stop、checkpoint 排程）
      module-agnostic，行為對齊 ``TESTrainer.train()`` 之共用骨幹。
    - 與 ``TESTrainer`` 共享同一組已驗證子元件（TrainerContext + step / validator
      / checkpointer / regularizer facade），不重寫其內部邏輯。
"""

from __future__ import annotations

import abc
import dataclasses
import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import IO, Any, Callable, Optional

import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .early_stop_dispatch import dispatch_early_stop
from .jsonl import dumps_record

logger = logging.getLogger(__name__)

_LOG_INTERVAL: int = 100  # train_step record 寫入頻率（每 N 步）


class TrainerBase(abc.ABC):
    """module-agnostic 訓練主迴圈骨架（mean / turb / base 共用）。

    子類別透過 override ``train_step`` / ``validate`` / ``save_monitor_vis``
    注入 module-specific 行為；本基底只負責 module-agnostic 的：

        - 啟動 header（n_params / val cases / config dump → log + jsonl）
        - 主迴圈步進（step 計數、throughput、pbar、train_step jsonl record）
        - 週期 validation（cold / warm split 分開記錄）
        - early stopping monitor 更新
        - 週期 checkpoint + monitor 視覺化排程

    Args:
        ctx: TrainerContext 共享狀態容器（含 model / cfg / loaders / optimizer …）。

    Notes:
        本基底刻意 **不** 自行建立 optimizer / scheduler / scaler / 子元件；
        這些由具體 trainer（MeanTrainer / TurbTrainer）於 ``__init__`` 建好後注入
        ``ctx``，與 ``TESTrainer`` 的建構順序一致，避免重複且分歧的初始化。
    """

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self._cfg = ctx.cfg
        self._model = ctx.model
        self._device = ctx.device
        self._output_dir = Path(ctx.output_dir)
        self._train_loader = ctx.train_loader
        self._val_cold_loader = ctx.val_cold_loader
        self._val_warm_loader = ctx.val_warm_loader
        self._early_stop = ctx.early_stopping
        self._scheduler = ctx.scheduler
        self._step = int(ctx.step)
        self._should_stop: bool = False
        self._metrics_fp: Optional[IO[str]] = ctx.metrics_file
        self._bg_executor: ThreadPoolExecutor | None = ctx.executor
        self._bg_jobs: list[Future] = []

    # ------------------------------------------------------------------
    # 抽象 hook（子類別必須 override）
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def train_step(self, batch: Optional[dict]) -> dict[str, float]:
        """執行一步梯度更新並回傳 metrics dict。

        Args:
            batch: 一個 DataLoader batch dict；rolling pool 自帶資料時可為 None。

        Returns:
            metrics dict（至少含 ``total`` / ``k`` / ``loss_version``）。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self, loader: Any) -> dict[str, float]:
        """在指定 val loader 上計算指標（不更新權重）。

        Args:
            loader: cold 或 warm val DataLoader。

        Returns:
            metrics dict（跨 batch 平均）。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_monitor_vis(self, ckpt_path: Path, do_visual: bool) -> None:
        """存 ckpt 後的 rollout / pool 視覺化（module-specific）。

        Args:
            ckpt_path: 剛存好的 checkpoint 路徑。
            do_visual: 是否到 vis_interval（False 時子類別可略過昂貴繪圖）。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # module-specific 委派 hook（預設委派回具體 trainer；可 override）
    # ------------------------------------------------------------------

    def select_step_batch(self) -> tuple[Any, bool]:
        """取得本步要餵入 ``train_step`` 的 batch。

        預設行為：從 train_loader 取下一 batch（StopIteration 時重啟 iterator）。
        子類別（rolling pool）可 override 回傳 ``(None, True)`` 表示自帶資料。

        Returns:
            (batch, is_pool_step)；is_pool_step=True 時 batch 可為 None。
        """
        if not hasattr(self, "_data_iter") or self._data_iter is None:
            self._data_iter = iter(self._train_loader)
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self._train_loader)
            batch = next(self._data_iter)
        return batch, False

    def save_checkpoint(self, tag: Optional[str] = None) -> Path:
        """儲存 checkpoint（委派至具體 trainer 的權威實作）。

        具體 trainer 透過 ``ctx._save_checkpoint`` 注入；缺失時 ``RuntimeError``。
        """
        save_fn = getattr(self._ctx, "_save_checkpoint", None)
        if not callable(save_fn):
            raise RuntimeError(
                "TrainerBase.save_checkpoint：具體 trainer 未注入 ctx._save_checkpoint"
            )
        return save_fn(tag=tag)

    # ------------------------------------------------------------------
    # 共用工具（module-agnostic）
    # ------------------------------------------------------------------

    def _write_jsonl(self, record: dict, force_flush: bool = False) -> None:
        """寫一筆 JSON record 至 metrics.json（line-delimited）。"""
        if self._metrics_fp is not None:
            self._metrics_fp.write(dumps_record(record) + "\n")
            if force_flush:
                self._metrics_fp.flush()

    def _validate_split(self, split_name: str, loader: Any) -> dict[str, float]:
        """單一 split 驗證 + log + jsonl（委派 module-specific ``validate``）。"""
        split_metrics = self.validate(loader)
        if not split_metrics:
            return {}
        val_parts = "  ".join(
            f"{k}={v:.4f}" for k, v in sorted(split_metrics.items())
        )
        logger.info("[step %6d] VAL %s: %s", self._step, split_name, val_parts)
        self._write_jsonl(
            {"event": "val", "split": split_name, "step": self._step, **split_metrics},
            force_flush=True,
        )
        return split_metrics

    def _emit_startup(self) -> None:
        """啟動 header：n_params / val cases / config dump → log + jsonl。"""
        n_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
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
            "%s 開始訓練，device=%s, max_steps=%d, n_params=%s (%.2fM)",
            type(self).__name__,
            self._device,
            self._cfg.max_steps,
            f"{n_params:,}",
            n_params / 1e6,
        )
        logger.info("val split — cold: %d cases, warm: %d cases", val_cold_n, val_warm_n)
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

    # ------------------------------------------------------------------
    # 主迴圈
    # ------------------------------------------------------------------

    def train(self) -> None:
        """執行完整訓練主迴圈直至 max_steps（module-agnostic 骨架）。

        步進骨架對齊 ``TESTrainer.train()`` 之共用結構；module-specific 的單步、
        驗證、視覺化分別委派至 ``train_step`` / ``validate`` / ``save_monitor_vis``。
        """
        self._emit_startup()
        self._model.train()
        t0 = time.time()
        _window_t = t0

        _log_ctx = logging_redirect_tqdm()
        _log_ctx.__enter__()
        pbar = tqdm(
            total=self._cfg.max_steps,
            initial=self._step,
            desc=type(self).__name__,
            unit="step",
            dynamic_ncols=True,
            colour="white",
        )

        try:
            while self._step < self._cfg.max_steps and not self._should_stop:
                self._ctx.step = self._step
                batch, is_pool = self.select_step_batch()
                metrics = self.train_step(None if is_pool else batch)
                self._step = self._ctx.step
                elapsed = time.time() - t0

                # ── throughput（每 100 步；前 100 步每步）──────────────
                steps_per_sec: Optional[float] = None
                _early = self._step < 100
                if (self._step % 100 == 0 and self._step > 0) or _early:
                    window_elapsed = time.time() - _window_t
                    n_window = 1 if _early else 100
                    steps_per_sec = n_window / max(window_elapsed, 1e-6)
                    _window_t = time.time()

                # ── train_step jsonl record ───────────────────────────
                lr = self._scheduler.get_last_lr()[0]
                if self._step % _LOG_INTERVAL == 0 or _early:
                    json_record: dict = {
                        "event": "train_step",
                        "step": self._step,
                        "loss": metrics.get("total"),
                        "k": metrics.get("k"),
                        "v": metrics.get("loss_version"),
                        "lr": lr,
                        "t": elapsed,
                        "grad_norm": metrics.get("grad_norm"),
                    }
                    for key, value in metrics.items():
                        if key in json_record or key == "nan_keys":
                            continue
                        if isinstance(value, (int, float, str)) or value is None:
                            json_record[key] = value
                    if steps_per_sec is not None:
                        json_record["steps_per_sec"] = steps_per_sec
                    self._write_jsonl(json_record, force_flush=True)

                pbar.set_postfix(
                    loss=f"{metrics.get('total', float('nan')):.4f}",
                    k=metrics.get("k", 0),
                    v=metrics.get("loss_version", "?"),
                    lr=f"{lr:.2e}",
                    refresh=False,
                )
                pbar.update(1)

                # ── 週期 validation ──────────────────────────────────
                if self._step % self._cfg.val_interval == 0:
                    self._run_validation_block()

                # ── 週期 checkpoint + monitor vis ────────────────────
                if self._step % self._cfg.ckpt_interval == 0:
                    ckpt_path = self.save_checkpoint()
                    do_visual = self._step % self._cfg.vis_interval == 0
                    try:
                        self.save_monitor_vis(ckpt_path, do_visual)
                    except Exception as _vis_err:
                        logger.warning(
                            "monitor vis skipped at step %d: %s", self._step, _vis_err
                        )

            if not self._should_stop:
                self.save_checkpoint(tag="final")
                logger.info("訓練完成，共 %d steps", self._step)
            else:
                logger.info(
                    "訓練提前結束（EarlyStopping）step=%d", self._step
                )
        finally:
            pbar.close()
            _log_ctx.__exit__(None, None, None)
            if self._metrics_fp is not None:
                self._metrics_fp.flush()
            if self._bg_executor is not None:
                self._bg_executor.shutdown(wait=True)
                self._bg_executor = None

    def _run_validation_block(self) -> None:
        """週期 validation：cold / warm split 分開 → early stop monitor 更新。"""
        cold_metrics: dict[str, float] = {}
        warm_metrics: dict[str, float] = {}
        if hasattr(self._model, "set_aux_metrics_enabled"):
            self._model.set_aux_metrics_enabled(True)
        for split_name, loader in (
            ("cold", self._val_cold_loader),
            ("warm", self._val_warm_loader),
        ):
            if loader is None:
                continue
            split_metrics = self._validate_split(split_name, loader)
            if split_name == "cold":
                cold_metrics = split_metrics
            else:
                warm_metrics = split_metrics
        if hasattr(self._model, "set_aux_metrics_enabled"):
            self._model.set_aux_metrics_enabled(False)

        # 分派與 monitor 解析都委派給共用實作 `schedule.select_early_stop_monitor`，
        # 不在此保留第二份平行實作。
        if dispatch_early_stop(
            es_cfg=self._cfg.early_stop,
            step=self._step,
            cold_metrics=cold_metrics,
            warm_metrics=warm_metrics,
            vg_metrics=None,          # 基底類無 vg_* 視覺門控
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
            self.save_checkpoint(tag="early_stop_best")
