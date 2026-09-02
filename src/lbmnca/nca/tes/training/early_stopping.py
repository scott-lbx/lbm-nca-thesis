"""TESEarlyStopping — val_loss plateau 與崩潰偵測。

與 nan_patience 分工：
    nan_patience    → 單步 NaN / loss spike（訓練數值不穩定）
    TESEarlyStopping → val_loss 多 epoch 無改善（學習停滯）或大幅發散（崩潰）

觸發後由 TESTrainer 寫入 early_stop checkpoint 並結束 train()。
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from ..config import TESEarlyStopConfig

logger = logging.getLogger(__name__)


class TESEarlyStopping:
    """監控 validation loss 的 plateau / divergence，輸出停止信號。

    Args:
        cfg: TESEarlyStopConfig，含 monitor、patience、mode、min_delta、
            diverge_threshold、restore_best、start_step 等欄位。
    """

    def __init__(self, cfg: TESEarlyStopConfig) -> None:
        self._cfg = cfg
        self._best_val: float = float("inf") if cfg.mode == "min" else float("-inf")
        self._best_step: int = 0
        self._best_state: Optional[dict[str, torch.Tensor]] = None
        self._no_improve_count: int = 0
        self._triggered: bool = False
        self._trigger_reason: str = ""

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def update(
        self,
        val_loss: float,
        step: int,
        model_state: dict[str, torch.Tensor],
    ) -> bool:
        """新 val_loss 更新狀態，回傳是否應停止訓練。

        Args:
            val_loss: 當前 validation loss（對應 cfg.monitor key）。
            step: 當前訓練步數。
            model_state: model.state_dict()（用於 restore_best）。

        Returns:
            True 表示應立即停止訓練。
        """
        if not self._cfg.enabled or step < self._cfg.start_step:
            return False

        improved = self._is_improved(val_loss)

        if improved:
            self._best_val = val_loss
            self._best_step = step
            self._no_improve_count = 0
            if self._cfg.restore_best:
                self._best_state = {k: v.cpu().clone() for k, v in model_state.items()}
            logger.info(
                "[EarlyStopping] step=%d  新最佳 %s=%.6f",
                step, self._cfg.monitor, val_loss,
            )
        else:
            self._no_improve_count += 1
            logger.info(
                "[EarlyStopping] step=%d  無改善 %d/%d（best=%.6f @ step %d）",
                step, self._no_improve_count, self._cfg.patience,
                self._best_val, self._best_step,
            )

        # Plateau 停止
        if self._no_improve_count >= self._cfg.patience:
            self._triggered = True
            self._trigger_reason = (
                f"plateau: {self._no_improve_count} 次無改善"
                f"（patience={self._cfg.patience}）"
            )
            logger.warning("[EarlyStopping] 觸發 — %s", self._trigger_reason)
            return True

        # Divergence 停止（val_loss 相對最佳值暴增）
        if self._cfg.mode == "min" and self._best_val > 0:
            ratio = val_loss / (self._best_val + 1e-10)
            if ratio > self._cfg.diverge_threshold:
                self._triggered = True
                self._trigger_reason = (
                    f"diverge: val_loss/best={ratio:.2f}"
                    f"（threshold={self._cfg.diverge_threshold}）"
                )
                logger.warning("[EarlyStopping] 觸發 — %s", self._trigger_reason)
                return True

        return False

    def as_jsonl_record(self, step: int) -> dict[str, object]:
        """回傳供寫入 metrics.json 的狀態快照（line-delimited）。

        Args:
            step: 當前訓練步數。

        Returns:
            包含 early_stop 事件欄位的 dict，可直接序列化為 JSONL 一行。
        """
        return {
            "event": "early_stop",
            "step": step,
            "triggered": self._triggered,
            "reason": self._trigger_reason,
            "best_val": self._best_val,
            "best_step": self._best_step,
            "no_improve_count": self._no_improve_count,
            "patience": self._cfg.patience,
            "monitor": self._cfg.monitor,
        }

    # ------------------------------------------------------------------
    # 屬性
    # ------------------------------------------------------------------

    @property
    def triggered(self) -> bool:
        """是否已觸發早停。"""
        return self._triggered

    @property
    def trigger_reason(self) -> str:
        """觸發原因說明字串（未觸發時為空字串）。"""
        return self._trigger_reason

    @property
    def best_val(self) -> float:
        """目前觀測到的最佳 val_loss。"""
        return self._best_val

    @property
    def best_step(self) -> int:
        """達到最佳 val_loss 的訓練步數。"""
        return self._best_step

    @property
    def best_state(self) -> Optional[dict[str, torch.Tensor]]:
        """最佳 checkpoint 的 model state_dict（restore_best=False 時為 None）。"""
        return self._best_state

    @property
    def no_improve_count(self) -> int:
        """連續無改善的次數。"""
        return self._no_improve_count

    # ------------------------------------------------------------------
    # 內部
    # ------------------------------------------------------------------

    def _is_improved(self, val_loss: float) -> bool:
        """判斷 val_loss 是否相對 _best_val 有顯著改善（超過 min_delta）。

        Args:
            val_loss: 當前 validation loss。

        Returns:
            True 表示有顯著改善。
        """
        if self._cfg.mode == "min":
            return val_loss < self._best_val - self._cfg.min_delta
        return val_loss > self._best_val + self._cfg.min_delta
