"""training/mean_trainer.py — LC decompose Phase 0 mean module 專用 trainer（additive §3）。

``MeanTrainer(TrainerBase)`` override 三個 hook，將 module-specific 行為委派至
已驗證的 ``TESTrainer`` 子元件：

    - ``train_step``        → MeanStep（``TESTrainer._step_mean.run``）
    - ``validate``          → ``TESTrainer._validate_split_mean``（EG-1 mean gate）
    - ``save_monitor_vis``  → mean rollout async vis（``TESTrainer._launch_async_vis``）
                              + pool vis（``TESTrainer._save_pool_vis``）

★定位（安全鐵則）：**additive 並存**。本類別內部持有一個已建好的 ``TESTrainer``
實例（backing），透過委派重用其全部已驗證邏輯 — **不重寫、不改動 trainer.py**。
``train.py`` 預設仍走 ``TESTrainer.train()``；本類別供後續切換 shim 時接入。

前置條件（與 ``TESTrainer`` 一致，缺即報錯）：``cfg.mean_module_enabled`` 為 True。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .trainer_base import TrainerBase

logger = logging.getLogger(__name__)


class MeanTrainer(TrainerBase):
    """LC decompose Phase 0 mean module trainer（委派 TESTrainer 子元件）。

    Args:
        backing: 已建構完成的 ``TESTrainer`` 實例（提供 ctx + 子元件 + 權威方法）。

    Raises:
        ValueError: ``backing._cfg.mean_module_enabled`` 非 True（mean trainer 前置）。
    """

    def __init__(self, backing: Any) -> None:
        if not backing._cfg.mean_module_enabled:
            raise ValueError(
                "MeanTrainer 要求 cfg.mean_module_enabled=True；"
                f"got mean_module_enabled={backing._cfg.mean_module_enabled}"
            )
        super().__init__(backing._ctx)
        self._backing = backing

    # ------------------------------------------------------------------
    # 抽象 hook override
    # ------------------------------------------------------------------

    def select_step_batch(self) -> tuple[Any, bool]:
        """mean module 一律走 mean_step（pool_step 只適用 trajectory）→ 取 GT batch。"""
        return super().select_step_batch()

    def train_step(self, batch: Optional[dict]) -> dict[str, float]:
        """單步 → MeanStep（Phase 0 mean_module，含 mean gate scheduled sampling）。"""
        return self._backing._step_mean.run(batch)

    def validate(self, loader: Any) -> dict[str, float]:
        """EG-1 mean gate 定量評估（§6.1 四項；委派 TESTrainer._validate_split_mean）。"""
        return self._backing._validate_split_mean(loader)

    def save_monitor_vis(self, ckpt_path: Path, do_visual: bool) -> None:
        """mean rollout vis（canonical = async lc_vis_external）+ pool vis。

        async vis 由獨立 subprocess 載 ckpt（device=cpu，不碰 live training）；
        pool vis 須畫當下 live pool → 主線 inline。
        """
        if do_visual:
            self._backing._launch_async_vis(ckpt_path)
            try:
                self._backing._save_pool_vis()
            except Exception as _pv_err:
                logger.warning("Pool vis skipped at step %d: %s", self._step, _pv_err)
