"""訓練期繪圖派發器 —— 每個 ``vis_interval`` 由 trainer 呼叫一次。

收錄：``loss_breakdown``（分項 train/eval cold/warm）、``val_vel``/``val_vor``
場域對比、``vis_gate_trends``（視覺 gate 趨勢）、``train_health``（grad/lr）、
``delta_ctu_audit``（分層視覺）。不畫 total loss——加權總和的高低沒有判讀意義，
一律看分項曲線。
"""
from __future__ import annotations

import logging

from . import (
    delta_ctu_audit,
    loss_breakdown,
    train_health,
    val_vel_field_compare,
    val_vor_field_compare,
    vis_gate_trends,
)

_logger = logging.getLogger(__name__)

_MODULES = [
    loss_breakdown,
    vis_gate_trends,
    train_health,
    val_vel_field_compare,
    val_vor_field_compare,
    delta_ctu_audit,
]


def plot_all(run_dir: str, step: int | None = None) -> None:
    """產生全部訓練圖，PNG 寫至 ``run_dir/plots/``。"""
    for mod in _MODULES:
        try:
            mod.plot(run_dir, step)
        except Exception as exc:
            _logger.debug("plot %s skipped: %s", mod.__name__, exc)
