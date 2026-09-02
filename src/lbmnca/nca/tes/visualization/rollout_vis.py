"""rollout_vis.py — rollout 繪圖函式的集中匯出點。

實作位於 ``lbmnca.nca.plots.train``（命名慣例：模組名 = 輸出 PNG 檔名）：

  - ``plot_eval_sequence`` / ``plot_eval_sequence_multi`` / ``two_window_converged``
    → ``plots/train/mean_eval_sequence.py``
  - ``plot_rollout_error_lines`` → ``plots/train/rollout_error.py``
  - ``plot_pool_vel_grid`` → ``plots/train/pool_vel.py``
"""
from __future__ import annotations

from lbmnca.nca.plots.train.mean_eval_sequence import (
    plot_eval_sequence,
    plot_eval_sequence_multi,
    two_window_converged,
)
from lbmnca.nca.plots.train.pool_vel import plot_pool_vel_grid
from lbmnca.nca.plots.train.rollout_error import plot_rollout_error_lines

__all__ = [
    "plot_eval_sequence",
    "plot_eval_sequence_multi",
    "plot_rollout_error_lines",
    "plot_pool_vel_grid",
    "two_window_converged",
]
