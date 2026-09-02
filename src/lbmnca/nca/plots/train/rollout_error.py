"""rollout_error.py — rollout 誤差曲線（subplots MAE / MSE / SSIM，每 sample 一線）。

png：``rollout_error.png``。mean phase 與 turb phase 用不同顏色，標記 mean→turb 轉換點。
由 trainer 的 rollout 視覺化流程餵 curve 資料呼叫。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_rollout_error_lines"]


def plot_rollout_error_lines(
    curves: Sequence[dict],
    out_path: Path,
    title: str,
    transition_ctu: Optional[float] = None,
) -> None:
    """rollout 誤差曲線：subplots MAE / MSE / SSIM，每個 sample 一條線。

    Args:
        curves: 每個 sample dict{ctu[list], mae[list], mse[list], ssim[list], color, phase}。
        out_path: 輸出 PNG。
        title: 圖標題。
        transition_ctu: 若給定，於該 x 畫垂直虛線標記 mean→turb 轉換點。
    """
    if not curves:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    metrics = [("mae", "MAE (fluid, norm)"), ("mse", "MSE (fluid, norm)"),
               ("ssim", "vel SSIM")]
    seen_phase: set[str] = set()
    for ax, (key, label) in zip(axes, metrics):
        for cv in curves:
            x = np.asarray(cv["ctu"], dtype=np.float64)
            y = np.asarray(cv[key], dtype=np.float64)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 1:
                continue
            phase = cv.get("phase", "")
            lbl = phase if phase not in seen_phase else None
            seen_phase.add(phase)
            ax.plot(x[m], y[m], color=cv.get("color", "tab:blue"),
                    alpha=0.7, linewidth=1.2, label=lbl)
        if transition_ctu is not None:
            ax.axvline(transition_ctu, color="k", linestyle="--", linewidth=1.0,
                       label="mean→turb")
        ax.set_xlabel("CTU")
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        if key != "ssim":
            ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
