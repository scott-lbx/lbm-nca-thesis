"""pool_vel.py — Experience Pool vel 監控（首 8 / 末 8 slot，col8×row2，固定軸）。

png：``pool_vel.png``。row0 = pool 首 8 slot（round-robin 新灌），row1 = 末 8 slot（累積偏離）。
固定軸 [0, vmax] 便於跨 vis_interval 比較 pool 演化。
由 trainer 的 pool 視覺化流程餵 vel 資料呼叫。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_pool_vel_grid"]


def _solid_nan(field: np.ndarray, solid: np.ndarray) -> np.ndarray:
    out = np.squeeze(field).astype(np.float64).copy()
    if solid is not None and out.shape == solid.shape:
        out[solid] = np.nan
    return out


def plot_pool_vel_grid(
    first_vel: "list[np.ndarray]",
    last_vel: "list[np.ndarray]",
    solid: np.ndarray,
    out_path: Path,
    title: str,
    vmax: float = 0.14,   # vis_range SSOT (stats.json vel_max)；診斷圖固定值域
) -> None:
    """Pool vel 監控：首 8 / 末 8 slot 的 vel_mag，col8×row2，固定值域 [0, vmax]。"""
    n = min(len(first_vel), len(last_vel), 8)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n + 0.8, 5),
                             constrained_layout=True, squeeze=False)
    im = None
    for col in range(n):
        for row, vels in ((0, first_vel), (1, last_vel)):
            v = _solid_nan(vels[col], solid)
            im = axes[row, col].imshow(v, vmin=0.0, vmax=vmax, cmap="plasma",
                                       origin="upper", aspect="equal")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        axes[0, col].set_title(f"#{col}", fontsize=9)
    axes[0, 0].set_ylabel("first 8 slots", fontsize=10)
    axes[1, 0].set_ylabel("last 8 slots", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01)
    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
