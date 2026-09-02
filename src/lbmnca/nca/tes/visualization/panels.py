"""TES 視覺化面板：軌跡比較圖與訓練 Loss 曲線。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch

_RHO_IDX: int = 0
_JX_IDX: int = 3
_JY_IDX: int = 5
_EPS: float = 1e-8


def plot_trajectory_comparison(
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    fluid_mask: torch.Tensor,
    save_path: Path | None = None,
    title: str = "TES Trajectory Comparison",
) -> plt.Figure:
    """繪製預測 vs 目標軌跡面板（速度大小，最多顯示 6 步）。

    三列 grid：第一列 LBM 目標、第二列 TES 預測、第三列絕對誤差。
    若 save_path 不為 None 則儲存後關閉 figure（避免 GUI 記憶體洩漏）。

    Args:
        pred_traj: ``[k, 9, H, W]`` 預測軌跡（CPU tensor）。
        target_traj: ``[k, 9, H, W]`` LBM 目標軌跡。
        fluid_mask: ``[1, H, W]`` 流體遮罩，非流體區域歸零。
        save_path: 儲存路徑；None 則不儲存。
        title: 圖表標題。

    Returns:
        matplotlib Figure 物件。
    """
    pred_np: np.ndarray = pred_traj.detach().cpu().float().numpy()
    tgt_np: np.ndarray = target_traj.detach().cpu().float().numpy()
    mask_np: np.ndarray = fluid_mask.detach().cpu().float().numpy()[0]

    k: int = min(pred_np.shape[0], 6)
    fig, axes = plt.subplots(3, k, figsize=(3 * k, 9))
    if k == 1:
        axes = axes[:, None]

    for i in range(k):
        pred_vm: np.ndarray = _vel_magnitude(pred_np[i]) * mask_np
        tgt_vm: np.ndarray = _vel_magnitude(tgt_np[i]) * mask_np
        err: np.ndarray = np.abs(pred_vm - tgt_vm)

        vmax: float = max(float(tgt_vm.max()), 1e-6)

        axes[0, i].imshow(tgt_vm, cmap="plasma", vmin=0, vmax=vmax)
        axes[0, i].set_title(f"LBM t={i+1}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(pred_vm, cmap="plasma", vmin=0, vmax=vmax)
        axes[1, i].set_title(f"TES t={i+1}", fontsize=9)
        axes[1, i].axis("off")

        axes[2, i].imshow(err, cmap="hot", vmin=0, vmax=vmax * 0.2)
        axes[2, i].set_title(f"|err| t={i+1}", fontsize=9)
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("LBM", fontsize=10)
    axes[1, 0].set_ylabel("TES", fontsize=10)
    axes[2, 0].set_ylabel("|Error|", fontsize=10)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_loss_curves(
    loss_history: list[dict[str, float]],
    save_path: Path | None = None,
) -> plt.Figure:
    """繪製訓練 Loss 曲線（total + semantic components）。

    若 save_path 不為 None 則儲存後關閉 figure（避免 GUI 記憶體洩漏）。

    Args:
        loss_history: 每訓練步的 metrics dict 列表；各 dict 須含 "step" 鍵。
        save_path: 儲存路徑；None 則不儲存。

    Returns:
        matplotlib Figure 物件。
    """
    steps: list[float] = [m["step"] for m in loss_history if "step" in m]
    total: list[float] = [m.get("total", float("nan")) for m in loss_history]
    traj: list[float] = [m.get("obsmphy_traj_mae", float("nan")) for m in loss_history]
    delta: list[float] = [m.get("obsmphy_delta_mae", float("nan")) for m in loss_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(steps, total, label="Total", lw=1.5)
    ax1.set_ylabel("Total Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(steps, traj, label="ObsMphy_Traj_Mae", lw=1.2)
    ax2.plot(steps, delta, label="ObsMphy_Delta_Mae", lw=1.2)
    ax2.set_ylabel("Component Loss")
    ax2.set_xlabel("Step")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("TES Training Loss Curves", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def _vel_magnitude(physics: np.ndarray) -> np.ndarray:
    """從 9ch physics array 計算速度大小（正規化空間近似）。

    Args:
        physics: ``[9, H, W]`` numpy array，channel 順序依 TES state 定義。

    Returns:
        ``[H, W]`` 速度大小 array。
    """
    rho: np.ndarray = physics[_RHO_IDX] + _EPS
    ux: np.ndarray = physics[_JX_IDX] / rho
    uy: np.ndarray = physics[_JY_IDX] / rho
    return np.sqrt(ux ** 2 + uy ** 2 + _EPS)
