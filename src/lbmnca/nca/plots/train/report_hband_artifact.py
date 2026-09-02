"""Eval plot: HBand artifact report — building audience, no FFT background required.

2-row layout:
  Row 0: GT velocity | Pred velocity | |Error|
  Row 1: per-row error bar chart | band component | HBand ratio bar chart

Called from trainer._generate_checkpoint_images; outputs to eval_ckpt_*/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def _vel_rgb(
    field: np.ndarray, mask: np.ndarray, vmin: float = 0.0, vmax: float = 0.14
) -> np.ndarray:
    d01 = np.clip((field - vmin) / (vmax - vmin + 1e-8), 0, 1)
    rgb = plt.get_cmap("plasma")(d01)[:, :, :3]
    gray = np.full_like(rgb, 0.5)
    return np.where((mask > 0.5)[:, :, None], rgb, gray)


def _err_rgb(
    error: np.ndarray, mask: np.ndarray, vmax: float | None = None
) -> np.ndarray:
    vals = error[mask > 0.5]
    if vmax is None:
        vmax = float(np.percentile(vals, 99)) if len(vals) else float(error.max())
    d01 = np.clip(error / (vmax + 1e-8), 0, 1)
    rgb = plt.get_cmap("Reds")(d01)[:, :, :3]
    gray = np.full_like(rgb, 0.88)
    return np.where((mask > 0.5)[:, :, None], rgb, gray)


def _hband_ratio(field: np.ndarray, mask: np.ndarray) -> float:
    mean = (field * mask).sum() / max(mask.sum(), 1)
    F = np.fft.rfft2((field - mean) * mask)
    power = np.abs(F) ** 2
    return float(power[1:, 0].sum() / (power.sum() - power[0, 0] + 1e-12))


def plot_hband_artifact(
    pred_vel: np.ndarray,
    gt_vel: np.ndarray,
    fluid_mask: np.ndarray,
    case_info: str,
    output_path: Path | str,
) -> None:
    """HBand artifact report for one case — building-audience readable.

    Args:
        pred_vel: [H, W] float — predicted velocity magnitude.
        gt_vel: [H, W] float — ground-truth velocity magnitude.
        fluid_mask: [H, W] float 0/1 — fluid pixels = 1.
        case_info: title string.
        output_path: full path to write PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_f = np.asarray(pred_vel, dtype=np.float64)
    gt_f = np.asarray(gt_vel, dtype=np.float64)
    mask_f = np.asarray(fluid_mask, dtype=np.float64)

    H, W = gt_f.shape
    error_f = np.abs(pred_f - gt_f)
    row_err = (error_f * mask_f).sum(axis=1) / (mask_f.sum(axis=1) + 1e-8)
    band_2d = np.outer(row_err, np.ones(W))

    gt_ratio = _hband_ratio(gt_f, mask_f)
    pred_ratio = _hband_ratio(pred_f, mask_f)

    fig_w = 18.0
    h_vel = 3.6
    h_anal = 3.8
    fig = plt.figure(figsize=(fig_w, h_vel + h_anal + 1.8))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[h_vel, h_anal],
        hspace=0.46,
        wspace=0.28,
        top=0.92,
        bottom=0.05,
        left=0.06,
        right=0.97,
    )

    # Row 0: velocity comparison
    for col, (img, panel_title) in enumerate(
        [
            (_vel_rgb(gt_f, mask_f), "GT 速度場（標準答案）"),
            (_vel_rgb(pred_f, mask_f), "Pred 速度場（模型輸出）"),
            (_err_rgb(error_f, mask_f), "|誤差|  =  |GT − Pred|\n（紅色 = 誤差大）"),
        ]
    ):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, origin="lower", aspect="auto")
        ax.set_title(panel_title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 1 Left: per-row error bar chart
    ax_row = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(H)
    norm_err = row_err / (row_err.max() + 1e-8)
    bar_colors = plt.get_cmap("Reds")(norm_err * 0.85 + 0.1)
    ax_row.barh(y_pos, row_err, color=bar_colors, height=1.0, linewidth=0)
    ax_row.set_xlabel("平均速度誤差", fontsize=9)
    ax_row.set_ylabel("圖像垂直位置（y 行）", fontsize=9)
    ax_row.set_title("各水平行的平均誤差\n峰值行 ≈ 水平帶假影所在位置", fontsize=9)
    ax_row.set_ylim(0, H)
    ax_row.tick_params(labelsize=7)
    ax_row.xaxis.grid(True, ls="--", lw=0.5, alpha=0.45)
    ax_row.set_axisbelow(True)

    # Row 1 Middle: band component visualisation
    ax_band = fig.add_subplot(gs[1, 1])
    vmax_b = (
        float(np.percentile(band_2d[mask_f > 0.5], 99))
        if (mask_f > 0.5).any()
        else band_2d.max()
    )
    ax_band.imshow(
        _err_rgb(band_2d, mask_f, vmax=vmax_b), origin="lower", aspect="auto"
    )
    ax_band.set_title(
        "水平帶成分視覺化\n（每行誤差展開成色帶；紅色 = 帶狀假影強烈位置）", fontsize=9
    )
    ax_band.set_xticks([])
    ax_band.set_yticks([])

    # Row 1 Right: HBand ratio bar chart
    ax_bar = fig.add_subplot(gs[1, 2])
    labels = ["GT\n（標準）", "Pred\n（模型輸出）"]
    values = [gt_ratio, pred_ratio]
    colors = ["#888888", "#D73027"]
    bars = ax_bar.bar(
        labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5
    )
    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + values[-1] * 0.04,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=bar.get_facecolor(),
        )
    ax_bar.set_ylabel("水平帶能量佔比", fontsize=9)
    ax_bar.set_title("水平條帶假影強度指標\n（越高 = 假影越嚴重）", fontsize=9)
    ax_bar.set_ylim(0, max(pred_ratio * 1.35, gt_ratio * 3, 0.02))
    ax_bar.yaxis.grid(True, ls="--", lw=0.5, alpha=0.45)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(labelsize=8)

    fig.suptitle(
        f"HBand Artifact — {case_info}\n"
        f"GT={gt_ratio:.3f}  Pred={pred_ratio:.3f}  "
        "判讀：Pred bar 越接近 GT bar 越好 ↓",
        fontsize=12,
        y=0.97,
    )
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
