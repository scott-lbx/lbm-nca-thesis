"""Eval plot: HBand GT vs Pred FFT deep diagnostic — one figure per case.

3-row layout:
  Row 0: Pred 2D FFT | GT 2D FFT | Diff FFT
  Row 1: kx=0 slice comparison | excess heatmap (kx=0~29 × all ky)
  Row 2: GT velocity field | Pred velocity field

Called from trainer._generate_checkpoint_images; outputs to eval_ckpt_*/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def _fft_power(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = mask.sum()
    mean = (field * mask).sum() / max(count, 1)
    centered = (field - mean) * mask
    F = np.fft.rfft2(centered)
    return np.abs(F) ** 2  # [H, W//2+1]


def plot_hband_debug(
    pred_vel: np.ndarray,
    gt_vel: np.ndarray,
    fluid_mask: np.ndarray,
    case_info: str,
    output_path: Path | str,
) -> None:
    """GT vs Pred FFT deep diagnostic for one case.

    Args:
        pred_vel: [H, W] float — predicted velocity magnitude (raw or normalised).
        gt_vel: [H, W] float — ground-truth velocity magnitude.
        fluid_mask: [H, W] float 0/1 — fluid pixels = 1.
        case_info: title string (e.g. "cold_case001 | Re=300").
        output_path: full path to write PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_f = np.asarray(pred_vel, dtype=np.float64)
    gt_f   = np.asarray(gt_vel,   dtype=np.float64)
    mask_f = np.asarray(fluid_mask, dtype=np.float64)

    pred_pow = _fft_power(pred_f, mask_f)
    gt_pow   = _fft_power(gt_f,   mask_f)
    H, W_half = pred_pow.shape

    pred_total = pred_pow[1:, :].sum() + 1e-12
    gt_total   = gt_pow[1:, :].sum()   + 1e-12

    ky     = np.arange(1, H)
    kx_arr = np.arange(W_half)
    ky_arr = np.arange(H)

    fig_w = 18.0
    h_fft  = 4.0
    h_anal = 3.5
    h_vel  = fig_w * 0.43 * (gt_f.shape[0] / gt_f.shape[1])
    fig_h  = h_fft + h_anal + h_vel + 2.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(3, 6, figure=fig,
                  height_ratios=[h_fft, h_anal, h_vel],
                  hspace=0.50, wspace=0.40,
                  top=0.94, bottom=0.04, left=0.07, right=0.97)

    # Row 0: 2D FFT panels
    diff_pow = np.log1p(pred_pow) - np.log1p(gt_pow)
    vmax_d   = np.percentile(np.abs(diff_pow), 98)

    for ax_fft, fft_data, title in [
        (fig.add_subplot(gs[0, 0:2]), np.log1p(pred_pow), "Pred 2D FFT (log power)"),
        (fig.add_subplot(gs[0, 2:4]), np.log1p(gt_pow),   "GT 2D FFT (log power)"),
    ]:
        ax_fft.pcolormesh(kx_arr, ky_arr, fft_data, cmap="inferno", shading="nearest")
        ax_fft.set_xscale("symlog", linthresh=1)
        ax_fft.set_yscale("symlog", linthresh=1)
        ax_fft.axvline(0.5, color="red", lw=1.5, ls="--", alpha=0.9, label="kx=0")
        ax_fft.set_title(title, fontsize=10)
        ax_fft.set_xlabel("kx [symlog]")
        ax_fft.set_ylabel("ky [symlog]")
        ax_fft.legend(fontsize=8)

    ax_diff = fig.add_subplot(gs[0, 4:6])
    ax_diff.pcolormesh(kx_arr, ky_arr, diff_pow, cmap="RdBu_r",
                       vmin=-vmax_d, vmax=vmax_d, shading="nearest")
    ax_diff.set_xscale("symlog", linthresh=1)
    ax_diff.set_yscale("symlog", linthresh=1)
    ax_diff.axvline(0.5, color="black", lw=1.5, ls="--", alpha=0.9, label="kx=0")
    ax_diff.set_title("Pred − GT log-FFT（紅=Pred多，藍=GT多）", fontsize=10)
    ax_diff.set_xlabel("kx [symlog]")
    ax_diff.legend(fontsize=8)

    # Row 1 Left: kx=0 slice
    pred_frac0 = pred_pow[1:, 0] / pred_total
    gt_frac0   = gt_pow[1:, 0]   / gt_total
    excess0    = float(np.clip(pred_frac0 - gt_frac0, 0, None).mean())

    ax_kx0 = fig.add_subplot(gs[1, 0:3])
    ax_kx0.plot(ky, pred_frac0, color="#D73027", lw=1.8, label="Pred kx=0")
    ax_kx0.plot(ky, gt_frac0,   color="#4575B4", lw=1.8, label="GT kx=0")
    ax_kx0.fill_between(ky, gt_frac0, pred_frac0,
                        where=(pred_frac0 >= gt_frac0), alpha=0.38,
                        color="#D73027", label="Pred > GT（假影區）")
    ax_kx0.fill_between(ky, gt_frac0, pred_frac0,
                        where=(gt_frac0 > pred_frac0), alpha=0.38,
                        color="#4575B4", label="GT > Pred")
    ax_kx0.set_title(f"kx=0 切片比較（HBand excess={excess0:.4f}）", fontsize=9)
    ax_kx0.set_xlabel("ky")
    ax_kx0.set_ylabel("佔總功率比例")
    ax_kx0.set_yscale("symlog", linthresh=1e-7)
    ax_kx0.yaxis.grid(True, which="both", ls="--", lw=0.5, alpha=0.55)
    ax_kx0.set_axisbelow(True)
    ax_kx0.legend(fontsize=8)
    ax_kx0.set_xlim(0, H // 2)

    # Row 1 Right: kx×ky excess heatmap
    N_KX_HEAT = 30
    excess_heat = np.zeros((H - 1, N_KX_HEAT))
    for kx_i in range(N_KX_HEAT):
        p_ki = pred_pow[1:, kx_i] / pred_total
        g_ki = gt_pow[1:, kx_i]   / gt_total
        excess_heat[:, kx_i] = p_ki - g_ki
    vmax_h = np.percentile(np.abs(excess_heat), 98)

    ax_heat = fig.add_subplot(gs[1, 3:6])
    ax_heat.pcolormesh(np.arange(N_KX_HEAT), np.arange(1, H), excess_heat,
                       cmap="RdBu_r", vmin=-vmax_h, vmax=vmax_h, shading="nearest")
    ax_heat.set_yscale("symlog", linthresh=1)
    ax_heat.yaxis.grid(True, which="both", ls="--", lw=0.5, alpha=0.4, color="white")
    ax_heat.set_axisbelow(False)
    ax_heat.set_title(
        f"Pred−GT excess heatmap（kx=0~{N_KX_HEAT - 1} × 所有ky）\n"
        "紅=Pred假影  藍=GT較強  白=接近零",
        fontsize=9,
    )
    ax_heat.set_xlabel("kx")
    ax_heat.set_ylabel("ky [symlog]")

    # Row 2: velocity fields
    for ax_v, field, panel_title in [
        (fig.add_subplot(gs[2, 0:3]), gt_f,   "GT 速度場"),
        (fig.add_subplot(gs[2, 3:6]), pred_f, "Pred 速度場"),
    ]:
        ax_v.imshow(field, cmap="viridis", origin="lower", aspect="equal")
        ax_v.set_title(panel_title, fontsize=9)
        ax_v.set_xticks([])
        ax_v.set_yticks([])

    fig.suptitle(
        f"HBand Debug — {case_info}\n"
        "判讀：kx=0 紅線越貼藍線越好；excess heatmap 越接近白色越好 ↓",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
