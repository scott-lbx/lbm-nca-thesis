"""Plot: Vorticity field comparison via FieldColorizer — per-sample GT-derived range.

Reads val_fields_latest.npz.
Layout: 8 columns — first 4 cold, last 4 warm.
Rows: IC (t0) / Ground Truth (t0+n) / Predicted / |Error|.

每欄（一個 sample）用「該欄 GT」推導的對稱範圍 [-a,a]，IC/GT/pred 共用同 scale →
pred 渦旋若過阻尼/弱會誠實顯弱（按 GT scale 畫，非各自 auto-range）。每欄 a 寫在
欄標題（無單一共用 colorbar，因各欄 scale 不同）。詳見 _range.py。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ._range import err_range, gt_vor_range
from ._utils import ensure_plot_dir, save_plot_versions
from lbmnca.nca.utils.FieldColorizer import FieldColorizer

_N_COLD = 4
_N_WARM = 4

_colorizer = FieldColorizer()


def _to_rgb_vor(field_hw: np.ndarray, mask_hw: np.ndarray,
                vmin: float, vmax: float) -> np.ndarray:
    t = torch.from_numpy(field_hw.astype(np.float32)).unsqueeze(0)
    m = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0)
    rgb = _colorizer._colorize(t.unsqueeze(0), vmin, vmax, _colorizer._vor_lut, m.unsqueeze(0))
    return rgb[0].permute(1, 2, 0).numpy()


def _load_split(data: np.lib.npyio.NpzFile, prefix: str, n: int):
    """Return (target_vor, ic_vor, pred_vor, fluid_mask, t0_arr, n_arr) for split."""
    def _key(k):
        full = f"{prefix}_{k}"
        if full not in data:
            return None
        arr = data[full][:n]
        return None if arr.size == 0 else arr

    return (
        _key("target_vor"), _key("ic_vor"), _key("pred_vor"),
        _key("fluid_mask"), _key("t0"), _key("n_steps"),
    )


def plot(run_dir: str, step: int | None = None, vor_range: float | None = None) -> None:
    """繪製 val 渦度場對比。

    Args:
        run_dir: run 目錄（讀 val_fields_latest.npz）。
        step: 圖檔步數標記。
        vor_range: 給定則全欄用此對稱固定值域 [-vor_range, +vor_range]（vis_range
            SSOT，跨 run 可比）；None 則用 per-sample GT-derived 範圍。
    """
    npz_path = Path(run_dir) / "val_fields_latest.npz"
    if not npz_path.exists():
        return

    data = np.load(npz_path)
    vis_step = int(data["step"])
    is_mean = bool(data["mean_module_mode"])
    is_turb = bool(data["turb_module_mode"])

    cold = _load_split(data, "cold", _N_COLD)
    warm = _load_split(data, "warm", _N_WARM)

    # fall back to legacy single-split keys (old npz without cold_/warm_ prefix)
    if cold[0] is None and "target_vor" in data:
        arr = data["target_vor"]
        n_leg = len(arr)
        n_c = min(n_leg // 2, _N_COLD)
        n_w = min(n_leg - n_c, _N_WARM)
        fm = data["fluid_mask"] if "fluid_mask" in data else np.ones_like(arr)
        dummy_t0 = np.zeros(n_leg, dtype=np.int32)
        dummy_n  = np.ones(n_leg, dtype=np.int32)
        cold = (data["target_vor"][:n_c], data["ic_vor"][:n_c],
                data["pred_vor"][:n_c], fm[:n_c], dummy_t0[:n_c], dummy_n[:n_c])
        warm = (data["target_vor"][n_c:n_c+n_w], data["ic_vor"][n_c:n_c+n_w],
                data["pred_vor"][n_c:n_c+n_w], fm[n_c:n_c+n_w],
                dummy_t0[n_c:n_c+n_w], dummy_n[n_c:n_c+n_w])

    c_tgt, c_ic, c_pred, c_fm, c_t0, c_n = cold
    w_tgt, w_ic, w_pred, w_fm, w_t0, w_n = warm

    n_c = len(c_tgt) if c_tgt is not None else 0
    n_w = len(w_tgt) if w_tgt is not None else 0
    N = n_c + n_w
    if N == 0:
        return

    # 自動偵測 cold/warm 混合狀態
    has_cold = n_c > 0
    has_warm = n_w > 0
    if has_cold and has_warm:
        split_desc = f"cold: {n_c}  warm: {n_w}"
    elif has_cold:
        split_desc = f"cold-only: {n_c}  (no warm data)"
    else:
        split_desc = f"warm-only: {n_w}  (no cold data)"

    plot_dir = ensure_plot_dir(run_dir)

    col_ic   = list(c_ic)  if c_ic  is not None else []
    col_tgt  = list(c_tgt) if c_tgt is not None else []
    col_pred = list(c_pred) if c_pred is not None else []
    col_fm   = list(c_fm)  if c_fm  is not None else []
    col_t0   = list(c_t0.ravel()) if c_t0 is not None else [0]*n_c
    col_n    = list(c_n.ravel())  if c_n  is not None else [1]*n_c
    col_split = ["cold"] * n_c

    col_ic   += list(w_ic)   if w_ic   is not None else []
    col_tgt  += list(w_tgt)  if w_tgt  is not None else []
    col_pred += list(w_pred) if w_pred is not None else []
    col_fm   += list(w_fm)   if w_fm   is not None else []
    col_t0   += list(w_t0.ravel()) if w_t0 is not None else [0]*n_w
    col_n    += list(w_n.ravel())  if w_n  is not None else [1]*n_w
    col_split += ["warm"] * n_w

    # 固定 vis_range（給定）→ 全欄同對稱尺度，跨 run 可比；否則 per-sample GT 範圍
    if vor_range is not None:
        col_a = [float(vor_range)] * N
    else:
        col_a = [gt_vor_range(col_tgt[i], col_fm[i])[1] for i in range(N)]
    col_emax = [err_range(col_pred[i], col_tgt[i], col_fm[i])[1] for i in range(N)]

    error_per_col = [
        np.abs(col_pred[i] - col_tgt[i]) * (col_fm[i] > 0.5)
        for i in range(N)
    ]
    row_data_per_col = [
        [col_ic[i], col_tgt[i], col_pred[i], error_per_col[i]]
        for i in range(N)
    ]

    # 列標籤依訓練模式
    if is_mean:
        row_labels = ["IC (parent mean)", "GT (mean flow)", "Pred (mean)", "|Error|"]
    elif is_turb:
        row_labels = ["IC (mean+fluct t₀)", "GT (mean+fluct)", "Pred (mean+fluct)", "|Error|"]
    else:
        row_labels = ["IC (t₀)", "GT (t₀+n)", "Predicted", "|Error|"]

    n_rows = 4
    aspect = 640 / 360
    cell_h = 2.0
    cell_w = cell_h * aspect
    fig_w = cell_w * N + 0.9
    fig_h = cell_h * n_rows + 1.0

    fig, axes = plt.subplots(
        n_rows, N,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.04, "wspace": 0.04},
    )
    axes = np.atleast_2d(axes)

    for row_idx in range(n_rows):
        is_err = row_idx == 3
        for col_idx in range(N):
            ax = axes[row_idx, col_idx]
            fm_hw = col_fm[col_idx]
            f_field = row_data_per_col[col_idx][row_idx]
            a = col_a[col_idx]

            if is_err:
                f = f_field.copy()
                f[fm_hw < 0.5] = np.nan
                cmap_obj = plt.get_cmap("Greys").copy()
                cmap_obj.set_bad(color="#b0b0b0")
                ax.imshow(f, cmap=cmap_obj, vmin=0.0, vmax=col_emax[col_idx],
                          origin="upper", interpolation="nearest")
                ax.text(0.02, 0.98, f"<={col_emax[col_idx]:.3f}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=6,
                        color="#222", bbox=dict(facecolor="white", alpha=0.6,
                                                edgecolor="none", pad=0.5))
            else:
                f_clean = np.where(fm_hw > 0.5, f_field, 0.0)
                rgb = _to_rgb_vor(f_clean, fm_hw, -a, a)
                ax.imshow(rgb, origin="upper", interpolation="nearest")

            ax.axis("off")
            if col_idx == 0:
                ax.text(-0.06, 0.5, row_labels[row_idx],
                        transform=ax.transAxes,
                        va="center", ha="right", fontsize=9, fontweight="bold")
            if row_idx == 0:
                split_label = col_split[col_idx]
                t0_val = int(col_t0[col_idx])
                n_val  = int(col_n[col_idx])
                color = "#1a6fcf" if split_label == "cold" else "#c0392b"
                if is_mean:
                    n_label = f"k={n_val}(mean)"
                elif is_turb:
                    n_label = f"n={n_val}(turb)"
                else:
                    n_label = f"n={n_val}"
                ax.set_title(f"[{split_label}] t₀={t0_val} {n_label}\n±{a:.3f}",
                             fontsize=7, pad=3, color=color)

        # cold|warm 垂直分隔線（僅 cold+warm 混合時顯示）
        if has_cold and has_warm and row_idx == 0:
            ax_div = axes[row_idx, n_c - 1]
            ax_div.plot(
                [1.0, 1.0], [0.0, 1.0],
                color="#888888", linewidth=1.5,
                transform=ax_div.transAxes, clip_on=False,
            )

    if is_mean:
        gt_desc = "GT=mean_flow_post_conv"
        mode_tag = "[mean module]"
    elif is_turb:
        gt_desc = "GT=mean+fluct (full U)"
        mode_tag = "[turb module]"
    else:
        gt_desc = "random t₀, random n"
        mode_tag = ""

    if vor_range is not None:
        scale_desc = f"fixed vis_range ±{vor_range:.3f} (跨 run 可比)"
    else:
        scale_desc = "per-sample GT-scaled"
    fig.suptitle(
        f"Step: {vis_step} | Val Vorticity {mode_tag} ({scale_desc})  "
        f"[{split_desc}]  {gt_desc}\n"
        f"判讀：Pred 渦結構/方向應對齊 GT；|Error| 越暗越好 ↓",
        fontsize=10, y=1.002,
    )
    save_plot_versions(fig, plot_dir, "val_vor_field_compare", step, dpi=110, bbox_inches="tight")
    plt.close(fig)
