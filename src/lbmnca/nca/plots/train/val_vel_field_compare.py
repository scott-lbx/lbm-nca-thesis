"""Plot: Velocity magnitude field comparison — per-sample GT-derived range.

Reads val_fields_latest.npz.
Layout: 8 columns — first 4 cold, last 4 warm.
Rows: IC (t0) / Ground Truth (t0+n) / Predicted / |Error|.

每欄（一個 sample）用「該欄 GT」推導的 vmin/vmax，IC/GT/pred 共用同 scale →
pred 過阻尼/暗淡會誠實顯暗（按 GT scale 畫，非各自 auto-range）。每欄量值寫在
欄標題（無單一共用 colorbar，因各欄 scale 不同）。詳見 _range.py。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ._range import err_range, gt_vel_range
from ._utils import ensure_plot_dir, save_plot_versions
from lbmnca.nca.utils.FieldColorizer import FieldColorizer

_N_COLD = 4
_N_WARM = 4

_colorizer = FieldColorizer()


def _to_rgb_vel(field_hw: np.ndarray, mask_hw: np.ndarray,
                vmin: float, vmax: float) -> np.ndarray:
    t = torch.from_numpy(field_hw.astype(np.float32)).unsqueeze(0)
    m = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0)
    rgb = _colorizer._colorize(t.unsqueeze(0), vmin, vmax, _colorizer._vel_lut, m.unsqueeze(0))
    return rgb[0].permute(1, 2, 0).numpy()


def _to_scalar_vel(arr: np.ndarray) -> np.ndarray:
    """Accept current [N,H,W] velocity magnitude or legacy [N,2,H,W] vectors."""
    if arr.ndim == 4 and arr.shape[1] == 2:
        return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
    return arr


def _load_split(data, prefix: str, n: int):
    def _key(k):
        full = f"{prefix}_{k}"
        if full not in data:
            return None
        arr = data[full][:n]
        return None if arr.size == 0 else arr

    return (
        _key("target_vel"), _key("ic_vel"), _key("pred_vel"),
        _key("fluid_mask"), _key("t0"), _key("n_steps"),
    )


def plot(run_dir: str, step: int | None = None, vel_max: float | None = None) -> None:
    """繪製 val 速度場對比。

    Args:
        run_dir: run 目錄（讀 val_fields_latest.npz）。
        step: 圖檔步數標記。
        vel_max: 給定則全欄用此固定值域 [0, vel_max]（vis_range SSOT，跨 run 可比）；
            None 則用 per-sample GT-derived 範圍（過阻尼診斷對比）。
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

    # fall back to legacy keys without split prefix
    if cold[0] is None and "target_vel" in data:
        arr = _to_scalar_vel(data["target_vel"])
        ic_arr = _to_scalar_vel(data["ic_vel"])
        pred_arr = _to_scalar_vel(data["pred_vel"])
        n_leg = len(arr)
        n_c = min(n_leg // 2, _N_COLD)
        n_w = min(n_leg - n_c, _N_WARM)
        fm = data["fluid_mask"] if "fluid_mask" in data else np.ones_like(arr)
        dummy_t0 = np.zeros(n_leg, dtype=np.int32)
        dummy_n = np.ones(n_leg, dtype=np.int32)
        cold = (arr[:n_c], ic_arr[:n_c], pred_arr[:n_c],
                fm[:n_c], dummy_t0[:n_c], dummy_n[:n_c])
        warm = (arr[n_c:n_c+n_w], ic_arr[n_c:n_c+n_w], pred_arr[n_c:n_c+n_w],
                fm[n_c:n_c+n_w], dummy_t0[n_c:n_c+n_w], dummy_n[n_c:n_c+n_w])

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

    # flatten per-column
    col_tgt = list(c_tgt) if c_tgt is not None else []
    col_ic = list(c_ic) if c_ic is not None else []
    col_pred = list(c_pred) if c_pred is not None else []
    col_fm = list(c_fm) if c_fm is not None else []
    col_t0 = list(c_t0.ravel()) if c_t0 is not None else [0] * n_c
    col_n = list(c_n.ravel()) if c_n is not None else [1] * n_c
    col_split = ["cold"] * n_c

    col_tgt += list(w_tgt) if w_tgt is not None else []
    col_ic += list(w_ic) if w_ic is not None else []
    col_pred += list(w_pred) if w_pred is not None else []
    col_fm += list(w_fm) if w_fm is not None else []
    col_t0 += list(w_t0.ravel()) if w_t0 is not None else [0] * n_w
    col_n += list(w_n.ravel()) if w_n is not None else [1] * n_w
    col_split += ["warm"] * n_w

    # 固定 vis_range（給定）→ 全欄同尺度，跨 run 可比；否則 per-sample GT 範圍
    if vel_max is not None:
        col_vmax = [float(vel_max)] * N
    else:
        col_vmax = [gt_vel_range(col_tgt[i], col_fm[i])[1] for i in range(N)]
    col_emax = [err_range(col_pred[i], col_tgt[i], col_fm[i])[1] for i in range(N)]

    plot_dir = ensure_plot_dir(run_dir)
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

    # 列標籤依訓練模式
    if is_mean:
        row_labels = ["IC (parent mean)", "GT (mean flow)", "Pred (mean)", "|Error|"]
    elif is_turb:
        row_labels = ["IC (mean+fluct t₀)", "GT (mean+fluct)", "Pred (mean+fluct)", "|Error|"]
    else:
        row_labels = ["IC (t₀)", "GT (t₀+n)", "Predicted", "|Error|"]

    for row_idx in range(n_rows):
        is_err = row_idx == 3
        for col_idx in range(N):
            ax = axes[row_idx, col_idx]
            fm_hw = col_fm[col_idx]
            vmax = col_vmax[col_idx]

            if row_idx == 0:
                field = col_ic[col_idx]
            elif row_idx == 1:
                field = col_tgt[col_idx]
            elif row_idx == 2:
                field = col_pred[col_idx]
            else:
                field = np.abs(col_pred[col_idx] - col_tgt[col_idx]) * (fm_hw > 0.5)

            if is_err:
                f = field.copy()
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
                f_clean = np.where(fm_hw > 0.5, field, 0.0)
                rgb = _to_rgb_vel(f_clean, fm_hw, 0.0, vmax)
                ax.imshow(rgb, origin="upper", interpolation="nearest")

            ax.axis("off")
            if col_idx == 0:
                ax.text(-0.06, 0.5, row_labels[row_idx],
                        transform=ax.transAxes,
                        va="center", ha="right", fontsize=9, fontweight="bold")
            if row_idx == 0:
                split_label = col_split[col_idx]
                t0_val = int(col_t0[col_idx])
                n_val = int(col_n[col_idx])
                color = "#1a6fcf" if split_label == "cold" else "#c0392b"
                if is_mean:
                    n_label = f"k={n_val}(mean)"
                elif is_turb:
                    n_label = f"n={n_val}(turb)"
                else:
                    n_label = f"n={n_val}"
                ax.set_title(f"[{split_label}] t₀={t0_val} {n_label}\nvmax={vmax:.3f}",
                             fontsize=7, pad=3, color=color)

    if is_mean:
        gt_desc = "GT=mean_flow_post_conv"
        mode_tag = "[mean module]"
    elif is_turb:
        gt_desc = "GT=mean+fluct (full U)"
        mode_tag = "[turb module]"
    else:
        gt_desc = "random t₀+n"
        mode_tag = ""

    if vel_max is not None:
        scale_desc = f"fixed vis_range [0,{vel_max:.3f}] (跨 run 可比)"
        read_desc = "判讀：全欄同固定 scale → 跨 run/case 可直接比亮度；Pred 越暗 = 越弱"
    else:
        scale_desc = "per-sample GT-scaled"
        read_desc = "判讀：每欄 IC/GT/Pred 同 scale → Pred 越暗 = 越過阻尼"
    fig.suptitle(
        f"Step: {vis_step} | Val Velocity {mode_tag} ({scale_desc})  "
        f"[{split_desc}]  {gt_desc}\n"
        f"{read_desc}；|Error| 越暗越好 ↓",
        fontsize=10, y=1.002,
    )
    save_plot_versions(fig, plot_dir, "val_vel_field_compare", step, dpi=110, bbox_inches="tight")
    plt.close(fig)
