"""mean_eval_sequence.py — rollout 序列圖（rows: vel / vort / error，cols: IC→序列→target）。

png：``mean_eval_sequence.png``（mean_module）/ ``turb_sequence.png``（turb_module，共用
plotter）。由 trainer 的 rollout 視覺化流程餵場資料呼叫（非走 ``plot_all`` 的 metrics-reader
慣例；此圖需模型 rollout 場資料）。

收斂判定（mean_module）用 ``mae_ssim_converged``：pred 對 **time-average** mean target 的
MAE/SSIM 達視覺門檻即收斂。LBM 的 2-window KE 判準是為**瞬態渦街場**（振盪 limit-cycle）
設計，mean_module 收斂到的是 time-average 場（無振盪），半窗均值差恆為 ~0 而失去鑑別力，
故此處不適用。``two_window_converged`` 保留供 turb_module 使用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm

__all__ = [
    "plot_eval_sequence",
    "plot_eval_sequence_multi",
    "two_window_converged",
    "mae_ssim_converged",
    "_solid_nan",
]

# mean_module 收斂門檻（嚴格視覺正確；tunable）。norm 空間 MAE（9ch L1）+ struct_vel RGB SSIM。
_CONV_MAE_THR: float = 0.03      # ≤ → 視為達標（對應 VAL mean_mse gate <0.05 的 MAE 量級）
_CONV_SSIM_THR: float = 0.90     # ≥ → 視為達標（對齊 mean_jx_ssim gate 0.90）


def mae_ssim_converged(
    mae_history: Sequence[float],
    ssim_history: Sequence[float],
    mae_thr: float = _CONV_MAE_THR,
    ssim_thr: float = _CONV_SSIM_THR,
) -> tuple[bool, int]:
    """time-average 場收斂判定：標記 pred 對 mean target **最相似（最佳 plateau onset）** 的 snapshot。

    標的是 **SSIM 峰值 plateau 的 onset**（第一個達 ``max_ssim − tol`` 的 step），即場已達
    「best stable」之處。mean 場 rollout 會持續改善，若改標「第一個跨門檻」的 step 會標得太早
    ——其後的 snapshot 視覺反而更像 target。過演化（SSIM 升後降）時 argmax 自動取峰值前的
    plateau，不會標到退化段。

    Args:
        mae_history: 各 rollout step 的 MAE（與 ssim_history 平行，index i ↔ step i+1）。
        ssim_history: 各 rollout step 的 struct_vel RGB SSIM。
        mae_thr: MAE 收斂門檻（norm 空間；在 onset step 須 ≤ 此值才算達標）。
        ssim_thr: SSIM 收斂門檻（rollout 內 **峰值** 須 ≥ 此值才宣告收斂）。

    Returns:
        ``(converged, onset_idx)``；未達標回 ``(False, -1)``。onset_idx 為 history 索引（best plateau 起點）。
    """
    n = min(len(mae_history), len(ssim_history))
    if n == 0:
        return False, -1
    # 峰值 SSIM = 最相似時刻；tie-break 取最早（plateau onset）
    best_i = max(range(n), key=lambda i: ssim_history[i])
    best_ssim = ssim_history[best_i]
    tol = 0.01  # plateau 容差：第一個達 (峰值 − tol) 的 step = best-stable onset
    onset_i = next((i for i in range(n) if ssim_history[i] >= best_ssim - tol), best_i)
    if best_ssim >= ssim_thr and mae_history[onset_i] <= mae_thr:
        return True, onset_i
    return False, -1


def _solid_nan(field: np.ndarray, solid: np.ndarray) -> np.ndarray:
    out = np.squeeze(field).astype(np.float64).copy()
    if solid is not None and out.shape == solid.shape:
        out[solid] = np.nan
    return out


def two_window_converged(
    ke_history: Sequence[float],
    ctu_history: Sequence[float],
    window_ctu: float = 5.0,
    eps: float = 0.05,
) -> tuple[bool, float]:
    """LBM 權威 2-window KE 收斂判定（複刻 lbmnca.lbm check_ke_convergence_twowindow）。

    取最近 2×window_ctu 切兩半比均值，對渦街振盪魯棒（半窗平均抵消振幅、留慢漂移）。

    Args:
        ke_history: 各 snapshot 的 KE 值（與 ctu_history 平行）。
        ctu_history: 各 snapshot 的 CTU 值。
        window_ctu: 半窗 CTU 大小（每半窗應 ≥1 shedding 週期，~5 CTU）。
        eps: |Δmean|/mean 收斂門檻（~0.05）。

    Returns:
        ``(converged, ratio)``。
    """
    n = min(len(ke_history), len(ctu_history))
    if n < 4:
        return False, 1.0
    ctu_arr = np.asarray(ctu_history[:n], dtype=np.float64)
    ke_arr = np.asarray(ke_history[:n], dtype=np.float64)
    t_end = float(ctu_arr[-1])
    t_mid = t_end - window_ctu
    t_start = t_end - 2.0 * window_ctu
    if t_start < float(ctu_arr[0]):
        return False, 1.0
    m1_mask = (ctu_arr > t_start) & (ctu_arr <= t_mid)
    m2_mask = (ctu_arr > t_mid) & (ctu_arr <= t_end)
    if int(m1_mask.sum()) < 2 or int(m2_mask.sum()) < 2:
        return False, 1.0
    m1 = float(np.mean(ke_arr[m1_mask]))
    m2 = float(np.mean(ke_arr[m2_mask]))
    ratio = abs(m2 - m1) / (abs(0.5 * (m1 + m2)) + 1e-10)
    return ratio < eps, ratio


def plot_eval_sequence(
    cols: Sequence[dict],
    solid: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """rollout 序列圖：rows = vel / vort / error，cols = IC → 序列 → target。

    Args:
        cols: 每欄 dict{label, vel[H,W], vort[H,W], err[H,W]|None, color}。
        solid: ``[H, W]`` bool，solid 區（plot 設 nan）。
        title: 圖標題。
        out_path: 輸出 PNG 路徑。
    """
    n = len(cols)
    if n == 0:
        return

    vels = [_solid_nan(c["vel"], solid) for c in cols]
    vors = [_solid_nan(c["vort"], solid) for c in cols]
    errs = [(_solid_nan(c["err"], solid) if c.get("err") is not None else None)
            for c in cols]

    vel_pos = np.concatenate([x[np.isfinite(x) & (x > 0)].ravel() for x in vels])
    vmin = max(np.percentile(vel_pos, 1), 1e-4) if vel_pos.size else 1e-4
    vmax = vel_pos.max() if vel_pos.size else 1.0
    vel_norm = LogNorm(vmin=vmin, vmax=vmax)

    vor_abs = max((float(np.nanmax(np.abs(x))) for x in vors
                   if np.isfinite(np.nanmax(np.abs(x)))), default=1.0)
    vor_abs = vor_abs if vor_abs > 0 else 1.0
    vor_norm = SymLogNorm(linthresh=max(vor_abs * 1e-2, 1e-4),
                          vmin=-vor_abs, vmax=vor_abs, base=10)

    err_vals = np.concatenate([x[np.isfinite(x)].ravel()
                               for x in errs if x is not None]) \
        if any(x is not None for x in errs) else np.array([])
    err_max = float(np.percentile(err_vals, 99)) if err_vals.size else 1.0
    err_max = err_max if err_max > 0 else 1.0

    fig, axes = plt.subplots(3, n, figsize=(2.4 * n + 1.2, 8),
                             constrained_layout=True, squeeze=False)
    im_vel = im_vor = im_err = None
    for col, c in enumerate(cols):
        clr = c.get("color", "black")
        # 收斂時點：紅色粗框 heatmap border（覆寫該欄色），標題加「✓CONV」紅字
        is_conv = bool(c.get("converged", False))
        border_clr = "red" if is_conv else clr
        border_lw = 4.0 if is_conv else 2.0
        im_vel = axes[0, col].imshow(vels[col], norm=vel_norm, cmap="viridis",
                                     origin="upper", aspect="equal")
        im_vor = axes[1, col].imshow(vors[col], norm=vor_norm, cmap="RdBu_r",
                                     origin="upper", aspect="equal")
        if errs[col] is not None:
            im_err = axes[2, col].imshow(errs[col], vmin=0, vmax=err_max,
                                         cmap="inferno", origin="upper", aspect="equal")
        else:
            axes[2, col].axis("off")
        _ttl = c["label"] + ("\n✓CONV" if is_conv else "")
        axes[0, col].set_title(_ttl, fontsize=9, color=border_clr,
                               fontweight=("bold" if is_conv else "normal"))
        for r in range(3):
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
            for sp in axes[r, col].spines.values():
                sp.set_color(border_clr); sp.set_linewidth(border_lw)
    axes[0, 0].set_ylabel("vel_mag (log)", fontsize=10)
    axes[1, 0].set_ylabel("vorticity (symlog)", fontsize=10)
    axes[2, 0].set_ylabel("|pred − target| vel", fontsize=10)
    if im_vel is not None:
        fig.colorbar(im_vel, ax=axes[0, :].tolist(), fraction=0.012, pad=0.01)
    if im_vor is not None:
        fig.colorbar(im_vor, ax=axes[1, :].tolist(), fraction=0.012, pad=0.01)
    if im_err is not None:
        fig.colorbar(im_err, ax=axes[2, :].tolist(), fraction=0.012, pad=0.01)
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _shared_norms(
    vels: Sequence[np.ndarray],
    vors: Sequence[np.ndarray],
    errs: Sequence[Optional[np.ndarray]],
) -> tuple[LogNorm, SymLogNorm, float]:
    """跨全部欄位算共用 color norm（vel log / vort symlog / err max）。

    多 sample 比較圖用同一 norm → 不同 Re 的能量差異可直接目視比較
    （各自 normalize 會抹平 Re 間的量級差）。

    Args:
        vels: 各欄 vel_mag（solid 已設 nan）。
        vors: 各欄 vorticity（solid 已設 nan）。
        errs: 各欄 error（None 表無誤差欄）。

    Returns:
        ``(vel_norm, vor_norm, err_max)``。
    """
    vel_pos = np.concatenate([x[np.isfinite(x) & (x > 0)].ravel() for x in vels]) \
        if vels else np.array([])
    vmin = max(np.percentile(vel_pos, 1), 1e-4) if vel_pos.size else 1e-4
    vmax = vel_pos.max() if vel_pos.size else 1.0
    vel_norm = LogNorm(vmin=vmin, vmax=vmax)

    vor_abs = max((float(np.nanmax(np.abs(x))) for x in vors
                   if np.isfinite(np.nanmax(np.abs(x)))), default=1.0)
    vor_abs = vor_abs if vor_abs > 0 else 1.0
    vor_norm = SymLogNorm(linthresh=max(vor_abs * 1e-2, 1e-4),
                          vmin=-vor_abs, vmax=vor_abs, base=10)

    err_vals = np.concatenate([x[np.isfinite(x)].ravel()
                               for x in errs if x is not None]) \
        if any(x is not None for x in errs) else np.array([])
    err_max = float(np.percentile(err_vals, 99)) if err_vals.size else 1.0
    err_max = err_max if err_max > 0 else 1.0
    return vel_norm, vor_norm, err_max


def plot_eval_sequence_multi(
    samples: Sequence[dict],
    title: str,
    out_path: Path,
) -> None:
    """多 sample（不同 Re）rollout 序列比較圖。

    每個 sample 一個 3-row block（vel / vort / err），垂直堆疊，左側 ylabel 標 Re。
    **所有 sample 共用同一 color norm**（vel/vort/err）→ 跨 Re 能量差異可直接目視比較。
    sample 依 Re 升序排列。

    Args:
        samples: list of dict{re: float, cols: Sequence[dict], solid: np.ndarray[H,W] bool}。
                 cols 格式同 ``plot_eval_sequence``（label/vel/vort/err/color/converged）。
        title: 圖標題。
        out_path: 輸出 PNG 路徑。
    """
    samples = [s for s in samples if s.get("cols")]
    if not samples:
        return
    samples = sorted(samples, key=lambda s: s.get("re", 0.0))
    n_samp = len(samples)
    n_cols = max(len(s["cols"]) for s in samples)

    # 各 sample solid-nan 後的場（cache 供 norm + render）
    prep: list[dict] = []
    all_vels: list[np.ndarray] = []
    all_vors: list[np.ndarray] = []
    all_errs: list[Optional[np.ndarray]] = []
    for s in samples:
        solid = s.get("solid")
        vels = [_solid_nan(c["vel"], solid) for c in s["cols"]]
        vors = [_solid_nan(c["vort"], solid) for c in s["cols"]]
        errs = [(_solid_nan(c["err"], solid) if c.get("err") is not None else None)
                for c in s["cols"]]
        prep.append({"vels": vels, "vors": vors, "errs": errs, "cols": s["cols"],
                     "re": s.get("re", 0.0)})
        all_vels.extend(vels); all_vors.extend(vors); all_errs.extend(errs)

    vel_norm, vor_norm, err_max = _shared_norms(all_vels, all_vors, all_errs)

    n_rows = 3 * n_samp
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.4 * n_cols + 1.6, 2.0 * n_rows),
                             constrained_layout=True, squeeze=False)
    im_vel = im_vor = im_err = None
    for si, p in enumerate(prep):
        r0 = 3 * si
        for col, c in enumerate(p["cols"]):
            clr = c.get("color", "black")
            is_conv = bool(c.get("converged", False))
            border_clr = "red" if is_conv else clr
            border_lw = 4.0 if is_conv else 2.0
            im_vel = axes[r0, col].imshow(p["vels"][col], norm=vel_norm, cmap="viridis",
                                          origin="upper", aspect="equal")
            im_vor = axes[r0 + 1, col].imshow(p["vors"][col], norm=vor_norm, cmap="RdBu_r",
                                              origin="upper", aspect="equal")
            if p["errs"][col] is not None:
                im_err = axes[r0 + 2, col].imshow(p["errs"][col], vmin=0, vmax=err_max,
                                                  cmap="inferno", origin="upper",
                                                  aspect="equal")
            else:
                axes[r0 + 2, col].axis("off")
            if si == 0:
                _ttl = c["label"] + ("\n✓CONV" if is_conv else "")
                axes[r0, col].set_title(_ttl, fontsize=9, color=border_clr,
                                        fontweight=("bold" if is_conv else "normal"))
            for rr in range(3):
                axes[r0 + rr, col].set_xticks([]); axes[r0 + rr, col].set_yticks([])
                for sp in axes[r0 + rr, col].spines.values():
                    sp.set_color(border_clr); sp.set_linewidth(border_lw)
        # 多畫的空欄關掉（sample 欄數不一時）
        for col in range(len(p["cols"]), n_cols):
            for rr in range(3):
                axes[r0 + rr, col].axis("off")
        axes[r0, 0].set_ylabel(f"Re={p['re']:.0f}\nvel (log)", fontsize=9)
        axes[r0 + 1, 0].set_ylabel("vort (symlog)", fontsize=9)
        axes[r0 + 2, 0].set_ylabel("|pred-tgt|", fontsize=9)
    if im_vel is not None:
        fig.colorbar(im_vel, ax=axes[0::3].ravel().tolist(), fraction=0.010, pad=0.01)
    if im_vor is not None:
        fig.colorbar(im_vor, ax=axes[1::3].ravel().tolist(), fraction=0.010, pad=0.01)
    if im_err is not None:
        fig.colorbar(im_err, ax=axes[2::3].ravel().tolist(), fraction=0.010, pad=0.01)
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
