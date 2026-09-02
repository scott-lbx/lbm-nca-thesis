"""Plot: delta_ctu 分層 visual audit。

版面：欄 = Target | IC | Pred | Error，列 = 3 bucket{0.2,1.0,2.0 CTU} × 3 代表{worst/median/best}。
case 選取依 per-case fluid-masked vel MAE。vel 與 vort 各一張。

需 val_fields_latest.npz 含 {split}_delta_ctu（由 trainer 記錄）。缺則跳過。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._range import err_range, gt_vel_range, gt_vor_range
from ._utils import ensure_plot_dir, save_plot_versions

_BUCKETS = [
    ("0.2 CTU", 0.1, 0.5),
    ("1.0 CTU", 0.5, 1.5),
    ("2.0 CTU", 1.5, 3.0),
]


def _load_cases(npz, field: str):
    """彙整 cold+warm 兩 split 的 per-case 陣列。回傳 list of dict 或 None。"""
    cases = []
    for split in ("cold", "warm"):
        dkey = f"{split}_delta_ctu"
        pkey = f"{split}_pred_{field}"
        tkey = f"{split}_target_{field}"
        ikey = f"{split}_ic_{field}"
        mkey = f"{split}_fluid_mask"
        if dkey not in npz or pkey not in npz:
            continue
        delta = np.asarray(npz[dkey]).reshape(-1)
        pred = np.asarray(npz[pkey])
        tgt = np.asarray(npz[tkey])
        ic = np.asarray(npz[ikey])
        mask = np.asarray(npz[mkey]) if mkey in npz else np.ones_like(pred)
        n = min(len(delta), pred.shape[0])
        for i in range(n):
            m = mask[i]
            denom = max(float(m.sum()), 1.0)
            err = float((np.abs(pred[i] - tgt[i]) * m).sum() / denom)
            cases.append({
                "split": split, "delta": float(delta[i]),
                "target": tgt[i], "ic": ic[i], "pred": pred[i],
                "mask": m, "err": err,
            })
    return cases or None


def _pick(cases: list[dict], lo: float, hi: float):
    """bucket 內依 err 取 worst/median/best。"""
    grp = sorted([c for c in cases if lo <= c["delta"] < hi], key=lambda c: c["err"])
    if not grp:
        return []
    worst = grp[-1]
    best = grp[0]
    median = grp[len(grp) // 2]
    return [("worst", worst), ("median", median), ("best", best)]


def _plot_field(cases, field, plot_dir, step, symmetric, cmap, is_turb: bool = False):
    """每列（一個 case）用該列 GT 推導 vmin/vmax，Target/IC/Pred 共用同 scale。"""
    rows = []
    for bname, lo, hi in _BUCKETS:
        for tag, case in _pick(cases, lo, hi):
            rows.append((f"{bname}\n{tag} ({case['split']})\nerr={case['err']:.3g}", case))

    # mean_module fallback: NaN delta_ctu → no CTU bucket，改顯示 worst/median/best
    is_mean_fallback = False
    if not rows and cases:
        sorted_c = sorted(cases, key=lambda c: c["err"])
        best = sorted_c[0]
        median = sorted_c[len(sorted_c) // 2]
        worst = sorted_c[-1]
        rows = [
            (f"worst (err={worst['err']:.3g})", worst),
            (f"median (err={median['err']:.3g})", median),
            (f"best (err={best['err']:.3g})", best),
        ]
        is_mean_fallback = True

    if not rows:
        return

    # 欄位標籤依訓練模式
    if is_mean_fallback:
        col_labels = ["Target (mean flow)", "IC (parent mean)", "Pred (mean)", "Error"]
    elif is_turb:
        col_labels = ["Target (mean+fluct)", "IC (mean+fluct)", "Pred (mean+fluct)", "Error"]
    else:
        col_labels = ["Target", "IC", "Pred", "Error"]

    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 2.7 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, 4)
    for j, c in enumerate(col_labels):
        axes[0, j].set_title(c, fontsize=11)

    for i, (label, case) in enumerate(rows):
        m = case["mask"] > 0.5
        if symmetric:
            vmin, vmax = gt_vor_range(case["target"], case["mask"])
            scale_txt = f"±{vmax:.3f}"
        else:
            vmin, vmax = gt_vel_range(case["target"], case["mask"])
            scale_txt = f"vmax={vmax:.3f}"
        err_lim = err_range(case["pred"], case["target"], case["mask"])[1]
        for j, key in enumerate(["target", "ic", "pred"]):
            f = np.where(m, case[key], np.nan)
            axes[i, j].imshow(f, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        err = np.where(m, np.abs(case["pred"] - case["target"]), np.nan)
        im = axes[i, 3].imshow(err, cmap="magma", vmin=0.0, vmax=err_lim, origin="lower")
        axes[i, 3].set_xticks([]); axes[i, 3].set_yticks([])
        axes[i, 0].set_ylabel(f"{label}\n{scale_txt}", fontsize=8)
        fig.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

    if is_mean_fallback:
        title = f"Mean Flow {field.upper()} Audit [step={step}] — worst/median/best"
    elif is_turb:
        title = f"Turb (mean+fluct) {field.upper()} Audit [step={step}] — worst/median/best per CTU bucket"
    else:
        title = f"Delta-CTU {field.upper()} Audit [step={step}] — worst/median/best per bucket"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    save_plot_versions(fig, plot_dir, f"delta_ctu_{field}_audit", step, dpi=100)
    plt.close(fig)


def plot(run_dir: str, step: int | None = None) -> None:
    npz_path = Path(run_dir) / "val_fields_latest.npz"
    if not npz_path.exists():
        return
    with np.load(npz_path) as npz:
        if not any(f"{s}_delta_ctu" in npz for s in ("cold", "warm")):
            return
        is_turb = bool(npz["turb_module_mode"])
        plot_dir = ensure_plot_dir(run_dir)
        vel_cases = _load_cases(npz, "vel")
        if vel_cases:
            _plot_field(vel_cases, "vel", plot_dir, step, symmetric=False,
                        cmap="viridis", is_turb=is_turb)
        vor_cases = _load_cases(npz, "vor")
        if vor_cases:
            _plot_field(vor_cases, "vor", plot_dir, step, symmetric=True,
                        cmap="RdBu_r", is_turb=is_turb)
