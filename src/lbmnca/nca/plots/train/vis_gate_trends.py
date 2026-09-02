"""Plot: visual gate 趨勢（每 1000）— 資料源 vis_gate_history.jsonl。

不使用 total loss。展示分項 must-pass gate + cold/warm 結構指標趨勢。
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from ._utils import (
    ensure_plot_dir,
    load_vis_gate_history,
    load_vis_gate_split,
    save_plot_versions,
)


def _agg_series(vg, key):
    xs, ys = [], []
    for r in vg:
        v = r.get("aggregate", {}).get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != fv:  # NaN
            continue
        xs.append(r["step"])
        ys.append(fv)
    return xs, ys


def plot(run_dir: str, step: int | None = None) -> None:
    vg = load_vis_gate_history(run_dir)
    if len(vg) < 1:
        return

    plot_dir = ensure_plot_dir(run_dir)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cur = vg[-1]["step"]
    fig.suptitle(f"Visual Gate Trends [step={cur}] — PASS/FAIL 由分項 gate 決定（非 total loss）", fontsize=10)

    # [0,0] composite + strict (連續趨勢，僅參考)
    ax = axes[0, 0]
    for key, sty, c, lab in [("vg_composite", "o-", "tab:blue", "composite"),
                             ("vg_composite_strict", "s--", "tab:purple", "strict")]:
        xs, ys = _agg_series(vg, key)
        if xs:
            ax.plot(xs, ys, sty, markersize=3, linewidth=1.2, color=c, label=lab)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Composite (參考，不裁決)")

    # [0,1] gate n_pass
    ax = axes[0, 1]
    xs, ys = [], []
    for r in vg:
        g = r.get("gate", {})
        if "n_pass" in g:
            xs.append(r["step"])
            ys.append(g["n_pass"])
    if xs:
        ax.plot(xs, ys, "o-", markersize=3, linewidth=1.3, color="tab:green")
        nt = vg[-1].get("gate", {}).get("n_total", 5)
        ax.axhline(nt, linestyle="--", linewidth=0.7, color="tab:green", alpha=0.5, label=f"all pass ({nt})")
        ax.set_ylim(0, (nt or 5) + 0.5)
        ax.legend(fontsize=7)
    ax.set_title("Gate n_pass ↑ (must-pass 通過數)")

    # [0,2] IC
    ax = axes[0, 2]
    xs, ys = _agg_series(vg, "IC_Copy_Score")
    if xs:
        ax.plot(xs, ys, "o-", markersize=3, linewidth=1.2, color="tab:cyan")
    ax.axhline(0.55, linestyle="--", linewidth=0.7, color="gray", alpha=0.5, label="0.55")
    ax.axhline(1.10, linestyle="--", linewidth=0.7, color="tab:red", alpha=0.5, label="1.10")
    ax.legend(fontsize=7)
    ax.set_title("IC_Copy_Score (gate 0.55–1.10)")

    # [1,0] StructVel_SSIM cold/warm
    ax = axes[1, 0]
    for mode, c in [("cold", "tab:orange"), ("warm", "tab:red")]:
        xs, ys = load_vis_gate_split(run_dir, mode, "SSIM")
        if xs:
            ax.plot(xs, ys, "o-", markersize=3, linewidth=1.2, color=c, label=mode)
    ax.axhline(0.65, linestyle="--", linewidth=0.6, color="tab:orange", alpha=0.5, label="cold 0.65")
    ax.axhline(0.78, linestyle="--", linewidth=0.6, color="tab:red", alpha=0.5, label="warm 0.78")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6)
    ax.set_title("StructVel_SSIM cold/warm ↑")

    # [1,1] VortHighMask_IoU cold/warm
    ax = axes[1, 1]
    for mode, c in [("cold", "tab:orange"), ("warm", "tab:red")]:
        xs, ys = load_vis_gate_split(run_dir, mode, "VortIoU")
        if xs:
            ax.plot(xs, ys, "o-", markersize=3, linewidth=1.2, color=c, label=mode)
    ax.axhline(0.50, linestyle="--", linewidth=0.6, color="tab:orange", alpha=0.5, label="cold 0.50")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6)
    ax.set_title("VortHighMask_IoU cold/warm ↑")

    # [1,2] HBand agg / max_case
    ax = axes[1, 2]
    for key, c, lab in [("HorizontalBand_FFT", "tab:red", "agg"),
                        ("HBand_max_case", "tab:brown", "max_case")]:
        xs, ys = _agg_series(vg, key)
        if xs:
            ax.plot(xs, ys, "o-", markersize=3, linewidth=1.1, color=c, label=lab)
    ax.axhline(0.30, linestyle="--", linewidth=0.7, color="tab:brown", alpha=0.5, label="max 0.30")
    ax.legend(fontsize=6)
    ax.set_title("HBand agg / max_case ↓")

    for row in axes:
        for ax in row:
            ax.set_xlabel("step", fontsize=7)
            ax.tick_params(labelsize=7)
    plt.tight_layout()
    save_plot_versions(fig, plot_dir, "vis_gate_trends", step, dpi=120)
    plt.close(fig)
