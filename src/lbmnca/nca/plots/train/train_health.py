"""Plot: training health — grad_norm / lr / scaler_scale (非評估，僅穩定度檢視)。"""
from __future__ import annotations

import matplotlib.pyplot as plt

from ._utils import ensure_plot_dir, load_metrics, rolling_mean, save_plot_versions


def plot(run_dir: str, step: int | None = None) -> None:
    train, _ = load_metrics(run_dir)
    if len(train) < 2:
        return

    plot_dir = ensure_plot_dir(run_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cur_step = train[-1]["step"]
    fig.suptitle(f"Training Health [step={cur_step}] — 穩定度檢視（非評估指標）", fontsize=10)

    # grad_norm
    ax = axes[0]
    gr = [r for r in train if r.get("grad_norm") is not None]
    if len(gr) >= 2:
        gs = [r["step"] for r in gr]
        gn = [r["grad_norm"] for r in gr]
        ax.plot(gs, gn, alpha=0.15, linewidth=0.5, color="tab:purple")
        if len(gn) >= 50:
            idx, sm = rolling_mean(gn, 50)
            ax.plot([gs[i] for i in idx], sm, linewidth=1.3, color="tab:purple", label="avg50")
        gn_max = max(gn) if gn else 1.0
        clip_val = 1.0
        ax.axhline(clip_val, linestyle="--", linewidth=0.8, color="tab:red", alpha=0.5, label=f"clip({clip_val})")
        # 自動縮放：當實際梯度遠小於 clip 時，縮窄 y 範圍讓曲線可見
        ylim_top = max(gn_max * 1.5, clip_val * 0.05)
        ax.set_ylim(0, ylim_top)
        ax.legend(fontsize=7)
    ax.set_title("Grad Norm")

    # lr
    ax = axes[1]
    lr_r = [r for r in train if r.get("lr") is not None]
    if len(lr_r) >= 2:
        ax.plot([r["step"] for r in lr_r], [r["lr"] for r in lr_r], color="tab:brown", linewidth=1.2)
        ax.set_yscale("log")
    ax.set_title("Learning Rate")

    # scaler scale
    ax = axes[2]
    sc = [r for r in train if r.get("scaler_scale") is not None]
    if len(sc) >= 2:
        ax.plot([r["step"] for r in sc], [r["scaler_scale"] for r in sc], color="tab:gray", linewidth=1.0)
        ax.set_yscale("log")
    ax.set_title("AMP Scaler Scale")

    for ax in axes:
        ax.set_xlabel("step", fontsize=7)
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    save_plot_versions(fig, plot_dir, "train_health", step, dpi=110)
    plt.close(fig)
