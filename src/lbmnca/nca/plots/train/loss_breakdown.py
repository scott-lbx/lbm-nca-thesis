"""Plot: loss breakdown — 每個分項 loss/MAE 一個 subplot，曲線分 train / eval_cold / eval_warm。

規格：
- 不畫 total loss、不畫 weighted stacked area——加權總和的高低沒有判讀意義。
- train 曲線用 EMA（batch 雜訊高）。train records 為 aggregate（不分 cold/warm）；
  metrics 若含 {key}_cold / {key}_warm 則自動加畫 train 分離曲線。
- eval 分 cold / warm（val records 已分 split）。
- 視覺 gate 指標（SSIM/IoU/IC/HBand）見 vis_gate_trends.png，不在此重複。
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt

from ._utils import ensure_plot_dir, ema, load_metrics, save_plot_versions

_NON_LOSS_KEYS = {
    "event", "step", "loss", "total", "k", "v", "loss_version", "lr", "t",
    "grad_norm", "scaler_scale", "gpu_mem_mb", "steps_per_sec",
    "t_start", "global_t_start", "nan_consec", "split",
    "dcs_enabled", "frame_window", "physical_horizon_ctu",
    "delta_ctu_min", "delta_ctu_mean", "delta_ctu_max",
    "delta_ctu_bucket_0p2", "delta_ctu_bucket_1p0", "delta_ctu_bucket_2p0",
}
# 9ch MAE + vel_mag 一律顯示（即使非加權 loss term）
_MAE_KEYS = ["rho_mae", "jx_mae", "jy_mae", "vel_mag_mae"]
# mean_module：loss 分項（無 w_ 前綴）+ VAL 端 SSIM/品質 metric。
# SSIM 用 VAL records 的 mean_jx_ssim（gate eval 計算；higher=better）——train-step
# SSIM 在 AMP fp16 下不穩，故以 VAL SSIM 為品質曲線。
# 診斷用的 jac_*（gate_calls／active_calls／ratio_last／raw_last）刻意不列：那是計數
# 與「最後一次的值」，與 total 不同尺度，畫進來會讓縱軸失去意義；要看請直接讀
# metrics.json。
_MEAN_MODULE_KEYS = [
    "mse", "ssim_term", "chuber", "stay", "delta_focus",
    "overflow", "fp_reg", "rel_l2",
    "mean_jx_ssim", "mean_rel_l2", "mean_drift_2k",
]
# turb_module：turb_step 回傳的逐項 loss/penalty + VAL 端品質 metric。
_TURB_MODULE_KEYS = [
    "mse", "mse_wf", "acc_term", "rij_term", "tke_term", "spec_term",
    "ke_ratio_pen", "ic_copy_pen", "hband_pen", "struct_term", "pod_term", "tvar_term",
    "fluct_ke_ratio", "turb_acc", "turb_mse",
]


def _active_loss_keys(records: list[dict]) -> list[str]:
    """偵測 active 分項：w_ 前綴的 loss term base name + 固定 MAE keys + mean_module keys。

    per-term 值通常記在 VAL records（train_step 僅 aggregate loss），故需掃 train+val。
    """
    semantic = set()
    for rec in records:
        for key, val in rec.items():
            if key.startswith("w_") and isinstance(val, (int, float)):
                base = key[2:]
                # 只在 base 真有 value series 時才加（否則如 w_f 無對應 "f" 值→空 panel）
                if base not in _NON_LOSS_KEYS and any(r.get(base) is not None for r in records):
                    semantic.add(base)
    keys = sorted(semantic)
    for k in _MAE_KEYS:
        if any(r.get(k) is not None for r in records) and k not in keys:
            keys.append(k)
    # mean_module + turb_module 直接記錄的分項
    for k in _MEAN_MODULE_KEYS + _TURB_MODULE_KEYS:
        if any(r.get(k) is not None for r in records) and k not in keys:
            keys.append(k)
    return keys


def _series(records: list[dict], key: str):
    xs, ys = [], []
    for r in records:
        v = r.get(key)
        if v is None:
            continue
        xs.append(r["step"])
        ys.append(float(v))
    return xs, ys


def plot(run_dir: str, step: int | None = None) -> None:
    train, val = load_metrics(run_dir)
    if len(train) < 2 and len(val) < 2:
        return
    active = _active_loss_keys(train + val)
    if not active:
        return

    val_cold = [r for r in val if r.get("split") == "cold"]
    val_warm = [r for r in val if r.get("split") == "warm"]

    plot_dir = ensure_plot_dir(run_dir)
    n = len(active)
    n_cols = min(4, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows), squeeze=False)
    cur = train[-1]["step"]
    fig.suptitle(
        f"Loss Breakdown [step={cur}] — 分項 ↓（train EMA / eval cold / eval warm）",
        fontsize=11,
    )

    for i, key in enumerate(active):
        ax = axes[i // n_cols][i % n_cols]
        # train aggregate (EMA)
        xs, ys = _series(train, key)
        if xs:
            ax.plot(xs, ys, alpha=0.10, linewidth=0.4, color="tab:blue")
            ax.plot(xs, ema(ys, 0.1), linewidth=1.2, color="tab:blue", label="train(EMA)")
        # forward-compatible train split
        for sk, c in [(f"{key}_cold", "tab:cyan"), (f"{key}_warm", "tab:pink")]:
            sx, sy = _series(train, sk)
            if sx:
                ax.plot(sx, ema(sy, 0.1), linewidth=1.0, color=c, label=sk.split("_")[-1] + "(tr)")
        # eval cold / warm
        for recs, c, lab in [(val_cold, "tab:orange", "eval_cold"), (val_warm, "tab:red", "eval_warm")]:
            ex, ey = _series(recs, key)
            if ex:
                ax.plot(ex, ey, "o-", markersize=3, linewidth=1.0, color=c, label=lab)
        ax.set_title(f"{key} ↓", fontsize=8)
        if ys and all(v > 0 for v in ys):
            ax.set_yscale("log")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("step", fontsize=6)
        if i == 0:
            ax.legend(fontsize=6)

    # hide unused
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    plt.tight_layout()
    save_plot_versions(fig, plot_dir, "loss_breakdown", step, dpi=120)
    plt.close(fig)
