"""TES-NCA M-series 階段審閱視覺化評估圖表模組。

所有公開函式接受預先計算的 numpy array，輸出 PNG 檔案。
不依賴 model import 或 torch——純 numpy + matplotlib。
由 TESTrainer._generate_checkpoint_images() 呼叫。

每個 checkpoint（每隔 ckpt_interval 步）產生以下圖表：
    plot_sequence_strip()       — 時序演化條（GT / Pred / |Error|）
    plot_hband_spectrum()       — 2D FFT HBand 偽影診斷
    plot_icswap_demo()          — warm-start IC 演化示範
    plot_stratified_report()    — VIS-GATE 各層 bar chart（按 Re / mode 分層）
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matplotlib 延遲載入
# ---------------------------------------------------------------------------


def _mpl():
    """延遲載入 matplotlib 並強制 Agg backend + 繁體中文字型。

    Returns:
        已設定好字型與 backend 的 matplotlib.pyplot module。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as _fm

    # 繁中可顯示字型（優先 JhengHei；避免 CJK glyph 缺字方框）
    _avail = {f.name for f in _fm.fontManager.ttflist}
    for _f in ("Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"):
        if _f in _avail:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break
    return plt


# ---------------------------------------------------------------------------
# 時序演化條帶圖
# ---------------------------------------------------------------------------


def plot_sequence_strip(
    gt_frames: list[np.ndarray],      # N × [H, W] vel_mag，值域 [0, 1]
    pred_frames: list[np.ndarray],    # N × [H, W] vel_mag，值域 [0, 1]
    frame_labels: list[str],          # N 個字串，例如 ["t=0","t=3","t=6",...]
    case_info: str,                   # 子標題（case_key + ic_type + Re）
    output_path: Path | str,
    error_vmax: float = 0.25,
) -> None:
    """3 列（GT / Pred / |Error|）× N 欄時序演化條帶圖。

    各列說明：
        Row 0 — GT（LBM 參考解）
        Row 1 — Pred（TES 模型預測）
        Row 2 — |Error|（絕對差值，hot colormap）

    Args:
        gt_frames: N 個 [H, W] GT 速度量 array，值域 [0, 1]。
        pred_frames: N 個 [H, W] 預測速度量 array，值域 [0, 1]。
        frame_labels: 每欄的時間標籤字串列表，長度須與 gt_frames 相同。
        case_info: 圖表主標題文字（通常包含 case_key、ic_type、Re 資訊）。
        output_path: 輸出 PNG 路徑；上層目錄若不存在則自動建立。
        error_vmax: |Error| 行的 colormap 上限值，預設 0.25。

    Returns:
        None。直接將圖表寫入 output_path。
    """
    plt = _mpl()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(gt_frames)
    if n == 0:
        logger.warning("plot_sequence_strip: no frames provided, skipping %s", output_path)
        return

    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 8.0))
    if n == 1:
        axes = axes[:, np.newaxis]

    row_labels = ["GT (LBM)", "Pred (TES)", "|Error|"]
    cmaps      = ["viridis",  "viridis",    "hot"]
    vmaxs      = [1.0,         1.0,          error_vmax]

    for ci in range(n):
        gt   = np.clip(gt_frames[ci],   0, 1)
        pred = np.clip(pred_frames[ci], 0, 1)
        err  = np.abs(gt - pred)
        imgs = [gt, pred, err]

        for ri, (img, cmap, vmax) in enumerate(zip(imgs, cmaps, vmaxs)):
            ax = axes[ri, ci]
            ax.imshow(img, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(row_labels[ri], fontsize=9, labelpad=4)
            if ri == 0:
                ax.set_title(frame_labels[ci], fontsize=8)

    fig.suptitle(case_info, fontsize=9, y=1.005)
    plt.tight_layout()
    plt.savefig(output_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sequence strip → %s", output_path.name)


# ---------------------------------------------------------------------------
# HBand FFT 頻譜診斷圖
# ---------------------------------------------------------------------------


def plot_hband_spectrum(
    pred_vel_mag: np.ndarray,    # [H, W] float，速度量（原始或正規化）
    tgt_vel_mag: np.ndarray,     # [H, W] float
    fluid_mask: np.ndarray,      # [H, W] float 0/1
    case_info: str,
    output_path: Path | str,
) -> None:
    """2 面板 HBand FFT 偽影診斷圖。

    左面板：Pred 速度場的 2D FFT log-power，以青色虛線標出 kx=0 欄。
    右面板：kx=0 切片比較（Pred vs GT），超出量（偽影）以填色標示。

    判讀方式（HBand spectrum reading guide）：
      • 左圖：Pred 速度場的 2D FFT log-power（亮 = 高頻能量）。
        cyan 虛線標出 kx=0（純垂直條紋方向），若這條線特別亮表示
        Pred 出現「水平條紋偽影」。
      • 右圖：把左圖 kx=0 那條線抽出來，紅 = Pred、藍 = GT，
        x 軸 = ky（垂直空間頻率）。**紅線越貼藍線越好**；
        紅 > 藍 的範圍以淡紅色填滿 → 即 "Excess artifact" 偽影能量。
      • 右上角 "HBand excess (mean)" 是該偽影量的純量分數，
        **越低越好 ↓**（PASS 門檻 < 0.34）。

    Args:
        pred_vel_mag: [H, W] 預測速度量 array。
        tgt_vel_mag: [H, W] GT 速度量 array。
        fluid_mask: [H, W] 流體遮罩（0=固體，1=流體）。
        case_info: 圖表標題文字。
        output_path: 輸出 PNG 路徑；上層目錄若不存在則自動建立。

    Returns:
        None。直接將圖表寫入 output_path。
    """
    plt = _mpl()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _fft_power(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
        count = mask.sum()
        if count < 1:
            return np.zeros((field.shape[0], field.shape[1] // 2 + 1), dtype=np.float32)
        mean = (field * mask).sum() / count
        centered = (field - mean) * mask
        F = np.fft.rfft2(centered)
        return np.abs(F) ** 2  # [H, W//2+1]

    pred_power = _fft_power(pred_vel_mag, fluid_mask)
    tgt_power  = _fft_power(tgt_vel_mag,  fluid_mask)
    H = pred_power.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.suptitle(
        f"HBand FFT — {case_info}\n"
        f"判讀：右圖紅線(Pred)越貼藍線(GT)越好 ↓；紅色 'Excess artifact' 面積越小越好 ↓ "
        f"(HBand excess PASS<0.34)",
        fontsize=9,
    )

    # 面板 1：2D FFT log-power（Pred）
    log_pred = np.log1p(pred_power)
    axes[0].imshow(log_pred, aspect="auto", origin="lower", cmap="inferno")
    axes[0].axvline(x=0.5, color="cyan", linewidth=1.5, linestyle="--", alpha=0.9)
    axes[0].set_xlabel("kx index (horiz freq)", fontsize=9)
    axes[0].set_ylabel("ky index (vert freq)", fontsize=9)
    axes[0].set_title("FFT power of pred (log) — cyan: kx=0 axis", fontsize=9)

    # 面板 2：kx=0 切片（排除 DC ky=0）
    pred_kx0 = pred_power[1:, 0]
    tgt_kx0  = tgt_power[1:, 0]
    ky_range = np.arange(1, H)

    pred_total = pred_power[1:, :].sum() + 1e-12
    tgt_total  = tgt_power[1:, :].sum()  + 1e-12

    pred_frac = pred_kx0 / pred_total
    tgt_frac  = tgt_kx0  / tgt_total

    axes[1].plot(ky_range, pred_frac,  color="tomato",    linewidth=1.8, label="Pred (TES)")
    axes[1].plot(ky_range, tgt_frac,   color="steelblue", linewidth=1.8, label="GT (LBM)")
    axes[1].fill_between(
        ky_range, tgt_frac, pred_frac,
        where=(pred_frac > tgt_frac),
        alpha=0.35, color="tomato", label="Excess artifact",
    )
    axes[1].set_xlabel("ky index (vertical frequency)", fontsize=9)
    axes[1].set_ylabel("Fraction of total FFT power", fontsize=9)
    axes[1].set_title("kx=0 slice: artifactual HBand energy", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, max(H // 2, 1))

    # HBand 分數標注
    excess = float(np.clip(pred_frac - tgt_frac, 0, None).mean())
    axes[1].annotate(
        f"HBand excess (mean): {excess:.4f}",
        xy=(0.97, 0.95), xycoords="axes fraction",
        fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    logger.info("HBand spectrum → %s", output_path.name)


# ---------------------------------------------------------------------------
# Warm-start 演化示範圖
# ---------------------------------------------------------------------------


def plot_icswap_demo(
    ic_vel: np.ndarray,            # [H, W] t=0 速度量（warm IC，已套遮罩）
    pred_seq: list[np.ndarray],    # M × [H, W] 選定時步的預測速度量
    gt_seq: list[np.ndarray],      # M × [H, W] 相同時步的 GT 速度量
    frame_labels: list[str],       # M 個標籤，例如 ["t=1","t=3","t=6","final"]
    case_info: str,
    output_path: Path | str,
) -> None:
    """Warm-start 演化示範圖。

    展示：warm IC（t=0）→ Pred 演化  vs  warm IC（t=0）→ GT 演化。
    直接示範論文核心能力：接續前一版流場的自迴歸推進。

    Row 0（GT 路徑）：  IC | GT t=1 | GT t=3 | ... | GT final
    Row 1（Pred 路徑）：IC | Pred t=1 | Pred t=3 | ... | Pred final
    Row 2（Error）：    — | |Error| t=1 | ... | |Error| final

    Args:
        ic_vel: [H, W] warm IC 速度量 array（t=0）。
        pred_seq: M 個 [H, W] 預測速度量 array（選定時步）。
        gt_seq: M 個 [H, W] GT 速度量 array（與 pred_seq 對應相同時步）。
        frame_labels: 各 rollout 欄的時間標籤列表，長度須與 pred_seq 相同。
        case_info: 圖表主標題文字。
        output_path: 輸出 PNG 路徑；上層目錄若不存在則自動建立。

    Returns:
        None。直接將圖表寫入 output_path。
    """
    plt = _mpl()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rollout = len(pred_seq)
    total_cols = 1 + n_rollout  # [IC] + rollout 各欄
    col_labels = ["Warm IC\n(t=0)"] + frame_labels

    fig, axes = plt.subplots(3, total_cols, figsize=(2.6 * total_cols, 8.0))
    row_labels = ["GT (LBM)", "Pred (TES)", "|Error|"]

    for ri in range(3):
        # 第 0 欄：warm IC（各列共用）
        ax0 = axes[ri, 0]
        ax0.imshow(np.clip(ic_vel, 0, 1), cmap="viridis", vmin=0, vmax=1.0,
                   interpolation="nearest", origin="lower")
        ax0.set_xticks([])
        ax0.set_yticks([])
        if ri == 0:
            ax0.set_title(col_labels[0], fontsize=8)
        ax0.set_ylabel(row_labels[ri], fontsize=9, labelpad=4)

        for ci in range(n_rollout):
            gt_f   = np.clip(gt_seq[ci],   0, 1)
            pred_f = np.clip(pred_seq[ci], 0, 1)
            err_f  = np.abs(gt_f - pred_f)
            imgs   = [gt_f, pred_f, err_f]
            cmaps  = ["viridis", "viridis", "hot"]
            vmaxs  = [1.0, 1.0, 0.25]

            ax = axes[ri, 1 + ci]
            ax.imshow(imgs[ri], cmap=cmaps[ri], vmin=0, vmax=vmaxs[ri],
                      interpolation="nearest", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(col_labels[1 + ci], fontsize=8)

    fig.suptitle(f"Warm-start Evolution Demo — {case_info}", fontsize=9, y=1.005)
    plt.tight_layout()
    plt.savefig(output_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    logger.info("IC swap demo → %s", output_path.name)


# ---------------------------------------------------------------------------
# 分層 bar chart 報告
# ---------------------------------------------------------------------------


def plot_stratified_report(
    vis_gate_detailed: dict[str, Any],
    output_path: Path | str,
) -> None:
    """2 列 bar chart：VIS-GATE 指標按 ic_type（上）與 Re 四分位（下）分層。

    三欄：IC_Copy_Score、HorizontalBand_FFT、StructVel_SSIM。
    紅虛線標示 PASS 門檻。

    判讀方式（Stratified report reading guide）：
      • 紅虛線 = PASS 門檻；藍/紅 bar 為各分層 (cold/warm 或 Re 四分位) 的指標值。
      • IC（IC_Copy_Score）   ↑ 越高越好（PASS ≥ 0.70）：
          0=完全 IC-copy 沒演化；1=演化幅度與 GT 相當；>1=過度發散。
      • HBand                ↓ 越低越好（PASS ≤ 0.34）：
          速度場水平條紋偽影量，越接近 0 越乾淨。
      • SSIM（StructVel）    ↑ 越高越好（PASS ≥ 0.82）：
          結構相似度，越接近 1 越像 GT。
      • bar 上方 n=k 表示該層樣本數；任一層 < PASS 線即代表該情境失敗。

    Args:
        vis_gate_detailed: VIS-GATE 詳細結果 dict，包含以下 key：
            - "step" (int)：訓練步數。
            - "by_mode" (dict)：按 ic_type 分層的指標 dict。
            - "by_re_quartile" (dict)：按 Re 四分位分層的指標 dict。
        output_path: 輸出 PNG 路徑；上層目錄若不存在則自動建立。

    Returns:
        None。直接將圖表寫入 output_path。
    """
    plt = _mpl()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    step     = vis_gate_detailed.get("step", 0)
    by_mode  = vis_gate_detailed.get("by_mode", {})
    by_re    = vis_gate_detailed.get("by_re_quartile", {})

    metrics    = ["IC",    "HBand",  "SSIM"]
    thresholds = {"IC": 0.70, "HBand": 0.34, "SSIM": 0.82}
    colors     = {"IC": "#4878CF", "HBand": "#E24A33", "SSIM": "#348ABD"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6))
    fig.suptitle(
        f"Step {step} — Stratified VIS-GATE Metrics\n"
        f"判讀：紅虛線=PASS 門檻；IC↑ (≥0.70)、HBand↓ (≤0.34)、SSIM↑ (≥0.82)；"
        f"bar 上 n=樣本數",
        fontsize=10, y=1.01,
    )

    def _bar_row(ax_row: list[Any], strat: dict[str, Any], row_title_suffix: str) -> None:
        """繪製單列（3 個子圖）的 bar chart，每欄對應一個指標。

        Args:
            ax_row: 含 3 個 matplotlib Axes 物件的列表。
            strat: 分層資料 dict，格式為 {層名: {指標: 值, "n": 樣本數}}。
            row_title_suffix: 子圖標題後綴，說明分層方式（如 "mode (cold/warm)"）。
        """
        names = list(strat.keys())
        for mi, metric in enumerate(metrics):
            ax = ax_row[mi]
            vals = [strat.get(n, {}).get(metric, float("nan")) for n in names]
            ns   = [strat.get(n, {}).get("n", 0) for n in names]
            xs   = np.arange(len(names))
            bars = ax.bar(xs, [v if not np.isnan(v) else 0 for v in vals],
                          color=colors[metric], alpha=0.75, width=0.6)
            ax.axhline(thresholds[metric], color="red", linestyle="--",
                       linewidth=1.2, label=f"PASS={thresholds[metric]}")
            max_v = max([v for v in vals if not np.isnan(v)] or [1.0])
            ax.set_ylim(0, max(1.05, max_v * 1.12))
            ax.set_xticks(xs)
            ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
            ax.set_ylabel(metric, fontsize=9)
            ax.set_title(f"{metric} by {row_title_suffix}", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            for bar, n, v in zip(bars, ns, vals):
                label = f"n={n}" if not np.isnan(v) else "n/a"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    label, ha="center", fontsize=7,
                )

    _bar_row(list(axes[0]), by_mode, "mode (cold/warm)")
    _bar_row(list(axes[1]), by_re,   "Re quartile")

    plt.tight_layout()
    plt.savefig(output_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    logger.info("Stratified report → %s", output_path.name)
