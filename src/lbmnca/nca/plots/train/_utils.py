"""Shared utilities for training plots."""

from __future__ import annotations

import json
from pathlib import Path

import logging as _logging
import warnings as _warnings

import matplotlib

matplotlib.use("Agg")

# 缺字 substitution 也會走 warnings.warn（UserWarning，非 logging）；一併靜音。
_warnings.filterwarnings("ignore", message=r"Glyph \d+ .* missing from font")

import matplotlib.font_manager as _fm
import matplotlib.pyplot as _plt


def _setup_zh_font() -> None:
    # U+2212（unicode 減號）修復：
    #   - axes.unicode_minus=False → 座標軸負值 tick 用 ASCII '-'（CJK 字型無 U+2212）。
    #   - mathtext.fontset='dejavusans' → mathtext 的減號走 DejaVu（內建、含 U+2212），
    #     不再 fallback 到 'default'/STIXGeneral 而噴 glyph 缺字 warning。
    #   兩者無條件設定（即使找不到 CJK 字型也要生效）。
    _plt.rcParams["axes.unicode_minus"] = False
    _plt.rcParams["mathtext.fontset"] = "dejavusans"
    candidates = ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"]
    available = {f.name for f in _fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            _plt.rcParams["font.family"] = "sans-serif"
            _plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            break
    # matplotlib 缺字 substitution 屬純視覺 noise（render 正確）；靜音其 logger，
    # 避免 async vis log 被 "does not have a glyph"/"Substituting symbol" 洗版。
    _logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
    _logging.getLogger("matplotlib.mathtext").setLevel(_logging.ERROR)


_setup_zh_font()


def load_metrics(run_dir: str) -> tuple[list[dict], list[dict]]:
    """Parse metrics.json (新版) 或 metrics.json (舊 run) → (train_records, val_records)."""
    run_path = Path(run_dir)
    path = run_path / "metrics.json"
    if not path.exists():
        path = run_path / "metrics.jsonl"
    train: list[dict] = []
    val: list[dict] = []
    if not path.exists():
        return train, val
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = rec.get("event", "")
            if event == "train_step":
                train.append(rec)
            elif event == "val":
                val.append(rec)
    return train, val


def load_vis_gate_history(run_dir: str) -> list[dict]:
    """Parse eval_output/vis_gate_history.jsonl → list of records (one per vis step).

    Each record has keys: step, aggregate (IC_Copy_Score, StructVel_RGB_SSIM,
    HorizontalBand_FFT, HBand_Q1_low, HBand_max_case, vg_composite,
    vg_composite_strict), by_mode, by_re_quartile.
    """
    path = Path(run_dir) / "eval_output" / "vis_gate_history.jsonl"
    records: list[dict] = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def ensure_plot_dir(run_dir: str) -> Path:
    p = Path(run_dir) / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_plot_versions(
    fig, plot_dir: Path, stem: str, step: int | None, **savefig_kwargs
) -> None:
    """Save latest plot only (overwrite each time)."""
    fig.savefig(plot_dir / f"{stem}.png", **savefig_kwargs)


def rolling_mean(values: list[float], window: int) -> tuple[list[int], list[float]]:
    """Returns (indices_into_values, smoothed) via box filter."""
    import numpy as np

    if len(values) < window:
        return list(range(len(values))), [float(v) for v in values]
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    smooth = np.convolve(arr, kernel, mode="valid")
    return list(range(window - 1, len(values))), smooth.tolist()


def ema(values: list[float], alpha: float = 0.1) -> list[float]:
    """Exponential moving average (train 曲線去 batch 雜訊；HC-2a)。

    alpha 越小越平滑。回傳與輸入等長的平滑序列（對齊原 step）。
    """
    if not values:
        return []
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out


def load_vis_gate_split(run_dir: str, mode: str, key: str) -> tuple[list[int], list[float]]:
    """從 vis_gate_history.jsonl 取某 mode(cold/warm) 的某 by_mode metric 趨勢。

    mode: "cold" | "warm"；key: "SSIM" | "VortIoU" | "HBand" | ...
    回傳 (steps, values)，跳過 NaN/缺值。
    """
    import math

    steps: list[int] = []
    vals: list[float] = []
    for rec in load_vis_gate_history(run_dir):
        bm = rec.get("by_mode", {}).get(mode, {})
        v = bm.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        steps.append(int(rec["step"]))
        vals.append(fv)
    return steps, vals
