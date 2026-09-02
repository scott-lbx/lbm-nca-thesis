"""個案視覺 Gate 指標模組：依 Reynolds number、G_vec 複雜度、IC 類型進行分層分析。

由 TESTrainer 呼叫，負責建立 vis_gate_detailed.json。
所有計算均為純 Python/NumPy/PyTorch，不執行模型前向傳播。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..utils.evidence import write_run_evidence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 個案逐一指標計算
# ---------------------------------------------------------------------------


def compute_per_case_visual_gate(
    pred_finals: list[Tensor],
    tgt_finals: list[Tensor],
    ic_states: list[Tensor],
    fluid_masks: list[Tensor],
    meta_list: list[dict[str, Any]],
    normalizer: Any,
    pod_basis: dict[str, Tensor] | None = None,
    pod_modes: int | None = None,
) -> list[dict[str, Any]]:
    """逐一計算每個 case 的 VIS-GATE 視覺指標。

    每個輸入 list 的元素對應一個 case（無 batch dim）。
    回傳與 meta_list 順序一致的 per-case 結果 dict 清單。

    Args:
        pred_finals: 每個元素為 [9, H, W] 正規化物理通道預測結果。
        tgt_finals: 每個元素為 [9, H, W] 正規化物理通道目標值。
        ic_states: 每個元素為 [9, H, W] 正規化初始條件（t=0 IC）。
        fluid_masks: 每個元素為 [1, H, W] float 流體遮罩。
        meta_list: 每個元素為含 {case_key, re, gvec_norm, ic_type, mask_id} 的 dict。
        normalizer: TESCondNormalizer 實例，供反正規化使用。
        pod_basis: POD 基底 dict（key 為通道名稱，value 為 Tensor），可為 None。
        pod_modes: 使用的 POD 模態數，可為 None（使用 pod_basis 全部）。

    Returns:
        與 meta_list 等長的 dict list，每個 dict 含 meta 欄位與各指標數值。
    """
    from .visual_metrics import (
        ic_copy_score,
        horizontal_band_score_fft,
        struct_vel_rgb_ssim,
        struct_vort_highmask_iou,
        struct_vort_rgb_ssim,
        vort_signed_corr,
        vort_topk_mae,
    )
    from .pod_metrics import compute_pod_metrics

    results: list[dict[str, Any]] = []
    for pred, tgt, ic, fm, meta in zip(pred_finals, tgt_finals, ic_states, fluid_masks, meta_list):
        p = pred.unsqueeze(0)   # [1, 9, H, W]
        t = tgt.unsqueeze(0)
        c = ic.unsqueeze(0)
        f = fm.unsqueeze(0) if fm.dim() == 3 else fm  # [1, 1, H, W]

        try:
            ic_s = ic_copy_score(p, c, t, f)
        except Exception as e:
            logger.debug("ic_copy_score failed: %s", e)
            ic_s = float("nan")

        try:
            hband = horizontal_band_score_fft(p, t, f, normalizer)
        except Exception as e:
            logger.debug("hband_fft failed: %s", e)
            hband = float("nan")

        try:
            ssim = struct_vel_rgb_ssim(p, t, normalizer, f)
        except Exception as e:
            logger.debug("struct_vel_ssim failed: %s", e)
            ssim = float("nan")

        try:
            vort_iou = struct_vort_highmask_iou(p, t, f, normalizer)
        except Exception as e:
            logger.debug("vort_highmask_iou failed: %s", e)
            vort_iou = float("nan")

        try:
            vort_ssim = struct_vort_rgb_ssim(p, t, normalizer, f)
        except Exception as e:
            logger.debug("struct_vort_ssim failed: %s", e)
            vort_ssim = float("nan")

        try:
            vort_topk = vort_topk_mae(p, t, f, normalizer)
        except Exception as e:
            logger.debug("vort_topk_mae failed: %s", e)
            vort_topk = float("nan")

        try:
            vort_corr = vort_signed_corr(p, t, f, normalizer)
        except Exception as e:
            logger.debug("vort_signed_corr failed: %s", e)
            vort_corr = float("nan")

        pod_metrics: dict[str, float] = {}
        if pod_basis is not None:
            try:
                pod_metrics = compute_pod_metrics(p, t, f, normalizer, pod_basis, pod_modes)
            except Exception as e:
                logger.debug("pod metrics failed: %s", e)

        results.append({
            **meta,
            "IC_Copy_Score": float(ic_s),
            "HorizontalBand_FFT": float(hband),
            "StructVel_SSIM": float(ssim),
            "StructVort_HighMask_IoU": float(vort_iou),
            "StructVort_RGB_SSIM": float(vort_ssim),
            "VortTopK_MAE": float(vort_topk),
            "VortSignedCorr": float(vort_corr),
            **{k: float(v) for k, v in pod_metrics.items()},
        })

    return results


# ---------------------------------------------------------------------------
# 分層統計輔助函式
# ---------------------------------------------------------------------------


def _group_stats(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """計算一組 case 的彙總統計量。

    Args:
        cases: per-case 結果 dict 的清單。

    Returns:
        含 n、各指標 mean/std 的彙總 dict；若清單為空則回傳 {"n": 0}。
    """
    if not cases:
        return {"n": 0}
    ic_vals = [c["IC_Copy_Score"] for c in cases if not np.isnan(c.get("IC_Copy_Score", float("nan")))]
    hb_vals = [c["HorizontalBand_FFT"] for c in cases if not np.isnan(c.get("HorizontalBand_FFT", float("nan")))]
    ss_vals = [c["StructVel_SSIM"] for c in cases if not np.isnan(c.get("StructVel_SSIM", float("nan")))]
    iou_vals = [c["StructVort_HighMask_IoU"] for c in cases if not np.isnan(c.get("StructVort_HighMask_IoU", float("nan")))]
    vss_vals = [c["StructVort_RGB_SSIM"] for c in cases if not np.isnan(c.get("StructVort_RGB_SSIM", float("nan")))]
    vtk_vals = [c["VortTopK_MAE"] for c in cases if not np.isnan(c.get("VortTopK_MAE", float("nan")))]
    vsc_vals = [c["VortSignedCorr"] for c in cases if not np.isnan(c.get("VortSignedCorr", float("nan")))]
    pod_r2_vals = [c["PODCoeff_R2"] for c in cases if not np.isnan(c.get("PODCoeff_R2", float("nan")))]
    pod_mae_vals = [c["PODCoeff_MAE"] for c in cases if not np.isnan(c.get("PODCoeff_MAE", float("nan")))]
    pod_rel_vals = [c["PODCoeff_RelL2"] for c in cases if not np.isnan(c.get("PODCoeff_RelL2", float("nan")))]
    pod_cos_vals = [c["PODCoeff_Cosine"] for c in cases if not np.isnan(c.get("PODCoeff_Cosine", float("nan")))]
    return {
        "n": len(cases),
        "IC":      float(np.mean(ic_vals)) if ic_vals else float("nan"),
        "IC_std":  float(np.std(ic_vals))  if ic_vals else float("nan"),
        "HBand":   float(np.mean(hb_vals)) if hb_vals else float("nan"),
        "SSIM":    float(np.mean(ss_vals)) if ss_vals else float("nan"),
        "VortIoU": float(np.mean(iou_vals)) if iou_vals else float("nan"),
        "VortSSIM": float(np.mean(vss_vals)) if vss_vals else float("nan"),
        "VortTopK_MAE": float(np.mean(vtk_vals)) if vtk_vals else float("nan"),
        "VortSignedCorr": float(np.mean(vsc_vals)) if vsc_vals else float("nan"),
        "POD_R2":  float(np.mean(pod_r2_vals)) if pod_r2_vals else float("nan"),
        "POD_MAE": float(np.mean(pod_mae_vals)) if pod_mae_vals else float("nan"),
        "POD_RelL2": float(np.mean(pod_rel_vals)) if pod_rel_vals else float("nan"),
        "POD_Cosine": float(np.mean(pod_cos_vals)) if pod_cos_vals else float("nan"),
    }


def stratify_metrics(
    per_case: list[dict[str, Any]],
    re_bins: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """依 ic_type、Re 四分位、G_vec 複雜度對 per-case 結果進行分層彙總。

    Args:
        per_case: compute_per_case_visual_gate 回傳的 per-case 結果清單。
        re_bins: 自訂 Re 分箱 dict，格式為 {name: (lo, hi)}；
                 None 則自動以四分位數切割。

    Returns:
        含以下 key 的 dict：by_mode、by_re_quartile、by_gvec_complexity、
        worst_3、best_3。若 per_case 為空則回傳 {}。
    """
    if not per_case:
        return {}

    # ── 依 ic_type 分層 ──────────────────────────────────────────────────────
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for c in per_case:
        mode = c.get("ic_type", "unknown")
        by_mode.setdefault(mode, []).append(c)
    strat_mode = {k: _group_stats(v) for k, v in by_mode.items()}

    # ── 依 Re 四分位分層 ─────────────────────────────────────────────────────
    re_vals = [c.get("re", 0.0) for c in per_case]
    if re_bins is None and re_vals:
        q25 = float(np.percentile(re_vals, 25))
        q50 = float(np.percentile(re_vals, 50))
        q75 = float(np.percentile(re_vals, 75))
        re_min, re_max = float(min(re_vals)), float(max(re_vals))
        re_bins = {
            "Q1_low":  (re_min, q25),
            "Q2_mid":  (q25, q50),
            "Q3_high": (q50, q75),
            "Q4_turb": (q75, re_max),
        }

    strat_re: dict[str, Any] = {}
    if re_bins:
        by_re: dict[str, list[dict[str, Any]]] = {k: [] for k in re_bins}
        for c in per_case:
            re_val = c.get("re", 0.0)
            placed = False
            for qname, (lo, hi) in re_bins.items():
                if lo <= re_val <= hi:
                    by_re[qname].append(c)
                    placed = True
                    break
            # 未落入任何區間時歸入最後一個 bin（通常為最高 Re 區）
            if not placed:
                by_re[list(re_bins.keys())[-1]].append(c)
        for qname, cases in by_re.items():
            stats = _group_stats(cases)
            stats["re_range"] = [float(re_bins[qname][0]), float(re_bins[qname][1])]
            strat_re[qname] = stats

    # ── 依 G_vec norm 代理複雜度分層 ─────────────────────────────────────────
    norms = [c.get("gvec_norm", 0.0) for c in per_case]
    strat_gvec: dict[str, Any] = {}
    if len(norms) >= 3:
        p33 = float(np.percentile(norms, 33))
        p66 = float(np.percentile(norms, 66))

        def _gvec_group(norm: float) -> str:
            if norm <= p33:
                return "simple"
            if norm <= p66:
                return "medium"
            return "complex"

        by_gvec: dict[str, list[dict[str, Any]]] = {"simple": [], "medium": [], "complex": []}
        for c in per_case:
            by_gvec[_gvec_group(c.get("gvec_norm", 0.0))].append(c)
        for name, cases in by_gvec.items():
            strat_gvec[name] = _group_stats(cases)
        strat_gvec["simple"]["gvec_norm_max"]  = round(p33, 4)
        strat_gvec["medium"]["gvec_norm_max"]  = round(p66, 4)
        strat_gvec["complex"]["gvec_norm_max"] = round(float(max(norms)), 4)

    # ── 最差 / 最佳 case 摘要 ────────────────────────────────────────────────
    def _brief(c: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_key":  c.get("case_key", ""),
            "re":        c.get("re", 0.0),
            "ic_type":   c.get("ic_type", ""),
            "gvec_norm": round(c.get("gvec_norm", 0.0), 4),
            "IC":        round(c.get("IC_Copy_Score", float("nan")), 4),
            "HBand":     round(c.get("HorizontalBand_FFT", float("nan")), 4),
            "SSIM":      round(c.get("StructVel_SSIM", float("nan")), 4),
            "POD_R2":    round(c.get("PODCoeff_R2", float("nan")), 4),
            "POD_MAE":   round(c.get("PODCoeff_MAE", float("nan")), 4),
            "POD_RelL2": round(c.get("PODCoeff_RelL2", float("nan")), 4),
        }

    sorted_ic    = sorted(per_case, key=lambda c: c.get("IC_Copy_Score", 0.0))
    sorted_hband = sorted(per_case, key=lambda c: c.get("HorizontalBand_FFT", 0.0), reverse=True)
    sorted_ssim  = sorted(per_case, key=lambda c: c.get("StructVel_SSIM", 0.0))
    sorted_pod   = sorted(per_case, key=lambda c: c.get("PODCoeff_R2", float("-inf")))

    return {
        "by_mode":            strat_mode,
        "by_re_quartile":     strat_re,
        "by_gvec_complexity": strat_gvec,
        "worst_3": {
            "IC":     [_brief(c) for c in sorted_ic[:3]],
            "HBand":  [_brief(c) for c in sorted_hband[:3]],
            "SSIM":   [_brief(c) for c in sorted_ssim[:3]],
            "POD_R2": [_brief(c) for c in sorted_pod[:3]],
        },
        "best_3": {
            "IC":     [_brief(c) for c in sorted_ic[-3:][::-1]],
            "POD_R2": [_brief(c) for c in sorted_pod[-3:][::-1]],
        },
    }


# ---------------------------------------------------------------------------
# 分項 must-pass gate（取代單一乘積決定 PASS/FAIL）
# ---------------------------------------------------------------------------

# 預設閾值（per-stage 可由 config 覆寫）
DEFAULT_GATE_THRESHOLDS: dict[str, float] = {
    "cold_ssim_min":  0.65,   # cold StructVel_SSIM >=
    "cold_iou_min":   0.50,   # cold VortHighMask_IoU >=
    "warm_ssim_min":  0.78,   # warm StructVel_SSIM >=
    "hband_max_max":  0.30,   # HBand_max_case <=
    "ic_min":         0.55,   # IC >= (過低 = under-evolve / IC-copy)
    "ic_max":         1.10,   # IC <= (over-copy / divergence；IC clamp 上限 1.0)
}


def evaluate_gate(
    agg: dict[str, Any],
    strat: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """執行分項 must-pass gate，判斷評估是否通過。

    所有 check 均必須 pass，gate_pass 才為 True。
    NaN 值視為該 check fail。

    Args:
        agg: 彙總指標 dict，含 HBand_max_case、IC_Copy_Score 等欄位。
        strat: stratify_metrics 回傳的分層結果 dict，含 by_mode 子 dict。
        thresholds: 覆寫預設閾值的 dict，可為 None（使用 DEFAULT_GATE_THRESHOLDS）。

    Returns:
        含 checks（各項判定結果）、gate_pass（bool）、n_pass、n_total 的 dict。
    """
    th = dict(DEFAULT_GATE_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    by_mode = strat.get("by_mode", {})
    cold = by_mode.get("cold", {}) or {}
    warm = by_mode.get("warm", {}) or {}

    def _f(v: Any) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return v

    cold_ssim = _f(cold.get("SSIM"))
    cold_iou  = _f(cold.get("VortIoU"))
    warm_ssim = _f(warm.get("SSIM"))
    hband_max = _f(agg.get("HBand_max_case"))
    ic        = _f(agg.get("IC_Copy_Score"))

    def _ge(val: float, thr: float) -> bool:
        return (not np.isnan(val)) and val >= thr

    def _le(val: float, thr: float) -> bool:
        return (not np.isnan(val)) and val <= thr

    checks: dict[str, dict[str, Any]] = {
        "cold_ssim":  {"value": cold_ssim, "threshold": th["cold_ssim_min"], "pass": _ge(cold_ssim, th["cold_ssim_min"])},
        "cold_iou":   {"value": cold_iou,  "threshold": th["cold_iou_min"],  "pass": _ge(cold_iou, th["cold_iou_min"])},
        "warm_ssim":  {"value": warm_ssim, "threshold": th["warm_ssim_min"], "pass": _ge(warm_ssim, th["warm_ssim_min"])},
        "hband_max":  {"value": hband_max, "threshold": th["hband_max_max"], "pass": _le(hband_max, th["hband_max_max"])},
        "ic_range":   {"value": ic, "threshold": [th["ic_min"], th["ic_max"]],
                       "pass": (not np.isnan(ic)) and th["ic_min"] <= ic <= th["ic_max"]},
    }
    gate_pass = all(c["pass"] for c in checks.values())
    n_pass = sum(1 for c in checks.values() if c["pass"])
    return {"checks": checks, "gate_pass": gate_pass, "n_pass": n_pass, "n_total": len(checks)}


# ---------------------------------------------------------------------------
# JSON 結構建構與儲存
# ---------------------------------------------------------------------------


def build_vis_gate_detailed(
    step: int,
    per_case: list[dict[str, Any]],
    aggregate: dict[str, Any],
    re_bins: dict[str, tuple[float, float]] | None = None,
    gate_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """建立完整的 vis_gate_detailed.json 資料結構。

    Args:
        step: 當前訓練步數（checkpoint step）。
        per_case: compute_per_case_visual_gate 回傳的 per-case 結果清單。
        aggregate: 來自 trainer 的彙總指標 dict。
        re_bins: 自訂 Re 分箱，None 則自動四分位切割。
        gate_thresholds: 覆寫預設閾值的 dict，可為 None。

    Returns:
        含 step、aggregate、gate、stratification、per_case 的完整 dict。
    """
    strat = stratify_metrics(per_case, re_bins)
    agg = dict(aggregate)

    # 將分層的 HBand 指標提升至頂層彙總欄位
    q1_hband = strat.get("by_re_quartile", {}).get("Q1_low", {}).get("HBand", float("nan"))
    worst_hband_list = strat.get("worst_3", {}).get("HBand", [])
    max_case_hband = worst_hband_list[0]["HBand"] if worst_hband_list else float("nan")
    agg["HBand_Q1_low"] = float(q1_hband)
    agg["HBand_max_case"] = float(max_case_hband)

    # vg_composite（與 trainer._add_vg_composite 保持一致）
    # 注意：用明確的 None 判斷而非 `or` fallback，因為 0.0 是有效指標值，
    # `0.0 or default` 會錯誤地替換為預設值。
    def _safe_float(val: Any, default: float) -> float:
        return float(val) if val is not None else default

    ic_c    = min(max(_safe_float(agg.get("IC_Copy_Score"), 0.0), 0.0), 1.0)
    ssim_c  = max(_safe_float(agg.get("StructVel_RGB_SSIM"), 0.0), 0.0)
    hband_c = min(max(_safe_float(agg.get("HorizontalBand_FFT"), 1.0), 0.0), 1.0)
    agg["vg_composite"] = ic_c * ssim_c * (1.0 - hband_c)

    # vg_composite_strict：對未達目標的指標施以更強懲罰
    def _c01(x: float) -> float:
        return max(0.0, min(1.0, x))

    hband_q1 = _safe_float(agg.get("HBand_Q1_low"), 1.0)
    agg["vg_composite_strict"] = (
        _c01((_safe_float(agg.get("IC_Copy_Score"), 0.0) - 0.55) / 0.30)
        * _c01((_safe_float(agg.get("StructVel_RGB_SSIM"), 0.0) - 0.70) / 0.18)
        * _c01((0.15 - _safe_float(agg.get("HorizontalBand_FFT"), 1.0)) / 0.15)
        * _c01((0.20 - hband_q1) / 0.20)
    )

    # 分項 must-pass gate（PASS/FAIL 主裁；vg_composite 僅保留為連續趨勢）
    gate = evaluate_gate(agg, strat, gate_thresholds)
    agg["gate_pass"]   = gate["gate_pass"]
    agg["gate_n_pass"] = gate["n_pass"]

    return {
        "step": step,
        "aggregate": agg,
        "gate": gate,
        **strat,
        "per_case": per_case,
    }


def save_vis_gate_detailed(eval_output_dir: Path, data: dict[str, Any]) -> None:
    """將 vis_gate_detailed evidence 寫入磁碟，並附加至歷史 JSONL。

    寫入兩份帶身分後綴（`__nockpt_{YYYYMMDD}`）且內嵌 `_provenance` 的 evidence：

    - `vis_gate_detailed`：最新完整結果（run 內滾動，允許覆寫）
    - `vis_gate_detailed_step{step:06d}`：當步快照（保留 per-case 詳情）

    並將精簡版（不含 per_case）附加至 `vis_gate_history.jsonl`（append-only，
    本身不會覆寫既有內容，故不套用禁覆寫機制）。

    身分說明：訓練期寫入時該 step 的 ckpt 尚未落地，無檔案內容可雜湊，
    故 `ckpt=None`；世代身分由 run 目錄 + step 承擔，兩者寫入 `_provenance.extra`
    （見 `lbmnca/nca/tes/utils/evidence.py`）。

    Args:
        eval_output_dir: 評估輸出目錄路徑，不存在時自動建立。
        data: build_vis_gate_detailed 回傳的完整 dict。
    """
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = eval_output_dir.parent
    step = int(data["step"])

    # 最新完整結果 + 每步快照（同一份 payload，兩個 stem）
    for stem in ("vis_gate_detailed", f"vis_gate_detailed_step{step:06d}"):
        write_run_evidence(eval_output_dir, stem, data, run_dir=run_dir, step=step)

    # 附加精簡版至歷史記錄（省略 per_case 清單以節省空間）
    history_entry: dict[str, Any] = {
        "step":      step,
        "aggregate": data.get("aggregate", {}),
        "by_mode":   data.get("by_mode", {}),
        "gate":      data.get("gate", {}),
    }
    re_q = data.get("by_re_quartile", {})
    if re_q:
        history_entry["by_re_quartile"] = {
            k: {kk: vv for kk, vv in v.items() if kk not in ("re_range",)}
            for k, v in re_q.items()
        }

    with open(eval_output_dir / "vis_gate_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry, ensure_ascii=False) + "\n")

    logger.debug(
        "vis_gate_detailed evidence saved (step=%d, %d cases)", step, len(data.get("per_case", []))
    )


# eval_summary 只保留跨 run 可比的 gate 指標，刻意排除 total loss（禁用於跨 run 排名）。
_EVAL_SUMMARY_AGG_KEYS: tuple[str, ...] = (
    "IC_Copy_Score", "StructVel_RGB_SSIM", "StructVort_RGB_SSIM",
    "StructVort_HighMask_IoU", "VortTopK_MAE", "VortSignedCorr",
    "HorizontalBand_FFT", "HBand_max_case", "HBand_Q1_low",
    "vg_composite", "vg_composite_strict", "gate_pass", "gate_n_pass",
)


def save_eval_summary(eval_output_dir: Path, detailed: dict[str, Any]) -> Path:
    """寫跨 run 可比的評估摘要 evidence（gate 布林 + 分項 + 分層）。

    單一實作：`TESTrainer._write_eval_summary` 與 `Validator.write_summary`
    兩者皆委派至本函式（DRY）。

    寫 `eval_summary_latest`（帶 `__nockpt_{YYYYMMDD}` 身分後綴 + `_provenance`，
    run 內滾動故允許覆寫）+ `eval_summary.jsonl`（append-only 歷史）。
    不含 total loss（禁用於跨 run 排名）。

    Args:
        eval_output_dir: 評估輸出目錄（通常為 `{run_dir}/eval_output`）。
        detailed: `build_vis_gate_detailed` 回傳的 detailed dict。

    Returns:
        `eval_summary_latest` evidence 的實際寫出路徑。
    """
    agg = detailed.get("aggregate", {})
    summary: dict[str, Any] = {
        "step": detailed.get("step"),
        "gate": detailed.get("gate", {}),
        "aggregate": {k: agg.get(k) for k in _EVAL_SUMMARY_AGG_KEYS},
        "by_mode": detailed.get("by_mode", {}),
        "by_re_quartile": detailed.get("by_re_quartile", {}),
    }
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    written = write_run_evidence(
        eval_output_dir, "eval_summary_latest", summary,
        run_dir=eval_output_dir.parent, step=detailed.get("step"),
    )
    with open(eval_output_dir / "eval_summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return written
