"""求解迴圈內時均（``/mean_accum`` 分段和）的讀取端工具。

寫入端在 ``src/lbm_mrt_les/core/time_mean_accumulator.py``（需要 taichi）。
本模組**不依賴 taichi**，供 H5→Zarr 轉檔與離線分析腳本共用，避免度量與
窗選規則出現第二份實作（DEEP_AUDIT_2026-07 Stage 1.15）。

核心關係式（段時均以步數為權重合成任意窗時均）::

    mean(window) = sum_i n_i * mean_i / sum_i n_i     （i 取窗內各段）

所有欄位皆為 **raw moment 空間**（未正規化）；正規化一律留給呼叫端，
沿用既有 ``tanh((x - M_REF) / S_combined)`` 慣例。
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

MEAN_ACCUM_GROUP: str = "mean_accum"
"""H5 內分段時均群組名。"""

SEGMENT_MEAN_KEY: str = f"{MEAN_ACCUM_GROUP}/segment_mean"
"""存在性判斷用的資料集路徑（`in f` 檢查，不用帶預設值的 config fallback）。"""


def has_mean_accum(f: h5py.File) -> bool:
    """H5 是否含分段時均（新欄位存在性檢查）。"""
    return SEGMENT_MEAN_KEY in f


def read_mean_accum(f: h5py.File) -> dict[str, Any] | None:
    """讀 ``/mean_accum`` 群組。

    Args:
        f: 已開啟的 H5 檔。

    Returns:
        dict（含 segment_mean/n/step0/step1/ctu0/ctu1 與 attrs）或 None（無此群組）。
    """
    if not has_mean_accum(f):
        return None
    g = f[MEAN_ACCUM_GROUP]
    return {
        "segment_mean":  np.asarray(g["segment_mean"][:], dtype=np.float32),
        "segment_n":     np.asarray(g["segment_n"][:], dtype=np.int64),
        "segment_step0": np.asarray(g["segment_step0"][:], dtype=np.int64),
        "segment_step1": np.asarray(g["segment_step1"][:], dtype=np.int64),
        "segment_ctu0":  np.asarray(g["segment_ctu0"][:], dtype=np.float64),
        "segment_ctu1":  np.asarray(g["segment_ctu1"][:], dtype=np.float64),
        "attrs":         {k: v for k, v in g.attrs.items()},
    }


def combine_segments(
    segment_mean: np.ndarray,
    segment_n: np.ndarray,
    sel: np.ndarray,
) -> np.ndarray:
    """把選中的段以步數為權重合成窗時均。

    Args:
        segment_mean: [S, 9, H, W] 段時均。
        segment_n: [S] 各段樣本數。
        sel: [S] bool，選中的段。

    Returns:
        [9, H, W] float64 窗時均。

    Raises:
        ValueError: 未選中任何段。
    """
    if not np.any(sel):
        raise ValueError("combine_segments: 未選中任何段")
    w = segment_n[sel].astype(np.float64)
    m = segment_mean[sel].astype(np.float64)
    return np.tensordot(w / w.sum(), m, axes=(0, 0))


def select_segments(
    accum: dict[str, Any],
    start_ctu: float,
    end_ctu: float | None = None,
) -> np.ndarray:
    """選出完全落在 [start_ctu, end_ctu] 內的段。

    只取**整段**落在窗內者（段起點 >= start_ctu），不做段內插值——段是累加
    的最小不可分單位，部份採用會混入窗外時間。

    Args:
        accum: :func:`read_mean_accum` 的輸出。
        start_ctu: 窗起點（CTU）。負值表示不限制（取全部段）。
        end_ctu: 窗終點（CTU），None = 不限制。

    Returns:
        [S] bool 選擇遮罩。
    """
    c0 = accum["segment_ctu0"]
    c1 = accum["segment_ctu1"]
    sel = np.ones(len(c0), dtype=bool) if start_ctu < 0.0 else (c0 >= start_ctu - 1e-9)
    if end_ctu is not None:
        sel &= c1 <= end_ctu + 1e-9
    return sel


def window_mean(
    accum: dict[str, Any],
    start_ctu: float,
    end_ctu: float | None = None,
) -> dict[str, Any] | None:
    """算指定窗的累加時均，並附贈前半／後半分半時均（自我檢查用）。

    分半以**段數**對切（段長固定 → 等同時間對切），兩半各自是同一條累加
    軌跡的獨立子窗，其差量 ``Δ_self(前半|後半)`` 即該時均的不確定度尺度，
    **不需要 snapshot 就能量測**。

    Args:
        accum: :func:`read_mean_accum` 的輸出。
        start_ctu: 窗起點（CTU）；負值 = 全部段。
        end_ctu: 窗終點（CTU），None = 到最後一段。

    Returns:
        dict（mean / n_steps / ctu0 / ctu1 / n_segments / half1 / half2 …），
        或 None（窗內無任何完整段）。
    """
    sel = select_segments(accum, start_ctu, end_ctu)
    if not np.any(sel):
        return None
    idx = np.flatnonzero(sel)
    seg_mean, seg_n = accum["segment_mean"], accum["segment_n"]

    mean = combine_segments(seg_mean, seg_n, sel)
    out: dict[str, Any] = {
        "mean":        mean,
        "n_steps":     int(seg_n[idx].sum()),
        "n_segments":  int(idx.size),
        "ctu0":        float(accum["segment_ctu0"][idx[0]]),
        "ctu1":        float(accum["segment_ctu1"][idx[-1]]),
        "half1":       None,
        "half2":       None,
        "half_n":      (0, 0),
    }
    if idx.size >= 2:
        cut = idx.size // 2
        s1 = np.zeros_like(sel)
        s2 = np.zeros_like(sel)
        s1[idx[:cut]] = True
        s2[idx[cut:]] = True
        out["half1"] = combine_segments(seg_mean, seg_n, s1)
        out["half2"] = combine_segments(seg_mean, seg_n, s2)
        out["half_n"] = (int(seg_n[s1].sum()), int(seg_n[s2].sum()))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 度量（與監督者 1.12 null-control 完全相同的定義）
# ══════════════════════════════════════════════════════════════════════════════


def rel_l2(x: np.ndarray, ref: np.ndarray) -> float:
    """``‖x − ref‖ / ‖x‖``（分母是 x 本身，比照 1.12 null-control 定義）。"""
    denom = float(np.linalg.norm(x))
    if denom <= 0.0:
        return float("nan")
    return float(np.linalg.norm(x - ref) / denom)


def delta_norm_fluid(
    mean_x: np.ndarray,
    mean_ref: np.ndarray,
    m_ref: np.ndarray,
    s_combined: np.ndarray,
    fluid_hw: np.ndarray,
) -> float:
    """tanh 正規化、fluid-masked、9 通道 rel_l2（與模型誤差同一種度量）。

    Args:
        mean_x: [9, H, W] raw moment 時均（被評估者，作分母）。
        mean_ref: [9, H, W] raw moment 時均（參考）。
        m_ref: [9, 1, 1] M_REF。
        s_combined: [9, 1, 1] S_combined。
        fluid_hw: [H, W] bool，True = 流體（= ~solid mask）。

    Returns:
        rel_l2 純量。
    """
    nx = np.tanh((mean_x - m_ref) / s_combined)
    nr = np.tanh((mean_ref - m_ref) / s_combined)
    fm3 = np.broadcast_to(fluid_hw, mean_x.shape)
    return rel_l2(nx[fm3], nr[fm3])


def channel_rms_diff(
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    m_ref: np.ndarray,
    s_combined: np.ndarray,
    fluid_hw: np.ndarray,
) -> np.ndarray:
    """分通道 RMS 差（tanh 空間、fluid-masked），供混疊診斷。

    Returns:
        [9] float64，各通道的 RMS(|a − b|)。
    """
    na = np.tanh((mean_a - m_ref) / s_combined)
    nb = np.tanh((mean_b - m_ref) / s_combined)
    d = (na - nb)[:, fluid_hw]                       # [9, n_fluid]
    return np.sqrt((d.astype(np.float64) ** 2).mean(axis=1))
