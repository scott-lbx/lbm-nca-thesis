"""橋接 lbmnca.morphospace.physics_proxy.gvec_api → G_vec 14D。

若 lbmnca.morphospace 無法 import（路徑不存在、API 不符），
回退至 fallback g_vec（從 global_stats 的均值，供測試用）。

★ 2026-07-26 解析度修正（DEEP_AUDIT_2026-07 主題 2，critical）
-------------------------------------------------------------
訓練用的 g_vec 是在 **1920×1080 原生 mask** 上計算的，但前端把幾何光柵化成
**640×360**（MaskRasterizer）後直接呼叫 compute_gvec_from_mask，**全程無尺度還原**。

G_vec 14 維中有 5 維是**原始像素長度**（A1/A2/B1/D1/D3），另有 1 維量綱為 [1/L]
（C1），全都沒有以 L_char 無因次化 → 部署端餵給模型的條件向量系統性偏移。

同一支 regime_features.py 對 L_char 有明確 ×SIM_SCALE=3.0 的還原，對 g_vec 卻沒有
（:136 vs :106）——這使 4-methodology.tex:416/418「以與訓練階段完全同源之程序」
的宣稱不成立。

實測（12 個真實 mask，對照訓練端 1920×1080 真值的相對誤差）：

    ==========================  ==============  ==============
    dim                          修正後（上採樣）  修正前（直接算）
    ==========================  ==============  ==============
    A1_min_inter_obstacle_gap        2.02%          65.34%
    A2_mean_inter_obstacle_gap       1.12%          65.77%
    B1_throat_width                  0.55%          66.47%
    C1_frontal_area_density          0.44%         198.92%
    D1_mean_wall_distance            0.30%          66.08%
    D3_hydraulic_diameter            0.53%          66.56%
    ──────────────────────────  ──────────────  ──────────────
    整體平均                         0.58%          38.04%
    整體最大                         5.88%         202.82%
    ==========================  ==============  ==============

修正方式為**最近鄰上採樣回訓練解析度後再計算**，而非只對 6 個維度做 ×3 / ÷3
的尺度校正。理由：實測比值並非精確的 1/3（0.3330–0.3393），純尺度校正會留下
0.3–1.9% 的殘差；且離散型與非線性的量（連通域、tortuosity、shelter index）
是否嚴格尺度不變並無保證，逐維校正等於對每一維各下一個未驗證的假設。

殘餘的 0.58% 來自前端只有 640×360 的資訊量（3px 量化），不可再降。
成本：約 0.26 s／次，只在幾何改變時觸發、不在渲染迴圈內
（見 docs/DEPLOYMENT.md「唯一允許 CPU 的操作」）。

防回歸測試：``scripts/qa/verify_gvec_resolution.py``
"""
from __future__ import annotations

import hashlib

import numpy as np

# g_vec stats mean（無真實 morphospace 時的預設值）
_FALLBACK_G_VEC = np.array([
    2.5218, 5.1209, 10.7372, 0.02516, 0.7308, 29.4326, 2.0427,
    0.0, 0.02456, 0.003270, 0.4499, 5.8943, 36.4049, 1.0,
], dtype=np.float32)


def _try_import_gvec_api():
    """嘗試 import lbmnca.morphospace 的 gvec_api。

    Returns:
        (compute_fn, active_indices) 若成功；(None, None) 若失敗。
    """
    try:
        from lbmnca.morphospace.physics_proxy.gvec_api import (
            compute_gvec_from_mask,
            GVEC_ACTIVE_INDICES,
        )
        return compute_gvec_from_mask, GVEC_ACTIVE_INDICES
    except (ImportError, ModuleNotFoundError):
        return None, None


class GVecUnavailableError(RuntimeError):
    """lbmnca.morphospace gvec_api 不可用且未允許 fallback。"""


_gvec_cache: dict[bytes, np.ndarray] = {}

# ── G_vec 計算解析度（訓練契約）────────────────────────────────────────────
# 訓練端的 g_vec 一律算在 1920×1080 原生 mask 上（dataset/*/masks/*.png 實測尺寸）。
# 前端推論域為 640×360，兩者相差整數倍 3 → 可用精確的最近鄰上採樣還原。
GVEC_TRAIN_H: int = 1080
GVEC_TRAIN_W: int = 1920


def _upsample_to_train_res(mask_np: np.ndarray) -> np.ndarray:
    """把前端解析度的 mask 最近鄰上採樣到訓練解析度。

    只支援整數倍放大（640×360 → 1920×1080 為 ×3）。非整數倍或已是訓練解析度
    時原樣返回，並由呼叫端記錄警告——**不做插值近似**，因為那會在邊界產生
    訓練端不存在的中間值，反而引入新的分布偏移。

    Args:
        mask_np: [H, W] float32/bool，0=solid、非 0=fluid。

    Returns:
        [GVEC_TRAIN_H, GVEC_TRAIN_W] 的 mask；無法整數放大時回傳原陣列。
    """
    h, w = mask_np.shape[:2]
    if (h, w) == (GVEC_TRAIN_H, GVEC_TRAIN_W):
        return mask_np
    if h <= 0 or w <= 0:
        return mask_np
    fy, fx = GVEC_TRAIN_H // h, GVEC_TRAIN_W // w
    if fy < 1 or fx < 1 or fy * h != GVEC_TRAIN_H or fx * w != GVEC_TRAIN_W:
        return mask_np          # 非整數倍 → 交由呼叫端警告
    return np.repeat(np.repeat(mask_np, fy, axis=0), fx, axis=1)


_warned_resolution: set[tuple[int, int]] = set()


def get_gvec(mask_np: np.ndarray, allow_fallback: bool = False) -> np.ndarray:
    """從 binary mask 計算 G_vec 14D，相同 mask 直接返回快取結果。

    嘗試呼叫 lbmnca.morphospace 的 gvec_api。失敗時（UI-09 修正，去 silent）：
      - ``allow_fallback=True``（config ``geometry.allow_gvec_fallback``，
        開發模式）→ 回 fallback 常數向量並大聲警告；
      - 否則 raise ``GVecUnavailableError``——demo 條件全錯比 crash 更糟，
        由呼叫端顯示錯誤 banner。

    Args:
        mask_np: [H, W] float32, 0=solid, 1=fluid
        allow_fallback: 允許退回 g_vec stats 均值常數向量。

    Returns:
        [14] float32 G_vec（raw，未正規化）

    Raises:
        GVecUnavailableError: gvec_api 不可用且 allow_fallback=False。
    """
    key = hashlib.sha256(mask_np.tobytes()).digest()
    if key in _gvec_cache:
        return _gvec_cache[key].copy()

    compute_fn, _ = _try_import_gvec_api()
    err: Exception | None = None
    if compute_fn is not None:
        try:
            # ★ 解析度對齊：訓練 g_vec 算在 1920×1080，此處先上採樣還原，
            # 否則 5 個像素長度維與 C1 會系統性偏移（見模組 docstring 實測表）。
            mask_train = _upsample_to_train_res(mask_np)
            if mask_train.shape[:2] != (GVEC_TRAIN_H, GVEC_TRAIN_W):
                shp = (int(mask_np.shape[0]), int(mask_np.shape[1]))
                if shp not in _warned_resolution:
                    _warned_resolution.add(shp)
                    print(
                        f"[gvec_bridge] WARN 無法把 {shp[0]}×{shp[1]} 整數倍上採樣到 "
                        f"訓練解析度 {GVEC_TRAIN_H}×{GVEC_TRAIN_W}；g_vec 的像素長度維"
                        f"（A1/A2/B1/D1/D3）與 C1 將偏離訓練分布。"
                    )
            result = compute_fn(mask_train).astype(np.float32)
            _gvec_cache[key] = result
            return result.copy()
        except Exception as e:
            err = e
    if allow_fallback:
        print(f"[gvec_bridge] WARN FALLBACK G_VEC (constants, dev only): {err or 'gvec_api import failed'}")
        result = _FALLBACK_G_VEC.copy()
        _gvec_cache[key] = result
        return result.copy()
    raise GVecUnavailableError(
        f"lbmnca.morphospace.physics_proxy.gvec_api 不可用（{err or 'import 失敗'}）。"
        "開發模式可設 geometry.allow_gvec_fallback: true"
    )
