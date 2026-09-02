"""L_char 公開 API（特徵長度單一真相）。

計算 Reynolds 數與 SDF tanh 正規化所用的特徵長度 L_char。
此 API 是 lbmnca.morphospace → lbmnca.lbm / lbmnca.nca / lbmnca.bridge 的穩定介面，
比照 :mod:`gvec_api` 的定位。

★ 本模組是 L_char 的**唯一實作**。任何 repo 不得自行實作特徵長度公式。
  背景：2026-06-29 訓練端把公式從 ``min(w, h)`` 改為 y-extent 後，
  部署端（lbmnca.bridge）與 S1_5 生成腳本未同步，導致同一幾何在訓練與推論
  得到不同的 L_char（實測系統性低估 ~26.5%），而 L_char 同時決定 Re 條件與
  SDF 的 tanh 尺度 → 雙重偏移。詳見
  ``docs/reviews/DEEP_AUDIT_2026-07/fixes/FIX_B_Lchar統一.md``。

定義
----
來流沿 **+x**，L_char 取單一建物在 **y 軸（垂直流向，cross-flow）** 的投影寬度，
全體建物取最大值。這是主導 ``Re = U·D/nu`` 與 Strouhal 數的特徵尺度。

不是流向深度，也不是 ``min(x-extent, y-extent)``——後者取向盲，對迎風扁寬建物
會低估 L_char，使實際 Re 高於標籤值（最高誤標 ~4.6×）。

兩個入口
--------
- :func:`l_char_from_buildings` — 由建物參數 ``(cx, cy, w, h)`` 計算（config 生成端）
- :func:`l_char_from_mask` — 由 rasterized binary mask 計算（mask 端、部署端）

兩者對同一幾何應給出一致結果（mask 版因離散化與連通域處理會有數 px 差異）。
一致性由 ``scripts/qa/verify_lchar_consistency.py`` 驗證。
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion

# ── 演算法常數（原 src/lbmnca/lbm/lbm_mrt_les/public/geometry.py）────────────────

#: 形態學侵蝕迭代數，用於分離黏連建築。
EROSION_ITER: int = 3

#: 連通域面積上限（domain 面積佔比）。超過者視為背景/邊框，排除。
AREA_FRAC_MAX: float = 0.05

#: TES H5 相對於 ROI（1920×1080）的空間降採樣倍率。
#: SDF 的 tanh 尺度為 ``L_char_px / H5_DOWNSAMPLE``。
#: ★ 原先在三處各自硬編（01 ``H5_DOWNSAMPLE``、02/03 ``_SIM_TO_H5_SCALE``），此處收斂。
H5_DOWNSAMPLE: int = 3

#: mask 軸序。
#:   ``"xy"`` → ``axis0 = x``（流向）、``axis1 = y``（橫流向）。Taichi ``(nx, ny)`` 慣例。
#:   ``"yx"`` → ``axis0 = y``（橫流向）、``axis1 = x``（流向）。NumPy 影像 ``[H, W]`` 慣例。
MaskLayout = Literal["xy", "yx"]


def l_char_from_buildings(buildings: Sequence[Sequence[float]]) -> int:
    """由建物參數計算 L_char。

    建物 tuple 為 ``(cx, cy, w, h)``，其中 ``h`` 為 y 方向（垂直流向）尺寸。
    取全體建物 ``h`` 的最大值。

    Args:
        buildings: 建物序列，每項至少 4 個元素 ``(cx, cy, w, h)``。

    Returns:
        L_char（像素，至少為 1）。

    Raises:
        ValueError: buildings 為空。
    """
    if len(buildings) == 0:
        raise ValueError("buildings 為空，無法計算 L_char")
    return max(1, max(int(round(b[3])) for b in buildings))


def l_char_from_mask(
    solid_mask: np.ndarray,
    *,
    layout: MaskLayout,
) -> int:
    """由 binary solid mask 計算 L_char。

    演算法：
        1. 形態學侵蝕（``EROSION_ITER`` px）分離黏連建築
        2. Connected component 標記
        3. 面積過濾：排除 > domain ``AREA_FRAC_MAX`` 的大型連通域
        4. 在原始（未侵蝕）mask 上、以侵蝕後 bbox 外擴 ``EROSION_ITER`` 的 ROI
           重新量測，取 y-extent（垂直流向投影寬度）
        5. 全域取最大值

    若面積過濾後無合格連通域，退回取「面積最小」的連通域之 y-extent
    （fallback 與主路徑同樣使用 y-extent，不切換公式）。

    Args:
        solid_mask: 2D 陣列，True/非零 = 固體。
        layout: 軸序，見 :data:`MaskLayout`。**必填**，不提供預設——
            軸序猜錯會把流向深度當成橫流寬度，是無聲的物理錯誤。

    Returns:
        L_char（像素，至少為 1）。

    Raises:
        ValueError: mask 非 2D，或 layout 非法。
    """
    solid = np.asarray(solid_mask, dtype=bool)
    if solid.ndim != 2:
        raise ValueError(f"solid_mask 必須為 2D，收到 shape={solid.shape}")

    if layout == "yx":
        # 轉成內部統一的 (x, y) 表示
        solid = solid.T
    elif layout != "xy":
        raise ValueError(f"未知 layout: {layout!r}（可用 'xy' / 'yx'）")

    nx, ny = solid.shape

    solid_eroded = binary_erosion(solid, iterations=EROSION_ITER)
    labeled, n_features = ndimage.label(solid_eroded)

    if n_features == 0:
        labeled, n_features = ndimage.label(solid)
        if n_features == 0:
            return max(1, ny // 8)

    area_max = int(nx * ny * AREA_FRAC_MAX)
    max_l = 0

    for label_id in range(1, n_features + 1):
        region = labeled == label_id
        if int(np.sum(region)) > area_max:
            continue

        x_idx = np.where(np.any(region, axis=1))[0]
        y_idx = np.where(np.any(region, axis=0))[0]
        if len(x_idx) == 0 or len(y_idx) == 0:
            continue

        x0 = max(0, x_idx[0] - EROSION_ITER)
        x1 = min(nx - 1, x_idx[-1] + EROSION_ITER)
        y0 = max(0, y_idx[0] - EROSION_ITER)
        y1 = min(ny - 1, y_idx[-1] + EROSION_ITER)
        roi = solid[x0 : x1 + 1, y0 : y1 + 1]

        x_real = np.where(np.any(roi, axis=1))[0]
        y_real = np.where(np.any(roi, axis=0))[0]
        if len(x_real) == 0 or len(y_real) == 0:
            continue

        # y_real 的 extent = 垂直流向投影寬度（cross-flow）
        max_l = max(max_l, int(y_real[-1] - y_real[0] + 1))

    if max_l == 0:
        min_area, fallback_l = nx * ny, 1
        for label_id in range(1, n_features + 1):
            region = labeled == label_id
            area = int(np.sum(region))
            if area < min_area:
                x_idx = np.where(np.any(region, axis=1))[0]
                y_idx = np.where(np.any(region, axis=0))[0]
                if len(x_idx) > 0 and len(y_idx) > 0:
                    min_area = area
                    fallback_l = int(y_idx[-1] - y_idx[0] + 1)
        max_l = fallback_l

    return max(1, max_l)


def sdf_tanh_scale(l_char_px: float, *, downsample: int = H5_DOWNSAMPLE) -> float:
    """SDF tanh 正規化的尺度。

    C0 通道為 ``tanh(SDF_h5 / sdf_tanh_scale(L_char))``，其中 SDF 以 H5 解析度
    （ROI 的 1/``downsample``）的像素為單位。

    ★ 原先此換算在三處重複（``norm_api.sdf_forward``、``tes_backend._normalize_static``、
      ``tes_adapter._normalize_static``），常數還有兩個名字。此處收斂為單一定義。

    Args:
        l_char_px: L_char，單位為 **ROI 像素**（1920×1080 尺度）。
        downsample: H5 相對 ROI 的降採樣倍率。

    Returns:
        tanh 尺度（H5 像素），下限 1.0。
    """
    return max(1.0, float(l_char_px) / float(downsample))
