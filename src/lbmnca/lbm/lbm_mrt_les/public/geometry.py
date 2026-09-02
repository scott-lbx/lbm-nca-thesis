"""
公開 API: 幾何特性計算 -- L_char、阻塞率、邊界距離。

取代並整合以下位置的邏輯：
  - src/tools/config_utils/geometry.py (calc_l_char, calc_max_blockage)

原 config_utils/geometry.py 中的 fill_geometry() 入口保留為薄包裝，
內部委派至此模組的函式。

★ L_char 已於 2026-07-26 收斂至單一真相
  ``src/lbmnca/morphospace/physics_proxy/lchar_api.py``。本模組的
  :func:`compute_characteristic_length` 現為委派包裝，並 re-export
  buildings 版入口供 01 的 config 生成腳本使用。
  背景見 ``docs/reviews/DEEP_AUDIT_2026-07/fixes/FIX_B_Lchar統一.md``。
"""

import sys
from pathlib import Path
from typing import Sequence, TypedDict

import numpy as np

from .mask import png_to_binary_mask

# L_char 的單一真相在 morphospace，不得於此重寫公式——靜默改用別的公式正是曾修正過的 bug。
from lbmnca.morphospace.physics_proxy.lchar_api import (
    AREA_FRAC_MAX as _AREA_FRAC_MAX,
    EROSION_ITER as _EROSION_ITER,
    l_char_from_buildings,
    l_char_from_mask,
)


class BlockageMetrics(TypedDict):
    max_blockage: float
    open_fraction: float


class BoundaryExtentMetrics(TypedDict):
    inlet_px: int
    outlet_px: int
    top_px: int
    bottom_px: int
    inlet_over_lchar: float
    outlet_over_lchar: float
    top_over_lchar: float
    bottom_over_lchar: float


def compute_characteristic_length(
    solid_mask: np.ndarray,
    nx: int,
    ny: int,
) -> int:
    """計算 L_char：最大單一建築的迎風橫向（cross-flow）投影寬度（像素）。

    ★ 委派至 ``physics_proxy.lchar_api.l_char_from_mask``（L_char 單一真相）。
    演算法（3px 侵蝕分離黏連建築 → connected component → 面積過濾 → y-extent →
    取最大）與參數常數皆定義於該模組；此處僅保留 01 既有簽名，不重複實作公式。

    Args:
        solid_mask: 布林陣列 (nx, ny)，Taichi 慣例（axis0=x 流向、axis1=y 橫向）。
            True = 固體。
        nx: domain 寬度（保留簽名相容；實際由 mask shape 推得）。
        ny: domain 高度（同上）。

    Returns:
        特徵長度（像素），至少為 1。
    """
    return l_char_from_mask(solid_mask, layout="xy")


def compute_characteristic_length_from_buildings(
    buildings: Sequence[Sequence[float]],
) -> int:
    """計算 L_char：由建物參數 ``(cx, cy, w, h)`` 取 max(h)（cross-flow 迎風寬）。

    ★ 委派至 ``physics_proxy.lchar_api.l_char_from_buildings``（L_char 單一真相）。
    供 config 生成腳本（gen_c_series / supplement_cold_cases / gen_s1_5）呼叫，
    取代它們原本各自內嵌的公式。

    Args:
        buildings: 建物序列，每項 ``(cx, cy, w, h)``。

    Returns:
        特徵長度（像素），至少為 1。
    """
    return l_char_from_buildings(buildings)


def compute_characteristic_length_from_png(
    png_path: str,
    invert: bool,
    nx: int,
    ny: int,
) -> int:
    """便利包裝：讀取 PNG 後計算 L_char。

    Args:
        png_path: 遮罩 PNG 路徑。
        invert: 亮色為固體的旗標。
        nx: domain 寬度。
        ny: domain 高度。

    Returns:
        特徵長度（像素）。
    """
    solid_hw = png_to_binary_mask(png_path, invert, nx, ny)
    solid_taichi = solid_hw.T  # (nx, ny)
    return compute_characteristic_length(solid_taichi, nx, ny)


def compute_blockage_metrics(
    solid_mask: np.ndarray,
    nx: int,
    ny: int,
    pad_right: int = 512,
    buffer: int = 128,
) -> BlockageMetrics:
    """計算各 X 截面的最大阻塞率與開放比。

    使用 5 像素滾動平均後取最大值，排除入口 5% 及右側 sponge 區域。

    Args:
        solid_mask: 布林陣列 (nx, ny)，Taichi 慣例。
        nx: domain 寬度。
        ny: domain 高度。
        pad_right: 右側 padding（sponge 區域）。
        buffer: 額外緩衝區。

    Returns:
        包含 max_blockage 與 open_fraction 的字典。
    """
    solid = np.asarray(solid_mask, dtype=bool)
    x_start = max(1, int(nx * 0.05))
    x_end = min(nx - 1, nx - pad_right - buffer)
    roi = solid[x_start:x_end, :]

    if roi.shape[0] == 0:
        return BlockageMetrics(max_blockage=0.0, open_fraction=1.0)

    blockage_per_x = np.mean(roi.astype(np.float32), axis=1)
    window = 5
    if len(blockage_per_x) >= window:
        kernel = np.ones(window, dtype=np.float32) / window
        smoothed = np.convolve(blockage_per_x, kernel, mode="valid")
    else:
        smoothed = blockage_per_x

    max_b = float(np.max(smoothed))
    return BlockageMetrics(max_blockage=max_b, open_fraction=1.0 - max_b)


def compute_blockage_metrics_from_png(
    png_path: str,
    invert: bool,
    nx: int,
    ny: int,
    pad_right: int = 512,
    buffer: int = 128,
) -> BlockageMetrics:
    """便利包裝：讀取 PNG 後計算阻塞率。

    Args:
        png_path: 遮罩 PNG 路徑。
        invert: 亮色為固體的旗標。
        nx: domain 寬度。
        ny: domain 高度。
        pad_right: 右側 padding。
        buffer: 額外緩衝區。

    Returns:
        包含 max_blockage 與 open_fraction 的字典。
    """
    solid_hw = png_to_binary_mask(png_path, invert, nx, ny)
    solid_taichi = solid_hw.T  # (nx, ny)
    return compute_blockage_metrics(solid_taichi, nx, ny, pad_right, buffer)


def compute_boundary_extent_metrics(
    solid_mask: np.ndarray,
    nx: int,
    ny: int,
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bot: int,
    l_char_px: int,
) -> BoundaryExtentMetrics:
    """計算邊界距離：各邊到最近固體格點的距離（px 及 L_char 比值）。

    Args:
        solid_mask: 布林陣列 (nx, ny)，Taichi 慣例。
        nx: domain 寬度。
        ny: domain 高度。
        pad_left: 左側 padding（入口側）。
        pad_right: 右側 padding（出口側）。
        pad_top: 上方 padding。
        pad_bot: 下方 padding。
        l_char_px: 特徵長度（像素）。

    Returns:
        包含各邊 px 距離及 L_char 比值的字典。
    """
    solid = np.asarray(solid_mask, dtype=bool)
    solid_xs = np.where(np.any(solid, axis=1))[0]
    solid_ys = np.where(np.any(solid, axis=0))[0]

    if len(solid_xs) == 0 or len(solid_ys) == 0:
        lc = max(l_char_px, 1)
        return BoundaryExtentMetrics(
            inlet_px=pad_left,
            outlet_px=pad_right,
            top_px=pad_top,
            bottom_px=pad_bot,
            inlet_over_lchar=pad_left / lc,
            outlet_over_lchar=pad_right / lc,
            top_over_lchar=pad_top / lc,
            bottom_over_lchar=pad_bot / lc,
        )

    first_solid_x = int(solid_xs[0])
    last_solid_x = int(solid_xs[-1])
    first_solid_y = int(solid_ys[0])
    last_solid_y = int(solid_ys[-1])

    inlet_px = first_solid_x
    outlet_px = (nx - 1) - last_solid_x
    bottom_px = first_solid_y
    top_px = (ny - 1) - last_solid_y

    lc = max(l_char_px, 1)
    return BoundaryExtentMetrics(
        inlet_px=inlet_px,
        outlet_px=outlet_px,
        top_px=top_px,
        bottom_px=bottom_px,
        inlet_over_lchar=inlet_px / lc,
        outlet_over_lchar=outlet_px / lc,
        top_over_lchar=top_px / lc,
        bottom_over_lchar=bottom_px / lc,
    )
