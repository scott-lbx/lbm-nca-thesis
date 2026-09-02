"""
公開 API: Mask I/O 與 SDF 計算。

專案中所有 mask 相關操作的正式實作 (canonical implementation)。
取代並整合以下位置的重複邏輯：
  - src/tools/config_utils/mask_io.py
  - src/lbm_mrt_les/utils/mask_utils.py
  - src/lbm_mrt_les/io/LBM_Writer.py (SDF 區段)
"""

import cv2
import numpy as np
import scipy.ndimage


def png_to_binary_mask(
    png_path: str,
    invert: bool,
    nx: int,
    ny: int,
) -> np.ndarray:
    """讀取 PNG 遮罩，回傳影像慣例 (H, W) 的二值固體遮罩。

    Args:
        png_path: 灰階 PNG 檔案路徑。
        invert: True 時亮色像素 (>127) 視為固體；False 時暗色為固體。
        nx: domain 寬度（像素），對應 resize 目標寬。
        ny: domain 高度（像素），對應 resize 目標高。

    Returns:
        布林陣列 (ny, nx)，影像慣例 (row=y, col=x)。True = 固體, False = 流體。

    Raises:
        ValueError: 無法讀取 PNG 時拋出。
    """
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"無法讀取圖片: {png_path}")

    if img.shape != (ny, nx):
        img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_NEAREST)

    solid_hw = (img > 127) if invert else (img < 127)
    return solid_hw  # (ny, nx) = (H, W)


def png_to_binary_mask_embedded(
    png_path: str,
    invert: bool,
    nx: int,
    ny: int,
    embed_offset_col: int,
    embed_offset_row: int,
) -> np.ndarray:
    """POI mask PNG を solver domain (ny, nx) に埋め込む。境界は free flow で埋める。

    Args:
        png_path: POI mask PNG 路徑（mask 原始尺寸，不等於 nx, ny）。
        invert: True → 亮色為固體；False → 暗色為固體。
        nx: solver domain 寬度（px）。
        ny: solver domain 高度（px）。
        embed_offset_col: mask 左邊緣距 solver domain 左邊界的距離（image convention，col 方向）。
        embed_offset_row: mask 上邊緣距 solver domain 上邊界的距離（image convention，row 方向）。

    Returns:
        布林陣列 (ny, nx)，影像慣例。True = 固體，False = 流體。
        外部 sponge+buffer 區域全部填 False（free flow）。

    Raises:
        ValueError: 無法讀取 PNG 時拋出。
    """
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"無法讀取圖片: {png_path}")

    mask_h, mask_w = img.shape[:2]
    solid_nat = (img > 127) if invert else (img < 127)

    # 建立全域 False（free flow），再把 POI mask 嵌入
    full_solid_hw = np.zeros((ny, nx), dtype=bool)
    full_solid_hw[
        embed_offset_row : embed_offset_row + mask_h,
        embed_offset_col : embed_offset_col + mask_w,
    ] = solid_nat
    return full_solid_hw


def binary_mask_to_solver_mask(mask_hw: np.ndarray) -> np.ndarray:
    """將影像慣例 (H, W) 遮罩轉置為 Taichi field 慣例 (nx, ny)。

    Args:
        mask_hw: 布林或浮點遮罩，形狀 (H, W)。

    Returns:
        形狀 (W, H) = (nx, ny) 的陣列，可直接填入 Taichi field。
    """
    return mask_hw.T


def binary_mask_to_sdf(mask_hw: np.ndarray) -> np.ndarray:
    """從二值遮罩計算 Signed Distance Field (SDF)。

    流體區域為正值，固體區域為負值。

    Args:
        mask_hw: 二值遮罩 (H, W)，數值 > 0.5 視為固體。

    Returns:
        float32 陣列 (H, W)。正值 = 流體距離, 負值 = 固體距離。
    """
    solid = (np.asarray(mask_hw, dtype=np.float32) > 0.5).astype(np.float32)
    dist_fluid = scipy.ndimage.distance_transform_edt(1.0 - solid)
    dist_solid = scipy.ndimage.distance_transform_edt(solid)
    return (dist_fluid - dist_solid).astype(np.float32)
