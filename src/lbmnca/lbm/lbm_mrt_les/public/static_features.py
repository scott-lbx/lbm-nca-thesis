"""
公開 API: ML 資料集的靜態特徵計算。

產出 (4, H, W) static_mask 堆疊：
  C0: SDF (signed distance field)
  C1: upstream_obstacle_sum (cumsum / width)
  C2: X_norm (位置嵌入, x / L_char_px)
  C3: Y_norm (位置嵌入, y / L_char_px)

取代並整合以下位置的邏輯：
  - src/post_process/h5_to_zarr_pipeline.py (compute_upstream_obstacle_sum)
  - src/post_process/zarr_utils.py (compute_position_embedding)
  - src/lbm_mrt_les/io/LBM_Writer.py (SDF 計算)
"""

import numpy as np

from .mask import binary_mask_to_sdf


def compute_upstream_obstacle_sum(binary_mask_hw: np.ndarray) -> np.ndarray:
    """計算累積上游障礙物密度。

    對每個像素 (y, x)，回傳其上游（左側，沿 axis=1）的固體像素佔比。

    Args:
        binary_mask_hw: (H, W) 遮罩，數值 > 0.5 視為固體。

    Returns:
        float32 陣列 (H, W)，數值範圍 [0, 1]。
    """
    solid = (np.asarray(binary_mask_hw, dtype=np.float32) > 0.5).astype(np.float32)
    _, width = solid.shape
    if width <= 0:
        return np.zeros_like(solid, dtype=np.float32)
    return np.cumsum(solid, axis=1, dtype=np.float32) / float(width)


def compute_position_embedding(
    target_h: int,
    target_w: int,
    l_char_px: float = 1.0,
    crop_w_px: float = 1.0,
    crop_h_px: float = 1.0,
) -> np.ndarray:
    """計算域相對位置嵌入網格，輸出 [0, 1]。

    X_01 = (x + 0.5) / target_w   → [0, 1]，0 = 入口側，1 = 出口側
    Y_01 = (y + 0.5) / target_h   → [0, 1]，0 = 底部，1 = 頂部

    與 L_char_px 無關，適合 NCA 學習入口/出口邊界差異。
    l_char_px / crop_w_px / crop_h_px 參數保留以維持向後相容，但不再使用。

    Args:
        target_h: 最終輸出高度（像素）。
        target_w: 最終輸出寬度（像素）。
        l_char_px: 保留參數（未使用）。
        crop_w_px: 保留參數（未使用）。
        crop_h_px: 保留參數（未使用）。

    Returns:
        float32 陣列 (2, H, W)，包含 [X_01, Y_01]，值域 (0, 1)。
    """
    x_coords = (np.arange(target_w, dtype=np.float32) + 0.5) / target_w
    y_coords = (np.arange(target_h, dtype=np.float32) + 0.5) / target_h

    x_grid = np.tile(x_coords, (target_h, 1))
    y_grid = np.tile(y_coords[:, np.newaxis], (1, target_w))

    return np.stack([x_grid, y_grid], axis=0).astype(np.float32)


def build_static_feature_stack(
    binary_mask_hw: np.ndarray,
    l_char_px: float,
    crop_w_px: float,
    crop_h_px: float,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """建構完整的 (4, target_h, target_w) 靜態特徵堆疊，供 ML 訓練使用。

    通道配置：
        C0: SDF（流體為正，固體為負）
        C1: upstream_obstacle_sum
        C2: X_norm（位置嵌入）
        C3: Y_norm（位置嵌入）

    輸入遮罩可以是任意大小（通常是 resize 後的 target 尺寸）。
    若輸入尺寸與 target 不同，會自動 resize。

    注意：回傳的是原始值（未正規化）。Z-score 正規化由
    h5_to_zarr_pipeline 使用 global_stats 在後續階段執行。

    Args:
        binary_mask_hw: (H, W) 二值遮罩，> 0.5 = 固體。
        l_char_px: 特徵長度（像素，resize 前的原始值）。
        crop_w_px: resize 前的原始裁切寬度。
        crop_h_px: resize 前的原始裁切高度。
        target_h: 輸出高度。
        target_w: 輸出寬度。

    Returns:
        float32 陣列 (4, target_h, target_w)。
    """
    import cv2

    mask = np.asarray(binary_mask_hw, dtype=np.float32)
    h_in, w_in = mask.shape

    # 若輸入尺寸與目標不同，先 resize
    if h_in != target_h or w_in != target_w:
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)

    sdf = binary_mask_to_sdf(mask)
    upstream = compute_upstream_obstacle_sum(mask)
    pos = compute_position_embedding(target_h, target_w, l_char_px, crop_w_px, crop_h_px)

    return np.stack([sdf, upstream, pos[0], pos[1]], axis=0).astype(np.float32)
