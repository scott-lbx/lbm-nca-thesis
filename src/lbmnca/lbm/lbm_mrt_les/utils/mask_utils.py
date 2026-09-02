"""
mask_utils.py -- Solver 端的遮罩載入。

薄包裝層，委派至 public.mask。保留 create_mask(config, png_path)
簽名供 run_one_case.py 使用。
"""

import os

import numpy as np

from ..public.mask import (
    binary_mask_to_solver_mask,
    png_to_binary_mask,
    png_to_binary_mask_embedded,
)


def create_mask(config: dict, png_path: str) -> np.ndarray:
    """從 config 與 PNG 路徑載入遮罩，回傳 (nx, ny) Taichi 慣例陣列。

    v2.0 embed 模式：當 config["mask"] 含 embed_offset_col/row 時，
    PNG 以 POI 原始尺寸讀入，嵌入 solver domain（nx, ny）中；
    邊界填 False（free flow）。

    Args:
        config: 完整 case config 字典（需含 config["mask"] 與 config["simulation"]）。
        png_path: 遮罩 PNG 檔案路徑。

    Returns:
        布林陣列 (nx, ny)。True = 固體。
    """
    nx = config["simulation"]["nx"]
    ny = config["simulation"]["ny"]

    if config["mask"]["enable"] and config["mask"]["type"] == "png":
        if not png_path or not os.path.exists(png_path):
            raise FileNotFoundError(f"遮罩檔案不存在: {png_path}")
        invert = config["mask"]["invert"]
        embed_col = config["mask"].get("embed_offset_col", 0)  # optional — embed 模式才有
        embed_row = config["mask"].get("embed_offset_row", 0)  # optional — embed 模式才有
        if embed_col > 0 or embed_row > 0:
            mask_hw = png_to_binary_mask_embedded(
                png_path, invert, nx, ny, embed_col, embed_row
            )
        else:
            mask_hw = png_to_binary_mask(png_path, invert, nx, ny)
        return binary_mask_to_solver_mask(mask_hw)

    return np.zeros((nx, ny), dtype=bool)
