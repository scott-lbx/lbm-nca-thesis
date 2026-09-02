"""
config_utils/mask_io.py -- 薄包裝層，委派至 public.mask。

正式實作位於 src/lbm_mrt_les/public/mask.py。
"""

from lbmnca.lbm.lbm_mrt_les.public.mask import binary_mask_to_solver_mask, png_to_binary_mask


def load_solid_mask(png_path: str, invert: bool, nx: int, ny: int):
    """讀取 PNG 遮罩，回傳 (nx, ny) Taichi 慣例的布林陣列。

    委派至 public.mask.png_to_binary_mask + transpose。
    """
    mask_hw = png_to_binary_mask(png_path, invert, nx, ny)
    return binary_mask_to_solver_mask(mask_hw)
