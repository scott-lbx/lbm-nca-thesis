"""任意尺寸 PNG mask → canonical R×R float32 array。

輸入：flat_mask_gen 輸出的 PNG（0=建物/固體，255=自由流）。
輸出：(1, R, R) float32，occupancy convention：1=建物，0=自由流。
      use_sdf=True 時輸出 SDF ∈ [-1, 1]（正=自由流，負=建物內）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def mask_to_canonical(
    mask_path: Path,
    R: int = 64,
    pad_left: int = 500,
    pad_top: int = 384,
    inner_px: int = 768,
    use_sdf: bool = False,
) -> np.ndarray:
    """PNG mask → canonical (1, R, R) float32 tensor-ready array。

    Args:
        mask_path: PNG 路徑（0=建物，255=自由流）。
        R: 輸出解析度（R×R）。
        pad_left: 左邊距 pixels（flat_mask_gen.PAD_LEFT）。
        pad_top: 上邊距 pixels（flat_mask_gen.PAD_TOP）。
        inner_px: inner domain 邊長 pixels（flat_mask_gen._INNER_PX）。
        use_sdf: True → 輸出 SDF float；False → 輸出 binary float。

    Returns:
        np.ndarray shape (1, R, R), dtype float32.
    """
    img = Image.open(mask_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    # Crop inner domain
    crop = arr[pad_top:pad_top + inner_px, pad_left:pad_left + inner_px]

    # PNG: 0=建物 → invert: 1=建物（occupancy map）
    occ = 1.0 - crop / 255.0  # float, 1=building, 0=free

    # Resize to R×R
    resample = Image.BILINEAR if R < inner_px else Image.NEAREST
    img_r = Image.fromarray((occ * 255).astype(np.uint8), mode="L")
    img_r = img_r.resize((R, R), resample)
    occ_r = np.array(img_r, dtype=np.float32) / 255.0

    # 二值化（消除插值模糊）
    occ_r = (occ_r >= 0.5).astype(np.float32)

    if use_sdf:
        canonical = _binary_to_sdf(occ_r)
    else:
        canonical = occ_r  # [0, 1] binary

    return canonical[np.newaxis]  # (1, R, R)


def _binary_to_sdf(occ: np.ndarray) -> np.ndarray:
    """Binary occupancy (1=建物) → normalized SDF ∈ [-1, 1]。

    SDF > 0：自由流；SDF < 0：建物內部；SDF = 0：邊界。
    """
    from scipy.ndimage import distance_transform_edt

    occ_bool = occ > 0.5
    d_inside = distance_transform_edt(occ_bool)   # 建物內部距離
    d_outside = distance_transform_edt(~occ_bool)  # 自由流距離
    sdf = d_outside - d_inside  # positive=free, negative=building
    max_d = float(np.abs(sdf).max())
    if max_d < 1e-8:
        return np.zeros_like(sdf, dtype=np.float32)
    return (sdf / max_d).astype(np.float32)
