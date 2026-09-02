"""Mask perturbation utilities for TES warm-start dataset generation.

Four geometry perturbation operations on binary urban geometry masks.
Each operation changes only one dimension (single building), verifying
validity after perturbation.

Mask convention: True=solid(building), False=fluid.
Input mask shape: (H, W) bool array.
"""

from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np
import scipy.ndimage


# ── Domain constants (flat_diverse_gen compatible) ────────────────────────────
# Masks are 1920×1080, inner domain starts at PAD_LEFT from left, PAD_TOP from top
MASK_W: int = 1920
MASK_H: int = 1080
PAD_LEFT: int = 256
PAD_TOP: int = 156
PAD_RIGHT: int = 896   # right margin = 1920 - 256 - 768
PAD_BOT: int = 156     # bottom margin = 1080 - 156 - 768
INNER_W: int = 768
INNER_H: int = 768


# ── Validity check ──────────────────────────────────────────────────────────

def verify_mask_validity(mask_hw: np.ndarray) -> tuple[bool, str]:
    """Check if a perturbed mask is physically valid for LBM simulation.

    Args:
        mask_hw: Boolean mask (H, W), True=solid.

    Returns:
        (valid, reason): True with empty reason if valid; False with description.
    """
    h, w = mask_hw.shape

    # 1. No solid pixel on domain boundary (would block inlet/outlet)
    if mask_hw[0, :].any() or mask_hw[-1, :].any():
        return False, "solid pixels on top/bottom boundary"
    if mask_hw[:, 0].any() or mask_hw[:, -1].any():
        return False, "solid pixels on left/right boundary"

    # 2. BCR < 0.8 (total solid fraction)
    bcr = float(mask_hw.sum()) / (h * w)
    if bcr >= 0.8:
        return False, f"BCR={bcr:.3f} >= 0.8"

    # 3. Flow path not completely blocked: flood fill from inlet column
    # Simple check: left column is all fluid → outlet column is reachable
    fluid = ~mask_hw
    if not fluid[:, 0].any():
        return False, "inlet column fully blocked"
    if not fluid[:, -1].any():
        return False, "outlet column fully blocked"

    return True, ""


# ── Building extraction ──────────────────────────────────────────────────────

def _extract_buildings(mask_hw: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected solid regions as individual buildings.

    Returns:
        (labels_hw, n_buildings): labeled array and count (labels 1..n).
    """
    labeled, n = scipy.ndimage.label(mask_hw.astype(np.int32))
    return labeled, n


def _building_bbox(labels_hw: np.ndarray, label_id: int) -> tuple[int, int, int, int]:
    """Return (row_min, row_max, col_min, col_max) for a labeled building."""
    rows, cols = np.where(labels_hw == label_id)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


# ── Perturbation operations ──────────────────────────────────────────────────

def perturb_translate(
    mask_hw: np.ndarray,
    dx_range: tuple[int, int] = (3, 8),
    max_tries: int = 100,
    rng: Optional[random.Random] = None,
) -> Optional[np.ndarray]:
    """Translate one randomly chosen building by (dx, dy) pixels.

    Args:
        mask_hw: Input boolean mask (H, W).
        dx_range: Min/max absolute displacement per axis (px).
        max_tries: Maximum random attempts before giving up.
        rng: Optional Random instance for reproducibility.

    Returns:
        Perturbed mask or None if all attempts failed validity check.
    """
    if rng is None:
        rng = random.Random()

    labeled, n = _extract_buildings(mask_hw)
    if n == 0:
        return None

    for _ in range(max_tries):
        bld_id = rng.randint(1, n)
        building_pixels = (labeled == bld_id)

        dx = rng.randint(dx_range[0], dx_range[1]) * rng.choice([-1, 1])
        dy = rng.randint(dx_range[0], dx_range[1]) * rng.choice([-1, 1])

        # Shift the building pixels
        shifted = np.roll(np.roll(building_pixels, dy, axis=0), dx, axis=1)

        # Build candidate mask
        candidate = mask_hw.copy()
        candidate[building_pixels] = False   # remove original
        candidate[shifted] = True            # place shifted

        valid, _ = verify_mask_validity(candidate)
        if valid:
            return candidate

    return None


def perturb_scale(
    mask_hw: np.ndarray,
    scale_range: tuple[float, float] = (0.90, 1.10),
    max_tries: int = 100,
    rng: Optional[random.Random] = None,
) -> Optional[np.ndarray]:
    """Scale one randomly chosen building around its centroid.

    Args:
        mask_hw: Input boolean mask (H, W).
        scale_range: Min/max scale factor.
        max_tries: Maximum random attempts.
        rng: Optional Random instance.

    Returns:
        Perturbed mask or None on failure.
    """
    if rng is None:
        rng = random.Random()

    labeled, n = _extract_buildings(mask_hw)
    if n == 0:
        return None

    for _ in range(max_tries):
        bld_id = rng.randint(1, n)
        rmin, rmax, cmin, cmax = _building_bbox(labeled, bld_id)

        cx = (cmin + cmax) / 2.0
        cy = (rmin + rmax) / 2.0
        scale = rng.uniform(scale_range[0], scale_range[1])

        # New bounding box after scaling
        half_w = (cmax - cmin) / 2.0 * scale
        half_h = (rmax - rmin) / 2.0 * scale

        new_cmin = int(round(cx - half_w))
        new_cmax = int(round(cx + half_w))
        new_rmin = int(round(cy - half_h))
        new_rmax = int(round(cy + half_h))

        # Clamp to domain (leaving 1px margin for boundary check)
        H, W = mask_hw.shape
        new_cmin = max(1, new_cmin)
        new_cmax = min(W - 2, new_cmax)
        new_rmin = max(1, new_rmin)
        new_rmax = min(H - 2, new_rmax)

        if new_cmax <= new_cmin or new_rmax <= new_rmin:
            continue

        candidate = mask_hw.copy()
        # Remove original building
        candidate[labeled == bld_id] = False
        # Place scaled building
        candidate[new_rmin:new_rmax + 1, new_cmin:new_cmax + 1] = True

        valid, _ = verify_mask_validity(candidate)
        if valid:
            return candidate

    return None


def perturb_add_bldg(
    mask_hw: np.ndarray,
    size_range: tuple[int, int] = (15, 40),
    max_bcr_increase: float = 0.05,
    max_tries: int = 200,
    rng: Optional[random.Random] = None,
) -> Optional[np.ndarray]:
    """Add one small rectangular building in the inner domain area.

    Args:
        mask_hw: Input boolean mask (H, W).
        size_range: Min/max building side length (px).
        max_bcr_increase: Maximum allowed BCR increase.
        max_tries: Maximum random placement attempts.
        rng: Optional Random instance.

    Returns:
        Perturbed mask or None if placement fails.
    """
    if rng is None:
        rng = random.Random()

    H, W = mask_hw.shape
    orig_bcr = float(mask_hw.sum()) / (H * W)

    # Define placement region (inner domain with margin)
    margin = 10
    x_lo = PAD_LEFT + margin
    x_hi = PAD_LEFT + INNER_W - margin
    y_lo = PAD_TOP + margin
    y_hi = PAD_TOP + INNER_H - margin

    for _ in range(max_tries):
        bw = rng.randint(size_range[0], size_range[1])
        bh = rng.randint(size_range[0], size_range[1])

        if x_lo + bw >= x_hi or y_lo + bh >= y_hi:
            continue

        cx = rng.randint(x_lo + bw // 2, x_hi - bw // 2)
        cy = rng.randint(y_lo + bh // 2, y_hi - bh // 2)

        r0 = cy - bh // 2
        r1 = cy + bh // 2
        c0 = cx - bw // 2
        c1 = cx + bw // 2

        r0, r1 = max(1, r0), min(H - 2, r1)
        c0, c1 = max(1, c0), min(W - 2, c1)

        # Check overlap with existing buildings
        if mask_hw[r0:r1 + 1, c0:c1 + 1].any():
            continue

        candidate = mask_hw.copy()
        candidate[r0:r1 + 1, c0:c1 + 1] = True

        new_bcr = float(candidate.sum()) / (H * W)
        if new_bcr - orig_bcr > max_bcr_increase:
            continue

        valid, _ = verify_mask_validity(candidate)
        if valid:
            return candidate

    return None


def perturb_remove_bldg(
    mask_hw: np.ndarray,
    min_buildings: int = 4,
    rng: Optional[random.Random] = None,
) -> Optional[np.ndarray]:
    """Remove the smallest building (by area) from the mask.

    Args:
        mask_hw: Input boolean mask (H, W).
        min_buildings: Minimum number of buildings required to proceed.
        rng: Optional Random instance (unused, for API consistency).

    Returns:
        Perturbed mask or None if too few buildings.
    """
    labeled, n = _extract_buildings(mask_hw)
    if n < min_buildings:
        return None

    # Find smallest building by pixel count
    areas = [(int((labeled == i).sum()), i) for i in range(1, n + 1)]
    areas.sort()
    smallest_id = areas[0][1]

    candidate = mask_hw.copy()
    candidate[labeled == smallest_id] = False

    valid, _ = verify_mask_validity(candidate)
    return candidate if valid else None


# ── Batch perturbation ────────────────────────────────────────────────────────

PERTURB_FUNCS = {
    "translate":   perturb_translate,
    "scale":       perturb_scale,
    "add_bldg":    perturb_add_bldg,
    "remove_bldg": perturb_remove_bldg,
}


def apply_perturbation(
    mask_hw: np.ndarray,
    perturb_type: str,
    rng: Optional[random.Random] = None,
    **kwargs,
) -> Optional[np.ndarray]:
    """Apply a named perturbation.

    Args:
        mask_hw: Input boolean mask.
        perturb_type: One of 'translate', 'scale', 'add_bldg', 'remove_bldg'.
        rng: Optional Random instance.
        **kwargs: Forwarded to the perturbation function.

    Returns:
        Perturbed mask or None on failure.

    Raises:
        ValueError: If perturb_type is unknown.
    """
    if perturb_type not in PERTURB_FUNCS:
        raise ValueError(
            f"Unknown perturb_type '{perturb_type}'. "
            f"Valid: {list(PERTURB_FUNCS.keys())}"
        )
    fn = PERTURB_FUNCS[perturb_type]
    if perturb_type == "remove_bldg":
        return fn(mask_hw, rng=rng, **kwargs)
    return fn(mask_hw, rng=rng, **kwargs)


def generate_variants(
    mask_hw: np.ndarray,
    perturb_types: list[str] = ("translate", "scale", "add_bldg", "remove_bldg"),
    seed: Optional[int] = None,
    magnitudes: Optional[dict[str, float]] = None,
) -> dict[str, Optional[np.ndarray]]:
    """Generate one variant per perturbation type.

    Args:
        mask_hw: Source boolean mask (H, W).
        perturb_types: Which perturbation types to apply.
        seed: Optional random seed for reproducibility.
        magnitudes: Optional dict mapping perturb_type → scalar magnitude.
            translate  → max pixel shift; dx_range = (mag//2, mag)
            scale      → deviation;       scale_range = (1-mag, 1+mag)
            add_bldg   → max side px;     size_range = (mag//2, mag)
            remove_bldg → ignored

    Returns:
        Dict mapping perturb_type → variant mask (or None if failed).
    """
    rng = random.Random(seed)
    results: dict[str, Optional[np.ndarray]] = {}
    for pt in perturb_types:
        kwargs: dict = {}
        if magnitudes and pt in magnitudes:
            mag = magnitudes[pt]
            if pt == "translate" and mag > 0:
                lo = max(3, int(mag * 0.5))
                hi = max(lo + 1, int(mag))
                kwargs["dx_range"] = (lo, hi)
            elif pt == "scale" and mag > 0:
                kwargs["scale_range"] = (max(0.10, 1.0 - mag), min(2.0, 1.0 + mag))
            elif pt == "add_bldg" and mag > 0:
                lo = max(15, int(mag * 0.5))
                hi = max(lo + 1, int(mag))
                kwargs["size_range"] = (lo, hi)
        results[pt] = apply_perturbation(mask_hw, pt, rng=rng, **kwargs)
    return results
