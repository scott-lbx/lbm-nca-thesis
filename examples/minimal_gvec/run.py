"""Generate a few building layouts and compute their geometric descriptors.

This is the smallest end-to-end slice of the pipeline: it needs no GPU, no
dataset and no trained weights, so it is a good first thing to run after
installing. It exercises layout generation, mask rasterisation, and the
14-dimensional `G_vec`.

Usage::

    python examples/minimal_gvec/run.py
    python examples/minimal_gvec/run.py --n 8 --seed 7 --out-dir /tmp/gvec_demo

Writes the mask PNGs and a `gvec.csv` into the output directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from lbmnca.morphospace.physics_proxy.gvec_api import (
    GVEC_DIM,
    GVEC_FEATURE_NAMES,
    compute_gvec_from_mask,
)


def _random_layout(rng: np.random.Generator, h: int, w: int, n_buildings: int) -> np.ndarray:
    """Rasterise a few non-overlapping rectangles into a solid mask.

    Deliberately simpler than the sampler used to build the real dataset: this
    example is about the descriptor, not about covering the design space.

    Args:
        rng: Source of randomness.
        h: Mask height in pixels.
        w: Mask width in pixels.
        n_buildings: How many rectangles to place.

    Returns:
        Boolean array of shape ``(h, w)`` where ``True`` marks solid.
    """
    mask = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(200):
        if placed >= n_buildings:
            break
        bw = int(rng.integers(w // 24, w // 8))
        bh = int(rng.integers(h // 16, h // 5))
        x0 = int(rng.integers(w // 8, w - bw - w // 8))
        y0 = int(rng.integers(h // 10, h - bh - h // 10))
        # Reject a placement that touches an existing building, so the gap
        # features have something meaningful to measure.
        pad = 8
        window = mask[max(0, y0 - pad) : y0 + bh + pad, max(0, x0 - pad) : x0 + bw + pad]
        if window.any():
            continue
        mask[y0 : y0 + bh, x0 : x0 + bw] = True
        placed += 1
    return mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="how many layouts to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, object]] = []
    for i in range(args.n):
        n_buildings = int(rng.integers(3, 9))
        mask = _random_layout(rng, args.height, args.width, n_buildings)

        # PNG convention: 0 is solid (building), 255 is open. The boolean array
        # convention is the other way round, True is solid.
        try:
            from PIL import Image

            png_path = args.out_dir / f"layout_{i:02d}.png"
            Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(png_path)
        except ImportError:
            png_path = None

        # Two mask polarities coexist in this pipeline and mixing them up is
        # silent: the descriptor comes back full of plausible numbers describing
        # the background as one huge building. PNG files use 0 for solid, which
        # is what `solid_is_zero=True` (the default) expects; the boolean array
        # here uses True for solid, so the flag has to be turned off.
        g_vec = compute_gvec_from_mask(mask, solid_is_zero=False)
        assert g_vec.shape == (GVEC_DIM,), f"expected {GVEC_DIM} dimensions, got {g_vec.shape}"

        print(f"\nlayout_{i:02d}  ({n_buildings} buildings" + (f", {png_path.name})" if png_path else ")"))
        for name, value in zip(GVEC_FEATURE_NAMES, g_vec):
            print(f"    {name:<38} {value: .5f}")

        row: dict[str, object] = {"layout": f"layout_{i:02d}", "n_buildings": n_buildings}
        row.update(dict(zip(GVEC_FEATURE_NAMES, (float(v) for v in g_vec))))
        rows.append(row)

    csv_path = args.out_dir / "gvec.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}")
    print("What each dimension actually measures: see docs/GVEC_SPEC.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
