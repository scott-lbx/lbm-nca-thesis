"""Shared constants + helpers for dataset config setup scripts.

Used by setup_cold.py and setup_warm.py. Holds the canonical TES solver
domain + boundary configuration so cold/warm setups stay in lock-step.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from PIL import Image


# ── Solver domain constants (TES, fixed) ──────────────────────────────────────

U_INLET_LU: float = 0.05

NX: int = 2560   # 1920 (mask) + 160 (in) + 480 (out)
NY: int = 1440   # 1080 (mask) + 180 + 180

EMBED_OFFSET_COL: int = 160
EMBED_OFFSET_ROW: int = 180

DOMAIN_ZONES: dict[str, Any] = {
    "sponge_in":               80,
    "sponge_out":             240,
    "sponge_top":              90,
    "sponge_bot":              90,
    "buffer_in":               80,
    "buffer_out":             240,
    "buffer_top":              90,
    "buffer_bot":              90,
    "buffer":                   0,
    "sponge_strength":        3.0,
    "sponge_relax_strength":  0.0,
    "sponge_reference_u_factor": 1.0,
    "sponge_reference_v_target": 0.0,
    "sponge_reference_mode":  "uniform_flow",
}


# ── Generic helpers ───────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_nu(l_char_px: float, re: int) -> float:
    """nu = U_inlet_lu * L_char_px / Re."""
    return float(U_INLET_LU * l_char_px / re)


def _save_yaml(data: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)


def _save_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_mask_bool(png_path: Path) -> np.ndarray:
    """Load mask PNG -> bool (H, W). True = solid.

    Convention: 0 = solid (building), 255 = fluid. pixel < 127 -> solid.
    """
    img = Image.open(str(png_path)).convert("L")
    return np.array(img, dtype=np.uint8) < 127


def _save_mask_png(mask_hw: np.ndarray, out_path: Path) -> None:
    """Save bool solid mask (True=solid) as PNG (0=solid, 255=fluid)."""
    pixels = np.where(mask_hw, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(str(out_path))


# ── Case YAML builder ─────────────────────────────────────────────────────────

def _build_case_config(
    *,
    case_id: str,
    geometry_id: str,
    dataset_name: str,
    ic_type: str,
    re: int,
    l_char_px: float,
    n_obstacles: int,
    bcr: float,
    mask_abs_path: str,
    total_ctus: float,
    spinup_ctus: float = 0.0,
    ke_convergence_threshold: float = 0.001,
    # warm-only fields
    parent_h5_path: Optional[str] = None,
    parent_case_id: Optional[str] = None,
    parent_dataset: Optional[str] = None,
    warm_k_level: Optional[int] = None,
    perturb_type: Optional[str] = None,
    perturb_magnitude: Optional[float] = None,
) -> dict[str, Any]:
    """Assemble a complete per-case YAML config dict (TES schema)."""
    nu = _compute_nu(l_char_px, re)

    cfg: dict[str, Any] = {
        "simulation": {
            "nx":                    NX,
            "ny":                    NY,
            "name":                  case_id,
            "nu":                    round(nu, 8),
            "ghost_moments_s":       1.2,
            "characteristic_length": float(l_char_px),
            "rho_in":                1.002,
            "rho_out":               1.0,
            "smagorinsky_constant":  0.1,
            "compute_step_size":     200,
            "warmup_steps":          0,
            "max_steps":             0,
        },
        "boundary_condition": {
            "type":  [0, 2, 1, 2],
            "value": [[0.05, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        },
        "mask": {
            "enable":           True,
            "type":             "png",
            "invert":           False,
            "path":             mask_abs_path,
            "embed_offset_col": EMBED_OFFSET_COL,
            "embed_offset_row": EMBED_OFFSET_ROW,
        },
        "domain_zones": dict(DOMAIN_ZONES),
    }

    tes: dict[str, Any] = {
        "case_id":                  case_id,
        "geometry_id":              geometry_id,
        "dataset_name":             dataset_name,
        "ic_type":                  ic_type,
        "total_ctus":               float(total_ctus),
        "spinup_ctus":              float(spinup_ctus),
        "re_target":                re,
        "n_obstacles":              n_obstacles,
        "bcr":                      round(float(bcr), 6),
        "g_vec":                    [],
        "ke_convergence_threshold": ke_convergence_threshold,
    }

    if ic_type == "warm":
        if parent_h5_path is None:
            raise ValueError(f"parent_h5_path required for warm case '{case_id}'")
        tes["parent_h5_path"]    = parent_h5_path
        tes["parent_case_id"]    = parent_case_id or ""
        tes["parent_dataset"]    = parent_dataset or ""
        tes["warm_k_level"]      = warm_k_level if warm_k_level is not None else 1
        tes["perturb_type"]      = perturb_type or ""
        tes["perturb_magnitude"] = float(perturb_magnitude) if perturb_magnitude is not None else 0.0

    cfg["tes"] = tes
    return cfg
