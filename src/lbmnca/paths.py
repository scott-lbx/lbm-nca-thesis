"""Central path resolution for the whole pipeline.

Every location that used to be an absolute path on the author's machine is
resolved here, from an environment variable with a sensible default. Nothing
outside this module should contain a machine-specific path.

Environment variables
---------------------
``LBMNCA_DATA_ROOT``
    Where generated data lives: masks, per-case configs, HDF5 trajectories and
    the converted Zarr datasets. Default: ``<repo>/data``.
``LBMNCA_CHECKPOINT_ROOT``
    Where model weights and normalisation statistics live.
    Default: ``<repo>/checkpoints``.

The dataset is not distributed with this repository; see
``docs/DATA_GENERATION.md`` for how to regenerate it into ``LBMNCA_DATA_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "data_root",
    "checkpoint_root",
    "dataset_dir",
    "zarr_dir",
    "design_space_dir",
    "resolve_placeholders",
]

#: Repository root, derived from this file's location (``src/lbmnca/paths.py``).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    """Read a directory from the environment, falling back to ``default``."""
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


def data_root() -> Path:
    """Root for all generated data."""
    return _env_path("LBMNCA_DATA_ROOT", REPO_ROOT / "data")


def checkpoint_root() -> Path:
    """Root for model weights and normalisation statistics."""
    return _env_path("LBMNCA_CHECKPOINT_ROOT", REPO_ROOT / "checkpoints")


def dataset_dir(name: str) -> Path:
    """Directory holding one simulated dataset (masks, configs, raw HDF5)."""
    return data_root() / "datasets" / name


def zarr_dir() -> Path:
    """Directory holding the converted Zarr datasets used for training."""
    return data_root() / "zarr"


def design_space_dir() -> Path:
    """Directory holding the sampled population and the frozen PCA cube."""
    return data_root() / "design_space"


#: Placeholders accepted inside configuration files.
_PLACEHOLDERS = {
    "${DATA_ROOT}": data_root,
    "${DATASET_ROOT}": data_root,
    "${ZARR_ROOT}": zarr_dir,
    "${CHECKPOINT_ROOT}": checkpoint_root,
    "${REPO_ROOT}": lambda: REPO_ROOT,
}


def resolve_placeholders(value: str) -> str:
    """Expand ``${...}`` placeholders in a configuration string.

    Args:
        value: A configuration value that may contain placeholders such as
            ``"${ZARR_ROOT}/D1-1_Cold_M400C400_unified.zarr"``.

    Returns:
        The string with every recognised placeholder replaced by an absolute
        path. Unrecognised text is returned unchanged.
    """
    for token, getter in _PLACEHOLDERS.items():
        if token in value:
            value = value.replace(token, str(getter()).replace("\\", "/"))
    return value
