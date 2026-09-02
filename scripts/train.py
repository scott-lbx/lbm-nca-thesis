"""Train the surrogate.

A thin wrapper over ``lbmnca.nca.tes.train`` that fails early with a readable
message when the dataset is missing, rather than deep inside the loader.

Usage::

    python scripts/train.py --config configs/nca/main.yaml
    python scripts/train.py --config configs/nca/main.yaml --resume runs/main/ckpt_step0002500.pth

The dataset is not distributed with this repository. See
``docs/DATA_GENERATION.md`` for how to regenerate it, and set
``LBMNCA_DATA_ROOT`` if you keep it outside the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from lbmnca.paths import data_root, resolve_placeholders


def _check_dataset(config_path: Path) -> list[str]:
    """Return the configured Zarr paths that do not exist.

    Args:
        config_path: Training configuration to inspect.

    Returns:
        Missing paths, in the order the configuration lists them.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    wanted: list[str] = []
    for key in ("zarr_paths", "val_zarr_paths", "stats_paths"):
        if key in cfg:
            wanted.extend(cfg[key])
    return [p for p in (resolve_placeholders(str(w)) for w in wanted) if not Path(p).exists()]


def main() -> int:
    argv = sys.argv[1:]
    if "--config" in argv:
        config_path = Path(argv[argv.index("--config") + 1])
        if not config_path.is_file():
            print(f"[error] configuration not found: {config_path}", file=sys.stderr)
            return 2
        missing = _check_dataset(config_path)
        if missing:
            print(
                f"[error] the dataset is missing {len(missing)} of the files this configuration\n"
                f"        needs. LBMNCA_DATA_ROOT is currently {data_root()}.\n",
                file=sys.stderr,
            )
            for path in missing[:6]:
                print(f"          {path}", file=sys.stderr)
            if len(missing) > 6:
                print(f"          ... and {len(missing) - 6} more", file=sys.stderr)
            print(
                "\n        The dataset is not distributed with this repository. Regenerate it\n"
                "        by following docs/DATA_GENERATION.md, or point LBMNCA_DATA_ROOT at an\n"
                "        existing copy.",
                file=sys.stderr,
            )
            return 2

    from lbmnca.nca.tes.train import main as train_main

    return train_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
