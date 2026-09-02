"""Load a trained checkpoint and run the surrogate, without any dataset present.

The training entry point checks for the Zarr dataset before it builds anything,
so it cannot be used on a machine that has only the weights. This script goes
straight to the model factory instead, which needs the architecture
configuration and nothing else.

Usage::

    python scripts/run_inference.py \\
        --checkpoint checkpoints/main_v4.pth \\
        --config configs/nca/main.yaml \\
        --steps 8

With ``--mask`` it runs the surrogate on a layout from a PNG; without one it
only builds the model and reports its identity, which is enough to verify an
installation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from lbmnca.nca.tes.config import load_tes_config
from lbmnca.nca.tes.models.model_factory import build_tes_model
from lbmnca.paths import REPO_ROOT

#: Architecture fields recorded inside a checkpoint at training time.
_IDENTITY_KEYS: tuple[str, ...] = (
    "model.arch",
    "model.state_dim",
    "model.static_dim",
    "model.physics_dim",
    "model.hidden_dim",
    "model.cond_dim",
    "model.nca_channels",
)


def verify_identity(ckpt: dict, cfg) -> list[str]:
    """Compare the checkpoint's recorded architecture against the configuration.

    Loading is not strict, so a mismatched configuration would leave parts of the
    network randomly initialised and still produce plausible-looking output. That
    failure is silent, which is why this check exists.

    Args:
        ckpt: The loaded checkpoint dictionary.
        cfg: Configuration produced by ``load_tes_config``.

    Returns:
        Field descriptions that disagree; empty when the two match.
    """
    bad: list[str] = []
    for key in _IDENTITY_KEYS:
        if key not in ckpt:
            continue
        attr = key.split(".", 1)[1]
        want = ckpt[key]
        got = getattr(cfg.model, attr)
        if got != want:
            bad.append(f"{attr}: checkpoint={want} config={got}")
    return bad


def load_model(ckpt_path: Path, config_path: Path, device: str) -> torch.nn.Module:
    """Build the network from its configuration and load the trained weights.

    Args:
        ckpt_path: Slimmed checkpoint under ``checkpoints/``.
        config_path: The configuration the checkpoint was trained with.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        The model in evaluation mode.

    Raises:
        KeyError: The checkpoint has no ``model`` entry.
        ValueError: The checkpoint and the configuration describe different
            architectures.
    """
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"{ckpt_path} has no 'model' key; top level holds {sorted(ckpt)[:12]}")

    cfg = load_tes_config(str(config_path))
    mismatch = verify_identity(ckpt, cfg)
    if mismatch:
        raise ValueError(
            "Checkpoint and configuration describe different architectures:\n  "
            + "\n  ".join(mismatch)
        )

    model = build_tes_model(cfg.model)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing parameters, first few: {list(missing)[:5]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected parameters, first few: {list(unexpected)[:5]}")

    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    step = ckpt["step"] if "step" in ckpt else "unrecorded"
    print(f"[ok] loaded {ckpt_path.name}: step={step}, parameters={n_params:,}")
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "main_v4.pth")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "nca" / "main.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mask", type=Path, default=None, help="layout PNG to run the surrogate on")
    ap.add_argument("--steps", type=int, default=8, help="number of update steps when --mask is given")
    ap.add_argument("--out", type=Path, default=None, help="where to write the predicted field as .npy")
    args = ap.parse_args()

    for path in (args.checkpoint, args.config):
        if not path.is_file():
            print(f"[error] not found: {path}", file=sys.stderr)
            if path.suffix == ".pth":
                print("        Weights are stored with Git LFS. Try: git lfs pull", file=sys.stderr)
            return 2

    model = load_model(args.checkpoint, args.config, args.device)

    if args.mask is None:
        print("[ok] model built and weights loaded. Pass --mask to run it on a layout.")
        return 0

    from lbmnca.morphospace.physics_proxy.gvec_api import compute_gvec

    g_vec = compute_gvec(str(args.mask))
    print(f"[ok] G_vec from {args.mask.name}: {np.round(g_vec, 4).tolist()}")
    print(
        "[note] Running the surrogate on a raw mask also needs the static channels and the\n"
        "       physical parameters (Re, inlet velocity, characteristic length). The live\n"
        "       bridge assembles those for you: python -m lbmnca.bridge.main"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
