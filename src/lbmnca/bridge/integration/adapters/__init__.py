"""Model backend adapters. Selected by config.model.backend.

Each adapter implements src.integration.model_backend.ModelBackend.
"""
from __future__ import annotations


def build_backend(config: dict):
    """Factory: build the ModelBackend named by config['model']['backend']."""
    backend = config["model"]["backend"]
    H = config["resolution"]["H"]
    W = config["resolution"]["W"]
    device = config["model"].get("device", "cuda")

    if backend == "dummy":
        from lbmnca.bridge.integration.adapters.dummy_adapter import DummyAdapter
        return DummyAdapter(
            H, W, device=device,
            hidden_channels=int(config["model"].get("hidden_channels", 16)),
        )
    if backend == "tes":
        from lbmnca.bridge.integration.adapters.tes_adapter import TESAdapter
        return TESAdapter(H, W, config, device=device)
    if backend == "decompose":
        from lbmnca.bridge.integration.adapters.decompose_adapter import DecomposeAdapter
        return DecomposeAdapter(H, W, config, device=device)
    raise ValueError(f"Unknown model backend: {backend!r}")


__all__ = ["build_backend"]
