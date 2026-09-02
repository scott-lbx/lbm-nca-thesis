"""Tests for the H-series multi-scale utility ops."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.models.multiscale_ops import (
    downsample_mask,
    downsample_state,
    expand_block_halo,
    make_block_mask,
    upsample_state,
)


def test_downsample_then_upsample_roundtrip_shape() -> None:
    x = torch.randn(2, 4, 16, 16)
    d = downsample_state(x, ratio=4, mode="area")
    assert d.shape == (2, 4, 4, 4)
    u = upsample_state(d, (16, 16), mode="bilinear")
    assert u.shape == (2, 4, 16, 16)


def test_downsample_ratio_one_is_identity() -> None:
    x = torch.randn(1, 3, 8, 8)
    assert torch.equal(downsample_state(x, ratio=1), x)


def test_downsample_mask_is_binary() -> None:
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., :8, :8] = 1.0
    coarse = downsample_mask(mask, ratio=4, threshold=0.5)
    assert coarse.shape == (1, 1, 4, 4)
    assert set(coarse.unique().tolist()) <= {0.0, 1.0}


def test_make_block_mask_top_k() -> None:
    # 4 blocks of 4x4 in a 8x8 grid, 25% active → 1 block
    score = torch.zeros(1, 1, 8, 8)
    score[..., :4, :4] = 1.0  # top-left block has high score
    mask = make_block_mask(score, block_size=4, active_ratio=0.25)
    assert mask.shape == (1, 1, 2, 2)
    assert mask[0, 0, 0, 0].item() == 1.0
    assert mask.sum().item() == 1.0


def test_make_block_mask_invalid_ratio() -> None:
    score = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError):
        make_block_mask(score, block_size=4, active_ratio=1.5)


def test_expand_block_halo_grows_mask() -> None:
    mask = torch.zeros(1, 1, 4, 4)
    mask[0, 0, 1, 1] = 1.0
    expanded = expand_block_halo(mask, halo_blocks=1)
    # 3×3 dilation around (1,1) → 9 ones in the 4×4 block grid
    assert expanded.sum().item() == 9.0


def test_expand_block_halo_zero_passthrough() -> None:
    mask = torch.zeros(1, 1, 4, 4)
    mask[0, 0, 0, 0] = 1.0
    expanded = expand_block_halo(mask, halo_blocks=0)
    assert torch.equal(expanded, mask)
