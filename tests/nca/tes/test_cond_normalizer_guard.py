"""TESCondNormalizer 退化維守衛（g 側與 p 側行為必須一致）。

DEEP_AUDIT_2026-07 Stage 2.9。含負向測試：竄改 std 使其退化 -> 必須輸出 0。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lbmnca.nca.tes.data.normalizer import TESCondNormalizer

_DEG = TESCondNormalizer.DEGENERATE_STD


def _make(g_std_over: dict[int, float] | None = None,
          p_std_over: dict[int, float] | None = None) -> TESCondNormalizer:
    """建一個各維皆健康的 normalizer，再依 override 把指定維壓成退化。"""
    g_std = np.full(14, 2.0, dtype=np.float32)
    p_std = np.array([100.0, 50.0, 40.0], dtype=np.float32)
    for i, v in (g_std_over or {}).items():
        g_std[i] = v
    for i, v in (p_std_over or {}).items():
        p_std[i] = v
    return TESCondNormalizer(
        g_mean=np.full(14, 1.0, dtype=np.float32), g_std=g_std,
        p_mean=np.array([300.0, 0.05, 180.0], dtype=np.float32), p_std=p_std,
    )


# ---------------------------------------------------------------- 正向：健康維
def test_healthy_dims_are_plain_zscore():
    nz = _make()
    g = torch.full((2, 14), 5.0)
    p = torch.tensor([[400.0, 0.07, 220.0], [200.0, 0.03, 140.0]])
    assert torch.allclose(nz.normalize_g(g), (g - 1.0) / (2.0 + 1e-8))
    exp = (p - torch.tensor([300.0, 0.05, 180.0])) / (
        torch.tensor([100.0, 50.0, 40.0]) + 1e-8)
    assert torch.allclose(nz.normalize_p(p), exp)


# -------------------------------------------------- 負向：竄改 std 使其退化 -> 0
@pytest.mark.parametrize("bad_std", [0.0, 1e-8, 1e-7, _DEG])
@pytest.mark.parametrize("idx", [0, 7, 13])
def test_g_degenerate_dim_outputs_zero(idx: int, bad_std: float):
    nz = _make(g_std_over={idx: bad_std})
    out = nz.normalize_g(torch.full((3, 14), 1e6))     # 極端 OOD 輸入
    assert out[:, idx].abs().max().item() == 0.0
    other = [j for j in range(14) if j != idx]
    assert out[:, other].abs().min().item() > 0.0      # 其餘維不受影響


@pytest.mark.parametrize("bad_std", [0.0, 1e-8, 1e-7, _DEG])
@pytest.mark.parametrize("idx", [0, 1, 2])
def test_p_degenerate_dim_outputs_zero(idx: int, bad_std: float):
    nz = _make(p_std_over={idx: bad_std})
    out = nz.normalize_p(torch.tensor([[1e6, 1e6, 1e6]]))
    assert out[:, idx].abs().max().item() == 0.0


def test_p_guard_kills_u_inlet_blowup():
    """真實情境：p_std[1]=1.259e-07（canonical 合併值），部署給 U_inlet=0.06。"""
    nz = _make(p_std_over={1: 1.2590757592079171e-07})
    out = nz.normalize_p(torch.tensor([[300.0, 0.06, 180.0]]))
    assert out[0, 1].item() == 0.0
    assert abs(out[0, 0].item()) < 10.0 and abs(out[0, 2].item()) < 10.0


# ------------------------------------------------------- 兩側守衛行為必須一致
@pytest.mark.parametrize("bad_std", [0.0, 1e-8, 1e-7, _DEG])
def test_g_and_p_guards_agree(bad_std: float):
    """同一 std、同一輸入偏移下，g 側與 p 側必須給出相同結果。"""
    nz = _make(g_std_over={3: bad_std}, p_std_over={1: bad_std})
    g = torch.zeros(1, 14); g[0, 3] = 1.0 + 123.0        # 偏離 g_mean=1.0 共 123
    p = torch.tensor([[300.0, 0.05 + 123.0, 180.0]])     # 偏離 p_mean 同樣 123
    assert nz.normalize_g(g)[0, 3].item() == nz.normalize_p(p)[0, 1].item() == 0.0


def test_threshold_is_exclusive_above():
    """略高於門檻者不得被歸零（守衛不可過度攔截）。"""
    nz = _make(p_std_over={1: _DEG * 10})
    out = nz.normalize_p(torch.tensor([[300.0, 0.05 + 1e-3, 180.0]]))
    assert out[0, 1].abs().item() > 0.0


# ------------------------------------------- 釘住 cond_17D 的 idx 15 佈局
def test_cond17d_u_inlet_index_is_15():
    """cond_17D = [g_norm(14), p_norm(3)]，U_inlet 是 p 的第 2 維 → 全域 idx 15。

    ★2026-08-05：原測 cond_19D（17D ＋ 兩個時間維）。時間維在 canonical 上恆為
    常數填補，已隨 19D 一併移除；idx 15 的佈局不受影響（時間維接在尾端 17/18）。
    """
    nz = _make(p_std_over={1: 1e-8})
    cond = nz.build_cond17d(torch.zeros(1, 14), torch.zeros(1, 3))
    assert cond.shape == (1, 17)
    assert cond[0, 15].item() == 0.0
