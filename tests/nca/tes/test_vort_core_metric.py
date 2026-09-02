"""HD-M6：vortex core metric 測試（percentile + absolute floor）。

驗證 struct_vort_highmask_iou 的 percentile/abs_floor 擴充 + vort_core_mae/signed_corr。
用合成 9ch Mphy 場（identity normalizer）直接控制 vorticity。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.eval.visual_metrics import (
    struct_vort_highmask_iou,
    vort_core_mae,
    vort_core_signed_corr,
)


class _IdNorm:
    """identity normalizer：denorm_mphy 直接回傳輸入（測試用，rho=1）。

    ★2026-06-17（a24b8f5）colormap vis_range SSOT 重構後，struct_vort_highmask_iou /
    vort_core_mae / vort_core_signed_corr 的 abs_floor_frac 分支改讀
    ``normalizer.vor_vis_range``（原本硬編碼模組常數 ``_VOR_RANGE=0.04``，見該 commit
    diff）。此處補回同值常數，維持本檔 fixture 與production 讀取介面一致
    （非 fallback：這是測試專用假 normalizer 的固定回傳值，不影響生產碼的
    「vis_range 必來自 stats.json、缺鍵直接報錯」規則）。
    """

    def denorm_mphy(self, x):
        return x

    @property
    def vor_vis_range(self):
        return 0.04


def _field_from_velocity(vx: np.ndarray, vy: np.ndarray) -> torch.Tensor:
    """組 [1,9,H,W]：ch0=rho=1, ch3=jx=vx, ch5=jy=vy（vort = dvy/dx - dvx/dy）。"""
    H, W = vx.shape
    f = np.zeros((1, 9, H, W), dtype=np.float32)
    f[0, 0] = 1.0
    f[0, 3] = vx
    f[0, 5] = vy
    return torch.from_numpy(f)


def _make_vortex(H=48, W=48, cx=24, cy=24, strength=0.05, r=8):
    """造一個 Gaussian 渦：vy = strength·(x-cx)·exp(-d²/2r²) → vort 在中心達峰、向外遞減。

    使 |vort| 場有連續變化（非常數塊），讓高 percentile 能挑出真正 core peak、
    且 SignedCorr 在 core 上有非零變異。回 (field, fluid_mask)。
    """
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    env = np.exp(-d2 / (2.0 * r * r)).astype(np.float32)
    vy = (strength * (xs - cx) * env).astype(np.float32)
    vx = np.zeros((H, W), dtype=np.float32)
    fm = torch.ones(1, 1, H, W)
    return _field_from_velocity(vx, vy), fm


def test_legacy_p85_unchanged_default():
    """abs_floor_frac=0 預設 → legacy 行為（perfect pred IoU≈1）。"""
    f, fm = _make_vortex()
    nm = _IdNorm()
    iou = struct_vort_highmask_iou(f, f, fm, nm, percentile=85.0)
    assert iou > 0.95  # perfect pred


def test_perfect_pred_iou_near_one_all_percentiles():
    f, fm = _make_vortex()
    nm = _IdNorm()
    for pct in (85.0, 95.0, 97.0, 99.0):
        iou = struct_vort_highmask_iou(f, f, fm, nm, percentile=pct)
        assert iou > 0.9, f"perfect pred p{pct} IoU={iou}"


def test_shifted_core_lowers_iou():
    """pred 渦核位移 → IoU 下降。"""
    f_tgt, fm = _make_vortex(cx=24)
    f_pred, _ = _make_vortex(cx=34)  # 位移 10px
    nm = _IdNorm()
    iou_same = struct_vort_highmask_iou(f_tgt, f_tgt, fm, nm, percentile=99.0)
    iou_shift = struct_vort_highmask_iou(f_pred, f_tgt, fm, nm, percentile=99.0)
    assert iou_shift < iou_same
    assert iou_shift < 0.5


def test_near_zero_target_absfloor_returns_nan():
    """near-zero 渦量場 + abs_floor → 回 NaN（不量 noise）。"""
    H = W = 48
    rng = np.random.default_rng(0)  # 固定 seed，避免 flaky（偶發 noise 超 floor）
    tiny = (rng.standard_normal((H, W)) * 1e-5).astype(np.float32)  # 近零 noise
    f = _field_from_velocity(np.zeros((H, W), np.float32), tiny)
    fm = torch.ones(1, 1, H, W)
    nm = _IdNorm()
    # 無 floor：仍給數值（量 noise）
    iou_nofloor = struct_vort_highmask_iou(f, f, fm, nm, percentile=99.0, abs_floor_frac=0.0)
    assert not np.isnan(iou_nofloor)
    # 有 floor（25% VOR_RANGE）：target high-mask 近空 → NaN
    iou_floor = struct_vort_highmask_iou(f, f, fm, nm, percentile=99.0, abs_floor_frac=0.25)
    assert np.isnan(iou_floor)


def test_strong_vortex_passes_absfloor():
    """強渦場（> floor）+ abs_floor → 不被擋，給有效 IoU。"""
    f, fm = _make_vortex(strength=0.05, r=8)  # vort 可達 ~0.05 > 0.25*0.04=0.01
    nm = _IdNorm()
    iou = struct_vort_highmask_iou(f, f, fm, nm, percentile=99.0, abs_floor_frac=0.25)
    assert not np.isnan(iou)
    assert iou > 0.9


def test_empty_fluid_mask_nan():
    f, _ = _make_vortex()
    fm = torch.zeros(1, 1, 48, 48)
    nm = _IdNorm()
    assert np.isnan(struct_vort_highmask_iou(f, f, fm, nm, percentile=99.0))


def test_vort_core_mae_perfect_zero():
    f, fm = _make_vortex()
    nm = _IdNorm()
    mae = vort_core_mae(f, f, fm, nm, percentile=99.0)
    assert mae < 1e-6


def test_vort_core_signed_corr_perfect_one():
    f, fm = _make_vortex()
    nm = _IdNorm()
    corr = vort_core_signed_corr(f, f, fm, nm, percentile=99.0)
    assert corr > 0.99


def test_vort_core_mae_floor_near_zero_nan():
    H = W = 48
    rng = np.random.default_rng(1)  # 固定 seed，避免 flaky
    tiny = (rng.standard_normal((H, W)) * 1e-5).astype(np.float32)
    f = _field_from_velocity(np.zeros((H, W), np.float32), tiny)
    fm = torch.ones(1, 1, H, W)
    nm = _IdNorm()
    assert np.isnan(vort_core_mae(f, f, fm, nm, percentile=99.0, abs_floor_frac=0.25))
