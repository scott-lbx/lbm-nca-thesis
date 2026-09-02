"""static_builder + regime_features：static 4ch 與 cond_raw 的建構（純 numpy，無 GPU）。"""

from __future__ import annotations

import numpy as np
import pytest

from lbmnca.bridge.core.static_builder import build_static_raw
from lbmnca.bridge.geometry.regime_features import (
    BCR_TRAIN_RANGE,
    L_CHAR_DEFAULT,
    L_CHAR_TRAIN_RANGE,
    SIM_SCALE,
    compute_cond_raw,
    compute_l_char,
)

H, W = 360, 640


def mask_with_box(cx, cy, w, h):
    """[H,W] float32，1=fluid、0=solid；中央放一個矩形障礙。"""
    m = np.ones((H, W), dtype=np.float32)
    m[cy - h // 2 : cy + h // 2, cx - w // 2 : cx + w // 2] = 0.0
    return m


CFG = {"physics": {"re_levels": {1: 150, 2: 300, 3: 500}, "u_inlet_lu": 0.05}}


# ── static_builder ───────────────────────────────────────────────────────────


def test_static_raw_shape_and_channels():
    st = build_static_raw(mask_with_box(320, 180, 40, 60))
    assert st.shape == (4, H, W)
    assert st.dtype == np.float32


def test_sdf_sign_convention():
    """C0 SDF：fluid > 0、solid <= 0（v3 tanh_lchar 下門檻恆為 0）。"""
    m = mask_with_box(320, 180, 40, 60)
    sdf = build_static_raw(m)[0]
    assert (sdf[m > 0.5] > 0).all()
    assert (sdf[m < 0.5] <= 0).all()


def test_sdf_grows_with_distance_from_obstacle():
    m = mask_with_box(320, 180, 40, 60)
    sdf = build_static_raw(m)[0]
    near = sdf[180, 320 + 25]  # 貼著障礙右緣
    far = sdf[180, 320 + 200]  # 遠離
    assert far > near > 0


def test_pos_channels_are_unit_ramps():
    """C2/C3 = [0,1] 線性座標（raw identity，正規化階段才 passthrough）。"""
    st = build_static_raw(mask_with_box(320, 180, 40, 60))
    x_norm, y_norm = st[2], st[3]
    assert x_norm[0, 0] == pytest.approx(0.0)
    assert x_norm[0, -1] == pytest.approx(1.0)
    assert y_norm[0, 0] == pytest.approx(0.0)
    assert y_norm[-1, 0] == pytest.approx(1.0)
    assert np.allclose(x_norm[0], x_norm[-1])  # 沿 y 不變
    assert np.allclose(y_norm[:, 0], y_norm[:, -1])  # 沿 x 不變


def test_upstream_zero_before_obstacle_and_positive_after():
    """C1 累積上游障礙密度：障礙上游應為 0，下游 > 0。"""
    m = mask_with_box(320, 180, 40, 60)
    up = build_static_raw(m)[1]
    assert up[180, 100] == pytest.approx(0.0)
    assert up[180, 500] > 0.0


def test_empty_mask_is_finite():
    st = build_static_raw(np.ones((H, W), dtype=np.float32))
    assert np.isfinite(st).all()


# ── L_char ───────────────────────────────────────────────────────────────────


def test_l_char_is_cross_flow_extent_times_sim_scale():
    """L_char = 迎風橫向（y）投影寬 × SIM_SCALE（360×640 → 1920×1080）。"""
    m = mask_with_box(320, 180, 40, 60)  # y 方向 60 px
    assert compute_l_char(m) == pytest.approx(60.0 * SIM_SCALE)


def test_l_char_takes_max_over_buildings():
    m = np.ones((H, W), dtype=np.float32)
    m[100:130, 100:120] = 0.0  # y-extent 30
    m[200:270, 400:420] = 0.0  # y-extent 70 ← 最大
    assert compute_l_char(m) == pytest.approx(70.0 * SIM_SCALE)


def test_l_char_default_when_no_solid():
    assert compute_l_char(np.ones((H, W), dtype=np.float32)) == L_CHAR_DEFAULT


# ── cond_raw ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("level,re", [(1, 150.0), (2, 300.0), (3, 500.0)])
def test_wind_level_maps_to_re(level, re):
    c = compute_cond_raw(
        mask_with_box(320, 180, 40, 60), level, 0.0, CFG, g_vec=[0.0] * 14
    )
    assert c["Re"] == re
    assert c["p_vec"][0] == re


def test_p_vec_layout():
    """p_vec = [Re, U_inlet_lu, L_char_simpx]。"""
    c = compute_cond_raw(mask_with_box(320, 180, 40, 60), 2, 0.0, CFG, g_vec=[0.0] * 14)
    assert c["p_vec"] == [c["Re"], c["U_inlet"], c["L_char"]]
    assert c["U_inlet"] == 0.05


def test_u_inlet_missing_raises_not_defaults():
    """禁 fallback：缺 u_inlet_lu 必須 KeyError，不得靜默代 0.06。"""
    bad = {"physics": {"re_levels": {2: 300}}}
    with pytest.raises(KeyError):
        compute_cond_raw(mask_with_box(320, 180, 40, 60), 2, 0.0, bad, g_vec=[0.0] * 14)


def test_bcr_matches_solid_fraction():
    m = mask_with_box(320, 180, 40, 60)
    c = compute_cond_raw(m, 2, 0.0, CFG, g_vec=[0.0] * 14)
    assert c["BCR"] == pytest.approx(float((m < 0.5).sum()) / (H * W))


def test_out_of_distribution_warns_but_does_not_block():
    """守門是軟警示：超出訓練分布仍回完整 cond_raw。"""
    huge = mask_with_box(320, 180, 300, 300)  # L_char 900 sim px，遠超上界
    c = compute_cond_raw(huge, 2, 0.0, CFG, g_vec=[0.0] * 14)
    assert any("L_char" in w for w in c["warnings"])
    assert c["p_vec"][2] > L_CHAR_TRAIN_RANGE[1]
    assert c["g_vec"] is not None


def test_in_distribution_has_no_warnings():
    # 兩個守門要同時滿足：L_char = y-extent × 3 ∈ [32,128] → y-extent ≤ 42 px；
    # BCR ≥ 0.01 → solid ≥ 2304 px。100×40 = 4000 px（BCR 0.017、L_char 120）兩者皆過。
    m = mask_with_box(320, 180, 100, 40)
    c = compute_cond_raw(m, 2, 0.0, CFG, g_vec=[0.0] * 14)
    assert L_CHAR_TRAIN_RANGE[0] <= c["L_char"] <= L_CHAR_TRAIN_RANGE[1]
    assert BCR_TRAIN_RANGE[0] <= c["BCR"] <= BCR_TRAIN_RANGE[1]
    assert c["warnings"] == []


def test_wind_angle_passed_through():
    c = compute_cond_raw(
        mask_with_box(320, 180, 40, 60), 2, 37.5, CFG, g_vec=[0.0] * 14
    )
    assert c["wind_angle"] == 37.5
