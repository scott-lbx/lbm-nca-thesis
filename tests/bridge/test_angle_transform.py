"""AngleTransform：旋轉數學與座標契約。"""
from __future__ import annotations

import math

import pytest

from lbmnca.bridge.geometry.angle_transform import AngleTransform


def test_identity_is_noop():
    pts = [[0.0, 0.0], [3.0, 4.0], [-2.0, 7.5]]
    out = AngleTransform.no_transform().to_canonical(pts)
    for (x, y), (ox, oy) in zip(pts, out):
        assert ox == pytest.approx(x)
        assert oy == pytest.approx(y)


@pytest.mark.parametrize("deg", [0.0, 15.0, 30.0, 90.0, 180.0, -45.0])
def test_to_canonical_aligns_that_direction_to_plus_x(deg):
    """一條指向 deg 的向量，轉 canonical 後應落在 +x 上。"""
    v = [[0.0, 0.0], [math.cos(math.radians(deg)), math.sin(math.radians(deg))]]
    (_, tip) = AngleTransform(deg).to_canonical(v)
    assert tip[0] == pytest.approx(1.0, abs=1e-12)
    assert tip[1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("deg", [13.0, 47.0, -80.0])
def test_rotation_preserves_lengths_and_angles(deg):
    """旋轉是剛體變換：邊長與夾角不變。"""
    pts = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]
    out = AngleTransform(deg).to_canonical(pts)
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        oa, ob = out[i], out[(i + 1) % 4]
        assert math.dist(oa, ob) == pytest.approx(math.dist(a, b), abs=1e-12)


@pytest.mark.parametrize("deg", [23.0, -67.0])
def test_center_only_shifts_result(deg):
    """旋轉中心只改變平移量；呼叫端隨後的 bbox 正規化會把它消掉。"""
    pts = [[1.0, 2.0], [5.0, 2.0], [5.0, 9.0]]
    a = AngleTransform(deg).to_canonical(pts, center=(0.0, 0.0))
    b = AngleTransform(deg).to_canonical(pts, center=(3.3, -1.7))
    dx = b[0][0] - a[0][0]
    dy = b[0][1] - a[0][1]
    for pa, pb in zip(a, b):
        assert pb[0] - pa[0] == pytest.approx(dx, abs=1e-12)
        assert pb[1] - pa[1] == pytest.approx(dy, abs=1e-12)


def test_no_inverse_velocity_rotation_api():
    """GGUI 只顯示 canonical frame，不存在速度場逆旋轉路徑。

    先前 ``rotate_velocity_field`` / ``get_inverse_params`` 全 repo 無呼叫點，
    存在只會讓讀者誤以為畫面已轉回 Rhino 方位。移除後由本測固定該契約。
    """
    import lbmnca.bridge.geometry.angle_transform as mod

    assert not hasattr(mod, "rotate_velocity_field")
    assert not hasattr(AngleTransform, "get_inverse_params")
