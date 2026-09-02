"""geometry_ingest：兩套 payload 方言 + GH world 方言的旋轉不變性。

核心迴歸：旋轉 boundary 不得改變建物在 canonical 座標系裡的形狀。舊實作先用 world
AABB 正規化、再由 window 旋轉，兩步合起來會把建物切變成平行四邊形。
"""
from __future__ import annotations

import math

import pytest

from lbmnca.bridge.core.geometry_ingest import _wind_angle_from_corners, ingest_analyze

SIM_W, SIM_H = 1920.0, 1080.0


def rot(x: float, y: float, deg: float) -> list[float]:
    a = math.radians(deg)
    return [x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)]


def rect_world(cx, cy, w, h, deg=0.0):
    """world 座標的旋轉矩形（CCW 4 頂點）。"""
    return [rot(cx - w / 2, cy - h / 2, deg), rot(cx + w / 2, cy - h / 2, deg),
            rot(cx + w / 2, cy + h / 2, deg), rot(cx - w / 2, cy + h / 2, deg)]


def edge_lengths(poly):
    n = len(poly)
    return [math.dist(poly[i], poly[(i + 1) % n]) for i in range(n)]


def corner_angles(poly):
    """每個頂點的內角（度）。矩形應恆為 90。"""
    n = len(poly)
    out = []
    for i in range(n):
        p, c, q = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        v1 = (p[0] - c[0], p[1] - c[1])
        v2 = (q[0] - c[0], q[1] - c[1])
        cosang = ((v1[0] * v2[0] + v1[1] * v2[1])
                  / (math.hypot(*v1) * math.hypot(*v2)))
        out.append(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    return out


# ── web 扁平 sim-px 方言 ──────────────────────────────────────────────────────

def test_web_dialect_detected_and_normalized():
    cmd = ingest_analyze({
        "action": "analyze",
        "buildings": [{"cx": 960, "cy": 540, "w": 192, "h": 108}],
        "wind_level": 3, "mode": "warm",
    })
    assert cmd["wind_angle"] == 0.0
    assert cmd["wind_level"] == 3
    assert cmd["mode"] == "warm"
    (poly,) = cmd["polygons"]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert min(xs) == pytest.approx((960 - 96) / SIM_W)
    assert max(xs) == pytest.approx((960 + 96) / SIM_W)
    assert min(ys) == pytest.approx((540 - 54) / SIM_H)
    assert max(ys) == pytest.approx((540 + 54) / SIM_H)


def test_web_dialect_clamps_out_of_domain():
    """超出 sim 域的矩形被夾到 [0,1]，不得產生負座標。"""
    cmd = ingest_analyze({
        "action": "analyze",
        "buildings": [{"cx": 20, "cy": 20, "w": 400, "h": 400}],
        "wind_level": 2, "mode": "fresh",
    })
    for x, y in cmd["polygons"][0]:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_web_dialect_empty_buildings_falls_to_world_branch():
    """buildings 為空 → 不是 web 方言，走 world 分支且不炸。"""
    cmd = ingest_analyze({"action": "analyze", "buildings": []})
    assert cmd["polygons"] == []


# ── GH world 多邊形方言 ──────────────────────────────────────────────────────

def test_wind_angle_from_longest_edge():
    corners = rect_world(96, 54, 192, 108, 30.0)
    assert _wind_angle_from_corners(corners) == pytest.approx(30.0, abs=1e-6)


MASK_W, MASK_H = 640, 360      # 推論域；[0,1] 座標乘回去才是實際光柵化的像素


def to_px(poly):
    """[0,1] 正規化座標 → mask 像素座標。

    [0,1] 正規化本身是非等向的（x 除以 boundary 長邊、y 除以短邊），因為 [0,1]²
    代表的是一塊 16:9 的區域；要判斷形狀有沒有被切變，必須乘回 640×360 像素空間。
    """
    return [[x * MASK_W, y * MASK_H] for x, y in poly]


@pytest.mark.parametrize("deg", [0.0, 12.5, 30.0, 45.0, 77.0, -25.0])
def test_rotating_boundary_does_not_distort_buildings(deg):
    """★核心迴歸：boundary 轉任意角度，建物在 mask 像素空間裡形狀不變。

    boundary 本身是 16:9（= sim 域長寬比）→ 映到 640×360 是等向縮放，正方形建物
    應維持正方形（各邊等長、四角 90°）。舊實作先用 world AABB 正規化、再由 window
    旋轉，在 deg != 0 時會把它切變成平行四邊形。
    """
    bw, bh = 192.0, 108.0
    cmd = ingest_analyze({
        "action": "analyze",
        "boundary_world": rect_world(bw / 2, bh / 2, bw, bh, deg),
        "buildings_world": [rect_world(60.0, 40.0, 10.0, 10.0, deg)],
        "wind_level": 2, "mode": "fresh",
    })
    poly = to_px(cmd["polygons"][0])
    lens = edge_lengths(poly)
    assert max(lens) / min(lens) == pytest.approx(1.0, abs=1e-6), f"邊長不等：{lens}"
    for ang in corner_angles(poly):
        assert ang == pytest.approx(90.0, abs=1e-4)


@pytest.mark.parametrize("deg", [0.0, 12.5, 30.0, 45.0, 77.0, -25.0])
def test_rotating_boundary_preserves_building_aspect(deg):
    """非正方形建物的長寬比也不隨 boundary 旋轉改變（切變會同時改長寬比）。"""
    bw, bh = 192.0, 108.0
    cmd = ingest_analyze({
        "action": "analyze",
        "boundary_world": rect_world(bw / 2, bh / 2, bw, bh, deg),
        "buildings_world": [rect_world(60.0, 40.0, 16.0, 8.0, deg)],
        "wind_level": 2, "mode": "fresh",
    })
    lens = edge_lengths(to_px(cmd["polygons"][0]))
    long_e, short_e = max(lens), min(lens)
    # world 16:8 = 2.0；boundary 16:9 等向映到 640×360 → 比值不變
    assert long_e / short_e == pytest.approx(2.0, rel=1e-6)


@pytest.mark.parametrize("deg", [0.0, 17.0, 45.0, -60.0])
def test_building_position_is_rotation_invariant(deg):
    """同一棟建物隨 boundary 一起轉，正規化後的位置應與未旋轉時相同。"""
    bw, bh = 192.0, 108.0
    cmd = ingest_analyze({
        "action": "analyze",
        "boundary_world": rect_world(bw / 2, bh / 2, bw, bh, deg),
        "buildings_world": [rect_world(48.0, 27.0, 12.0, 8.0, deg)],
        "wind_level": 2, "mode": "fresh",
    })
    got = cmd["polygons"][0]
    base = ingest_analyze({
        "action": "analyze",
        "boundary_world": rect_world(bw / 2, bh / 2, bw, bh, 0.0),
        "buildings_world": [rect_world(48.0, 27.0, 12.0, 8.0, 0.0)],
        "wind_level": 2, "mode": "fresh",
    })["polygons"][0]
    # 先四捨五入再排序：頂點座標相同、但浮點尾差會讓 x 相等時的排序 tie-break 反覆，
    # 直接比對未經穩定化的排序結果會偽陽性。
    def key(poly):
        return sorted((round(x, 9), round(y, 9)) for x, y in poly)

    for (gx, gy), (bx, by) in zip(key(got), key(base)):
        assert gx == pytest.approx(bx, abs=1e-6)
        assert gy == pytest.approx(by, abs=1e-6)


def test_boundary_fills_unit_square():
    """boundary 自身正規化後應恰好填滿 [0,1]²（長邊 → x）。"""
    corners = rect_world(96, 54, 192, 108, 40.0)
    cmd = ingest_analyze({
        "action": "analyze",
        "boundary_world": corners,
        "buildings_world": [corners],          # 拿 boundary 自己當建物
        "wind_level": 2, "mode": "fresh",
    })
    poly = cmd["polygons"][0]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert min(xs) == pytest.approx(0.0, abs=1e-9)
    assert max(xs) == pytest.approx(1.0, abs=1e-9)
    assert min(ys) == pytest.approx(0.0, abs=1e-9)
    assert max(ys) == pytest.approx(1.0, abs=1e-9)


def test_degenerate_polygons_dropped():
    """頂點少於 3 的多邊形被丟棄，不進 polygons。"""
    cmd = ingest_analyze({
        "action": "analyze",
        "boundary_world": rect_world(96, 54, 192, 108, 0.0),
        "buildings_world": [[[10.0, 10.0], [20.0, 10.0]],           # 只有 2 點
                            rect_world(60.0, 40.0, 8.0, 8.0, 0.0)],
        "wind_level": 2, "mode": "fresh",
    })
    assert len(cmd["polygons"]) == 1


def test_empty_boundary_does_not_divide_by_zero():
    cmd = ingest_analyze({"action": "analyze", "boundary_world": [],
                          "buildings_world": []})
    assert cmd["wind_angle"] == 0.0
    assert cmd["polygons"] == []
