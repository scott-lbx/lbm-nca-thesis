"""Taichi kernel（CPU arch）：mask 光柵化與 MRT → 速度/壓力推導。

★一律 arch=ti.cpu。訓練常常同時在跑，搶 VRAM 會把它 OOM 掉。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

H, W = 360, 640


@pytest.fixture(scope="module")
def gpu_fields(ti_cpu):
    from lbmnca.bridge.render.gpu_fields import GPUFields

    return GPUFields(H, W)


@pytest.fixture(scope="module")
def rasterizer(ti_cpu):
    from lbmnca.bridge.geometry.mask_rasterizer import MaskRasterizer

    return MaskRasterizer(H, W)


def polys_to_edges(polygons):
    from lbmnca.bridge.ggui.window import _polygons_to_edges

    return _polygons_to_edges(polygons)


def rasterize(rasterizer, polygons):
    edges, n = polys_to_edges(polygons)
    rasterizer.rasterize(edges, n)
    return rasterizer.get_mask_numpy()


def norm_rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# ── MaskRasterizer ───────────────────────────────────────────────────────────

def test_empty_scene_is_all_fluid(rasterizer):
    m = rasterize(rasterizer, [])
    assert m.shape == (H, W)
    assert m.min() == 1.0


def test_rect_area_matches_expected_fraction(rasterizer):
    """winding-number 光柵化的固體面積應等於矩形佔比（±1 px 邊界誤差）。"""
    m = rasterize(rasterizer, [norm_rect(0.25, 0.25, 0.75, 0.5)])
    solid = float((m < 0.5).sum())
    expected = 0.5 * 0.25 * H * W
    assert solid == pytest.approx(expected, rel=0.02)


def test_rect_is_where_it_was_drawn(rasterizer):
    """x∈[0.25,0.75]→列、y∈[0.25,0.5]→行（geometry_ingest 的軸序約定）。"""
    m = rasterize(rasterizer, [norm_rect(0.25, 0.25, 0.75, 0.5)])
    assert m[int(0.375 * H), int(0.5 * W)] == 0.0     # 框內
    assert m[int(0.8 * H), int(0.5 * W)] == 1.0       # 框下方
    assert m[int(0.375 * H), int(0.05 * W)] == 1.0    # 框左方


def test_two_disjoint_rects_both_present(rasterizer):
    m = rasterize(rasterizer, [norm_rect(0.1, 0.1, 0.2, 0.2),
                               norm_rect(0.7, 0.6, 0.85, 0.8)])
    assert m[int(0.15 * H), int(0.15 * W)] == 0.0
    assert m[int(0.7 * H), int(0.78 * W)] == 0.0


def test_rasterize_is_idempotent(rasterizer):
    """重複光柵化同一幾何結果相同（field 有被清乾淨，不累積）。"""
    p = [norm_rect(0.3, 0.3, 0.6, 0.55)]
    a = rasterize(rasterizer, p).copy()
    b = rasterize(rasterizer, p).copy()
    assert np.array_equal(a, b)


def test_previous_geometry_cleared(rasterizer):
    """換成空場景後不得殘留上一次的固體。"""
    rasterize(rasterizer, [norm_rect(0.3, 0.3, 0.6, 0.55)])
    m = rasterize(rasterizer, [])
    assert m.min() == 1.0


def test_rotated_polygon_area_preserved(rasterizer):
    """旋轉 45° 的正方形面積不變（光柵化不偏袒軸對齊）。"""
    cx, cy, r = 0.5, 0.5, 0.12
    diamond = [[cx + r, cy], [cx, cy + r * W / H], [cx - r, cy],
               [cx, cy - r * W / H]]
    m = rasterize(rasterizer, [diamond])
    solid = float((m < 0.5).sum())
    d1 = 2 * r * W                    # 水平對角線（px）
    d2 = 2 * r * (W / H) * H          # 垂直對角線（px）；r 已按 W/H 補償過長寬比
    assert solid == pytest.approx(d1 * d2 / 2, rel=0.05)


def test_polygons_to_edges_caps_at_max_edges():
    """邊數上限保護：超量不得越界寫入。"""
    many = [norm_rect(0.0, 0.0, 0.01, 0.01)] * 800     # 3200 邊 > MAX_EDGES 2000
    edges, n = polys_to_edges(many)
    assert n <= edges.shape[0]


# ── field_ops.derive_fields（MRT 矩量 → 速度/壓力）────────────────────────────

def uniform_mrt(rho=1.0, jx=0.05, jy=0.0):
    """[9,H,W] 均勻 MRT 矩量場（denorm/raw 空間）。"""
    f = np.zeros((9, H, W), dtype=np.float32)
    f[0] = rho
    f[3] = jx
    f[5] = jy
    return f


def derive(gpu_fields, field_np):
    import taichi as ti
    import torch

    from lbmnca.bridge.render.field_ops import derive_fields

    gpu_fields.physics.from_torch(torch.from_numpy(field_np).contiguous())
    derive_fields(gpu_fields.physics, gpu_fields.vel_x, gpu_fields.vel_y,
                  gpu_fields.vel_mag, gpu_fields.pressure)
    ti.sync()
    return (gpu_fields.vel_x.to_numpy(), gpu_fields.vel_y.to_numpy(),
            gpu_fields.vel_mag.to_numpy(), gpu_fields.pressure.to_numpy())


def test_velocity_is_momentum_over_density(gpu_fields):
    vx, vy, vmag, _ = derive(gpu_fields, uniform_mrt(rho=2.0, jx=0.1, jy=0.0))
    assert vx.mean() == pytest.approx(0.05, abs=1e-6)      # j/rho
    assert abs(vy).max() == pytest.approx(0.0, abs=1e-6)
    assert vmag.mean() == pytest.approx(0.05, abs=1e-6)


def test_velocity_magnitude_is_euclidean(gpu_fields):
    vx, vy, vmag, _ = derive(gpu_fields, uniform_mrt(rho=1.0, jx=0.03, jy=0.04))
    assert vmag.mean() == pytest.approx(math.hypot(0.03, 0.04), abs=1e-6)


def test_pressure_is_rho_over_three(gpu_fields):
    """LBM 等溫狀態方程 p = rho·cs² = rho/3。"""
    _, _, _, p = derive(gpu_fields, uniform_mrt(rho=1.0))
    assert p.mean() == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_derive_fields_is_finite_on_zero_density(gpu_fields):
    """rho=0（固體區的退化情形）不得產生 NaN/Inf。"""
    f = uniform_mrt(rho=0.0, jx=0.0, jy=0.0)
    for a in derive(gpu_fields, f):
        assert np.isfinite(a).all()


def test_vorticity_of_shear_flow_has_expected_sign(gpu_fields):
    """u_x = k·y 的剪切流 → ω = ∂v/∂x − ∂u/∂y = −k（常數負值）。"""
    import taichi as ti

    from lbmnca.bridge.render.field_ops import derive_fields, derive_vorticity

    import torch

    k = 1e-4
    f = np.zeros((9, H, W), dtype=np.float32)
    f[0] = 1.0
    f[3] = (k * np.arange(H, dtype=np.float32))[:, None]     # jx 隨 y 線性增加
    gpu_fields.physics.from_torch(torch.from_numpy(f).contiguous())
    derive_fields(gpu_fields.physics, gpu_fields.vel_x, gpu_fields.vel_y,
                  gpu_fields.vel_mag, gpu_fields.pressure)
    derive_vorticity(gpu_fields.vel_x, gpu_fields.vel_y, gpu_fields.vorticity, H, W)
    ti.sync()
    w = gpu_fields.vorticity.to_numpy()[2:-2, 2:-2]
    assert w.mean() < 0
    assert np.isfinite(w).all()
