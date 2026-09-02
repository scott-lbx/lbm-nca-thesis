"""geometry_ingest — server-side ingest of raw world-coordinate geometry from GH.

GH stays thin: it sends building parameters; all normalization happens here.

支援兩種 payload 格式：

  新格式（dataset 對齊）：
    {action, buildings:[{cx,cy,w,h}], wind_level, mode}
    cx/cy/w/h 為 1920×1080 sim px 座標（與 1M dataset 生成完全一致）
    wind_angle 固定 0°（+X 方向）

  舊格式（legacy）：
    {action, mode, wind_level, boundary_world:[[x,y]*4],
     buildings_world:[[[x,y],...], ...]}

Output is the normalized payload the GGUI command handler expects:
    {polygons: [[ [x,y],... ], ...], wind_angle, wind_level, mode}

★座標契約：``polygons`` 一律是 **canonical 方向**（風沿 +x）的 [0,1] 正規化座標。
world 方言在此就地旋轉 −θ 後才 bbox 正規化，故 GGUI 端不得再轉第二次
（理由見 ``src/geometry/angle_transform.py`` 模組 docstring）。
``wind_angle`` 仍然回傳，供 log／UI 顯示使用者當初畫的方位。
"""
from __future__ import annotations

import math
from typing import Any

from lbmnca.bridge.geometry.angle_transform import AngleTransform

# 1M dataset 生成時的全圖解析度（對應 mask 歸一化基準）
_SIM_W: float = 1920.0
_SIM_H: float = 1080.0


def ingest_analyze(data: dict) -> dict[str, Any]:
    """Convert a raw GH 'analyze' message into a normalized command payload.

    自動偵測 payload 格式：若 buildings 為 dict 列表（含 cx/cy/w/h 鍵）→ 新格式；
    否則走舊格式（boundary_world + buildings_world）。

    Returns:
        {polygons (normalized [0,1]), wind_angle (deg), wind_level, mode}
    """
    buildings_raw = data.get("buildings") or []
    wind_level = int(data.get("wind_level", 2))
    mode = data.get("mode", "fresh")

    if buildings_raw and isinstance(buildings_raw[0], dict) and "cx" in buildings_raw[0]:
        # 新格式：N × {cx, cy, w, h}（1920×1080 sim px）
        polygons = [_rect_to_polygon(b) for b in buildings_raw]
        return {
            "polygons": polygons,
            "wind_angle": 0.0,
            "wind_level": wind_level,
            "mode": mode,
        }

    # 舊格式（legacy）：boundary_world + buildings_world
    corners = data.get("boundary_world") or []
    buildings_world = data.get("buildings_world") or []
    wind_angle = _wind_angle_from_corners(corners)

    # ★先旋轉進 boundary 自身的座標系（長邊 → +x），再取 bbox 正規化。
    #   不可先用 world AABB 正規化再旋轉：旋轉後 boundary 的 AABB 比 boundary 本身大、
    #   長寬比也不同，x/y 被不同倍率壓縮，之後在該非等向空間裡旋轉會把建物切變成
    #   平行四邊形（見 angle_transform 模組 docstring）。
    xf = AngleTransform(wind_angle)
    corners_c = xf.to_canonical(corners)
    min_x, min_y, w, h = _bbox(corners_c)

    def norm(pt):
        return [(pt[0] - min_x) / w, (pt[1] - min_y) / h]

    polygons = [
        [norm(p) for p in xf.to_canonical(poly)]
        for poly in buildings_world
        if len(poly) >= 3
    ]
    return {
        "polygons": polygons,
        "wind_angle": wind_angle,
        "wind_level": wind_level,
        "mode": mode,
    }


def _rect_to_polygon(b: dict) -> list[list[float]]:
    """將 {cx, cy, w, h}（1920×1080 px）轉為 [0,1] 歸一化閉合多邊形。

    mask_rasterizer 使用 winding number（[0,1] 歸一化座標）：
        j / W → x 方向（列）
        i / H → y 方向（行）
    故 sim_x / 1920 → mask 列方向，sim_y / 1080 → mask 行方向。
    """
    cx, cy = float(b["cx"]), float(b["cy"])
    hw, hh = float(b["w"]) / 2.0, float(b["h"]) / 2.0
    x0 = max(0.0, min(1.0, (cx - hw) / _SIM_W))
    x1 = max(0.0, min(1.0, (cx + hw) / _SIM_W))
    y0 = max(0.0, min(1.0, (cy - hh) / _SIM_H))
    y1 = max(0.0, min(1.0, (cy + hh) / _SIM_H))
    # closed polygon（首尾相同）
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def _bbox(corners: list) -> tuple[float, float, float, float]:
    if not corners:
        return 0.0, 0.0, 1.0, 1.0
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    min_x, min_y = min(xs), min(ys)
    w = max(1e-6, max(xs) - min_x)
    h = max(1e-6, max(ys) - min_y)
    return min_x, min_y, w, h


def _wind_angle_from_corners(corners: list) -> float:
    """Wind direction = longest edge of the boundary rectangle (deg, CCW from +X)."""
    if len(corners) < 2:
        return 0.0
    best_len = -1.0
    best_ang = 0.0
    n = len(corners)
    for i in range(n):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        ln = dx * dx + dy * dy
        if ln > best_len:
            best_len = ln
            best_ang = math.degrees(math.atan2(dy, dx))
    return best_ang
