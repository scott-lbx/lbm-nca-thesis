"""make_mock_geo — 生成符合 1M dataset 參數空間的假 GeometryInputs（standalone 測試用）。

座標系：1920×1080 sim px（與 dataset 生成完全一致）。
mask 解析度：360×640（H×W）= sim / 3。
"""
from __future__ import annotations

import numpy as np
import torch

from lbmnca.nca.inference.geometry_types import GeometryInputs

# dataset config.py 定義的有效放置範圍（sim px）
_CX_LO, _CX_HI = 262, 1018
_CY_LO, _CY_HI = 162, 918
_W_MIN, _W_MAX = 25, 220
_H_MIN, _H_MAX = 25, 220
_SIM_SCALE = 3  # 1920×1080 → 640×360

# G_vec stats 均值（無 morphospace 時的 fallback）
_FALLBACK_G_VEC = np.array([
    2.5218, 5.1209, 10.7372, 0.02516, 0.7308, 29.4326, 2.0427,
    0.0, 0.02456, 0.003270, 0.4499, 5.8943, 36.4049, 1.0,
], dtype=np.float32)

# Re levels（對齊 C-series config）
_RE_LEVELS = {1: 150, 2: 300, 3: 500, 4: 750, 5: 1000}


def make_mock_geo(
    n_buildings: int = 3,
    wind_level: int = 2,
    seed: int = 42,
    device: str = "cuda",
) -> GeometryInputs:
    """生成 N 棟隨機矩形建物的 GeometryInputs（mask 解析度 360×640）。

    Args:
        n_buildings: 建物棟數（3–20）。
        wind_level: 風力等級 1–5。
        seed: random seed。
        device: "cuda" 或 "cpu"。

    Returns:
        GeometryInputs with mask [360,640], static_raw [4,360,640].
    """
    rng = np.random.default_rng(seed)
    H, W = 360, 640

    mask_np = np.ones((H, W), dtype=np.float32)  # 全 fluid

    for _ in range(n_buildings):
        cx_sim = int(rng.integers(_CX_LO, _CX_HI + 1))
        cy_sim = int(rng.integers(_CY_LO, _CY_HI + 1))
        bw_sim = int(rng.integers(_W_MIN, _W_MAX + 1))
        bh_sim = int(rng.integers(_H_MIN, _H_MAX + 1))
        # 轉換到 mask 座標（/3）
        cx, cy = cx_sim // _SIM_SCALE, cy_sim // _SIM_SCALE
        bw, bh = max(1, bw_sim // _SIM_SCALE), max(1, bh_sim // _SIM_SCALE)
        r0 = max(0, cy - bh // 2)
        r1 = min(H, cy + bh // 2 + 1)
        c0 = max(0, cx - bw // 2)
        c1 = min(W, cx + bw // 2 + 1)
        mask_np[r0:r1, c0:c1] = 0.0  # solid

    static_raw = _build_static_raw(mask_np)

    re = float(_RE_LEVELS.get(wind_level, 300))
    u_inlet = 0.05
    l_char = 64.0
    cond_raw = {
        "wind_level": wind_level,
        "Re": re,
        "U_inlet": u_inlet,
        "u_inlet": u_inlet,
        "L_char": l_char,
        "g_vec": _FALLBACK_G_VEC.tolist(),
        "p_vec": [re, u_inlet, l_char],
        "BCR": float((mask_np < 0.5).sum()) / float(H * W),
        "wind_angle": 0.0,
        "warnings": [],
    }

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    return GeometryInputs(
        mask=torch.from_numpy(mask_np).to(dev),
        static_raw=torch.from_numpy(static_raw).to(dev),
        cond_raw=cond_raw,
        H=H,
        W=W,
    )


def _build_static_raw(mask_np: np.ndarray) -> np.ndarray:
    """近似版 static_raw，對 smoke test 足夠（不需匯入前端的 static_builder）。"""
    H, W = mask_np.shape
    fluid = mask_np > 0.5
    solid = ~fluid

    try:
        from scipy.ndimage import distance_transform_edt
        dist_fluid = distance_transform_edt(fluid)
        dist_solid = distance_transform_edt(solid)
        sdf = np.where(fluid, dist_fluid, -dist_solid).astype(np.float32)
    except ImportError:
        # scipy 不可用 → 零 SDF（dummy backend 不依賴 SDF 數值）
        sdf = np.zeros((H, W), dtype=np.float32)

    upstream = np.zeros((H, W), dtype=np.float32)
    inlet_w = max(1, int(W * 0.05))
    upstream[:, :inlet_w] = fluid[:, :inlet_w].astype(np.float32)

    x_norm = np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1))
    y_norm = np.tile(np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None], (1, W))

    return np.stack([sdf, upstream, x_norm, y_norm], axis=0)


__all__ = ["make_mock_geo"]
