"""從 binary mask 構建 TES NCA 需要的 4ch static field。

Static channel 定義（對應 global_stats_unified.json static_norm）：
    C0: SDF_norm  — 到最近固體的有符號距離（流體正，固體負），z-score 正規化
    C1: upstream_norm — 累積上游障礙物密度 cumsum(solid)/width（委派 lbmnca.lbm，
        訓練同源），z-score 正規化
    C2: pos_x_norm — pixel x 座標，z-score 正規化
    C3: pos_y_norm — pixel y 座標，z-score 正規化

mask 慣例：0.0=solid, 1.0=fluid（對應 MaskRasterizer 輸出）

★ C1 語義修正（2026-07-28 W1-K；DEEP_AUDIT B3-T10／B6-02-實證-02）
------------------------------------------------------------------
訓練資料的 C1 由 lbmnca.lbm ``compute_upstream_obstacle_sum``（沿 x 的
``cumsum(solid)/width`` 累積場）產生（``tes_h5_to_zarr`` Pass2）。本檔曾自寫
「左 5% 流體 inlet marker」——兩者**非零支撐完全互斥**，且 marker 仍套
cumsum 分布的 z-score 常數，逐像素 |Δ| 平均 2.83σ、最大 23.89σ。
現改為委派 01 的實作（單一真相），禁止在本 repo 重寫公式。
★注意慣例反轉：01 以 >0.5 為 solid，本 repo mask 是 0=solid。
"""

from __future__ import annotations

import numpy as np

from lbmnca.lbm.lbm_mrt_les.public.static_features import compute_upstream_obstacle_sum


def _upstream_obstacle_sum(mask_np: np.ndarray) -> np.ndarray:
    """C1 委派 lbmnca.lbm（訓練同源）：沿 x 的 cumsum(solid)/width。

    Args:
        mask_np: [H, W]，本 repo 慣例 0=solid、1=fluid。

    Returns:
        float32 [H, W]，值域 [0, 1]。
    """
    # 慣例反轉：lbmnca.lbm 以 >0.5 為 solid；本 repo mask 0=solid、1=fluid。
    solid = (np.asarray(mask_np, dtype=np.float32) <= 0.5).astype(np.float32)
    return compute_upstream_obstacle_sum(solid).astype(np.float32)


def _sdf_px(mask_np: np.ndarray) -> np.ndarray:
    """Signed distance field in px. Tries Taichi JFA first, falls back to scipy EDT."""
    from lbmnca.bridge.geometry.jfa_sdf import (
        compute_sdf_taichi,
    )  # lazy import avoids ti.init() order issue

    sdf = compute_sdf_taichi(mask_np)
    if sdf is not None:
        return sdf
    from scipy.ndimage import distance_transform_edt

    fluid = mask_np > 0.5
    solid = ~fluid
    return np.where(
        fluid, distance_transform_edt(fluid), -distance_transform_edt(solid)
    ).astype(np.float32)


def build_static_raw(mask_np: np.ndarray) -> np.ndarray:
    """Build UN-normalized static 4ch for the model-agnostic backend path.

    C0: signed distance in **pixels**（fluid >0, solid <0）— backends derive
        fluid mask from >0；TES v3 慣例為 per-case tanh(SDF_px / L_char_h5px)
        （01 tes_h5_to_zarr Pass2），由 adapter 端套用，故此處保留 px 單位。
    C1: upstream_obstacle_sum = cumsum(solid)/width（委派 lbmnca.lbm，訓練同源）
    C2: X_norm in [0,1]
    C3: Y_norm in [0,1]

    Args:
        mask_np: [H, W] float32, 0=solid, 1=fluid.

    Returns:
        [4, H, W] float32 raw static. Adapters normalize as their model needs.
    """
    H, W = mask_np.shape
    sdf = _sdf_px(mask_np)  # fluid > 0, solid < 0, px 單位

    upstream = _upstream_obstacle_sum(mask_np)

    x_norm = np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1))
    y_norm = np.tile(np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None], (1, W))

    return np.stack([sdf, upstream, x_norm, y_norm], axis=0)


def build_static_from_mask(
    mask_np: np.ndarray,
    static_norm_params: dict,
) -> np.ndarray:
    """從 binary mask 構建正規化 static 4ch。

    Args:
        mask_np: [H, W] float32，0=solid, 1=fluid
        static_norm_params: load_static_norm_params() 的輸出，
            格式為 {"sdf": (mean, std), "upstream": ..., "pos_x": ..., "pos_y": ...}

    Returns:
        [4, H, W] float32，已 z-score 正規化
    """
    H, W = mask_np.shape

    # C0: SDF（有符號距離）fluid > 0, solid < 0, px 單位
    sdf_raw = _sdf_px(mask_np)

    # C1: 累積上游障礙物密度（委派 lbmnca.lbm，訓練同源）
    upstream_raw = _upstream_obstacle_sum(mask_np)

    # C2: pos_x（pixel x 座標）
    pos_x_raw = np.tile(np.arange(W, dtype=np.float32), (H, 1))

    # C3: pos_y（pixel y 座標）
    pos_y_raw = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))

    # z-score 正規化
    def znorm(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
        return ((arr - mean) / (std + 1e-8)).astype(np.float32)

    sdf_m, sdf_s = static_norm_params["sdf"]
    up_m, up_s = static_norm_params["upstream"]
    px_m, px_s = static_norm_params["pos_x"]
    py_m, py_s = static_norm_params["pos_y"]

    return np.stack(
        [
            znorm(sdf_raw, sdf_m, sdf_s),
            znorm(upstream_raw, up_m, up_s),
            znorm(pos_x_raw, px_m, px_s),
            znorm(pos_y_raw, py_m, py_s),
        ],
        axis=0,
    )
