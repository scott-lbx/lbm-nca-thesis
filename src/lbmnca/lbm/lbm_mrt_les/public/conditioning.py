"""
公開 API: ML 訓練用 conditioning vector 計算。

10D 向量（全部為 lattice 單位或無因次量）：
  [Re, U_inlet, L_char_px, BCR, AR, perimeter_density, U_peak_ratio,
   wt_to_real_world_scaler, open_fraction, nu_lb]

idx 0-6 與舊版 7D 向後兼容。idx 7-9 為 Phase 6 新增。
idx 7 為風洞→真實尺度的無因次縮放比（= m_per_px / dx_model）。

取代以下位置的邏輯：
  - src/post_process/h5_to_zarr_pipeline.py (compute_conditioning)
"""

from typing import Any, Dict

import numpy as np

COND_DIM = 10


def _safe_float(v: Any) -> float:
    """將任意值轉為 float，失敗時回傳 NaN。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def compute_perimeter_density(binary_mask_hw: np.ndarray) -> float:
    """計算建築邊界密度（邊界格點數 / 總格點數）。

    Args:
        binary_mask_hw: (H, W) 遮罩，> 0.5 = 固體。

    Returns:
        邊界密度，範圍 [0, 1]。
    """
    solid = (np.asarray(binary_mask_hw, dtype=np.float32) > 0.5).astype(np.float32)
    h, w = solid.shape
    fluid_neighbors = (
        (1.0 - np.roll(solid, 1, axis=1))
        + (1.0 - np.roll(solid, -1, axis=1))
        + (1.0 - np.roll(solid, 1, axis=0))
        + (1.0 - np.roll(solid, -1, axis=0))
    )
    boundary = solid * (fluid_neighbors > 0)
    return float(boundary.sum()) / float(h * w)


def compute_u_peak_ratio(
    mean_field_9hw: np.ndarray,
    u_inlet_lu: float,
) -> float:
    """計算 U_peak_ratio = 第 99 百分位速度 / 入口速度。

    Args:
        mean_field_9hw: (9, H, W) MRT moment 時均場（原始 lattice 值）。
            通道 0 = rho, 通道 3 = jx, 通道 5 = jy。
        u_inlet_lu: 入口速度（lattice 單位）。

    Returns:
        峰值速度比。
    """
    rho = np.maximum(mean_field_9hw[0].astype(np.float32), 1e-6)
    ux = mean_field_9hw[3].astype(np.float32) / rho
    uy = mean_field_9hw[5].astype(np.float32) / rho
    u_mag = np.sqrt(ux**2 + uy**2)
    return float(np.percentile(u_mag, 99)) / max(u_inlet_lu, 1e-8)


def build_training_conditioning(
    case_meta: Dict[str, Any],
    binary_mask_hw: np.ndarray,
    mean_field_9hw: np.ndarray,
) -> np.ndarray:
    """從 case metadata 與場量建構 10D conditioning vector。

    向量內容（全部使用 lattice 單位或無因次量）：
      [0] Re                      -- runtime 量測的 Reynolds number
      [1] U_inlet                  -- 入口速度（lattice unit）
      [2] L_char_px                -- 特徵長度（像素）
      [3] BCR                      -- 建蔽率
      [4] AR                       -- domain 長寬比
      [5] perimeter_density        -- 建築邊界複雜度
      [6] U_peak_ratio             -- 99th 百分位速度 / 入口速度
      [7] wt_to_real_world_scaler  -- 風洞→真實尺度縮放比（無因次）
      [8] open_fraction            -- 1 - max_blockage（入口開放比）
      [9] nu_lb                    -- lattice viscosity

    Zarr 存放原始值；ML DataLoader 在載入時套用 z-score 正規化。
    向後兼容：idx 0-6 與舊 7D 格式完全相同。

    Args:
        case_meta: sim_results.json 中的 case metadata 字典。
        binary_mask_hw: (H, W) 二值遮罩。
        mean_field_9hw: (9, H, W) 時間平均 MRT moments（原始 lattice 值）。

    Returns:
        float32 陣列，形狀 (10,)。
    """
    params = case_meta.get("parameters", {})
    sim_outs = params.get("simulation_outputs", {})
    lat = params.get("lattice_inputs", {})
    # manifest 使用 geometry_features；舊 sim_results 使用 geometry_metadata
    geom = case_meta.get("geometry_features") or case_meta.get("geometry_metadata", {})
    mask_meta = case_meta.get("mask_metadata", {})

    # --- idx 0: Re ---
    # Phase 5: 優先使用 re_reference_for_dataset（runtime 量測的 inlet plane mean Re）
    re_family = case_meta.get("reynolds_family", {})
    re_num = _safe_float(re_family.get("re_reference_for_dataset"))
    if np.isnan(re_num):
        re_num = _safe_float(sim_outs.get("actual_reynolds_number"))
    if np.isnan(re_num):
        re_num = _safe_float(sim_outs.get("estimated_reynolds_number", 0.0))

    # --- idx 1: U_inlet（lattice 單位）---
    u_inlet_lu = _safe_float(lat.get("inlet_velocity_lu", 0.0))

    # --- idx 2: L_char（像素）---
    l_char_px = _safe_float(lat.get("characteristic_length_px"))
    if np.isnan(l_char_px):
        l_char_px = _safe_float(geom.get("characteristic_length_px", 0.0))

    # --- idx 3: BCR ---
    bcr = float(np.mean((binary_mask_hw > 0.5).astype(np.float32)))

    # --- idx 4: AR ---
    h, w = binary_mask_hw.shape
    ar = float(w) / float(h) if h > 0 else 1.0

    # --- idx 5: perimeter_density ---
    peri = compute_perimeter_density(binary_mask_hw)

    # --- idx 6: U_peak_ratio ---
    u_peak = compute_u_peak_ratio(mean_field_9hw, u_inlet_lu)

    # --- idx 7: wt_to_real_world_scaler（無因次）---
    # 優先從 manifest 直接讀取；fallback 從 m_per_px / dx_model 計算
    wt_scaler = _safe_float(params.get("wt_to_real_world_scaler"))
    if np.isnan(wt_scaler):
        m_per_px = _safe_float(mask_meta.get("m_per_px", 0.0))
        wt_cell_size = _safe_float(
            params.get("wind_tunnel_model_scale", {}).get("cell_size_m", 0.0)
        )
        if m_per_px > 0 and wt_cell_size > 0:
            wt_scaler = m_per_px / wt_cell_size
        else:
            wt_scaler = 0.0

    # --- idx 8: open_fraction（新增）---
    open_fraction = _safe_float(geom.get("open_fraction"))
    if np.isnan(open_fraction):
        max_blockage = _safe_float(geom.get("max_blockage", 0.0))
        open_fraction = 1.0 - (max_blockage if not np.isnan(max_blockage) else 0.0)

    # --- idx 9: nu_lb（新增）---
    # manifest 使用 kinematic_viscosity_lu；部分舊格式使用 nu_lb 或 nu
    nu_lb = _safe_float(lat.get("nu_lb"))
    if np.isnan(nu_lb):
        nu_lb = _safe_float(lat.get("kinematic_viscosity_lu"))
    if np.isnan(nu_lb):
        nu_lb = _safe_float(params.get("nu", 0.0))

    return np.array(
        [re_num, u_inlet_lu, l_char_px, bcr, ar, peri, u_peak,
         wt_scaler, open_fraction, nu_lb],
        dtype=np.float32,
    )


# --- Conditioning 欄位名稱（供 Zarr attrs + global_stats 使用）---
CONDITIONING_LABELS = [
    "Re",
    "U_inlet_lu",
    "L_char_px",
    "BCR",
    "AR",
    "perimeter_density",
    "U_peak_ratio",
    "wt_to_real_world_scaler",
    "open_fraction",
    "nu_lb",
]
