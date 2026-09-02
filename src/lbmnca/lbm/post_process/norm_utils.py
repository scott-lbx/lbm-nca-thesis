"""
MRT 9 通道正規化工具：raw lattice → baseline-deviation → tanh rescaled。

三階段正規化：
  Stage 1 (baseline-deviation): ñ_i   = (m_i - M_REF_i) / S_i
  Stage 2 (group rescale):      ñ̃_i   = ñ_i / A_i         where A_i = group-shared max(P99.5(|ñ|))
  Stage 3 (soft-clip):          out_i  = tanh(ñ̃_i)         → 值域 (-1, 1)，極值 soft 壓縮

  M_REF = 靜止參考態 [ρ₀, -2ρ₀, ρ₀, 0, 0, 0, 0, 0, 0]
  S     = Chapman-Enskog velocity-scale 因子（由 U_ref 決定）
  A     = 群組共享 P99.5(|ñ|)，同群組取 max → 保留群組內 +/- 比例

群組設計（NORM_GROUPS）：
  momentum [3,4,5,6]：jx/jy/flux 共享 A → 各向同性，流速方向保留
  stress   [7,8]    ：pxx/pxy 共享 A → 應力張量方向保留

正反轉換：
  forward: out   = tanh((raw - M_REF) / S_combined)    S_combined = S * A
  inverse: raw   = atanh(out) * S_combined + M_REF     atanh 需 clip(-0.9999, 0.9999)

理論依據：doc/doc3_physics_informed_normalization_theory.md
"""

from pathlib import Path
from typing import List

import h5py
import numpy as np
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# 物理常數
# ══════════════════════════════════════════════════════════════════════════════

RHO_0: float = 1.0
CS_SQ: float = 1.0 / 3.0
U_REF_SAFETY_FACTOR: float = 1.2
NUM_CHANNELS: int = 9

# MRT moment 通道索引（D2Q9: [rho, e, eps, jx, qx, jy, qy, pxx, pxy]）
CH_JX: int = 3  # x-momentum（ρ≈1 → jx ≈ ux）
CH_JY: int = 5  # y-momentum（ρ≈1 → jy ≈ uy）

# 靜止參考態 m_i^ref（D2Q9 MRT, ρ₀=1, u=v=0）
M_REF: np.ndarray = np.array(
    [RHO_0, -2.0 * RHO_0, RHO_0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)

# Rescale 預設參數
RESCALE_PERCENTILE: float = 99.5
RESCALE_SAMPLES_PER_CASE: int = 50_000

# 通道群組：同群組共享 rescale A，保留群組內通道的 +/- 量值比例
# MRT D2Q9 channel order: [rho, e, eps, jx, qx, jy, qy, pxx, pxy]
NORM_GROUPS: dict[str, list[int]] = {
    "density":  [0],           # ρ：獨立，密度波動獨立量級
    "energy":   [1, 2],        # e, ε：能量模，共享
    "momentum": [3, 4, 5, 6],  # jx, flux_x, jy, flux_y：各向同性共享 → 流速方向保留
    "stress":   [7, 8],        # pxx, pxy：應力張量共享 → 張量方向保留
}

# tanh 逆變換用 clip 邊界（防止 atanh(±1) = ±inf）
TANH_INV_CLIP: float = 0.9999


# ══════════════════════════════════════════════════════════════════════════════
# Velocity-scale 因子
# ══════════════════════════════════════════════════════════════════════════════


def compute_velocity_scales(u_ref: float, rho_0: float = RHO_0) -> np.ndarray:
    """從特徵速度 U_ref 計算 9 通道 velocity-scale 縮放因子 S_i。

    Chapman-Enskog 量級分析：
      S_0 = ρ₀ Ma²           (density  ~ Ma²)
      S_1 = S_2 = 3ρ₀ U²     (energy   ~ 3U²)
      S_3..S_6 = ρ₀ U         (momentum ~ U)
      S_7 = S_8 = ρ₀ U²       (stress   ~ U²)

    Args:
        u_ref: 全域特徵速度（lattice 單位）。
        rho_0: 參考密度。

    Returns:
        S 陣列 (9,)，dtype=float32。
    """
    ma_sq = 3.0 * u_ref ** 2
    return np.array(
        [
            rho_0 * ma_sq,            # m0: density
            3.0 * rho_0 * u_ref ** 2, # m1: energy
            3.0 * rho_0 * u_ref ** 2, # m2: energy²
            rho_0 * u_ref,            # m3: x-momentum
            rho_0 * u_ref,            # m4: x-energy flux
            rho_0 * u_ref,            # m5: y-momentum
            rho_0 * u_ref,            # m6: y-energy flux
            rho_0 * u_ref ** 2,       # m7: normal stress
            rho_0 * u_ref ** 2,       # m8: shear stress
        ],
        dtype=np.float32,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 群組共享 A 工具
# ══════════════════════════════════════════════════════════════════════════════


def apply_group_max_rescale(
    rescale_abs_max: np.ndarray,
    groups: dict[str, list[int]] | None = None,
) -> np.ndarray:
    """同群組通道取 max(A_i)，保留群組內 +/- 量值比例。

    例：momentum 群組 [3,4,5,6]：jx/jy 共享同一 A，
    流速方向 atan2(jy, jx) 在正規化後保持不變。

    Args:
        rescale_abs_max: per-channel P99.5(|ñ|)，shape (9,)。
        groups: 群組定義，None 時使用 NORM_GROUPS。

    Returns:
        group-shared rescale_abs_max，shape (9,)。
    """
    if groups is None:
        groups = NORM_GROUPS
    result = rescale_abs_max.copy().astype(np.float32)
    for name, indices in groups.items():
        group_max = float(max(result[i] for i in indices))
        for i in indices:
            result[i] = group_max
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 正規化轉換（單一陣列）
# ══════════════════════════════════════════════════════════════════════════════


def _broadcast(vec: np.ndarray, ndim: int) -> np.ndarray:
    """將 (9,) 向量 reshape 為可廣播形狀。

    ndim=3 → (9,1,1)；ndim=4 → (1,9,1,1)。
    """
    if ndim == 3:
        return vec.reshape(NUM_CHANNELS, 1, 1)
    if ndim == 4:
        return vec.reshape(1, NUM_CHANNELS, 1, 1)
    raise ValueError(f"不支援的 ndim={ndim}，預期 3 或 4")


# ── Stage 1: baseline-deviation ──


def baseline_deviation_forward(
    raw: np.ndarray,
    m_ref: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """raw MRT → baseline-deviation normalized。

    ñ = (m - M_REF) / S

    Args:
        raw: MRT moments (9,H,W) 或 (T,9,H,W)。
        m_ref: 靜止參考態 (9,)。
        s: velocity-scale 因子 (9,)。

    Returns:
        normed，與 raw 同形狀，dtype=float32。
    """
    return ((raw - _broadcast(m_ref, raw.ndim)) / _broadcast(s, raw.ndim)).astype(
        np.float32
    )


def baseline_deviation_inverse(
    normed: np.ndarray,
    m_ref: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """baseline-deviation normalized → raw MRT。

    m = ñ × S + M_REF

    Args:
        normed: 正規化後 (9,H,W) 或 (T,9,H,W)。
        m_ref: 靜止參考態 (9,)。
        s: velocity-scale 因子 (9,)。

    Returns:
        raw，與 normed 同形狀，dtype=float32。
    """
    return (normed * _broadcast(s, normed.ndim) + _broadcast(m_ref, normed.ndim)).astype(
        np.float32
    )


# ── Stage 2: rescale to ~(-1, 1) ──


def rescale_forward(
    normed: np.ndarray,
    rescale_abs_max: np.ndarray,
) -> np.ndarray:
    """baseline-deviation normed → rescaled ~(-1, 1)。

    scaled = ñ / A

    Args:
        normed: baseline-deviation 後 (9,H,W) 或 (T,9,H,W)。
        rescale_abs_max: per-channel rescale 因子 (9,)。

    Returns:
        rescaled，99.5% 值落在 (-1, 1)。
    """
    return (normed / _broadcast(rescale_abs_max, normed.ndim)).astype(np.float32)


def rescale_inverse(
    scaled: np.ndarray,
    rescale_abs_max: np.ndarray,
) -> np.ndarray:
    """rescaled → baseline-deviation normed。

    ñ = scaled × A
    """
    return (scaled * _broadcast(rescale_abs_max, scaled.ndim)).astype(np.float32)


# ── Combined 捷徑（raw ↔ tanh-scaled，一步完成）──


def raw_to_scaled(
    raw: np.ndarray,
    m_ref: np.ndarray,
    s: np.ndarray,
    rescale_abs_max: np.ndarray,
) -> np.ndarray:
    """raw MRT → tanh rescaled ∈ (-1, 1)，三階段合併。

    out = tanh((raw - M_REF) / S_combined)    S_combined = S * A（group-shared）

    tanh soft-clip：P99.5 值映射到 tanh(1) ≈ 0.76，保留極端值梯度。
    """
    s_combined = s * rescale_abs_max
    linear = (raw - _broadcast(m_ref, raw.ndim)) / _broadcast(s_combined, raw.ndim)
    return np.tanh(linear).astype(np.float32)


def raw_to_scaled_linear(
    raw: np.ndarray,
    m_ref: np.ndarray,
    s: np.ndarray,
    rescale_abs_max: np.ndarray,
) -> np.ndarray:
    """raw MRT → linear rescaled（不含 tanh），供比較/診斷用。

    scaled = (raw - M_REF) / S_combined
    """
    s_combined = s * rescale_abs_max
    return ((raw - _broadcast(m_ref, raw.ndim)) / _broadcast(s_combined, raw.ndim)).astype(
        np.float32
    )


def scaled_to_raw(
    scaled: np.ndarray,
    m_ref: np.ndarray,
    s: np.ndarray,
    rescale_abs_max: np.ndarray,
) -> np.ndarray:
    """tanh rescaled → raw MRT，三階段反正規化合併。

    m = atanh(clip(out, ±TANH_INV_CLIP)) × S_combined + M_REF

    clip 防止 atanh(±1) = ±inf（zarr float16 量化可能產生邊界值）。
    """
    s_combined = s * rescale_abs_max
    clipped = np.clip(scaled, -TANH_INV_CLIP, TANH_INV_CLIP)
    linear = np.arctanh(clipped).astype(np.float32)
    return (linear * _broadcast(s_combined, linear.ndim) + _broadcast(m_ref, linear.ndim)).astype(
        np.float32
    )


def scaled_to_raw_linear(
    scaled: np.ndarray,
    m_ref: np.ndarray,
    s: np.ndarray,
    rescale_abs_max: np.ndarray,
) -> np.ndarray:
    """linear rescaled → raw MRT（不含 atanh），供舊格式相容/診斷用。

    m = scaled × S_combined + M_REF
    """
    s_combined = s * rescale_abs_max
    return (scaled * _broadcast(s_combined, scaled.ndim) + _broadcast(m_ref, scaled.ndim)).astype(
        np.float32
    )


# ══════════════════════════════════════════════════════════════════════════════
# Dataset-level 統計：計算 rescale 因子
# ══════════════════════════════════════════════════════════════════════════════


def compute_rescale_abs_max(
    h5_paths: List[Path],
    m_ref: np.ndarray,
    s: np.ndarray,
    percentile: float = RESCALE_PERCENTILE,
    samples_per_case: int = RESCALE_SAMPLES_PER_CASE,
) -> np.ndarray:
    """掃描所有 H5 的 mean_moments_field，計算 per-channel P{percentile}(|ñ|)。

    使用隨機取樣降低記憶體用量（每 case 取 samples_per_case 個像素）。

    Args:
        h5_paths: H5 檔案路徑列表。
        m_ref: 靜止參考態 (9,)。
        s: velocity-scale 因子 (9,)。
        percentile: 目標百分位數（預設 99.5）。
        samples_per_case: 每個 case 最大取樣像素數。

    Returns:
        rescale_abs_max: per-channel rescale 因子 (9,)，dtype=float32。
    """
    m_ref_3d = m_ref.reshape(NUM_CHANNELS, 1, 1)
    s_3d = s.reshape(NUM_CHANNELS, 1, 1)

    all_abs: list[list[np.ndarray]] = [[] for _ in range(NUM_CHANNELS)]
    rng = np.random.default_rng(42)

    for h5_path in tqdm(h5_paths, desc=f"[Rescale] 計算 P{percentile}"):
        with h5py.File(h5_path, "r") as f:
            raw = f["mean_moments_field"][:].astype(np.float32)  # (9, H, W)

        normed = (raw - m_ref_3d) / s_3d
        n_pixels = normed.shape[1] * normed.shape[2]
        n_sample = min(samples_per_case, n_pixels)
        idx = rng.choice(n_pixels, n_sample, replace=False)

        for ch in range(NUM_CHANNELS):
            all_abs[ch].append(np.abs(normed[ch].reshape(-1)[idx]))

    rescale_abs_max = np.zeros(NUM_CHANNELS, dtype=np.float32)
    for ch in range(NUM_CHANNELS):
        combined = np.concatenate(all_abs[ch])
        val = float(np.percentile(combined, percentile))
        rescale_abs_max[ch] = max(val, 1e-8)  # 防止除零

    print(f"  rescale_abs_max (P{percentile}): {rescale_abs_max.tolist()}")
    return rescale_abs_max
