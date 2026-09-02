"""regime_features — compute the model-agnostic cond_raw superset from geometry.

03 produces RAW regime stats; each adapter takes what it needs:
  - GC-NCA: 7D regime [Re, U_inlet, L_char, BCR, AR, perimeter_density, U_peak_ratio]
  - TES: g_vec(14D) + p_vec(3D) + ctu (g_vec via gvec_bridge)

See docs/MODEL_BOUNDARY_CONTRACT.md §4 and PLAN_02 §4.

L_char 慣例（★2026-07-26 起委派至單一真相 ``physics_proxy.lchar_api``）：
    l_char = 迎風橫向（cross-flow, y-extent）投影寬度，全體建物取最大
單位 = 模擬解析度 1920×1080 的 mask px；TES 推論域 640×360 為其 1/3 縮尺，
故本模組在 360×640 mask 上量測後 ×SIM_SCALE(=3) 還原至 p_vec 慣例。
Re 即定義在此 l_char 上（nu = U_inlet × l_char / Re），給錯 → cond 正規化後 OOD。

⚠ 本檔先前使用舊公式 ``max(min(w, h))``（未同步 2026-06-29 訓練端修正），
  對真實 case 系統性低估 L_char，見 docs/reviews/DEEP_AUDIT_2026-07/fixes/FIX_B_Lchar統一.md。
"""

from __future__ import annotations

import numpy as np

from lbmnca.morphospace.physics_proxy.lchar_api import l_char_from_mask

# 模擬 mask（1920×1080）→ TES 推論域（640×360）縮尺
SIM_SCALE: float = 3.0
# C-series 訓練分布守門（sim px；超出 → cond_raw["warnings"]，不擋推論）
L_CHAR_TRAIN_RANGE: tuple[float, float] = (32.0, 128.0)
BCR_TRAIN_RANGE: tuple[float, float] = (0.01, 0.30)
L_CHAR_DEFAULT: float = 64.0  # 無固體時的 fallback（訓練分布眾數附近）


def compute_cond_raw(
    mask_np: np.ndarray,
    wind_level: int,
    wind_angle: float,
    config: dict,
    g_vec: list | None = None,
) -> dict:
    """Build cond_raw dict from a fluid mask + wind controls.

    Args:
        mask_np: [H, W] float32, 0=solid, 1=fluid.
        wind_level: 1|2|3 -> Re via config["physics"]["re_levels"].
        wind_angle: degrees (already extracted from boundary long edge).
        config: app config (re_levels, u_inlet_lu).
        g_vec: optional 14D geometry vector (for TES); None if unavailable.

    Returns:
        cond_raw superset dict.
    """
    H, W = mask_np.shape
    fluid = mask_np > 0.5
    solid = ~fluid

    n_total = float(H * W)
    bcr = float(solid.sum()) / n_total  # blockage ratio

    # bbox of solids -> aspect ratio (fallback to domain aspect)
    ar = float(W) / float(H)
    if solid.any():
        ys, xs = np.where(solid)
        bw = max(1, xs.max() - xs.min() + 1)
        bh = max(1, ys.max() - ys.min() + 1)
        ar = float(bw) / float(bh)

    # perimeter density: solid boundary pixels / domain area
    perim = _perimeter_pixels(solid)
    perimeter_density = float(perim) / n_total

    re_levels = config["physics"]["re_levels"]
    re = float(
        re_levels[wind_level] if wind_level in re_levels else re_levels.get(2, 300)
    )
    # ★ 禁 fallback（2026-07-28，DEEP_AUDIT_2026-07 Stage 2.9）：
    # 原為 `.get("u_inlet_lu", 0.06)`，而訓練契約的固定值是 **0.05**
    # （814/814 生產 config、670/670 canonical node 皆然）。該 key 一旦被刪或
    # 改名，fallback 就靜默供給 0.06 —— 因 U_inlet 是零變異維（p_std ~= 1.3e-7），
    # 正規化後 cond 分量會變成 +7.4e4，實測使 rollout 場崩壞（rel-L2 5448%）。
    # 缺鍵一律直接 KeyError，不得以預設值代替。
    u_inlet = float(config["physics"]["u_inlet_lu"])
    l_char = compute_l_char(mask_np)  # sim px（01 C-series 慣例）
    u_peak_ratio = 1.0 + bcr  # crude proxy (blockage -> accel)

    # 訓練分布守門（軟警示，不擋推論——論文「誠實揭露」論述）
    warnings: list[str] = []
    if not (L_CHAR_TRAIN_RANGE[0] <= l_char <= L_CHAR_TRAIN_RANGE[1]):
        warnings.append(
            f"L_char={l_char:.0f}px 超出訓練分布 {L_CHAR_TRAIN_RANGE}（建物過大/過小）"
        )
    if not (BCR_TRAIN_RANGE[0] <= bcr <= BCR_TRAIN_RANGE[1]):
        warnings.append(f"BCR={bcr:.3f} 超出訓練分布 {BCR_TRAIN_RANGE}")

    return {
        "wind_angle": float(wind_angle),
        "wind_level": int(wind_level),
        "Re": re,
        "U_inlet": u_inlet,
        "u_inlet": u_inlet,  # dummy adapter alias
        "L_char": l_char,
        "BCR": bcr,
        "AR": ar,
        "perimeter_density": perimeter_density,
        "U_peak_ratio": u_peak_ratio,
        "g_vec": list(g_vec) if g_vec is not None else None,
        "p_vec": [re, u_inlet, l_char],
        "delta_ctu": None,
        "cumulative_ctu": None,
        "warnings": warnings,
    }


def compute_l_char(mask_np: np.ndarray) -> float:
    """L_char（sim px）：委派至 ``physics_proxy.lchar_api``（L_char 單一真相）。

    ★ 2026-07-26 行為修正（DEEP_AUDIT_2026-07 主題 2）。
    本函式原本用 ``max(min(w, h))``，未隨 2026-06-29 訓練端改為 y-extent 而同步，
    導致**部署時餵給模型的 L_char 與訓練契約不一致**（實測系統性低估 ~26.5%），
    且 L_char 同時決定 p_vec[2] 的 Re 條件與 C0 SDF 的 tanh 尺度 → 雙重偏移。

    修正理由是**一致性**：模型是以訓練端定義學的，部署端必須用同一個定義，
    與哪個公式物理上更合理無關。

    Args:
        mask_np: [H, W] float32, 0=solid, 1=fluid（360×640 推論域）。
            軸序為 numpy 影像慣例（axis0=y 橫流向、axis1=x 流向）→ layout="yx"。

    Returns:
        l_char（sim px）；無固體 → L_CHAR_DEFAULT。
    """
    solid = mask_np < 0.5
    if not solid.any():
        return L_CHAR_DEFAULT
    l_char_mask_px = l_char_from_mask(solid, layout="yx")
    return float(l_char_mask_px) * SIM_SCALE


def _perimeter_pixels(solid: np.ndarray) -> int:
    """Count solid pixels adjacent to a fluid pixel (4-neighbourhood)."""
    if not solid.any():
        return 0
    pad = np.pad(solid, 1, mode="edge")
    fluid_neigh = ~pad[:-2, 1:-1] | ~pad[2:, 1:-1] | ~pad[1:-1, :-2] | ~pad[1:-1, 2:]
    return int((solid & fluid_neigh).sum())
