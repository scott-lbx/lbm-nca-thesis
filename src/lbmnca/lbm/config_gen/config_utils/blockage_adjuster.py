"""
config_utils/blockage_adjuster.py

從阻塞率 + 最小建築間距推導安全入口速度 u_inlet（velocity-inlet 模式）。

【物理推導】
  1. 截面平均加速: u_section = u_inlet / (1 - max_blockage)
  2. 最窄缺口集中: 窄缺口阻力高，流體會繞道 → 用 sqrt 衰減
  3. 角落 + 暫態:   u_peak ≈ u_gap × CORNER_TRANSIENT_FACTOR
  4. 安全條件:      u_peak < MA_LIMIT × CS

  有 min_distance_px 時：更精確估算峰值加速（sqrt 模型 + cap）
  沒有時：fallback 用 MA_SAFETY_FACTOR 粗估
"""

import math
from typing import Any, Dict

from .constants import CS, MA_LIMIT, MA_SAFETY_FACTOR, MIN_OPEN

# 角落效應 + 渦旋暫態倍率
CORNER_TRANSIENT_FACTOR = 1.5
# 缺口集中倍率上限（物理上不可能無限集中）
MAX_GAP_CONCENTRATION = 4.0


def _estimate_peak_velocity_ratio(mask_ctx: Dict[str, Any]) -> float:
    """估算 u_peak / u_inlet 的倍率。

    有 min_distance_px 時：
      section_accel × sqrt(avg_gap / min_gap) × corner_factor
      sqrt 反映窄缺口的高阻力分流效應，cap 在 MAX_GAP_CONCENTRATION。

    沒有時：fallback 用 MA_SAFETY_FACTOR / open_fraction。

    Args:
        mask_ctx: 含 max_blockage, min_distance_px, ny, n_buildings。

    Returns:
        peak_velocity / u_inlet 的估計倍率。
    """
    max_blockage = mask_ctx["max_blockage"]
    open_fraction = max(MIN_OPEN, 1.0 - max_blockage)
    section_accel = 1.0 / open_fraction

    min_dist = mask_ctx.get("min_distance_px", 0)
    ny = mask_ctx.get("ny", 0)
    n_buildings = mask_ctx.get("n_buildings", 0)

    if min_dist > 0 and ny > 0 and n_buildings > 1:
        # 精確模型
        open_width = ny * open_fraction
        n_gaps = max(1, n_buildings + 1)
        avg_gap = open_width / n_gaps

        # sqrt 衰減：窄缺口阻力高，不會吸引全部比例的流量
        raw_conc = avg_gap / min_dist
        gap_concentration = min(MAX_GAP_CONCENTRATION, math.sqrt(max(1.0, raw_conc)))

        return section_accel * gap_concentration * CORNER_TRANSIENT_FACTOR
    else:
        return MA_SAFETY_FACTOR / open_fraction


def fill_blockage_adj(case_result: dict, mask_ctx: dict, sim_ctx: dict) -> None:
    """從 max_blockage + min_distance_px 計算安全 u_inlet。

    Args:
        case_result: 個案計算結果（會被寫入 u_inlet 等欄位）。
        mask_ctx: per-mask context（須含 max_blockage, min_distance_px）。
        sim_ctx: 共用 sim context（須含 rho_out）。

    Writes:
        case_result["u_inlet"]       float  安全入口速度（格子單位）
        case_result["open_fraction"] float  開放比
        case_result["rho_in_case"]   float  等效 rho_in（供 metadata 使用）
        case_result["peak_accel"]    float  預估峰值加速倍率
    """
    peak_accel = _estimate_peak_velocity_ratio(mask_ctx)
    u_inlet = MA_LIMIT * CS / peak_accel

    open_fraction = max(MIN_OPEN, 1.0 - mask_ctx["max_blockage"])
    case_result["u_inlet"] = u_inlet
    case_result["open_fraction"] = open_fraction
    case_result["peak_accel"] = round(peak_accel, 2)
    rho_out = sim_ctx["rho_out"]
    case_result["rho_in_case"] = rho_out + 1.5 * u_inlet ** 2
