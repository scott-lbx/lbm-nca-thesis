"""
config_utils/nu_sampler.py

Velocity-inlet 模式下的 nu 反推與 target Re 匹配。

核心邏輯：給定 u_inlet 和 target Re，直接推導 nu = u_inlet * L_char / Re。
不再需要 nu_lb_list 搜索。
"""

from typing import Any, Dict, List, Tuple

from .constants import CS, MA_LIMIT, TAU_MIN, RE_MAX


def derive_nu_for_target_re(
    u_inlet: float,
    l_char: int,
    target_re: float,
) -> float:
    """從 u_inlet 和 target Re 反推 nu。

    Args:
        u_inlet: 入口速度（格子單位）。
        l_char: 特徵長度（像素）。
        target_re: 目標 Reynolds number。

    Returns:
        格子運動黏度 nu_lb。
    """
    return u_inlet * l_char / target_re


def collect_cases_for_target_re(
    u_inlet: float,
    l_char: int,
    target_reynolds_numbers: List[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """對每個 target Re 推導 nu，檢查可行性。

    Args:
        u_inlet: 入口速度（格子單位）。
        l_char: 特徵長度（像素）。
        target_reynolds_numbers: 目標 Re 列表。

    Returns:
        (selected, skipped) 兩個列表。
    """
    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    ma = u_inlet / CS

    for target_re in sorted(target_reynolds_numbers):
        nu = derive_nu_for_target_re(u_inlet, l_char, target_re)
        tau = 3.0 * nu + 0.5

        if ma > MA_LIMIT:
            skipped.append({
                "target_reynolds_number": float(target_re),
                "reason": f"Ma={ma:.4f} > {MA_LIMIT}",
            })
            continue

        if tau < TAU_MIN:
            skipped.append({
                "target_reynolds_number": float(target_re),
                "reason": f"tau={tau:.4f} < {TAU_MIN} (nu={nu:.6f})",
            })
            continue

        if target_re > RE_MAX:
            skipped.append({
                "target_reynolds_number": float(target_re),
                "reason": f"Re={target_re} > {RE_MAX}",
            })
            continue

        selected.append({
            "nu_lb": float(nu),
            "target_reynolds_number": float(target_re),
            "estimated_reynolds_number": float(target_re),  # 精確匹配
            "re_relative_error_to_target": 0.0,
            "selection_mode": "velocity_inlet",
        })

    return selected, skipped
