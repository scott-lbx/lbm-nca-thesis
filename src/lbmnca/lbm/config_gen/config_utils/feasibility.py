"""
config_utils/feasibility.py

LBM 模擬參數的物理可行性檢查（velocity-inlet 版本）。

三重保護（按嚴重程度排序）：
  1. Ma 上限：確保低馬赫假設成立
  2. tau 下限：確保 LBM 鬆弛格式數值穩定
  3. Re 上限：確保 2D 都市場景不崩潰

設計原則：
  - 純函式，無副作用
  - 回傳 (ok, reason) tuple，由呼叫端決定如何處理
"""

from .constants import CS, MA_LIMIT, TAU_MIN, RE_MAX


def check_feasibility(
    u_inlet: float,
    nu_lb: float,
    l_char: int,
) -> tuple[bool, str]:
    """檢查一組 (u_inlet, nu_lb, l_char) 是否物理可行。

    Args:
        u_inlet: 入口速度（格子單位）。
        nu_lb: 格子運動黏度。
        l_char: 特徵長度（像素）。

    Returns:
        (True, "")          → 可行
        (False, reason_str) → 不可行
    """
    Ma = u_inlet / CS
    tau = 3.0 * nu_lb + 0.5
    Re = u_inlet * l_char / nu_lb if nu_lb > 0 else float("inf")

    if Ma > MA_LIMIT:
        return False, (
            f"Ma={Ma:.4f} > {MA_LIMIT}  (u_inlet={u_inlet:.5f} lu/step)"
        )

    if tau < TAU_MIN:
        nu_min = (TAU_MIN - 0.5) / 3.0
        return False, (
            f"tau={tau:.4f} < {TAU_MIN}  (nu_lb={nu_lb:.6f})。"
            f"需 nu_lb >= {nu_min:.6f}"
        )

    if Re > RE_MAX:
        return False, (
            f"Re={Re:.0f} > {RE_MAX}  "
            f"(u_inlet={u_inlet:.5f}, nu={nu_lb:.6f}, L_char={l_char}px)"
        )

    return True, ""
