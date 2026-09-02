"""
Preview helpers for velocity-inlet config generation.
"""

from typing import List, Tuple

from .constants import CS, MA_LIMIT, MA_SAFETY_FACTOR, MIN_OPEN, RE_MAX, TAU_MIN


def print_re_preview(sim_ctx: dict, l_char_range: Tuple[int, int]) -> None:
    """印出 velocity-inlet 模式下的 Re / nu / tau 預覽表。"""
    target_res = sim_ctx.get("target_reynolds_numbers", [])
    l_min, l_max = l_char_range

    sep = "=" * 90
    print(sep)
    print("  Velocity-Inlet Re Preview")
    print(sep)
    print(f"  MA_LIMIT={MA_LIMIT}  MA_SAFETY_FACTOR={MA_SAFETY_FACTOR}  TAU_MIN={TAU_MIN}")
    print(f"  mask L_char range: {l_min} ~ {l_max} px")
    if target_res:
        print(f"  target Re: {', '.join(f'{v:.0f}' for v in target_res)}")
    print()

    # 用不同 blockage 展示 u_inlet 範圍
    print(f"  {'blockage':>10}  {'open_frac':>10}  {'u_inlet':>10}  {'Ma':>8}", end="")
    for re in target_res:
        print(f"  {'nu@Re=' + str(int(re)):>14}", end="")
    print()
    print("  " + "-" * (44 + 16 * len(target_res)))

    for blockage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        open_frac = max(MIN_OPEN, 1.0 - blockage)
        accel = MA_SAFETY_FACTOR / open_frac
        u_inlet = MA_LIMIT * CS / accel
        ma = u_inlet / CS
        row = f"  {blockage:>10.1%}  {open_frac:>10.1%}  {u_inlet:>10.5f}  {ma:>8.4f}"
        for re in target_res:
            nu = u_inlet * l_min / re  # 用最小 L_char（最嚴格）
            tau = 3.0 * nu + 0.5
            flag = "!!" if tau < TAU_MIN else "  "
            row += f"  {flag}{nu:>12.6f}"
        print(row)

    print()
    print("  Notes:")
    print(f"    - u_inlet = MA_LIMIT * CS * open_fraction / MA_SAFETY_FACTOR")
    print(f"    - nu = u_inlet * L_char / Re_target（精確匹配）")
    print(f"    - tau < {TAU_MIN} 的組合會被 feasibility check 跳過")
    print(f"    - !! 標記表示 tau 過低（用最小 L_char={l_min}px 估算）")
    print(sep + "\n")


def print_summary(
    sim_ctx: dict,
    success: int,
    skipped: int,
    l_min: int,
    l_max: int,
) -> None:
    """印出 config 生成摘要。"""
    print("=" * 60)
    print(f"[Done] Generated {success} configs, skipped {skipped}.")
    if sim_ctx.get("target_reynolds_numbers"):
        target_text = ", ".join(f"{v:.0f}" for v in sim_ctx["target_reynolds_numbers"])
        print(f"[Targets] Re = {target_text}")
    print(f"[L_char] {l_min} ~ {l_max} px")
    print("=" * 60)
