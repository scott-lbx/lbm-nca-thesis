"""
config_utils/steps_calc.py

衍生物理量與步數計算，結果寫入 case_result dict。

Velocity-inlet 模式：u_inlet 已知，steps_per_ctu = L_char / u_inlet，
不需要 Bernoulli 估算或 U_STEP_FACTOR 修正。
"""

from .constants import CS


def fill_physics_and_steps(case_result: dict, mask_ctx: dict, sim_ctx: dict) -> None:
    """計算衍生物理量（Ma, Re, tau）與步數，結果寫入 case_result。

    Args:
        case_result: 須含 u_inlet, nu_lb。
        mask_ctx: 須含 l_char。
        sim_ctx: 須含 rho_out 即時間相關參數。

    Writes:
        case_result 中的物理量與步數欄位。
    """
    u_inlet = case_result["u_inlet"]
    nu_lb = case_result["nu_lb"]
    l_char = mask_ctx["l_char"]
    rho_out = sim_ctx["rho_out"]
    U_phys = sim_ctx["U_phys"]
    nu_air = sim_ctx["nu_air"]

    Ma = u_inlet / CS
    tau = 3.0 * nu_lb + 0.5
    Re = u_inlet * l_char / nu_lb

    vel_scale = U_phys / u_inlet if u_inlet > 1e-9 else 0
    dx_mm = (nu_air / (vel_scale * nu_lb)) * 1000 if (vel_scale * nu_lb) > 1e-9 else 0

    # --- CTU 步數計算（u_inlet 已知，直接算）---
    steps_per_ctu = max(1, int(l_char / u_inlet))

    # --- 時間參數（per-Re CTU 覆寫優先）---
    target_re = case_result.get("target_reynolds_number")
    ctu_per_re = sim_ctx.get("ctu_per_re", {})
    re_ctu = ctu_per_re.get(int(target_re), {}) if target_re else {}

    warmup_ctu = float(re_ctu.get("warmup", sim_ctx.get("warmup_passes", 8)))
    total_ctu = float(re_ctu.get("total", sim_ctx.get("total_passes", 30)))
    start_record_ctu = float(
        re_ctu.get("start_record", sim_ctx.get("start_record_passes", 9))
    )
    saves_per_ctu = float(sim_ctx.get("saves_per_ctu", 10))

    # 步數換算
    interval = max(1, int(steps_per_ctu / saves_per_ctu))
    warmup_steps = int(warmup_ctu * steps_per_ctu)
    max_steps = int(total_ctu * steps_per_ctu)
    start_record_step = int(start_record_ctu * steps_per_ctu)

    case_result.update(
        {
            "u_inlet": u_inlet,
            "Ma": Ma,
            "Re": Re,
            "tau": tau,
            "dx_mm": dx_mm,
            "steps_per_ctu": steps_per_ctu,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "start_record_step": start_record_step,
            "interval": interval,
        }
    )
