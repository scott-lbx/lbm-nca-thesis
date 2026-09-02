"""`acoustic_period_steps` 的單元測試。

★ 為什麼需要這個測試（DEEP_AUDIT_2026-07 / v2 L0-a）
-----------------------------------------------------
本函式的回傳值是 **GT 重生成的分段界規格**（`R.0h ②` → v2 `L4-a`）。
稽核期間發生過一次實際事故：監督者派工時給的是**未經 ``1/(1-M^2)`` 修正**的
``17736``，而正解是 ``17870.23``。該錯值一度存活在執行計劃的操作行上，
若照抄進重生成 config，會使 15-87 GPU 小時的產出帶著失配 0.85% 的分段界
（正解 0.10%）。

事故的根因不是算錯，是**沒有任何測試把正確值釘住**。本檔即該防線。

⚠ 本測試會匯入 ``time_mean_accumulator``，該模組 top-level ``import taichi``
→ **訓練進行中不要跑**（會搶 GPU 導致 OOM）。
"""

from __future__ import annotations

import math

import pytest

from lbmnca.lbm.lbm_mrt_les.core.time_mean_accumulator import CS_LU, acoustic_period_steps

# 生產母體常數（DEEP_AUDIT Stage 1.17：掃全部 814 個生產 config，
# nx=2560 為 814/814、u_inlet=0.05 為 814/814）
_PROD_NX: int = 2560
_PROD_U_INLET: float = 0.05

# 監督者手算並經獨立覆核的值（Stage 1.16 修 3）
_EXPECTED_PROD: float = 17870.226971793752

# 未經 1/(1-M^2) 修正的錯值 —— 曾經進入執行計劃的操作行
_WRONG_UNCORRECTED: float = 17736.0


def test_production_value_is_bit_exact() -> None:
    """生產母體參數下的回傳值必須逐 bit 等於已覆核的常數。

    這是本檔的核心斷言：釘住重生成分段界的唯一真相。
    """
    assert acoustic_period_steps(_PROD_NX, _PROD_U_INLET) == _EXPECTED_PROD


def test_mach_correction_is_applied() -> None:
    """必須含 ``1/(1-M^2)`` 修正 —— 這正是當初出錯的地方。"""
    got = acoustic_period_steps(_PROD_NX, _PROD_U_INLET)
    uncorrected = 4.0 * _PROD_NX / CS_LU

    # 未修正值約 17736.2，與正解差 0.75%
    assert uncorrected == pytest.approx(17736.200269505303, rel=1e-12)
    assert got > uncorrected, "缺少 1/(1-M^2) 修正（修正必使週期變長）"

    mach = _PROD_U_INLET / CS_LU
    assert got == pytest.approx(uncorrected / (1.0 - mach**2), rel=1e-15)


def test_rejects_the_historical_wrong_value() -> None:
    """明確拒絕曾經誤用的未修正值，避免同一個錯再次被寫回。"""
    got = acoustic_period_steps(_PROD_NX, _PROD_U_INLET)
    assert abs(got - _WRONG_UNCORRECTED) / got > 1e-3, (
        f"回傳值 {got} 過於接近已知錯值 {_WRONG_UNCORRECTED} —— "
        "1/(1-M^2) 修正可能又被移除了"
    )


def test_zero_velocity_reduces_to_quarter_wave() -> None:
    """``u_inlet = 0`` 時 Mach = 0，應退化為純四分之一波週期 ``4*nx/c_s``。"""
    assert acoustic_period_steps(_PROD_NX, 0.0) == pytest.approx(
        4.0 * _PROD_NX / CS_LU, rel=1e-15
    )


def test_scales_linearly_with_nx() -> None:
    """週期正比於 ``nx``（縱向模態只依賴求解域長度）。

    這條同時是 Stage 1.11 步驟 5「``ny`` 加倍後主峰完全不變」的形式化。
    """
    a = acoustic_period_steps(1280, _PROD_U_INLET)
    b = acoustic_period_steps(2560, _PROD_U_INLET)
    assert b == pytest.approx(2.0 * a, rel=1e-15)


@pytest.mark.parametrize(
    "nx, u_inlet",
    [
        (0, 0.05),      # nx 非正
        (-1, 0.05),     # nx 非正
        (2560, -0.01),  # 速度為負
    ],
)
def test_rejects_invalid_inputs(nx: int, u_inlet: float) -> None:
    """非法輸入必須 ``raise``，⛔ 不得靜默回傳可疑值。"""
    with pytest.raises(ValueError):
        acoustic_period_steps(nx, u_inlet)


def test_rejects_supersonic() -> None:
    """Mach >= 1 時公式不適用，必須 ``raise``。"""
    with pytest.raises(ValueError):
        acoustic_period_steps(_PROD_NX, CS_LU)  # 恰好 Mach = 1


def test_ctu_conversion_at_production_l_char_range() -> None:
    """★ 換算成 CTU 後**逐 case 不同** —— 這是段長必須以「步」定義的理由。

    ``L_char in [25, 220] px``、``u_inlet = 0.05`` →
    ``steps_per_ctu = L_char / u_inlet``，故
    ``T_CTU = T_steps / steps_per_ctu`` 隨 ``L_char`` 變動。
    """
    t_steps = acoustic_period_steps(_PROD_NX, _PROD_U_INLET)

    t_ctu_small = t_steps / (25.0 / _PROD_U_INLET)   # L_char = 25 px
    t_ctu_large = t_steps / (220.0 / _PROD_U_INLET)  # L_char = 220 px

    assert t_ctu_small == pytest.approx(35.74, abs=0.05)
    assert t_ctu_large == pytest.approx(4.06, abs=0.05)
    # 兩端相差近 9 倍 → 以 CTU 定義的固定段長無法對所有 case 與聲學週期公度
    assert t_ctu_small / t_ctu_large > 8.0


def test_formula_matches_independent_reimplementation() -> None:
    """以獨立重寫的公式交叉驗證，避免測試只是複述實作。"""

    def reference(nx: int, u: float) -> float:
        cs = 1.0 / math.sqrt(3.0)
        return (4.0 * nx / cs) / (1.0 - (u / cs) ** 2)

    for nx in (640, 1280, 2560, 5120):
        for u in (0.0, 0.01, 0.05, 0.1):
            assert acoustic_period_steps(nx, u) == pytest.approx(
                reference(nx, u), rel=1e-15
            )
