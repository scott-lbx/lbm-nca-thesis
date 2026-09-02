"""LES Smagorinsky 非平衡應力範數正確性回歸測試（GPU-free，純 numpy）。

驗證 2026-06-24 稽核確認的 LES bug 修正（docs/LBM_SOLVER_PHYSICS_AUDIT_2026-06-24.md §2.1-2.3）：
  - m[7] = Π_xx − Π_yy（正應力「差」），m[8] = Π_xy（用實際 M 矩陣驗證）。
  - 正確範數 Q = sqrt(2·Π:Π) = sqrt(m7² + 4·m8²)（修正後 line 369）。
  - 舊式 sqrt(2·m7² + 2·m8²) 對無跡偏應力給錯值（正應變高估、剪切低估）。

執行：python tests/test_les_strain_norm.py
"""

from __future__ import annotations

import sys

import numpy as np

# Windows 預設 cp950 console 無法輸出 U+2212/Π 等字元，會使含斷言全過的測試因 print 崩潰（exit 1）。
# 強制 stdout UTF-8，避免誤判測試失敗（不影響任何斷言邏輯）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 與 LBM2D_MRT_LES.py:166-179 完全一致的 D2Q9 MRT 轉換矩陣
M = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
    ],
    dtype=np.float64,
)

# 由 M 的動量列回讀格子速度分量：row3 = jx = e_x、row5 = jy = e_y
EX = M[3].copy()
EY = M[5].copy()


def momentum_flux(f: np.ndarray) -> tuple[float, float, float]:
    """由分布 f 直接算動量通量張量分量 (Π_xx, Π_yy, Π_xy)。"""
    pi_xx = float(np.sum(EX * EX * f))
    pi_yy = float(np.sum(EY * EY * f))
    pi_xy = float(np.sum(EX * EY * f))
    return pi_xx, pi_yy, pi_xy


def test_m7_m8_identity() -> None:
    """對任意 f：m[7] = Π_xx − Π_yy、m[8] = Π_xy（線性恆等，與是否無跡無關）。"""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        f = rng.standard_normal(9)
        m = M @ f
        pi_xx, pi_yy, pi_xy = momentum_flux(f)
        assert abs(m[7] - (pi_xx - pi_yy)) < 1e-9, "m7 ≠ Π_xx − Π_yy"
        assert abs(m[8] - pi_xy) < 1e-9, "m8 ≠ Π_xy"
    print("[PASS] m7 = Π_xx − Π_yy、m8 = Π_xy（2000 隨機 f）")


def test_corrected_norm_matches_Q() -> None:
    """無跡偏應力（Π_yy = −Π_xx）下，修正範數 = Q = sqrt(2·Π:Π)；舊式不符。"""
    max_old_err = 0.0
    for a in np.linspace(-0.1, 0.1, 21):  # 法向應力差幅度 (Π_xx = a, Π_yy = −a)
        for s in np.linspace(-0.1, 0.1, 21):  # 剪切 Π_xy = s
            m7 = (a) - (-a)  # Π_xx − Π_yy = 2a
            m8 = s
            # 真值：Q = sqrt(2·Π:Π), Π:Π = Π_xx²+Π_yy²+2Π_xy² = 2a²+2s²
            Q = np.sqrt(2.0 * (a * a + (-a) * (-a) + 2.0 * s * s))
            new_norm = np.sqrt(m7 * m7 + 4.0 * m8 * m8)  # 修正後 (line 369)
            old_norm = np.sqrt(2.0 * m7 * m7 + 2.0 * m8 * m8)  # 舊 buggy
            assert abs(new_norm - Q) < 1e-12, f"new_norm 不等於 Q at a={a},s={s}"
            max_old_err = max(max_old_err, abs(old_norm - Q))
    print(
        f"[PASS] 修正範數 sqrt(m7²+4m8²) == Q（441 組 a,s）；舊式最大偏差 {max_old_err:.4f}"
    )


def test_directional_bias_sign() -> None:
    """確認舊式方向性偏誤：純正應變高估、純剪切低估。"""
    # 純正應變 a=0.1, s=0 → m7=0.2, m8=0
    Q_ext = np.sqrt(2 * (2 * 0.1**2))
    old_ext = np.sqrt(2 * 0.2**2)
    assert old_ext > Q_ext, "舊式純正應變應高估"
    # 純剪切 a=0, s=0.1 → m7=0, m8=0.1
    Q_shear = np.sqrt(2 * (2 * 0.1**2))
    old_shear = np.sqrt(2 * 0.1**2)
    assert old_shear < Q_shear, "舊式純剪切應低估"
    print(
        f"[PASS] 方向偏誤：純正應變舊式 {old_ext/Q_ext:.3f}× 高估、"
        f"純剪切舊式 {old_shear/Q_shear:.3f}× 低估"
    )


if __name__ == "__main__":
    test_m7_m8_identity()
    test_corrected_norm_matches_Q()
    test_directional_bias_sign()
    print("\n所有 LES 範數測試通過 ✓")
