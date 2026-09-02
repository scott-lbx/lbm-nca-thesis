"""`physics_gate.surface_cp` 的壓差阻力符號回歸（純 CPU：合成場 + stub converter）。

`pressure_drag_coeff` ＝ ``−Σ_ring Cp · n_x / A_frontal``。式子前面那個負號是把
「指向流體的外法線」轉成流向阻力所必需的；少了它，鈍體會回**負**阻力。round-4 把這
件事寫成 docstring 裡的一個實測數字，round-5 改成本檔的斷言——數字只存在這裡，
`surface_cp` 的 docstring 不再重寫一份。

合成場（刻意選成可手算的組態，讓斷言值不是「跑出來就抄下來」）：
``H×W = 64×128`` 全流體，置入一個 12×12 固體方塊；方塊迎風側 4 欄 ρ=1.02、背風側
4 欄 ρ=0.98，其餘 ρ=1.0，``u_inf=0.05``、``cp_ring_px=1``。此時
``p = ρ/3``、``q_inf = 0.5·ρ∞·u_inf²``，貼壁環的迎風 12 格 Cp=+16/3、背風 12 格
Cp=−16/3，上下兩列的 ``n_x`` 為 0，故 ``drag = 64 / 12``（＝迎風欄數 × 每格貢獻 ÷
迎風投影高度），見 :func:`test_bluff_body_drag_sign_and_magnitude` 的推導斷言。

不需要 zarr、不需要 GPU、不需要真的 `FieldConverter`。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.eval import physics_gate as pg

_H, _W = 64, 128
_BX0, _BX1 = 40, 51                     # 固體方塊的欄範圍（含端點，寬 12）
_BY0, _BY1 = 26, 37                     # 列範圍（含端點，高 12）
_BAND_PX = 4                            # 迎風／背風高低壓帶的欄數
_RHO_UP, _RHO_DOWN, _RHO_FAR = 1.02, 0.98, 1.0
_U_INF = 0.05


class _StubNormalizer:
    """`FieldConverter.normalizer` 的最小替身：`denorm_mphy` 直接回預先組好的 raw 場。"""

    def __init__(self, raw: torch.Tensor) -> None:
        self._raw = raw

    def denorm_mphy(self, x: torch.Tensor) -> torch.Tensor:
        """回傳固定的 raw 場（本測試不驗 denorm 本身）。"""
        return self._raw


class _StubConverter:
    """只提供 `surface_cp` 會讀到的 ``pressure`` 與 ``rho`` 的 `FieldConverter` 替身。

    ``pressure`` 依 LBM 等溫狀態方程 ``p = c_s² ρ = ρ/3`` 由同一份 ρ 場導出，
    與 `FieldConverter.toVphy` 的定義一致。
    """

    def __init__(self, rho: np.ndarray) -> None:
        raw = torch.zeros(1, 9, *rho.shape, dtype=torch.float32)
        raw[0, 0] = torch.from_numpy(rho).float()
        self._raw = raw
        self._pressure = raw[:, 0:1] / 3.0
        self.normalizer = _StubNormalizer(raw)

    def toVphy(self, phys: torch.Tensor, is_norm: bool = True) -> dict[str, torch.Tensor]:
        """回傳固定的 ``pressure``（與傳入的 ``phys`` 無關，本測試不需要）。"""
        zero = torch.zeros_like(self._pressure)
        return {
            "vel_x": zero,
            "vel_y": zero,
            "vel_mag": zero,
            "pressure": self._pressure,
        }


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """合成鈍體場：方塊迎風側高壓帶、背風側低壓帶。

    Returns:
        ``(fluid, rho)``，皆為 ``[H, W]``。
    """
    fluid = np.ones((_H, _W), dtype=bool)
    fluid[_BY0 : _BY1 + 1, _BX0 : _BX1 + 1] = False
    rho = np.full((_H, _W), _RHO_FAR, dtype=np.float64)
    rho[:, _BX0 - _BAND_PX : _BX0] = _RHO_UP
    rho[:, _BX1 + 1 : _BX1 + 1 + _BAND_PX] = _RHO_DOWN
    return fluid, rho


def _cfg() -> pg.PhysicsGateConfig:
    """建 `PhysicsGateConfig`（能譜三欄位為必填，這裡給測試夾具值）。"""
    return pg.PhysicsGateConfig(
        spectrum_k_lo_frac=0.05, spectrum_k_hi_frac=0.35, spectrum_r2_min=0.8
    )


def _run(cfg: pg.PhysicsGateConfig | None = None) -> dict:
    """跑一次 `surface_cp`。"""
    fluid, rho = _scene()
    return pg.surface_cp(
        torch.zeros(1, 9, _H, _W),
        torch.from_numpy(fluid.astype(np.float32))[None, None],
        _StubConverter(rho),
        cfg=cfg if cfg is not None else _cfg(),
        u_inf=_U_INF,
    )


def test_bluff_body_drag_sign_and_magnitude() -> None:
    """★核心回歸：迎風高壓 + 背風低壓的鈍體必須回**正**阻力，且值等於手算。

    手算：``q_inf = 0.5 · 1.0 · 0.05² = 1.25e-3``；迎風貼壁欄的
    ``Cp = (1.02/3 − 1.0/3)/q_inf = 16/3``、背風為其相反數；兩側的 ``|n_x| = 0.5``
    （固體指示函數中央差分），各 12 格；上下兩列的 ``n_x = 0``。故
    ``−Σ Cp·n_x = 2 × 12 × (16/3) × 0.5 = 64``，除以迎風投影高度 12 ⇒ ``16/3``。
    """
    cp_up = (_RHO_UP / 3.0 - _RHO_FAR / 3.0) / (0.5 * _RHO_FAR * _U_INF**2)
    body_px = _BY1 - _BY0 + 1
    expected = 2 * body_px * cp_up * 0.5 / body_px

    out = _run()
    assert out["n_ring"] == 4 * body_px, "貼壁環格數變動 → 手算的 12+12+12+12 前提失效"
    assert out["rho_ref"] == pytest.approx(_RHO_FAR)
    assert out["pressure_drag_coeff"] > 0.0, (
        "迎風高壓、背風低壓的鈍體回負阻力 —— `surface_cp` 的 "
        "``F_x = −∮ p n_x dS`` 那個負號八成被拿掉了"
    )
    # rel=1e-5：stub 走 float32（與正式 `FieldConverter` 同精度），非物理差異。
    assert out["pressure_drag_coeff"] == pytest.approx(expected, rel=1e-5), (
        f"實得 {out['pressure_drag_coeff']}、手算 {expected}"
    )


def test_cp_ring_is_symmetric_about_far_field() -> None:
    """貼壁環的 Cp 極值對稱於遠場（迎風 +、背風 −），確認 p_ref 取的是遠場而非環上平均。"""
    out = _run()
    cp_up = (_RHO_UP / 3.0 - _RHO_FAR / 3.0) / (0.5 * _RHO_FAR * _U_INF**2)
    assert out["cp_max"] == pytest.approx(cp_up, rel=1e-5)
    assert out["cp_min"] == pytest.approx(-cp_up, rel=1e-5)
    assert out["cp_mean"] == pytest.approx(0.0, abs=1e-4)


def test_illegal_ref_band_raises_even_without_bodies() -> None:
    """★非法的 cp 參考帶在**沒有量體**時仍要早期失敗（同 `separation_points` 的紀律）。"""
    cfg = pg.PhysicsGateConfig(
        spectrum_k_lo_frac=0.05,
        spectrum_k_hi_frac=0.35,
        spectrum_r2_min=0.8,
        cp_ref_col_lo_frac=0.5,
        cp_ref_col_hi_frac=0.5,
    )
    fluid = np.ones((_H, _W), dtype=bool)          # 全流體，_body_components 回空 list
    with pytest.raises(ValueError):
        pg.surface_cp(
            torch.zeros(1, 9, _H, _W),
            torch.from_numpy(fluid.astype(np.float32))[None, None],
            _StubConverter(np.full((_H, _W), _RHO_FAR)),
            cfg=cfg,
            u_inf=_U_INF,
        )
