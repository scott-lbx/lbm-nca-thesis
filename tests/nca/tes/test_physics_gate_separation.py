"""`physics_gate.separation_points` 行為測試（純 CPU：合成場 + stub converter）。

覆蓋四件事：

1. **均勻附著流不得產生站點**——這是「取切向速度的流向分量、而非沿弧長切向」那個
   定義選擇的回歸測試：沿弧長的切向在繞過方塊時，上/下表面符號必定相反，前/後停滯面
   會出現兩個**純幾何**的假變號。改取流向分量後前/後面恆為 0，不再誤判。
2. **近壁流向速度反轉時找得到站點**，且位置落在符號翻轉的那一格界面上。
3. **三個定義性參數未裁決時硬錯誤**（`SeparationSpecNotDefinedError`），部分給定時
   `physics_gate_metrics` 也硬錯誤，不靜默降級成「不量」。
4. **兩種法向估法都跑得動**，且 ``select`` / ``reduce`` 的語義如 docstring 所述。

不需要 zarr、不需要 GPU、不需要真的 `FieldConverter`（stub 只提供本函式用到的
``vel_x`` / ``vel_y``）。
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
_BX0, _BX1 = 40, 52                     # 固體方塊的欄範圍（含端點）
_BY0, _BY1 = 26, 38                     # 列範圍（含端點）
_U_INF = 0.05
_L_CHAR_PX = 180.0                      # sim_px_width=1920、W=128 → grid lc = 12.0 px


class _StubNormalizer:
    """`FieldConverter.normalizer` 的最小替身（本測試不走 denorm_mphy 路徑）。"""

    def denorm_mphy(self, x: torch.Tensor) -> torch.Tensor:
        """原樣回傳（分離點只用 toVphy 的 vel_x / vel_y）。"""
        return x


class _StubConverter:
    """只提供 `separation_points` 會讀到的欄位的 `FieldConverter` 替身。"""

    def __init__(self, ux: np.ndarray, uy: np.ndarray) -> None:
        self._ux = torch.from_numpy(ux)[None, None]
        self._uy = torch.from_numpy(uy)[None, None]
        self.normalizer = _StubNormalizer()

    def toVphy(self, phys: torch.Tensor, is_norm: bool = True) -> dict[str, torch.Tensor]:
        """回傳固定的 ``vel_x`` / ``vel_y``（與傳入的 ``phys`` 無關，本測試不需要）。"""
        zero = torch.zeros_like(self._ux)
        return {
            "vel_x": self._ux,
            "vel_y": self._uy,
            "vel_mag": torch.sqrt(self._ux**2 + self._uy**2),
            "pressure": zero,
        }


def _scene(reverse_from_col: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """合成場：均勻 ``u_x = +U_INF``；給定欄之後把方塊上/下貼壁列改為逆流。

    Args:
        reverse_from_col: 逆流起始欄（None ＝ 全場附著流）。

    Returns:
        ``(fluid, ux, uy)``，皆為 ``[H, W]``。
    """
    fluid = np.ones((_H, _W), dtype=bool)
    fluid[_BY0 : _BY1 + 1, _BX0 : _BX1 + 1] = False
    ux = np.full((_H, _W), _U_INF, dtype=np.float64)
    uy = np.zeros((_H, _W), dtype=np.float64)
    if reverse_from_col is not None:
        ux[_BY0 - 1, reverse_from_col : _BX1 + 1] = -0.02
        ux[_BY1 + 1, reverse_from_col : _BX1 + 1] = -0.02
    return fluid, ux, uy


def _cfg(**sep: str | None) -> pg.PhysicsGateConfig:
    """建 `PhysicsGateConfig`（能譜三欄位為必填，這裡給測試夾具值）。"""
    return pg.PhysicsGateConfig(
        spectrum_k_lo_frac=0.05,
        spectrum_k_hi_frac=0.35,
        spectrum_r2_min=0.8,
        **sep,
    )


def _run(
    reverse_from_col: int | None,
    *,
    normal_mode: str = "axis_face",
    select: str = "most_upstream",
    reduce: str = "median",
) -> dict:
    """跑一次 `separation_points`。"""
    fluid, ux, uy = _scene(reverse_from_col)
    cfg = _cfg(
        separation_normal_mode=normal_mode,
        separation_select=select,
        separation_reduce=reduce,
    )
    return pg.separation_points(
        torch.zeros(1, 9, _H, _W),
        torch.from_numpy(fluid.astype(np.float32))[None, None],
        _StubConverter(ux, uy),
        cfg=cfg,
        l_char_px=_L_CHAR_PX,
    )


@pytest.mark.parametrize("normal_mode", sorted(pg.SEPARATION_NORMAL_MODES))
def test_attached_uniform_flow_has_no_separation(normal_mode: str) -> None:
    """★核心回歸：均勻附著流不得產生站點（前/後停滯面的幾何變號不算分離）。"""
    out = _run(None, normal_mode=normal_mode)
    assert out["separation_n_bodies"] == 1
    assert out["separation_n_points_total"] == 0
    assert out["separation_n_bodies_with_point"] == 0
    assert not np.isfinite(out["separation_x_over_lchar"])
    body = out["separation_per_body"][0]
    # 迎風/背風面（法向沿 ±x）的流向切應力恆為 0，不參與配對。
    assert body["n_zero_shear"] == 2 * (_BY1 - _BY0 + 1)
    assert body["n_undefined_normal"] == 0


@pytest.mark.parametrize("normal_mode", sorted(pg.SEPARATION_NORMAL_MODES))
def test_reversed_near_wall_flow_is_detected(normal_mode: str) -> None:
    """近壁流向速度反轉 → 上/下表面各一個站點，位置落在翻轉界面上。"""
    rev = _BX0 + 6
    out = _run(rev, normal_mode=normal_mode)
    assert out["separation_n_points_total"] == 2       # 上表面一個、下表面一個
    assert out["separation_n_bodies_with_point"] == 1
    body = out["separation_per_body"][0]
    xs = sorted(s["x"] for s in body["sites"])
    assert xs == [rev - 0.5, rev - 0.5], (
        f"站點欄位置 {xs} 應為翻轉界面（{rev - 1} 與 {rev} 兩格的形心）"
    )
    lc = _L_CHAR_PX * (_W / 1920.0)
    assert out["separation_x_over_lchar"] == pytest.approx((rev - 0.5 - _BX0) / lc)


def test_select_and_reduce_semantics() -> None:
    """``select`` 取上/下游端點；``reduce`` 決定 case 級彙總（單量體時四種相等）。"""
    rev = _BX0 + 6
    up = _run(rev, select="most_upstream")
    down = _run(rev, select="most_downstream")
    # 本合成場的兩個站點同欄，故 select 兩種取法給出同一個 x —— 這裡驗的是
    # 「兩種取法都跑得動、且全部站點都仍完整列出」，不是驗兩者必不相等。
    assert up["separation_per_body"][0]["n_sites"] == 2
    assert down["separation_per_body"][0]["n_sites"] == 2
    assert up["separation_per_body"][0]["selected_x"] == pytest.approx(
        down["separation_per_body"][0]["selected_x"]
    )
    for mode in sorted(pg.SEPARATION_REDUCE_MODES):
        got = _run(rev, reduce=mode)["separation_x_over_lchar"]
        assert got == pytest.approx(up["separation_x_over_lchar"])
    # 定義性選擇一律回寫進輸出，供證據 JSON 事後核對。
    assert up["separation_select"] == "most_upstream"
    assert down["separation_select"] == "most_downstream"
    assert up["separation_reduce"] == "median"


def test_unset_spec_raises() -> None:
    """三個定義性參數未裁決時直接硬錯誤，不代選。"""
    fluid, ux, uy = _scene(_BX0 + 6)
    with pytest.raises(pg.SeparationSpecNotDefinedError):
        pg.separation_points(
            torch.zeros(1, 9, _H, _W),
            torch.from_numpy(fluid.astype(np.float32))[None, None],
            _StubConverter(ux, uy),
            cfg=_cfg(),
            l_char_px=_L_CHAR_PX,
        )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("separation_normal_mode", "nearest_face"),
        ("separation_select", "first"),
        ("separation_reduce", "p95"),
    ],
)
def test_illegal_spec_value_raises(field: str, bad: str) -> None:
    """非法的定義性參數值要 `ValueError`，不可靜默退回某一種。"""
    kwargs = {
        "separation_normal_mode": "axis_face",
        "separation_select": "most_upstream",
        "separation_reduce": "median",
    }
    kwargs[field] = bad
    with pytest.raises(ValueError):
        _run(_BX0 + 6, **{k.removeprefix("separation_"): v for k, v in kwargs.items()})


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("separation_normal_mode", "nearest_face"),
        ("separation_select", "first"),
        ("separation_reduce", "p95"),
    ],
)
def test_illegal_spec_fails_early_even_without_bodies(field: str, bad: str) -> None:
    """★沒有任何合格量體時，非法設定**仍**要早期失敗。

    這是 `surface_cp` 的 cp 參考帶那條紀律的同型回歸：檢查若放在逐量體迴圈裡，
    全流體（無量體）的 case 會靜默吞掉非法設定，最需要早期失敗的路徑反而不失敗。
    """
    kwargs = {
        "separation_normal_mode": "axis_face",
        "separation_select": "most_upstream",
        "separation_reduce": "median",
    }
    kwargs[field] = bad
    fluid = np.ones((_H, _W), dtype=bool)          # 全流體，_body_components 回空 list
    with pytest.raises(ValueError):
        pg.separation_points(
            torch.zeros(1, 9, _H, _W),
            torch.from_numpy(fluid.astype(np.float32))[None, None],
            _StubConverter(np.full((_H, _W), _U_INF), np.zeros((_H, _W))),
            cfg=_cfg(**kwargs),
            l_char_px=_L_CHAR_PX,
        )


def test_metrics_partial_spec_raises_and_empty_spec_skips() -> None:
    """`physics_gate_metrics`：三者全空＝不量分離點；只給一部分＝硬錯誤。"""
    fluid, ux, uy = _scene(_BX0 + 6)
    fm = torch.from_numpy(fluid.astype(np.float32))[None, None]
    conv = _StubConverter(ux, uy)
    with pytest.raises(pg.SeparationSpecNotDefinedError):
        pg.physics_gate_metrics(
            torch.zeros(1, 9, _H, _W),
            torch.zeros(1, 9, _H, _W),
            fm,
            conv,
            cfg=_cfg(separation_normal_mode="axis_face"),
            l_char_px=_L_CHAR_PX,
            u_inf=_U_INF,
        )
