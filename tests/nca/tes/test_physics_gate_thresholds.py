"""physics_gate 判定器測試（純字典，無 zarr/GPU/FieldConverter）。

只測 `evaluate_thresholds` 的**判定語義**：門檻未填時硬錯誤、鍵必須真的存在、
方向依 `PASS_DIRECTIONS` 分支、退化模型不得靜默通過。六項判準的量測函式本身需要
`FieldConverter` 與真實場，不在本檔範圍。

★本檔用的門檻值純屬**測試夾具**，與 D-R4-2 的作者裁決無關，也不寫回模組全域——
每個測試都以 `monkeypatch.setattr` 暫時替換 `PASS_THRESHOLDS`。
"""
from __future__ import annotations

import sys
from pathlib import Path

import contextlib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.eval import physics_gate as pg

# 生產用的已凍結門檻。這是權威副本：`physics_gate.PASS_THRESHOLDS` 必須與此一致，
# 下方 `test_thresholds_are_the_approved_values` 即比對本常數。
_APPROVED_THRESHOLDS: dict[str, float] = {
    "recirc_area_ratio": 0.01016,
    "reattach_length_lchar": 0.26643,
    "wake_width_growth": 0.05319,
    "cp_mean": 0.36043,
    "divj_over_lbm_floor": 2.0,
    "spectrum_logslope": 0.31487,
}

# 測試夾具門檻（與上表無關，僅供本檔造假資料用）：五項「貼近 GT」型用絕對偏差上限，
# divj 用倍數上限。
_FIXTURE_THRESHOLDS: dict[str, float] = {
    "recirc_area_ratio": 0.02,
    "reattach_length_lchar": 0.5,
    "wake_width_growth": 0.05,
    "cp_mean": 0.10,
    "divj_over_lbm_floor": 2.0,
    "spectrum_logslope": 0.5,
}

# GT 場的參考值（帶有真實尾流結構）。
_GT: dict[str, float] = {
    "gt_recirc_area_ratio": 0.080,
    "gt_reattach_length_lchar": 1.900,
    "gt_wake_width_growth": 0.310,
    "gt_cp_mean": -0.250,
    "gt_spectrum_logslope": -2.600,
}


def _metrics(**pred: float) -> dict[str, float]:
    """組出 `physics_gate_metrics` 形狀的最小 metric dict（pred 值由參數給）。"""
    m: dict[str, float] = dict(_GT)
    m.update(pred)
    return m


def _good() -> dict[str, float]:
    """與 GT 幾乎一致的健康模型。"""
    return _metrics(
        recirc_area_ratio=0.078,
        reattach_length_lchar=1.850,
        wake_width_growth=0.300,
        cp_mean=-0.240,
        divj_over_lbm_floor=1.400,
        spectrum_logslope=-2.550,
    )


def _degenerate() -> dict[str, float]:
    """把尾流完全填平的退化模型（recirc/Lr/成長率歸零、能譜過度耗散）。"""
    return _metrics(
        recirc_area_ratio=0.0,
        reattach_length_lchar=0.0,
        wake_width_growth=0.0,
        cp_mean=-0.240,
        divj_over_lbm_floor=1.400,
        spectrum_logslope=-5.000,
    )


@contextlib.contextmanager
def _patched(obj, name: str, value):
    """暫時替換模組屬性（避免污染其他測試）。"""
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


class TestThresholdTables:
    def test_threshold_keys_are_real_metric_keys(self):
        """門檻的鍵必須是量測函式真的會產出的 metric key（surface_cp 是函式名，不是 key）。"""
        assert "surface_cp" not in pg.PASS_THRESHOLDS
        assert set(pg.PASS_THRESHOLDS) == set(pg.PASS_DIRECTIONS)
        assert set(pg.PASS_DIRECTIONS.values()) <= set(pg._VALID_DIRECTIONS)

    def test_thresholds_are_the_approved_values(self):
        """物理閘門門檻為已凍結的六個值，且沒有任何 None。

        鎖確切數值而非只檢查「有定義」：門檻若能隨結果調整，判準就會自我實現。
        要改動任何一個值，理由不得是「因為模型沒通過」。
        """
        assert pg.PASS_THRESHOLDS == _APPROVED_THRESHOLDS
        assert not any(v is None for v in pg.PASS_THRESHOLDS.values())

    def test_missing_threshold_still_raises(self):
        """門檻被清空（任一項為 None）時仍必須硬錯誤，不得靜默放行。"""
        with pytest.raises(pg.ThresholdsNotDefinedError):
            broken = dict(pg.PASS_THRESHOLDS)
            broken["cp_mean"] = None
            with _patched(pg, "PASS_THRESHOLDS", broken):
                pg.evaluate_thresholds(_good())


class TestEvaluateThresholds:
    @pytest.fixture(autouse=True)
    def _fill(self, monkeypatch):
        monkeypatch.setattr(pg, "PASS_THRESHOLDS", dict(_FIXTURE_THRESHOLDS))

    def test_healthy_model_passes_all(self):
        assert all(pg.evaluate_thresholds(_good()).values())

    def test_degenerate_model_is_caught(self):
        """★核心回歸：填平尾流的退化模型不得全項通過。"""
        v = pg.evaluate_thresholds(_degenerate())
        assert not v["recirc_area_ratio"]
        assert not v["reattach_length_lchar"]
        assert not v["wake_width_growth"]
        assert not v["spectrum_logslope"]

    def test_missing_metric_key_raises_valueerror(self):
        """缺鍵要明確報 ValueError，不可讓裸 KeyError 從比較式漏出來。"""
        m = _good()
        del m["gt_recirc_area_ratio"]
        with pytest.raises(ValueError):
            pg.evaluate_thresholds(m)

    def test_nonfinite_is_not_a_pass(self):
        m = _good()
        m["wake_width_growth"] = float("nan")
        assert pg.evaluate_thresholds(m)["wake_width_growth"] is False

    def test_divj_direction_is_le(self):
        """divj 是六項中唯一方向為 ``le``（越小越好）的判準：GT floor 的 10 倍必須判 False。"""
        m = _good()
        m["divj_over_lbm_floor"] = 10.0
        assert not pg.evaluate_thresholds(m)["divj_over_lbm_floor"]

    def test_direction_table_drives_comparison(self, monkeypatch):
        """方向由 PASS_DIRECTIONS 決定，不硬編：改成 ge 後同一組數字翻面。"""
        monkeypatch.setattr(
            pg, "PASS_DIRECTIONS",
            dict(pg.PASS_DIRECTIONS, divj_over_lbm_floor="ge"))
        m = _good()
        m["divj_over_lbm_floor"] = 10.0
        assert pg.evaluate_thresholds(m)["divj_over_lbm_floor"]
