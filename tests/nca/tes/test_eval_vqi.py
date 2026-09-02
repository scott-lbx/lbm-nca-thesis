"""
tests/tes/test_eval_vqi.py
=========================
VQI 視覺品質指數的冒煙測試（`src.tes.eval.vqi`）。

驗的是**運算過程是否與設計相符**，不是模型好不好。全部以合成場為輸入，
每一項都有一個可獨立驗算的預期值，不依賴任何 ckpt、zarr 或 GPU。

刻度是逐案的：``+1`` 為該案真值、``−1`` 為該案零場（動量兩通道歸零），
兩個端點由 `build_anchors` 當場由該案的零場導出，不取自任何母體。

分十一組

一　頻譜類項的專責性　把只含單一已知結構的場餵進去，確認該項有反應而其餘項沒有
二　遮罩腐蝕與流體限定　確認腐蝕半徑正確，且頻譜確實只取流體內部
三　偏差量與壓縮　`deviation`／`bounded_deviation` 的方向性、比值項封閉式、防除零常數無關性
四　刻度建構的健全性　`build_anchors` 的 usable 判定與 noise_floor 登錄
五　子分數與合成的核心保證　±1 端點、可低於 −1、`combine` 取 min、反函數
六　否決門檻與否決邏輯　`veto_thresholds` 與 `combine` 的否決判定
七　項組、標籤與參照場的一致性
八　物理有效性前置檢查　`validity`
九　`compute_terms` 的錯誤處理
十　對外入口 `compute_vqi`　端到端評分與逐案獨立性
十一　舊三基線介面已完全移除

Run from project root:
    python -m pytest tests/tes/test_eval_vqi.py -v
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.eval.vqi import (  # noqa: E402
    ARTIFACT_TAGS, CANONICAL_TERM_SET, CLEAN_TAGS, DEAD_TAGS, ERODE_PX,
    LATTICE_SOUND_SPEED, MOMENTUM_CHANNELS, TANH_DIVISOR, TERM_SETS, TERMS,
    VETO_REF_TERM, VQIScore, bounded_deviation, build_anchors, combine,
    compute_terms, compute_vqi, deviation, resolve_term_set, zero_field,
    subscore_inverse, subscores, veto_reference_fields, veto_thresholds,
    validity,
)
from lbmnca.nca.tes.eval.vqi.terms import (  # noqa: E402
    IDEAL_BY_KIND, _erode, band_prominence, checker_energy, diag_prominence,
    hf_band_energy, kinetic_energy, peak_speed, total_variation,
)

H, W = 96, 128      # 頻譜／腐蝕類測試用的大網格
SH, SW = 32, 32     # 端到端 compute_vqi 測試用的小網格


# ───────────────────────── 共用 fixtures／輔助函式 ─────────────────────────

@pytest.fixture()
def mask() -> torch.Tensor:
    """全流體遮罩 ``[1,1,H,W]``。"""
    return torch.ones(1, 1, H, W)


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    """回傳 ``(ys, xs)``，形狀分別為 ``[H,1]`` 與 ``[1,W]``。"""
    return (torch.arange(H, dtype=torch.float32).view(H, 1),
            torch.arange(W, dtype=torch.float32).view(1, W))


def _smooth(amp: float = 1.0) -> torch.Tensor:
    """低頻平滑場 ``[1,H,W]``，作為「乾淨流場」的替身。"""
    ys, xs = _grid()
    return (amp * (1.0 + 0.3 * torch.sin(ys / H * 2 * np.pi)
                   * torch.cos(xs / W * 2 * np.pi))).unsqueeze(0)


def _checker(amp: float = 0.2) -> torch.Tensor:
    """Nyquist 棋盤格 ``[1,H,W]``。"""
    ys, xs = _grid()
    return (amp * (-1.0) ** (ys + xs)).unsqueeze(0)


def _hstripe(amp: float = 0.2) -> torch.Tensor:
    """水平條紋，沿 x 均勻、沿 y 變化，能量落在 ``kx≈0``。"""
    ys, _ = _grid()
    return (amp * torch.sin(ys * 0.9)).unsqueeze(0).expand(1, H, W).clone()


def _dstripe(amp: float = 0.2) -> torch.Tensor:
    """斜向條紋，能量落在頻譜對角線。"""
    ys, xs = _grid()
    return (amp * torch.sin((ys + xs) * 0.9)).unsqueeze(0)


class _StubNorm:
    """最小 `FieldNormalizer` 替身，供合成場的端到端測試使用。

    ``denorm_mphy`` 為恆等映射；``vel_vis_max``／``vor_vis_range`` 給
    stats.json 的全域 p99.5 SSOT 值，供固定視覺範圍類指標（SSIM 等）換算。
    """

    @staticmethod
    def denorm_mphy(x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def vel_vis_max(self) -> float:
        return 0.14

    @property
    def vor_vis_range(self) -> float:
        return 0.016


NORM = _StubNorm()


def _mphy_field(seed: int = 0) -> torch.Tensor:
    """造一個 ``[1,9,SH,SW]`` 的合成真值場：密度為 1、動量為一個平滑渦旋加小擾動。

    9 通道對齊 MRT 矩空間慣例，VQI 只讀 ``m0``（密度）與動量兩通道
    （`MOMENTUM_CHANNELS`），其餘通道不參與五項計算，故填 0。
    """
    g = torch.Generator().manual_seed(seed)
    ys = torch.arange(SH, dtype=torch.float32).view(SH, 1)
    xs = torch.arange(SW, dtype=torch.float32).view(1, SW)
    cy, cx = SH / 2.0, SW / 2.0
    r = ((ys - cy) ** 2 + (xs - cx) ** 2).sqrt()
    swirl = torch.exp(-r.square() / (2.0 * (SH / 4.0) ** 2))
    jx = 0.04 * swirl * torch.cos(r * 0.3) + 0.004 * torch.randn(SH, SW, generator=g)
    jy = 0.04 * swirl * torch.sin(r * 0.3) + 0.004 * torch.randn(SH, SW, generator=g)
    x = torch.zeros(1, 9, SH, SW)
    x[:, 0] = 1.0
    x[:, MOMENTUM_CHANNELS[0]] = jx
    x[:, MOMENTUM_CHANNELS[1]] = jy
    return x


def _mphy_batch(n: int, seed0: int = 100) -> torch.Tensor:
    """疊 ``n`` 個獨立的合成真值場為一批，形狀 ``[n,9,SH,SW]``。"""
    return torch.cat([_mphy_field(seed=seed0 + i) for i in range(n)], dim=0)


def _hand_anchor(kind: str, zero: float, group: str = "texture") -> dict[str, dict]:
    """手造單項刻度，供歸一數學（`subscores`／`combine`／`subscore_inverse`）的獨立驗算。

    直接複製 `build_anchors` 的內部邏輯，略過項組耦合，使測試可以自由選擇
    任意 ``kind``／``zero`` 組合。``zero`` 即該案零場在該項上的原始值，
    單位由它當場導出。
    """
    unit = bounded_deviation(kind, zero)
    usable = math.isfinite(unit) and unit > 0.0
    return {"X": {"ideal": IDEAL_BY_KIND[kind], "zero": zero, "unit": float(unit),
                  "kind": kind, "group": group, "noise_floor": float("nan"),
                  "usable": usable, "reason": ""}}


def _still_for(term_set: str, override: dict[str, float] | None = None,
              ) -> dict[str, float]:
    """幫項組裡的每一項給一個有鑑別度的零場值（0.1），供健全性測試覆寫其一。

    0.1 對三種方向皆非理想值、皆為有限值，故預設情形下每一項都可用。
    """
    terms, _veto_ref = resolve_term_set(term_set)
    still = {name: 0.1 for name in terms}
    if override:
        still.update(override)
    return still


def _flow(rho: float = 1.0, jx: float = 0.05, jy: float = 0.0) -> torch.Tensor:
    """造一個九通道等值場，密度與動量可指定，供有效性檢查測試使用。"""
    x = torch.zeros(1, 9, 8, 8)
    x[:, 0] = rho
    x[:, 3] = jx * rho
    x[:, 5] = jy * rho
    return x


# ───────────────────────── 一　頻譜類項的專責性 ─────────────────────────

def test_checker_energy_只對棋盤格有反應(mask: torch.Tensor) -> None:
    """Nyquist 角落佔比須被棋盤格拉高，被平滑場壓低。"""
    base = checker_energy(_smooth(), mask)
    cb = checker_energy(_smooth() + _checker(), mask)
    assert cb > base * 50, f"棋盤格未拉高 Checker_Ratio，base={base:.3e} cb={cb:.3e}"


def test_checker_energy_對棋盤格的反應遠強於對條紋(mask: torch.Tensor) -> None:
    """棋盤項的專責性。

    ★判準取**三種擾動之間的相對強弱**，不取對乾淨場的倍率。
    `_smooth` 是解析低頻場，其 Nyquist 能量在 1e-16 量級即浮點零，
    任何擾動除以它都會得到天文數字，那個倍率沒有意義。
    """
    cb = checker_energy(_smooth() + _checker(), mask)
    hs = checker_energy(_smooth() + _hstripe(), mask)
    ds = checker_energy(_smooth() + _dstripe(), mask)
    assert cb > hs * 100, f"棋盤格未壓過水平條紋，cb={cb:.3e} hs={hs:.3e}"
    assert cb > ds * 100, f"棋盤格未壓過斜向條紋，cb={cb:.3e} ds={ds:.3e}"


def test_band_prominence_只對水平條紋有反應(mask: torch.Tensor) -> None:
    """``kx≈0`` 的凸起度須被水平條紋拉高，且不被斜向條紋拉高。"""
    base = band_prominence(_smooth(), mask)
    hs = band_prominence(_smooth() + _hstripe(), mask)
    ds = band_prominence(_smooth() + _dstripe(), mask)
    assert hs > base * 2, f"水平條紋未拉高 Band_Prom，base={base:.4f} hs={hs:.4f}"
    assert ds < hs, f"斜向條紋在水平項上不得高於水平條紋，ds={ds:.4f} hs={hs:.4f}"


def test_diag_prominence_只對斜向條紋有反應(mask: torch.Tensor) -> None:
    """對角楔形佔比須被斜向條紋拉高，且不被水平條紋或棋盤格拉高。

    棋盤格那一項是本項的設計重點——環帶上界 0.40 排除 Nyquist 角落，
    棋盤格的能量因而不進本項，與 `Checker_Ratio` 不重複計分。
    """
    base = diag_prominence(_smooth(), mask)
    ds = diag_prominence(_smooth() + _dstripe(), mask)
    hs = diag_prominence(_smooth() + _hstripe(), mask)
    cb = diag_prominence(_smooth() + _checker(), mask)
    assert ds > base * 2, f"斜向條紋未拉高 Diag_Prom，base={base:.4f} ds={ds:.4f}"
    assert hs < ds, f"水平條紋在斜向項上不得高於斜向條紋，hs={hs:.4f} ds={ds:.4f}"
    assert cb < ds, f"棋盤格洩入斜向項，cb={cb:.4f} ds={ds:.4f}"


def test_hf_band_雙側(mask: torch.Tensor) -> None:
    """中高頻環帶佔比為雙側，過度平滑使其下降、顆粒使其上升。"""
    ref = hf_band_energy(_smooth(), mask)
    g = torch.Generator().manual_seed(0)
    noisy = hf_band_energy(_smooth() + 0.1 * torch.randn(1, H, W, generator=g), mask)
    flat = hf_band_energy(_smooth(0.0) + 1.0, mask)     # 常數場，無任何結構
    assert noisy > ref, f"顆粒未拉高 HF_Band，ref={ref:.4e} noisy={noisy:.4e}"
    assert flat <= ref, f"無結構場未壓低 HF_Band，ref={ref:.4e} flat={flat:.4e}"


# ───────────────────────── 二　遮罩腐蝕與流體限定 ─────────────────────────

def test_erode_只在固體周圍內縮() -> None:
    """腐蝕的作用域是固體周圍，計算域邊界不動。

    ``max_pool2d`` 以負無限大補邊，全流體遮罩的補集恆為零，池化後仍為零。
    要擋的是固體邊界的階梯，計算域邊界沒有固體、不構成該問題。
    """
    full = torch.ones(1, 1, H, W)
    assert torch.equal(_erode(full), full), "全流體遮罩不應被內縮"

    fm = full.clone()
    fm[:, :, 40:56, 40:56] = 0.0
    e = _erode(fm)
    r = ERODE_PX
    assert e[0, 0, 40 - r, 48] == pytest.approx(0.0), "固體上緣 r 像素內未被移除"
    assert e[0, 0, 40 - r - 1, 48] == pytest.approx(1.0), "內縮超出 r 像素"
    assert e[0, 0, 0, 0] == pytest.approx(1.0), "計算域角落不應被移除"
    removed = int((fm - e).sum().item())
    assert removed == (16 + 2 * r) ** 2 - 16 * 16, "移除的環寬與 ERODE_PX 不符"


def test_腐蝕排除固體鄰近的擾動() -> None:
    """腐蝕要擋的是固體周圍那一圈，該處的擾動不得進入頻譜。

    同一塊棋盤格補丁，貼在固體鄰接處者應被腐蝕吃掉，貼在流體內部者應留下。
    """
    fm = torch.ones(1, 1, H, W)
    fm[:, :, 40:56, 40:56] = 0.0
    r = ERODE_PX
    near, far = _smooth().clone(), _smooth().clone()
    patch = _checker()[0, 40 - r:40, 40:56]
    near[0, 40 - r:40, 40:56] += patch                    # 緊貼固體上緣
    far[0, 4:4 + r, 40:56] += patch                       # 離固體很遠
    e_near = checker_energy(near, fm)
    e_far = checker_energy(far, fm)
    assert e_near < e_far / 10, (
        f"固體鄰近的擾動未被腐蝕排除，near={e_near:.3e} far={e_far:.3e}")


def test_共同邊緣效應在比值中對消() -> None:
    """遮罩與腐蝕造出的硬邊為預測與真值共有，比值須因而回到 1。

    這是邊緣效應真正的處理方式，腐蝕只把階梯往內移、並未消除它。
    """
    fm = torch.ones(1, 1, H, W)
    fm[:, :, 40:56, 40:56] = 0.0
    v = _smooth()
    for fn in (checker_energy, band_prominence, diag_prominence, hf_band_energy):
        ratio = fn(v, fm) / max(fn(v.clone(), fm), 1e-30)
        assert ratio == pytest.approx(1.0, abs=1e-9), f"{fn.__name__} 的自比不為 1"


def test_kinetic_energy_與_peak_只算流體區() -> None:
    """固體區歸零，不得計入動能均值；峰值取 p99.9 而非最大值。"""
    fm = torch.ones(1, 1, H, W)
    fm[:, :, :, W // 2:] = 0.0
    v = torch.ones(1, H, W)
    v[:, :, W // 2:] = 1e6                       # 固體區塞極大值
    assert kinetic_energy(v, fm) == pytest.approx(1.0, rel=1e-6)
    assert peak_speed(v, fm) == pytest.approx(1.0, rel=1e-6)


def test_peak_speed_不被單一格點主導() -> None:
    """單一離群格點不得改變 p99.9。"""
    fm = torch.ones(1, 1, H, W)
    v = torch.ones(1, H, W)
    before = peak_speed(v, fm)
    v[0, 0, 0] = 1e6
    assert peak_speed(v, fm) == pytest.approx(before, rel=1e-6)


def test_total_variation_對顆粒有反應(mask: torch.Tensor) -> None:
    """總變差須隨紋理總量上升。"""
    g = torch.Generator().manual_seed(1)
    tv_clean = total_variation(_smooth(), mask)
    tv_noisy = total_variation(_smooth() + 0.1 * torch.randn(1, H, W, generator=g), mask)
    assert tv_noisy > tv_clean * 2


# ───────────────────────── 三　偏差量與壓縮 ─────────────────────────

@pytest.mark.parametrize("kind,x,expected", [
    ("up", 0.4, 0.6), ("down", 0.4, 0.4),
    ("ratio", math.e, 1.0), ("ratio", 1.0 / math.e, 1.0),
])
def test_deviation依方向計算(kind: str, x: float, expected: float) -> None:
    """0 為理想，越大越差；比值項取 ``|ln x|``，過強與過弱同分。"""
    assert deviation(kind, x) == pytest.approx(expected, abs=1e-9)


def test_deviation下限夾在零() -> None:
    """理想值不可超越，超越只可能是浮點誤差，須夾到 0 而非給負偏差。"""
    assert deviation("up", 1.0 + 1e-8) == 0.0
    assert deviation("down", -1e-8) == 0.0
    assert deviation("ratio", 1.0 + 1e-8) >= 0.0


@pytest.mark.parametrize("r", [4.0, 0.25, 2.0, 0.5, 10.0, 0.1, 3.0, 1.0])
def test_比值項的壓縮偏差等於封閉式(r: float) -> None:
    """``TANH_DIVISOR=2`` 使比值項的壓縮偏差恰為標準相對差 ``|r−1|/(r+1)``。"""
    closed = abs(r - 1.0) / (r + 1.0)
    assert bounded_deviation("ratio", r) == pytest.approx(closed, abs=1e-9)
    assert TANH_DIVISOR == pytest.approx(2.0)


def test_比值項雙側對稱() -> None:
    """互為倒數的兩個讀數須得同分，``4×`` 與 ``0.25×`` 不例外。"""
    a = _hand_anchor("ratio", 8.0)
    s_hi = subscores({"X": 3.0}, a)["X"]
    s_lo = subscores({"X": 1.0 / 3.0}, a)["X"]
    assert s_hi == pytest.approx(s_lo, abs=1e-9)


def test_比值項下限與防除零常數無關() -> None:
    """零場在比值項的偏差由 ``max(x, 1e-12)`` 這個防除零常數決定，
    但壓縮（tanh）後與該常數的實際取值無關——``1e-6`` 與 ``1e-30`` 皆落在
    ``1.0`` 附近，不像未壓縮的 ``|ln x|`` 會相差近一倍（13.8 對 27.6）。
    """
    assert bounded_deviation("ratio", 0.0) == pytest.approx(1.0, abs=1e-9)
    assert bounded_deviation("ratio", 1e-30) == pytest.approx(1.0, abs=1e-9)
    assert bounded_deviation("ratio", 1e-6) == pytest.approx(1.0, abs=1e-5)


# ───────────────────────── 四　刻度建構的健全性 ─────────────────────────

def test_零場偏差非有限值時判為不可用() -> None:
    """零場的偏差為非有限值時，該項對『有沒有流動』毫無反應，須被擋下。"""
    still = _still_for("slim_pos5", {"Peak_Ratio": float("nan")})
    a = build_anchors(still, "slim_pos5")["Peak_Ratio"]
    assert a["usable"] is False
    assert "非有限值" in a["reason"]


def test_零場恰在理想值上時判為不可用() -> None:
    """零場的偏差為零時，該項沒有負向動態範圍，須被擋下而非產出無意義的分數。"""
    still = _still_for("slim_pos5", {"Peak_Ratio": IDEAL_BY_KIND["ratio"]})
    a = build_anchors(still, "slim_pos5")["Peak_Ratio"]
    assert a["usable"] is False
    assert "無鑑別度" in a["reason"]


def test_一般情形判為可用並登錄單位() -> None:
    """零場有正常偏差時該項可用，其單位即該偏差的壓縮值。"""
    still = _still_for("slim_pos5")
    a = build_anchors(still, "slim_pos5")["Peak_Ratio"]
    assert a["usable"] is True
    assert a["unit"] == pytest.approx(bounded_deviation("ratio", 0.1))
    assert a["zero"] == pytest.approx(0.1)
    assert a["ideal"] == 1.0
    assert a["kind"] == "ratio"


def test_noise_floor僅登錄不參與刻度() -> None:
    """`noise_floor` 只登錄供報告對照，換一個噪聲底線不得改變單位。"""
    still = _still_for("slim_pos5")
    a_with = build_anchors(still, "slim_pos5", {"Peak_Ratio": 0.987})["Peak_Ratio"]
    a_without = build_anchors(still, "slim_pos5")["Peak_Ratio"]
    assert a_with["noise_floor"] == pytest.approx(0.987)
    assert math.isnan(a_without["noise_floor"])
    assert a_with["unit"] == pytest.approx(a_without["unit"]), "noise_floor 不得改變刻度"


# ───────────────────────── 五　子分數與合成的核心保證 ─────────────────────────

@pytest.mark.parametrize("kind,zero", [
    ("up", 0.40), ("down", 0.05), ("ratio", 1.20), ("ratio", 8.0),
])
def test_subscores於兩端點恰為正負一(kind: str, zero: float) -> None:
    """``+1`` 為理想值（該案真值）、``−1`` 為零場，兩者由定義精確給出。"""
    a = _hand_anchor(kind, zero)
    ideal = IDEAL_BY_KIND[kind]
    assert subscores({"X": ideal}, a)["X"] == pytest.approx(1.0, abs=1e-9)
    assert subscores({"X": zero}, a)["X"] == pytest.approx(-1.0, abs=1e-9)


def test_低於零場的讀數仍可分辨強弱() -> None:
    """漸近但嚴格單調，兩個都比零場更差的讀數不得被夾成同一個值。"""
    a = _hand_anchor("up", 0.4)
    worse = subscores({"X": 0.0}, a)["X"]
    worst = subscores({"X": -50.0}, a)["X"]
    assert worse < -1.0 and worst < -1.0
    assert worst < worse


def test_分數可以低於負一() -> None:
    """一個與真值反相關的場，比什麼都不做更糟——這是刻意保留的資訊，不夾板。

    語義對照 `VortSignedCorr`：理想值為 +1，零場的讀數約為 0（無結構可比），
    完全反相關（−1）須比零場更差。
    """
    a = _hand_anchor("up", 0.0)
    s = subscores({"X": -1.0}, a)["X"]
    assert s < -1.0


def test_combine取最小值而非平均() -> None:
    """**這是 VQI 存在的理由。** 四項好、一項極差時，`vqi` 須等於那一項，
    不得被其餘四項的高分稀釋。
    """
    anchors = {n: _hand_anchor("up", 0.0)["X"] | {"group": "struct"}
               for n in ("A", "B", "C", "D", "E")}
    raw = {"A": 0.90, "B": 0.85, "C": 0.95, "D": 0.99, "E": -0.5}   # E 極差
    out = combine(raw, anchors)
    s = subscores(raw, anchors)
    assert out.vqi == pytest.approx(s["E"])
    assert out.binding_term == "E"
    mean_s = sum(s.values()) / len(s)
    assert out.vqi < mean_s, "min 必須嚴格低於算術平均，否則等同被稀釋"


def test_不可用項不進入合成() -> None:
    """標為不可用的項須被跳過，不得把它的分數算進最小值。"""
    anchors = {
        "good": _hand_anchor("up", 0.0)["X"] | {"group": "struct"},
        "bad": _hand_anchor("up", 0.0)["X"] | {"group": "struct", "usable": False},
    }
    raw = {"good": 0.8, "bad": 0.0}
    out = combine(raw, anchors)
    assert out.vqi == pytest.approx(subscores(raw, anchors)["good"])
    assert out.binding_term == "good"


def test_vqi_無可用項時不靜默() -> None:
    """全部項不可用（或空）時須回 nan 並自動否決，不得回一個看似正常的分數。"""
    out = combine({}, {})
    assert np.isnan(out.vqi)
    assert out.veto is True


def test_subscore_inverse與subscores互為反函數() -> None:
    """`subscore_inverse` 須還原偏差量，正負兩段皆然，閉式解無需迭代。

    ★極端值（如 ``x=-20``）使壓縮後的偏差極貼近 tanh 的漸近值 1，
    `atanh` 在該處導數趨於無限大，浮點還原因而僅能到 ``1e-6`` 量級的精度，
    這是數值條件數的限制而非公式錯誤。
    """
    a = _hand_anchor("up", 0.4)
    for x in (1.0, 0.8, 0.40, 0.2, 0.05, -0.5, -20.0):
        s = subscores({"X": x}, a)["X"]
        assert subscore_inverse("X", s, a) == pytest.approx(
            deviation("up", x), rel=1e-6, abs=1e-6), f"原始值 {x} 還原不符"


def test_subscore_inverse拒絕越界分數() -> None:
    """分數對應的壓縮後偏差不落在 ``[0, 1)`` 時無從還原，須明白報錯。"""
    a = _hand_anchor("up", 0.4)
    for bad in (1.0001, -100.0):
        with pytest.raises(ValueError):
            subscore_inverse("X", bad, a)


def test_落檔用的_dict_把非有限值轉為_None() -> None:
    """`as_dict()` 不得輸出 nan。

    JSON 沒有 NaN 這個常值，`json.dumps` 預設卻會寫出 `NaN` 字面值，
    嚴格解析器讀到就拋 ParseError，讀數檔因而不可攜。
    Python 端保留 nan，落檔這一層必須是 None。
    """
    anchors = {n: _hand_anchor("up", 0.0)["X"] | {"group": "struct"}
               for n in ("A", "B")}
    out2 = combine({"A": float("nan")}, anchors)
    assert np.isnan(out2.vqi), "全部項不可用時 vqi 為 nan"
    d = out2.as_dict()
    assert d["vqi"] is None, "落檔層必須轉 None"
    dumped = json.dumps(d, allow_nan=False)     # allow_nan=False 時有 nan 即拋
    assert "NaN" not in dumped


def test_距離RMS對任何合法子分數皆有定義且不取代判準() -> None:
    """`vqi_rms` 是次要讀數，對負子分數仍有定義，但**不得取代 min 作判準**。"""
    anchors = {n: _hand_anchor("up", 0.0)["X"] | {"group": "struct"}
               for n in ("A", "B")}
    ok = combine({"A": 0.8, "B": 0.5}, anchors)
    assert np.isfinite(ok.vqi_rms) and -1.0 < ok.vqi_rms <= 1.0
    bad = combine({"A": 0.8, "B": -0.3}, anchors)
    assert bad.vqi < 0.0 and bad.vqi == pytest.approx(
        subscores({"B": -0.3}, {"B": anchors["B"]})["B"])
    assert np.isfinite(bad.vqi_rms), "距離 RMS 對任何合法子分數皆有定義"
    assert bad.vqi_rms > bad.vqi, "RMS 是一種平均，恆不低於最小值"


# ───────────────────────── 六　否決門檻與否決邏輯 ─────────────────────────

def test_resolve_term_set回傳二元組() -> None:
    """`resolve_term_set` 現在只回傳 ``(terms, veto_ref)``。"""
    result = resolve_term_set("full")
    assert len(result) == 2
    terms, veto_ref = result
    assert isinstance(terms, dict) and isinstance(veto_ref, dict)


def test_未知項組拋出KeyError() -> None:
    with pytest.raises(KeyError):
        resolve_term_set("__no_such_term_set__")


def test_veto_reference_fields的類別名不帶百分比() -> None:
    """四類偽影的類別名為固定字面值，振幅已由 `VETO_REF_AMPLITUDE` 定死，
    不重複寫進名字。
    """
    t = _mphy_field(seed=9)
    m = torch.ones(1, 1, SH, SW)
    refs = veto_reference_fields(t, m)
    assert set(refs) == {"棋盤格", "白雜訊", "水平條紋", "斜向條紋"}


def test_veto門檻取多個參照中最不敏感的一級() -> None:
    """語義為「這一項所負責的失效裡，最輕但仍必須被擋下的那一級」，
    取多個參照裡**最高**的子分數（即最寬鬆的那一級），門檻因而較嚴。
    """
    still = _still_for("slim_pos5")
    anchors = build_anchors(still, "slim_pos5")
    veto_ref = {
        "棋盤格": {"RAPSD_Shape": 5.0},
        "白雜訊": {"RAPSD_Shape": 3.0},
        "水平條紋": {"RAPSD_Shape": 0.5},     # 偏差最小，即最輕的一級
        "斜向條紋": {"RAPSD_Shape": 1.0},
    }
    tau = veto_thresholds(veto_ref, anchors, "slim_pos5")
    s_all = {k: subscores(v, anchors)["RAPSD_Shape"] for k, v in veto_ref.items()}
    assert tau["RAPSD_Shape"] == pytest.approx(max(s_all.values()))
    assert tau["RAPSD_Shape"] == pytest.approx(s_all["水平條紋"])


def test_未列於參照表的項不獲得門檻() -> None:
    """沒有否決參照的項不設門檻，不參與否決。"""
    still = _still_for("slim_pos5")
    anchors = build_anchors(still, "slim_pos5")
    tau = veto_thresholds({"棋盤格": {"RAPSD_Shape": 5.0}}, anchors, "slim_pos5")
    assert "StructVel_RGB_SSIM" not in tau


def test_否決只看紋理群() -> None:
    """結構與能量項再低也不觸發否決，那是 VQI 數值的職責。"""
    anchors = {
        "S": _hand_anchor("up", 0.0)["X"] | {"group": "struct"},
        "T": _hand_anchor("up", 0.0)["X"] | {"group": "texture"},
    }
    tau = {"S": 0.5, "T": 0.5}
    out = combine({"S": 0.0, "T": 0.9}, anchors, tau)
    assert out.veto is False, "結構項不得觸發否決"
    out = combine({"S": 0.9, "T": 0.0}, anchors, tau)
    assert out.veto is True and out.veto_terms == ["T"]


def test_否決判定式為小於等於() -> None:
    """恰好等於門檻須觸發，這是參照退化本身也被擋下的原因。

    取零場本身（``raw == still``）當門檻最直接——其子分數由定義恰為 ``−1``
    （見 `test_subscores在零場上恰為負一` 系列），比較浮點誤差不會誤判邊界。
    """
    a = _hand_anchor("up", 0.4)
    anchors = {"T": a["X"] | {"group": "texture"}}
    tau = {"T": -1.0}
    assert combine({"T": 0.4}, anchors, tau).veto is True, "恰在零場，等於門檻須觸發"
    assert combine({"T": 0.41}, anchors, tau).veto is False, "優於零場，須不觸發"


def test_未給門檻的項不觸發否決() -> None:
    """沒有參照退化的項不參與否決，不得以預設值誤擋。"""
    anchors = {"T": _hand_anchor("up", 0.0)["X"] | {"group": "texture"}}
    assert combine({"T": 0.0}, anchors, {}).veto is False


# ───────────────────────── 七　項組、標籤與參照場的一致性 ─────────────────────────

def test_canonical項組為五項的slim_pos5() -> None:
    assert CANONICAL_TERM_SET == "slim_pos5"
    terms, _veto_ref = resolve_term_set(CANONICAL_TERM_SET)
    assert len(terms) == 5


def test_否決參照的項全部屬紋理群() -> None:
    """非紋理項設門檻不會生效，設了即為死設定。"""
    for term in VETO_REF_TERM:
        assert TERMS[term][0] == "texture", f"{term} 不屬紋理群卻設了否決門檻"


def test_視覺校準標籤互斥() -> None:
    """三組標籤不得重疊，否則驗收第七關的判定自相矛盾。"""
    groups = [set(ARTIFACT_TAGS), set(DEAD_TAGS), set(CLEAN_TAGS)]
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert not (a & b), f"標籤重疊 {a & b}"


@pytest.mark.parametrize("term_set", sorted(TERM_SETS))
def test_每個項組的resolve均回傳合法二元組(term_set: str) -> None:
    terms, veto_ref = resolve_term_set(term_set)
    assert terms, f"{term_set} 項組為空"
    assert isinstance(veto_ref, dict)
    for name in veto_ref:
        assert name in terms, f"{term_set} 的否決參照 {name} 不在項表內"


# ───────────────────────── 八　物理有效性前置檢查 ─────────────────────────

def test_乾淨場通過有效性檢查() -> None:
    """密度為正且速度遠低於晶格聲速時不得誤報。"""
    assert validity(_flow(), torch.ones(1, 1, 8, 8), NORM) == []


def test_負密度判為無效() -> None:
    """密度為負在物理上不存在，且速度由 j/rho 得出，跨零時該除法發散。"""
    x = _flow()
    x[:, 0, 3, 3] = -0.2
    assert "流體區出現負密度" in validity(x, torch.ones(1, 1, 8, 8), NORM)


def test_超過晶格聲速判為無效() -> None:
    """超過即離開 LBM 低馬赫近似的適用範圍。"""
    x = _flow(jx=LATTICE_SOUND_SPEED * 1.1)
    assert "速度大小超過晶格聲速" in validity(x, torch.ones(1, 1, 8, 8), NORM)


def test_非有限值判為無效() -> None:
    """NaN 與 Inf 須在做任何統計之前先攔下。"""
    x = _flow()
    x[:, 3, 0, 0] = float("nan")
    assert "出現非有限值" in validity(x, torch.ones(1, 1, 8, 8), NORM)


def test_固體區的異常不計入() -> None:
    """檢查只看流體區，固體格點的填充值不得觸發。"""
    x = _flow()
    x[:, 0, 0, :] = -1.0
    fm = torch.ones(1, 1, 8, 8)
    fm[:, :, 0, :] = 0.0
    assert validity(x, fm, NORM) == []


# ───────────────────────── 九　`compute_terms` 的錯誤處理 ─────────────────────────

def test_compute_terms單項失敗記nan不中斷其他項(caplog: pytest.LogCaptureFixture) -> None:
    """單一項失敗須被攔下記 nan，不得讓整批計算中斷，且須記警告日誌不靜默。"""
    target = _mphy_field(seed=5)
    m = torch.ones(1, 1, SH, SW)
    with caplog.at_level(logging.WARNING, logger="tes.eval.vqi"):
        out = compute_terms(target, target, m, None, CANONICAL_TERM_SET)
    terms, _veto_ref = resolve_term_set(CANONICAL_TERM_SET)
    assert set(out) == set(terms), "缺 normalizer 時每一項仍須有回傳值（記 nan）"
    assert all(math.isnan(v) for v in out.values()), "五項皆需要 normalizer，皆應記為 nan"
    assert "計算失敗" in caplog.text


# ───────────────────────── 十　對外入口 `compute_vqi` ─────────────────────────

def test_subscores在真值上恰為正一() -> None:
    """把真值當自己的預測餵回去，五項與合成後的 vqi 皆須恰為 ``+1``。"""
    t = _mphy_field(seed=1)
    m = torch.ones(1, 1, SH, SW)
    r = compute_vqi(t, t, m, NORM, term_set=CANONICAL_TERM_SET)[0]
    assert r.invalid is False
    for name, s in r.sub.items():
        assert s == pytest.approx(1.0, abs=1e-4), f"{name} 在真值上未達 +1，得 {s}"
    assert r.vqi == pytest.approx(1.0, abs=1e-4)


def test_subscores在零場上恰為負一() -> None:
    """把零場（動量全歸零）當預測，五項與合成後的 vqi 皆須恰為 ``−1``。

    這由刻度的定義直接保證——零場本身就是 `build_anchors` 用來訂單位的那個場，
    ``raw == still`` 使 ``d = unit``，故 ``S = 1 − 2·unit/unit = −1``。
    """
    t = _mphy_field(seed=2)
    m = torch.ones(1, 1, SH, SW)
    z = zero_field(t)
    r = compute_vqi(z, t, m, NORM, term_set=CANONICAL_TERM_SET, check_validity=False)[0]
    for name, s in r.sub.items():
        assert s == pytest.approx(-1.0, abs=1e-6), f"{name} 在零場上未達 -1，得 {s}"
    assert r.vqi == pytest.approx(-1.0, abs=1e-6)


def test_逐案分數不受批次中其他案例影響() -> None:
    """刻度由該案自己的真值當場導出，同一張 target 換到批次的哪個位置都須同分。"""
    t1, t2, t3 = _mphy_field(seed=11), _mphy_field(seed=12), _mphy_field(seed=13)
    p1 = t1 + 0.3 * (zero_field(t1) - t1)     # 介於真值與零場之間，非兩端點
    p2 = t2 + 0.3 * (zero_field(t2) - t2)
    p3 = t3 + 0.3 * (zero_field(t3) - t3)
    m1 = torch.ones(1, 1, SH, SW)

    solo = compute_vqi(p1, t1, m1, NORM, term_set=CANONICAL_TERM_SET,
                       check_validity=False)[0]

    batch_pred = torch.cat([p2, p1, p3], dim=0)
    batch_target = torch.cat([t2, t1, t3], dim=0)
    batch_mask = m1.expand(3, -1, -1, -1)
    triple = compute_vqi(batch_pred, batch_target, batch_mask, NORM,
                         term_set=CANONICAL_TERM_SET, check_validity=False)[1]

    assert solo.vqi == pytest.approx(triple.vqi, abs=1e-6)
    for k in solo.sub:
        assert solo.sub[k] == pytest.approx(triple.sub[k], abs=1e-6), (
            f"{k} 隨批次位置改變：solo={solo.sub[k]} triple={triple.sub[k]}")


def test_遮罩批次為一時對整批廣播() -> None:
    """單張遮罩配多案預測是合法的，逐案評分時重複使用同一張。"""
    pred = _mphy_batch(2)
    single_mask = torch.ones(1, 1, SH, SW)
    rs = compute_vqi(pred, pred, single_mask, NORM, term_set=CANONICAL_TERM_SET,
                     check_validity=False)
    assert len(rs) == 2
    for r in rs:
        assert isinstance(r, VQIScore)
        assert r.vqi == pytest.approx(1.0, abs=1e-4)


def test_批次維度不符即報錯不靜默() -> None:
    """既非等長也非可廣播時必須報錯，不得默默對錯案。"""
    pred = _mphy_batch(3)
    bad_mask = torch.ones(2, 1, SH, SW)
    with pytest.raises(ValueError, match="批次維度"):
        compute_vqi(pred, pred, bad_mask, NORM, term_set=CANONICAL_TERM_SET,
                   check_validity=False)


def test_noise_floor長度不符即報錯() -> None:
    """`noise_floor` 須逐案給定，長度與批次不符時直接報錯。"""
    pred = _mphy_batch(2)
    m = torch.ones(2, 1, SH, SW)
    with pytest.raises(ValueError):
        compute_vqi(pred, pred, m, NORM, term_set=CANONICAL_TERM_SET,
                   noise_floor=[{}], check_validity=False)


def test_未給noise_floor時noise_ceiling為nan() -> None:
    t = _mphy_field(seed=21)
    m = torch.ones(1, 1, SH, SW)
    r = compute_vqi(t, t, m, NORM, term_set=CANONICAL_TERM_SET,
                    check_validity=False)[0]
    assert math.isnan(r.noise_ceiling)


def test_給定noise_floor時填入noise_ceiling() -> None:
    """`noise_floor` 為該案真值兩段時均窗的逐項值，登錄為 `noise_ceiling`。"""
    t = _mphy_field(seed=22)
    m = torch.ones(1, 1, SH, SW)
    half = t + 0.005 * torch.randn_like(t)
    nf = compute_terms(half, t, m, NORM, CANONICAL_TERM_SET)
    r = compute_vqi(t, t, m, NORM, term_set=CANONICAL_TERM_SET,
                    noise_floor=[nf], check_validity=False)[0]
    assert math.isfinite(r.noise_ceiling)


def test_compute_vqi對未知項組提早失敗() -> None:
    """項組名稱錯就別往下走，不得算到一半才報錯。"""
    t = _mphy_field(seed=23)
    m = torch.ones(1, 1, SH, SW)
    with pytest.raises(KeyError):
        compute_vqi(t, t, m, NORM, term_set="__no_such_term_set__")


def test_vqi恆等於逐項子分數取最小() -> None:
    """min 聚合是「視覺一票否決」的連續量對應，端到端亦不例外。"""
    t = _mphy_field(seed=24)
    m = torch.ones(1, 1, SH, SW)
    p = t + 0.5 * (zero_field(t) - t)
    r = compute_vqi(p, t, m, NORM, term_set=CANONICAL_TERM_SET,
                    check_validity=False)[0]
    assert r.vqi == pytest.approx(min(r.sub.values()))
    assert r.sub[r.binding_term] == pytest.approx(r.vqi)
    assert set(r.sub) == set(r.raw)


# ───────────────────────── 十一　舊三基線介面已完全移除 ─────────────────────────

@pytest.mark.parametrize("name", [
    "S_AT_FLOOR", "FLOOR_ANCHOR_GROUP", "FLOOR_ANCHOR_TERM",
    "WORST_ANCHOR_GROUP", "WORST_ANCHOR_TERM",
    "build_vqi", "build_vqi_from_files", "VQIScorer",
])
def test_舊三基線介面已從對外套件移除(name: str) -> None:
    """★這些符號不是仍在但未使用，而是真的從對外介面上拿掉了，import 即 ImportError。"""
    with pytest.raises(ImportError):
        exec(f"from src.tes.eval.vqi import {name}")  # noqa: S102


@pytest.mark.parametrize("name", [
    "S_AT_FLOOR", "FLOOR_ANCHOR_GROUP", "FLOOR_ANCHOR_TERM",
])
def test_舊三基線介面已從terms子模組移除(name: str) -> None:
    with pytest.raises(ImportError):
        exec(f"from src.tes.eval.vqi.terms import {name}")  # noqa: S102


@pytest.mark.parametrize("name", ["build_vqi", "build_vqi_from_files", "VQIScorer"])
def test_舊三基線介面已從score子模組移除(name: str) -> None:
    with pytest.raises(ImportError):
        exec(f"from src.tes.eval.vqi.score import {name}")  # noqa: S102


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
