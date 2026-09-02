"""物理判準量測：時均場的六項判準純量測，**不設通過門檻**。

另附一項**選配**量測——分離點 `separation_points()`，它**不在**六項判準內，
需三個無預設參數全部顯式給定才會執行，詳見下節「分層宣告」。

## 為什麼有這個模組

訓練端的質量守恆項在 canonical config 上**權重為 0**（loss term
`phys_mass_divj_absmean` 存在，但 `mean_divj_weight` 取 dataclass 預設 0.0＝未啟用，
詳見下方防自我實現第 4 點——不是「不存在」）。既有 gate 的判準都落在像素／影像層級
（mean_mse、mean_jx_ssim、StructVel/StructVort/Wake 系列 SSIM、banding 等），沒有
一項回答「這個流場在物理上像不像流場」——Strouhal 數、分離點、再接著長度、能譜斜率、
Cp、drag 這些物理量在既有評測的 metric key 清單裡完全缺席。

## 分層宣告

本 gate 量的是**時均場（mean flow）上可定義的物理量**。其中**六項已登記為判準**
（`PASS_THRESHOLDS` / `PASS_DIRECTIONS` 兩張表的鍵，六鍵）：

    回流面積比、再接著長度 Lr/L_char、尾流寬度成長率、表面 Cp、
    div(ρu) 相對 GT floor 的倍數、徑向能譜對數斜率。

**Strouhal 數與渦脫頻率不屬於本 gate**——它們是時間序列量，時均場上沒有定義，只能在
solver 層或 turb 模組上取得。

**分離點（壁剪應力的流向分量變號＝近壁流向速度反轉的位置）已實作為
`separation_points()`，但刻意不列入上面那六鍵。** 它是穩態 RANS 的標準量、在本模組
的輸入上完全可定義（為何取流向分量而非沿弧長切向，見 `_separation_sites`）。
缺的是判準定義而不是算力，故把以下三者做成無預設的必填參數（比照
`spectrum_k_*_frac` 的先例，不由實作者代選）：

- (a) 階梯狀（光柵化）邊界的法/切向估法 → `PhysicsGateConfig.separation_normal_mode`
  （合法值見 `SEPARATION_NORMAL_MODES`，已實作兩種估法供作者比對）。
- (b) 一個量體上多個變號位置報哪一個 → ``separation_select``
  （見 `SEPARATION_SELECT_MODES`；不論選哪個，**全部**站點仍完整列在 per-body 輸出）。
- (c) 逐量體結果如何彙總成 case 級純量 → ``separation_reduce``
  （見 `SEPARATION_REDUCE_MODES`）。

三者為 None（現行預設）時 `separation_points()` 直接 raise
`SeparationSpecNotDefinedError`；`physics_gate_metrics()` 在三者全空時**不量**分離點。
**分離點不進 `PASS_THRESHOLDS` / `PASS_DIRECTIONS` 兩張表**，本模組不自行登記。

Cp 與 drag 在時均場上可定義（表面壓力積分），故列入本 gate；drag 由 `surface_cp`
的壓力積分分量提供，不另設函式。

## 接線現況（別把本模組當成已生效的 gate）

名字裡的 gate 指的是「判準」，不是「已經卡在流程上的關卡」。重現指令::

    rg -n "from lbmnca\\.nca\\.tes\\.eval\\.physics_gate|from \\.physics_gate|import physics_gate" -g "*.py" -g "!src/lbmnca/nca/tes/eval/physics_gate.py"

指令尾巴那條排除規則是必要的：pattern 字串裡含字面 `import physics_gate`，不排除本檔
的話這行指令本身會被自己命中，照做會多出一筆不是 import 的命中。

生產路徑的 import 點集合、以及「其餘命中一律落在 `tests/` 下」這個性質，由
`tests/nca/tes/test_r4_countable_claims.py::test_physics_gate_production_import_points_are_two`
鎖住（數字寫在該斷言裡，本 docstring 不重寫一份）。沒有任何 gate、評測或訓練流程會因
這六項判準停下來。

## 防「判準自我實現」的結構性保證

判準隨結果調整、且判準本身被損失直接最小化，是需要防範的失效模式。以下 1-3 是本模組
實際做到的三條結構性限制，**只約束本檔、不約束整個 repo**；第 4 點是已知缺口，
說明為什麼不能宣稱六項判準全都與目標函數無關：

1. **不 import 任何 loss 模組**（import 條數與「`lbmnca.nca.tes.loss.*` 不在其中」皆由
   `tests/nca/tes/test_r4_countable_claims.py` 以 AST 機器複驗，數字不寫進本 docstring）。
2. **不接受任何可調權重參數**——`PhysicsGateConfig` 只含「量測區域怎麼取」的定義性
   參數，沒有 λ、沒有 weight，無法被最佳化器接線。
3. **通過門檻在任何模型量測之前寫死**（見 `PASS_THRESHOLDS` 的「這些值怎麼來的」段）：
   生成規則與機械產出的數字各自獨立成立，兩者皆早於首支模型判準量測，門檻不是照著
   結果挑的。門檻未填時 `evaluate_thresholds()` 仍直接 raise，不會靜默放行。
4. **已知缺口（不要宣稱本模組與目標函數完全隔離）**：六項判準中的
   `divj_over_lbm_floor` 存在一條通往損失函數的活路徑——
   `training/steps/mean_step.py` 的 `phys_mass_divj_absmean` 項由
   `mean_divj_weight` 啟用（canonical config 現值 0.0＝未啟用）。若該項被啟用，
   `divj_over_lbm_floor` 對受訓模型即**喪失獨立性**（判準被直接最小化），該情境下
   的結論必須由其餘五項判準承擔；`divj_over_lbm_floor` 只能降級為訓練目標的達成度
   回報。量測時應把當下的 `mean_divj_weight` 一併記錄，供事後判讀該次量測是否落在
   此缺口內。

## 計算圖

下列性質由 `tests/nca/tes/test_r4_countable_claims.py` 以 AST 機器複驗（項數寫在該檔的
斷言裡，本 docstring 不重寫一份）：六項判準函式、`separation_points`、
`physics_gate_metrics` 帶 `@torch.inference_mode()`；私有工具與 `evaluate_thresholds`
未裝飾但仍不建構計算圖——私有工具的呼叫點除 `_separation_sites` 內呼叫
`_outward_normals` 這一個純 numpy 巢狀對（其外層唯一呼叫點在已裝飾的
`separation_points` 內，遞移上仍在 inference_mode 下）外，全部落在已裝飾函式體內；
`evaluate_thresholds` 只做 dict / float / `np.isfinite` 比較。若日後從未裝飾的路徑
直接呼叫私有工具，這個保證即失效。

## 單位與座標約定

- 輸入 `phys_norm` 為 dataset-global norm 空間的 9ch physics 場 ``[1, 9, H, W]``；
  一切物理量在 `FieldConverter` 反正規化後計算（lattice unit）。
- `l_char_px` 取自 `p_vec[2]`，單位是 **1920×1080 sim px**；本模組依實際場寬換算成
  grid px（見 `PhysicsGateConfig.sim_px_width`）。
- `u_inf` 取自 `p_vec[1]`（`U_inlet_lu`），一律由呼叫端逐 case 傳入，非正即 `ValueError`。
- 流向為 +X（左進右出），下游 = x 增加方向。
- 陣列軸與物理軸的對應：``x`` ＝ axis 1（欄）、``y`` ＝ axis 0（列，**索引增加為 +Y**）。
  `_divj_norm_on` 的 ``∂jy/∂y = np.gradient(jy, axis=0)`` 與 `_outward_normals` /
  `_separation_sites` 的法向、切向都建立在這個對應上；`surface_cp` 的
  ``n_x = -grad_x`` 同源。此前提綁在 `FieldConverter` 的 ``vel_y = jy / rho``（無變號），
  由 `tests/nca/tes/test_r4_countable_claims.py::test_field_converter_vel_y_sign_convention`
  鎖住：慣例翻轉時該斷言會紅，這三處必須一起改。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, label

from lbmnca.nca.utils.FieldConverter import FieldConverter

from .spectral_metrics import radial_spectrum_logslope

__all__ = [
    "PASS_DIRECTIONS",
    "PASS_THRESHOLDS",
    "PhysicsGateConfig",
    "SEPARATION_NORMAL_MODES",
    "SEPARATION_REDUCE_MODES",
    "SEPARATION_SELECT_MODES",
    "SeparationSpecNotDefinedError",
    "ThresholdsNotDefinedError",
    "divj_over_lbm_floor",
    "evaluate_thresholds",
    "physics_gate_metrics",
    "recirc_area_ratio",
    "reattach_length_lchar",
    "separation_points",
    "spectrum_logslope_metric",
    "surface_cp",
    "wake_width_growth",
]

_EPS: float = 1e-8

_RHO_CH: int = 0
_JX_CH: int = 3
_JY_CH: int = 5


# ── 通過門檻 ──────────────────────────────────────────────────────────────────
PASS_THRESHOLDS: dict[str, Optional[float]] = {
    "recirc_area_ratio": 0.01016,
    "reattach_length_lchar": 0.26643,
    "wake_width_growth": 0.05319,
    "cp_mean": 0.36043,
    "divj_over_lbm_floor": 2.0,
    "spectrum_logslope": 0.31487,
}
"""六項判準的通過門檻。

鍵必須是 `physics_gate_metrics()` 實際產出的 metric key（`evaluate_thresholds()` 開頭
會斷言這件事）。表面壓力這一項以 `cp_mean` 代表——`surface_cp` 是**函式名**不是 metric
key，寫成 metric key 會在判定時 KeyError。

## 這些值怎麼來的

1. **生成規則先於任何統計凍結**：五項 ``abs_dev_from_gt`` 判準
   ``thr = max(0.5*IQR_GT, 2*u_GT*|median_GT|)``，``u_GT = 0.052``（GT 時均場取樣的
   相對不確定度）；``divj_over_lbm_floor`` 取先驗工程係數 2.0（守恆殘差不得超過同案
   GT floor 兩倍）。
2. **數字由工具機械產出、人未挑選**：在 val-warm split 的 GT 場上量測，零 model
   forward、零 ckpt 載入——看 GT 的統計不會洩露模型會偏多少，因為五項判準量的是
   「與同案 GT 的偏離量」。test split 連 GT 都沒讀。
3. **門檻的生成先於任何模型量測**，故門檻不是照著模型結果挑的。

**維護規則**：日後若要改動任一門檻，必須先說明改動的理由**不是**「因為模型沒通過」。
看著結果調門檻會讓判準喪失獨立性，是需要防範的失效模式。
"""

_VALID_DIRECTIONS: frozenset[str] = frozenset({"le", "ge", "abs_dev_from_gt"})

PASS_DIRECTIONS: dict[str, str] = {
    "recirc_area_ratio": "abs_dev_from_gt",
    "reattach_length_lchar": "abs_dev_from_gt",
    "wake_width_growth": "abs_dev_from_gt",
    "cp_mean": "abs_dev_from_gt",
    "divj_over_lbm_floor": "le",
    "spectrum_logslope": "abs_dev_from_gt",
}
"""每項判準的**比較方向**，語義如下（各方向的項數只寫在
`test_r4_countable_claims.py::test_pass_tables_stay_six_keys_without_separation`，
本檔位於 `tests/nca/tes/`）：

- ``le``: ``metric <= threshold`` 通過（越小越好）。六項中只有 `divj_over_lbm_floor`
  屬此類——它已經除過同 case 的 GT floor，1.0 即等同 GT。
- ``ge``: ``metric >= threshold`` 通過（越大越好）。目前無判準屬此類，保留為判定器的
  合法方向以免日後新增判準時被迫改寫比較邏輯。
- ``abs_dev_from_gt``: ``|metric − metrics["gt_" + key]| <= threshold`` 通過
  （**貼近 GT** 型）。其餘五項屬此類：回流面積比、Lr/L_char、尾流寬度成長率、Cp、
  能譜斜率都是「太大或太小都錯」——例如把尾流完全填平會讓
  ``recirc_area_ratio → 0``、``spectrum_logslope`` 大幅變負，單向 ``le`` 比較會把這
  種退化模型判成通過。

方向來自各判準函式 docstring 已載明的失效模式（量測語義），`evaluate_thresholds()`
不硬編任何方向。
"""


class ThresholdsNotDefinedError(RuntimeError):
    """在 `PASS_THRESHOLDS` 仍為 None 時要求判定通過與否所拋出的例外。"""


class SeparationSpecNotDefinedError(RuntimeError):
    """在三個 ``separation_*`` 定義性參數仍為 None 時要求量分離點所拋出的例外。"""


SEPARATION_NORMAL_MODES: frozenset[str] = frozenset({"mask_gradient", "axis_face"})
"""分離點的壁面法向估法（`PhysicsGateConfig.separation_normal_mode` 的合法值）。

- ``mask_gradient``: 外法線 ＝ ``−∇(solid indicator)`` 正規化（中央差分，會把 1 px
  以外的階梯資訊也平均進來，對 45° 斜面較平滑、對單格凸角較鈍）。與 `surface_cp` 的
  ``n_x = -grad_x`` 同一慣例。
- ``axis_face``: 外法線 ＝ 該環格四個軸向鄰居中「是固體」那幾面的反向單位向量和
  （純軸向面，忠實反映光柵化後的階梯，不做任何平滑）。

★兩者在階梯狀邊界上會給出不同的切向，進而可能給出不同的分離點位置。本模組
**不設預設值**，呼叫端必須顯式選一個。
"""

SEPARATION_SELECT_MODES: frozenset[str] = frozenset(
    {"most_upstream", "most_downstream"}
)
"""單一量體上出現多個變號位置時要報哪一個。

- ``most_upstream``: 取 x 最小（最上游）的那個站點。
- ``most_downstream``: 取 x 最大（最下游）的那個站點。

★無論選哪一個，**全部**站點一律完整列在 ``per_body[i]["sites"]`` 與
``separation_n_points_total``；本參數只決定「彙總用哪一個代表值」，不會讓其餘站點消失。
"""

SEPARATION_REDUCE_MODES: frozenset[str] = frozenset({"median", "mean", "min", "max"})
"""逐量體代表值如何彙總成 case 級純量。"""

_SEPARATION_REDUCERS: dict[str, Any] = {
    "median": np.median,
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
}


@dataclass
class PhysicsGateConfig:
    """物理判準的**量測定義**參數（區域怎麼取、譜怎麼擬合）。

    ★本 dataclass 不含任何損失權重。新增欄位前請確認它描述的是「量測本身的定義」，
    不是「模型該被罰多少」——後者不屬於本模組。

    徑向能譜的擬合區間三個欄位**沒有預設值**：區間選擇會直接改變斜率數值，任何隱含
    預設等同於偷偷替作者做判準決定。呼叫端必須顯式給定。

    Attributes:
        spectrum_k_lo_frac: 能譜擬合下界，佔最大波數 bin 的比例，(0, 1)。
        spectrum_k_hi_frac: 能譜擬合上界，同上，須大於下界。
        spectrum_r2_min: 擬合可信度門檻（統計性，非物理判準）；R² 低於此值標 ill_fit。
        sim_px_width: `p_vec[2]` 的 L_char 所依據的 sim 解析度寬（C-series = 1920）。
        recirc_vel_x_threshold: 判為回流的 denorm ``u_x`` 上界（lattice unit）。
        min_body_area_px: 量體連通元件的最小面積，低於此視為雜點捨去。
        exclude_border_bodies: True 時排除貼到 domain 邊界的固體連通元件
            （牆體/入口封堵不是量體，會污染 Lr 與 Cp）。
        max_bodies: 逐量體量測的上限（超過則取面積最大的前 N 個）。
        wake_u_frac: 尾流判定門檻，``|u| < wake_u_frac · u_inf`` 視為尾流內。
        wake_n_stations: 下游取樣站位數。
        wake_station_start_frac: 首站位置＝量體下游端 + 該比例 ×（至出口的距離）。
        wake_station_end_frac: 末站位置，同上。
        cp_ring_px: 表面 Cp 取樣環厚度（自固體向流體膨脹的像素數）。
        cp_ref_col_lo_frac: 參考壓力 p∞ / ρ∞ 取樣帶的**起始**欄位置（佔場寬比例）。
            預設 0.06 而非 0（貼入口）是刻意的：模型 rollout 在近入口有一條窄欄
            偽影帶（W=640 時實測尖峰落在 col≈31-32，即 frac≈0.05），GT 場沒有這條帶。
            參考帶若落進去，pred 與 GT 的 p_ref 取樣條件不對稱，會系統性偏移每一個
            Cp 與 `pressure_drag_coeff` 的 pred-vs-GT 比較。**現行預設只避開了實測
            尖峰本身，不保證避開更寬的偽影餘裕區**（別把它讀成「已完全清出偽影
            帶」）；要完全避開需更大的 `cp_ref_col_lo_frac`。前端的入口修正純屬
            render layer、不回寫模型輸出，故偽影在本模組讀到的場上仍然存在，換模型
            或換解析度前應重新確認尖峰位置。
        cp_ref_col_hi_frac: 參考壓力取樣帶的**結束**欄位置（佔場寬比例，須大於下界）。
        divj_erode_iters: divj「侵蝕版」的 fluid 侵蝕迭代數（對齊
            `warmchain_stability_probe.field_metrics` 的 iterations=2）。
        divj_wall_band_px: divj「壁帶版」的壁面帶寬（自固體膨脹的像素數）。
        separation_normal_mode: 分離點的壁面法向估法，合法值見
            `SEPARATION_NORMAL_MODES`。★``None`` ＝**尚未裁決**，不是預設值：
            `separation_points()` 讀到 None 直接 raise
            `SeparationSpecNotDefinedError`，與 `PASS_THRESHOLDS` 全 None 時
            `evaluate_thresholds()` 硬錯誤是同一手法（會改變數值的定義性選擇不由實作者
            代選）。三個 ``separation_*`` 欄位必須**同時**給定或同時留空。
        separation_select: 多個變號位置時報哪一個，合法值見 `SEPARATION_SELECT_MODES`。
        separation_reduce: 逐量體代表值的彙總方式，合法值見 `SEPARATION_REDUCE_MODES`。
    """

    # ── 無預設（必填）：擬合區間是判準定義的一部分 ──
    spectrum_k_lo_frac: float
    spectrum_k_hi_frac: float
    spectrum_r2_min: float

    # ── 有預設：幾何/區域取法 ──
    sim_px_width: int = 1920
    recirc_vel_x_threshold: float = 0.0
    min_body_area_px: int = 16
    exclude_border_bodies: bool = True
    max_bodies: int = 32
    wake_u_frac: float = 0.9
    wake_n_stations: int = 8
    wake_station_start_frac: float = 0.05
    wake_station_end_frac: float = 0.90
    cp_ring_px: int = 1
    cp_ref_col_lo_frac: float = 0.06
    cp_ref_col_hi_frac: float = 0.10
    divj_erode_iters: int = 2
    divj_wall_band_px: int = 3

    # ── ★None 不是預設值，是「尚未裁決」的哨兵：讀到 None 即硬錯誤 ──
    #    （dataclass 語法要求有預設值的欄位排在無預設欄位之後，故只能寫成 = None；
    #     語義上它與上面「有預設」那一組不同類，勿依樣新增帶真實預設的判準參數。）
    separation_normal_mode: Optional[str] = None
    separation_select: Optional[str] = None
    separation_reduce: Optional[str] = None


# ── 內部工具（純幾何/取場，無物理判定）────────────────────────────────────────


def _fluid_bool(fluid_mask: torch.Tensor) -> np.ndarray:
    """``[1, 1, H, W]`` fluid mask → ``[H, W]`` bool（1=fluid）。"""
    return (fluid_mask[0, 0] > 0.5).detach().cpu().numpy()


def _vphy_np(
    phys_norm: torch.Tensor, converter: FieldConverter
) -> dict[str, np.ndarray]:
    """norm physics 場 → denorm 的 vel_x / vel_y / vel_mag / pressure / rho / jx / jy。

    Args:
        phys_norm: ``[1, 9, H, W]`` dataset-global norm 空間 physics 場。
        converter: 綁定同一組 global stats 的 `FieldConverter`。

    Returns:
        鍵為場名、值為 ``[H, W]`` float64 numpy 陣列的字典。
    """
    vphy = converter.toVphy(phys_norm.float(), is_norm=True)
    raw = converter.normalizer.denorm_mphy(phys_norm.float())
    out = {
        "vel_x": vphy["vel_x"][0, 0],
        "vel_y": vphy["vel_y"][0, 0],
        "vel_mag": vphy["vel_mag"][0, 0],
        "pressure": vphy["pressure"][0, 0],
        "rho": raw[0, _RHO_CH],
        "jx": raw[0, _JX_CH],
        "jy": raw[0, _JY_CH],
    }
    return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in out.items()}


def _body_components(
    fluid: np.ndarray, cfg: PhysicsGateConfig
) -> list[dict[str, Any]]:
    """固體連通元件（量體）清單，已依 cfg 濾掉雜點與貼邊元件。

    Args:
        fluid: ``[H, W]`` bool，1=fluid。
        cfg: 量測定義參數。

    Returns:
        每個量體一筆 dict：``area``（px）、``cy``（形心列）、``x_max``（下游端欄）、
        ``x_min``、``mask``（``[H, W]`` bool）。依面積由大到小排序、取前 ``max_bodies`` 個。
    """
    solid = ~fluid
    lab, n = label(solid)
    h, w = fluid.shape
    bodies: list[dict[str, Any]] = []
    for i in range(1, int(n) + 1):
        m = lab == i
        area = int(m.sum())
        if area < int(cfg.min_body_area_px):
            continue
        ys, xs = np.nonzero(m)
        touches_border = bool(
            ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1
        )
        if cfg.exclude_border_bodies and touches_border:
            continue
        bodies.append(
            {
                "area": area,
                "cy": float(ys.mean()),
                "x_min": int(xs.min()),
                "x_max": int(xs.max()),
                "mask": m,
            }
        )
    bodies.sort(key=lambda b: -b["area"])
    return bodies[: int(cfg.max_bodies)]


def _l_char_grid_px(l_char_px: float, width: int, cfg: PhysicsGateConfig) -> float:
    """把 sim 解析度下的 L_char 換算成本場 grid 像素。

    Args:
        l_char_px: `p_vec[2]`，單位為 ``cfg.sim_px_width`` 解析度下的像素。
        width: 本場的欄數 W。
        cfg: 量測定義參數。

    Returns:
        grid 座標下的 L_char（像素）。

    Raises:
        ValueError: ``l_char_px`` 非正時拋出（無效的 case metadata，不可靜默通過）。
    """
    if not np.isfinite(l_char_px) or l_char_px <= 0.0:
        raise ValueError(f"L_char 非正：{l_char_px}（p_vec[2] 無效，拒絕靜默代入預設）")
    return float(l_char_px) * (float(width) / float(cfg.sim_px_width))


def _divj_norm_on(
    jx: np.ndarray, jy: np.ndarray, region: np.ndarray
) -> float:
    """指定區域上的 ``mean|∇·(ρu)| / mean‖∇(ρu)‖``（分子分母都用 j＝ρu、不除 ρ）。
    「區域怎麼取」由呼叫端決定。

    梯度在**整場**上計算後才用 ``region`` 取值：緊貼固體的格點其梯度模板會取到固體格
    的 j=0，該處的值含固/流階躍的人工項。呼叫端若給的 ``region`` 未把界面排除
    （例如未侵蝕的全流體區），回值就不是純粹的流場守恆殘差。

    中央差分對 2Δ 成分的響應與其盲區範圍，見 `divj_over_lbm_floor` 的限制第 2 條。

    Args:
        jx: ``[H, W]`` denorm 的 x 方向動量 ρu_x。
        jy: ``[H, W]`` denorm 的 y 方向動量 ρu_y。
        region: ``[H, W]`` bool，量測區域。

    Returns:
        正規化散度殘差；區域為空時回 nan。
    """
    if not region.any():
        return float("nan")
    jx_x = np.gradient(jx, axis=1)
    jx_y = np.gradient(jx, axis=0)
    jy_x = np.gradient(jy, axis=1)
    jy_y = np.gradient(jy, axis=0)
    jgm = np.sqrt(jx_x**2 + jx_y**2 + jy_x**2 + jy_y**2)
    return float(np.abs(jx_x + jy_y)[region].mean() / (jgm[region].mean() + _EPS))


def _outward_normals(
    body: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """量體表面**指向流體**的單位外法線場（只有貼壁環上的值有意義）。

    座標約定與 `_divj_norm_on` 一致：``x`` ＝ axis 1（欄，流向 +X）、``y`` ＝ axis 0
    （列，索引增加為 +Y）。

    Args:
        body: ``[H, W]`` bool，該量體的固體遮罩。
        mode: `SEPARATION_NORMAL_MODES` 之一。

    Returns:
        ``(n_x, n_y, defined)``：前兩者為 ``[H, W]`` float64 單位向量分量，
        ``defined`` 為 ``[H, W]`` bool（法線模長為 0 的格點為 False，該處切向無定義）。

    Raises:
        ValueError: ``mode`` 不在 `SEPARATION_NORMAL_MODES` 內時拋出。
    """
    if mode not in SEPARATION_NORMAL_MODES:
        raise ValueError(
            f"separation_normal_mode 非法：{mode!r}；"
            f"合法值＝{sorted(SEPARATION_NORMAL_MODES)}"
        )
    b = body.astype(np.float64)
    if mode == "mask_gradient":
        g_y, g_x = np.gradient(b)                  # ∇body 指向固體
        n_x, n_y = -g_x, -g_y                      # 取負 → 指向流體（同 surface_cp）
    else:                                          # axis_face
        west = np.zeros_like(b); west[:, 1:] = b[:, :-1]    # 西鄰為固體 → 外法線 +x
        east = np.zeros_like(b); east[:, :-1] = b[:, 1:]    # 東鄰為固體 → 外法線 −x
        north = np.zeros_like(b); north[1:, :] = b[:-1, :]  # 列−1 鄰為固體 → 外法線 +y
        south = np.zeros_like(b); south[:-1, :] = b[1:, :]  # 列+1 鄰為固體 → 外法線 −y
        n_x = west - east
        n_y = north - south
    mag = np.sqrt(n_x**2 + n_y**2)
    defined = mag > _EPS
    inv = np.where(defined, 1.0 / np.where(defined, mag, 1.0), 0.0)
    return n_x * inv, n_y * inv, defined


def _separation_sites(
    ux: np.ndarray,
    uy: np.ndarray,
    fluid: np.ndarray,
    body: np.ndarray,
    mode: str,
) -> dict[str, Any]:
    """單一量體的壁剪應力**流向分量**變號站點（純 numpy，無 torch）。

    以無滑移壁面條件把壁剪應力的**符號**化約為第一層流體格的切向速度符號：
    ``τ_w ∝ ∂u_tan/∂n ≈ (u_tan(ring) − 0) / Δn``，``Δn = 1 px > 0`` →
    ``sign(τ_w) = sign(u_tan)``。切向速度向量取 ``u_tan = u − (u·n) n``（把法向分量
    扣掉），符號則取其**流向（x）分量** ``u_tan · x̂``。

    ★為什麼取流向分量、而不是沿弧長參數的 ``u_t = u·(−n_y, n_x)``：後者的正向會隨著
    繞行輪廓而轉，均勻附著流繞過一個方塊時，上表面 ``u_t > 0``、下表面 ``u_t < 0``，
    在前/後停滯面必定各出現一次**純幾何**的變號——那不是分離。改取流向分量後，法向
    沿 ±x 的面（方塊的迎風/背風面）其 ``u_tan · x̂`` 恆為 0（不判為變號），而
    ``sign`` 變化只發生在近壁流向速度真的反轉的地方，即 2D 分離的標準判據。

    站點定義：貼壁環（``binary_dilation(body, 1) & fluid``）上，8-鄰接到相反非零符號
    的格點；把這些格點以 8-連通標號，**每個連通元件算一個站點**（否則變號處兩側的
    格點會被重複計成兩個站點），站點座標取該元件的形心。符號恰為 0 的格點（含上述
    迎風/背風面）不參與配對，故 pos 與 neg 之間隔著 0 帶時**不**算變號。

    Args:
        ux: ``[H, W]`` denorm ``u_x``。
        uy: ``[H, W]`` denorm ``u_y``。
        fluid: ``[H, W]`` bool，1=fluid。
        body: ``[H, W]`` bool，該量體的固體遮罩。
        mode: `SEPARATION_NORMAL_MODES` 之一。

    Returns:
        包含 ``sites``（每站 ``{x, y, n_px}``）、``n_ring``、``n_zero_shear``
        （流向切應力恰為 0 的環格數，含法向沿 ±x 的迎風/背風面）、
        ``n_undefined_normal``（法線無定義的環格數）的字典。

    Raises:
        ValueError: ``mode`` 非法時由 `_outward_normals` 拋出。
    """
    ring = binary_dilation(body, iterations=1) & fluid
    n_ring = int(ring.sum())
    if n_ring == 0:
        return {"sites": [], "n_ring": 0, "n_zero_shear": 0, "n_undefined_normal": 0}
    n_x, n_y, defined = _outward_normals(body, mode)
    u_n = ux * n_x + uy * n_y                       # 法向分量
    u_t = np.where(ring & defined, ux - u_n * n_x, 0.0)  # 切向速度的流向（x）分量
    pos = ring & defined & (u_t > 0.0)
    neg = ring & defined & (u_t < 0.0)
    conn8 = np.ones((3, 3), dtype=bool)            # 8-連通結構元
    marked = (pos & binary_dilation(neg, structure=conn8)) | (
        neg & binary_dilation(pos, structure=conn8)
    )
    sites: list[dict[str, float]] = []
    if marked.any():
        lab, n = label(marked, structure=conn8)
        for i in range(1, int(n) + 1):
            ys, xs = np.nonzero(lab == i)
            sites.append(
                {"x": float(xs.mean()), "y": float(ys.mean()), "n_px": float(xs.size)}
            )
    return {
        "sites": sites,
        "n_ring": n_ring,
        "n_zero_shear": int((ring & defined & (u_t == 0.0)).sum()),
        "n_undefined_normal": int((ring & ~defined).sum()),
    }


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """最小平方直線擬合，回 ``(slope, r2)``；點數不足或退化回 ``(nan, nan)``。"""
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return float("nan"), float("nan")
    xs, ys = x[ok], y[ok]
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = float("nan") if ss_tot <= _EPS else 1.0 - float(np.sum(resid**2)) / ss_tot
    return float(slope), r2


# ── 六項判準 ──────────────────────────────────────────────────────────────────


@torch.inference_mode()
def recirc_area_ratio(
    phys_norm: torch.Tensor,
    fluid_mask: torch.Tensor,
    converter: FieldConverter,
    *,
    cfg: PhysicsGateConfig,
) -> dict[str, float]:
    """判準一：回流面積比＝denorm ``u_x < 0`` 的流體像素佔全部流體像素的比例。

    時均場上回流區是幾何的直接後果（量體背風側）。比例過低表示模型把尾流「填平」，
    過高表示產生了不存在的逆流結構。

    Args:
        phys_norm: ``[1, 9, H, W]`` norm 空間 physics 場。
        fluid_mask: ``[1, 1, H, W]``，1=fluid。
        converter: 綁定同一組 global stats 的 `FieldConverter`。
        cfg: 量測定義參數。

    Returns:
        包含 ``recirc_area_ratio``（比例）與 ``n_fluid``（流體像素數）的字典。
    """
    fluid = _fluid_bool(fluid_mask)
    v = _vphy_np(phys_norm, converter)
    n_fluid = int(fluid.sum())
    if n_fluid == 0:
        return {"recirc_area_ratio": float("nan"), "n_fluid": 0.0}
    neg = v["vel_x"][fluid] < float(cfg.recirc_vel_x_threshold)
    return {"recirc_area_ratio": float(neg.mean()), "n_fluid": float(n_fluid)}


@torch.inference_mode()
def reattach_length_lchar(
    phys_norm: torch.Tensor,
    fluid_mask: torch.Tensor,
    converter: FieldConverter,
    *,
    cfg: PhysicsGateConfig,
    l_char_px: float,
) -> dict[str, Any]:
    """判準二：再接著長度 Lr / L_char（真正的長度量，非面積佔比）。

    沿每個量體形心的下游中心線取 denorm ``u_x``（不是 ``vel_mag`` —— 後者無號、
    量不出回流），找由負轉正的第一個 x（線性內插求次像素），除以 L_char。

    無回流泡（中心線首個流體格已 ``u_x >= 0``）時回 0.0 並標 ``no_recirc``；
    直到出口都未轉正時回截尾長度並標 ``truncated``——**``no_recirc`` / ``truncated``
    這兩種邊界情形逐量體都給得出有限值、不回 nan**，避免下游 `np.isfinite` 過濾把這些
    樣本靜默丟掉而虛報平均。★但這不是「本函式一律不回 nan」：`_body_components` 濾完
    後沒有任何合格量體時，``reattach_length_lchar`` 仍為 nan（空 list 無中位數）；
    全部量體都被 ``crossed_other_body`` 排除時，``reattach_length_lchar_clean``
    同樣為 nan。

    ★多量體版面的已知限制：中心線掃描遇到固體格是「跳過」而非停止，因此上游量體的
    掃描可能**穿過**下游另一棟建物才找到轉正點，量出的長度不是該量體自身的回流泡；
    量體彼此的尾流也互相干擾。這種情形以 ``crossed_other_body`` 旗標標出，並另報一
    個排除它們的中位數 ``reattach_length_lchar_clean``。**主報值
    ``reattach_length_lchar`` 仍為全部量體的中位數**（不靜默換定義），判讀時應同時看
    ``n_crossed_other_body``。本旗標只偵測「掃描路徑上、**在離開本量體並遇到第一格
    流體之後**是否再跨過固體」（實作以 ``seen_fluid`` 守衛：從 ``x_max + 1`` 起掃、
    第一格流體出現前碰到的固體不算，那通常是本量體自身的階梯狀邊緣），且不判斷該固體
    是另一棟建物還是同一量體的凹形回折。

    Args:
        phys_norm: ``[1, 9, H, W]`` norm 空間 physics 場。
        fluid_mask: ``[1, 1, H, W]``，1=fluid。
        converter: 綁定同一組 global stats 的 `FieldConverter`。
        cfg: 量測定義參數。
        l_char_px: `p_vec[2]`（sim 解析度下的特徵長度，見 `sim_px_width`）。

    Returns:
        包含以下鍵的字典：
        - ``reattach_length_lchar``: **全部**量體結果的中位數（無合格量體時為 nan）。
        - ``reattach_length_lchar_clean``: 排除 ``crossed_other_body`` 後的中位數
          （全部量體都被排除時為 nan）。
        - ``per_body``: 逐量體 dict（``lr_over_lchar`` / ``no_recirc`` /
          ``truncated`` / ``crossed_other_body`` / ``cy`` / ``x_max`` / ``area``）。
        - ``n_bodies``: 參與量測的量體數。
        - ``n_no_recirc`` / ``n_truncated`` / ``n_crossed_other_body``: 三種邊界情形
          的計數。
    """
    fluid = _fluid_bool(fluid_mask)
    v = _vphy_np(phys_norm, converter)
    ux = v["vel_x"]
    h, w = fluid.shape
    lc = _l_char_grid_px(l_char_px, w, cfg)
    bodies = _body_components(fluid, cfg)

    per_body: list[dict[str, Any]] = []
    for b in bodies:
        row = int(round(b["cy"]))
        row = int(np.clip(row, 0, h - 1))
        x0 = int(b["x_max"]) + 1
        cross_px: Optional[float] = None
        no_recirc = False
        truncated = False
        prev_x: Optional[int] = None
        prev_u: float = 0.0
        seen_neg = False
        seen_fluid = False
        crossed_other_body = False
        for x in range(x0, w):
            if not fluid[row, x]:
                if seen_fluid:                     # 已離開本量體後又遇固體 → 跨過別的量體
                    crossed_other_body = True
                continue
            seen_fluid = True
            u = float(ux[row, x])
            if u < 0.0:
                seen_neg = True
                prev_x, prev_u = x, u
                continue
            if not seen_neg:                       # 首個流體格就已順流 → 無回流泡
                no_recirc = True
                break
            if prev_x is None:                     # 理論不可達（seen_neg 蘊含 prev_x）
                no_recirc = True
                break
            frac = abs(prev_u) / (abs(prev_u) + u + _EPS)
            cross_px = float(prev_x) + frac * float(x - prev_x)
            break
        if no_recirc:
            lr = 0.0
        elif cross_px is None:                     # 到出口仍未轉正 → 截尾（右設限）
            truncated = True
            lr = float(w - 1 - b["x_max"]) / lc
        else:
            lr = float(cross_px - b["x_max"]) / lc
        per_body.append(
            {
                "lr_over_lchar": float(lr),
                "no_recirc": bool(no_recirc),
                "truncated": bool(truncated),
                "crossed_other_body": bool(crossed_other_body),
                "cy": float(b["cy"]),
                "x_max": int(b["x_max"]),
                "area": int(b["area"]),
            }
        )

    vals = [p["lr_over_lchar"] for p in per_body]
    clean = [p["lr_over_lchar"] for p in per_body if not p["crossed_other_body"]]
    return {
        "reattach_length_lchar": float(np.median(vals)) if vals else float("nan"),
        "reattach_length_lchar_clean": (
            float(np.median(clean)) if clean else float("nan")
        ),
        "per_body": per_body,
        "n_bodies": len(per_body),
        "n_no_recirc": sum(1 for p in per_body if p["no_recirc"]),
        "n_truncated": sum(1 for p in per_body if p["truncated"]),
        "n_crossed_other_body": sum(1 for p in per_body if p["crossed_other_body"]),
    }


@torch.inference_mode()
def wake_width_growth(
    phys_norm: torch.Tensor,
    fluid_mask: torch.Tensor,
    converter: FieldConverter,
    *,
    cfg: PhysicsGateConfig,
    u_inf: float,
) -> dict[str, Any]:
    """判準三：尾流寬度成長率＝尾流縱向寬度對下游距離的線性回歸斜率。

    在最下游量體之後取數個 x 站位，逐站數「該欄流體格中 ``|u| < wake_u_frac · u_inf``」
    的列數作為尾流縱向寬度，對 x 做最小平方回歸。斜率為無因次（px/px），可跨 case 比。

    Args:
        phys_norm: ``[1, 9, H, W]`` norm 空間 physics 場。
        fluid_mask: ``[1, 1, H, W]``，1=fluid。
        converter: 綁定同一組 global stats 的 `FieldConverter`。
        cfg: 量測定義參數。
        u_inf: 入口速度（`p_vec[1]`，lattice unit）。

    Returns:
        包含以下鍵的字典：
        - ``wake_width_growth``: 回歸斜率（無因次）。
        - ``wake_width_r2``: 回歸決定係數。
        - ``stations``: 逐站 ``{x, width_px}``。
        - ``n_stations``: 有效站位數。

    Raises:
        ValueError: ``u_inf`` 非正時拋出。
    """
    if not np.isfinite(u_inf) or u_inf <= 0.0:
        raise ValueError(f"u_inf 非正：{u_inf}（p_vec[1] 無效，拒絕靜默代入預設）")
    fluid = _fluid_bool(fluid_mask)
    v = _vphy_np(phys_norm, converter)
    vmag = v["vel_mag"]
    h, w = fluid.shape
    bodies = _body_components(fluid, cfg)
    if not bodies:
        return {
            "wake_width_growth": float("nan"),
            "wake_width_r2": float("nan"),
            "stations": [],
            "n_stations": 0,
        }
    x_body = max(int(b["x_max"]) for b in bodies)
    span = float(w - 1 - x_body)
    xs_f = np.linspace(
        float(cfg.wake_station_start_frac),
        float(cfg.wake_station_end_frac),
        int(cfg.wake_n_stations),
    )
    thr = float(cfg.wake_u_frac) * float(u_inf)
    stations: list[dict[str, float]] = []
    for f in xs_f:
        x = int(round(x_body + f * span))
        x = int(np.clip(x, 0, w - 1))
        col_fluid = fluid[:, x]
        if not col_fluid.any():
            continue
        width = int(np.count_nonzero(col_fluid & (vmag[:, x] < thr)))
        stations.append({"x": float(x), "width_px": float(width)})
    if len(stations) < 3:
        return {
            "wake_width_growth": float("nan"),
            "wake_width_r2": float("nan"),
            "stations": stations,
            "n_stations": len(stations),
        }
    slope, r2 = _linfit(
        np.array([s["x"] for s in stations], dtype=np.float64),
        np.array([s["width_px"] for s in stations], dtype=np.float64),
    )
    return {
        "wake_width_growth": slope,
        "wake_width_r2": r2,
        "stations": stations,
        "n_stations": len(stations),
    }


@torch.inference_mode()
def surface_cp(
    phys_norm: torch.Tensor,
    fluid_mask: torch.Tensor,
    converter: FieldConverter,
    *,
    cfg: PhysicsGateConfig,
    u_inf: float,
) -> dict[str, Any]:
    """判準四：表面壓力係數 Cp ＝ ``(p − p∞) / (0.5 ρ∞ U∞²)``。

    p 由 ch0 的 ρ 經等溫狀態方程 ``p = c_s² ρ = ρ/3`` 換算（`FieldConverter.toVphy`
    的 ``pressure``）；取樣點為緊貼固體的 1 px 流體環（LBM 的固體格本身無有效巨觀量）。
    p∞ / ρ∞ 取 ``[cp_ref_col_lo_frac, cp_ref_col_hi_frac)`` 欄段內的流體平均——**不貼
    入口**，理由見 `PhysicsGateConfig.cp_ref_col_lo_frac`（近入口偽影帶）。

    同時回報壓力積分的流向分量 ``pressure_drag_coeff``＝
    ``−Σ_ring Cp · n_x / A_frontal``（``A_frontal`` 取量體最大迎風投影高度，grid px），
    作為時均場上可定義的壓差阻力代理。

    ★符號慣例：``n_x`` 是**指向流體的外法線** x 分量（以固體指示函數的 −∂/∂x 近似），
    而流向阻力為 ``F_x = −∮ p n_x dS``，故式子前面的負號是必要的。加了負號後，迎風
    高壓（Cp>0, n_x<0）與背風低壓（Cp<0, n_x>0）都貢獻**正**阻力；合成鈍體場的重現
    設定、手算推導與實測值一律寫在
    `tests/nca/tes/test_physics_gate_surface_cp.py::test_bluff_body_drag_sign_and_magnitude`
    的斷言裡，本 docstring 不重寫一份。

    黏性摩擦分量未計，故命名為 pressure drag 而非 drag；離散環積分本身也不是精確的
    表面積分。**且多量體版面下分子與分母的量體集合不一致**：分子是全部合格量體的壓力
    合力（環由聯集膨脹而得），分母只取其中迎風投影高度最大的**單一**量體，因此該值
    只在單量體版面上具有 2D C_d 的形式。這個量只能與同法量測的 GT 相比，**不可**直接
    對文獻 C_d 宣稱數值。

    Args:
        phys_norm: ``[1, 9, H, W]`` norm 空間 physics 場。
        fluid_mask: ``[1, 1, H, W]``，1=fluid。
        converter: 綁定同一組 global stats 的 `FieldConverter`。
        cfg: 量測定義參數。
        u_inf: 入口速度（`p_vec[1]`，lattice unit）。

    Returns:
        包含 ``cp_mean`` / ``cp_max`` / ``cp_min`` / ``cp_std`` /
        ``pressure_drag_coeff`` / ``n_ring`` / ``p_ref`` / ``rho_ref`` /
        ``cp_ref_col_lo`` / ``cp_ref_col_hi``（參考帶實際欄範圍）的字典。

    Raises:
        ValueError: ``u_inf`` 非正、或 ``cp_ref_col_lo_frac`` / ``cp_ref_col_hi_frac``
            不滿足 ``0 <= lo < hi <= 1`` 時拋出。兩項檢查都在取場之前執行，**與該 case
            有沒有合格量體無關**（非法 config 一律早期失敗）。
    """
    if not np.isfinite(u_inf) or u_inf <= 0.0:
        raise ValueError(f"u_inf 非正：{u_inf}（p_vec[1] 無效，拒絕靜默代入預設）")
    # ★config 合法性檢查必須在「無量體 → 回 empty」之前：放在後面的話，沒有合格量體
    #   的 case 會靜默吞掉非法的 lo/hi 設定，最需要早期失敗的路徑反而不失敗。
    if not (0.0 <= float(cfg.cp_ref_col_lo_frac) < float(cfg.cp_ref_col_hi_frac) <= 1.0):
        raise ValueError(
            "cp 參考帶非法：需 0 <= cp_ref_col_lo_frac < cp_ref_col_hi_frac <= 1，"
            f"得到 ({cfg.cp_ref_col_lo_frac}, {cfg.cp_ref_col_hi_frac})"
        )
    fluid = _fluid_bool(fluid_mask)
    v = _vphy_np(phys_norm, converter)
    p = v["pressure"]
    rho = v["rho"]
    h, w = fluid.shape

    bodies = _body_components(fluid, cfg)
    empty = {
        "cp_mean": float("nan"),
        "cp_max": float("nan"),
        "cp_min": float("nan"),
        "cp_std": float("nan"),
        "pressure_drag_coeff": float("nan"),
        "n_ring": 0.0,
        "p_ref": float("nan"),
        "rho_ref": float("nan"),
        "cp_ref_col_lo": float("nan"),
        "cp_ref_col_hi": float("nan"),
    }
    if not bodies:
        return empty

    col_lo = int(np.clip(int(round(float(cfg.cp_ref_col_lo_frac) * w)), 0, w - 1))
    col_hi = int(np.clip(int(round(float(cfg.cp_ref_col_hi_frac) * w)), col_lo + 1, w))
    ref_region = np.zeros_like(fluid)
    ref_region[:, col_lo:col_hi] = True
    ref_region &= fluid
    if not ref_region.any():
        return empty
    p_ref = float(p[ref_region].mean())
    rho_ref = float(rho[ref_region].mean())
    q_inf = 0.5 * rho_ref * float(u_inf) ** 2
    if q_inf <= _EPS:
        return empty

    solid_bodies = np.zeros_like(fluid)
    for b in bodies:
        solid_bodies |= b["mask"]
    ring = binary_dilation(solid_bodies, iterations=int(cfg.cp_ring_px)) & fluid
    if not ring.any():
        return empty

    cp = (p - p_ref) / q_inf
    cp_ring = cp[ring]

    # 壓差阻力代理：外法線 x 分量以固體指示函數的 x 梯度近似（指向流體為正）。
    # ★分子跨量體、分母單量體：ring 由 solid_bodies（全部合格量體聯集）膨脹而來，故
    #   分子是**所有量體**的壓力合力；分母 frontal_px 卻只取**單一最大量體**的迎風
    #   投影高度。單量體版面上這就是 2D C_d 的形式（合力 / 迎風投影高度）；多量體版面
    #   上它不是任何一種常規 C_d，只能與同法量測的 GT 相比。
    # ★負號：F_x = −∮ p n_x dS（n_x 為指向流體的外法線），少了它鈍體會回負阻力。
    grad_x = np.gradient(solid_bodies.astype(np.float64), axis=1)
    n_x = -grad_x                                   # 固體→流體方向
    frontal_px = float(
        max(int(np.count_nonzero(b["mask"].any(axis=1))) for b in bodies)
    )
    drag = float(-np.sum(cp[ring] * n_x[ring]) / max(frontal_px, _EPS))

    return {
        "cp_mean": float(cp_ring.mean()),
        "cp_max": float(cp_ring.max()),
        "cp_min": float(cp_ring.min()),
        "cp_std": float(cp_ring.std()),
        "pressure_drag_coeff": drag,
        "n_ring": float(int(ring.sum())),
        "p_ref": p_ref,
        "rho_ref": rho_ref,
        "cp_ref_col_lo": float(col_lo),
        "cp_ref_col_hi": float(col_hi),
    }


@torch.inference_mode()
def divj_over_lbm_floor(
    phys_norm: torch.Tensor,
    gt_norm: torch.Tensor,
    fluid_mask: torch.Tensor,
    converter: FieldConverter,
    *,
    cfg: PhysicsGateConfig,
) -> dict[str, float]:
    """判準五：``div(ρu)`` 殘差相對**同一場 GT** 的倍數（三個區域變體）。

    LBM 的穩態質量守恆為 ``∇·(ρu) = ∇·j = 0``；離散化與量測本身有一個非零 floor，
    故絕對值無意義，必須除以同一 case 的 GT 場在同一區域下的殘差，回報倍數。

    三個變體同時輸出：

    - ``eroded``: fluid 侵蝕 ``divj_erode_iters`` 次（避固體邊界導數）。
    - ``raw``: 完全不侵蝕的全流體區——侵蝕會把壁面附近（誤差最大處）整段切掉。
    - ``wall_band``: 只取貼壁 ``divj_wall_band_px`` px 帶（
      ``binary_dilation(solid, divj_wall_band_px) & fluid``，只跟著量體走、不含域外
      框）——把壁面誤差單獨拉出來看。

    兩個必須跟著數字一起引用的限制：

    1. **``raw`` 與 ``wall_band`` 含 mask 邊緣人工項。** `_divj_norm_on` 先對**整場**
       跑 `np.gradient` 再限制到區域，梯度模板因此會看到固體格的 j=0；在固/流界面上
       這個階躍本身就產生大的分子與分母，不是流場的守恆殘差。侵蝕（``eroded``）換來
       乾淨梯度但切掉壁面附近，不侵蝕換來壁面覆蓋但混入界面階躍——兩者都只可
       **pred 與 GT 同法相比**（本函式回的倍數即是同法相比），不可當絕對守恆殘差引用。
    2. **散度算子是中央差分，對 2Δ（Nyquist）波長零響應。** `_divj_norm_on` 用
       `np.gradient`，其核 ``[-0.5, 0, 0.5]`` 的頻率響應為 ``i·sin(k)``，在 k=π
       （2 格週期的 checkerboard／grain）恰為 0——**恰好 2Δ 的守恆違反成分在本函式的
       三個變體上增益皆為 0＝完全量不到，接近 2Δ 的成分也被大幅衰減**（增益
       ``|sin k|`` 在 k→π 時趨於 0），故此處的倍數是守恆違反的下界，不是全部。
       通量式（staggered）算子對 2Δ 有非零響應，見 `tes.eval.metrics` 模組的
       `mass_conservation_residual_flux`；**本函式尚未提供通量式變體**。緩解（但不
       消除）：本函式報的是 pred/GT 倍數，盲區在分子分母上部分抵消。

    Args:
        phys_norm: ``[1, 9, H, W]`` 模型輸出（norm 空間）。
        gt_norm: ``[1, 9, H, W]`` 同 case 的 GT（norm 空間，同一 stats 空間）。
        fluid_mask: ``[1, 1, H, W]``，1=fluid。
        converter: 綁定同一組 global stats 的 `FieldConverter`。
        cfg: 量測定義參數。

    Returns:
        每個變體三個鍵：``divj_{variant}_pred`` / ``divj_{variant}_gt`` /
        ``divj_over_lbm_floor_{variant}``（倍數）。另含
        ``divj_over_lbm_floor``＝``eroded`` 變體（與既有 stage0_divj 對齊的主報值）。
    """
    fluid = _fluid_bool(fluid_mask)
    solid = ~fluid
    pv = _vphy_np(phys_norm, converter)
    gv = _vphy_np(gt_norm, converter)

    regions = {
        "eroded": binary_erosion(fluid, iterations=int(cfg.divj_erode_iters)),
        "raw": fluid,
        "wall_band": binary_dilation(solid, iterations=int(cfg.divj_wall_band_px))
        & fluid,
    }
    out: dict[str, float] = {}
    for name, region in regions.items():
        d_pred = _divj_norm_on(pv["jx"], pv["jy"], region)
        d_gt = _divj_norm_on(gv["jx"], gv["jy"], region)
        out[f"divj_{name}_pred"] = d_pred
        out[f"divj_{name}_gt"] = d_gt
        out[f"divj_over_lbm_floor_{name}"] = (
            float(d_pred / d_gt) if np.isfinite(d_gt) and d_gt > _EPS else float("nan")
        )
    out["divj_over_lbm_floor"] = out["divj_over_lbm_floor_eroded"]
    return out


@torch.inference_mode()
def spectrum_logslope_metric(
    phys_norm: torch.Tensor,
    fluid_mask: torch.Tensor,
    converter: FieldConverter,
    *,
    cfg: PhysicsGateConfig,
) -> dict[str, Any]:
    """判準六：速度量值場的徑向能譜對數斜率。

    固體區以流體平均值填補後再做 FFT（對齊
    `warmchain_stability_probe.field_metrics` 的 ``hf_vel`` 作法），避免固體邊緣的
    階躍在譜上跨大範圍波數地蓋過流場本身的能量。斜率變正 = 高頻能量過多（grain）；
    變負 = 過度耗散。

    Args:
        phys_norm: ``[1, 9, H, W]`` norm 空間 physics 場。
        fluid_mask: ``[1, 1, H, W]``，1=fluid。
        converter: 綁定同一組 global stats 的 `FieldConverter`。
        cfg: 量測定義參數（提供擬合區間，見 `PhysicsGateConfig`）。

    Returns:
        `spectral_metrics.radial_spectrum_logslope` 的回傳，鍵前綴為 ``spectrum_``。
    """
    fluid = _fluid_bool(fluid_mask)
    v = _vphy_np(phys_norm, converter)
    vel = v["vel_mag"]
    if not fluid.any():
        return {
            "spectrum_logslope": float("nan"),
            "spectrum_r2": float("nan"),
            "spectrum_ill_fit": True,
            "spectrum_n_points": 0,
        }
    filled = np.where(fluid, vel, float(vel[fluid].mean()))
    res = radial_spectrum_logslope(
        filled,
        k_lo_frac=float(cfg.spectrum_k_lo_frac),
        k_hi_frac=float(cfg.spectrum_k_hi_frac),
        r2_min=float(cfg.spectrum_r2_min),
    )
    return {
        "spectrum_logslope": res["logslope"],
        "spectrum_r2": res["r2"],
        "spectrum_ill_fit": res["ill_fit"],
        "spectrum_n_points": res["n_points"],
        "spectrum_k_lo": res["k_lo"],
        "spectrum_k_hi": res["k_hi"],
    }


# ── 選配量測：分離點（★不在 `PASS_THRESHOLDS` / `PASS_DIRECTIONS` 兩張表內）──────
#    刻意排在六項判準之後：它是本模組量得出來、但尚未被登記為判準的物理量。


@torch.inference_mode()
def separation_points(
    phys_norm: torch.Tensor,
    fluid_mask: torch.Tensor,
    converter: FieldConverter,
    *,
    cfg: PhysicsGateConfig,
    l_char_px: float,
) -> dict[str, Any]:
    """時均場上的分離點：壁剪應力**流向分量**變號位置（**不屬於六項判準**）。

    穩態/時均場上的分離點定義為壁剪應力 ``τ_w`` 變號（近壁流向速度反轉）之處。本函式
    以無滑移壁面條件把它化約為第一層流體格切向速度的**流向分量**符號（推導與「為何取
    流向分量而非沿弧長的切向」見 `_separation_sites`），沿每個量體的 1 px 貼壁環找符號
    變化，位置以 ``(x_sep − 量體最上游欄) / L_char`` 無因次化。

    **三個定義性參數沒有預設值**：`PhysicsGateConfig.separation_normal_mode`
    （階梯狀邊界的法向估法）、``separation_select``（一個量體上多個變號位置報哪一個）、
    ``separation_reduce``（逐量體如何彙總成 case 級純量）。三者為 None 時直接
    raise `SeparationSpecNotDefinedError`，不代選——與 `PASS_THRESHOLDS` 未填時
    `evaluate_thresholds()` 硬錯誤同一手法。

    **本量目前不是判準**：`PASS_THRESHOLDS` / `PASS_DIRECTIONS` 仍是六鍵，分離點只
    報數字。

    ★已知限制（要跟著數字一起引用）：
    1. 光柵化的階梯邊界上，法向與切向都只有 1 px 尺度的解析度；``mask_gradient`` 與
       ``axis_face`` 在同一個凸角上可能給出不同切向，進而給出不同的站點數。兩種估法
       的差異本身即是本量的不確定度，判讀時應至少跑過兩種比對。
    2. 站點是「符號變化的位置」，**不區分分離（attached→separated）與再接著
       （separated→attached）**；後者的物理量另由 `reattach_length_lchar` 提供。
    3. 只用第一層流體格 ⇒ 假設近壁速度剖面在 1 px 內單調。網格解析度不足的細窄縫隙
       上這個假設會失效，該處的符號可能由數值雜訊決定。

    Args:
        phys_norm: ``[1, 9, H, W]`` norm 空間 physics 場。
        fluid_mask: ``[1, 1, H, W]``，1=fluid。
        converter: 綁定同一組 global stats 的 `FieldConverter`。
        cfg: 量測定義參數（須已填三個 ``separation_*`` 欄位）。
        l_char_px: `p_vec[2]`（sim 解析度下的特徵長度，見 `sim_px_width`）。

    Returns:
        包含以下鍵的字典：
        - ``separation_x_over_lchar``: case 級純量（依 ``separation_reduce`` 彙總；
          沒有任何量體找到站點時為 nan）。
        - ``separation_n_bodies``: 參與量測的量體數。
        - ``separation_n_bodies_with_point``: 至少找到一個站點的量體數。
        - ``separation_n_points_total``: 全部量體的站點總數（不受 ``select`` 影響）。
        - ``separation_per_body``: 逐量體 dict（``sites`` 完整列出、``selected_x``、
          ``selected_x_over_lchar``、``n_ring`` / ``n_zero_shear`` /
          ``n_undefined_normal``、``x_min`` / ``area``）。
        - ``separation_normal_mode`` / ``separation_select`` / ``separation_reduce``:
          本次量測實際使用的定義（字串，寫進證據 JSON 供事後核對）。

    Raises:
        SeparationSpecNotDefinedError: 三個 ``separation_*`` 欄位未全部填妥時拋出。
        ValueError: 任一 ``separation_*`` 值不在對應的合法集合內、或 ``l_char_px``
            非正時拋出。
    """
    spec = {
        "separation_normal_mode": cfg.separation_normal_mode,
        "separation_select": cfg.separation_select,
        "separation_reduce": cfg.separation_reduce,
    }
    unset = sorted(k for k, v in spec.items() if v is None)
    if unset:
        raise SeparationSpecNotDefinedError(
            f"分離點的定義性參數尚未由作者裁決：{unset}。"
            "三者會改變數值，刻意不給預設；請顯式指定"
            f"（合法值：normal_mode={sorted(SEPARATION_NORMAL_MODES)}、"
            f"select={sorted(SEPARATION_SELECT_MODES)}、"
            f"reduce={sorted(SEPARATION_REDUCE_MODES)}）。"
        )
    normal_mode = str(cfg.separation_normal_mode)
    select = str(cfg.separation_select)
    reduce_mode = str(cfg.separation_reduce)
    # ★三個值的合法性一律在取場之前檢查（與 `surface_cp` 的 cp 參考帶同一紀律）：
    #   放到迴圈裡的話，沒有合格量體的 case 會靜默吞掉非法設定，最需要早期失敗的
    #   路徑反而不失敗。normal_mode 在 `_outward_normals` 內另有一道相同的檢查。
    if normal_mode not in SEPARATION_NORMAL_MODES:
        raise ValueError(
            f"separation_normal_mode 非法：{normal_mode!r}；"
            f"合法值＝{sorted(SEPARATION_NORMAL_MODES)}"
        )
    if select not in SEPARATION_SELECT_MODES:
        raise ValueError(
            f"separation_select 非法：{select!r}；"
            f"合法值＝{sorted(SEPARATION_SELECT_MODES)}"
        )
    if reduce_mode not in SEPARATION_REDUCE_MODES:
        raise ValueError(
            f"separation_reduce 非法：{reduce_mode!r}；"
            f"合法值＝{sorted(SEPARATION_REDUCE_MODES)}"
        )

    fluid = _fluid_bool(fluid_mask)
    v = _vphy_np(phys_norm, converter)
    ux, uy = v["vel_x"], v["vel_y"]
    _, w = fluid.shape
    lc = _l_char_grid_px(l_char_px, w, cfg)
    bodies = _body_components(fluid, cfg)

    per_body: list[dict[str, Any]] = []
    selected: list[float] = []
    n_points_total = 0
    picker = min if select == "most_upstream" else max
    for b in bodies:
        res = _separation_sites(ux, uy, fluid, b["mask"], normal_mode)
        sites = res["sites"]
        n_points_total += len(sites)
        if sites:
            pick = picker(sites, key=lambda s: s["x"])
            sel_x = float(pick["x"])
            sel_norm = float(sel_x - float(b["x_min"])) / lc
            selected.append(sel_norm)
        else:
            sel_x = float("nan")
            sel_norm = float("nan")
        per_body.append(
            {
                "sites": [
                    {
                        "x": s["x"],
                        "y": s["y"],
                        "n_px": s["n_px"],
                        "x_over_lchar": float(s["x"] - float(b["x_min"])) / lc,
                    }
                    for s in sites
                ],
                "n_sites": len(sites),
                "selected_x": sel_x,
                "selected_x_over_lchar": sel_norm,
                "n_ring": res["n_ring"],
                "n_zero_shear": res["n_zero_shear"],
                "n_undefined_normal": res["n_undefined_normal"],
                "x_min": int(b["x_min"]),
                "area": int(b["area"]),
            }
        )

    return {
        "separation_x_over_lchar": (
            float(_SEPARATION_REDUCERS[reduce_mode](np.asarray(selected)))
            if selected
            else float("nan")
        ),
        "separation_n_bodies": len(per_body),
        "separation_n_bodies_with_point": len(selected),
        "separation_n_points_total": n_points_total,
        "separation_per_body": per_body,
        "separation_normal_mode": normal_mode,
        "separation_select": select,
        "separation_reduce": reduce_mode,
    }


# ── 彙總 ──────────────────────────────────────────────────────────────────────


@torch.inference_mode()
def physics_gate_metrics(
    phys_norm: torch.Tensor,
    gt_norm: torch.Tensor,
    fluid_mask: torch.Tensor,
    converter: FieldConverter,
    *,
    cfg: PhysicsGateConfig,
    l_char_px: float,
    u_inf: float,
) -> dict[str, Any]:
    """一次跑完六項判準，回傳扁平的 metric 字典（不判定通過與否）。

    分離點（`separation_points`）**只在三個 ``separation_*`` 定義性參數全部填妥時
    才一併量測**；未填時本函式不量分離點（現行預設即此路徑）。三者填妥時會另加
    ``separation_*`` 與 ``gt_separation_*`` 系列鍵。**分離點不進 `PASS_THRESHOLDS`**，
    故不影響 `evaluate_thresholds`。部分填妥（例如只給 normal_mode）視為設定錯誤，
    直接讓 `SeparationSpecNotDefinedError` 冒出來，不靜默降級成不量。

    Args:
        phys_norm: ``[1, 9, H, W]`` 模型輸出（norm 空間）。
        gt_norm: ``[1, 9, H, W]`` 同 case 的 GT（norm 空間）。
        fluid_mask: ``[1, 1, H, W]``，1=fluid。
        converter: 綁定同一組 global stats 的 `FieldConverter`。
        cfg: 量測定義參數。
        l_char_px: `p_vec[2]`。
        u_inf: `p_vec[1]`。

    Returns:
        六項判準的合併字典；``per_body`` / ``stations`` 等結構欄位一併保留。
        另含 ``gt_*`` 前綴的**五項**單場量（回流面積比、Lr/L_char、尾流寬度成長率、
        Cp、能譜斜率在 GT 場上的同法量測）——沒有 GT 對照就無法判斷這些物理量是否
        合理，`evaluate_thresholds` 的 ``abs_dev_from_gt`` 方向也依賴這些鍵。
        `divj_over_lbm_floor` 本身已是 pred/GT 倍數，不另出 ``gt_`` 版本。
        分離點若啟用，pred 側與 GT 側同法各量一次（``separation_*`` /
        ``gt_separation_*``）。

    Raises:
        SeparationSpecNotDefinedError: 三個 ``separation_*`` 欄位只填了一部分時拋出
            （全部留空＝不量分離點，不算錯誤）。
    """
    sep_spec = (
        cfg.separation_normal_mode,
        cfg.separation_select,
        cfg.separation_reduce,
    )
    n_sep_set = sum(1 for s in sep_spec if s is not None)
    if n_sep_set not in (0, len(sep_spec)):
        raise SeparationSpecNotDefinedError(
            "三個 separation_* 定義性參數必須同時給定或同時留空，"
            f"目前只給了 {n_sep_set}/{len(sep_spec)} 個：{sep_spec}"
        )
    with_separation = n_sep_set == len(sep_spec)
    out: dict[str, Any] = {}
    out.update(recirc_area_ratio(phys_norm, fluid_mask, converter, cfg=cfg))
    out.update(
        reattach_length_lchar(
            phys_norm, fluid_mask, converter, cfg=cfg, l_char_px=l_char_px
        )
    )
    out.update(
        wake_width_growth(phys_norm, fluid_mask, converter, cfg=cfg, u_inf=u_inf)
    )
    out.update(surface_cp(phys_norm, fluid_mask, converter, cfg=cfg, u_inf=u_inf))
    out.update(divj_over_lbm_floor(phys_norm, gt_norm, fluid_mask, converter, cfg=cfg))
    out.update(spectrum_logslope_metric(phys_norm, fluid_mask, converter, cfg=cfg))
    if with_separation:
        out.update(
            separation_points(
                phys_norm, fluid_mask, converter, cfg=cfg, l_char_px=l_char_px
            )
        )

    gt: dict[str, Any] = {}
    gt.update(recirc_area_ratio(gt_norm, fluid_mask, converter, cfg=cfg))
    gt.update(
        reattach_length_lchar(
            gt_norm, fluid_mask, converter, cfg=cfg, l_char_px=l_char_px
        )
    )
    gt.update(wake_width_growth(gt_norm, fluid_mask, converter, cfg=cfg, u_inf=u_inf))
    gt.update(surface_cp(gt_norm, fluid_mask, converter, cfg=cfg, u_inf=u_inf))
    gt.update(spectrum_logslope_metric(gt_norm, fluid_mask, converter, cfg=cfg))
    if with_separation:
        gt.update(
            separation_points(
                gt_norm, fluid_mask, converter, cfg=cfg, l_char_px=l_char_px
            )
        )
    for k, val in gt.items():
        out[f"gt_{k}"] = val
    return out


def evaluate_thresholds(metrics: dict[str, Any]) -> dict[str, bool]:
    """依 `PASS_THRESHOLDS` 與 `PASS_DIRECTIONS` 判定通過與否。

    比較方式逐項由 `PASS_DIRECTIONS` 決定（``le`` / ``ge`` / ``abs_dev_from_gt``），
    不對全部判準硬編同一個方向——單向比較會把「把尾流完全填平」這種退化解判成通過。

    非有限值（nan / inf）一律判為 **False**：量不出來不等於通過。

    Args:
        metrics: `physics_gate_metrics` 的回傳（``abs_dev_from_gt`` 方向的判準另需
            對應的 ``gt_*`` 鍵，`physics_gate_metrics` 已一併輸出）。

    Returns:
        每項判準的通過旗標，鍵同 `PASS_THRESHOLDS`。

    Raises:
        ThresholdsNotDefinedError: `PASS_THRESHOLDS` 尚有 None（現行狀態）時一律拋出。
            這是刻意的——「門檻未定義」必須是硬錯誤，不能靜默退化成全部通過。
        ValueError: 兩張表的鍵不一致、方向值非法、或 `metrics` 缺少某項判準所需的鍵
            （含 ``abs_dev_from_gt`` 需要的 ``gt_*``）。缺鍵在此明確報錯，不讓裸
            KeyError 從比較式裡漏出來。
    """
    if set(PASS_THRESHOLDS) != set(PASS_DIRECTIONS):
        raise ValueError(
            "PASS_THRESHOLDS 與 PASS_DIRECTIONS 的鍵集合不一致："
            f"{sorted(set(PASS_THRESHOLDS) ^ set(PASS_DIRECTIONS))}"
        )
    bad_dir = {k: d for k, d in PASS_DIRECTIONS.items() if d not in _VALID_DIRECTIONS}
    if bad_dir:
        raise ValueError(
            f"PASS_DIRECTIONS 含非法方向 {bad_dir}；合法值＝{sorted(_VALID_DIRECTIONS)}"
        )
    missing = [k for k, v in PASS_THRESHOLDS.items() if v is None]
    if missing:
        raise ThresholdsNotDefinedError(
            "物理判準的通過門檻有缺項："
            f"{missing}。六項門檻已由作者批准（見 PASS_THRESHOLDS "
            "docstring）；出現 None 表示有人清空了它。要重新定義門檻，"
            "必須在看到任何模型數字之前寫死。"
        )

    required: list[str] = []
    for k, direction in PASS_DIRECTIONS.items():
        required.append(k)
        if direction == "abs_dev_from_gt":
            required.append(f"gt_{k}")
    absent = [k for k in required if k not in metrics]
    if absent:
        raise ValueError(
            f"metrics 缺少判定所需的鍵：{absent}。"
            "PASS_THRESHOLDS/PASS_DIRECTIONS 的鍵必須是 physics_gate_metrics() 實際"
            "產出的 metric key。"
        )

    out: dict[str, bool] = {}
    for k, thr_opt in PASS_THRESHOLDS.items():
        thr = float(thr_opt)                       # 上面已排除 None
        direction = PASS_DIRECTIONS[k]
        val = float(metrics[k])
        if not np.isfinite(val):
            out[k] = False
            continue
        if direction == "le":
            out[k] = bool(val <= thr)
        elif direction == "ge":
            out[k] = bool(val >= thr)
        else:                                      # abs_dev_from_gt
            ref = float(metrics[f"gt_{k}"])
            out[k] = bool(np.isfinite(ref) and abs(val - ref) <= thr)
    return out
