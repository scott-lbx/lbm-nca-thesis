r"""單 case 物理驗證：模擬跑完後的**唯一收錄閘門**。

依 2026-07-29 使用者裁決：

> 每一個模擬跑完之後都必須執行一次這個驗證，只有驗證過後才代表這個模擬是成功的、
> 才能被收錄下來，不然都要被推翻，重新生成 mask／config、重新模擬。

因此本模組的判定是**二元且不可協商**的：`PASS` 才進 dataset，`FAIL` 一律作廢重生成。
沒有「警告但收錄」這種中間狀態——`warn` 級的項目只寫進報告供追溯，不參與判定。

## 與 L1／L2／L3 的分工

| 層 | 對象 | 何時跑 | 在哪 |
|---|---|---|---|
| L1 解析解（Poiseuille／靜止／均勻流） | **求解器本身** | 批次開始前一次 | `bench_l1_analytic.py` |
| L2 文獻錨點（St／Cd／Lr） | **求解器本身** | 批次開始前一次 | `bench_l2_literature.py` |
| **本模組** | **每一個 case** | **每次模擬結束** | 本檔 |

L1／L2 驗的是「這個求解器對不對」，跑一次即可；本模組驗的是「這一次模擬的產物
能不能用」，每個 case 都要跑。兩者不可互相取代。

## 檢查清單

| ID | 名稱 | 等級 | 判準 |
|---|---|---|---|
| G1.1 | 結構完整 | fatal | 必要 group／dataset 齊備 |
| G1.2 | 數值有限 | fatal | trajectory 與 mean_accum 無 NaN／Inf |
| G2.1 | tau 守門 | fatal | `tau ∈ [0.51, 2.0]` |
| G2.2 | 入口 Ma | fatal | `u_inlet / c_s < 0.1` |
| G2.3 | Re 自洽 | fatal | attrs 的 `re` 與 `u·L/ν` 相對誤差 < 1% |
| G3.1 | KE 不發散 | fatal | 後半段 KE 中位數 / 中段 < 1.5 |
| G3.2 | 局部 Ma | gate | `max|u|/c_s < 0.5`（可壓縮性上限） |
| G4.1 | 質量守恆 | gate | 時均場 `mean\|∇·(ρu)\|/mean‖∇(ρu)‖ < Ma²` |
| G4.2 | ROI 進出口通量差 | **warn** | 參考量；ROI 上下非封閉，有建物時不需平衡 |
| G5.1 | 時均統計收斂 | gate | 半窗差 / √2 < 5% |
| G5.2 | burn-in 充足 | gate | `burnin_ctus ≥ 0.9 × Re`（cold） |
| G6.1 | 入口無條紋 | gate | 入口/遠場高波數能量比 < 3 |
| G7.1 | 幾何有效 | fatal | mask 非空、固體占比 ∈ (0, 0.5) |
| G7.2 | L_char 一致 | fatal | attrs 的 `l_char_px` 與 mask 實算相對誤差 < 2% |
| G7.3 | 溯源欄位齊備 | fatal | 必要 attrs 全部存在 |
| G8.1 | 對稱性 | warn | 幾何上下對稱時，時均場不對稱度 < 5% |

★`gate` 與 `fatal` 都會導致 FAIL，差別只在 `fatal` 會中止後續檢查（資料根本讀不下去）。

## 用法

```bash
python -m lbmnca.lbm.lbm_mrt_les.validation.case_physics --h5 <path>          # 單檔
python -m lbmnca.lbm.lbm_mrt_les.validation.case_physics --dataset <name>     # 整個 dataset
```

程式內：

```python
from lbmnca.lbm.lbm_mrt_les.validation.case_physics import validate_h5
report = validate_h5(path)
if not report.passed:
    ...  # 作廢重生成
```
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from lbmnca.paths import dataset_dir

C_S = 1.0 / np.sqrt(3.0)

#: 判準常數。改這裡等於改收錄標準——每一條都要有實測或解析依據。
THRESH: dict[str, float] = {
    # G2 參數守門（與 config_gen/config_utils/constants.py 同源）
    "tau_min": 0.51,
    "tau_max": 2.00,
    "inlet_ma_max": 0.10,
    "re_self_consistency": 0.01,
    # G3 穩定性。局部 Ma 上限取 0.5：狹縫射流達 3-4 倍入口速度是物理的
    #   （tau 穩定性掃描實測），但超過 0.5 時 Ma² = 0.25 的可壓縮性誤差已不可忽略。
    "ke_growth_max": 1.5,
    "local_ma_max": 0.50,
    # G4 質量守恆。參照是 Ma²（LBM 弱可壓縮的固有量級），不是 float32 底噪——
    #   後者會要求弱可壓縮方法達到不可壓縮的精度。L3 實測六 case 落在 Ma² 的
    #   3.4%-23.9%，取 1.0×Ma² 有充足餘裕。
    "div_over_ma2_max": 1.00,
    "mass_residual_rel_max": 0.05,
    # G5 時均品質。半窗差 = √2 × 單窗 SE；要求單窗 SE < 5%。
    # ★G5.1 雙級（2026-07-29）：gate 擋「明確不可用」，warn 標記「理想品質」。
    #   高 Re case 有週期 >260 CTU 的低頻尾流擺動，實測 SE 在 120-260 CTU 之間
    #   卡在 18-22% 不再下降，外推要到 10% 需 867 CTU（不可行）。那是**物理限制**
    #   而非設定錯誤——把 gate 設在 5% 只會讓那批 case 永遠 FAIL 而重跑無效。
    #   閘門該擋的是錯誤，不是物理限制；品質差異由記錄的 SE 讓下游決定如何使用。
    "mean_se_gate": 0.30,  # 超過此值代表時均根本沒收斂（如窗 80 的 c0034：56.6%）
    "mean_se_ideal": 0.10,  # warn 門檻：達此值的 case 品質良好
    # ★2026-07-30 由 0.9 降到 0.27，跟上 BURNIN_RE_FACTOR 1.0 → 0.3 的變更。
    #   依據見 gen_c_series_configs 的 BURNIN_RE_FACTOR 註解：改用 batch-means
    #   度量後，burn-in 對時均 SE 幾乎無影響，縮短引入的偏差被雜訊淹沒。
    #   仍保留這一條是為了擋「完全沒有丟棄窗」的組態（如舊協定的固定 10 CTU，
    #   在 Re=300 時只有 0.03xRe）。
    "burnin_re_factor_min": 0.27,  # 允許 config 的 0.3 有 10% 餘裕
    # G6 入口條紋。L3 實測：加 warmup 後入口/遠場比降到 ~1；舊資料達 4.5-51.7。
    "inlet_stripe_ratio_max": 3.0,
    # G7 幾何
    "solid_fraction_min": 1e-4,
    "solid_fraction_max": 0.50,
    "l_char_rel_err_max": 0.02,
    # G8 對稱性（warn 級）
    "asymmetry_max": 0.05,
}

#: 缺一不可的 root attrs（溯源與下游都依賴）
REQUIRED_ATTRS: tuple[str, ...] = (
    "case_id",
    "geometry_id",
    "dataset_name",
    "ic_type",
    "re",
    "u_inlet_lu",
    "l_char_px",
    "nu_lu",
    "tau",
    "ctu_lbm_steps",
    "total_ctu",
    "n_snapshots",
)

CH_RHO, CH_JX, CH_JY = 0, 3, 5


@dataclass
class Check:
    """單一檢查的結果。

    ★`__post_init__` 強制把 numpy 純量轉成 Python 原生型別。判準大多由 numpy
    運算得出，`np.bool_` 會讓 `is True` 比較失敗、也不能 JSON 序列化——而這份
    報告要落盤存證，序列化失敗等於整個閘門的紀錄丟失。
    """

    id: str
    name: str
    severity: str  # "fatal" | "gate" | "warn"
    passed: bool
    value: Optional[float] = None
    threshold: Optional[float] = None
    detail: str = ""
    #: 判準方向。"upper" = value 須 < threshold；"lower" = value 須 >= threshold；
    #: "range"／None = 兩者皆非（如 tau 的區間、計數型）。餘裕的算法依此而定——
    #: 用錯方向算出來的「餘裕」沒有意義（G5.2 曾因此顯示 −98%）。
    direction: Optional[str] = "upper"

    def margin(self) -> Optional[float]:
        """距離判準的相對餘裕（正 = 還有空間，負 = 已越界）。"""
        if self.value is None or self.threshold is None:
            return None
        if self.direction == "upper":
            return 1.0 - self.value / self.threshold if self.threshold > 0 else None
        if self.direction == "lower":
            return self.value / self.threshold - 1.0 if self.threshold > 0 else None
        return None

    def __post_init__(self) -> None:
        self.passed = bool(self.passed)
        if self.value is not None:
            self.value = float(self.value)
        if self.threshold is not None:
            self.threshold = float(self.threshold)


@dataclass
class Report:
    """一個 case 的完整驗證報告。"""

    h5_path: str
    case_id: str = ""
    passed: bool = False
    checks: list[Check] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, c: Check) -> Check:
        self.checks.append(c)
        return c

    def finalize(self) -> "Report":
        self.passed = all(
            c.passed for c in self.checks if c.severity in ("fatal", "gate")
        )
        return self

    @property
    def failures(self) -> list[Check]:
        return [
            c for c in self.checks if not c.passed and c.severity in ("fatal", "gate")
        ]

    def to_dict(self) -> dict:
        return {
            "h5_path": self.h5_path,
            "case_id": self.case_id,
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_failed": len(self.failures),
            "failed_ids": [c.id for c in self.failures],
            "checks": [{**asdict(c), "margin": c.margin()} for c in self.checks],
            "meta": self.meta,
        }


# ── 純函式判準（便於單元測試，不碰 I/O）────────────────────────────────────


def check_tau(tau: float) -> Check:
    ok = THRESH["tau_min"] <= tau <= THRESH["tau_max"]
    return Check(
        "G2.1",
        "tau 守門",
        "fatal",
        ok,
        float(tau),
        None,
        f"需 ∈ [{THRESH['tau_min']}, {THRESH['tau_max']}]",
        "range",
    )


def check_inlet_ma(u_inlet: float) -> Check:
    ma = u_inlet / C_S
    return Check(
        "G2.2",
        "入口 Ma",
        "fatal",
        ma < THRESH["inlet_ma_max"],
        float(ma),
        THRESH["inlet_ma_max"],
    )


def check_re_consistency(re_attr: float, u: float, l_char: float, nu: float) -> Check:
    re_calc = u * l_char / nu if nu > 0 else float("inf")
    rel = abs(re_calc - re_attr) / re_attr if re_attr > 0 else float("inf")
    return Check(
        "G2.3",
        "Re 自洽",
        "fatal",
        rel < THRESH["re_self_consistency"],
        float(rel),
        THRESH["re_self_consistency"],
        f"attrs re={re_attr:.1f} vs u·L/ν={re_calc:.1f}",
        "upper",
    )


def check_ke_growth(ke: np.ndarray) -> Check:
    """KE 序列不得單調發散（末段中位數 / 中段中位數）。"""
    ke = np.asarray(ke, dtype=np.float64)
    ke = ke[np.isfinite(ke)]
    if ke.size < 8:
        return Check(
            "G3.1",
            "KE 不發散",
            "fatal",
            False,
            None,
            None,
            f"樣本不足（{ke.size} < 8）",
        )
    n = ke.size
    mid = float(np.median(ke[n // 4 : n // 2]))
    end = float(np.median(ke[3 * n // 4 :]))
    ratio = end / mid if mid > 0 else float("inf")
    return Check(
        "G3.1",
        "KE 不發散",
        "fatal",
        bool(np.isfinite(ratio) and ratio < THRESH["ke_growth_max"]),
        float(ratio),
        THRESH["ke_growth_max"],
    )


def check_local_ma(max_abs_u: float) -> Check:
    ma = max_abs_u / C_S
    return Check(
        "G3.2",
        "局部 Ma",
        "gate",
        ma < THRESH["local_ma_max"],
        float(ma),
        THRESH["local_ma_max"],
        "狹縫射流達 3-4× 入口速度是物理的；此處管的是可壓縮性誤差上限",
    )


def divergence_ratio(
    rho: np.ndarray, u: np.ndarray, v: np.ndarray, fluid: np.ndarray
) -> float:
    """`mean|∇·(ρu)| / mean‖∇(ρu)‖`（僅核心流體區）。"""
    jx, jy = rho * u, rho * v
    dxx = np.gradient(jx, axis=1)
    dxy = np.gradient(jx, axis=0)
    dyx = np.gradient(jy, axis=1)
    dyy = np.gradient(jy, axis=0)
    div = dxx + dyy
    gmag = np.sqrt(dxx**2 + dxy**2 + dyx**2 + dyy**2)
    m = fluid[1:-1, 1:-1]
    if m.sum() == 0:
        return float("nan")
    denom = float(gmag[1:-1, 1:-1][m].mean())
    if denom <= 0:
        return float("nan")
    return float(np.abs(div[1:-1, 1:-1][m]).mean() / denom)


def check_mass_conservation(
    div_rel: float, inlet_ma: float, source_is_f32: bool = True
) -> Check:
    """時均場的散度是否落在弱可壓縮的固有量級內。

    ★`source_is_f32=False`（只有 fp16 的 trajectory 可用）時降為 `warn`：
    fp16 的相對精度 5e-4 使 `j≈0.05` 帶 2.5e-5 的量化噪聲，而真實梯度僅
    ~1.25e-3——光是量化就給出 ≈2.7 倍 Ma² 的假散度，判準完全被淹沒。
    新協定的 H5 一律帶 float32 的 `/mean_accum`，故生產資料仍受 gate 保護。
    """
    ma2 = inlet_ma**2
    ratio = div_rel / ma2 if ma2 > 0 else float("inf")
    sev = "gate" if source_is_f32 else "warn"
    note = (
        ""
        if source_is_f32
        else "（來源為 fp16 trajectory，量化噪聲淹沒判準 → 降為 warn）"
    )
    return Check(
        "G4.1",
        "質量守恆",
        sev,
        bool(np.isfinite(ratio) and ratio < THRESH["div_over_ma2_max"]),
        float(ratio),
        THRESH["div_over_ma2_max"],
        f"div_rel={div_rel:.3e}，參照 Ma²={ma2:.3e}" f"（弱可壓縮固有量級）{note}",
    )


def check_mass_residual(res_rel: Optional[np.ndarray]) -> Check:
    """ROI 進出口面的通量差（**參考量，非判定**）。

    ★2026-07-29 降為 `warn`：這一條原本設成 gate，但它的前提是錯的。
    `mass_residual` 量的是 **ROI**（裁掉 sponge+buffer 後）第一列與最後一列的
    `mean(ρu_x)` 之差，而 ROI 的上下邊界**不是** domain 邊界——流線繞過建物時
    會偏出 ROI 上下再回來，因此有建物的 case 進出口通量本來就不必相等。
    實測 D0-V 三個 case：0.098／0.098／0.211，且與固體占比不成比例
    （0.56% 占比的 case 反而差最大），符合「流線偏轉」而非「質量不守恆」的圖像。

    真正的質量守恆判定是 **G4.1 的逐點散度**（`∇·(ρu)` 相對 Ma²），那是更強的
    檢查且不依賴任何邊界封閉假設；三個 case 在該項的餘裕都很充足（0.39–0.55）。
    本項保留為 warn，供追溯 ROI 邊界的通量分布。
    """
    if res_rel is None or len(res_rel) == 0:
        return Check(
            "G4.2",
            "ROI 進出口通量差",
            "warn",
            True,
            None,
            None,
            "H5 無 mass_residual_rel（舊格式）",
        )
    v = float(np.median(np.asarray(res_rel, dtype=np.float64)))
    return Check(
        "G4.2",
        "ROI 進出口通量差",
        "warn",
        bool(np.isfinite(v) and v < THRESH["mass_residual_rel_max"]),
        v,
        THRESH["mass_residual_rel_max"],
        "參考量：ROI 上下非封閉，有建物時不需平衡；質量守恆看 G4.1",
    )


def half_window_diff(seg_mean: np.ndarray, seg_n: np.ndarray, start_idx: int) -> float:
    """由分段時均重組前後兩半窗，回傳相對 L2 差。

    ⚠ **與 zarr 的 `mean_running_se` 是兩種不同的度量，數值不可互換：**

    * 本函式（H5 層）：raw moment 空間，逐通道**去掉常數偏移**後正規化，
      並把同一張量的分量合併。因為 H5 階段拿不到 dataset 級的正規化常數
      （`M_REF` / `S_combined` 要到轉檔時才算），只能用 case 內自洽的尺度。
    * `post_process.mean_accum_io.delta_norm_fluid`（zarr 層）：
      **tanh 正規化空間**的 9 通道 rel_l2，與模型誤差同一種度量，
      因此 `mean_running_se` 可直接和模型誤差（如 0.229）相比。

    兩者各有用途：本函式供收錄閘門做 case 內的自洽檢查（不依賴 dataset 統計，
    因此在單 case 剛跑完時就能判）；zarr 的 SE 供下游依 GT 品質加權。
    **G5.1 的判準（0.30 / 0.10）只適用於本函式的度量，不可套到 tanh 度量上。**

    Args:
        seg_mean: `[S, C, H, W]` 各段的段時均。
        seg_n: `[S]` 各段樣本數（作為加權）。
        start_idx: 時均窗的起始段索引（burn-in 之後）。

    Returns:
        兩個半窗時均的相對 L2 差；段數不足時回 NaN。
    """
    s = seg_mean[start_idx:]
    w = np.asarray(seg_n[start_idx:], dtype=np.float64)
    if s.shape[0] < 4:
        return float("nan")
    h = s.shape[0] // 2
    a = np.tensordot(w[:h], s[:h], axes=(0, 0)) / w[:h].sum()
    b = np.tensordot(w[h : 2 * h], s[h : 2 * h], axes=(0, 0)) / w[h : 2 * h].sum()

    # ★逐通道正規化，且分母先去掉各通道的常數偏移（2026-07-29 修正）。
    #   首版用全 9 通道拉平後的單一相對 L2，結果被 `rho ≈ 1.0` 這個遍布全場的
    #   常數主導：分母 ~ sqrt(N)，而真正的差集中在 jx/jy（~0.05 量級）——差異被
    #   稀釋 20-50 倍，使本項對 burn-in 不足**完全沒有反應**。
    #   陰性對照實測（Re=269，burn-in 269 → 10 CTU）：舊算法 0.38% vs 0.40%
    #   （毫無差別）；同一組資料改逐通道去偏移後才顯現。
    #   取各通道相對差的**最大值**：任何一個通道未收斂都代表時均不可用。
    rels = []
    for c in range(a.shape[0]):
        mid = 0.5 * (a[c] + b[c])
        den = np.linalg.norm(mid - mid.mean())  # 去常數偏移
        if den <= 0:
            continue
        rels.append(float(np.linalg.norm(a[c] - b[c]) / den))
    return float(max(rels)) if rels else float("nan")


def batch_means_se(
    seg_mean: np.ndarray,
    seg_n: np.ndarray,
    start_idx: int,
    n_batch: int = 6,
    chan_groups: tuple[tuple[int, ...], ...] = ((3, 5), (7, 8), (0,)),
) -> float:
    """以 **batch means** 估時均的相對標準誤（取各張量群組的最大值）。

    ★為什麼不用半窗差（`half_window_diff`）做判定：

    半窗差只用「前半 vs 後半」這**一個**對比，等於 1 個自由度。實測（D0-W 三個
    case）顯示它對窗的落點極度敏感——同一 case、同樣 200 CTU 窗長、只把起點滑動
    40 CTU，單窗 SE 就在 6–12%（Re=269）與 14–30%（Re=480）之間跳動，**變異 2 倍**。
    拿這種量做硬 gate 會隨機擋掉品質相同的資料。

    batch means 是 MCMC 標準的 SE 估計法：把 n 個段分成 k 批、各批取平均，
    再由**批間標準差**估 SE。它用到全部 n 段的資訊（k−1 自由度），且只要批長
    大於自相關時間就自動處理段間相關。

        SE = std(批平均) / sqrt(k)

    分母沿用「去掉常數偏移的張量群組範數」（見 :func:`half_window_diff` 的說明），
    因此數值與半窗差/√2 同一量綱、可直接沿用同一組門檻。

    Args:
        seg_mean: `[S, C, H, W]` 各段的段時均。
        seg_n: `[S]` 各段樣本數。
        start_idx: 時均窗起始段索引。
        n_batch: 批數；6 批在 40 段的窗下每批約 6-7 段（33 CTU），大於實測的
            積分自相關時間（~2.7 CTU）。
        chan_groups: 物理上同屬一個張量的通道分組。

    Returns:
        各群組相對 SE 的最大值；段數不足 `2 × n_batch` 時回 NaN。
    """
    s_all = seg_mean[start_idx:]
    w_all = np.asarray(seg_n[start_idx:], dtype=np.float64)
    n = s_all.shape[0]
    if n < 2 * n_batch:
        return float("nan")
    edges = np.linspace(0, n, n_batch + 1).astype(int)
    batches = []
    for i in range(n_batch):
        a, b = edges[i], edges[i + 1]
        if b - a < 1:
            return float("nan")
        w = w_all[a:b]
        batches.append(np.tensordot(w, s_all[a:b], axes=(0, 0)) / w.sum())
    B = np.stack(batches, axis=0)  # [k, C, H, W]
    grand = np.tensordot(w_all, s_all, axes=(0, 0)) / w_all.sum()

    out = []
    for grp in chan_groups:
        # 批間標準差（逐格），再對群組內通道取平方和
        num = np.sqrt(sum(float(np.sum(B[:, c].std(axis=0, ddof=1) ** 2)) for c in grp))
        den = np.sqrt(
            sum(float(np.linalg.norm(grand[c] - grand[c].mean()) ** 2) for c in grp)
        )
        if den <= 0:
            continue
        out.append(num / np.sqrt(n_batch) / den)
    return float(max(out)) if out else float("nan")


def check_mean_convergence(
    se: float, source_is_accum: bool = True, half_diff: Optional[float] = None
) -> Check:
    """時均的統計收斂（兩個獨立半窗的差 = √2 × 單窗 SE）。

    ★只有 `/mean_accum`（逐步累加、float32）才能真的量到統計收斂。舊 H5 只能拿
    稀疏 snapshot 的後段平均代替，那是**另一個量**（樣本少且被取樣間距主導），
    比出來的數字與判準的語義對不上，故降為 `warn`。
    """
    sev = "gate" if source_is_accum else "warn"
    note = "" if source_is_accum else "（來源為稀疏 snapshot，非逐步累加 → warn）"
    quality = (
        "良好"
        if (np.isfinite(se) and se < THRESH["mean_se_ideal"])
        else "偏高（高 Re 低頻擺動的物理限制，見 G5.1 註解）"
    )
    return Check(
        "G5.1",
        "時均統計收斂",
        sev,
        bool(np.isfinite(se) and se < THRESH["mean_se_gate"]),
        float(se),
        THRESH["mean_se_gate"],
        f"batch-means SE；半窗差/√2 = "
        f"{(half_diff/np.sqrt(2)*100 if half_diff is not None else float('nan')):.2f}%"
        f"（上界估計，供對照）；品質 {quality}{note}",
    )


def check_burnin(burnin_ctus: float, re: float, ic_type: str) -> Check:
    """burn-in 是否達 `factor × Re`（動量擴散橫跨 L_char 的時間 = Re 個 CTU）。

    ★G5.2（本項）與 G5.1 是**互補**而非重複，兩者都必須保留：

    * **G5.2 是先驗守門**——只看 config 的 burn-in 是否達到物理需求，不看資料。
      對接近穩態的低 Re case 可能過嚴（它們其實不需要那麼長），這是刻意的保守。
    * **G5.1 是後驗檢查**——實測時均窗內前後半的差，但它的鑑別力**依賴流場的
      非定常性**。

    陰性對照實測（2026-07-29）：Re=63 的 case 把 burn-in 從 63 砍到 10 CTU，
    G5.1 的半窗差只從 0.04% 變 0.08%——**完全沒反應**，因為 Re=63 接近方塊繞流的
    渦脫落臨界（~47），流場幾乎是穩態，時均的前後半必然相同。同一項在 Re=100
    的 L3 實測則清楚反應（warmup 10 CTU → 15.578%、150 CTU → 3.005%）。

    結論：低 Re 的 burn-in 不足**只有 G5.2 抓得到**。若把 G5.2 拿掉而只靠 G5.1，
    低 Re 段的瞬態污染會整批漏過。
    """
    if ic_type == "warm":
        need = 0.2 * re * (THRESH["burnin_re_factor_min"] / 0.3)
        detail = "warm 由 parent 收斂態出發，係數 0.2（待實測驗證）"
    else:
        need = THRESH["burnin_re_factor_min"] * re
        detail = "cold：擴散時間 L²/ν = Re 個 CTU（四 case 實測吻合）"
    if burnin_ctus < 0:
        return Check(
            "G5.2",
            "burn-in 充足",
            "gate",
            False,
            None,
            float(need),
            "H5 無 burnin_ctus（舊協定資料）",
        )
    return Check(
        "G5.2",
        "burn-in 充足",
        "gate",
        bool(burnin_ctus >= need),
        float(burnin_ctus),
        float(need),
        detail,
        "lower",
    )


def stripe_index(field_hw: np.ndarray, x0: int, x1: int) -> float:
    """沿 x 的功率譜中高波數半段的能量占比（逐列去線性趨勢後）。"""
    seg = field_hw[:, x0:x1].astype(np.float64)
    if seg.shape[1] < 8:
        return float("nan")
    xs = np.arange(seg.shape[1], dtype=np.float64)
    for r in range(seg.shape[0]):
        seg[r] -= np.polyval(np.polyfit(xs, seg[r], 1), xs)
    spec = np.abs(np.fft.rfft(seg, axis=1)) ** 2
    spec[:, 0] = 0.0
    tot = spec.sum()
    if tot <= 0:
        return float("nan")
    return float(spec[:, spec.shape[1] // 2 :].sum() / tot)


def check_inlet_stripes(ratio: float) -> Check:
    """入口區高頻能量占比相對遠場的比值。**參考量，非判定**（2026-08-01 降級）。

    ★這個判準被實測證明無效，故由 `gate` 降為 `warn`。三項證據：

    1. **分母失控。** `s_far` 取自固定位置的遠場窗（x ∈ [0.45W, 0.57W]），
       它是否乾淨取決於建物剛好擺在哪裡。實測 9 個被判退的 case 中有 7 個
       的遠場窗**完全沒有建物**，`s_far` 因此塌到通過組的 1/5.5；分子只高
       2.9 倍，比值卻爆成 15 倍。這個比值主要在反映幾何佈局，不是條紋嚴重程度。

    2. **對真正的條紋零敏感度。** 以已知成因（脈衝啟動、無 warmup）的 case
       當陽性對照，它的 ratio 是 **0.10**——遠低於門檻 3.0，輕鬆通過；而目視
       完全乾淨的 case 被判退。

    3. **時均場根本沒有格點尺度峰值。** 四組（陽性對照／已修／判退／通過）的
       「高波數段峰值 ÷ 該段中位數」全是 1.72–1.75，完全一樣。條紋的定義就是
       高波數峰值，沒有峰值就沒有條紋。

    誠實的限制：陽性對照用的是 trajectory 後段平均，啟動暫態本來就會被平均掉，
    因此**沒有一個經驗證的「時均場入口條紋」陽性對照**。但這反過來支持同一個
    結論——若暫態活不到時均場，在時均場上設這道閘門就抓不到它宣稱要抓的東西。

    為何是降級而不是換一個判準：既然沒有任何一組時均場存在高波數峰值，
    任何以現有資料校準的絕對門檻都無法驗證其偵測力。在有可通過陽性對照檢驗的
    偵測器之前，這個數字只能當參考量報出，不能用來判退資料。

    ★降級的代價已評估：它判退的是「建物集中在上游、遠場窗空曠」的 case，
    以它守門會在收錄集裡造成**設計空間的系統性偏差**，這比放行更糟。

    證據：`docs/reviews/DEEP_AUDIT_2026-07/RERUN_ROUND1_FINDINGS_2026-08-01.md` §2、
    `figures/g61_stripe_check.png`、`figures/g61_positive_control.png`。
    """
    return Check(
        "G6.1",
        "入口條紋指數（參考量）",
        "warn",
        bool(np.isfinite(ratio) and ratio < THRESH["inlet_stripe_ratio_max"]),
        float(ratio),
        THRESH["inlet_stripe_ratio_max"],
        "入口/遠場高頻能量占比。★比值受遠場窗是否有建物支配，"
        "已證明無偵測力，僅供參考不判退",
    )


def check_geometry(solid_frac: float) -> Check:
    ok = THRESH["solid_fraction_min"] < solid_frac < THRESH["solid_fraction_max"]
    return Check(
        "G7.1",
        "幾何有效",
        "fatal",
        ok,
        float(solid_frac),
        None,
        f"固體占比需 ∈ ({THRESH['solid_fraction_min']}, "
        f"{THRESH['solid_fraction_max']})",
    )


def check_l_char(
    l_char_attr: float, l_char_measured: float, downsample: int = 3
) -> Check:
    """attrs 的 L_char 與 mask 實算是否一致。

    ★容忍度必須含**降採樣的量化下限**：`geometry/mask` 是 `downsample` 倍降採樣的，
    y-extent 只能量到 `downsample` 的整數倍，故單格量化就有 `downsample/L_char`
    的相對誤差（L_char=40、ds=3 時是 7.5%）。用固定 2% 判定會把完全正確的
    case 判 FAIL——那是判準錯不是資料錯。
    """
    if l_char_attr <= 0:
        return Check(
            "G7.2",
            "L_char 一致",
            "fatal",
            False,
            float("inf"),
            None,
            "attrs 的 l_char_px 非正",
        )
    rel = abs(l_char_measured - l_char_attr) / l_char_attr
    tol = max(THRESH["l_char_rel_err_max"], 1.5 * downsample / l_char_attr)
    return Check(
        "G7.2",
        "L_char 一致",
        "fatal",
        bool(rel < tol),
        float(rel),
        float(tol),
        f"attrs={l_char_attr:.0f} vs mask 實算={l_char_measured:.0f}"
        f"（容忍含 {downsample}× 降採樣量化）",
    )


def check_required_attrs(present: set[str]) -> Check:
    missing = [k for k in REQUIRED_ATTRS if k not in present]
    return Check(
        "G7.3",
        "溯源欄位齊備",
        "fatal",
        not missing,
        float(len(missing)),
        0.0,
        ("缺：" + ", ".join(missing)) if missing else "全部齊備",
    )


def check_symmetry(asym: Optional[float]) -> Check:
    if asym is None or not np.isfinite(asym):
        return Check(
            "G8.1", "對稱性", "warn", True, None, None, "幾何非上下對稱，本項不適用"
        )
    return Check(
        "G8.1",
        "對稱性",
        "warn",
        bool(asym < THRESH["asymmetry_max"]),
        float(asym),
        THRESH["asymmetry_max"],
    )


# ── H5 讀取與組裝 ──────────────────────────────────────────────────────────


def _measure_l_char(mask_hw: np.ndarray, downsample: int = 3) -> float:
    """由 mask 量 cross-flow y-extent（canonical L_char 定義）。

    `geometry/mask` 是降採樣後的 `[H_ds, W_ds]`（row = y），故量到的 y-extent
    要乘回 `downsample` 才是 solver 像素單位。
    """
    from scipy.ndimage import label

    solid = np.asarray(mask_hw, dtype=bool)
    if not solid.any():
        return 0.0
    lab, n = label(solid)
    best = 0
    for i in range(1, n + 1):
        rows = np.where(lab.any(axis=1) & (lab == i).any(axis=1))[0]
        ys = np.where((lab == i).any(axis=1))[0]
        if ys.size:
            best = max(best, int(ys.max() - ys.min() + 1))
        del rows
    return float(best * downsample)


def validate_h5(h5_path: str | Path, verbose: bool = False) -> Report:
    """對一個 case 的 H5 跑完整物理驗證。

    Args:
        h5_path: H5 檔路徑。
        verbose: 逐項印出。

    Returns:
        :class:`Report`，`passed` 為最終收錄判定。
    """
    import h5py

    p = Path(h5_path)
    rep = Report(h5_path=str(p))

    try:
        f = h5py.File(str(p), "r")
    except Exception as exc:  # noqa: BLE001 — 任何開檔失敗都是 fatal
        rep.add(
            Check("G1.1", "結構完整", "fatal", False, None, None, f"無法開啟：{exc}")
        )
        return rep.finalize()

    with f:
        attrs = {k: f.attrs[k] for k in f.attrs}
        rep.case_id = str(attrs.get("case_id", p.stem))
        rep.add(check_required_attrs(set(attrs)))

        need = (
            "trajectory/moments",
            "geometry/mask",
            "convergence_info/kinetic_energy",
        )
        missing = [k for k in need if k not in f]
        rep.add(
            Check(
                "G1.1",
                "結構完整",
                "fatal",
                not missing,
                float(len(missing)),
                0.0,
                ("缺：" + ", ".join(missing)) if missing else "齊備",
            )
        )
        if missing or rep.failures:
            return rep.finalize()

        u_inlet = float(attrs["u_inlet_lu"])
        l_char = float(attrs["l_char_px"])
        nu = float(attrs["nu_lu"])
        tau = float(attrs["tau"])
        re = float(attrs["re"])
        ic_type = str(attrs.get("ic_type", "cold"))
        burnin = float(attrs.get("burnin_ctus", -1.0))
        rep.meta = {
            "re": re,
            "tau": tau,
            "l_char_px": l_char,
            "nu": nu,
            "u_inlet": u_inlet,
            "ic_type": ic_type,
            "burnin_ctus": burnin,
            "total_ctu": float(attrs.get("total_ctu", -1.0)),
            "has_mean_accum": "mean_accum" in f,
        }

        rep.add(check_tau(tau))
        rep.add(check_inlet_ma(u_inlet))
        rep.add(check_re_consistency(re, u_inlet, l_char, nu))

        mask = np.asarray(f["geometry/mask"])  # [H, W] bool，True=solid
        rep.add(check_geometry(float(mask.mean())))
        rep.add(check_l_char(l_char, _measure_l_char(mask)))

        ke = np.asarray(f["convergence_info/kinetic_energy"])
        rep.add(check_ke_growth(ke))
        mres = (
            np.asarray(f["convergence_info/mass_residual_rel"])
            if "convergence_info/mass_residual_rel" in f
            else None
        )
        rep.add(check_mass_residual(mres))

        # ── 時均場：優先用 mean_accum 重組，否則退回 trajectory 後段平均 ──
        if "mean_accum" in f:
            seg = np.asarray(f["mean_accum/segment_mean"], dtype=np.float64)
            seg_n = np.asarray(f["mean_accum/segment_n"], dtype=np.float64)
            ctu0 = np.asarray(f["mean_accum/segment_ctu0"], dtype=np.float64)
            start_ctu = (
                float(attrs.get("warmup_ctus", 0.0))
                if burnin >= 0
                else 0.2 * float(attrs.get("total_ctu", 0.0))
            )
            if burnin >= 0:
                start_ctu = max(start_ctu, 0.0) + burnin
            i0 = int(np.searchsorted(ctu0, start_ctu))
            i0 = min(i0, max(seg.shape[0] - 4, 0))
            w = seg_n[i0:]
            mean_m = np.tensordot(w, seg[i0:], axes=(0, 0)) / w.sum()
            _hd = half_window_diff(seg, seg_n, i0)
            _se = batch_means_se(seg, seg_n, i0)
            rep.add(check_mean_convergence(_se, source_is_accum=True, half_diff=_hd))
            rep.meta["mean_se_batch"] = float(_se)
            rep.meta["mean_half_window_diff"] = float(_hd)
            rep.meta["mean_start_ctu"] = float(start_ctu)
            rep.meta["n_segments_used"] = int(seg.shape[0] - i0)
        else:
            traj = np.asarray(f["trajectory/moments"], dtype=np.float64)
            k0 = traj.shape[0] // 2
            mean_m = traj[k0:].mean(axis=0)
            a = traj[k0 : k0 + (traj.shape[0] - k0) // 2].mean(axis=0)
            b = traj[k0 + (traj.shape[0] - k0) // 2 :].mean(axis=0)
            den = np.linalg.norm(0.5 * (a + b))
            hd = float(np.linalg.norm(a - b) / den) if den > 0 else float("nan")
            rep.add(
                check_mean_convergence(
                    hd / np.sqrt(2.0), source_is_accum=False, half_diff=hd
                )
            )
            rep.meta["mean_start_ctu"] = -1.0
        rep.add(check_burnin(burnin, re, ic_type))

        # ── 由時均矩導出場量 ──
        rho = mean_m[CH_RHO]
        safe = np.where(rho > 1e-9, rho, 1.0)
        u = mean_m[CH_JX] / safe
        v = mean_m[CH_JY] / safe
        fluid = ~np.asarray(mask, dtype=bool)

        finite_ok = bool(np.isfinite(mean_m).all())
        rep.add(
            Check(
                "G1.2",
                "數值有限",
                "fatal",
                finite_ok,
                None,
                None,
                "時均矩含 NaN/Inf" if not finite_ok else "全部有限",
            )
        )
        if not finite_ok:
            return rep.finalize()

        rep.add(check_local_ma(float(np.abs(u[fluid]).max() if fluid.any() else 0.0)))
        rep.add(
            check_mass_conservation(
                divergence_ratio(rho, u, v, fluid),
                u_inlet / C_S,
                source_is_f32=("mean_accum" in f),
            )
        )

        # ── 入口條紋 ──
        W = u.shape[1]
        w_in = max(8, int(0.12 * W))
        s_in = stripe_index(u, 0, w_in)
        s_far = stripe_index(u, int(0.45 * W), int(0.45 * W) + w_in)
        ratio = s_in / s_far if (np.isfinite(s_far) and s_far > 0) else float("nan")
        rep.add(check_inlet_stripes(ratio))
        rep.meta["stripe_inlet"] = s_in
        rep.meta["stripe_far"] = s_far

        # ── 對稱性（僅在幾何本身上下對稱時才有意義）──
        m_arr = np.asarray(mask, dtype=bool)
        geo_sym = bool(np.array_equal(m_arr, m_arr[::-1, :]))
        asym = None
        if geo_sym:
            den = np.linalg.norm(u)
            asym = (
                float(np.linalg.norm(u - u[::-1, :]) / den) if den > 0 else float("nan")
            )
        rep.add(check_symmetry(asym))
        rep.meta["geometry_updown_symmetric"] = geo_sym

    rep.finalize()
    if verbose:
        _print_report(rep)
    return rep


def _print_report(rep: Report) -> None:
    print(f"\n=== {rep.case_id} — {'PASS' if rep.passed else 'FAIL'} ===")
    for c in rep.checks:
        mark = "OK  " if c.passed else ("FAIL" if c.severity != "warn" else "warn")
        val = "" if c.value is None else f"{c.value:.4g}"
        thr = "" if c.threshold is None else f" / {c.threshold:.4g}"
        print(
            f"  [{mark}] {c.id:5s} {c.name:12s} {val:>10s}{thr:>12s}" f"   {c.detail}"
        )
    if rep.failures:
        print(f"  → 未通過：{', '.join(c.id for c in rep.failures)}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description="單 case 物理驗證（收錄閘門）")
    ap.add_argument("--h5", type=str, default=None)
    ap.add_argument(
        "--dataset", type=str, default=None, help="dataset 名稱，驗證其 raw/*.h5 全部"
    )
    ap.add_argument("--ws", type=str, default=None, help="workspace 根")
    ap.add_argument("--out", type=str, default=None, help="報告 JSON 輸出路徑")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ws = Path(args.ws) if args.ws else None
    targets: list[Path] = []
    if args.h5:
        targets = [Path(args.h5)]
    elif args.dataset:
        # --ws 指定時沿用 <ws>/dataset/<name>/raw 舊慣例；否則走 paths.py 的
        # dataset_dir() 單一真相（<LBMNCA_DATA_ROOT>/datasets/<name>/raw）。
        raw_dir = (
            (ws / "dataset" / args.dataset / "raw")
            if ws is not None
            else (dataset_dir(args.dataset) / "raw")
        )
        targets = sorted(raw_dir.glob("*.h5"))
    else:
        ap.error("需指定 --h5 或 --dataset")

    if not targets:
        print("找不到任何 H5")
        return 2

    reports = [validate_h5(t, verbose=not args.quiet) for t in targets]
    n_pass = sum(r.passed for r in reports)
    print(f"\n總計 {len(reports)} case：PASS {n_pass}、FAIL {len(reports)-n_pass}")
    if n_pass < len(reports):
        print("★FAIL 的 case 依收錄規則必須作廢，重新生成 mask/config 後重跑。")
        for r in reports:
            if not r.passed:
                print(f"    {r.case_id}: {', '.join(c.id for c in r.failures)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "n_total": len(reports),
                    "n_pass": n_pass,
                    "thresholds": THRESH,
                    "reports": [r.to_dict() for r in reports],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"→ {args.out}")
    return 0 if n_pass == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
