"""VQI 的刻度。**兩個端點逐案定死，起點不參與定義。**

| 端點 | 取自 | 分數 |
|---|---|---|
| 上 | 該案的真值 | $+1$ |
| 下 | 該案的零場，即動量全歸零 | $-1$ |

刻度因而是**絕對的**——同一個分數在任何案例、任何母體上代表同一件事，
即「離真值多近、離零場多遠」。暖啟動起點只是落在卷面上的一個點，
其分數為量出來的結果而非定義。需要相對讀數時自行相減

    進步幅度 = VQI(預測) − VQI(起點)

## 為什麼起點不能當零點

以起點為零時量到的是**進步幅度**而非絕對品質。同一份輸出，起點越差則
分數越高。更要緊的是，起點若本身充滿雜訊，一個仍帶雜訊的輸出會因為
「比起點好」而得正分，可是那張畫面根本不能用。絕對刻度不受此影響，
帶雜訊的輸出無論起點多差都落在下端點附近。

## 偏差量為什麼要壓成有界

`deviation` 對比值項取 $|\\ln r|$，在 $r \\to 0$ 時發散。零場的高頻能量與
峰值在數學上恰為零，其偏差因而由 ``max(x, 1e-12)`` 這個防除零常數決定
而非由物理決定，實測為 $|\\ln 10^{-12}| = 27.63$。直接拿它當下端點時
$u$ 被推到兩百倍以上，「幅值 $\\times 0.25$」這種明顯錯的場會被壓到
$-0.009$ 而幾乎等於起點。

改以 $\\hat d = \\tanh(d / 2)$ 壓進 $[0, 1)$。三個性質

1. 零場恰好落在 $\\hat d = 1$，故其分數恰為 $-1$。
2. 對比值項而言 $\\tanh(|\\ln r| / 2) \\equiv |r-1| / (r+1)$，即標準的相對差。
   **除數 2 由此式導出而非調參**，$4\\times$ 與 $0.25\\times$ 仍然同分。
3. tanh 在大輸入時飽和，防除零常數取 $10^{-6}$ 或 $10^{-30}$ 皆使零場落在
   $1.0$，刻度因而與該常數無關。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .terms import IDEAL_BY_KIND, resolve_term_set

__all__ = ["build_anchors", "deviation", "bounded_deviation", "subscores",
           "subscore_inverse", "veto_thresholds", "TANH_DIVISOR"]

#: 壓縮偏差量的除數。取 2 使比值項的壓縮恰為 $|r-1|/(r+1)$，見模組說明。
TANH_DIVISOR: float = 2.0


def deviation(kind: str, x: float) -> float:
    """把原始值換算為偏差量。0 為理想，越大越差，恆非負。

    五項的量綱互不相通，相似度落在 0 至 1、頻譜距離落在 0 至 15、能量比為倍數。
    先統一成「離理想有多遠」，其後的歸一才有共同的操作對象。

    Args:
        kind: ``"up"``（越大越好）、``"down"``（越小越好）或 ``"ratio"``（比值）。
        x: 該項的原始值。

    Returns:
        偏差量，恆非負。比值項取 ``|ln x|``，**4× 與 0.25× 因而得同分**，
        過強與過弱一視同仁。取線性的 ``|x − 1|`` 會讓過強被罰得比過弱重
        （3× 得 2.0 而 1∕3 只得 0.667），且比值歸零時的偏差僅為 1
        而低於白雜訊使高頻能量爆到十七倍的 15.9，方向完全相反。

    ★ **下限夾在 0。** 理想值取自指標定義（相似度 1、比值 1、距離 0），
    定義上不可超越，故負偏差只可能是浮點誤差。實測 `VortSignedCorr` 於真值
    對自己得 ``1 + 7e-8``，不夾則壓縮後超過零場的單位而使分數破 1。
    """
    if kind == "up":
        d = IDEAL_BY_KIND["up"] - x
    elif kind == "down":
        d = x - IDEAL_BY_KIND["down"]
    else:
        d = abs(math.log(max(x, 1e-12) / IDEAL_BY_KIND["ratio"]))
    return max(0.0, d)


def bounded_deviation(kind: str, x: float) -> float:
    """把偏差量壓進 ``[0, 1)``。見模組說明。

    Args:
        kind: 見 `deviation`。
        x: 該項的原始值。

    Returns:
        壓縮後的偏差量。零場在比值項上落在 ``1.0``（到機器精度），
        與防除零常數無關。
    """
    return math.tanh(deviation(kind, x) / TANH_DIVISOR)


def build_anchors(zero: dict[str, float], term_set: str = "full",
                  noise_floor: dict[str, float] | None = None,
                  ) -> dict[str, dict[str, Any]]:
    """由該案的零場導出每一項的刻度單位。

    單位即零場的壓縮後偏差量。真值的偏差為零、零場的偏差為該單位，
    分數 $S = 1 - 2\\hat d / \\hat d_{zero}$ 因而在兩者上恰為 $+1$ 與 $-1$。

    項被判為**不可用**的唯一情形是零場的偏差為零或非有限值，
    即該項對「有沒有流動」毫無反應。實測五項無一如此。

    Args:
        zero: 該案零場的逐項原始值。**必須來自該案自己的真值**，
            由 `fields.zero_field` 造出。
        term_set: 項組名稱。
        noise_floor: 該案真值自身兩段時均窗的逐項值。**不參與刻度**，
            僅登錄供報告時對照，其分數即該案的實務天花板。

    Returns:
        項名 → ``{"ideal","zero","unit","kind","group","noise_floor",
        "usable","reason"}``。
    """
    terms, _veto_ref = resolve_term_set(term_set)
    nf = noise_floor or {}
    out: dict[str, dict[str, Any]] = {}
    for name, (grp, dirn, _fn) in terms.items():
        s = zero.get(name, float("nan"))
        unit = bounded_deviation(dirn, s) if np.isfinite(s) else float("nan")
        rec: dict[str, Any] = {
            "ideal": IDEAL_BY_KIND[dirn], "zero": s, "unit": float(unit),
            "kind": dirn, "group": grp,
            "noise_floor": nf.get(name, float("nan")),
            "usable": True, "reason": "",
        }
        if not np.isfinite(unit):
            rec.update({"usable": False, "reason": "零場的偏差為非有限值"})
        elif unit <= 0.0:
            rec.update({"usable": False,
                        "reason": "零場恰在理想值上，該項對有無流動無鑑別度"})
        out[name] = rec
    return out


def subscores(raw: dict[str, float],
              anchors: dict[str, dict[str, Any]]) -> dict[str, float]:
    """把原始項值換算為以兩端點為刻度的子分數。

    ```
    S = 1 − 2 · tanh(deviation(x) / 2) / unit
    ```

    ``+1`` 為該案的真值、``−1`` 為該案的零場。刻度**全段同一個單位**，
    不分正負兩段，故任意兩個讀數之差可直接相比。

    ★ 分數可以低於 ``−1``。零場在相似度與相關係數上的讀數約為 0，而這兩項
    的理想值為 1、下界為 −1，故一個與真值反相關的場會比零場更遠。
    語義為「產出錯的結構比什麼都不產出更糟」，不夾板以保留該資訊。

    不可用的項與非有限的讀數直接跳過，不進入合成。
    """
    s: dict[str, float] = {}
    for name, a in anchors.items():
        if not a["usable"] or name not in raw or not np.isfinite(raw[name]):
            continue
        d = bounded_deviation(a["kind"], raw[name])
        s[name] = float(1.0 - 2.0 * d / a["unit"])
    return s


def subscore_inverse(name: str, s: float,
                     anchors: dict[str, dict[str, Any]]) -> float:
    """由子分數回推偏差量。閉式解，與 `subscores` 互為反函數。

    ```
    δ = 2 · artanh( (1 − S) · unit / 2 )
    ```

    Args:
        name: 項名。
        s: 該項的子分數。
        anchors: `build_anchors` 的輸出。

    Returns:
        壓縮前的偏差量 δ。★比值項的 δ 不帶號，還原得 ``x ∈ {e^δ, e^−δ}``；
        該項本就設計為過強與過弱同分，欲完全還原原始值須另存符號。

    Raises:
        ValueError: 該項不可用，或分數對應的壓縮後偏差不落在 ``[0, 1)``。
    """
    a = anchors[name]
    if not a["usable"]:
        raise ValueError(f"{name} 不可用（{a['reason']}），無從還原")
    d_hat = (1.0 - s) * a["unit"] / 2.0
    if not (0.0 <= d_hat < 1.0):
        raise ValueError(
            f"{name} 的子分數 {s} 對應的壓縮後偏差為 {d_hat}，"
            f"不落在 [0, 1)。tanh 的值域上界不可達，該分數無原像")
    return float(TANH_DIVISOR * math.atanh(d_hat))


def veto_thresholds(veto_ref: dict[str, dict[str, float]],
                    anchors: dict[str, dict[str, Any]],
                    term_set: str = "full") -> dict[str, float]:
    """逐項的否決門檻＝該項在**它自己負責偵測的最輕偽影**上的子分數。

    多個參照時取其中**最高**的子分數，即較寬鬆的那一個偽影，門檻因而較嚴，
    四類偽影才會全部觸發。

    ★ 門檻與刻度同為逐案量，其參照場由該案的真值當場造出
    （見 `fields.veto_reference_fields`），母體層級的量完全不進入。

    Args:
        veto_ref: 偽影類別名 → 該類輕級在該案上的逐項原始值。
        anchors: `build_anchors` 的輸出。
        term_set: 項組名稱。

    Returns:
        項名 → τ。未列於參照表的項不設門檻，不參與否決。
    """
    _terms, ref_tags = resolve_term_set(term_set)
    tau: dict[str, float] = {}
    for term, refs in ref_tags.items():
        vals = []
        for r in refs:
            row = veto_ref.get(r)
            if row is None:
                continue
            v = subscores(row, anchors).get(term)
            if v is not None:
                vals.append(v)
        if vals:
            tau[term] = float(max(vals))
    return tau
