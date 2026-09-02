"""VQI 的合成、否決，以及對外的唯一入口 `compute_vqi`。

## 最小呼叫

刻度是逐案量，由該案的真值當場導出，沒有任何需要事先載入的錨點檔。
評分因而是一次呼叫的事

```python
from lbmnca.nca.tes.eval.vqi import compute_vqi

rs = compute_vqi(pred, target, fluid_mask, normalizer)

rs[0].vqi            # +1 為該案真值、−1 為該案零場
rs[0].binding_term   # 把分數壓下去的那一項
rs[0].veto           # 是否出現視覺不可接受的偽影
```

四個輸入分別是模型的直接輸出、其真值、流體遮罩與 `FieldNormalizer`。
反正規化、遮罩腐蝕、頻譜、造零場、造否決參照、歸一、合成全部在內部完成。

## 逐案，不聚合

回傳一律是 ``list[VQIScore]``，長度等於批次大小。**沒有整批聚合的選項**，
因為每一案有自己的尺，把不同案例的子分數混在一起平均不具語義。
要母體層級的數字時對回傳值取中位或平均，並一併報告離散度。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .anchors import build_anchors, subscores, veto_thresholds
from .fields import zero_field, veto_reference_fields
from .terms import CANONICAL_TERM_SET, compute_terms, resolve_term_set, validity

__all__ = ["VQIScore", "combine", "case_scale", "compute_vqi", "noise_ceiling_of",
           "VETO_TIE_TOL"]

#: 否決比較的平手容差。判準是「子分數落在門檻**或以下**」，而門檻本身取自
#: 參照偽影的子分數，故參照級恆為平手。同一張場經由兩條不同的運算路徑算出來
#: 會差到最後幾位，純以 ``<=`` 比較時該級的否決率因而不穩定。
#:
#: 取值由實測界定：浮點噪聲的上界為 8.3e-07，真正的間距下界為 0.080，
#: 兩者相隔五個數量級。取 1e-5 落在其間，為噪聲的十二倍而為真實間距的八千
#: 分之一，故它只吸收平手，不可能把不該否決的場掃進來。
VETO_TIE_TOL: float = 1e-5


@dataclass(frozen=True)
class VQIScore:
    """單一張預測場的評分結果。

    刻度為**絕對的**，$+1$ 為該案的真值、$-1$ 為該案的零場。

    Attributes:
        vqi: 全部子分數的最小值。判準一律吃這個。``+1`` 為該案的真值、
            ``−1`` 為該案的零場（動量全歸零）。刻度全段同一個單位，
            故任意兩個讀數之差可直接相比，跨案與跨母體亦然。
            **暖啟動起點不對應任何固定分數**，其分數為量出來的結果；
            需要「比不動作進步多少」時自行相減。
            分數可以低於 ``−1``，語義為「產出錯的結構比什麼都不產出更糟」。
        binding_term: 把分數壓下去的那一項。
        veto: 是否出現視覺不可接受的偽影。**與 `vqi` 分開報，不可互相取代**。
        veto_terms: 觸發否決的項。
        sub: 逐項歸一後的子分數。
        raw: 逐項的原始值，供回查。
        noise_ceiling: 該案真值自身兩段時均窗互比所得的分數，即**實務天花板**。
            量的是「這把尺在這一案上量得多細」，未提供半窗時均時為 ``nan``。
            ★引用 `vqi` 時應並列本欄——天花板低的案，其分數的可信範圍很窄。
        vqi_rms: 距離均方根。**次要讀數，與判準無關**，表示整體離真值多遠。

            ``VQI_rms = 1 − sqrt( mean_i (1 − S_i)² )``

            RMS 較算術平均更重罰極端項，但**仍是一種平均**，
            四項完美而一項落到零場時它仍給正分而 `vqi` 給 ``−1``。
            判準一律吃 `vqi`，本欄只回答「整體有多好」。
        invalid: 是否未通過物理有效性前置檢查。**為真時 `vqi` 不具意義**，
            因為一個帶負密度的場根本不是流場。與 `veto` 分開——
            後者是視覺判斷，本欄是硬性事實。
        invalid_reasons: 違反的判準。
    """

    vqi: float
    binding_term: str | None
    veto: bool
    veto_terms: list[str]
    sub: dict[str, float]
    raw: dict[str, float] = field(default_factory=dict)
    noise_ceiling: float = float("nan")
    vqi_rms: float = float("nan")
    invalid: bool = False
    invalid_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """轉為可 JSON 序列化的 dict。

        ★非有限值一律轉 ``None``。``json.dumps`` 預設會把 ``nan`` 寫成 `NaN`
        字面值，該字面值不在 RFC 8259 之內，嚴格解析器讀到就拋 ParseError，
        讀數檔因而不可攜。Python 端仍保留 ``nan``，只有落檔這一層轉 ``None``。
        """
        def _j(x: float) -> float | None:
            return x if isinstance(x, float) and math.isfinite(x) else (
                x if not isinstance(x, float) else None)

        return {"vqi": _j(self.vqi), "binding_term": self.binding_term,
                "veto": self.veto, "veto_terms": self.veto_terms,
                "sub": {k: _j(v) for k, v in self.sub.items()},
                "raw": {k: _j(v) for k, v in self.raw.items()},
                "noise_ceiling": _j(self.noise_ceiling),
                "vqi_rms": _j(self.vqi_rms),
                "invalid": self.invalid, "invalid_reasons": self.invalid_reasons}


def combine(raw: dict[str, float], anchors: dict[str, dict[str, Any]],
            tau: dict[str, float] | None = None,
            invalid_reasons: list[str] | None = None,
            noise_ceiling: float = float("nan")) -> VQIScore:
    """把逐項原始值合成為單一視覺品質指數。

    **取全部子分數的最小值**，這是「視覺一票否決」最直接的連續量對應，
    且可指出是哪一項把分數壓下去。★取平均會把單一項的強訊號稀釋掉——
    實測滿振幅棋盤格在平均下讀 ``+0.23``，一張全是網格振盪的畫面因而拿到正分，
    因為其餘四項還好。取最小值則由該項直接定分，讀 ``−1``。

    子分數為 ``+1`` 於該案的真值、``−1`` 於該案的零場，全段同一個單位，
    故任意兩個讀數之差可直接相比。分數可以低於 ``−1``，語義為
    「產出錯的結構比什麼都不產出更糟」，不夾板以保留該資訊。

    否決**只看紋理群**且與數值分開。結構與能量項再低也不觸發否決，
    那是 VQI 數值的職責。

    Args:
        raw: 逐項原始值。
        anchors: `build_anchors` 的輸出。
        tau: 逐項否決門檻。未給定時不判否決。
        invalid_reasons: `terms.validity` 的輸出。非空即標記 `invalid`。
        noise_ceiling: 該案的實務天花板，見 `VQIScore.noise_ceiling`。

    Returns:
        `VQIScore`。全部項皆不可用時 `vqi` 為 ``nan`` 且自動否決，
        **不回一個看似正常的分數**。
    """
    s = subscores(raw, anchors)
    inv = list(invalid_reasons or [])
    if not s:
        return VQIScore(vqi=float("nan"), binding_term=None, veto=True,
                        veto_terms=["無可用項"], sub={}, raw=raw,
                        invalid=bool(inv), invalid_reasons=inv)
    binding = min(s, key=lambda k: s[k])
    vetoed = [k for k, x in s.items()
              if anchors[k]["group"] == "texture" and tau
              and x <= tau.get(k, float("-inf")) + VETO_TIE_TOL]
    vals = list(s.values())
    rms = float(1.0 - np.sqrt(np.mean([(1.0 - x) ** 2 for x in vals])))
    return VQIScore(vqi=float(s[binding]), binding_term=binding, veto=bool(vetoed),
                    veto_terms=vetoed, sub=s, raw=raw,
                    noise_ceiling=float(noise_ceiling), vqi_rms=rms,
                    invalid=bool(inv), invalid_reasons=inv)


def case_scale(target: torch.Tensor, fluid_mask: torch.Tensor, normalizer,
               term_set: str = CANONICAL_TERM_SET,
               noise_floor: dict[str, float] | None = None,
               ) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """由單一案例的真值導出該案的刻度與否決門檻。

    **這是刻度的唯一來源。** 兩個端點與門檻全部由 `target` 當場造出，
    母體層級的量一概不進入，同一張真值在任何批次、任何母體上都給出同一把尺。

    一般評分請用 `compute_vqi`，本函式供需要檢視錨點本身的稽核使用。

    Args:
        target: ``[1,9,H,W]`` 正規化真值場。
        fluid_mask: ``[1,1,H,W]``，流體為 1。
        normalizer: `FieldNormalizer`。
        term_set: 項組名稱。
        noise_floor: 該案真值兩段時均窗的逐項值。**不參與刻度**，
            僅登錄供報告時對照。

    Returns:
        ``(anchors, tau)``。
    """
    zero = compute_terms(zero_field(target), target, fluid_mask,
                         normalizer, term_set)
    anchors = build_anchors(zero, term_set, noise_floor)
    refs = {name: compute_terms(f, target, fluid_mask, normalizer, term_set)
            for name, f in veto_reference_fields(target, fluid_mask).items()}
    return anchors, veto_thresholds(refs, anchors, term_set)


def _slice(x: torch.Tensor, i: int, n: int, name: str) -> torch.Tensor:
    """取批次第 ``i`` 案。``N=1`` 時視為對整批廣播，其餘不符即報錯。"""
    if x.shape[0] == n:
        return x[i:i + 1]
    if x.shape[0] == 1:
        return x
    raise ValueError(
        f"{name} 的批次維度為 {x.shape[0]}，與預測場的 {n} 不符，"
        f"且不為 1（可廣播）。逐案評分無從對齊，請檢查呼叫端的切片")


def noise_ceiling_of(half1: torch.Tensor, half2: torch.Tensor,
                     fluid_mask: torch.Tensor, normalizer,
                     term_set: str = CANONICAL_TERM_SET) -> float:
    """該案的實務天花板，即真值自身兩段時均窗互比所得的分數。

    語義為「這把尺在這一案上量得多細」。天花板低的案，其分數的可信範圍很窄，
    模型讀數落在天花板附近即已達量測極限，再往上的差異是噪聲不是訊號。

    Args:
        half1: ``[1,9,H,W]`` 前半窗時均場。
        half2: ``[1,9,H,W]`` 後半窗時均場。
        fluid_mask: ``[1,1,H,W]``，流體為 1。
        normalizer: `FieldNormalizer`。
        term_set: 項組名稱。

    Returns:
        以 `half1` 為真值時 `half2` 的 VQI。
    """
    anchors, tau = case_scale(half1, fluid_mask, normalizer, term_set)
    raw = compute_terms(half2, half1, fluid_mask, normalizer, term_set)
    return combine(raw, anchors, tau).vqi


def compute_vqi(pred: torch.Tensor, target: torch.Tensor,
                fluid_mask: torch.Tensor, normalizer, *,
                term_set: str = CANONICAL_TERM_SET,
                noise_floor: list[dict[str, float]] | None = None,
                check_validity: bool = True) -> list[VQIScore]:
    """**對外唯一入口。** 對一批預測場評分。

    刻度由每一案的真值當場導出，故不需要任何事先載入的錨點檔，
    也不需要指定母體。同一張真值在任何批次、任何母體上都給出同一把尺。

    Args:
        pred: ``[N,9,H,W]`` 正規化預測場，即**模型的直接輸出**。
        target: ``[N,9,H,W]`` 正規化真值場。``N=1`` 時對整批廣播。
        fluid_mask: ``[N,1,H,W]``，流體為 1。``N=1`` 時對整批廣播。
        normalizer: `FieldNormalizer`，供內部反正規化。
        term_set: 項組名稱，預設為驗收通過的五項組。
        noise_floor: 逐案的半窗時均逐項值，長度須等於 ``N``。給定時填入
            每一案的 `VQIScore.noise_ceiling`。
        check_validity: 是否跑物理有效性前置檢查。**正式評測一律為真**，
            關掉只用於不帶真實 `normalizer` 的合成測試。做成顯式參數而非
            自動略過，是為了讓「沒檢查」在呼叫端看得見。

    Returns:
        ``list[VQIScore]``，長度為 ``N``，逐案獨立評分。

    Raises:
        ValueError: `noise_floor` 的長度與批次不符。

    Example:
        ```python
        rs = compute_vqi(pred, target, fluid_mask, normalizer)
        print([r.vqi for r in rs])
        print(float(np.median([r.vqi for r in rs])))
        ```
    """
    resolve_term_set(term_set)               # 早期失敗，項組名稱錯就別往下走
    n = pred.shape[0]
    if noise_floor is not None and len(noise_floor) != n:
        raise ValueError(
            f"noise_floor 有 {len(noise_floor)} 筆而批次為 {n} 案，逐案對不起來")
    # ★真值與遮罩皆為單案廣播時，整批共用同一把尺，故只導出一次。
    #   造一把尺要算零場與四類參照場共五輪 `compute_terms`，對同一張真值重算
    #   純屬浪費——實測以同一真值評十張場時，逐張重算會多花九成的時間。
    shared = target.shape[0] == 1 and fluid_mask.shape[0] == 1
    cached: tuple[dict, dict] | None = None
    out: list[VQIScore] = []
    for i in range(n):
        p = _slice(pred, i, n, "pred")
        t = _slice(target, i, n, "target")
        m = _slice(fluid_mask, i, n, "fluid_mask")
        nf = noise_floor[i] if noise_floor is not None else None
        if shared and cached is not None and nf is None:
            anchors, tau = cached
        else:
            anchors, tau = case_scale(t, m, normalizer, term_set, nf)
            if shared and nf is None:
                cached = (anchors, tau)
        ceil = combine(nf, anchors, tau).vqi if nf else float("nan")
        inv = validity(p, m, normalizer) if check_validity else None
        out.append(combine(compute_terms(p, t, m, normalizer, term_set),
                           anchors, tau, inv, ceil))
    return out
