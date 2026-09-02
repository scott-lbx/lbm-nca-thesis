"""VQI　視覺品質指數。

把一張預測流場與其真值之間的差距，壓成**一個以視覺效果為準的分數**。

## 值域語義

刻度由該案自己的兩個場定死，是**絕對的**——同一個分數在任何案例、
任何母體上代表同一件事。

| 值 | 意義 |
|---|---|
| **+1** | 該案的真值。距離為零，不可超越 |
| 中間值 | 全段同一個單位，兩個讀數之差可直接相比 |
| **−1** | 該案的**零場**，即動量全歸零、逐點靜止 |
| **低於 −1** | 產出了錯的結構，比什麼都不產出更糟。不夾板以保留該資訊 |
| **否決旗標** | 出現視覺上不可接受的偽影。**與數值分開報，不可互相取代** |

**暖啟動起點不對應任何固定分數。** 它只是卷面上量出來的一點。
需要「比不動作進步多少」時自行相減

```python
gain = r_model.vqi - r_identity.vqi
```

以起點為零點的舊讀法量到的是進步幅度而非絕對品質，且起點若本身充滿雜訊，
一個仍帶雜訊的輸出會因為「比起點好」而得正分——那張畫面根本不能用。

## 用法

一次呼叫，四個輸入。沒有需要事先載入的錨點檔。

```python
from lbmnca.nca.tes.eval.vqi import compute_vqi

rs = compute_vqi(pred, target, fluid_mask, normalizer)

rs[0].vqi            # float，最小子分數，判準吃這個
rs[0].binding_term   # str，把分數壓下去的那一項
rs[0].veto           # bool，是否視覺不可接受
rs[0].invalid        # bool，是否未通過物理有效性前置檢查
rs[0].sub            # dict，逐項歸一後的子分數
rs[0].noise_ceiling  # float，該案的實務天花板，須與 vqi 並列引用
```

反正規化、遮罩腐蝕、頻譜、造零場、造否決參照、歸一、合成全部在內部完成。

## 輸入的三個硬性要求

1. 場為**正規化後的九通道**張量，反正規化由內部處理，呼叫端不要自己轉。
2. 流體遮罩沿用 canonical 導法，即 static 通道 0 **大於 0 代表流體**。
   寫反會讓整份量測跑在固體區上。
3. 預測與真值**逐案對齊**。刻度由真值導出，餵錯真值等於換了一把尺。

## 模組分工

| 模組 | 職責 |
|---|---|
| `terms` | 項的定義、項組、視覺校準標籤 |
| `fields` | 刻度所需的參照場，即零場與四類偽影的輕級 |
| `anchors` | 刻度建構與歸一數學 |
| `score` | 合成、否決、`compute_vqi` 入口 |
"""
from __future__ import annotations

from .anchors import (
    TANH_DIVISOR, bounded_deviation, build_anchors, deviation, subscore_inverse,
    subscores, veto_thresholds,
)
from .fields import (
    MOMENTUM_CHANNELS, VETO_REF_AMPLITUDE, zero_field, veto_reference_fields,
)
from .score import (
    VQIScore, case_scale, combine, compute_vqi, noise_ceiling_of,
)
from .terms import (
    ARTIFACT_TAGS, BASELINE_TAG, CANONICAL_TERM_SET, CLEAN_TAGS, DEAD_TAGS,
    ERODE_PX, EXCLUDED_TERMS, LATTICE_SOUND_SPEED, PEAK_Q, RAPSD_BINS,
    ZERO_TAG, TERM_SETS, TERMS, VETO_REF_TERM, compute_terms,
    resolve_term_set, validity,
)

__all__ = [
    # 對外唯一入口
    "compute_vqi",
    "VQIScore",
    # 逐案刻度，供稽核與 harness
    "case_scale",
    "noise_ceiling_of",
    "zero_field",
    "veto_reference_fields",
    # 內部構件
    "compute_terms",
    "build_anchors",
    "subscores",
    "subscore_inverse",
    "deviation",
    "bounded_deviation",
    "veto_thresholds",
    "combine",
    "resolve_term_set",
    # 定義表
    "TERMS",
    "TERM_SETS",
    "CANONICAL_TERM_SET",
    "EXCLUDED_TERMS",
    "VETO_REF_TERM",
    "VETO_REF_AMPLITUDE",
    "MOMENTUM_CHANNELS",
    "TANH_DIVISOR",
    "ARTIFACT_TAGS",
    "DEAD_TAGS",
    "CLEAN_TAGS",
    "BASELINE_TAG",
    "ZERO_TAG",
    "ERODE_PX",
    "RAPSD_BINS",
    "PEAK_Q",
    "LATTICE_SOUND_SPEED",
    "validity",
]
