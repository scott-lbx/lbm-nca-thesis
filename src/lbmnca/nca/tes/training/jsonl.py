"""metrics.json（line-delimited JSON）的寫入淨化層。

★為何需要這層：原本四個寫入點都是
``json.dumps(record, default=str)``。Python 的 ``json`` 預設 ``allow_nan=True``，
會把 ``float('nan')``／``inf`` 寫成裸 token ``NaN``／``Infinity`` —— 那**不是合法
JSON**（RFC 8259 沒有這些字面值）。後果分兩層：

1. Python 端 ``json.loads`` 容忍它，所以整條證據鏈從來不會報錯；
2. 但論文證據會經過非 Python 的解析（JS 前端、``jq``、pandas 的嚴格模式），
   在那裡不是解析失敗就是被轉成 ``null``／字串，**兩種都會讓「該步發散」這件事
   在下游變成無聲的資料缺漏**，而缺漏恰好與被量的性質相關（MNAR）。

**不採用直接 ``allow_nan=False``**：那會在訓練發散的當下、於 logging 層丟
``ValueError`` 殺掉整個 run —— 把「指標非有限」升級成「訓練崩潰」，代價與收益
不成比例，且 NaN 出現時最該保住的就是那幾行紀錄。

本模組的作法：先把非有限值換成 ``None``（合法 JSON 的 null），同時在同一筆
record 內留下 ``<key>__nonfinite`` 標記其原值（``"nan"``／``"inf"``／``"-inf"``），
再以 ``allow_nan=False`` 序列化。

若只靠 ``allow_nan=False``：它只檢查 Python 內建 ``float``。numpy／torch 的
純量（如 ``np.float32``／``np.float16``／0 維 ``torch.Tensor``）在到達這道
檢查前，會先被 ``default=str`` 轉成字串（例如 ``np.float32('nan')`` 變成
``{"x": "nan"}``，連 ``np.float32(1.5)`` 這種**有限值**也會變成字串
``"1.5"``），使 ``allow_nan=False`` 這道安全網對 numpy／torch 標量結構性
失效——而訓練 record 的數值多半來自 tensor，只是大部分路徑先呼叫了
``.item()`` 才僥倖是 Python float。

現在先把「0 維數值標量」（numpy scalar／0-d torch tensor／任何 ``numbers.Real``）
正規化成 Python ``float``／``int``，再走非有限值判定。``default=str`` 保留給真正
非數值的物件（``Path``、dataclass 等），那是刻意的既有行為。
"""

from __future__ import annotations

import json
import math
import numbers
from typing import Any

__all__ = [
    "NONFINITE_SUFFIX",
    "sanitize_record",
    "restore_record",
    "dumps_record",
    "as_number",
]

#: ``__nonfinite`` 標記值 → 還原用的浮點值。
_TAG_TO_VALUE: dict[str, float] = {
    "nan": float("nan"),
    "inf": float("inf"),
    "-inf": float("-inf"),
}

#: 標記某鍵原本是非有限值的後綴。
NONFINITE_SUFFIX: str = "__nonfinite"


def as_number(value: Any) -> Any:
    """把 0 維數值標量正規化成 Python ``int``／``float``；非數值原樣回傳。

    涵蓋 numpy scalar（``np.float32`` 等）、0-d ``torch.Tensor``、以及任何
    ``numbers.Real``。★不在此處 import numpy／torch——用 duck typing 判斷，
    避免本模組對它們產生 import 相依（也才能在沒裝 torch 的環境測試）。

    Args:
        value: 任意物件。

    Returns:
        數值標量 → Python ``int``／``float``；其餘原樣回傳。
    """
    if isinstance(value, bool):
        return value                      # bool 是 int 子類，但語義上不是數
    if isinstance(value, (int, float)):
        return value
    # numpy scalar／0-d tensor：有 .item() 且維度為 0
    if hasattr(value, "item"):
        ndim = getattr(value, "ndim", None)
        shape = getattr(value, "shape", None)
        if ndim == 0 or shape == ():
            try:
                got = value.item()
            except Exception:             # noqa: BLE001 — 取不出就當非數值
                return value
            if isinstance(got, (bool, int, float)):
                return got
            return value
    if isinstance(value, numbers.Real):
        try:
            return float(value)
        except Exception:                 # noqa: BLE001
            return value
    return value


def _tag(value: float) -> str:
    """回傳非有限浮點值的標記字串。"""
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def sanitize_record(obj: Any) -> Any:
    """遞迴把非有限浮點值換成 ``None``，並在所屬 dict 內加註 ``__nonfinite``。

    只有 dict 的直接值會被加註（list 內的元素僅換成 ``None``，因為 list 沒有
    可掛標記的鍵位）。bool 是 int 的子類但不是浮點，原樣保留。

    Args:
        obj: 任意可 JSON 化的巢狀結構。

    Returns:
        淨化後的新結構（不就地修改輸入）。
    """
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        marks: dict[str, str] = {}
        for key, val in obj.items():
            num = as_number(val)
            if isinstance(num, float) and not math.isfinite(num):
                out[key] = None
                if isinstance(key, str):
                    marks[f"{key}{NONFINITE_SUFFIX}"] = _tag(num)
                else:
                    # ★非字串 key 掛不上 <key>__nonfinite 標記——不能靜默丟掉
                    #   「這裡原本是 NaN」這件事，故集中記在一個列表欄位。
                    marks.setdefault(NONFINITE_SUFFIX, [])  # type: ignore[arg-type]
                    marks[NONFINITE_SUFFIX].append(  # type: ignore[union-attr]
                        {"key": str(key), "was": _tag(num)}
                    )
            else:
                out[key] = sanitize_record(val)
        # 標記後補；撞名時**不覆蓋**既有鍵，改掛 __nonfinite__dup 以免揭露消失
        for mkey, mval in marks.items():
            if mkey in out:
                out.setdefault(f"{mkey}__dup", mval)
            else:
                out[mkey] = mval
        return out
    if isinstance(obj, (list, tuple)):
        return [sanitize_record(v) for v in obj]
    num = as_number(obj)
    if isinstance(num, float) and not math.isfinite(num):
        return None
    return num


def restore_record(obj: Any) -> Any:
    """:func:`sanitize_record` 的反函式：把 ``null`` ＋ ``__nonfinite`` 標記還原成原值。

    ★為何需要：``metrics.json`` 的 ``startup`` event 內含完整的 config dump，
    若有呼叫端把它讀回來重新寫成 yaml 再 ``load_tes_config``，就會受影響——
    ``TESTrainConfig`` 有四個欄位（``tempw_init``／``tempw_final``／``jacw_init``／
    ``jacw_final``）預設為 ``float("nan")`` 當哨兵值，淨化後變成 ``null``，
    config round-trip 若不還原就會把 NaN 哨兵換成了 ``None``；同時多出來的
    ``__nonfinite`` 鍵會讓 config 載入器噴「未知欄位…將被忽略」。

    標記存在的目的正是讓這件事可逆：任何**把 metrics.json 的紀錄讀回來當資料用**
    的地方都應該先過本函式，而不是直接吃淨化後的形狀。

    ★可逆性的邊界：當 record 內**本來就有**一個名為
    ``<key>__nonfinite`` 的資料鍵、且 ``<key>`` 同時是非有限值時，round-trip
    **不可逆**——:func:`sanitize_record` 會把標記改掛到 ``__dup``，而本函式
    無法區分「原始資料」與「被擠開的標記」。實測
    ``{"loss": nan, "loss__nonfinite": nan}`` 還原後 ``loss`` 變 ``None``。
    該情形在實務上不可達（只有本模組自己產生 ``__nonfinite`` 後綴的鍵），
    故不為它增加複雜度，但**不得**宣稱無條件可逆；行為由
    ``tests/nca/tes/test_jsonl_sanitize.py`` 的撞名測試釘住。

    Args:
        obj: 由 :func:`sanitize_record` 產生（或 ``json.loads`` 讀回）的結構。

    Returns:
        還原後的新結構；``__nonfinite`` 標記鍵會被移除。
    """
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for key, val in obj.items():
            if isinstance(key, str) and key.endswith(NONFINITE_SUFFIX):
                continue                       # 標記本身不進結果
            tag = obj.get(f"{key}{NONFINITE_SUFFIX}") if isinstance(key, str) else None
            if val is None and isinstance(tag, str) and tag in _TAG_TO_VALUE:
                out[key] = _TAG_TO_VALUE[tag]
            else:
                out[key] = restore_record(val)
        # 非字串 key 的集中揭露（sanitize 的 NONFINITE_SUFFIX 列表欄位）
        bulk = obj.get(NONFINITE_SUFFIX)
        if isinstance(bulk, list):
            for item in bulk:
                if not isinstance(item, dict):
                    continue
                k, was = item.get("key"), item.get("was")
                if k in out and out[k] is None and was in _TAG_TO_VALUE:
                    out[k] = _TAG_TO_VALUE[was]
        return out
    if isinstance(obj, (list, tuple)):
        return [restore_record(v) for v in obj]
    return obj


def dumps_record(record: Any) -> str:
    """把一筆 record 序列化成合法的單行 JSON。

    Args:
        record: 待寫入 metrics.json 的 dict。

    Returns:
        單行 JSON 字串（不含換行）。

    Raises:
        ValueError: 淨化後仍存在**Python float** 的非有限值。

    Note:
        ``allow_nan=False`` **不是**完整的安全網——它只檢查 Python ``float``，
        numpy／torch 標量會先被 ``default=str`` 吃掉、永遠碰不到那道檢查（見模組
        docstring）。真正的保障來自 :func:`as_number` 的型別正規化；
        ``allow_nan=False`` 只是最後一道對 Python float 的斷言。
        回歸鎖見 ``tests/nca/tes/test_jsonl_sanitize.py``。
    """
    return json.dumps(sanitize_record(record), default=str, allow_nan=False)
