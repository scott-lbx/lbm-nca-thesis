"""val 子套件。

原本的 ``Validator`` 類已刪除：它是一次大重構產生的平行副本，全 repo 零實例化
（``TESTrainer`` 一直用自己的內嵌實作），卻在每次修正時被要求同步維護。與其
維護兩份必然分岔的實作，直接刪除死碼；聚合語義改由 :mod:`aggregate` 的純函式
單一持有。
"""
from .aggregate import POOL_KEYS, POOLED_METRICS, aggregate_val_metrics

__all__ = ["aggregate_val_metrics", "POOL_KEYS", "POOLED_METRICS"]
