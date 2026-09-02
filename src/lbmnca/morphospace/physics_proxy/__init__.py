"""Physics-Proxy Feature 模組。

提供從 binary geometry mask 解析式計算物理代理特徵的公共 API，
無需執行 CFD 模擬。

公共介面：
    compute_all   : 計算全部 22 個特徵（raw，未歸一化）
    FEATURE_NAMES : 特徵名稱清單（長度 22）
    FEATURE_DIM   : 特徵維度（= 22）
    ProxyStats    : 歸一化統計 dataclass
    fit_stats     : 從訓練集 mask 計算 mean/std
    normalize     : z-score 正規化
    denormalize   : 反正規化
    save_stats    : 持久化 ProxyStats → p_stats.json
    load_stats    : 讀取 p_stats.json → ProxyStats
"""

from .features import FEATURE_DIM, FEATURE_NAMES, compute_all
from .normalize import (
    ProxyStats,
    denormalize,
    fit_stats,
    load_stats,
    normalize,
    save_stats,
)

__all__ = [
    "compute_all",
    "FEATURE_NAMES",
    "FEATURE_DIM",
    "ProxyStats",
    "fit_stats",
    "normalize",
    "denormalize",
    "save_stats",
    "load_stats",
]
