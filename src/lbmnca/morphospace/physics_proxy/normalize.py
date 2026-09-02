"""Physics-Proxy Feature 歸一化模組。

提供 ProxyStats dataclass 及對應的 fit / normalize / save / load 工具函式。
所有特徵統一採用 z-score 正規化（mean=0, std=1）。

使用流程：
    1. 訓練資料集上 ``fit_stats(raw_features)`` → ProxyStats
    2. ``save_stats(stats, path)`` → p_stats.json（持久化）
    3. 推論時 ``load_stats(path)`` → ProxyStats
    4. ``normalize(raw_p, stats)`` → 正規化特徵向量
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM, FEATURE_NAMES


# ---------------------------------------------------------------------------
# ProxyStats dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProxyStats:
    """Physics-Proxy 特徵的歸一化統計。

    Attributes:
        mean: 各特徵的訓練集均值，shape [FEATURE_DIM]。
        std:  各特徵的訓練集標準差，shape [FEATURE_DIM]；
              std < EPS 的維度會被夾到 EPS，防止除以零。
        feature_names: 特徵名稱清單（與 FEATURE_NAMES 對應）。
        n_samples: fit 時使用的樣本數。
    """

    mean: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_DIM, dtype=np.float32))
    std: np.ndarray = field(default_factory=lambda: np.ones(FEATURE_DIM, dtype=np.float32))
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    n_samples: int = 0

    # std 小於此值時夾到此值，防止歸一化除以零
    _EPS: float = field(default=1e-6, init=False, repr=False)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.std = np.asarray(self.std, dtype=np.float32)
        # 夾住過小的 std
        self.std = np.maximum(self.std, self._EPS)


# ---------------------------------------------------------------------------
# 核心工具函式
# ---------------------------------------------------------------------------


def fit_stats(raw_features: np.ndarray) -> ProxyStats:
    """從原始特徵矩陣計算 ProxyStats。

    Args:
        raw_features: shape [N, FEATURE_DIM]，N 個樣本的 raw 特徵。
                      允許含有 inf（例如 tortuosity 不可達時），
                      計算時會自動排除 inf 樣本（per feature）。

    Returns:
        ProxyStats：包含 mean、std、feature_names、n_samples。

    Raises:
        ValueError: raw_features 的列數不等於 FEATURE_DIM。
    """
    if raw_features.ndim != 2 or raw_features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"raw_features 形狀應為 [N, {FEATURE_DIM}]，"
            f"實際收到 {raw_features.shape}。"
        )

    n_samples = raw_features.shape[0]
    means = np.zeros(FEATURE_DIM, dtype=np.float64)
    stds = np.zeros(FEATURE_DIM, dtype=np.float64)

    for d in range(FEATURE_DIM):
        col = raw_features[:, d]
        finite_mask = np.isfinite(col)
        finite_vals = col[finite_mask]
        if finite_vals.size > 1:
            means[d] = float(finite_vals.mean())
            stds[d] = float(finite_vals.std())
        elif finite_vals.size == 1:
            means[d] = float(finite_vals[0])
            stds[d] = 1.0  # 只有一個樣本，std 無意義，設為 1.0
        else:
            means[d] = 0.0
            stds[d] = 1.0  # 全部 inf，設 fallback

    return ProxyStats(
        mean=means.astype(np.float32),
        std=stds.astype(np.float32),
        feature_names=list(FEATURE_NAMES),
        n_samples=n_samples,
    )


def normalize(raw_p: np.ndarray, stats: ProxyStats) -> np.ndarray:
    """將 raw physics-proxy 特徵 z-score 正規化。

    Args:
        raw_p: Raw 特徵，shape [FEATURE_DIM] 或 [N, FEATURE_DIM]。
               inf 值在歸一化後會保持為 inf（呼叫端需處理）。
        stats: 由 fit_stats 或 load_stats 取得的 ProxyStats。

    Returns:
        正規化特徵，與輸入同 shape，dtype=float32。
    """
    raw = np.asarray(raw_p, dtype=np.float32)
    return (raw - stats.mean) / stats.std


def denormalize(norm_p: np.ndarray, stats: ProxyStats) -> np.ndarray:
    """將正規化特徵還原為 raw 值。

    Args:
        norm_p: 正規化特徵，shape [FEATURE_DIM] 或 [N, FEATURE_DIM]。
        stats: ProxyStats。

    Returns:
        Raw 特徵，與輸入同 shape，dtype=float32。
    """
    normed = np.asarray(norm_p, dtype=np.float32)
    return normed * stats.std + stats.mean


# ---------------------------------------------------------------------------
# 持久化（JSON）
# ---------------------------------------------------------------------------


def save_stats(stats: ProxyStats, path: str | Path) -> None:
    """將 ProxyStats 序列化為 JSON 檔案。

    Args:
        stats: 要儲存的 ProxyStats。
        path: 輸出路徑（例如 ``data/p_stats.json``）。
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "feature_names": stats.feature_names,
        "n_samples": stats.n_samples,
        "mean": stats.mean.tolist(),
        "std": stats.std.tolist(),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_stats(path: str | Path) -> ProxyStats:
    """從 JSON 檔案讀取 ProxyStats。

    Args:
        path: p_stats.json 的路徑。

    Returns:
        ProxyStats。

    Raises:
        FileNotFoundError: 檔案不存在。
        KeyError: JSON 缺少必要欄位。
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"p_stats.json 不存在：{in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        payload: dict = json.load(f)

    return ProxyStats(
        mean=np.array(payload["mean"], dtype=np.float32),
        std=np.array(payload["std"], dtype=np.float32),
        feature_names=payload["feature_names"],
        n_samples=payload["n_samples"],
    )


# ---------------------------------------------------------------------------
# 批量計算輔助（供 train.py 或 CLI 呼叫）
# ---------------------------------------------------------------------------


def compute_and_fit_stats(
    masks: list[np.ndarray],
    sim_cfg_list: list[dict],
) -> tuple[np.ndarray, ProxyStats]:
    """對一批 mask 計算 raw 特徵並 fit ProxyStats。

    Args:
        masks: mask 清單，每個 [H, W]，1=固體，0=流體。
        sim_cfg_list: 對應的模擬設定清單（長度與 masks 相同）。

    Returns:
        (raw_matrix [N, FEATURE_DIM], stats)。

    Raises:
        ValueError: masks 與 sim_cfg_list 長度不符。
    """
    from .features import compute_all

    if len(masks) != len(sim_cfg_list):
        raise ValueError(
            f"masks 數量 ({len(masks)}) 與 sim_cfg_list 數量 ({len(sim_cfg_list)}) 不符。"
        )

    raw_list: list[np.ndarray] = []
    for mask, sim_cfg in zip(masks, sim_cfg_list):
        try:
            raw_list.append(compute_all(mask, sim_cfg))
        except ValueError:
            # 空 mask（全固體或全流體）→ 跳過
            raw_list.append(np.full(FEATURE_DIM, np.nan, dtype=np.float32))

    raw_matrix = np.stack(raw_list, axis=0)  # [N, FEATURE_DIM]
    stats = fit_stats(raw_matrix)
    return raw_matrix, stats
