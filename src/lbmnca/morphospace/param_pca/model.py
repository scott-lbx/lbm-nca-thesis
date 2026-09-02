"""ParamPCA：StandardScaler + PCA（numpy SVD），凍結 transform fit/transform/save/load。

母群體 fit 一次 → 凍結（存 mean/scale/components/pca_mean）→ 任何新 mask 投影進**固定**空間。
這就是「母群體凍結破循環依賴」的落點。零 sklearn（純 numpy SVD）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ParamPCA:
    """標準化 + PCA。fit 後欄位皆 numpy；save/load 用 NPZ。"""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None        # (D,) 標準化中心
        self.scale_: np.ndarray | None = None       # (D,) 標準化尺度（零變異列=1，不縮放）
        self.keep_: np.ndarray | None = None        # (D,) bool，保留（非零變異）列
        self.components_: np.ndarray | None = None   # (k, D_keep) 主成分（列）
        self.explained_variance_ratio_: np.ndarray | None = None
        self.pca_mean_: np.ndarray | None = None     # (D_keep,) 標準化後再 center

    def fit(self, X: np.ndarray, n_components: int | None = None) -> "ParamPCA":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(0)
        std = X.std(0)
        self.keep_ = std > 1e-9                       # 丟零變異列（避免 /0）
        self.scale_ = np.where(self.keep_, std, 1.0)
        Z = (X - self.mean_) / self.scale_
        Z = Z[:, self.keep_]
        self.pca_mean_ = Z.mean(0)
        Zc = Z - self.pca_mean_
        _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        evr = (S ** 2) / (S ** 2).sum()
        k = Vt.shape[0] if n_components is None else min(n_components, Vt.shape[0])
        self.components_ = Vt[:k]
        self.explained_variance_ratio_ = evr[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X (N,D) → scores (N,k)。"""
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) / self.scale_
        Z = Z[:, self.keep_] - self.pca_mean_
        return Z @ self.components_.T

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x[None, :])[0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean_, scale=self.scale_, keep=self.keep_,
                 components=self.components_, evr=self.explained_variance_ratio_,
                 pca_mean=self.pca_mean_)

    @classmethod
    def load(cls, path: str | Path) -> "ParamPCA":
        d = np.load(Path(path))
        m = cls()
        m.mean_, m.scale_, m.keep_ = d["mean"], d["scale"], d["keep"]
        m.components_, m.explained_variance_ratio_ = d["components"], d["evr"]
        m.pca_mean_ = d["pca_mean"]
        return m
