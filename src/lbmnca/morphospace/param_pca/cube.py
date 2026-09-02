"""Cube4D — 4D 設計 cube：3 geom PC（whiten）+ Re（標準化）= 等權 4 軸，凍結。

geom = ParamPCA 前 3 主軸（PC1=密度、PC2=建物大小、PC3=位置）→ 各除以母群體 std（whiten 等權）。
Re   = uniform[re_lo,re_hi] 假設 → z 標準化（mean=(lo+hi)/2, std=(hi-lo)/√12）。
四軸皆 ~單位 std → Sample-Elimination / LHS / MST 距離等權。

凍結存 `cube4d.npz`（含 ParamPCA 全欄 + geom_std + re 範圍 + n_geom）。

用法：python -m lbmnca.morphospace.param_pca.cube \
            --pop data/design_space/population_big.npz \
            --out data/design_space/cube4d.npz --n-geom 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lbmnca.morphospace.param_pca.features import featurize_population
from lbmnca.morphospace.param_pca.model import ParamPCA
from lbmnca.paths import design_space_dir


class Cube4D:
    """3 geom PC（whiten）+ Re（標準化）= 4D 等權 cube。"""

    def __init__(self, pca: ParamPCA, geom_std: np.ndarray,
                 re_lo: float, re_hi: float, n_geom: int = 3) -> None:
        self.pca = pca
        self.geom_std = np.asarray(geom_std, float)     # (n_geom,)
        self.re_lo = float(re_lo)
        self.re_hi = float(re_hi)
        self.n_geom = int(n_geom)

    # ── fit ────────────────────────────────────────────────────────────
    @classmethod
    def fit(cls, X: np.ndarray, re_lo: float = 150, re_hi: float = 500,
            n_geom: int = 3) -> "Cube4D":
        pca = ParamPCA().fit(X)
        scores = pca.transform(X)[:, :n_geom]
        geom_std = scores.std(0)
        return cls(pca, geom_std, re_lo, re_hi, n_geom)

    # ── 投影 ───────────────────────────────────────────────────────────
    def geom_coords(self, X_feat: np.ndarray) -> np.ndarray:
        """featurize 後 X (N,21) → whiten 後 geom (N,n_geom)。"""
        return self.pca.transform(X_feat)[:, :self.n_geom] / self.geom_std

    def re_coord(self, re) -> np.ndarray:
        mid = (self.re_lo + self.re_hi) / 2.0
        sd = (self.re_hi - self.re_lo) / np.sqrt(12.0)
        return (np.asarray(re, float) - mid) / sd

    def to_cube(self, X_feat: np.ndarray, re) -> np.ndarray:
        """X (N,21) + re (N,) → 4D cube (N, n_geom+1)。"""
        g = self.geom_coords(X_feat)
        r = self.re_coord(re).reshape(-1, 1)
        return np.concatenate([g, r], axis=1)

    # ── 存 / 載 ────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        p = self.pca
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=p.mean_, scale=p.scale_, keep=p.keep_,
                 components=p.components_, evr=p.explained_variance_ratio_,
                 pca_mean=p.pca_mean_, geom_std=self.geom_std,
                 re_lo=self.re_lo, re_hi=self.re_hi, n_geom=self.n_geom)

    @classmethod
    def load(cls, path: str | Path) -> "Cube4D":
        d = np.load(Path(path))
        pca = ParamPCA()
        pca.mean_, pca.scale_, pca.keep_ = d["mean"], d["scale"], d["keep"]
        pca.components_, pca.explained_variance_ratio_ = d["components"], d["evr"]
        pca.pca_mean_ = d["pca_mean"]
        return cls(pca, d["geom_std"], float(d["re_lo"]), float(d["re_hi"]), int(d["n_geom"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=str, default=design_space_dir() / "population_big.npz")
    ap.add_argument("--out", type=str, default=design_space_dir() / "cube4d.npz")
    ap.add_argument("--n-geom", type=int, default=3)
    ap.add_argument("--re-lo", type=float, default=150)
    ap.add_argument("--re-hi", type=float, default=500)
    cfg = ap.parse_args()

    d = np.load(Path(cfg.pop))
    print(f"[load] {cfg.pop}: {int(d['counts'].shape[0])} 佈局 → featurize ...")
    X = featurize_population(d["params"], d["counts"])
    cube = Cube4D.fit(X, cfg.re_lo, cfg.re_hi, cfg.n_geom)
    cube.save(cfg.out)
    evr = cube.pca.explained_variance_ratio_
    print(f"[cube4d] n_geom={cfg.n_geom} + Re → 4D 凍結 {cfg.out}")
    print(f"  geom PC evr: {(evr[:cfg.n_geom]*100).round(2)}  cum={evr[:cfg.n_geom].sum()*100:.1f}%")
    print(f"  geom_std（whiten 前）: {cube.geom_std.round(3)}")
    print(f"  Re 軸: uniform[{cfg.re_lo},{cfg.re_hi}] → z（std={(cfg.re_hi-cfg.re_lo)/np.sqrt(12):.1f}）")
    print("  → 四軸標準化後皆 ~單位 std，SE/LHS/MST 距離等權")


if __name__ == "__main__":
    main()
