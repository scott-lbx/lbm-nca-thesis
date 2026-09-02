"""PCA 1D vs 2D 解釋性對照（自包含）。

三個量度：
  1. 累積解釋變異（1D=PC1、2D=PC1+2）+ 重建殘差比
  2. 描述子可回復度 R²（從 k PC 線性回歸各幾何描述子）→ 看「第 2 維加了什麼」
  3. 視覺：PC1 一維帶（色=mean_w）vs 2D 散布（色=mean_w）→ 1D 是否把「建物大小」壓掉

用法：python -m lbmnca.morphospace.param_pca.compare_dims \
            --pop data/design_space/population_big.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lbmnca.morphospace.param_pca import config as C
from lbmnca.morphospace.param_pca.features import (
    DESCRIPTOR_NAMES, descriptors_population, featurize_population,
)
from lbmnca.morphospace.param_pca.model import ParamPCA
from lbmnca.paths import design_space_dir


def _r2(scores_k: np.ndarray, y: np.ndarray) -> float:
    """從 k 維 PC 線性回歸 y 的 R²（含截距）。"""
    A = np.column_stack([scores_k, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=str, default=design_space_dir() / "population_big.npz")
    cfg = ap.parse_args()

    d = np.load(Path(cfg.pop))
    params, counts = d["params"], d["counts"]
    print(f"[load] {cfg.pop}: {len(counts)} 佈局、{len(params)} 建物")

    X = featurize_population(params, counts)
    D = descriptors_population(params, counts, C.W_TOTAL, C.H_TOTAL)
    pca = ParamPCA().fit(X)
    scores = pca.transform(X)
    evr = pca.explained_variance_ratio_

    print("\n=== 1. 累積解釋變異 / 重建殘差 ===")
    cum = np.cumsum(evr)
    for k in (1, 2, 3, 5):
        print(f"  {k}D: 解釋={cum[k-1]*100:5.2f}%   殘差={ (1-cum[k-1])*100:5.2f}%")
    print(f"  → 第 2 維淨增 = {evr[1]*100:.2f}%（{cum[1]*100:.1f}% - {cum[0]*100:.1f}%）")

    print("\n=== 2. 描述子可回復 R²（1D=PC1 vs 2D=PC1+2 vs 3D）===")
    print(f"  {'descriptor':14s}  1D     2D     3D    Δ(2D-1D)")
    for j, nm in enumerate(DESCRIPTOR_NAMES):
        r1 = _r2(scores[:, :1], D[:, j])
        r2 = _r2(scores[:, :2], D[:, j])
        r3 = _r2(scores[:, :3], D[:, j])
        print(f"  {nm:14s} {r1:5.2f}  {r2:5.2f}  {r3:5.2f}   {r2-r1:+.2f}")

    _plot(scores, D, evr, Path(cfg.pop).parent / "_compare_1d_2d.png")


def _plot(scores, D, evr, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    mean_w = D[:, DESCRIPTOR_NAMES.index("mean_w")]
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(1, 3, figsize=(21, 6))

    # panel1：累積解釋變異
    cum = np.cumsum(evr) * 100
    ax[0].bar(np.arange(1, 8), evr[:7] * 100, color="#88a", alpha=0.7, label="each PC")
    ax[0].plot(np.arange(1, 8), cum[:7], "o-r", label="cumulative")
    ax[0].axvline(1.5, color="0.6", ls="--"); ax[0].axvline(2.5, color="0.6", ls="--")
    ax[0].set_xlabel("PC"); ax[0].set_ylabel("解釋變異 %")
    ax[0].set_title(f"1D={cum[0]:.0f}%  2D={cum[1]:.0f}%（+{evr[1]*100:.0f}%）"); ax[0].legend()

    # panel2：1D（PC1 + jitter）色=mean_w → 看大小是否被壓掉
    sc = ax[1].scatter(scores[:, 0], rng.uniform(0, 1, len(scores)), s=5,
                       c=mean_w, cmap="viridis", alpha=0.5)
    fig.colorbar(sc, ax=ax[1], label="mean_w（建物大小）")
    ax[1].set_xlabel("PC1"); ax[1].set_yticks([])
    ax[1].set_title("1D（只 PC1）：同一 PC1 上 mean_w 混雜 → 大小被壓掉")

    # panel3：2D 色=mean_w → 看 PC2 是否分開大小
    sc = ax[2].scatter(scores[:, 0], scores[:, 1], s=5, c=mean_w, cmap="viridis", alpha=0.5)
    fig.colorbar(sc, ax=ax[2], label="mean_w（建物大小）")
    ax[2].set_xlabel(f"PC1 ({evr[0]*100:.0f}%)"); ax[2].set_ylabel(f"PC2 ({evr[1]*100:.0f}%)")
    ax[2].set_title("2D：PC2 把建物大小分開")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
