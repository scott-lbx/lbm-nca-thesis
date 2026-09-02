"""載母群體 NPZ → featurize → fit ParamPCA（凍結存檔）→ 解釋變異 / PC 意義 / scatter / locality。

用法：python -m lbmnca.morphospace.param_pca.inspect \
            --pop data/design_space/population.npz \
            --save-transform data/design_space/param_pca.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lbmnca.morphospace.param_pca import config as C
from lbmnca.morphospace.param_pca.features import (
    DESCRIPTOR_NAMES, FEATURE_NAMES, descriptors, descriptors_population,
    featurize, featurize_population, iter_layouts,
)
from lbmnca.morphospace.param_pca.model import ParamPCA
from lbmnca.paths import design_space_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=str, default=design_space_dir() / "population.npz")
    ap.add_argument("--save-transform", type=str, default=design_space_dir() / "param_pca.npz")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    d = np.load(Path(cfg.pop))
    params, counts = d["params"], d["counts"]
    layouts = list(iter_layouts(params, counts))
    print(f"[load] {cfg.pop}: {len(counts)} 佈局、{len(params)} 建物")

    X = featurize_population(params, counts)            # (N,21)
    D = descriptors_population(params, counts, C.W_TOTAL, C.H_TOTAL)
    print(f"[feat] X {X.shape}（{len(FEATURE_NAMES)} 維）")

    pca = ParamPCA().fit(X)
    pca.save(cfg.save_transform)
    print(f"[pca] 凍結 transform 存 {cfg.save_transform}（丟零變異列 {(~pca.keep_).sum()}）")
    scores = pca.transform(X)
    evr = pca.explained_variance_ratio_

    print("\n=== 1. 解釋變異 ===")
    cum = 0.0
    for i in range(min(6, len(evr))):
        cum += evr[i]
        print(f"  PC{i+1}: {evr[i]*100:5.2f}%   cum={cum*100:5.2f}%")

    print("\n=== 2. PC1/PC2 代表什麼（|r| vs 描述子）===")
    for pc in range(2):
        rs = sorted(((abs(np.corrcoef(scores[:, pc], D[:, j])[0, 1]), DESCRIPTOR_NAMES[j])
                     for j in range(len(DESCRIPTOR_NAMES))), reverse=True)
        print(f"  PC{pc+1} ~ " + ", ".join(f"{nm}:{r:.2f}" for r, nm in rs[:3]))

    print("\n=== 3. 小編輯 locality（最大建物 cx +9/18/27px → PCA-2D 位移）===")
    rng = np.random.RandomState(cfg.seed)
    samp = scores[rng.choice(len(scores), min(500, len(scores)), replace=False), :2]
    dd = np.linalg.norm(samp[:, None, :] - samp[None, :, :], axis=2)  # pairwise（numpy）
    unrel = float(np.median(dd[np.triu_indices(len(samp), k=1)]))
    print(f"  無關 mask PCA-2D 距離中位數 = {unrel:.3f}")
    for si in range(5):
        b0 = layouts[si]
        big = int(np.argmax(b0[:, 2] * b0[:, 3]))
        base = pca.transform_one(featurize(b0))[:2]
        ds = []
        for dx in (9, 18, 27):
            b = b0.copy(); b[big, 0] += dx
            ds.append(float(np.linalg.norm(pca.transform_one(featurize(b))[:2] - base)))
        mono = "單調" if ds[0] <= ds[1] <= ds[2] else "非單調"
        print(f"  root{si}: +9={ds[0]:.3f} +18={ds[1]:.3f} +27={ds[2]:.3f} "
              f"（+27/無關={ds[2]/unrel:.3f}, {mono}）")

    _plot(scores, D, evr, Path(cfg.pop).parent / "_param_pca_inspect.png")


def _plot(scores, D, evr, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for k, (cj, lbl) in enumerate([(3, "n_bldg"), (5, "mean_w")]):
        sc = axes[k].scatter(scores[:, 0], scores[:, 1], s=5, c=D[:, cj], cmap="coolwarm", alpha=0.5)
        fig.colorbar(sc, ax=axes[k], label=lbl)
        axes[k].set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); axes[k].set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
        axes[k].set_title(f"param-PCA-2D 色={lbl}")
    axes[2].plot(np.arange(1, len(evr) + 1), np.cumsum(evr) * 100, "o-")
    axes[2].axhline(80, color="r", ls="--", lw=0.8, label="80%")
    axes[2].set_xlabel("PC 數"); axes[2].set_ylabel("累積解釋變異 %")
    axes[2].set_title("param-PCA scree"); axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
