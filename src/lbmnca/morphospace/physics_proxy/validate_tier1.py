"""Tier 1 統計健全性驗證（Phase G4: 22-d 直接吃 PNG，無 R=64 resize）。

跑完後產出：
    1. data/g_vec/raw_native_<dataset>.npy — [N, 22] raw G_vec cache（PNG 原解析度，BBOX 計算；
       檔名依 --dataset-dir 命名空間化，見 dataset_cache_key()）
    2. docs/g_vec_tier1_report.md    — 綜合報告（spread, redundancy, degeneracy, distribution）

包含的測試：
    T1.1 spread 測試    ：(p90-p10)/(|median|+1e-3) > 0.05？
    T1.2 冗餘測試       ：|ρ_pair| ≤ 0.99 為 hard pass，0.95-0.99 候選冗餘
    T1.4 分布形狀檢查   ：percentile + unique_count
    退化驗證            ：unique > DEGENERACY_UNIQUE_LIMIT

使用方式：
    python -m lbmnca.morphospace.physics_proxy.validate_tier1 \\
        --dataset-dir dataset/CEDS_dataset/flat_diverse2000
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np
from PIL import Image

from lbmnca.paths import REPO_ROOT

from .features import FEATURE_DIM, FEATURE_NAMES, SimConfig, compute_all

# ---------------------------------------------------------------------------
# 資料集常數
# ---------------------------------------------------------------------------

# 無預設 mask pool：由呼叫端（CLI --dataset-dir，或 main() 的 masks_dir/meta_path
# 參數）指定，避免靜默套用某台機器上的特定資料集。

# Phase G4 起：不再做 R=64 resize；直接讀 PNG → invert → 二值化 → compute_all
# PNG 慣例：0=building, 255=fluid → 反轉得 1=building（compute_all 期望的格式）
PNG_BUILDING_THRESHOLD: int = 128

# 用於 E1（re_eff = Re*L/B1）與 E2（U/(1-C2)）；其他 features 不需要
DEFAULT_SIM_CFG: SimConfig = {
    "Re": 500.0,
    "U_inlet_lu": 0.04,
    "L_char_px": 6.0,
}

# 通過門檻
# T1.1：相對 spread（p90-p10）/ |median|，0.05 代表 middle 80% 的寬度至少佔 median 的 5%
T1_1_SPREAD_THRESHOLD: float = 0.05
# T1.2：高 corr 標記為 informational（不直接 fail），交給 Tier 3 LBM 決定
T1_2_CORR_HIGH: float = 0.99  # 極度重複（>0.99）才硬 fail
T1_2_CORR_REDUNDANT: float = 0.95  # 0.95-0.99 標記為 candidate redundant
DEGENERACY_UNIQUE_LIMIT: int = 5  # unique value 數量低於此值視為退化

CACHE_DIR: Path = REPO_ROOT / "data" / "g_vec"
REPORT_PATH: Path = REPO_ROOT / "docs" / "g_vec_tier1_report.md"


def dataset_cache_key(dataset_root: Path) -> str:
    """由 dataset 根目錄產生 cache 檔名用的識別字串，避免跨 dataset 撞名共用 cache。

    ``--dataset-dir``（及 ``--lbm-dataset-dir``）現為自由 CLI 參數，若 cache 檔名
    固定不變，先後對兩個不同 dataset 跑同一支腳本時第二次會靜默讀到第一次的
    stale cache（本專案禁止的靜默污染實驗）。格式：``<目錄名>-<8 hex 位絕對路徑
    hash>``；hash 部分用來區分兩個不同路徑但同名（如皆名為 ``masks``）的目錄。

    Args:
        dataset_root: dataset 根目錄（含 masks/ 與 mask_meta.json，或 LBM dataset
            根目錄）。

    Returns:
        可安全嵌入檔名的字串。
    """
    resolved = str(dataset_root.resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:8]
    return f"{dataset_root.name}-{digest}"


# ---------------------------------------------------------------------------
# 結果結構
# ---------------------------------------------------------------------------


@dataclass
class FeatureVerdict:
    """單一 feature 在 Tier 1 各測試的判定結果。"""

    name: str
    spread_ratio: float  # (p90-p10)/(|median|+1e-3)
    spread_pass: bool  # spread_ratio > T1_1_SPREAD_THRESHOLD
    max_corr_other: float
    max_corr_partner: str
    corr_redundant: bool  # 0.95 < |ρ| ≤ 0.99（候選冗餘）
    corr_extreme: bool  # |ρ| > 0.99（極度重複，硬 fail）
    n_unique: int
    is_degenerate: bool  # unique < 5
    p10: float
    p50: float
    p90: float

    @property
    def overall_pass(self) -> bool:
        # 硬 fail 條件：spread 過小、退化、極度重複
        return self.spread_pass and not self.is_degenerate and not self.corr_extreme


# ---------------------------------------------------------------------------
# 載入 + 計算 G_vec
# ---------------------------------------------------------------------------


def load_meta(meta_path: Path) -> list[dict]:
    """讀取 mask_meta.json。"""
    with meta_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_mask_native(mask_path: Path) -> np.ndarray:
    """讀 PNG → 反轉 → 二值化，回傳 [H, W] uint8 (1=building, 0=fluid)。

    PNG 慣例：0=building, 255=fluid。
    """
    arr = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return (arr < PNG_BUILDING_THRESHOLD).astype(np.uint8)


def compute_g_vec_batch(
    meta: list[dict],
    masks_dir: Path,
    sim_cfg: SimConfig,
) -> tuple[np.ndarray, list[str]]:
    """對 meta 中所有 mask 計算 G_vec（native 解析度，內部 BBOX）。

    Returns:
        (raw_matrix [N, 22] float32, mask_ids [N]) — 失敗的 entry 以 NaN row 填入。
    """
    raw_rows: list[np.ndarray] = []
    mask_ids: list[str] = []
    n_skip = 0
    n = len(meta)

    for i, entry in enumerate(meta):
        path = masks_dir / entry["file_name"]
        mid = entry.get("mask_id", entry["file_name"])
        try:
            binary = load_mask_native(path)
            raw_p = compute_all(binary, sim_cfg)
            raw_rows.append(raw_p)
        except Exception as e:
            print(f"  [skip] {mid}: {e}")
            raw_rows.append(np.full(FEATURE_DIM, np.nan, dtype=np.float32))
            n_skip += 1
        mask_ids.append(mid)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n} 計算完畢")

    matrix = np.stack(raw_rows, axis=0)
    print(f"[compute] 完成，跳過 {n_skip} 個無效 mask")
    return matrix, mask_ids


def load_or_compute(
    cache_path: Path,
    meta: list[dict],
    masks_dir: Path,
    sim_cfg: SimConfig,
    force: bool = False,
) -> np.ndarray:
    """從 cache 讀取 G_vec，或計算後存入 cache。"""
    if cache_path.exists() and not force:
        print(f"[cache] 載入 {cache_path}")
        return np.load(cache_path)
    print(f"[cache] 重新計算（native PNG + BBOX）...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    matrix, _ = compute_g_vec_batch(meta, masks_dir, sim_cfg)
    np.save(cache_path, matrix)
    print(f"[cache] 已儲存到 {cache_path}")
    return matrix


# ---------------------------------------------------------------------------
# Tier 1 測試
# ---------------------------------------------------------------------------


def t1_1_spread(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T1.1 robust 相對 spread 測試。

    spread_ratio = (p90 - p10) / (|median| + 1e-3)

    對「90% 樣本是同一值 + 少數 outlier」的退化情況有效。
    避免 fallback 上限值或極端 outlier 拉爆 (max-min) 造成假象低變異。

    Args:
        G: [N, 22] raw G_vec。

    Returns:
        (spread_ratio [22], pass_mask [22] bool)
    """
    spread = np.zeros(FEATURE_DIM, dtype=np.float64)
    for d in range(FEATURE_DIM):
        col = G[:, d]
        finite = col[np.isfinite(col)]
        if finite.size < 10:
            spread[d] = 0.0
            continue
        p10 = float(np.percentile(finite, 10))
        p50 = float(np.percentile(finite, 50))
        p90 = float(np.percentile(finite, 90))
        denom = abs(p50) + 1e-3
        spread[d] = (p90 - p10) / denom
    return spread, spread > T1_1_SPREAD_THRESHOLD


def t1_2_redundancy(
    G: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """T1.2 冗餘測試（pairwise correlation）。

    對每個 feature 找與其他 feature 的最大 |ρ|。
    分兩級門檻：
      - 0.95 < |ρ| ≤ 0.99：候選冗餘（informational，留給 Tier 3）
      - |ρ| > 0.99：極度重複，硬 fail

    Returns:
        (max_corr [22], redundant_mask [22], extreme_mask [22], partner_names [22])
    """
    finite_mask = np.all(np.isfinite(G), axis=1)
    if finite_mask.sum() < 10:
        return (
            np.zeros(FEATURE_DIM),
            np.zeros(FEATURE_DIM, dtype=bool),
            np.zeros(FEATURE_DIM, dtype=bool),
            [""] * FEATURE_DIM,
        )

    G_finite = G[finite_mask]
    stds = G_finite.std(axis=0)
    safe = stds > 1e-8
    corr_full = np.zeros((FEATURE_DIM, FEATURE_DIM), dtype=np.float64)
    if safe.sum() >= 2:
        corr_full[np.ix_(safe, safe)] = np.corrcoef(G_finite[:, safe].T)

    max_corr = np.zeros(FEATURE_DIM, dtype=np.float64)
    partners = [""] * FEATURE_DIM
    for d in range(FEATURE_DIM):
        if not safe[d]:
            continue
        row = np.abs(corr_full[d]).copy()
        row[d] = 0.0
        idx = int(np.argmax(row))
        max_corr[d] = float(row[idx])
        partners[d] = FEATURE_NAMES[idx]

    redundant = (max_corr > T1_2_CORR_REDUNDANT) & (max_corr <= T1_2_CORR_HIGH)
    extreme = max_corr > T1_2_CORR_HIGH
    return max_corr, redundant, extreme, partners


def t1_4_distribution(
    G: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """T1.4 分布形狀（percentile + unique count）。

    Returns:
        (n_unique [22], p10 [22], p50 [22], p90 [22])
    """
    n_unique = np.zeros(FEATURE_DIM, dtype=np.int64)
    p10 = np.zeros(FEATURE_DIM, dtype=np.float64)
    p50 = np.zeros(FEATURE_DIM, dtype=np.float64)
    p90 = np.zeros(FEATURE_DIM, dtype=np.float64)
    for d in range(FEATURE_DIM):
        col = G[:, d]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        n_unique[d] = int(np.unique(np.round(finite, decimals=6)).size)
        p10[d] = float(np.percentile(finite, 10))
        p50[d] = float(np.percentile(finite, 50))
        p90[d] = float(np.percentile(finite, 90))
    return n_unique, p10, p50, p90


def degeneracy_check(G: np.ndarray, n_unique: np.ndarray) -> np.ndarray:
    """退化判定：unique value < DEGENERACY_UNIQUE_LIMIT。

    Returns:
        is_degenerate [22] bool
    """
    return n_unique < DEGENERACY_UNIQUE_LIMIT


# ---------------------------------------------------------------------------
# 報告產生
# ---------------------------------------------------------------------------


def build_verdicts(
    spread: np.ndarray,
    spread_pass: np.ndarray,
    max_corr: np.ndarray,
    corr_redundant: np.ndarray,
    corr_extreme: np.ndarray,
    partners: list[str],
    n_unique: np.ndarray,
    is_degen: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> list[FeatureVerdict]:
    """組合各 feature 的最終判定。"""
    out: list[FeatureVerdict] = []
    for d in range(FEATURE_DIM):
        out.append(
            FeatureVerdict(
                name=FEATURE_NAMES[d],
                spread_ratio=float(spread[d]),
                spread_pass=bool(spread_pass[d]),
                max_corr_other=float(max_corr[d]),
                max_corr_partner=partners[d],
                corr_redundant=bool(corr_redundant[d]),
                corr_extreme=bool(corr_extreme[d]),
                n_unique=int(n_unique[d]),
                is_degenerate=bool(is_degen[d]),
                p10=float(p10[d]),
                p50=float(p50[d]),
                p90=float(p90[d]),
            )
        )
    return out


def write_report(
    verdicts: list[FeatureVerdict], n_samples: int, out_path: Path
) -> None:
    """寫出 g_vec_tier1_report.md。"""
    lines: list[str] = []
    lines.append("# G_vec Tier 1 統計健全性驗證報告\n")
    lines.append("**資料集**：flat_diverse2000（N=2000）")
    lines.append(f"**有效樣本**：{n_samples}")
    lines.append(
        f"**硬 fail 條件**：spread_ratio ≤ {T1_1_SPREAD_THRESHOLD}（近常數）, "
        f"unique ≤ {DEGENERACY_UNIQUE_LIMIT}（退化）, |ρ| > {T1_2_CORR_HIGH}（極度重複）"
    )
    lines.append(
        f"**informational**：{T1_2_CORR_REDUNDANT} < |ρ| ≤ {T1_2_CORR_HIGH} 標記為候選冗餘，留給 Tier 3 LBM 判定。"
    )
    lines.append("")
    lines.append("## 一、總表（按特徵 ID 排序）\n")
    lines.append(
        "| ID | 名稱 | spread | spread_pass | max\\|ρ\\| | partner | corr 標記 | "
        "n_unique | degen? | p10 / p50 / p90 | 整體 |"
    )
    lines.append(
        "|----|------|--------|-------------|---------|---------|-----------|"
        "----------|--------|------------------|------|"
    )
    for v in verdicts:
        sp_p = "[OK]" if v.spread_pass else "[FAIL]"
        if v.corr_extreme:
            corr_tag = "[EXTREME]"
        elif v.corr_redundant:
            corr_tag = "[redundant?]"
        else:
            corr_tag = "[OK]"
        deg_p = "[FAIL]" if v.is_degenerate else "[OK]"
        ovr = "[PASS]" if v.overall_pass else "[FAIL]"
        partner_short = (
            v.max_corr_partner.split("_", 1)[0] if v.max_corr_partner else "-"
        )
        lines.append(
            f"| {v.name.split('_', 1)[0]} | `{v.name}` | "
            f"{v.spread_ratio:.3f} | {sp_p} | "
            f"{v.max_corr_other:.3f} | {partner_short} | {corr_tag} | "
            f"{v.n_unique} | {deg_p} | "
            f"{v.p10:.3g} / {v.p50:.3g} / {v.p90:.3g} | {ovr} |"
        )
    lines.append("")

    lines.append("## 二、退化驗證（R²=1.0 三疑點）\n")
    suspect_ids = [
        "B3_void_connectivity",
        "B4_max_enclosed_fluid_area_ratio",
        "F3_outlet_reachability",
    ]
    for v in verdicts:
        if v.name in suspect_ids:
            verdict = (
                "**退化**（unique value 不足）"
                if v.is_degenerate
                else "**真實簡單**（分布合理但低變異）"
            )
            lines.append(
                f"- `{v.name}`：n_unique={v.n_unique}, "
                f"p10/50/90 = {v.p10:.4g} / {v.p50:.4g} / {v.p90:.4g} → {verdict}"
            )
    lines.append("")

    lines.append("## 三、硬 fail features（必須砍）\n")
    failed = [v for v in verdicts if not v.overall_pass]
    if not failed:
        lines.append("（無）")
    else:
        for v in failed:
            reasons: list[str] = []
            if not v.spread_pass:
                reasons.append(f"spread={v.spread_ratio:.3f} ≤ {T1_1_SPREAD_THRESHOLD}")
            if v.is_degenerate:
                reasons.append(f"unique={v.n_unique} ≤ {DEGENERACY_UNIQUE_LIMIT}")
            if v.corr_extreme:
                reasons.append(
                    f"|ρ|={v.max_corr_other:.3f} > {T1_2_CORR_HIGH} with {v.max_corr_partner}"
                )
            lines.append(f"- `{v.name}`：{'; '.join(reasons)}")
    lines.append("")

    lines.append("## 四、候選冗餘 features（informational，留給 Tier 3）\n")
    redundant = [v for v in verdicts if v.corr_redundant and v.overall_pass]
    if not redundant:
        lines.append("（無）")
    else:
        for v in redundant:
            lines.append(
                f"- `{v.name}` ↔ `{v.max_corr_partner}`：|ρ|={v.max_corr_other:.3f}"
            )
    lines.append("")

    lines.append("## 五、通過 features 列表\n")
    passed = [v for v in verdicts if v.overall_pass]
    lines.append(f"通過數：{len(passed)} / {FEATURE_DIM}\n")
    for v in passed:
        tag = " (候選冗餘)" if v.corr_redundant else ""
        lines.append(f"- `{v.name}`{tag}")
    lines.append("")

    lines.append("---\n")
    lines.append("*由 `src/physics_proxy/validate_tier1.py` 產生。*")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] 已寫入 {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(masks_dir: Path, meta_path: Path, force_recompute: bool = False) -> None:
    """跑完整 Tier 1 驗證。

    Args:
        masks_dir: mask PNG 目錄。
        meta_path: 對應的 mask_meta.json 路徑。
        force_recompute: 忽略既有 cache 強制重算。
    """
    print(f"[main] 讀取 metadata: {meta_path}")
    meta = load_meta(meta_path)
    print(f"[main] meta 共 {len(meta)} 筆")

    cache_key = dataset_cache_key(meta_path.parent)
    cache_path = CACHE_DIR / f"raw_native_{cache_key}.npy"
    G = load_or_compute(
        cache_path,
        meta,
        masks_dir,
        DEFAULT_SIM_CFG,
        force=force_recompute,
    )

    finite_rows = np.all(np.isfinite(G), axis=1).sum()
    print(f"[main] G shape={G.shape}, finite rows={finite_rows}")

    print("[main] 跑 T1.1 spread 測試...")
    spread, spread_pass = t1_1_spread(G)

    print("[main] 跑 T1.2 冗餘測試...")
    max_corr, corr_redundant, corr_extreme, partners = t1_2_redundancy(G)

    print("[main] 跑 T1.4 分布形狀...")
    n_unique, p10, p50, p90 = t1_4_distribution(G)

    is_degen = degeneracy_check(G, n_unique)

    verdicts = build_verdicts(
        spread,
        spread_pass,
        max_corr,
        corr_redundant,
        corr_extreme,
        partners,
        n_unique,
        is_degen,
        p10,
        p50,
        p90,
    )

    write_report(verdicts, int(finite_rows), REPORT_PATH)

    print("\n=== 終局摘要 ===")
    passed = [v.name for v in verdicts if v.overall_pass]
    failed = [v.name for v in verdicts if not v.overall_pass]
    print(f"通過：{len(passed)}/{FEATURE_DIM}")
    for v in verdicts:
        flag = "[PASS]" if v.overall_pass else "[FAIL]"
        print(f"  {flag} {v.name}")
    print(f"\n硬 fail ({len(failed)}):")
    for name in failed:
        print(f"  - {name}")
    redundant = [v.name for v in verdicts if v.corr_redundant and v.overall_pass]
    if redundant:
        print(f"\n候選冗餘 ({len(redundant)}, informational):")
        for name in redundant:
            print(f"  - {name}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="mask pool 目錄（含 masks/ 與 mask_meta.json，"
             "e.g. dataset/CEDS_dataset/flat_diverse2000）",
    )
    p.add_argument("--force", action="store_true", help="強制重新計算 G_vec cache")
    args = p.parse_args()
    main(
        masks_dir=args.dataset_dir / "masks",
        meta_path=args.dataset_dir / "mask_meta.json",
        force_recompute=args.force,
    )
