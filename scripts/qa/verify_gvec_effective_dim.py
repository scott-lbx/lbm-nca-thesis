"""verify_gvec_effective_dim.py — 從**資料層**查證 G_vec 的有效條件維度。

背景
----
DEEP_AUDIT_2026-07 Stage 2.5（`B2-02-宣稱-01`）：
`B4_max_enclosed_fluid_area_ratio` / `C2_blockage_ratio` / `F3_outlet_reachability`
三維在 C-series canonical 語料上樣本標準差**精確為 0**，
`normalizer.normalize_g` 因此對它們一律輸出 0 → **有效條件維度為 11，不是 14**。

GATE 1 的教訓是「算出來了卻沒進資料層 == 沒做」。本腳本就是把這件事
**做成可從資料層直接查詢**的常設工具：不依賴任何文件敘述，
直接讀 `dataset_zarr/{ds}_graph.json` 的 node `g_vec` 與 `{ds}_stats.json`
的 `cond_stats`，把有效維度算出來。

⚠ 退化集合**由所載入的 stats 決定**，不是固定 3 維。單獨用某個小 dataset
的 stats 時退化維可能更多（實測 `C2-1_loop_5_2` 為 6 維）。
canonical 訓練與部署走 `stats_policy: dataset_global_runtime`（多份 stats 依
case 數加權合併）→ 本腳本的 `[merged]` 區塊即該情境。

檢查項目
--------
1. 逐 dataset：由 node `g_vec` 實測各維 std／unique 值數 → 退化維與有效維度
2. 逐 dataset：`{ds}_stats.json` 的 `cond_stats.g_vec.std` 是否與實測一致
   （若不一致，代表 stats 是舊的／來自不同語料）
3. 合併語料（canonical 訓練與部署實際看到的分布）的退化維與有效維度
4. 與 `gvec_api.GVEC_INERT_DIMS_C_SERIES` / `GVEC_EFFECTIVE_DIM_C_SERIES`
   的宣告值比對 —— 不符即 FAIL（防止 code 內的宣告與資料漂移）
5. 本腳本的退化門檻與 `lbmnca.nca` 的 `TESCondNormalizer.DEGENERATE_STD`
   相同（該常數才是真正決定「哪些維被歸零」的那一個）

用法
----
    python scripts/qa/verify_gvec_effective_dim.py
    python scripts/qa/verify_gvec_effective_dim.py --datasets C1-3_Warm_M300C300R32S100

退出碼：0 = 宣告與資料一致，1 = 不一致。唯讀，不修改任何檔案。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_MORPH_ROOT = _SCRIPT_DIR.parent
_WS_ROOT = _MORPH_ROOT.parent

if str(_MORPH_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_MORPH_ROOT / "src"))

from lbmnca.morphospace.physics_proxy.gvec_api import (  # noqa: E402
    GVEC_DIM,
    GVEC_EFFECTIVE_DIM_C_SERIES,
    GVEC_FEATURE_NAMES,
    GVEC_INERT_DIMS_C_SERIES,
    find_degenerate_dims,
)

#: C-series canonical 語料（`docs/ARCHITECTURE.md` 實測 670 node / 748 edge）。
CANONICAL_DATASETS: list[str] = [
    "C0-3_Warm_M30C30R4S10",
    "C1-3_Warm_M300C300R32S100",
    "C2-1_loop_5_2",
    "C2-2_WarmGraph_M180C300R20",
]

#: 與 `lbmnca.nca` 的 `TESCondNormalizer.DEGENERATE_STD` 同值（由 check 5 釘住）。
DEGENERATE_STD: float = 1e-6

#: 真正決定「哪些維被歸零」的常數所在位置（唯讀 AST 解析，不 import lbmnca.nca）。
_NORMALIZER_PATH_REL: str = "src/lbmnca/nca/tes/data/normalizer.py"


def read_normalizer_degenerate_std(ws_root: Path) -> float | None:
    """以 AST 讀出 `TESCondNormalizer.DEGENERATE_STD`（不 import lbmnca.nca）。

    Args:
        ws_root: lbm-nca-thesis 根目錄。

    Returns:
        該常數值；檔案不存在或找不到常數時回 None。
    """
    import ast

    path = ws_root / _NORMALIZER_PATH_REL
    if not path.exists():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "TESCondNormalizer":
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == "DEGENERATE_STD"
                and stmt.value is not None
            ):
                return float(ast.literal_eval(stmt.value))
    return None


def load_node_gvecs(zarr_root: Path, dataset: str) -> np.ndarray:
    """讀取單一 dataset 的所有 node g_vec。

    Args:
        zarr_root: `dataset_zarr` 目錄。
        dataset: dataset 名稱（不含副檔名）。

    Returns:
        [N, GVEC_DIM] float64 陣列。

    Raises:
        FileNotFoundError: 找不到 `{dataset}_graph.json`。
        ValueError: g_vec 長度與 GVEC_DIM 不符（zarr 語義已漂移）。
    """
    graph_path = zarr_root / f"{dataset}_graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    g = np.array([n["g_vec"] for n in graph["nodes"]], dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != GVEC_DIM:
        raise ValueError(
            f"{graph_path.name} 的 g_vec 維度為 {g.shape[1]}，"
            f"與 gvec_api.GVEC_DIM={GVEC_DIM} 不符 —— zarr 語義已漂移。"
        )
    return g


def load_stats_gvec_std(zarr_root: Path, dataset: str) -> np.ndarray | None:
    """讀取 `{dataset}_stats.json` 內烘焙的 `cond_stats.g_vec.std`。

    Args:
        zarr_root: `dataset_zarr` 目錄。
        dataset: dataset 名稱。

    Returns:
        [GVEC_DIM] std 陣列；檔案不存在或缺欄位時回 None。
    """
    stats_path = zarr_root / f"{dataset}_stats.json"
    if not stats_path.exists():
        return None
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if "cond_stats" not in stats:
        return None
    return np.array(stats["cond_stats"]["g_vec"]["std"], dtype=np.float64)


def report_dataset(g: np.ndarray, stats_std: np.ndarray | None, label: str) -> list[int]:
    """印出單一語料的逐維統計與退化維，回傳退化維索引。

    Args:
        g: [N, GVEC_DIM] raw g_vec。
        stats_std: 該語料 stats.json 內的 std（可為 None）。
        label: 顯示用標籤。

    Returns:
        退化維索引清單。
    """
    stds = g.std(axis=0)
    deg = find_degenerate_dims(g, std_threshold=DEGENERATE_STD)
    print(f"\n[{label}] N = {g.shape[0]}")
    print(f"  {'idx':>3} {'name':<38} {'min':>12} {'max':>12} {'std':>13} {'uniq':>6}")
    for i, name in enumerate(GVEC_FEATURE_NAMES):
        col = g[:, i]
        flag = "  <-- 退化（正規化後恆 0）" if i in deg else ""
        print(
            f"  {i:>3} {name:<38} {col.min():>12.6g} {col.max():>12.6g} "
            f"{stds[i]:>13.6e} {len(np.unique(col)):>6}{flag}"
        )
    print(f"  -> 退化維 {len(deg)} 個：{[GVEC_FEATURE_NAMES[i] for i in deg]}")
    print(f"  -> 有效條件維度 = {GVEC_DIM} - {len(deg)} = {GVEC_DIM - len(deg)}")

    if stats_std is not None:
        # stats.json 內的 std 已被生成端加過 eps（實測 1e-8），故只比對「是否退化」
        stats_deg = [int(i) for i in np.where(stats_std <= DEGENERATE_STD)[0]]
        agree = stats_deg == deg
        print(
            f"  -> stats.json 的 cond_stats 判定退化維 = {stats_deg} "
            f"({'一致' if agree else '★不一致：stats 可能來自別的語料'})"
        )
    else:
        print("  -> [WARN] 找不到 stats.json 或缺 cond_stats，無法交叉比對")
    return deg


def main() -> int:
    """CLI 入口。

    Returns:
        0 = code 內的宣告與資料一致；1 = 不一致或資料缺失。
    """
    parser = argparse.ArgumentParser(description="從資料層查證 G_vec 有效條件維度")
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=_WS_ROOT / "dataset_zarr",
        help="dataset_zarr 目錄（預設由本檔位置推導）",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=CANONICAL_DATASETS,
        help="要檢查的 dataset 名稱（預設 = C-series canonical 四個）",
    )
    args = parser.parse_args()

    failures: list[str] = []
    per_ds: list[np.ndarray] = []

    print("=" * 96)
    print("G_vec 有效條件維度 —— 資料層實測（DEEP_AUDIT_2026-07 Stage 2.5）")
    print(f"退化判準：樣本 std <= {DEGENERATE_STD:g}"
          "（同 lbmnca.nca TESCondNormalizer.DEGENERATE_STD）")
    print("=" * 96)

    for ds in args.datasets:
        try:
            g = load_node_gvecs(args.zarr_root, ds)
        except (FileNotFoundError, ValueError) as exc:
            failures.append(f"{ds}: {exc}")
            print(f"\n[{ds}] [FAIL] {exc}")
            continue
        report_dataset(g, load_stats_gvec_std(args.zarr_root, ds), ds)
        per_ds.append(g)

    if not per_ds:
        print("\n[FAIL] 沒有任何 dataset 讀取成功")
        return 1

    merged = np.concatenate(per_ds, axis=0)
    print("\n" + "=" * 96)
    print("合併語料（canonical 訓練與部署實際看到的分布；"
          "stats_policy = dataset_global_runtime）")
    print("=" * 96)
    merged_deg = report_dataset(merged, None, "merged")

    # ── 與 gvec_api 內的宣告比對 ────────────────────────────────────────
    print("\n" + "=" * 96)
    print("與 gvec_api 宣告值比對")
    print("=" * 96)
    declared = sorted(GVEC_FEATURE_NAMES.index(n) for n in GVEC_INERT_DIMS_C_SERIES)
    print(f"  gvec_api.GVEC_INERT_DIMS_C_SERIES        = {list(GVEC_INERT_DIMS_C_SERIES)}")
    print(f"  資料層實測（合併語料）                    = "
          f"{[GVEC_FEATURE_NAMES[i] for i in merged_deg]}")
    if merged_deg != declared:
        failures.append(
            f"GVEC_INERT_DIMS_C_SERIES 宣告 {declared}，資料層實測 {merged_deg}"
        )
    if GVEC_EFFECTIVE_DIM_C_SERIES != GVEC_DIM - len(merged_deg):
        failures.append(
            f"GVEC_EFFECTIVE_DIM_C_SERIES 宣告 {GVEC_EFFECTIVE_DIM_C_SERIES}，"
            f"資料層實測 {GVEC_DIM - len(merged_deg)}"
        )
    # 常數值本身也要對得上
    for name, expected in GVEC_INERT_DIMS_C_SERIES.items():
        i = GVEC_FEATURE_NAMES.index(name)
        actual = float(merged[:, i][0])
        if abs(actual - expected) > 1e-9:
            failures.append(f"{name} 宣告常數 {expected}，資料層實測 {actual}")

    # 退化門檻必須與真正生效的那一個（lbmnca.nca normalizer）相同
    norm_std = read_normalizer_degenerate_std(_WS_ROOT)
    if norm_std is None:
        print(f"  [WARN] 讀不到 {_NORMALIZER_PATH_REL} 的 DEGENERATE_STD，跳過門檻比對")
    else:
        print(f"  TESCondNormalizer.DEGENERATE_STD        = {norm_std:g}"
              f"（本腳本用 {DEGENERATE_STD:g}）")
        if norm_std != DEGENERATE_STD:
            failures.append(
                f"退化門檻不一致：本腳本 {DEGENERATE_STD:g} vs "
                f"{_NORMALIZER_PATH_REL} {norm_std:g}"
            )

    print("")
    print("=" * 96)
    if failures:
        print(f"[FAIL] 宣告與資料層不一致（{len(failures)} 項）")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"[PASS] 資料層實測有效條件維度 = {GVEC_DIM - len(merged_deg)}"
          f"（宣告 {GVEC_EFFECTIVE_DIM_C_SERIES}），常數維與宣告值逐項相符。")
    print("       [WARN] 這三維並非本質退化 —— 合成幾何（中庭／貼入口量體／全高牆）")
    print("         上皆有非常數取值；是 C-series 遮罩生成慣例未涵蓋該幾何族。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
