"""閉合幾何環（loop）資料集的 cases.json 產生器（C2-1_loop_{N}_{k}）。

從一個 base cases.json 取一組建物佈局，選定一棟「mover」建物，讓它沿小圓 orbit
分成 N 個節點（角度 theta_i = 2*pi*i/N），繞回 2*pi 精確歸位 → 閉合幾何環。

圖結構 = circulant graph C_N(1..k)：
  - 節點（cold case）：P_0 … P_{N-1}，每個 = 一個 cold-start 幾何狀態。
  - 有向連結（warm case）：(i, (i+d) % N) for d in 1..k，IC = 來源節點收斂場、幾何 = 目標節點。
  N=5, k=2 時 C_5(1,2) = K_5（完全圖），前向連結 N*k = 10 條。

輸出（param_pca cases.json 格式，供既有 C-series 管線消費）：
  dataset/{dataset_name}/cases.json        # 5 cold + 10 warm 節點
  dataset/{dataset_name}/loop_manifest.json # 連結清單（d、chord、來源/目標）

後續沿用既有工具：
  render_cases.py → masks/ ；gen_c_series_configs.py → configs/ ；
  run_c_series_batch.py → raw/*.h5 ；tes_h5_to_zarr.py → zarr。

用法（統一 venv）：
    python -m lbmnca.morphospace.param_pca.gen_loop_cases \\
        --base-cases dataset/C0-1_Cold_M30C30/cases.json \\
        --base-case-id C0-1_Cold_M30C30_c0000 \\
        --mover-idx 1 --n-points 5 --k-degree 2 --orbit-radius 8 --re 300 \\
        --dataset-name C2-1_loop_5_2 --out-root dataset
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lbmnca.morphospace.param_pca import config as C


# ── 幾何 ─────────────────────────────────────────────────────────────────────

def _orbit_buildings(
    base_buildings: list[list[float]],
    mover_idx: int,
    orbit_center: tuple[float, float],
    orbit_radius: float,
    theta: float,
) -> list[list[float]]:
    """回傳將 mover 建物移到 orbit 角度 theta 的新 buildings（深拷貝）。

    Args:
        base_buildings: [[cx, cy, w, h], ...] base 佈局。
        mover_idx: 要 orbit 的建物索引。
        orbit_center: orbit 圓心 (cx0, cy0)。
        orbit_radius: orbit 半徑（px）。
        theta: orbit 角度（rad）。

    Returns:
        新 buildings 列；除 mover 外其餘建物不變。
    """
    cx0, cy0 = orbit_center
    new = [list(b) for b in base_buildings]
    mover = new[mover_idx]
    mover[0] = round(cx0 + orbit_radius * math.cos(theta))
    mover[1] = round(cy0 + orbit_radius * math.sin(theta))
    return new


def _check_in_domain(buildings: list[list[float]], case_id: str) -> list[str]:
    """檢查每棟建物中心落在 inner box 可放範圍內，回傳警告列（空 = 全部 OK）。"""
    warnings: list[str] = []
    for k, (cx, cy, w, h) in enumerate(buildings):
        if not (C.CX_LO <= cx <= C.CX_HI and C.CY_LO <= cy <= C.CY_HI):
            warnings.append(
                f"[{case_id}] 建物 {k} 中心 ({cx},{cy}) 超出 inner box "
                f"x[{C.CX_LO},{C.CX_HI}] y[{C.CY_LO},{C.CY_HI}]"
            )
    return warnings


# ── 節點 / 連結建構 ───────────────────────────────────────────────────────────

def build_loop_cases(
    base_buildings: list[list[float]],
    mover_idx: int,
    n_points: int,
    k_degree: int,
    orbit_radius: float,
    re: int,
    dataset_name: str,
) -> tuple[dict, dict]:
    """建構 loop 的 cases.json（nodes）與 loop_manifest.json（links）內容。

    Args:
        base_buildings: base 佈局 [[cx, cy, w, h], ...]。
        mover_idx: mover 建物索引。
        n_points: 節點數 N（>= 3）。
        k_degree: 鄰接度 k（1 <= k <= floor(N/2)）。
        orbit_radius: orbit 半徑（px）。
        re: 固定 Reynolds 數（所有 case 共用）。
        dataset_name: 資料集名（= warm 的 parent_dataset）。

    Returns:
        (cases_meta, manifest)：兩個可序列化 dict。
    """
    if n_points < 3:
        raise ValueError(f"n_points 須 >= 3，收到 {n_points}")
    if not (1 <= k_degree <= n_points // 2):
        raise ValueError(f"k_degree 須在 [1, {n_points // 2}]，收到 {k_degree}")

    cx0, cy0 = base_buildings[mover_idx][0], base_buildings[mover_idx][1]
    orbit_center = (cx0, cy0)

    # 各節點幾何（mover 沿 orbit）
    node_buildings: list[list[list[float]]] = []
    for i in range(n_points):
        theta = 2.0 * math.pi * i / n_points
        node_buildings.append(
            _orbit_buildings(base_buildings, mover_idx, orbit_center, orbit_radius, theta)
        )

    nodes: list[dict] = []
    cold_case_ids: list[str] = []

    # 1) cold 節點（parent_idx = -1）
    for i in range(n_points):
        case_id = f"{dataset_name}_p{i}"
        cold_case_ids.append(case_id)
        nodes.append({
            "case_id": case_id,
            "mask_id": case_id,
            "re": re,
            "n_buildings": len(node_buildings[i]),
            "buildings": node_buildings[i],
            "ic_type": "cold",
            "parent_idx": -1,
            "parent_case_id": "",
            "parent_dataset": "",
            "depth": 0,
            "shortcut_parents": [],
            "loop": {"role": "point", "point_idx": i},
        })

    # 2) warm 連結（每個 cold i → (i+d)%N，d in 1..k）
    links: list[dict] = []
    for i in range(n_points):
        for d in range(1, k_degree + 1):
            j = (i + d) % n_points
            case_id = f"{dataset_name}_w{i}{j}d{d}"
            chord = 2.0 * orbit_radius * math.sin(math.pi * d / n_points)
            nodes.append({
                "case_id": case_id,
                "mask_id": case_id,
                "re": re,
                "n_buildings": len(node_buildings[j]),
                "buildings": node_buildings[j],          # 幾何 = 目標節點 j
                "ic_type": "warm",
                "parent_idx": i,                          # IC 來源 = cold 節點 i
                "parent_case_id": cold_case_ids[i],
                "parent_dataset": dataset_name,
                "depth": 1,
                "shortcut_parents": [],
                "loop": {"role": "link", "src": i, "dst": j, "d": d,
                         "chord_px": round(chord, 2)},
            })
            links.append({
                "link_case_id": case_id,
                "src_point": i,
                "dst_point": j,
                "d": d,
                "chord_px": round(chord, 2),
                "src_case_id": cold_case_ids[i],
                "dst_geometry_case_id": cold_case_ids[j],
                "parent_h5_rel": f"dataset/{dataset_name}/raw/{cold_case_ids[i]}.h5",
            })

    cases_meta = {
        "dataset_id": dataset_name,
        "space": f"closed loop circulant C_{n_points}({k_degree}) — single-building orbit",
        "method": (
            f"orbit-translate mover building (idx {mover_idx}) on circle r={orbit_radius}px, "
            f"theta_i = 2*pi*i/{n_points}; warm links = ring-distance 1..{k_degree}"
        ),
        "re_range": [re, re],
        "n_points": n_points,
        "k_degree": k_degree,
        "orbit_radius_px": orbit_radius,
        "mover_idx": mover_idx,
        "orbit_center": [cx0, cy0],
        "n_cold": n_points,
        "n_warm": len(links),
        "n_cases": len(nodes),
        "chord_px_per_d": {
            d: round(2.0 * orbit_radius * math.sin(math.pi * d / n_points), 2)
            for d in range(1, k_degree + 1)
        },
        "nodes": nodes,
    }
    manifest = {
        "dataset_id": dataset_name,
        "n_points": n_points,
        "k_degree": k_degree,
        "orbit_radius_px": orbit_radius,
        "links": links,
    }
    return cases_meta, manifest


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="生成 loop 閉合環資料集 cases.json + manifest")
    ap.add_argument("--base-cases", required=True, help="base cases.json 路徑")
    ap.add_argument("--base-case-id", required=True, help="base 節點 case_id（取其 buildings）")
    ap.add_argument("--mover-idx", type=int, required=True, help="orbit 的建物索引")
    ap.add_argument("--n-points", type=int, default=5, help="節點數 N")
    ap.add_argument("--k-degree", type=int, default=2, help="鄰接度 k")
    ap.add_argument("--orbit-radius", type=float, default=8.0, help="orbit 半徑（px）")
    ap.add_argument("--re", type=int, default=300, help="固定 Reynolds 數")
    ap.add_argument("--dataset-name", required=True, help="輸出資料集名（如 C2-1_loop_5_2）")
    ap.add_argument("--out-root", required=True, help="dataset 根目錄（如 ../dataset）")
    cfg = ap.parse_args()

    base_path = Path(cfg.base_cases)
    base_meta = json.loads(base_path.read_text(encoding="utf-8"))
    base_node = next(
        (n for n in base_meta["nodes"] if n["case_id"] == cfg.base_case_id), None
    )
    if base_node is None:
        raise SystemExit(f"base cases.json 找不到 case_id={cfg.base_case_id}")
    base_buildings = [list(b) for b in base_node["buildings"]]
    if not (0 <= cfg.mover_idx < len(base_buildings)):
        raise SystemExit(f"mover-idx {cfg.mover_idx} 超出 base 建物數 {len(base_buildings)}")

    cases_meta, manifest = build_loop_cases(
        base_buildings=base_buildings,
        mover_idx=cfg.mover_idx,
        n_points=cfg.n_points,
        k_degree=cfg.k_degree,
        orbit_radius=cfg.orbit_radius,
        re=cfg.re,
        dataset_name=cfg.dataset_name,
    )

    # domain 邊界檢查（印警告，不中斷）
    warnings: list[str] = []
    for nd in cases_meta["nodes"][: cfg.n_points]:   # 只查 cold 節點（warm 幾何重複）
        warnings.extend(_check_in_domain(nd["buildings"], nd["case_id"]))
    for w in warnings:
        print(f"[WARN] {w}")

    out_dir = Path(cfg.out_root) / cfg.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases.json").write_text(
        json.dumps(cases_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "loop_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] {cfg.dataset_name}: {cases_meta['n_cold']} cold + {cases_meta['n_warm']} warm "
          f"= {cases_meta['n_cases']} cases")
    print(f"[OK] chord_px_per_d = {cases_meta['chord_px_per_d']}")
    print(f"[OK] 寫出 {out_dir / 'cases.json'}")
    print(f"[OK] 寫出 {out_dir / 'loop_manifest.json'}")
    if warnings:
        print(f"[WARN] {len(warnings)} 個 domain 邊界警告（見上）")


if __name__ == "__main__":
    main()
