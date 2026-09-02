r"""由 `dataset/{ds}/cases.json` 產生 zarr 層的 `dataset_zarr/{ds}_cases.json` sidecar。

## 這兩個 cases.json 不是同一個檔

- `dataset/{ds}/cases.json` —— 生成端的 SSOT，欄位是採樣器的原生 schema
  （`depth`、`parent_idx`、`cube`、`buildings`……）
- `dataset_zarr/{ds}_cases.json` —— 下游 sidecar，schema 由消費端決定：

      {"nodes": [{"case_id", "parent_case_id", "parent_dataset",
                  "tree_depth", "parent_idx"}, ...]}

  `migrate_zarr_v3_to_v4._synthesize_chain_graph` 與
  `src/lbmnca/nca/tes/data/pair_mean_lookup.build_pair_mean_lookup` 都讀它。

`regen_cases_json.py` 只處理四個 warm dataset 且參數寫死，本腳本補上通用路徑。

## 純 cold dataset

每個 case 各自獨立模擬、沒有 parent：`parent_case_id=""`、`tree_depth=0`、
`parent_idx=-1`。下游據此把全部 node 當 cold IC 處理，不會誤建 warm 配對。

## 只收錄通過物理閘門的 case

sidecar 的 node 清單若含被判退的 case，遷移階段會替它們建 v4 node，
判退的資料就繞過閘門進了下游。收錄與否的單一真相是 `dataset_meta.json`
的 ``status``——與 `tes_h5_to_zarr._filter_admitted` 同一套判準。

## 用法

```bash
python -m lbmnca.lbm.post_process.make_cases_sidecar \
    --datasets D0-1_Cold_M40C40 D1-1_Cold_M400C400
```
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lbmnca.paths import dataset_dir, zarr_dir

#: 與 tes_h5_to_zarr._filter_admitted 保持一致
_ADMITTED = ("success", "converged", "precomputed")


def _node_from(src: dict, ds: str, cid: str) -> dict:
    """把生成端 node 或 meta entry 統一成 sidecar schema。"""
    # 生成端用 `depth`，消費端用 `tree_depth`；warm 才有 parent。
    depth = int(src.get("tree_depth", src.get("depth", 0)))
    pid = str(src.get("parent_case_id", ""))
    # meta entry 沒有 parent_idx —— 依 parent 有無回推（生成端就是這個規則）
    pidx = int(src["parent_idx"]) if "parent_idx" in src else (0 if pid else -1)
    return {
        "case_id": cid,
        "parent_case_id": pid,
        "parent_dataset": str(src.get("parent_dataset", ds)),
        "tree_depth": depth,
        "parent_idx": pidx,
        # 以下為 migrate_zarr_v3_to_v4._synthesize_chain_graph 建 v4 node 時
        # 直接取用的欄位（缺任一個都是 KeyError，它刻意不給預設值）。
        "mask_id": str(src.get("mask_id", cid)),
        "re": float(src["re"]),
        "ic_type": str(src.get("ic_type", "cold")),
        # cube_geom / param_delta_norm 僅供製圖與取樣權重，consumer 用
        # .get() 讀，缺了不影響訓練——但有就一併帶上。
        "cube_geom": src.get("cube_geom"),
        "param_delta_norm": src.get("param_delta_norm"),
    }


def build_nodes(ds: str) -> tuple[list[dict], int]:
    """由生成端 cases.json + dataset_meta.json 組出 sidecar 的 node 清單。

    ★收錄集合以 `dataset_meta.json` 為準，不是以 `cases.json` 為準。
    判退後由 `WarmReplacer` 生成的替代 case（`_r1`、`_r1_r1`……）只寫進 meta，
    **不會回寫 cases.json**——實測 D2-1 有 32 個、DE1-1 有 7 個這種 case
    通過閘門並進了 v3 zarr。若只走 cases.json，這些 case 在 v4 遷移時沒有
    對應 node，等於 11% 的 warm 資料靜默消失。故 cases.json 缺的一律回退到
    meta entry（欄位齊全：mask_id／re／ic_type／depth／parent_case_id）。

    Returns:
        (node 清單, 由 meta 補回的 node 數)。
    """
    d = dataset_dir(ds)
    meta = json.loads((d / "dataset_meta.json").read_text(encoding="utf-8"))
    by_meta = {str(c["case_id"]): c for c in meta["cases"]
               if str(c.get("status")) in _ADMITTED}
    by_src = {str(n["case_id"]): n
              for n in json.loads((d / "cases.json").read_text(encoding="utf-8"))["nodes"]}

    nodes, n_fallback = [], 0
    for cid in by_meta:
        src = by_src.get(cid)
        if src is None:
            src = by_meta[cid]
            n_fallback += 1
        nodes.append(_node_from(src, ds, cid))
    return nodes, n_fallback


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description="產生 zarr 層 cases.json sidecar")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out-root", default=str(zarr_dir()))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        nodes, n_fb = build_nodes(ds)
        n_warm = sum(1 for n in nodes
                     if n["tree_depth"] >= 1 and n["parent_idx"] >= 0)
        p = out_root / f"{ds}_cases.json"
        p.write_text(json.dumps({"nodes": nodes}, indent=1,
                                ensure_ascii=False), encoding="utf-8")
        print(f"{ds}: {len(nodes)} node（warm child {n_warm}"
              + (f"，其中 {n_fb} 個由 meta 補回" if n_fb else "") + f"）→ {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
