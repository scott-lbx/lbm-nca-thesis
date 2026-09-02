r"""v3 cases → 單檔 mean-only unified zarr（`tes_unified_mean_v1`）。

## 為什麼是新的一支，而不是改 migrate + build_unified

mean-only 範疇下，`postconv` 與過渡段 `snapshots` **沒有任何讀取端**：
`src/lbmnca/nca/tes/data/graph_dataset.py` 只讀

    nodes/{id}/static        [4, H, W]
    nodes/{id}/mean_flow     [9, H, W]

而 `migrate_zarr_v3_to_v4.py` 有一整套圍繞「transit / postconv 切分」的
不變式（``n_transit + n_postconv == T``）與逐位元驗證。那些不變式在
只存一幀的資料上必然失敗，且它們保護的東西已經不存在。
與其把 1000 行拆掉一半，不如直接走一條短路徑，把圖合成邏輯 import 過來重用。

## 輸出

```
{ds}_unified.zarr/
├── .zattrs["topology"]  {version, format, dataset_id, nodes[], edges[]}
├── nodes/{node_id}/
│   ├── mean_flow          (9, H, W) float32   ← 訓練目標
│   ├── mean_flow_half1     (9, H, W) float32   ← VQI 上錨（GT 自噪）
│   ├── mean_flow_half2     (9, H, W) float32
│   └── static             (4, H, W) float32
└── edges/{edge_id}/          ← 只有拓撲，不含任何陣列
```

`parent_mean` edge 的 IC 取 src node 的 `mean_flow`；`cold_start` edge
在 mean module 下**不被消費**（`graph_split.py` 明確排除），故不存任何幀。

★ `format` 標為 ``tes_unified_mean_v1``。舊的 ``tes_unified_v1`` 帶 postconv
且用**量測**的正規化常數，兩者數值意義不同 —— 下游必須據此拒絕，不可互換。

## 用法

```bash
python -m lbmnca.lbm.post_process.build_unified_zarr --dataset D0-1_Cold_M40C40 \
    --src-root dataset_zarr_v2 --out-root dataset_zarr_v2 --execute
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from lbmnca.lbm.post_process import norm_frozen as nf
from lbmnca.paths import zarr_dir

FORMAT: str = "tes_unified_mean_v1"


# ══════════════════════════════════════════════════════════════════════════
# 圖合成
# ══════════════════════════════════════════════════════════════════════════


def synthesize_graph(ds: str, case_ids: set[str],
                     cases_json: Path) -> tuple[list[dict], list[dict]]:
    """由 ``{ds}_cases.json`` 的 parent 樹合成 node / edge。

    每個 case 同時是一個 node（其 post-conv attractor）與一條 canonical edge
    （抵達該 attractor 的路徑）。edge 的 ``src_node``：

    * parent 存在於 zarr　　　　→ parent 的 case_id（warm transfer）
    * ``parent_case_id`` 為空　 → ``None``（冷啟動）
    * parent 不在 zarr（孤兒）　→ ``None``（冷啟動；顯式化 v3 的靜默行為）

    Args:
        ds: dataset 名稱。
        case_ids: v3 zarr 內實際存在的 case_id 集合。
        cases_json: ``{ds}_cases.json`` sidecar 路徑。

    Returns:
        ``(nodes, edges)``，欄位與既有 topology 相容。
    """
    raw = json.loads(cases_json.read_text(encoding="utf-8"))["nodes"]
    kept = [n for n in raw if n["case_id"] in case_ids]
    in_zarr = {n["case_id"] for n in kept}

    parent_of: dict[str, str | None] = {}
    orphans: list[str] = []
    for n in kept:
        pid = n["parent_case_id"]
        if not pid:
            parent_of[n["case_id"]] = None
        elif pid in in_zarr:
            parent_of[n["case_id"]] = pid
        else:
            parent_of[n["case_id"]] = None
            orphans.append(n["case_id"])
    if orphans:
        print(f"  [graph] {len(orphans)} 個孤兒 case（parent 不在 zarr）→ cold_start")

    def _depth(cid: str) -> int:
        d, cur = 0, parent_of[cid]
        while cur is not None:
            d, cur = d + 1, parent_of[cur]
        return d

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for i, n in enumerate(sorted(kept, key=lambda x: x["case_id"])):
        cid = n["case_id"]
        eid = f"{ds}_e{i:04d}"
        src = parent_of[cid]
        nodes.append({
            "node_id":           cid,
            "mask_id":           n["mask_id"],
            "re":                float(n["re"]),
            "ic_type":           n["ic_type"],
            "tree_depth":        _depth(cid),
            "canonical_edge_id": eid,
            "cube_geom":         n.get("cube_geom"),
            "external":          False,
        })
        edges.append({
            "edge_id":     eid,
            "src_node":    src,
            "dst_node":    cid,
            "edge_type":   "chain",
            "ic_source":   "parent_mean" if src else "cold_start",
            "window_type": "full",
            "traj_ref":    None,
            "attractor":   None,
        })
    return nodes, edges


# ══════════════════════════════════════════════════════════════════════════
# 建置
# ══════════════════════════════════════════════════════════════════════════


def build(ds: str, src_root: Path, out_root: Path, execute: bool,
          overwrite: bool = False) -> Path:
    """v3 cases → mean-only unified 單檔。回傳輸出路徑。

    Args:
        overwrite: 允許就地重建。用 zarr 的 ``mode="w"`` 截斷 group，
            **不走檔案系統刪除** —— 專案的資料保護 hook 會擋 `.zarr` 的遞迴刪除，
            而那道 hook 該留著。

    Raises:
        RuntimeError: 輸出已存在且未給 ``overwrite``、或來源 zarr 的 schema
            不是凍結常數產出的 ``tes_v4``。
    """
    v3_path = src_root / f"{ds}.zarr"
    out_path = out_root / f"{ds}_unified.zarr"
    cases_json = src_root / f"{ds}_cases.json"
    for p in (v3_path, cases_json):
        if not p.exists():
            raise FileNotFoundError(f"來源缺檔：{p}")
    if out_path.exists() and not overwrite:
        raise RuntimeError(
            f"輸出已存在，拒絕覆寫：{out_path}（確定要重建請加 --overwrite）")

    v3 = zarr.open(str(v3_path), mode="r")
    a = dict(v3.attrs)
    # ★ 只吃凍結常數產出的來源。舊 v3（量測常數、固體未覆寫）數值意義不同，
    #   混進來不會報錯只會靜默產生不可比的資料集。
    if a.get("version") != "tes_v4" or a.get("norm_schema") != nf.SCHEMA_VERSION:
        raise RuntimeError(
            f"來源 schema 不符：version={a.get('version')!r} "
            f"norm_schema={a.get('norm_schema')!r}；"
            f"預期 'tes_v4' / {nf.SCHEMA_VERSION!r}。"
            "舊版 zarr 用的是量測而來的正規化常數，不可混用。")

    v3_cases = v3["cases"]
    case_ids = set(v3_cases.group_keys())
    nodes, edges = synthesize_graph(ds, case_ids, cases_json)
    n_cold = sum(1 for e in edges if e["ic_source"] == "cold_start")
    print(f"[{ds}]  nodes={len(nodes)}  edges={len(edges)}  "
          f"(cold_start {n_cold} / parent_mean {len(edges) - n_cold})")
    if not execute:
        print("  （dry-run，未寫入）")
        return out_path

    root = zarr.open_group(str(out_path), mode="w")
    ngrp = root.require_group("nodes")
    egrp = root.require_group("edges")

    n_running = 0
    for n in nodes:
        cid = n["node_id"]
        src = v3_cases[cid]
        dst = ngrp.require_group(cid)
        # mean_flow_post_conv → mean_flow（改名：unified 層不再有「post_conv」的
        # 對照組，欄位名保留舊詞只會讓讀者以為還有別種 mean）
        dst.create_dataset("mean_flow", data=np.asarray(src["mean_flow_post_conv"]),
                           chunks=(9, None, None), dtype="float32", overwrite=True)
        # ★ 兩段獨立半窗時均 —— VQI 上錨（GT 自噪）的唯一來源。
        #   不可用 mean_flow 與 mean_flow_running 代替：兩者逐位元相同。
        for _tag in ("half1", "half2"):
            _k = f"mean_flow_{_tag}"
            if _k in list(src.array_keys()):
                dst.create_dataset(_k, data=np.asarray(src[_k]),
                                   chunks=(9, None, None), dtype="float32",
                                   overwrite=True)
        if "mean_flow_half1" in list(src.array_keys()):
            n_running += 1
        dst.create_dataset("static", data=np.asarray(src["static"]),
                           chunks=(4, None, None), dtype="float32", overwrite=True)
        # g_vec / p_vec 進 topology（純量向量，放 attrs 比開 array 划算）
        n["g_vec"] = np.asarray(src["g_vec"]).tolist()
        n["p_vec"] = np.asarray(src["p_vec"]).tolist()
        dst.attrs.update({k: v for k, v in src.attrs.items()
                          if k not in ("ctu_values",)})

    # ★ edge 只存拓撲，不存任何幀。
    #   `graph_split.py` 實測：`cold_start` edge **不進 train、也不組 val/test**
    #   （mean module 的任務是 parent_mean(IC) → child_mean(target)，cold_start
    #   依定義沒有 parent）。實際被消費的只有 `parent_mean` edge，而它的 IC 取
    #   src node 的 `mean_flow`，不需要任何快照。
    #
    #   ⚠ 這一點對 D 系列的後果值得知道：D0-1 與 D1-1 是純 cold dataset，
    #   其 312 個 case 對 mean module 的訓練貢獻為 **0**；D2-1 的 86 個 orphan
    #   （parent 在 D1-1 或已被准入過濾剔除）同樣落在 cold_start。
    #   實際訓練集因此是 D2-1 的 173 條 parent_mean edge。
    #   跨 dataset 解析 orphan 的 parent 可讓它變成 259 條（+50 %），
    #   但那會改變訓練集組成、破壞與 -5 的可比性，故不在本次範圍。
    for e in edges:
        egrp.require_group(e["edge_id"])

    root.attrs["topology"] = {
        "version":    "tes_v4",
        "format":     FORMAT,
        "dataset_id": ds,
        "scope":      "mean_only",
        "norm_schema": nf.SCHEMA_VERSION,
        "stats_space": "frozen",
        "grid":       {"H": int(a["H"]), "W": int(a["W"])},
        "nodes":      nodes,
        "edges":      edges,
    }
    print(f"  寫入 {out_path}（{n_running} 個 node 有兩段半窗時均）")
    return out_path


def verify(ds: str, src_root: Path, out_root: Path) -> None:
    """逐 node 全陣列比對來源與輸出；任一不符即中止。

    Raises:
        AssertionError: 數值或結構不符。
    """
    v3 = zarr.open(str(src_root / f"{ds}.zarr"), mode="r")["cases"]
    u = zarr.open(str(out_root / f"{ds}_unified.zarr"), mode="r")
    topo = dict(u.attrs)["topology"]
    assert topo["format"] == FORMAT, topo["format"]

    for n in topo["nodes"]:
        cid = n["node_id"]
        s, d = v3[cid], u["nodes"][cid]
        assert np.array_equal(np.asarray(s["mean_flow_post_conv"]),
                              np.asarray(d["mean_flow"])), f"{cid}: mean_flow 不符"
        assert np.array_equal(np.asarray(s["static"]),
                              np.asarray(d["static"])), f"{cid}: static 不符"
        for _tag in ("half1", "half2"):
            _k = f"mean_flow_{_tag}"
            if _k in list(s.array_keys()):
                assert np.array_equal(np.asarray(s[_k]), np.asarray(d[_k])),                     f"{cid}: {_k} 不符"
    for e in topo["edges"]:
        arrs = list(u["edges"][e["edge_id"]].array_keys())
        assert not arrs, f"{e['edge_id']}: edge 不應含任何陣列，卻有 {arrs}"

    print(f"  驗證通過：{len(topo['nodes'])} nodes / {len(topo['edges'])} edges")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description="v3 → mean-only unified zarr")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--src-root", default=str(zarr_dir()))
    ap.add_argument("--out-root", default=None, help="預設同 --src-root")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="就地重建既有 unified（zarr mode=w 截斷，不刪檔）")
    a = ap.parse_args()
    src = Path(a.src_root)
    out = Path(a.out_root) if a.out_root else src
    out.mkdir(parents=True, exist_ok=True)
    build(a.dataset, src, out, a.execute, overwrite=a.overwrite)
    if a.execute:
        verify(a.dataset, src, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
