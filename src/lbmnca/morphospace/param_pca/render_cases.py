"""cases.json 的 rectangle params → mask PNG（規範佈局，0=建物 255=流體）。

自包含（numpy + PIL）。輸出 masks/{case_id}.png（1080×1920）+ warm DAG 拓樸序（root 先）。

用法：python -m lbmnca.morphospace.param_pca.render_cases \\
        --cases data/design_space/S1_5-Cube4D_R16C300/cases.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lbmnca.morphospace.param_pca import config as C


def render_mask(buildings: list[list[float]]) -> np.ndarray:
    """buildings: [[cx,cy,w,h]...] → uint8 (H,W)，0=建物 255=流體。"""
    mask = np.full((C.H_TOTAL, C.W_TOTAL), 255, np.uint8)
    for cx, cy, w, h in buildings:
        x0 = max(0, int(round(cx - w / 2)))
        x1 = min(C.W_TOTAL, int(round(cx + w / 2)))
        y0 = max(0, int(round(cy - h / 2)))
        y1 = min(C.H_TOTAL, int(round(cy + h / 2)))
        mask[y0:y1, x0:x1] = 0
    return mask


def topo_order(nodes: list[dict]) -> list[int]:
    """warm DAG 拓樸序（parent 先於 child；root 最先）→ index 列。"""
    order, done = [], set()

    def visit(i: int) -> None:
        if i in done:
            return
        p = nodes[i]["parent_idx"]
        if p >= 0 and p not in done:
            visit(p)
        done.add(i)
        order.append(i)

    for i in range(len(nodes)):
        visit(i)
    return order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="cases.json")
    cfg = ap.parse_args()

    cp = Path(cfg.cases)
    meta = json.loads(cp.read_text(encoding="utf-8"))
    nodes = meta["nodes"]
    out_dir = cp.parent
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    from PIL import Image
    bcrs = []
    for nd in nodes:
        m = render_mask(nd["buildings"])
        Image.fromarray(m, mode="L").save(masks_dir / f"{nd['case_id']}.png")
        bcrs.append(float((m <= 127).mean()))
        nd["mask_id"] = nd["case_id"]

    order = topo_order(nodes)
    meta["sim_order"] = [nodes[i]["case_id"] for i in order]   # root 先、warm 後
    meta["render_status"] = f"done: {len(nodes)} masks"
    cp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    bcrs = np.array(bcrs)
    print(f"[render] {len(nodes)} masks → {masks_dir}")
    print(f"  BCR: min={bcrs.min():.3f} med={np.median(bcrs):.3f} max={bcrs.max():.3f}")
    print(f"  sim_order（warm DAG 拓樸序）寫入 cases.json：{len(order)} 筆，root 先")


if __name__ == "__main__":
    main()
