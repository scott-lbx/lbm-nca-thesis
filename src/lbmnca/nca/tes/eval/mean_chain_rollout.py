"""沿最長 warm parent-child 鏈做「無 GT 鏈式 rollout」誤差累積實驗。

部署情境：幾何編輯 = 換 static、physics 沿用上一收斂場。本實驗串成鏈：root IC=GT，
之後**每個 child 用 parent 的「預測」收斂場當 warm IC（no GT）**，逐 case rollout，
量每階段對該 case GT mean 的 MAE。誤差是否沿鏈爆掉 = warm 鏈 robustness。

  stage 0 (root)  : IC = root GT mean      → rollout N 步 → pred_0（應守住 = fixed point）
  stage k (child) : IC = pred_{k-1}（上一 case 預測，no GT）+ 本 case static
                    → rollout N 步 → pred_k；MAE(pred_k, GT_k)

兩顆 ckpt 同鏈同 IC 跑，疊圖比較各階段 MAE。

★rollout 完全沿用 trainer._validate_split_mean 的逐步 model 呼叫（同 arch/cond/renorm）。
★只讀 ckpt + dataset，不訓練、不改 arch/loss。

用法（路徑相對 repo 根目錄；zarr 預設落在 ``data/zarr/``，見 ``lbmnca.paths``）：
    python -m lbmnca.nca.tes.eval.mean_chain_rollout \
        --config configs/nca/main.yaml \
        --v29 checkpoints/base_modern.pth \
        --v30 checkpoints/main_v4.pth \
        --zarr data/zarr/D2-1_Warm_M200C200R24S60_unified.zarr \
        --ctu-per-stage 100 --out runs/diagnostics/mean_chain_rollout.png
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset

from ..data.graph_dataset import collate_graph_batch as _collate_variable_length
from ..inference import protocol
from ..train_mean import build_mean_training
from .visual_metrics import struct_vel_rgb_ssim

_CTU_PER_STEP = 0.2  # mean gate pseudo-CTU 慣例（mean_gate_eval.ke_curve_convergence）


def _find_chain(cases_json_path: str) -> list[str]:
    """由 nodes.parent_idx 建樹，回傳最長 root→leaf 鏈的 case_id 串（root 在前）。"""
    nodes = json.load(open(cases_json_path, encoding="utf-8"))["nodes"]

    def up(i: int) -> list[int]:
        out, seen = [], set()
        while 0 <= i < len(nodes) and i not in seen:
            seen.add(i)
            out.append(i)
            p = nodes[i].get("parent_idx", -1)
            i = p if isinstance(p, int) else -1
        return out[::-1]

    chains = [up(i) for i in range(len(nodes))]
    chosen = max(chains, key=len)
    return [nodes[i]["case_id"] for i in chosen]


def _build_case_index(trainer) -> dict[str, tuple]:
    """case_key → (dataset, local_idx)，掃 train+val 全 split（只讀 _case_keys，不載資料）。"""
    index: dict[str, tuple] = {}
    loaders = [getattr(trainer, "_train_loader", None),
               getattr(trainer, "_val_warm_loader", None),
               getattr(trainer, "_val_cold_loader", None)]
    for ld in loaders:
        if ld is None:
            continue
        ds = ld.dataset
        subs = ds.datasets if isinstance(ds, ConcatDataset) else [ds]
        for sub in subs:
            keys = getattr(sub, "_case_keys", None)
            if keys is None:
                continue
            for li, k in enumerate(keys):
                index.setdefault(k, (sub, li))
    return index


def _resolve_present_chain(chain_ids: list[str], index: dict[str, tuple]) -> list[tuple[str, tuple]]:
    """把鏈裁成「都在 dataset 內」的最長連續子鏈（warm/converged 才有 mean_flow）。"""
    runs: list[list[tuple[str, tuple]]] = []
    cur: list[tuple[str, tuple]] = []
    for c in chain_ids:
        hit = c if c in index else next((k for k in index if k.startswith(c)), None)
        if hit is not None:
            cur.append((hit, index[hit]))
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return max(runs, key=len) if runs else []


def _select_cross_re(cases_json_path: str, index: dict[str, tuple], n_re: int
                     ) -> tuple[list[tuple[str, tuple]], list[str]]:
    """跨 Re 選一 case/Re（均勻取 n_re 個 Re 值），各獨立（無血緣交棒）。

    回傳 (items=[(case_key,(ds,li))…] 依 Re 升序, labels=["Re###"…])。
    """
    nodes = json.load(open(cases_json_path, encoding="utf-8"))["nodes"]
    by_re: dict[int, list[str]] = {}
    for nd in nodes:
        by_re.setdefault(int(nd["re"]), []).append(nd["case_id"])
    res_sorted = sorted(by_re)
    # 均勻抽 n_re 個 Re（含頭尾）
    if len(res_sorted) > n_re:
        pick_idx = np.linspace(0, len(res_sorted) - 1, n_re).round().astype(int)
        res_pick = [res_sorted[i] for i in sorted(set(pick_idx.tolist()))]
    else:
        res_pick = res_sorted
    items, labels = [], []
    for re in res_pick:
        hit = None
        for cid in by_re[re]:
            k = cid if cid in index else next((kk for kk in index if kk.startswith(cid)), None)
            if k is not None:
                hit = k
                break
        if hit is not None:
            items.append((hit, index[hit]))
            labels.append(f"Re{re}")
    return items, labels


def _masked_mae_mse(pred: Tensor, gt: Tensor, fm: Tensor) -> tuple[float, float, float]:
    """回傳 (mae, mse, rel_l2)；rel_l2=‖pred−gt‖₂/‖gt‖₂（fluid 區，除 target 幅值 → 去 magnitude 階層）。"""
    mask = fm.float().expand_as(pred)
    n = mask.sum().clamp_min(1.0)
    diff = (pred - gt) * mask
    mae = float((diff.abs().sum() / n).item())
    mse = float((diff.square().sum() / n).item())
    tgt_sq = (gt * mask).square().sum()
    rel_l2 = float((diff.square().sum().sqrt() / tgt_sq.sqrt().clamp_min(1e-8)).item())
    return mae, mse, rel_l2


@torch.inference_mode()
def _rollout_chain(trainer, chain: list[tuple[str, tuple]], n_steps: int,
                   handoff: bool = True, labels: Optional[list[str]] = None) -> list[dict]:
    """逐 stage rollout，回傳每 stage dict（逐步 MAE/MSE/rel_l2/SSIM + 末值）。

    handoff=True（chain 模式）：root IC=GT，child IC=上一 stage 預測場（no GT，鏈式累積）。
    handoff=False（cross-re 模式）：每 stage 各自 IC=GT、互不相干（單純比各 Re 難度）。
    labels：每 stage 顯示標籤（None→case 短碼；cross-re→"Re###"）。
    """
    model = trainer._model
    model.eval()
    sd = model._cfg.static_dim
    pd = model._cfg.physics_dim
    field_norm = trainer._get_field_norm()
    dev = trainer._device

    stages: list[dict] = []
    prev_pred: Optional[Tensor] = None
    for stage, (case_key, (ds, li)) in enumerate(chain):
        batch = _collate_variable_length([ds[li]])
        static = batch["static"].to(dev)
        gt_mean = batch["mean_flow_post_conv"].to(dev).float()
        g_vec = batch["g_vec"].to(dev)
        p_vec = batch["p_vec"].to(dev)

        # M2 renorm（source-local → dataset-global，與 _validate_split_mean 對齊）
        static, _, _g5 = trainer._renorm_source_to_global(
            static, gt_mean, gt_mean.unsqueeze(1), batch
        )
        gt_mean = _g5.squeeze(1)

        fluid_mask = trainer._get_fluid_mask(static)
        cond_norm = trainer._build_cond_norm(g_vec, p_vec)
        g_vec_norm = trainer._build_g_vec_norm(g_vec)

        # IC：chain → root=GT/child=上一預測場（no GT）；cross-re(handoff off) → 每 stage 都 GT
        ic = gt_mean.clone() if (prev_pred is None or not handoff) else prev_pred.clone()
        # ★走 protocol：手抄 apply_ic_mask + 裸 init_state
        #   會與訓練/部署走到不同的算子，且少傳 ke_anchor 在 ceiling 模式下直接 RuntimeError。
        state = protocol.initial_state(model, trainer._cfg, static, ic, fluid_mask)
        _anchor = protocol.resolve_ke_anchor(trainer._cfg, state, fluid_mask)

        mae_c, mse_c, rel_c, ssim_c, ctu_c = [], [], [], [], []
        with torch.amp.autocast("cuda", enabled=trainer._cfg.use_amp):
            for step in range(1, n_steps + 1):
                state = model(state, cond_norm, fluid_mask, 1,
                              return_trajectory=False, g_vec_norm=g_vec_norm,
                              ke_anchor=_anchor)
                phys = state[:, sd:sd + pd].float()
                mae, mse, rel = _masked_mae_mse(phys, gt_mean, fluid_mask)
                mae_c.append(mae)
                mse_c.append(mse)
                rel_c.append(rel)
                ssim_c.append(struct_vel_rgb_ssim(phys, gt_mean, field_norm, fluid_mask))
                ctu_c.append(step * _CTU_PER_STEP)
        prev_pred = state[:, sd:sd + pd].float().detach()
        lab = labels[stage] if labels is not None else case_key.split("_")[-1]
        stages.append({
            "stage": stage, "case": case_key, "label": lab,
            "ctu": ctu_c, "mae": mae_c, "mse": mse_c, "rel_l2": rel_c, "ssim": ssim_c,
            "mae_final": mae_c[-1], "rel_l2_final": rel_c[-1], "ssim_final": ssim_c[-1],
        })
        print(f"  stage {stage} {lab}: "
              f"MAE_final={mae_c[-1]:.4f} relL2_final={rel_c[-1]:.4f} SSIM_final={ssim_c[-1]:.3f}")
    model.train()
    return stages


def _plot(res29: list[dict], res30: list[dict],
          out_path: str, n_steps: int, suptitle: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2), constrained_layout=True)
    span = n_steps * _CTU_PER_STEP

    def _concat(res, key):
        xs, ys = [], []
        for st in res:
            base = st["stage"] * span
            xs.extend([base + c for c in st["ctu"]])
            ys.extend(st[key])
        return np.asarray(xs), np.asarray(ys)

    for key, ax, lbl in (("mae", axes[0], "MAE (fluid, norm — magnitude-sensitive)"),
                         ("rel_l2", axes[1], "rel-L2 (fluid, /||target|| — magnitude-removed)"),
                         ("ssim", axes[2], "StructVel SSIM (fluid)")):
        x29, y29 = _concat(res29, key)
        x30, y30 = _concat(res30, key)
        ax.plot(x29, y29, color="tab:red", lw=1.3, alpha=0.85, label="v29 baseline")
        ax.plot(x30, y30, color="tab:blue", lw=1.3, alpha=0.85, label="v30 cross-static")
        n = max(len(res29), len(res30))
        for s in range(n):
            ax.axvline(s * span, color="gray", ls=":", lw=0.7, alpha=0.6)
            src = res30[s] if s < len(res30) else res29[s]
            tag = src.get("label", src["case"].split("_")[-1])
            # y 用 axes-fraction（get_xaxis_transform：x=data, y=[0,1]）→ log/linear 兩 panel 通用
            ax.text(s * span + 0.04 * span, 0.98, tag, transform=ax.get_xaxis_transform(),
                    fontsize=7, rotation=90, va="top", color="dimgray")
        ax.set_xlabel(f"cumulative pseudo-CTU (stage span={span:.0f}, {n_steps} steps × {_CTU_PER_STEP})")
        ax.set_title(lbl, fontsize=11)
        ax.grid(True, alpha=0.3)
        if key in ("mae", "rel_l2"):
            ax.set_yscale("log")
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(suptitle, fontsize=12)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved plot: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="mean v29 vs v30 逐 stage rollout（chain / cross-re）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--v29", required=True)
    ap.add_argument("--v30", required=True)
    ap.add_argument("--zarr", required=True, help="warm zarr（取 _cases.json）")
    ap.add_argument("--mode", choices=["chain", "cross-re"], default="chain",
                    help="chain=最長血緣鏈無 GT 交棒；cross-re=跨 Re 各 case 獨立 IC=GT")
    ap.add_argument("--n-re", type=int, default=10, help="cross-re 模式取幾個 Re 值")
    ap.add_argument("--ctu-per-stage", type=float, default=100.0)
    ap.add_argument("--out", default="runs/diagnostics/mean_chain_rollout_v29_v30.png")
    args = ap.parse_args()

    n_steps = max(1, int(round(args.ctu_per_stage / _CTU_PER_STEP)))
    cases_json = args.zarr.replace(".zarr", "_cases.json")

    mean_trainer = build_mean_training(args.config, finetune=args.v29)
    trainer = mean_trainer._backing
    index = _build_case_index(trainer)

    if args.mode == "chain":
        chain_ids = _find_chain(cases_json)
        stages = _resolve_present_chain(chain_ids, index)
        labels = None
        handoff = True
        suptitle = ("Chain-wise warm rollout (root IC=GT, children no-GT warm-from-parent-prediction)\n"
                    f"longest warm chain, {len(stages)} stages | v29 vs v30")
    else:
        stages, labels = _select_cross_re(cases_json, index, args.n_re)
        handoff = False
        suptitle = ("Cross-Re independent rollout (each stage IC=GT, no handoff)\n"
                    f"{len(stages)} Re values | v29 vs v30 | MAE(magnitude) vs rel-L2(magnitude-removed)")
    print(f"mode={args.mode}, stages={len(stages)}, steps/stage={n_steps}")
    if len(stages) < 2:
        raise RuntimeError("present stages < 2")

    print(f"[v29] {args.v29}")
    res29 = _rollout_chain(trainer, stages, n_steps, handoff=handoff, labels=labels)
    trainer.load_weights_only(args.v30)
    print(f"[v30] {args.v30}")
    res30 = _rollout_chain(trainer, stages, n_steps, handoff=handoff, labels=labels)

    _plot(res29, res30, args.out, n_steps, suptitle)

    # 逐 stage 摘要表 + JSON
    print("\n%-6s %-12s %9s %9s %9s %9s %8s %8s" %
          ("stage", "label", "v29_MAE", "v30_MAE", "v29_relL2", "v30_relL2", "v29_SS", "v30_SS"))
    print("-" * 80)
    summary = []
    for s in range(len(stages)):
        a, b = res29[s], res30[s]
        print("%-6d %-12s %9.4f %9.4f %9.4f %9.4f %8.3f %8.3f" %
              (s, a["label"], a["mae_final"], b["mae_final"],
               a["rel_l2_final"], b["rel_l2_final"], a["ssim_final"], b["ssim_final"]))
        summary.append({"stage": s, "case": a["case"], "label": a["label"],
                        "v29_mae": a["mae_final"], "v30_mae": b["mae_final"],
                        "v29_rel_l2": a["rel_l2_final"], "v30_rel_l2": b["rel_l2_final"],
                        "v29_ssim": a["ssim_final"], "v30_ssim": b["ssim_final"]})
    out_json = args.out.replace(".png", ".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "stages": [c for c, _ in stages], "n_steps": n_steps,
                   "ctu_per_stage": args.ctu_per_stage, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"saved json: {out_json}")


if __name__ == "__main__":
    main()
