"""Physics-Proxy CLI：批量計算全部 mask 的代理特徵，輸出 p_stats.json。

使用方式：
    python -m lbmnca.morphospace.physics_proxy.cli \\
        --masks-dir ../dataset/CEDS_dataset/flat200/masks \\
        --metadata  ../dataset/CEDS_dataset/flat200/mask_meta.json \\
        --out       data/p_stats.json \\
        [--R 64] [--pad-left 256] [--pad-top 156] [--inner-px 768] \\
        [--Re 500.0] [--U 0.04] [--L 6.0]

也可直接從 vae_config.yaml 讀取全部參數：
    python -m lbmnca.morphospace.physics_proxy.cli --config configs/vae_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lbmnca.morphospace.canonicalize.pipeline import mask_to_canonical
from lbmnca.paths import REPO_ROOT
from .features import FEATURE_DIM, FEATURE_NAMES, SimConfig, compute_all
from .normalize import fit_stats, save_stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="批量計算 physics-proxy 特徵並輸出 p_stats.json"
    )
    p.add_argument("--config", type=Path, default=None,
                   help="vae_config.yaml 路徑（若提供則覆蓋其他參數）")
    p.add_argument("--masks-dir", type=Path, default=None)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("data/p_stats.json"))
    p.add_argument("--combined-meta", type=str, default=None,
                   help="combined mask_meta.json（含 per-entry source_dir），與 --masks-dir 二擇一")
    p.add_argument("--R", type=int, default=64)
    p.add_argument("--pad-left", type=int, default=256)
    p.add_argument("--pad-top", type=int, default=156)
    p.add_argument("--inner-px", type=int, default=768)
    p.add_argument("--Re", type=float, default=500.0)
    p.add_argument("--U", type=float, default=0.04,
                   help="U_inlet_lu（lattice units）")
    p.add_argument("--L", type=float, default=6.0,
                   help="L_char_px（pixels）")
    return p.parse_args()


def _load_config_yaml(path: Path) -> dict:
    import yaml
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_from_config(cfg: dict, out_override: Path | None) -> tuple[Path, Path, dict, SimConfig, Path]:
    """從 vae_config.yaml 解析出執行參數。

    Returns:
        (masks_dir, metadata_json, canonical_params, sim_cfg, out_path)
    """
    d = cfg["data"]
    pp = cfg["physics_proxy"]

    masks_dir = (REPO_ROOT / d["masks_dir"]).resolve()
    metadata_json = (REPO_ROOT / d["metadata_json"]).resolve()
    canonical_params = {
        "R": d["R"],
        "pad_left": d["pad_left"],
        "pad_top": d["pad_top"],
        "inner_px": d["inner_px"],
    }
    sim_cfg: SimConfig = {
        "Re": float(pp["ref_Re"]),
        "U_inlet_lu": float(pp["ref_U_inlet_lu"]),
        "L_char_px": float(pp["ref_L_char_px"]),
    }
    out_path = out_override if out_override else (REPO_ROOT / pp["p_stats_path"])
    return masks_dir, metadata_json, canonical_params, sim_cfg, out_path


def _entry_mask_path(entry: dict, default_masks_dir: Path) -> Path:
    """解析單一 entry 的 mask 絕對路徑。

    combined dataset 的 entry 含 "source_dir"（絕對路徑字串），
    一般 dataset 使用 default_masks_dir。
    """
    base = Path(entry["source_dir"]) if "source_dir" in entry else default_masks_dir
    return base / entry["file_name"]


def run(
    masks_dir: Path | None,
    metadata_json: Path | None,
    out_path: Path,
    R: int,
    pad_left: int,
    pad_top: int,
    inner_px: int,
    sim_cfg: SimConfig,
    combined_meta: str | None = None,
) -> None:
    """批量計算所有 mask 的 proxy 特徵，fit stats，儲存 JSON。

    Args:
        masks_dir: masks PNG 目錄（single-source 模式）。combined_meta 存在時可為 None。
        metadata_json: mask_meta.json 路徑（single-source 模式）。
        out_path: 輸出 p_stats.json 路徑。
        R: canonical 解析度。
        pad_left: 左 crop 偏移 px。
        pad_top: 上 crop 偏移 px。
        inner_px: inner domain 邊長 px。
        sim_cfg: 用於 E 群組特徵計算的模擬參數。
        combined_meta: combined metadata JSON 路徑（含 per-entry source_dir）。
                       若提供則忽略 masks_dir / metadata_json。
    """
    if combined_meta is not None:
        with open(combined_meta, encoding="utf-8") as f:
            meta: list[dict] = json.load(f)
        default_dir = Path(".")
    else:
        assert masks_dir is not None and metadata_json is not None
        with metadata_json.open(encoding="utf-8") as f:
            meta = json.load(f)
        default_dir = masks_dir

    entries = [e for e in meta if _entry_mask_path(e, default_dir).exists()]
    n = len(entries)
    print(f"[cli] 找到 {n} 個有效 mask（共 {len(meta)} 筆）")
    print(f"[cli] sim_cfg: Re={sim_cfg['Re']}, U={sim_cfg['U_inlet_lu']}, L={sim_cfg['L_char_px']}")

    raw_rows: list[np.ndarray] = []
    n_skip = 0

    for i, entry in enumerate(entries):
        path = _entry_mask_path(entry, default_dir)
        try:
            binary_arr = mask_to_canonical(
                path,
                R=R,
                pad_left=pad_left,
                pad_top=pad_top,
                inner_px=inner_px,
                use_sdf=False,
            )[0]  # (R, R) float32, 1=solid
            binary_uint8 = (binary_arr >= 0.5).astype(np.uint8)
            raw_p = compute_all(binary_uint8, sim_cfg)
            raw_rows.append(raw_p)
        except Exception as e:
            print(f"  [skip] {entry['file_name']}: {e}")
            raw_rows.append(np.full(FEATURE_DIM, np.nan, dtype=np.float32))
            n_skip += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} 處理完畢...")

    raw_matrix = np.stack(raw_rows, axis=0)  # [N, 22]
    print(f"[cli] 計算完畢（跳過 {n_skip} 個），開始 fit_stats...")

    stats = fit_stats(raw_matrix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_stats(stats, out_path)

    print(f"[cli] p_stats.json 已儲存至 {out_path}")
    print(f"[cli] n_samples={stats.n_samples}")
    print("\n特徵摘要（mean ± std）：")
    for name, m, s in zip(FEATURE_NAMES, stats.mean, stats.std):
        print(f"  {name:45s}  {m:10.4f} ± {s:.4f}")


def main() -> None:
    """CLI 入口點。"""
    args = _parse_args()

    if args.config is not None:
        cfg = _load_config_yaml(args.config)
        masks_dir, metadata_json, cp, sim_cfg, out_path = _resolve_from_config(
            cfg, args.out if args.out != Path("data/p_stats.json") else None
        )
        R, pad_left, pad_top, inner_px = (
            cp["R"], cp["pad_left"], cp["pad_top"], cp["inner_px"]
        )
    elif args.combined_meta is not None:
        masks_dir = None
        metadata_json = None
        R, pad_left, pad_top, inner_px = args.R, args.pad_left, args.pad_top, args.inner_px
        sim_cfg: SimConfig = {
            "Re": args.Re,
            "U_inlet_lu": args.U,
            "L_char_px": args.L,
        }
        out_path = args.out
    else:
        if args.masks_dir is None or args.metadata is None:
            print("錯誤：請提供 --config、--combined-meta 或同時提供 --masks-dir 與 --metadata",
                  file=sys.stderr)
            sys.exit(1)
        masks_dir = args.masks_dir.resolve()
        metadata_json = args.metadata.resolve()
        R, pad_left, pad_top, inner_px = args.R, args.pad_left, args.pad_top, args.inner_px
        sim_cfg = {
            "Re": args.Re,
            "U_inlet_lu": args.U,
            "L_char_px": args.L,
        }
        out_path = args.out

    run(
        masks_dir=masks_dir if not args.combined_meta else None,
        metadata_json=metadata_json if not args.combined_meta else None,
        out_path=(REPO_ROOT / out_path) if not out_path.is_absolute() else out_path,
        R=R,
        pad_left=pad_left,
        pad_top=pad_top,
        inner_px=inner_px,
        sim_cfg=sim_cfg,
        combined_meta=args.combined_meta,
    )


if __name__ == "__main__":
    main()
