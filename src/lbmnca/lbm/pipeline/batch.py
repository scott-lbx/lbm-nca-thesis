"""
batch.py — Generic C-series batch simulation runner.

Reads dataset/{name}/dataset_meta.json → runs all pending cases.

Warm datasets:
  - BFS order (depth 1 → depth 2 → ...): parents always before children
  - Failed warm case B → WarmReplacer creates D:
      * D picked from LOCAL NEIGHBORHOOD of B in 3D geom space (top-k KNN, random)
      * If B's parent is non-converged cold → substitute a converged cold root
      * Updates all direct children of B → point to D; regenerates their configs
      * Queues D at front before children
  - Startup: verify all cold roots are converged (warn on bad ones)

Cold datasets:
  - Topological order from dataset_meta.json (unchanged)

Usage:
    python -m lbmnca.lbm.pipeline.batch --dataset-name C1-1_Cold_M300C300
    python -m lbmnca.lbm.pipeline.batch --dataset-name C1-2_Warm_M300C300R32S100
    python -m lbmnca.lbm.pipeline.batch --dataset-name C0-2_Warm_M30C30R4S10 --dry-run
    python -m lbmnca.lbm.pipeline.batch --dataset-name C1-2_Warm_M300C300R32S100 --no-replace
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml

from lbmnca.lbm.lbm_mrt_les.pipeline.run_one_case_tes import run_tes_case
from lbmnca.paths import dataset_dir


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _save_meta(meta: dict, meta_path: Path) -> None:
    """原子寫入 `dataset_meta.json`。

    這個檔案在批次執行期間**每跑完一個 case 就重寫一次**，同時有其他行程在讀
    （健康監控、進度查詢）。直接 write_text 會有一段檔案內容不完整的視窗，
    讀方就會拿到截斷的 JSON。2026-08-01 的健康檢查即因此在印出任何訊息前就
    JSONDecodeError，只留下一行「腳本本身異常」。

    先寫同目錄的暫存檔再 os.replace——同一檔案系統上的 replace 是原子的，
    讀方要嘛看到舊版、要嘛看到新版，不會看到寫到一半的狀態。
    """
    tmp = meta_path.with_name(meta_path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, meta_path)


def _update_status(meta: dict, case_id: str, status: str, result: dict) -> None:
    for entry in meta["cases"]:
        if entry["case_id"] == case_id:
            entry["status"]      = status
            entry["started_at"]  = result.get("sim_start_utc", result.get("started_at", ""))
            entry["finished_at"] = _now()
            if status in ("success", "converged"):
                entry["stop_reason"]         = result.get("stop_reason", "")
                entry["convergence_ctu"]     = result.get("convergence_ctu", -1.0)
                entry["ke_variance_ratio"]   = result.get("ke_variance_ratio", -1.0)
                entry["n_snapshots"]         = result.get("n_snapshots_written", 0)
                entry["wall_time_total_s"]   = result.get("wall_time_total_s", -1.0)
                entry["wall_time_to_conv_s"] = result.get("wall_time_to_conv_s", -1.0)
                entry["sim_conv_utc"]        = result.get("sim_conv_utc", "")
            elif status == "replaced":
                entry["error"]       = result.get("reason", "unknown")
                entry["replaced_by"] = result.get("replaced_by", "")
            else:
                entry["error"] = result.get("reason", "unknown")
            return


# ── BFS order ─────────────────────────────────────────────────────────────────

def bfs_order(cases: list[dict]) -> list[str]:
    """Return case_ids sorted by BFS depth (skip precomputed/replaced)."""
    depth_groups: dict[int, list[str]] = defaultdict(list)
    for c in cases:
        if c.get("status") in ("precomputed", "replaced"):
            continue
        depth = int(c.get("warm_k_level", c.get("depth", 0)))
        depth_groups[depth].append(c["case_id"])
    result: list[str] = []
    for d in sorted(depth_groups.keys()):
        result.extend(depth_groups[d])
    return result


# ── Root verification ──────────────────────────────────────────────────────────

def _is_h5_converged(h5_path: Path) -> bool:
    if not h5_path.exists():
        return False
    try:
        with h5py.File(h5_path, "r") as fp:
            return bool(fp.attrs.get("converged", False))
    except OSError:
        return False


def verify_warm_roots(meta: dict) -> dict[str, bool]:
    """Check all cold roots (precomputed) for convergence. Warn on bad ones."""
    root_status: dict[str, bool] = {}
    for c in meta["cases"]:
        if c.get("status") != "precomputed":
            continue
        cid     = c["case_id"]
        cold_ds = c.get("cold_dataset", "")
        h5p     = dataset_dir(cold_ds) / "raw" / f"{cid}.h5"
        ok      = _is_h5_converged(h5p)
        root_status[cid] = ok
        if not ok:
            try:
                with h5py.File(h5p, "r") as fp:
                    stop = str(fp.attrs.get("stop_reason", "?"))
            except Exception:
                stop = "H5_missing_or_locked"
            print(f"  [ROOT WARN] {cid} NOT converged (stop={stop}) — "
                  f"depth-1 children will be auto-replaced on failure")
    bad = sum(1 for v in root_status.values() if not v)
    print(f"  [ROOT CHECK] {len(root_status)} roots: "
          f"{len(root_status)-bad} OK  {bad} non-converged")
    return root_status


# ── Child config regen ────────────────────────────────────────────────────────

def _regen_child_config(child_entry: dict, new_parent_id: str,
                        ds_dir: Path, dataset_id: str) -> None:
    """Update child's YAML to reference new parent H5."""
    cfg_path = ds_dir / child_entry["config_path"]
    if not cfg_path.exists():
        return
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    parent_ds = child_entry.get("parent_dataset", dataset_id)
    new_h5 = str(dataset_dir(parent_ds) / "raw" / f"{new_parent_id}.h5")
    cfg["tes"]["parent_h5_path"] = new_h5
    cfg["tes"]["parent_case_id"] = new_parent_id
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ── Warm replacer ─────────────────────────────────────────────────────────────

class WarmReplacer:
    """
    Lazy-loads population + geom data; creates LOCAL-NEIGHBORHOOD replacement
    warm cases when a warm simulation fails.

    Replacement D for failed B:
      1. Parent check: if B's parent (cold root) is non-converged → substitute
         with a different converged cold case from cold_dataset.
      2. Mask pick: find top-K nearest neighbours of B in 3D geom space
         (Cube4D PCA), pick randomly from unused K neighbours.
         Falls back to global random only when neighbourhood is exhausted.
      3. Config gen + meta update; returns (new_id, new_entry).

    cube_geom is stored in meta entries for replacement chains:
    if D itself later fails, the next replacement E uses D's stored geom.
    """

    _N_SUBSAMPLE = 200_000
    _K_NEIGHBORS = 20     # pick from top-K nearest

    def __init__(self, ds_dir: Path,
                 dataset_id: str, cold_dataset: str) -> None:
        self.ds_dir       = ds_dir
        self.dataset_id   = dataset_id
        self.cold_dataset = cold_dataset
        self._loaded      = False
        self._replace_count: dict[str, int] = {}
        # Populated in _load():
        self._params    = None
        self._counts    = None
        self._off       = None
        self._cand_idx  = None   # (n_sub,) pop layout indices in subsample
        self._geom_pts  = None   # (n_sub, 3) 3D geom coords for subsample
        self._used: set[int] = set()
        self._case_geom: dict[str, list[float]] = {}  # case_id → cube_geom[3]

    # ── Lazy load ──────────────────────────────────────────────────────────────

    def _load(self, meta: dict) -> None:
        if self._loaded:
            return
        from lbmnca.morphospace.param_pca import config as _C
        from lbmnca.morphospace.param_pca.cube import Cube4D
        from lbmnca.morphospace.param_pca.features import featurize
        from lbmnca.paths import design_space_dir
        self._C = _C

        # Population NPZ
        pop_npz = design_space_dir() / "population_big.npz"
        d = np.load(pop_npz)
        self._params = d["params"]
        self._counts = d["counts"]
        self._off = np.concatenate([[0], np.cumsum(self._counts)]).astype(np.int64)
        M = len(self._counts)

        # Cube4D for geom projection
        cube_npz = design_space_dir() / "cube4d.npz"
        self._cube4d = Cube4D.load(cube_npz)

        # Collect already-used pop indices
        cases_json = self.ds_dir / "cases.json"
        if cases_json.exists():
            cdata = json.loads(cases_json.read_text())
            for n in cdata.get("nodes", []):
                idx = n.get("pop_layout_idx", -1)
                if idx >= 0:
                    self._used.add(idx)
                if "cube_geom" in n:
                    self._case_geom[n["case_id"]] = n["cube_geom"]
        for c in meta.get("cases", []):
            idx = c.get("pop_layout_idx", -1)
            if idx >= 0:
                self._used.add(idx)
            if "cube_geom" in c:
                self._case_geom[c["case_id"]] = c["cube_geom"]

        # Subsample for neighbourhood search (exclude used)
        pool_arr = np.array([i for i in range(M) if i not in self._used], dtype=np.int64)
        n_sub = min(self._N_SUBSAMPLE, len(pool_arr))
        rng_sub = np.random.RandomState(seed=7)
        self._cand_idx = rng_sub.choice(pool_arr, n_sub, replace=False)

        print(f"  [WarmReplacer] featurizing {n_sub} candidates for geom lookup...")
        X_sub = np.stack([
            featurize(self._params[self._off[i]:self._off[i + 1]])
            for i in self._cand_idx
        ])
        self._geom_pts = self._cube4d.geom_coords(X_sub)  # (n_sub, 3)

        self._loaded = True
        print(f"  [WarmReplacer] ready: {M} layouts, {len(self._used)} used, "
              f"{n_sub} candidates indexed")

    # ── Mask render ────────────────────────────────────────────────────────────

    def _render_mask(self, buildings: list) -> "np.ndarray":
        C = self._C
        mask = np.full((C.H_TOTAL, C.W_TOTAL), 255, np.uint8)
        for cx, cy, w, h in buildings:
            x0 = max(0, int(round(cx - w / 2)))
            x1 = min(C.W_TOTAL, int(round(cx + w / 2)))
            y0 = max(0, int(round(cy - h / 2)))
            y1 = min(C.H_TOTAL, int(round(cy + h / 2)))
            mask[y0:y1, x0:x1] = 0
        return mask

    # ── Converged cold parent search ───────────────────────────────────────────

    def _find_converged_cold(self, exclude_ids: set[str]) -> tuple[str, str]:
        """Find any converged cold H5 in cold_dataset not in exclude_ids."""
        cold_raw = dataset_dir(self.cold_dataset) / "raw"
        for h5p in sorted(cold_raw.glob("*.h5")):
            if h5p.stem in exclude_ids:
                continue
            if _is_h5_converged(h5p):
                return h5p.stem, self.cold_dataset
        return "", ""

    # ── Local neighbourhood pick ───────────────────────────────────────────────

    def _pick_local(self, b_geom: list[float],
                    rng: "np.random.RandomState") -> tuple[int, list[float]]:
        """
        Pick a pop layout index near b_geom in 3D geom space.
        Returns (pop_layout_idx, geom_coords).
        """
        b_arr = np.array(b_geom, dtype=float)
        dists = np.linalg.norm(self._geom_pts - b_arr, axis=1)

        # Progressively expand search until we find K unused candidates
        for k in (self._K_NEIGHBORS, 50, 200, len(dists)):
            k = min(k, len(dists))
            cand_local = np.argpartition(dists, k - 1)[:k]
            usable = [
                li for li in cand_local
                if int(self._cand_idx[li]) not in self._used
            ]
            if usable:
                li = int(rng.choice(usable))
                pop_i = int(self._cand_idx[li])
                return pop_i, self._geom_pts[li].tolist()

        # Global fallback (all neighbours exhausted)
        print("  [WarmReplacer] neighbourhood exhausted — global fallback")
        pool = [i for i in range(len(self._counts)) if i not in self._used]
        pop_i = int(rng.choice(pool if pool else range(len(self._counts))))
        return pop_i, []

    # ── Main entry ─────────────────────────────────────────────────────────────

    def create_replacement(
        self,
        failed_entry: dict,
        meta: dict,
        rng: "np.random.RandomState",
    ) -> tuple[str, dict]:
        """
        Create replacement D for failed warm case B.
        Returns (new_case_id, new_meta_entry).
        """
        self._load(meta)

        failed_id      = failed_entry["case_id"]
        depth          = int(failed_entry.get("warm_k_level", 1))
        re             = int(failed_entry.get("re", 200))
        orig_parent_id = failed_entry.get("parent_case_id", "")
        orig_parent_ds = failed_entry.get("parent_dataset", self.dataset_id)

        # Increment replacement counter for this case
        n = self._replace_count.get(failed_id, 0) + 1
        self._replace_count[failed_id] = n
        new_id = f"{failed_id}_r{n}"

        # ── Parent check: must be converged cold ───────────────────────────────
        parent_h5 = dataset_dir(orig_parent_ds) / "raw" / f"{orig_parent_id}.h5"
        if _is_h5_converged(parent_h5):
            parent_id = orig_parent_id
            parent_ds = orig_parent_ds
        else:
            print(f"  [WarmReplacer] parent {orig_parent_id} not converged → "
                  f"searching for substitute cold root...")
            parent_id, parent_ds = self._find_converged_cold({orig_parent_id})
            if not parent_id:
                raise RuntimeError(
                    f"No converged cold case available as substitute for {failed_id}"
                )
            print(f"  [WarmReplacer] substitute root: {parent_id} ({parent_ds})")

        # ── Local neighbourhood pick for new mask ─────────────────────────────
        b_geom = failed_entry.get("cube_geom") or self._case_geom.get(failed_id)

        if b_geom:
            pop_idx, new_geom = self._pick_local(b_geom, rng)
        else:
            print(f"  [WarmReplacer] no geom info for {failed_id} → global pick")
            pool = [i for i in range(len(self._counts)) if i not in self._used]
            pop_idx  = int(rng.choice(pool if pool else range(len(self._counts))))
            new_geom = []

        self._used.add(pop_idx)
        buildings = self._params[self._off[pop_idx]:self._off[pop_idx + 1]].tolist()

        # ── Render + save mask ─────────────────────────────────────────────────
        mask = self._render_mask(buildings)
        from PIL import Image  # type: ignore
        (self.ds_dir / "masks").mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask, mode="L").save(
            self.ds_dir / "masks" / f"{new_id}.png"
        )

        # ── Generate config YAML ───────────────────────────────────────────────
        from lbmnca.lbm.config_gen.case_config import build_yaml_config
        from lbmnca.paths import data_root

        node_dict = {
            "case_id":        new_id,
            "mask_id":        new_id,
            "re":             re,
            "ic_type":        "warm",
            "parent_case_id": parent_id,
            "parent_dataset": parent_ds,
            "depth":          depth,
            "warm_k_level":   depth,
            "n_buildings":    int(len(buildings)),
            "buildings":      [[float(v) for v in b] for b in buildings],
        }
        cfg_dict = build_yaml_config(
            node_dict, self.dataset_id, self.ds_dir / "masks", data_root()
        )
        cfg_path = self.ds_dir / "configs" / f"{new_id}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)

        new_entry: dict = {
            "case_id":        new_id,
            "mask_id":        new_id,
            "re":             re,
            "ic_type":        "warm",
            "depth":          depth,
            "warm_k_level":   depth,
            "parent_case_id": parent_id,
            "parent_dataset": parent_ds,
            "cube_geom":      new_geom,   # stored for chained replacements
            "config_path":    f"configs/{new_id}.yaml",
            "h5_output_path": f"raw/{new_id}.h5",
            "status":         "pending",
            "replaced_from":  failed_id,
            "pop_layout_idx": pop_idx,
        }
        # Cache geom for potential future use
        if new_geom:
            self._case_geom[new_id] = new_geom

        print(f"  [WarmReplacer] {new_id}  Re={re}  depth={depth}  "
              f"parent={parent_id}  local_pick={'yes' if b_geom else 'no'}")
        return new_id, new_entry


def _parent_status(parent_ds: str, parent_id: str) -> str:
    """查 parent case 在其 dataset_meta 中的 status；查不到回空字串。

    刻意不快取：parent 可能在同一輪批次中稍後才跑成功，快取住舊值會讓
    本來可以執行的 warm case 被永久擋掉。相對於一個 case 數分鐘的模擬，
    讀一次 JSON 的成本可以忽略。
    """
    p = dataset_dir(parent_ds) / "dataset_meta.json"
    if not p.exists():
        return ""
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    for c in meta.get("cases", []):
        if c.get("case_id") == parent_id:
            return str(c.get("status", ""))
    return ""


# ── run_one ────────────────────────────────────────────────────────────────────

def run_one(entry: dict, ds_dir: Path, masks_dir: Path,
            configs_dir: Path, raw_dir: Path, dry_run: bool) -> tuple[str, dict]:
    """Returns (status, result_dict). status: 'skip' | 'success' | 'failed'."""
    case_id = entry["case_id"]

    if entry.get("status") in ("precomputed", "replaced"):
        return "skip", {}

    # ★ resume 判據必須與收錄閘門一致：**H5 存在 ≠ 已完成**。
    #   沒通過物理閘門的 case 一樣留著 H5（它跑完了，只是被判退），
    #   若只看檔案存在就跳過，補跑（改參數 + status 重設為 pending）會完全失效。
    #   只有 status 已是 success 的才是真正的成品，可以跳過。
    # ★ resume 判據必須與收錄閘門一致：**H5 存在 ≠ 已完成**。
    #   沒通過物理閘門的 case 一樣留著 H5（它跑完了，只是被判退），
    #   若只看檔案存在就跳過，補跑（改參數 + status 重設為 pending）會完全失效。
    #
    #   只有 pending 該重跑：
    #   - success：已收錄，跳過
    #   - failed ：已判退且**沒有**標記補跑。模擬是決定性的，原樣重跑必得同樣
    #              結果、被同樣理由再擋一次，純浪費。要補跑得先用
    #              prepare_rerun_failed.py 調整參數並改回 pending。
    h5_path = raw_dir / f"{case_id}.h5"
    if h5_path.exists():
        st = str(entry.get("status"))
        if st == "success":
            print(f"[SKIP] {case_id}  (已收錄)")
            return "skip", {}
        if st == "failed":
            print(f"[SKIP] {case_id}  (已判退，未標記補跑："
                  f"{str(entry.get('error', ''))[:60]})")
            return "skip", {}
        print(f"[RERUN] {case_id}  (status={st!r} → 重跑覆寫)")

    if entry.get("ic_type") == "warm":
        parent_id = entry.get("parent_case_id", "")
        parent_ds = entry.get("parent_dataset", "")
        if parent_ds and parent_id:
            parent_h5 = dataset_dir(parent_ds) / "raw" / f"{parent_id}.h5"
            if not parent_h5.exists():
                print(f"[WAIT] {case_id}  parent H5 missing: {parent_h5}")
                return "failed", {"reason": f"parent H5 missing: {parent_h5}"}
            # ★ 光有 H5 不夠：parent 必須**通過物理閘門**才算可用。
            #   被判退的 parent 一樣留著 H5，拿它當初始條件會把已知有問題的流場
            #   一路傳染給整條 warm 鏈，而且下游沒有任何機制會再發現。
            pst = _parent_status(parent_ds, parent_id)
            if pst != "success":
                print(f"[WAIT] {case_id}  parent 未通過收錄："
                      f"{parent_ds}/{parent_id} status={pst!r}")
                return "failed", {"reason": f"parent not admitted: "
                                            f"{parent_ds}/{parent_id} "
                                            f"status={pst}"}

    cfg_path = configs_dir / f"{case_id}.yaml"
    if not cfg_path.exists():
        print(f"[ERROR] {case_id}  config not found: {cfg_path}")
        return "failed", {"reason": f"config not found: {cfg_path}"}

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mask_id   = cfg["tes"]["geometry_id"]
    mask_path = str(masks_dir / f"{mask_id}.png")
    ic_type   = cfg["tes"]["ic_type"]
    re        = cfg["tes"]["re_target"]
    depth     = cfg["tes"].get("warm_k_level", 0)

    print(f"\n[RUN] {case_id}  ic={ic_type}  Re={re}  depth={depth}")

    if dry_run:
        print(f"  [dry-run] would run {case_id}")
        return "skip", {}

    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_tes_case(cfg, mask_path, str(h5_path))
    except Exception as e:
        traceback.print_exc()
        result = {"status": "Error", "reason": str(e)}

    if result.get("status") != "Success":
        print(f"  [FAIL] {result.get('reason', 'unknown')}")
        return "failed", result

    # ★物理驗證閘門（2026-07-29 使用者裁決）：模擬「跑完」不等於「成功」。
    #   每個 case 跑完必須通過 validation.case_physics，否則一律作廢——
    #   該 case 要重新生成 mask/config 再重跑，不得收錄。
    #   求解器層級的 L1/L2（解析解、文獻錨點）另在批次開始前跑一次，
    #   不在此處重複。
    from lbmnca.lbm.lbm_mrt_les.validation.case_physics import validate_h5  # noqa: PLC0415

    vrep = validate_h5(h5_path)
    result["physics_validation"] = vrep.to_dict()
    if not vrep.passed:
        ids = ", ".join(c.id for c in vrep.failures)
        print(f"  [REJECT] 物理驗證未通過：{ids}")
        for c in vrep.failures:
            print(f"            {c.id} {c.name}: value={c.value} "
                  f"thr={c.threshold}  {c.detail}")
        result["reason"] = f"physics_validation_failed: {ids}"
        return "failed", result

    print(f"  [DONE] convergence_ctu={result.get('convergence_ctu', -1):.1f}  "
          f"snaps={result.get('n_snapshots_written', 0)}  物理驗證 PASS")
    return "success", result


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-replace", action="store_true",
        help="Disable automatic replacement of failed warm cases",
    )
    ap.add_argument(
        "--max-cases", type=int, default=0,
        help="只跑前 N 個實際執行的 case（0=全部），冒煙測試用",
    )
    args = ap.parse_args()

    ds_dir      = dataset_dir(args.dataset_name)
    meta_path   = ds_dir / "dataset_meta.json"
    masks_dir   = ds_dir / "masks"
    configs_dir = ds_dir / "configs"
    raw_dir     = ds_dir / "raw"

    meta = _load_meta(meta_path)

    is_warm = any(
        e.get("ic_type") == "warm" and e.get("status") != "precomputed"
        for e in meta["cases"]
    )

    # ── Root verification (warm datasets) ─────────────────────────────────────
    if is_warm:
        print(f"[BATCH] {args.dataset_name}  warm — verifying cold roots...")
        verify_warm_roots(meta)

    # ── Simulation order ───────────────────────────────────────────────────────
    if is_warm:
        sim_order = bfs_order(meta["cases"])
        print(f"[BATCH] {args.dataset_name}  BFS order  "
              f"{len(sim_order)} active cases  dry_run={args.dry_run}")
    else:
        sim_order = meta.get("sim_order", [e["case_id"] for e in meta["cases"]])
        print(f"[BATCH] {args.dataset_name}  {len(sim_order)} cases  "
              f"dry_run={args.dry_run}")

    id_to_entry = {e["case_id"]: e for e in meta["cases"]}
    queue       = deque(sim_order)

    replacer: WarmReplacer | None = None
    rng_replace = np.random.RandomState(seed=42)
    if is_warm and not args.no_replace:
        replacer = WarmReplacer(
            ds_dir,
            dataset_id   = meta["dataset_id"],
            cold_dataset = meta.get("cold_dataset", ""),
        )

    total          = len(queue)
    skipped        = 0
    done           = 0
    failed_total   = 0
    replaced_count = 0

    attempted = 0
    while queue:
        if args.max_cases and attempted >= args.max_cases:
            print(f"\n[BATCH] --max-cases {args.max_cases} 達到，提前停止")
            break
        case_id = queue.popleft()
        entry   = id_to_entry.get(case_id)
        if entry is None:
            continue

        i = done + skipped + failed_total + 1
        print(f"\n[{i:03d}/{total}] ", end="")

        status, result = run_one(
            entry, ds_dir, masks_dir, configs_dir, raw_dir, args.dry_run
        )
        # ★只計真正跑過的。把 skip 也算進去會讓 --max-cases 1 的守閘用法空轉：
        #   每輪都先撞到已完成的 case、skip、計數達標即結束，永遠不推進。
        if status != "skip":
            attempted += 1

        if status == "skip":
            skipped += 1

        elif status == "success":
            done += 1

        else:  # failed
            failed_total += 1

            if replacer and entry.get("ic_type") == "warm":
                print(f"  [REPLACE] {case_id} failed — sampling local replacement...")
                try:
                    new_id, new_entry = replacer.create_replacement(
                        entry, meta, rng_replace
                    )

                    # Update direct children: re-root to new_id
                    children_updated: list[str] = []
                    for e in meta["cases"]:
                        if e.get("parent_case_id") == case_id:
                            e["parent_case_id"] = new_id
                            _regen_child_config(
                                e, new_id, ds_dir, meta["dataset_id"]
                            )
                            children_updated.append(e["case_id"])

                    # Register replacement
                    meta["cases"].append(new_entry)
                    id_to_entry[new_id] = new_entry
                    total += 1
                    result["replaced_by"] = new_id

                    # Insert replacement before its children
                    queue.appendleft(new_id)
                    replaced_count += 1

                    if children_updated:
                        print(f"  [REPLACE] children re-routed → {new_id}: "
                              f"{children_updated}")

                except Exception as exc:
                    print(f"  [REPLACE ERROR] {exc}")
                    traceback.print_exc()

        # ★ skip 代表「這一輪什麼都沒做」，meta 必須原封不動。
        #   舊寫法把 skip 映射成 success，並用空的 result 呼叫 _update_status，
        #   造成兩重破壞：(a) 未通過物理閘門的 case 被洗白成已收錄；
        #   (b) convergence_ctu / n_snapshots / wall_time 等真實欄位被
        #   -1.0 / 0 覆寫。跳過一個 case 不該產生任何寫入。
        if not args.dry_run and status != "skip":
            final_status = (
                "replaced"
                if (status == "failed" and result.get("replaced_by"))
                else status
            )
            _update_status(meta, case_id, final_status, result)
            _save_meta(meta, meta_path)

        print(
            f"  Progress: {i}/{total}  done={done}  skip={skipped}  "
            f"fail={failed_total}  replaced={replaced_count}"
        )

    print(f"\n[BATCH DONE] {args.dataset_name}")
    print(
        f"  total={total}  done={done}  skip={skipped}  "
        f"fail={failed_total}  replaced={replaced_count}"
    )
    if failed_total and not replaced_count:
        print(f"  WARNING: {failed_total} unreplaced failures — check dataset_meta.json")


if __name__ == "__main__":
    main()
