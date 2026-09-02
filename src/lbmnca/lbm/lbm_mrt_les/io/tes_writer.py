"""TES trajectory HDF5 writer — per docs/methodology/dataset_structure_tes.md §5 (v2).

每個 case 輸出：
  /attrs              — case-level metadata
  /geometry/mask      — bool [H5sy=360, H5sx=640] （唯一幾何欄位，v2）
  /g_vec              — 14D geometry proxy (fp32)
  /convergence_info/  — kinetic_energy, mass_residual per snapshot
  /trajectory/moments — [N_snap, 9, 360, 640] fp16, lzf+shuffle
  /trajectory/lbm_steps, /trajectory/ctu

v2 變更（2026-05-13）：
  - Staircase CTU 排程（取代 Fibonacci）
  - DOWNSAMPLE_FACTOR: 4 → 3（H5s = 640×360）
  - Crop: sponge+buffer 兩層一起裁（ROI = Mask = 1920×1080）
  - geometry 只存 mask（移除 sdf/upstream_norm/x_norm/y_norm）
  - compression: gzip → lzf
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import cv2
import h5py
import numpy as np


# ── Snapshot schedule ──────────────────────────────────────────────────────────

# Staircase CTU schedule phases — (Δ_ctu, from_ctu_exclusive, to_ctu_inclusive)
# None for to_ctu means "extend to total_ctus".
# Schedule auto-truncates by total_ctus, so cold (30 CTU) and warm (5 CTU)
# share one phase list — warm runs only Phase 0+1 before truncation.
#
# Unified phases:
#   Phase 0: Δ=0.2  t∈(0,  1]  →  5 pts  (inlet establishment)
#   Phase 1: Δ=1.0  t∈(0,  5]  →  5 pts  (IC impact, dedupes t=1.0 with Phase 0)
#   Phase 2: Δ=1.0  t∈(5, 20]  → 15 pts  (main flow buildup)
#   Phase 3: Δ=2.0  t∈(20,30]  →  5 pts  (quasi-steady)
#   Cold (30 CTU): 29 unique snapshots
#   Warm (5 CTU):   9 unique snapshots (Phases 0+1 only)
STAIRCASE_PHASES: list[tuple[float, float, float | None]] = [
    (0.2,  0.0,  1.0),
    (1.0,  0.0,  5.0),
    (1.0,  5.0, 20.0),
    (2.0, 20.0, None),
]

# Backwards-compat aliases (both point to unified STAIRCASE_PHASES).
STAIRCASE_PHASES_COLD = STAIRCASE_PHASES
STAIRCASE_PHASES_WARM = STAIRCASE_PHASES


def build_snapshot_schedule_staircase(
    effective_steps: int,
    steps_per_ctu: float,
    total_ctus: float,
    phases: list[tuple[float, float, float | None]] | None = None,
) -> list[int]:
    """Staircase CTU schedule: 分段均勻間距，密→疏，避免 Fibonacci 早期假收斂。

    Args:
        effective_steps: Total LBM steps in the recording window (after spinup).
        steps_per_ctu: l_char_px / u_inlet_lu (step count per one CTU).
        total_ctus: Total recording duration in CTU.
        phases: Phase list [(Δ_ctu, from_ctu, to_ctu|None), ...].
            Defaults to STAIRCASE_PHASES when None.

    Returns:
        Sorted list of LBM step integers (relative to start of recording window).
        Always includes effective_steps as the final entry.
    """
    if phases is None:
        phases = STAIRCASE_PHASES

    ctu_points: list[float] = []
    for step_ctu, from_ctu, to_ctu in phases:
        actual_to = min(to_ctu, total_ctus) if to_ctu is not None else total_ctus
        if from_ctu >= total_ctus:
            break
        t = from_ctu + step_ctu
        while t <= actual_to + 1e-9:
            if t <= total_ctus + 1e-9:
                ctu_points.append(round(t, 6))
            t = round(t + step_ctu, 6)

    step_set: set[int] = set()
    for ctu in ctu_points:
        s = max(1, min(effective_steps, round(ctu * steps_per_ctu)))
        step_set.add(s)
    step_set.add(effective_steps)
    return sorted(step_set)


def build_snapshot_schedule_dense(
    effective_steps: int,
    steps_per_ctu: float,
    total_ctus: float,
    spacing_ctu: float = 0.2,
) -> list[int]:
    """Dense 等距 CTU 排程：全程每 ``spacing_ctu`` CTU 取一幀（plan §3.4）。

    取代 staircase 稀疏排程，用於 warm-start 資料集（S1_4-WarmMixGeoRe）：
      - 發展區（warm-start 操作區）每步皆有 GT → 不 alias 脫落（Strouhal）。
      - 修正 Dy2 訓練 ``delta_ctu bucket [0.1,0.5)`` 無純 0.2 窗的 fallback bug。
    收斂後遠端 limit cycle 的冗餘尾段由 ``_run_trajectory_loop`` 的
    ``extra_ctu_after_conv`` 自動 cap（不需在此截斷）。

    Args:
        effective_steps: 記錄窗（spinup 後）的總 LBM 步數。
        steps_per_ctu: l_char_px / u_inlet_lu（每 CTU 的步數）。
        total_ctus: 記錄總時長（CTU）。
        spacing_ctu: 幀間距（CTU），預設 0.2。

    Returns:
        相對於記錄窗起點的 LBM step 整數排序清單；必含 ``effective_steps``。
    """
    if spacing_ctu <= 0.0:
        raise ValueError(f"spacing_ctu must be > 0, got {spacing_ctu}")

    step_set: set[int] = set()
    n_frames = int(round(total_ctus / spacing_ctu))
    for k in range(1, n_frames + 1):
        ctu = round(k * spacing_ctu, 6)
        if ctu > total_ctus + 1e-9:
            break
        s = max(1, min(effective_steps, round(ctu * steps_per_ctu)))
        step_set.add(s)
    step_set.add(effective_steps)
    return sorted(step_set)


def build_snapshot_schedule(total_steps: int) -> list[int]:
    """Deprecated: Fibonacci schedule. Use build_snapshot_schedule_staircase instead."""
    fib: list[int] = [1, 2]
    while fib[-1] + fib[-2] < total_steps:
        fib.append(fib[-1] + fib[-2])
    return sorted(set(fib + [total_steps]))


def total_steps_from_ctus(
    l_char_px: int,
    total_ctus: float = 30.0,
    u_inlet_lu: float = 0.05,
) -> int:
    """Compute total LBM steps for a given CTU target."""
    return max(1, round(total_ctus * l_char_px / u_inlet_lu))


# ── Writer ─────────────────────────────────────────────────────────────────────

class TESCaseWriter:
    """Accumulates trajectory snapshots and writes one TES HDF5 case file (v2).

    Usage::
        writer = TESCaseWriter(h5_path, case_meta, nx, ny, zones_cfg,
                               mask_data=mask, g_vec=gvec)
        writer.append_snapshot(step, ctu, moments_nxnyc)
        ...
        writer.finalize(converged=True, stop_reason="converged")
    """

    DOWNSAMPLE_FACTOR: int = 3   # ROIs(1920×1080) → H5s(640×360)

    def __init__(
        self,
        h5_path: str,
        case_meta: dict,
        nx: int,
        ny: int,
        zones_cfg: dict,
        mask_data: Optional[np.ndarray] = None,
        g_vec: Optional[np.ndarray] = None,
        spinup_ctus: float = 0.0,
    ) -> None:
        """
        Args:
            h5_path: Output HDF5 file path.
            case_meta: Dict of attrs to write at root (case_id, re, ic_type, …).
            nx: Solver domain width (lattice units).
            ny: Solver domain height (lattice units).
            zones_cfg: domain_zones config dict (sponge_*, buffer_in/out/top/bot).
            mask_data: Binary mask array (nx, ny) — solid = 1, free fluid = 0.
            g_vec: 14D geometry proxy vector (float32).
            spinup_ctus: CTU count to skip before recording snapshots (spinup phase, not stored).
        """
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        self.h5_path = h5_path
        self.case_meta = case_meta
        self.nx = nx
        self.ny = ny
        self.is_closed = False
        self.spinup_ctus = float(spinup_ctus)

        # ── Crop region: remove Sponge + Buffer on all 4 sides ──────────────
        # This gives ROI = Mask_Size = 1920×1080 (ROIs = Ms, no buffer included)
        sponge_in  = int(zones_cfg.get("sponge_in",  0))
        sponge_out = int(zones_cfg.get("sponge_out", 0))
        sponge_top = int(zones_cfg.get("sponge_top", 0))
        sponge_bot = int(zones_cfg.get("sponge_bot", 0))
        # Per-side buffer widths (buffer_* keys preferred; fallback to scalar 'buffer')
        scalar_buf = int(zones_cfg.get("buffer", 0))
        buf_in  = int(zones_cfg.get("buffer_in",  scalar_buf))
        buf_out = int(zones_cfg.get("buffer_out", scalar_buf))
        buf_top = int(zones_cfg.get("buffer_top", scalar_buf))
        buf_bot = int(zones_cfg.get("buffer_bot", scalar_buf))

        # Solver x-axis = horizontal (nx), y-axis = vertical (ny)
        self.slice_x = slice(sponge_in + buf_in, nx - sponge_out - buf_out)
        self.slice_y = slice(sponge_bot + buf_bot, ny - sponge_top - buf_top)
        self.crop_w  = (nx - sponge_out - buf_out) - (sponge_in + buf_in)
        self.crop_h  = (ny - sponge_top - buf_top) - (sponge_bot + buf_bot)

        if self.crop_w <= 0 or self.crop_h <= 0:
            raise ValueError(
                f"[TESWriter] Invalid crop area: {self.crop_w}×{self.crop_h}. "
                "Check domain_zones config (sponge_* + buffer_*)."
            )

        # Downsampled spatial dimensions (3×): H5s = (640, 360)
        self.H_ds = self.crop_h // self.DOWNSAMPLE_FACTOR
        self.W_ds = self.crop_w // self.DOWNSAMPLE_FACTOR

        # ── Snapshot buffers (accumulated in memory) ──
        self._steps:    list[int]         = []
        self._ctus:     list[float]       = []
        self._moments:  list[np.ndarray]  = []   # each [9, H_ds, W_ds] fp16
        self._ke_list:  list[float]       = []
        self._mass_res: list[float]       = []
        self._mass_res_rel: list[float]   = []   # 無因次：|Δ通量| / |入口通量|

        # ── 求解迴圈內時均累加器的輸出（DEEP_AUDIT_2026-07 Stage 1.15）──
        # None = 未啟用（既有行為，H5 不含 /mean_accum 群組）。
        self._mean_accum_payload: Optional[dict] = None

        # ── Open HDF5 and write static data ──
        self.f = h5py.File(h5_path, "w", libver="latest")
        self._write_geometry(mask_data)
        if g_vec is not None:
            self.f.create_dataset("g_vec", data=np.asarray(g_vec, dtype=np.float32))

        print(
            f"[TESWriter] Init: crop={self.crop_w}×{self.crop_h} "
            f"(ROI=Ms) → ds={self.W_ds}×{self.H_ds} (H5s)  "
            f"h5={os.path.basename(h5_path)}"
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _crop_downsample_fp16(self, moments_nxnyc: np.ndarray) -> np.ndarray:
        """(nx, ny, 9) float32 → (9, H_ds, W_ds) float16.

        Crop removes Sponge + Buffer on all 4 sides → ROI = Mask_Size.
        3× INTER_AREA downsample → H5_Size (640×360).
        """
        cropped = moments_nxnyc[self.slice_x, self.slice_y, :]     # (crop_w, crop_h, 9)
        img_hwc = cropped.transpose(1, 0, 2).astype(np.float32)    # (crop_h, crop_w, 9)

        channels = []
        for c in range(9):
            ch = cv2.resize(
                img_hwc[:, :, c],
                (self.W_ds, self.H_ds),
                interpolation=cv2.INTER_AREA,
            )
            channels.append(ch)

        return np.stack(channels, axis=0).astype(np.float16)        # (9, H_ds, W_ds)

    def _write_geometry(self, mask_data: Optional[np.ndarray]) -> None:
        """Write /geometry group with mask only (v2: no derived fields)."""
        geom = self.f.create_group("geometry")
        geom.attrs["shape_h5"]            = [self.H_ds, self.W_ds]          # H5s
        geom.attrs["shape_roi"]           = [self.crop_h, self.crop_w]      # ROIs = Ms
        geom.attrs["shape_solver"]        = [self.ny, self.nx]              # ny × nx
        geom.attrs["downsample_factor"]   = self.DOWNSAMPLE_FACTOR

        if mask_data is None:
            return

        # mask: (nx, ny) → crop → (crop_h, crop_w) binary → 3× downsample
        mask_crop = mask_data[self.slice_x, self.slice_y]           # (crop_w, crop_h)
        mask_hw   = mask_crop.transpose(1, 0).astype(np.float32)    # (crop_h, crop_w)
        mask_ds   = cv2.resize(mask_hw, (self.W_ds, self.H_ds), interpolation=cv2.INTER_NEAREST)
        mask_bin  = (mask_ds > 0.5)

        geom.create_dataset(
            "mask", data=mask_bin, compression="lzf"
        )
        # sdf / upstream_norm / x_norm / y_norm are NOT stored (v2).
        # Computed on-the-fly in DataLoader from mask; per-geometry cached in RAM.

    # ── Public API ─────────────────────────────────────────────────────────────

    def append_snapshot(
        self,
        step: int,
        ctu: float,
        moments_numpy: np.ndarray,
    ) -> None:
        """Accumulate one snapshot.

        Args:
            step: Current LBM step number.
            ctu: Current time in CTU.
            moments_numpy: Solver moments array (nx, ny, 9) float32.
        """
        if self.is_closed:
            return
        if self.spinup_ctus > 0.0 and ctu < self.spinup_ctus:
            return

        snap_fp16 = self._crop_downsample_fp16(moments_numpy)
        self._steps.append(step)
        self._ctus.append(float(ctu))
        self._moments.append(snap_fp16)

        # Kinetic energy proxy: mean(|j|²/ρ) in crop region
        # MRT D2Q9 moment ordering: [ρ, e, ε, jx, qx, jy, qy, pxx, pxy]
        rho = moments_numpy[self.slice_x, self.slice_y, 0].astype(np.float64)
        jx  = moments_numpy[self.slice_x, self.slice_y, 3].astype(np.float64)
        jy  = moments_numpy[self.slice_x, self.slice_y, 5].astype(np.float64)
        ke  = float(np.mean((jx**2 + jy**2) / (rho + 1e-10)))
        self._ke_list.append(ke)

        # Mass residual: |mean(ρux) at inlet face − mean(ρux) at outlet face|
        # ★同時存無因次版（2026-07-29）：絕對值沒有分母，讀者無從判斷「大不大」，
        #   且它隨 u_inlet 與 ROI 尺寸縮放，跨 case 不可比。分母取入口面通量本身。
        j_in = float(np.mean(jx[0, :]))
        mass_res = float(abs(j_in - float(np.mean(jx[-1, :]))))
        self._mass_res.append(mass_res)
        self._mass_res_rel.append(
            float(mass_res / abs(j_in)) if abs(j_in) > 1e-30 else float("nan")
        )

    def set_mean_accum_payload(self, payload: Optional[dict]) -> None:
        """設定求解迴圈內時均累加器的輸出，供 finalize 寫入 ``/mean_accum``。

        Args:
            payload: :meth:`core.time_mean_accumulator.TimeMeanAccumulator.to_h5_payload`
                的回傳值；None 表未啟用（H5 不寫該群組）。
        """
        self._mean_accum_payload = payload

    def _write_mean_accum(self) -> None:
        """寫 ``/mean_accum`` 群組（分段時均，DEEP_AUDIT_2026-07 Stage 1.15）。

        欄位（S = 段數）：
            segment_mean  [S, 9, H_ds, W_ds] float32  各段 raw moment 時均
            segment_n     [S] int64                    各段累加樣本數
            segment_step0 / segment_step1 [S] int64    段的步數區間（左開右閉）
            segment_ctu0  / segment_ctu1  [S] float64  段的 CTU 區間
        任意窗時均 = sum(n_i * mean_i) / sum(n_i)。
        """
        p = self._mean_accum_payload
        if p is None:
            return
        grp = self.f.create_group("mean_accum")
        seg = p["segment_mean"]
        grp.create_dataset(
            "segment_mean",
            data=seg,
            chunks=(1, seg.shape[1], seg.shape[2], seg.shape[3]),
            compression="lzf",
            shuffle=True,
        )
        for key in ("segment_n", "segment_step0", "segment_step1",
                    "segment_ctu0", "segment_ctu1"):
            grp.create_dataset(key, data=p[key])
        for k, v in p["attrs"].items():
            grp.attrs[k] = v
        print(
            f"[TESWriter] /mean_accum: {seg.shape[0]} 段 × {seg.shape[1:]} "
            f"float32 ({seg.nbytes / 1024**2:.1f} MiB 未壓縮) | "
            f"總步數={p['attrs']['n_steps_total']}"
        )

    def finalize(
        self,
        converged: bool = True,
        stop_reason: str = "converged",
        ke_convergence_step: int = -1,
        ke_variance_ratio: float = -1.0,
        convergence_ctu: float = -1.0,
        sim_start_utc: str = "",
        sim_end_utc: str = "",
        wall_time_total_s: float = -1.0,
        wall_time_to_conv_s: float = -1.0,
        sim_conv_utc: str = "",
        gt_quality: dict | None = None,
    ) -> None:
        """Write all accumulated data to HDF5 and close file.

        Args:
            converged: Overall success flag.
            stop_reason: One of {converged_oscillatory, hard_cutoff,
                stability_failure, runtime_error, aborted, converged}.
            ke_convergence_step: LBM step at first convergence (-1 if none).
            ke_variance_ratio: Final 2-window mean-difference ratio.
            convergence_ctu: CTU at first convergence (-1.0 if never converged).
                Training pipeline should use snapshots with ctu ≤ convergence_ctu;
                snapshots beyond are buffered for future analysis.
            gt_quality: ★2026-07-26 新增（DEEP_AUDIT_2026-07 修正 5+6）。
                GT 品質診斷欄位，逐一寫入 root attrs。供下游篩選 case、
                loss weighting，以及論文引用 GT 品質量化指標。
                典型鍵：mean_window_truncated（時均窗是否被 total_ctu 截斷）、
                postconv_ctu（實際 post-conv 窗長）、conv_k_required（持續性
                要求次數）、conv_final_trend_drift（累積漂移閘門最終值）等。
        """
        if self.is_closed:
            return

        n_snaps = len(self._steps)
        if n_snaps == 0:
            print(f"[TESWriter] Warning: no snapshots — closing empty file {self.h5_path}")
            self.f.close()
            self.is_closed = True
            return

        # ── Root attrs ──
        for k, v in self.case_meta.items():
            try:
                self.f.attrs[k] = v
            except Exception:
                self.f.attrs[k] = str(v)
        self.f.attrs["n_snapshots"] = n_snaps
        self.f.attrs["converged"]   = converged
        self.f.attrs["created_at"]  = datetime.now(timezone.utc).isoformat()
        self.f.attrs["spinup_ctus"]         = float(self.spinup_ctus)
        self.f.attrs["ke_convergence_step"] = int(ke_convergence_step)
        self.f.attrs["convergence_ctu"]     = float(convergence_ctu)
        self.f.attrs["stop_reason"]         = str(stop_reason)
        if ke_variance_ratio < 0.0 and len(self._ke_list) >= 5:
            arr = np.array(self._ke_list[-5:], dtype=np.float64)
            _mean = float(np.mean(arr))
            _std  = float(np.std(arr))
            ke_variance_ratio = float(_std / max(abs(_mean), 1e-10))
        self.f.attrs["ke_variance_ratio"] = float(max(0.0, ke_variance_ratio))
        self.f.attrs["sim_start_utc"]       = sim_start_utc
        self.f.attrs["sim_end_utc"]         = sim_end_utc
        self.f.attrs["wall_time_total_s"]   = float(wall_time_total_s)
        self.f.attrs["wall_time_to_conv_s"] = float(wall_time_to_conv_s)
        self.f.attrs["sim_conv_utc"]        = sim_conv_utc

        # ── GT 品質診斷（DEEP_AUDIT_2026-07 修正 5+6）──────────────────
        if gt_quality:
            for k, v in gt_quality.items():
                try:
                    self.f.attrs[k] = v
                except Exception:
                    self.f.attrs[k] = str(v)

        # ── Trajectory group ──
        traj = self.f.create_group("trajectory")
        moments_stack = np.stack(self._moments, axis=0)              # [N, 9, H_ds, W_ds]
        traj.create_dataset(
            "moments",
            data=moments_stack,
            chunks=(1, 9, self.H_ds, self.W_ds),
            compression="lzf",
            shuffle=True,
        )
        traj.create_dataset("lbm_steps", data=np.array(self._steps, dtype=np.int32))
        traj.create_dataset("ctu",       data=np.array(self._ctus,  dtype=np.float32))
        traj.attrs["compression"]    = "lzf+shuffle+fp16+downsample3x"
        traj.attrs["schedule_type"]  = "staircase_ctu"

        # ── Convergence info ──
        ci = self.f.create_group("convergence_info")
        ci.create_dataset(
            "kinetic_energy", data=np.array(self._ke_list, dtype=np.float32)
        )
        ci.create_dataset(
            "mass_residual",  data=np.array(self._mass_res, dtype=np.float32)
        )
        ci.create_dataset(
            "mass_residual_rel",
            data=np.array(self._mass_res_rel, dtype=np.float32)
        )
        ci.attrs["mass_residual_def"] = (
            "|mean(rho*ux)[inlet] - mean(rho*ux)[outlet]|；"
            "_rel 為其除以 |mean(rho*ux)[inlet]| 的無因次版（跨 case 可比）"
        )
        if len(self._ke_list) >= 2:
            dke = abs(self._ke_list[-1] - self._ke_list[-2])
            rel = dke / (abs(self._ke_list[-2]) + 1e-10)
        else:
            rel = 0.0
        ci.attrs["final_ke_change_pct"] = float(rel * 100)
        ci.attrs["stop_reason"]         = stop_reason

        # ── 求解迴圈內時均（Stage 1.15）──
        self._write_mean_accum()

        self.f.close()
        self.is_closed = True
        print(
            f"[TESWriter] Finalized: {n_snaps} snapshots | "
            f"converged={converged} | {self.h5_path}"
        )

    def close(self) -> None:
        """Alias for finalize()."""
        self.finalize()
