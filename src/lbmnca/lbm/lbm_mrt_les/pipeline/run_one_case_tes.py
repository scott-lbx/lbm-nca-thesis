"""TES trajectory case runner — staircase CTU snapshot schedule (v2).

每 case 產生 per dataset_structure_tes.md §5 (v2) 規格的 trajectory HDF5：
  - Staircase CTU 排程（Phase 1-4，密→疏，防止早期假收斂）
  - H5s = 640×360（3× downsample from ROIs = 1920×1080）
  - geometry 只存 mask（無 sdf/upstream_norm 等衍生場）
"""

from __future__ import annotations

import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import taichi as ti
from tqdm import tqdm

import cv2
import h5py

from .. import utils
from ..core.convergence_monitor import ConvergenceMonitor, SustainedConvergenceTracker
from ..core.LBM2D_MRT_LES import LBM2D_MRT_LES
from ..core.runtime import ensure_taichi_runtime, reset_taichi_runtime
from ..core.simulation_ops import check_stability
from ..core.time_mean_accumulator import TimeMeanAccumulator, acoustic_period_steps
from ..io.tes_writer import (
    TESCaseWriter,
    build_snapshot_schedule_dense,
    build_snapshot_schedule_staircase,
    total_steps_from_ctus,
)

_U_INLET_LU: float = 0.05    # fixed for all TES cases (Mach < 0.1)

#: warm IC 判定「父場此格點是否為有效流體」的 ρ 下界。父場的固體格點在 H5 裡
#: 是 ρ=0，流體格點在弱可壓縮 LBM 下貼近 1（實測 D 系列 0.99–1.03）。
#: 0.5 落在兩者之間、留有寬裕餘裕，不會誤判任何真實流體格點。
_WARM_IC_RHO_FLUID_MIN: float = 0.5
# TES v_threshold (CEDS used 0.25): local peak velocities around dense obstacle
# clusters can transiently reach ~5× u_inlet = 0.25, but tau >> 0.508 keeps the
# simulation numerically stable. 0.40 = hard kill if velocity truly diverges.
# 註：warmup 緩衝由 config 的 warmup_steps 控制（目前皆 0 → 速度熔斷自 step 1 生效）；
# 先前的 _STABILITY_WARMUP=500 常數從未被引用，已於 2026-06-29 移除（audit C7 死碼）。
_TES_V_THRESHOLD: float = 0.40
_DOWNSAMPLE_FACTOR: int = 3  # same as TESCaseWriter.DOWNSAMPLE_FACTOR (v2: 3×)


def _load_warm_start_ic(
    solver: LBM2D_MRT_LES,
    tes_cfg: Dict[str, Any],
    zones: Dict[str, Any],
) -> None:
    """Load G_A's final cold-start snapshot and initialize solver state.

    Reads parent H5 moments[-1] (fp16, 4× downsampled, cropped), upsamples
    back to full crop resolution, and fills the solver's rho/vel fields before
    calling solver.init_warm_start().

    Args:
        solver: Initialized LBM solver (geometry already loaded).
        tes_cfg: 'tes' section from case config (must have parent_h5_path).
        zones: 'domain_zones' section from case config.
    """
    parent_h5_path = tes_cfg.get("parent_h5_path", "")
    if not parent_h5_path or not os.path.exists(parent_h5_path):
        raise FileNotFoundError(
            f"[TES-warm] Parent H5 not found: '{parent_h5_path}'. "
            "Check parent_h5_path in tes config section."
        )

    nx, ny = solver.nx, solver.ny

    # Compute crop region (must match TESCaseWriter v2 logic exactly:
    # remove Sponge + Buffer on all 4 sides → ROI = Mask_Size = 1920×1080)
    sponge_in  = int(zones.get("sponge_in",  0))
    sponge_out = int(zones.get("sponge_out", 0))
    sponge_top = int(zones.get("sponge_top", 0))
    sponge_bot = int(zones.get("sponge_bot", 0))
    scalar_buf = int(zones.get("buffer",     0))
    buf_in     = int(zones.get("buffer_in",  scalar_buf))
    buf_out    = int(zones.get("buffer_out", scalar_buf))
    buf_top    = int(zones.get("buffer_top", scalar_buf))
    buf_bot    = int(zones.get("buffer_bot", scalar_buf))

    x_start = sponge_in  + buf_in
    x_end   = nx - sponge_out - buf_out
    y_start = sponge_bot + buf_bot
    y_end   = ny - sponge_top - buf_top
    crop_w  = x_end - x_start
    crop_h  = y_end - y_start

    # Load parent H5 final snapshot: [9, H_ds, W_ds] fp16
    with h5py.File(parent_h5_path, "r") as h5:
        moments_last = h5["trajectory/moments"][-1].astype(np.float32)  # [9, H_ds, W_ds]

    H_ds, W_ds = moments_last.shape[1], moments_last.shape[2]
    target_h   = crop_h
    target_w   = crop_w

    # Upsample each channel from [H_ds, W_ds] to [crop_h, crop_w]
    upsamp = np.empty((9, target_h, target_w), dtype=np.float32)
    for c in range(9):
        upsamp[c] = cv2.resize(
            moments_last[c], (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

    # Extract ρ, jx, jy (MRT moment order: 0=ρ, 3=jx, 5=jy)
    rho_crop = upsamp[0]     # [crop_h, crop_w]
    jx_crop  = upsamp[3]     # [crop_h, crop_w]
    jy_crop  = upsamp[5]     # [crop_h, crop_w]

    # ★父場的固體格點在 H5 裡是 ρ=0。warm 鏈連接的是**不同幾何**，父場的固體
    #   格點完全可能落在子案例的流體區——那裡沒有任何可用的父場資訊。
    #
    #   原本的 `rho_safe = max(rho, 1e-6)` 只防住除零，卻把插值殘留的微小 jx
    #   放大成天文數字的速度（jx=1e-9 → u=1e-3 尚可，jx=1e-4 → u=100）。
    #   下方的 solid 遮罩只清「子案例的固體」，清不到這些格點，於是第一次力
    #   計算就 NaN。實測 D 系列 warm pilot 4/4 全在 step 200 以
    #   `Force becomes NaN/Inf` 收場。
    #
    #   C 系列很少踩到是因為它的 warm 鏈是 1:1 擾動樹、父子幾何幾乎相同；
    #   D 系列的鏈由 MST 連接 grid-incremental 取樣，父子佈局差異大。
    #
    #   正解：判定該格點在**父場**是否為有效流體，不是的就回退到背景參考態，
    #   讓子案例自己把那塊區域跑出來（burn-in 本來就是為此存在）。
    parent_fluid = rho_crop > _WARM_IC_RHO_FLUID_MIN
    rho_safe = np.where(parent_fluid, rho_crop, 1.0)
    ux_crop  = np.where(parent_fluid, jx_crop / rho_safe, _U_INLET_LU)
    uy_crop  = np.where(parent_fluid, jy_crop / rho_safe, 0.0)
    rho_crop = rho_safe
    n_invalid = int((~parent_fluid).sum())

    # Build full solver domain arrays — default to reference state
    rho_full = np.ones((nx, ny),    dtype=np.float32)
    vel_full = np.zeros((nx, ny, 2), dtype=np.float32)
    vel_full[:, :, 0] = _U_INLET_LU   # uniform inlet flow as background

    # Fill crop region (stored as [crop_h, crop_w] → transpose to [crop_w, crop_h])
    # Solver indexing: (i=x, j=y) → rho_full[x_start:x_end, y_start:y_end]
    # H corresponds to y-axis (ny), W corresponds to x-axis (nx)
    rho_full[x_start:x_end, y_start:y_end] = rho_crop.T
    vel_full[x_start:x_end, y_start:y_end, 0] = ux_crop.T
    vel_full[x_start:x_end, y_start:y_end, 1] = uy_crop.T

    # 新幾何的固體節點歸零靜止（audit G6）：固體節點不參與碰撞/串流（collide_and_stream
    # 凍結其 f），若沿用父場流體動量會永久殘留於新 solid 內 → 污染 mean_flow、違反 no-slip。
    # solver.mask 已於幾何載入時設好（1.0=固體）。
    solid = solver.mask.to_numpy() > 0.5     # [nx, ny]
    rho_full[solid] = 1.0
    vel_full[solid] = 0.0

    solver.init_warm_start(rho_full, vel_full)
    print(
        f"[TES-warm] IC loaded from {os.path.basename(parent_h5_path)} "
        f"crop=({crop_w}×{crop_h}) upsamp=({target_w}×{target_h}) "
        f"父場無效格點={n_invalid}"
        f"（{n_invalid / (target_h * target_w):.2%}，已回退為背景參考態）"
    )


def _build_case_meta(config: Dict[str, Any], l_char_px: int) -> dict:
    """Assemble HDF5 root attrs from case config."""
    tes  = config["tes"]
    sim  = config["simulation"]
    nu   = float(sim["nu"])
    tau  = 3.0 * nu + 0.5

    return {
        "case_id":        tes.get("case_id",        sim.get("name", "unknown")),
        "geometry_id":    tes.get("geometry_id",    "unknown"),
        "re":             float(tes.get("re_target", _U_INLET_LU * l_char_px / nu)),
        "u_inlet_lu":     _U_INLET_LU,
        "l_char_px":      l_char_px,
        "nu_lu":          nu,
        "tau":            tau,
        "ctu_lbm_steps":  int(round(l_char_px / _U_INLET_LU)),
        "ic_type":        tes.get("ic_type", "cold"),
        "parent_case_id": tes.get("parent_case_id", ""),
        "parent_snapshot": int(tes.get("parent_snapshot", -1)),
        "total_ctu":      float(tes.get("total_ctus", 30.0)),
        "total_lbm_steps": total_steps_from_ctus(l_char_px, float(tes.get("total_ctus", 30.0))),
        # optional field; "dense_0.2" for warm-start datasets (plan §3.4)
        "schedule_type":    str(tes.get("snapshot_schedule", "staircase_ctu")),
        "bcr":            float(tes.get("bcr", 0.0)),
        "n_obstacles":    int(tes.get("n_obstacles", 0)),
        "spinup_ctus":       float(tes.get("spinup_ctus", 0.0)),
        # ★時均窗的三段結構（2026-07-29 L3）。舊 config 沒有這些鍵 → 寫 -1.0
        #   當哨兵，轉檔層據此走舊的固定比例路徑（版本分支，不是 fallback）。
        "warmup_ctus":       float(tes["warmup_ctus"])
                             if "warmup_ctus" in tes else -1.0,
        "burnin_ctus":       float(tes["burnin_ctus"])
                             if "burnin_ctus" in tes else -1.0,
        "mean_window_ctus":  float(tes["mean_window_ctus"])
                             if "mean_window_ctus" in tes else -1.0,
        "warm_k_level":      int(tes.get("warm_k_level", 0)),
        "perturb_type":      str(tes.get("perturb_type", "")),
        "perturb_magnitude": float(tes.get("perturb_magnitude", 0.0)),
        "parent_dataset":    str(tes.get("parent_dataset", "")),
        "dataset_name":      str(tes.get("dataset_name", "")),
        # ── C2-2 WarmGraph edge metadata（graph 資料集才有；舊 dataset 為空值）──
        "edge_id":            str(tes.get("edge_id", "")),
        "edge_type":          str(tes.get("edge_type", "")),
        "edit_axis":          str(tes.get("edit_axis", "")),
        "window_type":        str(tes.get("window_type", "")),
        "canonical_case_id":  str(tes.get("canonical_case_id", "")),
    }


def _run_trajectory_loop(
    solver: LBM2D_MRT_LES,
    writer: TESCaseWriter,
    snapshot_steps: list[int],
    total_steps: int,
    compute_batch: int,
    warmup_steps: int,
    steps_per_ctu: float,
    v_threshold: float = _TES_V_THRESHOLD,
    enable_conv_check: bool = True,
    conv_window_ctu: float = 5.0,
    conv_eps: float = 0.05,
    extra_ctu_after_conv: float = 10.0,
    min_convergence_recorded_ctu: float = 15.0,
    post_conv_spacing_ctu: float = 0.0,
    t_start: float = 0.0,
    conv_k_required: int = 3,
    conv_cum_eps: float = 0.05,
    conv_trend_gate: bool = True,
    mean_accum: TimeMeanAccumulator | None = None,
) -> Dict[str, Any]:
    """Main TES simulation loop with sustained two-window KE convergence detection.

    ★★ 收斂偵測的角色（2026-07-28，DEEP_AUDIT_2026-07 Stage 1.16 降級）
    ------------------------------------------------------------------
    這裡的收斂偵測**只做兩件事**：(a) 決定何時可以提早停止（省 GPU），
    (b) 給下游一個「大約什麼時候之後算 post-transient」的**粗略提示**。
    它**不再是 GT 時均正確性的關鍵路徑**——時均由 ``mean_accum`` 分段累加器
    產生，窗起點由轉檔器的協定常數決定（見
    ``post_process/tes_h5_to_zarr._resolve_mean_start_ctu``），與本處的閂鎖時刻無關。

    為什麼要降級（1.16 的實測理由）：two-window 判準的兩個半窗合計
    ``2 × conv_window_ctu = 10`` CTU，**短於它所監測訊號的主導模態週期**
    （高解析 cylinder 實測 ~11.1 CTU、調變深度 113%，1.11 步驟 4）
    → 原理上測不出穩態，只測得到相位差。母體 74.6% 的 ``convergence_ctu``
    卡在 floor ``2×window+5=15``，也與此一致。

    Convergence policy（★2026-07-26 修正，DEEP_AUDIT_2026-07）:
    two-window mean comparison 需**連續 conv_k_required 次成立**，且通過
    trend-line 累積漂移閘門，才宣告收斂；收斂時刻取連續成立的**起點**。

    舊行為為「首次命中即永久閂鎖」，TIER0 實測顯示該做法量到的是
    「首次隨機低於門檻的時刻」而非任何持續狀態：母體 74.6% 卡在 floor，
    宣告收斂後 89% 的 case 重新套用同一判準會 FAIL，且 KE 仍中位再漲 +82%。
    詳見 docs/reviews/DEEP_AUDIT_2026-07/fixes/TIER0_結果與機制定案.md

    On detection, continue ``extra_ctu_after_conv`` CTU more then stop with
    ``stop_reason='converged_oscillatory'``. If never converges, run until
    total_steps then stop with ``stop_reason='hard_cutoff'``.

    Args:
        solver: Initialized LBM solver.
        writer: TESCaseWriter that will receive snapshots.
        snapshot_steps: Sorted list of LBM step integers (staircase schedule).
        total_steps: Hard-cutoff step count.
        compute_batch: GPU steps per call.
        warmup_steps: Steps before velocity threshold applies.
        steps_per_ctu: L_char / U_inlet, step → CTU conversion.
        v_threshold: Velocity stability threshold.
        enable_conv_check: If False, run full schedule (warm cases).
        conv_window_ctu: Half-window for the 2-window mean test, in CTU.
            ⛔ **不要再把它讀成「約一個脫落週期」** —— 那句原註解是假陳述，
            已於 2026-07-28（Stage 1.16）撤除。理由：本判準監測的是**全域積分
            KE**，而 1.13 實測全域 KE 中幾乎不含脫落訊號（Re=100 cylinder 上
            真實脫落諧波比主峰弱 **4744 倍**），所以「一個脫落週期」根本不是
            這條訊號的特徵尺度。KE 上實際存在的主導模態週期約 **11.1 CTU**
            （1.11 步驟 4；1.11 步驟 5 再證 ny 加倍後該模態完全不變
            → 不是側向駐波），而本判準兩個半窗合計只有 10 CTU。
            ★ 這個值**目前沒有可靠依據可以重訂**（唯一能訂它的量是主導模態
            週期，而 1.14 已撤除週期偵測，生產端也無真值可驗收任何偵測器）。
            處置方式是**讓它不影響正確性**（見本函式頂部的降級說明），
            而不是猜一個新數字。
        conv_eps: |Δmean|/mean threshold (~0.05).
        extra_ctu_after_conv: Continue this many CTU after first convergence.
        min_convergence_recorded_ctu: Earliest CTU at which check is enabled
            (must be >= 2 × conv_window_ctu to allow both windows of data).
            ⚠ 生產母體 74.6% 的 convergence_ctu 正好卡在此 floor
            （2×5+5 = 15.0）→ 那個數字反映的是 floor，不是物理時刻。
        mean_accum: 求解迴圈內時均累加器（Stage 1.15），None = 未啟用。
            已由 solver 每步呼叫 accumulate()；此處只負責跨過段邊界時封段。

    Returns:
        metadata dict with status, reason, stop_reason, convergence_ctu, etc.
    """
    snap_idx        = 0
    convergence_ctu = -1.0
    last_ratio      = -1.0
    t_conv          = -1.0

    # ★ 持續性收斂追蹤器（取代舊的單次閂鎖）
    conv_tracker = SustainedConvergenceTracker(
        window_ctu=conv_window_ctu,
        eps=conv_eps,
        k_required=conv_k_required,
        cum_eps=conv_cum_eps,
        enable_trend_gate=conv_trend_gate,
    )

    current_step = 0
    exit_status  = "Success"
    exit_reason  = "Reached total_steps"
    stop_reason  = "hard_cutoff"

    pbar = tqdm(total=total_steps, unit="step", mininterval=1.0, dynamic_ncols=True)

    try:
        while current_step < total_steps:
            if snap_idx < len(snapshot_steps):
                next_target = min(snapshot_steps[snap_idx], total_steps)
            else:
                next_target = total_steps

            while current_step < next_target:
                batch = min(compute_batch, next_target - current_step)
                solver.run_step(batch)
                current_step += batch
                pbar.update(batch)

                # ── 時均分段（Stage 1.15）：跨過段邊界就封存一段 ──
                # 封段點解析度 = compute_batch 步（段的實際步數逐段記錄，
                # 不假設等長），對段長（數千步）而言可忽略。
                if mean_accum is not None:
                    mean_accum.maybe_close_segment()

                forces = solver.get_force()
                max_v  = solver.get_max_velocity()
                ok, reason = check_stability(
                    forces, max_v, current_step, warmup_step=warmup_steps,
                    v_threshold=v_threshold,
                )
                if not ok:
                    exit_status = "Failed"
                    exit_reason = reason
                    stop_reason = "stability_failure"
                    print(f"\n[TES] Stability failure: {reason}")
                    return {
                        "status":              exit_status,
                        "reason":              exit_reason,
                        "stop_reason":         stop_reason,
                        "final_steps":         current_step,
                        "convergence_ctu":     float(convergence_ctu),
                        "ke_convergence_step": -1,
                        "ke_variance_ratio":   float(last_ratio),
                        "wall_time_to_conv_s": -1.0,
                        "sim_conv_utc":        "",
                    }

            if snap_idx < len(snapshot_steps) and current_step >= snapshot_steps[snap_idx]:
                ctu_val = float(snapshot_steps[snap_idx]) / steps_per_ctu
                snap_idx += 1

                # ── Two-phase recording: dense pre-conv, coarse post-conv ──
                # Pre-conv → record every dense frame (0.2 CTU).
                # Post-conv → record only on the absolute post_conv_spacing grid
                # (e.g. 1.0 CTU), matching thinned legacy data. Long post-conv
                # window (extra_ctu_after_conv) → many periods for clean mean.
                if convergence_ctu < 0.0 or post_conv_spacing_ctu <= 0.0:
                    do_record = True
                else:
                    _g = ctu_val / post_conv_spacing_ctu
                    do_record = abs(_g - round(_g)) < 0.1

                if do_record:
                    moments = solver.get_moments_numpy()
                    writer.append_snapshot(current_step, ctu_val, moments)

                    recorded_ctu = ctu_val - writer.spinup_ctus

                    # ── Convergence detection (sustained two-window + trend gate) ──
                    # ★ 不再用 `convergence_ctu < 0.0` 當 guard 做單次閂鎖；
                    # tracker 內部要求連續 k 次成立且通過累積漂移閘門。
                    if (
                        enable_conv_check
                        and convergence_ctu < 0.0
                        and recorded_ctu >= min_convergence_recorded_ctu
                    ):
                        conv = conv_tracker.update(
                            writer._ke_list, writer._ctus, ctu_val
                        )
                        last_ratio = conv_tracker.last_ratio
                        if conv:
                            # 收斂時刻 = 連續成立的起點（統計穩態實際開始處）
                            convergence_ctu = conv_tracker.converged_ctu
                            t_conv = time.time()
                            print(
                                f"\n[TES] KE converged @ {convergence_ctu:.2f} CTU "
                                f"(sustained {conv_tracker.k_required}× "
                                f"ratio<{conv_eps}, last={last_ratio:.3e}, "
                                f"trend_drift={conv_tracker.last_drift:.3e}, "
                                f"trend-rejected {conv_tracker.n_rejected_by_trend}×), "
                                f"confirmed @ {ctu_val:.2f} CTU, "
                                f"continuing {extra_ctu_after_conv:.1f} CTU "
                                f"@ post-conv spacing {post_conv_spacing_ctu:.1f} CTU"
                            )

                # ── Post-convergence stop check ──
                if (
                    convergence_ctu > 0.0
                    and (ctu_val - convergence_ctu) >= extra_ctu_after_conv
                ):
                    stop_reason = "converged_oscillatory"
                    exit_reason = (
                        f"converged @ {convergence_ctu:.2f} CTU + "
                        f"{extra_ctu_after_conv:.1f} CTU buffer"
                    )
                    print(f"\n[TES] Stopping: {exit_reason}")
                    pbar.close()
                    return {
                        "status":              "Success",
                        "reason":              exit_reason,
                        "stop_reason":         stop_reason,
                        "final_steps":         current_step,
                        "convergence_ctu":     float(convergence_ctu),
                        "ke_convergence_step": int(round(convergence_ctu * steps_per_ctu)),
                        "ke_variance_ratio":   float(last_ratio),
                        "wall_time_to_conv_s": float(t_conv - t_start) if t_conv > 0 else -1.0,
                        "sim_conv_utc":        datetime.fromtimestamp(t_conv, tz=timezone.utc).isoformat() if t_conv > 0 else "",
                    }

    except KeyboardInterrupt:
        exit_status = "Aborted"
        exit_reason = "User Interrupted (Ctrl+C)"
        stop_reason = "aborted"
        print(f"\n[TES] {exit_reason}")

    except Exception as e:
        exit_status = "Error"
        exit_reason = f"Runtime Error: {e}"
        stop_reason = "runtime_error"
        print(f"\n[TES] {exit_reason}")
        traceback.print_exc()

    finally:
        pbar.close()

    # ── ★ 時均窗截斷偵測（修正 5）──────────────────────────────────
    # extra_ctu_after_conv 是「請求」不是「保證」：迴圈只跑到 total_steps，
    # 若 convergence_ctu + extra > total_ctu，時均窗會被**靜默**截短。
    # TIER0 實測母體約 0.9% 受影響（最糟 c0004：請求 50 CTU 只拿到 12 CTU / 7 幀）。
    # 此處顯式標記，不再靜默。
    total_ctu_cfg = float(total_steps) / steps_per_ctu if steps_per_ctu > 0 else -1.0
    final_ctu = float(current_step) / steps_per_ctu if steps_per_ctu > 0 else -1.0
    postconv_ctu = (final_ctu - convergence_ctu) if convergence_ctu >= 0 else -1.0
    truncated = bool(
        convergence_ctu >= 0
        and (convergence_ctu + extra_ctu_after_conv) > total_ctu_cfg + 1e-6
    )
    if truncated:
        print(
            f"\n[TES] ⚠ 時均窗被 total_ctu 截斷："
            f"conv@{convergence_ctu:.1f} + extra {extra_ctu_after_conv:.1f} "
            f"= {convergence_ctu + extra_ctu_after_conv:.1f} CTU "
            f"> total {total_ctu_cfg:.1f} CTU"
            f" → 實際 post-conv 窗僅 {postconv_ctu:.1f} CTU"
        )

    out = {
        "status":              exit_status,
        "reason":              exit_reason,
        "stop_reason":         stop_reason,
        "final_steps":         current_step,
        "convergence_ctu":     float(convergence_ctu),
        "ke_convergence_step": int(round(convergence_ctu * steps_per_ctu)) if convergence_ctu > 0 else -1,
        "ke_variance_ratio":   float(last_ratio),
        "wall_time_to_conv_s": float(t_conv - t_start) if t_conv > 0 else -1.0,
        "sim_conv_utc":        datetime.fromtimestamp(t_conv, tz=timezone.utc).isoformat() if t_conv > 0 else "",
        # ★ GT 品質診斷（修正 5 + 6）——供下游篩選 / loss weighting / 論文引用
        "mean_window_truncated": truncated,
        "postconv_ctu":          float(postconv_ctu),
        "extra_ctu_requested":   float(extra_ctu_after_conv),
    }
    out.update(conv_tracker.diagnostics())
    return out


def run_tes_case(
    config: Dict[str, Any],
    mask_path: str,
    h5_output_path: str,
) -> Dict[str, Any]:
    """Execute one TES cold-start trajectory case.

    Args:
        config: Parsed case YAML (must contain 'tes' section).
        mask_path: Path to geometry mask PNG.
        h5_output_path: Destination HDF5 file path.

    Returns:
        JSON-serializable metadata dict (same shape as run_case output).
    """
    metadata: Dict[str, Any] = {"status": "Error", "reason": "Unknown error"}
    solver = writer = None
    mean_accum: TimeMeanAccumulator | None = None

    try:
        sim_cfg  = config["simulation"]
        tes_cfg  = config["tes"]
        zones    = config["domain_zones"]

        ensure_taichi_runtime(arch=ti.gpu, device_memory_fraction=0.8, log_level=ti.INFO)

        # ── Mask + Solver ──
        mask = utils.create_mask(config, mask_path)
        solver = LBM2D_MRT_LES(config, mask_data=mask)

        ic_type = tes_cfg.get("ic_type", "cold")
        if ic_type == "warm":
            solver.init()                         # allocate GPU fields first
            _load_warm_start_ic(solver, tes_cfg, zones)
        else:
            solver.init()

        l_char    = int(sim_cfg["characteristic_length"])
        nu        = float(sim_cfg["nu"])
        re_design = _U_INLET_LU * l_char / nu if nu > 0 else float("inf")
        print(f"[TES] Re={re_design:.1f}  L_char={l_char}px  nu={nu:.5f}")

        # ── Staircase CTU snapshot schedule (v2) ──
        total_ctus    = float(tes_cfg.get("total_ctus", 30.0))
        spinup_ctus   = float(tes_cfg.get("spinup_ctus", 0.0))

        # ── Convergence detection params (two-window mean) ──
        # Both cold & warm: enabled with same params (2×window+5 CTU minimum).
        enable_conv_check  = bool(tes_cfg.get("enable_conv_check", True))
        conv_window_ctu    = float(tes_cfg.get("conv_window_ctu", 5.0))
        conv_eps           = float(tes_cfg.get("conv_eps", 0.05))
        extra_ctu_after_conv = float(tes_cfg["extra_ctu_after_conv"])  # bracket 存取：缺鍵即報錯（禁 fallback，舊預設 10.0 為 audit B4 判定不足值）
        post_conv_spacing_ctu = float(tes_cfg.get("post_conv_spacing_ctu", 0.0))
        _default_min_conv  = 2.0 * conv_window_ctu + 5.0
        min_convergence_ctu = float(tes_cfg.get("min_convergence_ctu", _default_min_conv))
        # ★ 持續性判準參數（2026-07-26 新增，DEEP_AUDIT_2026-07 修正 1+3）。
        # 舊 config 未帶這些鍵時，預設即為**修正後**行為——舊行為（單次閂鎖）
        # 已由 TIER0 證實會在 74.6% 的 case 誤判於 floor，不應作為預設。
        # 要重現舊行為請顯式設 conv_k_required=1 且 conv_trend_gate=false。
        conv_k_required = int(tes_cfg.get("conv_k_required", 3))
        conv_cum_eps    = float(tes_cfg.get("conv_cum_eps", 0.05))
        conv_trend_gate = bool(tes_cfg.get("conv_trend_gate", True))

        spinup_steps    = round(spinup_ctus * l_char / _U_INLET_LU) if spinup_ctus > 0 else 0
        steps_per_ctu   = float(l_char) / _U_INLET_LU
        effective_steps = total_steps_from_ctus(l_char, total_ctus)
        total_steps     = spinup_steps + effective_steps

        # Snapshot schedule selection (optional field; default = staircase for
        # backward-compat with S0-S3 datasets). "dense"/"dense_0.2" → uniform
        # spacing_ctu schedule for warm-start datasets (plan §3.4).
        schedule_type = str(tes_cfg.get("snapshot_schedule", "staircase")).lower()
        if schedule_type in ("dense", "dense_0.2", "dense0.2"):
            spacing_ctu  = float(tes_cfg.get("snapshot_spacing_ctu", 0.2))
            schedule_raw = build_snapshot_schedule_dense(
                effective_steps, steps_per_ctu, total_ctus, spacing_ctu=spacing_ctu
            )
            schedule_label = f"dense({spacing_ctu} CTU)"
        else:
            schedule_raw = build_snapshot_schedule_staircase(
                effective_steps, steps_per_ctu, total_ctus
            )
            schedule_label = "staircase"
        schedule      = [s + spinup_steps for s in schedule_raw]
        compute_batch = int(sim_cfg.get("compute_step_size", 50))

        print(
            f"[TES] {schedule_label} schedule: {len(schedule)} snapshots | "
            f"spinup={spinup_steps:,} + effective={effective_steps:,} = total={total_steps:,} steps | "
            f"spinup={spinup_ctus:.1f} CTU + record={total_ctus:.0f} CTU | "
            f"steps_per_ctu={steps_per_ctu:.0f} | conv_check={enable_conv_check} "
            f"min_conv_ctu={min_convergence_ctu:.1f} window={conv_window_ctu:.1f} "
            f"eps={conv_eps:.3f} extra={extra_ctu_after_conv:.1f}"
        )

        # ── G_vec from config ──
        g_vec_raw = tes_cfg.get("g_vec")
        g_vec = np.array(g_vec_raw, dtype=np.float32) if g_vec_raw else None

        # ── TES Writer ──
        case_meta = _build_case_meta(config, l_char)
        writer = TESCaseWriter(
            h5_path=h5_output_path,
            case_meta=case_meta,
            nx=solver.nx,
            ny=solver.ny,
            zones_cfg=zones,
            mask_data=mask,
            g_vec=g_vec,
            spinup_ctus=spinup_ctus,
        )

        # ── 求解迴圈內時均累加器（DEEP_AUDIT_2026-07 Stage 1.15）────────────
        # ★ 明確的「區塊存在性」分支（比照 H5 的 `key in f`），**不是**帶預設值
        #   的 config fallback：config 沒有 mean_accum 區塊 = 功能關閉（並印出
        #   訊息），有區塊時所有參數一律 bracket 存取、缺鍵即 KeyError。
        if "mean_accum" in tes_cfg:
            ma_cfg = tes_cfg["mean_accum"]
            if bool(ma_cfg["enabled"]):
                # ★ Stage 1.16 修 3：段長改以**步**為準，使段界可與求解域縱向
                #   聲學共振週期公度（1.17 發現該模態才是 GT 時均不確定度的主導項）。
                #   `segment_steps` 與 `segment_ctu` 二擇一，由累加器強制檢查。
                #   聲學週期由 solver 幾何與 BC 直接算出（非硬編常數），
                #   只作記錄與失配報告，**不設任何門檻**。
                t_acoustic = acoustic_period_steps(solver.nx, _U_INLET_LU)
                print(
                    f"[TES] 聲學共振週期（4*nx/c_s / (1-M^2)）= {t_acoustic:.1f} 步 "
                    f"= {t_acoustic / steps_per_ctu:.4f} CTU  (nx={solver.nx})"
                )
                if "segment_steps" in ma_cfg:
                    seg_steps = int(ma_cfg["segment_steps"])
                    seg_ctu = None
                elif "segment_ctu" in ma_cfg:
                    seg_steps = None
                    seg_ctu = float(ma_cfg["segment_ctu"])
                    print(
                        "[TES] [WARN] mean_accum 用 segment_ctu 定義段長 -> 段界無法與"
                        "聲學週期公度（該週期在 CTU 下逐 case 不同）。新資料建議改用 "
                        "segment_steps（見 time_mean_accumulator.acoustic_period_steps）"
                    )
                else:
                    raise KeyError(
                        "tes.mean_accum 必須提供 segment_steps 或 segment_ctu 其中之一"
                    )
                mean_accum = TimeMeanAccumulator(
                    solver=solver,
                    slice_x=writer.slice_x,
                    slice_y=writer.slice_y,
                    downsample=writer.DOWNSAMPLE_FACTOR,
                    h_ds=writer.H_ds,
                    w_ds=writer.W_ds,
                    steps_per_ctu=steps_per_ctu,
                    segment_ctu=seg_ctu,
                    segment_steps=seg_steps,
                    stride=int(ma_cfg["stride"]),
                    acoustic_period_steps_value=t_acoustic,
                )
                solver.attach_time_mean_accumulator(mean_accum)
            else:
                print("[TES] mean_accum.enabled=false -> 時均累加器關閉")
        else:
            print("[TES] config 無 tes.mean_accum 區塊 -> 時均累加器關閉（沿用 snapshot 事後平均）")

        # ── Main simulation loop ──
        v_thresh = float(tes_cfg.get("v_threshold", _TES_V_THRESHOLD))
        if v_thresh != _TES_V_THRESHOLD:
            print(f"[TES] v_threshold override: {v_thresh} (default={_TES_V_THRESHOLD})")
        t_sim_start = time.time()
        sim_start_utc = datetime.now(timezone.utc).isoformat()
        loop_result = _run_trajectory_loop(
            solver=solver,
            writer=writer,
            snapshot_steps=schedule,
            total_steps=total_steps,
            compute_batch=compute_batch,
            warmup_steps=int(sim_cfg.get("warmup_steps", 0)),
            steps_per_ctu=steps_per_ctu,
            v_threshold=v_thresh,
            enable_conv_check=enable_conv_check,
            conv_window_ctu=conv_window_ctu,
            conv_eps=conv_eps,
            extra_ctu_after_conv=extra_ctu_after_conv,
            min_convergence_recorded_ctu=min_convergence_ctu,
            post_conv_spacing_ctu=post_conv_spacing_ctu,
            t_start=t_sim_start,
            conv_k_required=conv_k_required,
            conv_cum_eps=conv_cum_eps,
            conv_trend_gate=conv_trend_gate,
            mean_accum=mean_accum,
        )
        t_sim_end = time.time()
        metadata.update(loop_result)
        metadata["sim_start_utc"]   = sim_start_utc
        metadata["sim_end_utc"]     = datetime.now(timezone.utc).isoformat()
        metadata["wall_time_total_s"] = float(t_sim_end - t_sim_start)

        # ── Convergence check ──
        converged    = metadata["status"] == "Success"
        stop_reason  = "converged" if converged else metadata.get("reason", "unknown")

        # ── 準穩態狀態紀錄（★ 2026-07-28，DEEP_AUDIT_2026-07 Stage 1.16）──────
        # 舊行為只 print 一行警告就丟掉：H5 照樣寫 converged=True 且
        # convergence_ctu >= 0，下游 tes_h5_to_zarr 的過濾器完全看不到這件事
        # → 「宣告收斂但收尾 KE 還在變 6.3%」的 case 靜默通過。
        #
        # 現在把狀態**寫進資料層**（H5 root attrs → zarr case attrs → v4 node）。
        # ⛔ 刻意**不**據此翻轉 converged、也**不**據此過濾任何 case：
        #    這個檢查只比較最後兩個 snapshot，統計力極弱，把它升格成門檻就會
        #    重蹈 1.13/1.11 的覆轍（「不可能失敗的判準」比沒有判準更糟）。
        #    誠實記錄狀態，讓下游自己決定要不要用，這是目前有證據支持的最強做法。
        metadata["quasi_steady_check"] = "insufficient_snapshots"
        metadata["quasi_steady_ke_change_pct"] = -1.0
        metadata["quasi_steady_source"] = "last_two_snapshots_rel_ke_change"
        metadata["quasi_steady_threshold_pct"] = 5.0
        if writer and len(writer._ke_list) >= 2:
            ke_last   = writer._ke_list[-1]
            ke_prev   = writer._ke_list[-2]
            ke_change = abs(ke_last - ke_prev) / (abs(ke_prev) + 1e-10)
            metadata["quasi_steady_ke_change_pct"] = float(ke_change * 100.0)
            metadata["quasi_steady_check"] = (
                "ke_change_over_threshold" if ke_change > 0.05 else "ke_change_within_threshold"
            )
            if ke_change > 0.05:
                print(
                    f"[TES] [WARN] 收尾 KE 變化 = {ke_change*100:.1f}% (> 5%)，"
                    f"可能未達準穩態；已寫入 quasi_steady_check="
                    f"{metadata['quasi_steady_check']}"
                    f"（狀態紀錄，不改變 converged / 不過濾此 case）"
                )

        if converged:
            # Collect Re measurement
            vel_np = solver.vel.to_numpy()
            # ROI inlet = crop x_start = sponge_in + buffer_in（audit B5：舊用 scalar 'buffer'=0
            # → 探針落在 sponge 內、量到被阻尼的速度，Re 量測失真）。
            x_roi  = int(zones["sponge_in"]) + int(zones["buffer_in"])
            x_roi  = max(1, min(x_roi, vel_np.shape[0] - 2))
            u_inlet_meas = float(np.mean(vel_np[x_roi, 1:-1, 0]))
            re_meas = u_inlet_meas * l_char / nu if nu > 0 else 0.0

            metadata["status"]                        = "Success"
            metadata["u_inlet_lattice_lu"]            = u_inlet_meas
            metadata["reynolds_number_lattice_actual"] = re_meas
            metadata["l_char_lattice_px"]             = l_char
            metadata["nu_lattice_lu"]                 = nu
            metadata["total_steps_executed"]          = metadata.get("final_steps", 0)
            metadata["n_snapshots_written"]           = len(writer._steps) if writer else 0
            metadata["h5_file"]                       = h5_output_path
            metadata["reason"]                        = "TES trajectory completed successfully"

    except Exception as e:
        metadata["status"] = "Error"
        metadata["reason"] = str(e)
        print(f"\n[TES CRITICAL] {e}")
        traceback.print_exc()

    finally:
        converged   = metadata.get("status") == "Success"
        stop_reason = metadata.get("stop_reason") or (
            "converged" if converged else metadata.get("reason", "unknown")
        )
        convergence_ctu = float(metadata.get("convergence_ctu", -1.0))

        # ── 時均累加器收尾（Stage 1.15）：封最後一段 → 交給 writer 寫 H5 ──
        # 放在 finally 內，模擬中途失敗（stability_failure / 例外）時已累加的
        # 段仍會落盤，不靜默丟棄。
        if mean_accum is not None and writer is not None:
            try:
                if solver is not None:
                    solver.attach_time_mean_accumulator(None)
                mean_accum.close_segment()
                writer.set_mean_accum_payload(mean_accum.to_h5_payload())
            except Exception as e:
                print(f"[TES] mean_accum 收尾錯誤：{e}")

        if writer:
            try:
                writer.finalize(
                    converged=converged,
                    stop_reason=stop_reason,
                    ke_convergence_step=metadata.get("ke_convergence_step", -1),
                    ke_variance_ratio=metadata.get("ke_variance_ratio", -1.0),
                    convergence_ctu=convergence_ctu,
                    sim_start_utc=metadata.get("sim_start_utc", ""),
                    sim_end_utc=metadata.get("sim_end_utc", ""),
                    wall_time_total_s=float(metadata.get("wall_time_total_s", -1.0)),
                    wall_time_to_conv_s=float(metadata.get("wall_time_to_conv_s", -1.0)),
                    sim_conv_utc=metadata.get("sim_conv_utc", ""),
                    gt_quality={
                        k: metadata[k]
                        for k in (
                            "mean_window_truncated", "postconv_ctu",
                            "extra_ctu_requested", "conv_k_required",
                            "conv_eps", "conv_window_ctu", "conv_cum_eps",
                            "conv_trend_gate", "conv_final_ratio",
                            "conv_final_trend_drift", "conv_n_rejected_by_trend",
                            # ★ Stage 1.16：準穩態狀態紀錄（非門檻，見上方說明）
                            "quasi_steady_check", "quasi_steady_ke_change_pct",
                            "quasi_steady_source", "quasi_steady_threshold_pct",
                        )
                        if k in metadata
                    },
                )
            except Exception as e:
                print(f"[TES] Writer finalize error: {e}")

        try:
            reset_taichi_runtime()
        except Exception:
            pass

    return metadata
