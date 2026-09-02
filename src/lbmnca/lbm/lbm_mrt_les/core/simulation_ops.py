import time
import traceback

import numpy as np
import taichi as ti
from tqdm import tqdm

from .. import utils
from .convergence_monitor import ConvergenceMonitor


def check_stability(
    forces,
    max_v,
    step_count,
    v_threshold=0.25,
    f_threshold=1e6,
    warmup_step=1000,
):
    """
    [數值監控] 檢查模擬是否穩定 (數值熔斷機制)

    Args:
        forces (list/array): [fx, fy] 當前總受力
        max_v (float): 全場最大流速 (LBM 單位)
        step_count (int): 當前模擬步數
        v_threshold (float): 速度熔斷閾值 (建議設為 0.25, 避免馬赫數過高)
        f_threshold (float): 受力熔斷閾值
        warmup_step (int): 預熱步數

    Returns:
        (bool, str): (是否穩定, 錯誤訊息)
    """
    # --- 1. Force Check (NaN/Inf 檢查永遠執行) ---
    fx, fy = forces[0], forces[1]

    if np.isnan(fx) or np.isnan(fy) or np.isinf(fx) or np.isinf(fy):
        return False, f"Force becomes NaN/Inf at step {step_count} (Fx={fx}, Fy={fy})"

    if abs(fx) > f_threshold or abs(fy) > f_threshold:
        return (
            False,
            f"Force exploded (> {f_threshold:.1e}) at step {step_count} (Fx={fx:.2e}, Fy={fy:.2e})",
        )

    # --- 2. Velocity Check ---
    if np.isnan(max_v) or np.isinf(max_v):
        return False, f"Velocity field contains NaN/Inf at step {step_count}"

    # 數值大小檢查 (給予緩衝期 Warmup)
    if step_count > warmup_step:
        # LBM 的穩定性限制：Ma < 1/sqrt(3) ~= 0.577
        # 為了保持不可壓縮假設與數值穩定，建議閾值設為 0.25
        if max_v > v_threshold:
            return (
                False,
                f"Velocity {max_v:.4f} exceeded stability threshold ({v_threshold}) at step {step_count}",
            )

    return True, ""


def run_simulation_loop(config, solver, viz, recorder, gui, writer, max_steps):
    """
    [核心迴圈] LBM 模擬主流程
    回傳: 包含執行狀態與物理參數的字典 (metadata)
    """
    # --- 1. 讀取設定與初始化 ---
    sim_cfg = config["simulation"]
    out_cfg = config["outputs"]
    zones = utils.get_zone_config(config)

    compute_step_size = sim_cfg["compute_step_size"]
    gui_interval = out_cfg["gui"]["interval_steps"]
    vid_interval = out_cfg["video"]["interval_steps"]
    data_interval = out_cfg["dataset"]["interval_steps"]
    start_record = out_cfg.get("start_record_step", 0)

    # --- Phase 3: 收斂監測 ---
    steps_per_ctu = sim_cfg.get("steps_per_ctu", max(1, compute_step_size * 20))
    time_policy = sim_cfg.get("time_length_policy", {})
    diag_interval = int(time_policy.get("diagnostic_interval_steps_target", steps_per_ctu // 4))
    diag_interval = max(compute_step_size, diag_interval)
    convergence_thresholds = time_policy.get("convergence_thresholds", None)

    monitor = ConvergenceMonitor(
        window_size=max(3, int(time_policy.get("convergence_min_samples", 9))),
        steps_per_ctu=steps_per_ctu,
    )

    current_steps = 0
    # mininterval=1.0：每秒最多重畫一次（Windows CMD 沒這參數時每步印新行）
    # dynamic_ncols=True：自動偵測終端寬度，避免換行污染
    pbar = tqdm(total=max_steps, unit="step", mininterval=1.0, dynamic_ncols=True)

    # [狀態追蹤] 初始化預設狀態
    exit_status = "Success"  # Success, Failed, Aborted, Error
    exit_reason = "Reached max_steps"  # 詳細原因

    # [Profile] 初始化計時器
    timings = {"compute": 0.0, "stability": 0.0, "viz_proc": 0.0, "hdf5_io": 0.0}

    try:
        while current_steps < max_steps:
            t_loop_start = time.perf_counter()

            # [Check 1] GUI 關閉檢查
            if gui and not gui.running:
                exit_status = "Aborted"
                exit_reason = "GUI closed by user"
                print(f"\n[Info] {exit_reason}")
                break

            # =========================================================
            # 1. 核心運算（GPU 連續計算，不中斷）
            # =========================================================
            t0 = time.perf_counter()
            solver.run_step(compute_step_size)
            current_steps += compute_step_size

            # 2. 穩定性檢查 + 收斂監測（共用一次 GPU→CPU sync）
            #    get_force 和 get_max_velocity 各觸發一次 sync，
            #    但它們緊接在一起，GPU pipeline 已排空。
            forces = solver.get_force()
            max_v = solver.get_max_velocity()
            timings["compute"] = (time.perf_counter() - t0) * 1000

            is_stable, reason = check_stability(
                forces, max_v, current_steps, warmup_step=sim_cfg["warmup_steps"]
            )
            if not is_stable:
                exit_status = "Failed"
                exit_reason = reason
                print(f"\n\033[91m[CRITICAL] Simulation Failed: {reason}\033[0m")
                break

            # Phase 3: 收斂診斷（重用已取得的 max_v/forces，不再重複 sync）
            if current_steps % diag_interval < compute_step_size:
                monitor._snapshot_count += 1
                monitor._max_v_history.append(float(max_v))
                monitor._ke_history.append(float(max_v * max_v))
                force_mag = float(np.sqrt(forces[0]**2 + forces[1]**2))
                monitor._mass_flux_history.append(force_mag)
                monitor._step_history.append(current_steps)
                # 更新 _latest（簡化版，避免重複呼叫 solver）
                snapshot = {
                    "step": current_steps,
                    "ctu": round(current_steps / monitor._steps_per_ctu, 2),
                    "max_v": round(float(max_v), 6),
                    "ke_proxy": round(float(max_v * max_v), 8),
                    "force_magnitude": round(force_mag, 6),
                }
                if len(monitor._ke_history) >= 3:
                    ke_arr = np.array(monitor._ke_history)
                    ke_mean = float(np.mean(ke_arr))
                    ke_std = float(np.std(ke_arr))
                    snapshot["ke_window_mean"] = round(ke_mean, 8)
                    snapshot["ke_window_rel_drift"] = round(ke_std / max(ke_mean, 1e-12), 6)
                    fmag_arr = np.array(monitor._mass_flux_history)
                    fmag_mean = float(np.mean(fmag_arr))
                    fmag_std = float(np.std(fmag_arr))
                    snapshot["force_window_mean"] = round(fmag_mean, 6)
                    snapshot["force_window_rel_drift"] = round(fmag_std / max(fmag_mean, 1e-12), 6)
                monitor._latest = snapshot
                monitor.history.append(snapshot)

            # 更新 CLI 進度條（純 default 樣式，不附 Fx/Fy/MaxV postfix）
            pbar.update(compute_step_size)

            # =========================================================
            # 4. 視覺化處理 (Visualization Pipeline)
            # =========================================================
            t0 = time.perf_counter()
            vel_field, mask_data = None, None # 延後獲取物理場
            
            need_viz = False
            is_gui_frame = out_cfg["gui"]["enable"] and (
                current_steps % gui_interval == 0
            )
            is_vid_frame = out_cfg["video"]["enable"] and (
                current_steps % vid_interval == 0
            ) and current_steps >= start_record

            img = None
            if is_gui_frame or is_vid_frame:
                vel_field, mask_data = solver.get_physical_fields()
                img = viz.process_frame(vel_field, mask_data)
                need_viz = True
            t1 = time.perf_counter()
            timings["viz_proc"] = (t1 - t0) * 1000 if need_viz else 0.0

            # --- 4.1 GUI ---
            t0 = time.perf_counter()
            if is_gui_frame and gui:
                gui.set_image(img)
                if out_cfg["gui"]["show_zone_overlay"]:
                    utils.draw_zone_overlay(gui, zones, y_offset=0.0)
                    utils.draw_zone_overlay(gui, zones, y_offset=0.5)
                gui.show()
            t1 = time.perf_counter()
            timings["gui_draw"] = (t1 - t0) * 1000 if is_gui_frame else 0.0

            # --- 4.2 Video ---
            t0 = time.perf_counter()
            if is_vid_frame and recorder:
                recorder.write_frame(np.transpose(img, (1, 0, 2)))
            t1 = time.perf_counter()
            timings["video_io"] = (t1 - t0) * 1000 if is_vid_frame else 0.0

            # =========================================================
            # 5. HDF5 數據集寫入
            # =========================================================
            t0_fetch = time.perf_counter()
            is_data_step = out_cfg["dataset"]["enable"] and (
                current_steps % data_interval == 0
            ) and current_steps >= start_record

            if is_data_step and writer:
                moments_raw = solver.get_moments_numpy()
                t1_fetch = time.perf_counter()
                timings["moment_fetch"] = (t1_fetch - t0_fetch) * 1000

                t0_io = time.perf_counter()
                writer.append(moments_raw)
                t1_io = time.perf_counter()
                timings["hdf5_io"] = (t1_io - t0_io) * 1000
            else:
                timings["moment_fetch"] = 0.0
                timings["hdf5_io"] = 0.0

            # =========================================================
            # [Profile Report Console]
            # =========================================================
            total_time = (time.perf_counter() - t_loop_start) * 1000
            if (current_steps // compute_step_size) % 10 == 0 and config["outputs"][
                "enable_profiling"
            ]:
                print(f"\n[Profile] Step {current_steps} | Loop: {total_time:.1f}ms")
                print(f"  ├─ LBM Compute:  {timings['compute']:.1f} ms")
                if need_viz:
                    print(
                        f"  ├─ Vis Pipeline: {timings['viz_proc'] + timings['gui_draw'] + timings['video_io']:.1f} ms"
                    )
                if is_data_step:
                    print(
                        f"  └─ \033[92m[SAVE]\033[0m HDF5 I/O:    {timings['moment_fetch'] + timings['hdf5_io']:.1f} ms"
                    )
                else:
                    print(f"  └─ [SKIP] HDF5 I/O:    0.0 ms")

    except KeyboardInterrupt:
        exit_status = "Aborted"
        exit_reason = "User Interrupted (Ctrl+C)"
        print(f"\n[Info] {exit_reason}")

    except Exception as e:
        # [Check 3] 捕捉未預期的程式錯誤 (如 IndexOutOfBounds, VRAM OOM)
        exit_status = "Error"
        exit_reason = f"Runtime Error: {str(e)}"
        print(f"\n\033[91m[ERROR] Exception occurred: {exit_reason}\033[0m")
        traceback.print_exc()  # 印出詳細錯誤位置

    finally:
        pbar.close()

    if writer is not None:
        try:
            writer.finalize()
        except Exception as e:
            print(f"[Warning] Writer finalize failed: {e}")

    # --- 建構完整回傳資料 ---
    metadata = {
        # 1. 執行狀態（供 Batch Runner 判讀）
        "status": exit_status,
        "reason": exit_reason,
        "final_steps": current_steps,
        "target_steps": max_steps,
        # 2. 物理參數（供資料分析）
        "re_val": float(solver.Re) if hasattr(solver, "Re") else 0.0,
        "u_max": (
            float(np.linalg.norm(solver.u_inlet)) if hasattr(solver, "u_inlet") else 0.0
        ),
        "D": float(config["simulation"]["characteristic_length"]),
        "nu": float(config["simulation"]["nu"]),
        # 3. Phase 3: 收斂監測摘要
        "convergence_summary": monitor.get_summary(),
    }

    return metadata
