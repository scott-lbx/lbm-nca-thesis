"""GGUI 主循環 — 整合 NCA runner、DLPack、colorize、playback。"""

from __future__ import annotations

import numpy as np
import taichi as ti
import torch

from lbmnca.bridge.core.session import SimPhase


def run_ggui(config: dict, session, cmd_queue, gpu_fields) -> None:
    """GGUI 主循環（必須在主 thread 呼叫）。

    Args:
        config: 設定字典，需包含 config["resolution"]["H"]、config["resolution"]["W"]
                以及 config["physics"]["lbm_to_ms"]、config["streamline"]["dt"]。
        session: NCA session 物件（含 nca_runner、trajectory、sim_phase 等欄位）。
        cmd_queue: 來自 WebSocket 背景執行緒的命令佇列（需支援 drain()）。
        gpu_fields: GPU field 集合物件。
    """
    H = config["resolution"]["H"]
    W = config["resolution"]["W"]

    # display_scale=1 → 視窗 = zarr 原生解析度 (W, H)=(640, 360)；放大倍率由 config 控
    scale = config["resolution"]["display_scale"]
    window = ti.ui.Window(
        "Wind Environment Analyzer",
        res=(W * scale, H * scale),
        vsync=True,
        show_window=True,
    )
    canvas = window.get_canvas()

    # ── 設定視窗永遠置頂 (Windows 限定) ──
    import sys

    if sys.platform == "win32":
        import ctypes

        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        hwnd = ctypes.windll.user32.FindWindowW(None, "Wind Environment Analyzer")
        if hwnd:
            ctypes.windll.user32.SetWindowPos(
                hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE
            )

    from lbmnca.bridge.ggui.playback import PlaybackState, PlaybackMode, update_playback
    from lbmnca.bridge.ggui.input_handler import (
        handle_keyboard,
        handle_mouse_timeline,
        handle_mouse_geometry,
    )
    from lbmnca.bridge.ggui.ui_draw import draw_timeline

    from lbmnca.bridge.render.field_ops import derive_fields, derive_vorticity, compute_vmax
    from lbmnca.bridge.render.colorize import (
        to_display,
        colorize_velocity,
        colorize_vorticity,
        colorize_pressure,
        colorize_comfort,
        colorize_ibfv,
    )
    from lbmnca.bridge.render.overlay import overlay_buildings, draw_wind_arrow
    from lbmnca.bridge.render.streamlines import trace_streamlines, init_seeds
    from lbmnca.bridge.render.ibfv_pass import IBFVPass
    from lbmnca.bridge.render.ibfv_params import IBFVParams
    from lbmnca.bridge.render.playback_director import PlaybackDirector

    from lbmnca.bridge.geometry.mask_rasterizer import MaskRasterizer
    from lbmnca.bridge.render.convergence_overlay import overlay_convergence
    from lbmnca.bridge.simulation.convergence_guard import ConvergenceGuard

    rasterizer = MaskRasterizer(H, W)
    if session.guard is None:
        session.guard = ConvergenceGuard()

    # 預設視覺化模式（config SSOT；鍵 1-6 仍可切換）
    session.viz_mode = config["render"]["default_viz_mode"]

    # ── demo_chain：warm case chain 自動駕駛 ─────────────────────────
    # 首 case inject_ic=True（start frame = GT mean）；其後 inject_ic=False →
    # decompose backend 以前一 case 凍結態 warm 接鏈（apply_ic_mask 處理新幾何）。
    # 每 case COMPLETED 後停 hold_seconds，再推下一 case；播畢自動關窗。
    chain_cfg = config.get("demo_chain") or {}
    chain_ids: list = list(chain_cfg["case_ids"]) if chain_cfg.get("enabled") else []
    chain_pos = -1
    chain_hold_t: float | None = None
    chain_finished = False

    def _push_chain_case(pos: int) -> None:
        from lbmnca.bridge.core.command_queue import Command

        cmd_queue.push(
            Command(
                action="analyze_zarr",
                payload={
                    "dataset": chain_cfg["dataset"],
                    "case_id": chain_ids[pos],
                    "wind_level": int(chain_cfg["wind_level"]),
                    "inject_ic": pos == 0,  # 只有第一個 start frame 用 GT mean
                },
            )
        )
        print(
            f"[chain] case {pos + 1}/{len(chain_ids)}: {chain_ids[pos]}"
            f"（inject_ic={pos == 0}）"
        )

    # 播放狀態（local，每幀更新）
    playback = PlaybackState()

    # 初始化 mask 為全流體（啟動前沒有建物）
    gpu_fields.mask.from_numpy(np.ones((H, W), dtype=np.float32))

    # 初始化種子點供流線用
    init_seeds(gpu_fields.seed_pos, H, W)

    # ── IBFV + PlaybackDirector（smooth_transition Phase A，mean-only 全鏈）──
    # ibfv：mean 層單 pass（綁 GPUFields ibfv_mean_a/b ping-pong field）。
    # director：MEAN_CONVERGING 收斂序列累積 + 慢放。turb 雙層為 Phase B（TODO）。
    ibfv_cfg = config["ibfv"]
    director_cfg = config["director"]
    ibfv_enabled = bool(ibfv_cfg["enabled"])
    director_enabled = bool(director_cfg["enabled"])

    mean_ibfv_params = IBFVParams(
        alpha=float(ibfv_cfg["mean_alpha"]),
        dt_advect=float(ibfv_cfg["mean_dt_advect"]),
        noise_freq=float(ibfv_cfg["mean_noise_freq"]),
        noise_time_rate=float(ibfv_cfg["mean_noise_time_rate"]),
    )
    ibfv_mean = IBFVPass(gpu_fields.ibfv_mean_a, gpu_fields.ibfv_mean_b, H, W)
    ibfv_seeded_for_geom = False  # 幾何改變後需重新 seed IBFV 紋理

    director = PlaybackDirector(
        rate=float(director_cfg["rate"]),
        easing=director_cfg["easing"],
        loop_mode="none",  # mean 收斂 = 有限序列，播完 hold
        crossfade_frames=float(director_cfg["crossfade_frames"]),
    )

    from lbmnca.bridge.ggui.profiling import make_profiler

    prof = make_profiler(config)
    lbm_to_ms = config["physics"]["lbm_to_ms"]
    stream_dt = config["streamline"]["dt"]

    # dirty-flag render：場只在「有變化」時重算（最大效能來源）。
    # 每幀仍重發 set_image / canvas.lines / show（GGUI 每幀重繪）。
    last_idx = -1
    last_viz = None
    last_gen = -1
    last_phase = None  # phase 轉換（如 MEAN_CONVERGING→COMPLETED）需重繪清 overlay
    geom_dirty = True
    snap_held = False  # 快照鍵 'p' 防連續觸發

    VMAX_UPDATE_INTERVAL = 10
    _vmax_cache: float = 1.0
    _vmax_frame: int = 0
    _frame_idx: int = 0

    # ── vis_range SSOT（stats.json，經 backend FieldNormalizer）：固定 colormap 尺度 ──
    #   不閃、與 02 訓練/sim-viz 同尺度、部署一致。取不到（dummy/fieldstore 等無模型
    #   測試 backend）則 vis_vel_max=None → 退回動態 compute_vmax（僅測試，無真模型）。
    def _resolve_field_norm(backend):
        vn = getattr(backend, "field_norm", None)
        if vn is not None:
            return vn
        for _sub in ("_turb_ad", "_mean_ad"):  # decompose backend：取子 adapter
            ad = getattr(backend, _sub, None)
            if ad is not None and getattr(ad, "field_norm", None) is not None:
                return ad.field_norm
        return None

    _field_norm = _resolve_field_norm(session.backend)
    # 02 vis_range SSOT API（vel_vis_max 等）為 concurrent WIP；用能力探測優雅降級，
    # 未落地時 → None/預設（動態 vmax），避免 AttributeError 崩 GGUI。
    vis_vel_max = (
        getattr(_field_norm, "vel_vis_max", None) if _field_norm is not None else None
    )
    vis_vor_range = (
        getattr(_field_norm, "vor_vis_range", None) if _field_norm is not None else None
    )
    vis_p_center = (
        getattr(_field_norm, "p_vis_center", 1.0 / 3.0)
        if _field_norm is not None
        else 1.0 / 3.0
    )
    vis_p_range = (
        getattr(_field_norm, "p_vis_range", 0.01) if _field_norm is not None else 0.01
    )
    if vis_vel_max is not None:
        print(
            f"[colorize] vis_range SSOT 固定尺度  vel_max={vis_vel_max:.4f} "
            f"vor_range={vis_vor_range:.4f}（stats.json）"
        )
    else:
        print("[colorize] backend 無 FieldNormalizer → 動態 vmax（測試 backend）")

    # ── FieldColorizer（lbmnca.nca，torch；plasma 固定範圍 + obstacle gray）──
    field_colorizer = None
    if config.get("use_field_colorizer"):
        try:
            from lbmnca.nca.utils.FieldColorizer import FieldColorizer as _FC

            _fc_dev = getattr(session.backend, "device", "cuda")
            _fc_obj = (
                _FC.from_normalizer(_field_norm)
                if (_field_norm is not None and hasattr(_FC, "from_normalizer"))
                else _FC()
            )
            field_colorizer = _fc_obj.to(_fc_dev).eval()
            print("[colorize] 使用 02 FieldColorizer (plasma, vis_range SSOT)")
        except Exception as e:  # noqa: BLE001
            print(f"[colorize] FieldColorizer 載入失敗，回退 Taichi colorize：{e}")

    # ── 錄影（config record，供事後抽幀分析）──────────────────────
    recorder = None
    rec_max_frames = 0
    rec_count = 0
    png_every = 0
    png_dir = ""
    rec_exit_on_complete = False
    rec_cfg = config.get("record") or {}
    if rec_cfg.get("enabled"):
        import cv2
        import os
        import time as _rec_time

        rec_path = rec_cfg.get("path", "ggui_capture.mp4")
        rec_fps = int(rec_cfg.get("fps", 30))
        recorder = cv2.VideoWriter(
            rec_path, cv2.VideoWriter_fourcc(*"mp4v"), rec_fps, (W, H)
        )
        rec_max_frames = int(float(rec_cfg.get("max_seconds", 40)) * rec_fps)
        rec_exit_on_complete = bool(
            rec_cfg["exit_on_complete"]
        )  # 錄完自動關窗（自動化測試）
        png_every = int(rec_cfg.get("png_every", 0))  # >0 → 同時輸出 PNG sequence
        if png_every > 0:
            # 每 run 獨立 timestamped 資料夾，防跨 run 覆蓋（P0-2）
            png_dir = os.path.join(
                rec_cfg.get("png_dir", "test_outputs/png_seq"),
                _rec_time.strftime("run_%Y%m%d_%H%M%S"),
            )
            os.makedirs(png_dir, exist_ok=True)
        print(
            f"[record] → {rec_path} ({W}x{H}@{rec_fps}, max {rec_max_frames} frames"
            f"{f', PNG every {png_every} → {png_dir}' if png_every > 0 else ''}"
            f"{', exit_on_complete' if rec_exit_on_complete else ''})"
        )

    # ── 初始化每秒存檔的功能 ──
    import os
    import shutil
    import time

    _temp_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../..", "test_outputs", "_temp"
        )
    )
    if os.path.exists(_temp_dir):
        try:
            shutil.rmtree(_temp_dir)
        except Exception as e:
            print(f"\033[91m[window] Warning: Could not clear _temp folder: {e}\033[0m")
    os.makedirs(_temp_dir, exist_ok=True)
    last_png_time = time.time()
    png_count = 0

    # ── 初始化選取框的 Taichi field ──
    selected_outline = ti.Vector.field(2, ti.f32, shape=20)

    # ── 時間軸統一（steps/sec）：model 2 steps/s、render 30 fps ──────────
    #   model step 用 wall-clock 節流（與 render fps 解耦）；render 用 frame cap。
    #   director rate = model_step_fps / render_fps → 1 model step 內插補滿 (render/model) 幀。
    model_step_fps = float(config["model"]["step_fps"])
    render_fps = float(config["render"]["fps_target"])
    show_hud = bool(config["render"]["show_hud"])
    _step_interval = 1.0 / model_step_fps
    _frame_interval = 1.0 / render_fps
    _last_step_t = 0.0
    if director_enabled:
        director.rate = (
            model_step_fps / render_fps
        )  # real-time 平滑內插（取代 config 慢放值）
        print(
            f"[timing] model {model_step_fps} steps/s | render {render_fps} fps | "
            f"director rate={director.rate:.3f}（每 {render_fps/model_step_fps:.0f} frame/step）"
        )

    while window.running:
        _loop_t0 = time.perf_counter()
        with prof.stage("cmd"):
            for cmd in cmd_queue.drain():
                _process_command(cmd, session, rasterizer, gpu_fields, playback)
                geom_dirty = True

        # ── demo_chain driver：COMPLETED + hold 到期 → 推下一 case ──────
        if chain_ids and not chain_finished:
            if chain_pos < 0:
                chain_pos = 0
                _push_chain_case(0)
            elif session.sim_phase == SimPhase.COMPLETED:
                if chain_hold_t is None:
                    chain_hold_t = time.time()
                elif time.time() - chain_hold_t >= float(chain_cfg["hold_seconds"]):
                    if chain_pos < len(chain_ids) - 1:
                        chain_pos += 1
                        chain_hold_t = None
                        _push_chain_case(chain_pos)
                    else:
                        chain_finished = True
                        print("[chain] 全部 case 播畢 → 自動關閉視窗")
            else:
                chain_hold_t = None

        with prof.stage("step"):
            # model step 節流到 model_step_fps（wall-clock）；render 每幀仍跑
            if _loop_t0 - _last_step_t >= _step_interval:
                _step_nca_if_running(session, gpu_fields, playback)
                _last_step_t = _loop_t0

        # ── 幾何改變：清 director 序列 + 標記 IBFV 待重新 seed ──────────
        if geom_dirty:
            if director_enabled:
                director.clear()
            ibfv_seeded_for_geom = False

        # ── IBFV/Director 路徑（mean-only 全鏈，獨立於 trajectory 取幀）──
        # 每 backend step 把 mean 層 moments 推進 director 收斂序列（MEAN 期累積）。
        is_ibfv = ibfv_enabled and session.viz_mode == "ibfv"
        if session.director_rate is not None:
            director.rate = session.director_rate
        if director_enabled and session.backend is not None:
            field_layers_fn = getattr(session.backend, "field_layers", None)
            if callable(field_layers_fn) and session.generation_head != last_gen:
                with prof.stage("director_push"):
                    mean_moments, _fluct = field_layers_fn()  # Phase A：fluct=None
                    if mean_moments is not None:
                        director.push(mean_moments.detach().clone())

        n_frames = session.trajectory.n_frames if session.trajectory else 0
        idx = (
            max(0, min(int(playback.playback_head), n_frames - 1))
            if n_frames > 0
            else -1
        )

        # IBFV 為 animated（每幀平流推進）→ 永遠重算，繞過 dirty-flag 閘門。
        is_stream = session.viz_mode == "streamline"
        need_render = is_ibfv or (
            geom_dirty
            or idx != last_idx
            or session.viz_mode != last_viz
            or session.generation_head != last_gen
            or session.sim_phase != last_phase
        )

        if is_ibfv:
            _render_ibfv(
                gpu_fields,
                director,
                ibfv_mean,
                mean_ibfv_params,
                derive_fields,
                compute_vmax,
                colorize_ibfv,
                seeded=ibfv_seeded_for_geom,
                H=H,
                W=W,
                prof=prof,
                field_colorizer=field_colorizer,
                backend=session.backend,
                tex_strength=float(ibfv_cfg["tex_strength"]),
                tex_vel_knee=float(ibfv_cfg["tex_vel_knee"]),
                vis_vel_max=vis_vel_max,
                inlet_fix_width=session.inlet_fix_width,
            )
            ibfv_seeded_for_geom = True
            with prof.stage("overlay"):
                overlay_buildings(gpu_fields.mask, gpu_fields.rgb)
            last_idx, last_viz, last_gen, geom_dirty = (
                idx,
                session.viz_mode,
                session.generation_head,
                False,
            )
            last_phase = session.sim_phase
        elif need_render:
            if n_frames > 0:
                frame = _border_fix(
                    session.trajectory.get_frame(idx), session.inlet_fix_width
                )  # 邊界偽影修正
                with prof.stage("dlpack"):
                    gpu_fields.physics.from_torch(frame.contiguous())
                with prof.stage("derive"):
                    derive_fields(
                        gpu_fields.physics,
                        gpu_fields.vel_x,
                        gpu_fields.vel_y,
                        gpu_fields.vel_mag,
                        gpu_fields.pressure,
                    )
                    derive_vorticity(
                        gpu_fields.vel_x, gpu_fields.vel_y, gpu_fields.vorticity, H, W
                    )
                with prof.stage("vmax"):
                    if vis_vel_max is not None:
                        vmax_val = vis_vel_max + 1e-8  # 固定 SSOT 尺度（不閃）
                    else:
                        if _frame_idx - _vmax_frame >= VMAX_UPDATE_INTERVAL:
                            compute_vmax(
                                gpu_fields.vel_mag, gpu_fields.mask, gpu_fields._vmax
                            )
                            _vmax_cache = float(gpu_fields._vmax[None])
                            _vmax_frame = _frame_idx
                        vmax_val = _vmax_cache + 1e-8
                with prof.stage("colorize"):
                    if (
                        session.viz_mode in ("velocity", "streamline")
                        and field_colorizer is not None
                        and n_frames > 0
                    ):
                        dev = frame.device  # frame 已於上方 border-fix
                        rho = frame[0].clamp_min(1e-6)
                        vmag = torch.sqrt(
                            (frame[3] / rho) ** 2 + (frame[5] / rho) ** 2
                        )[None]
                        mask_t = gpu_fields.mask.to_torch(device=dev)
                        rgb_t = field_colorizer.getVelColor(
                            vmag, mask_t[None], use_fixed_range=True
                        )
                        gpu_fields.rgb.from_torch(
                            rgb_t.permute(1, 2, 0).contiguous().clamp(0.0, 1.0)
                        )
                    else:
                        _colorize(
                            session.viz_mode,
                            gpu_fields,
                            vmax_val,
                            lbm_to_ms,
                            vor_range=vis_vor_range,
                            p_center=vis_p_center,
                            p_range=vis_p_range,
                        )
                if is_stream:
                    with prof.stage("streamtrace"):
                        trace_streamlines(
                            gpu_fields.vel_x,
                            gpu_fields.vel_y,
                            gpu_fields.mask,
                            gpu_fields.seed_pos,
                            gpu_fields.line_verts,
                            dt=stream_dt,
                            H=H,
                            W=W,
                        )
                        session._stream_all_seeds_dumped = True
            else:
                gpu_fields.rgb.fill(0.1)

            with prof.stage("overlay"):
                # activity overlay 只在均場收斂期有意義；凍結/COMPLETED 後
                # backend 不再更新 convergence field，再疊加即 stale 斑塊殘影
                if (
                    getattr(session.backend, "has_convergence", False)
                    and session.sim_phase == SimPhase.MEAN_CONVERGING
                ):
                    overlay_convergence(
                        gpu_fields.convergence, gpu_fields.mask, gpu_fields.rgb, 0.18
                    )
                overlay_buildings(gpu_fields.mask, gpu_fields.rgb)

            last_idx, last_viz, last_gen, geom_dirty = (
                idx,
                session.viz_mode,
                session.generation_head,
                False,
            )
            last_phase = session.sim_phase

        # 每幀重發（GGUI 每幀重繪 framebuffer）
        with prof.stage("show"):
            # 轉置 (H,W)→(W,H) 修正壓縮：保持 dataset_zarr 的 landscape 比例（640:360）
            to_display(gpu_fields.rgb, gpu_fields.rgb_disp)
            canvas.set_image(gpu_fields.rgb_disp)
            if is_stream and n_frames > 0:
                # ★2026-07-23：改用 canvas.circles 逐點畫，不用 canvas.lines。
                # 診斷過程（見 docs/audit/0723 fix log）：(1) 用 matplotlib 直接畫
                # trace_streamlines 算出的原始 verts 陣列（逐種子分開畫），流線完全正常
                # （自由流近水平、尾流正確繞行/成渦）——證明追蹤演算法本身沒錯；
                # (2) 但 canvas.lines 畫出來的畫面持續出現大量左下到右上的粗斜線；
                # (3) 排除了細線次像素 rasterize（調寬 width 無效）、種子間無斷點
                # （逐種子最後一格強制寫 (-1,-1) 仍無效）兩個假設，找不到 canvas.lines
                # 這個「連續多點」畫法本身在哪裡出錯——與其繼續猜 Taichi GGUI 內部實作，
                # 改用逐點畫（每點獨立、無配對/串接語意可出錯）徹底繞開這整類問題，
                # 且已用 matplotlib 驗證過這批點本身的座標是對的。
                canvas.circles(
                    centers=gpu_fields.line_verts, radius=0.0016, color=(1.0, 1.0, 0.5)
                )

            sel_idx = getattr(session, "selected_building_idx", -1)
            if sel_idx != -1 and getattr(session, "active_polygons", None):
                poly = session.active_polygons[sel_idx]
                n_pts = len(poly)
                pts = []
                for i in range(n_pts):
                    p1 = poly[i]
                    p2 = poly[(i + 1) % n_pts]
                    # Taichi GUI y-axis (bottom-up) perfectly matches our origin="lower" polygons
                    pts.append([p1[0], p1[1]])
                    pts.append([p2[0], p2[1]])
                pts_np = np.array(pts, dtype=np.float32)
                if len(pts_np) <= 20:
                    selected_outline.from_numpy(pts_np)
                    # 畫紅框
                    canvas.lines(selected_outline, width=0.003, color=(1.0, 0.0, 0.0))

            playback.total_frames = n_frames
            playback.generation_head = session.generation_head
            if show_hud:
                draw_timeline(canvas, playback, gpu_fields)
                _draw_status_overlay(window, session)
                _draw_inlet_fix_slider(window, session)

        with prof.stage("input"):
            kb_cmd = handle_keyboard(window)
            if kb_cmd:
                action = kb_cmd.get("action")
                sel_idx = getattr(session, "selected_building_idx", -1)
                if (
                    action in ("arrow_left", "arrow_right", "arrow_up", "arrow_down")
                    and sel_idx != -1
                    and getattr(session, "active_polygons", None)
                ):
                    dx, dy = 0.0, 0.0
                    step_x = 1.0 / W  # 1 pixel at a time
                    step_y = 1.0 / H
                    if action == "arrow_left":
                        dx = -step_x
                    elif action == "arrow_right":
                        dx = step_x
                    elif action == "arrow_up":
                        dy = -step_y
                    elif action == "arrow_down":
                        dy = step_y

                    poly = session.active_polygons[sel_idx]
                    for pt in poly:
                        pt[0] = max(0.0, min(1.0, pt[0] + dx))
                        pt[1] = max(0.0, min(1.0, pt[1] + dy))

                    _apply_geometry_update(
                        session,
                        rasterizer,
                        gpu_fields,
                        session.active_polygons,
                        session.wind_angle,
                        session.re_level,
                        "adapt",
                        "analyze",
                    )
                    geom_dirty = True
                elif action in ("arrow_left", "arrow_right"):
                    # fallback to step for timeline
                    kb_cmd["action"] = "step"
                    kb_cmd["delta"] = -1 if action == "arrow_left" else 1
                    _handle_kb_command(kb_cmd, session, playback)
                else:
                    _handle_kb_command(kb_cmd, session, playback)

            # 快照：按 'p' 存 GGUI 當前 framebuffer 為 PNG（edge-trigger 防連發）
            if window.is_pressed("p"):
                if not snap_held:
                    snap_path = (
                        f"test_outputs/ggui_snapshot_{session.total_frames:06d}.png"
                    )
                    window.save_image(snap_path)
                    print(f"[snapshot] {snap_path}")
                    snap_held = True
            else:
                snap_held = False
            seek_t = handle_mouse_timeline(window, playback.total_frames)
            if seek_t is not None:
                playback.playback_head = seek_t
                session.playback_mode = "seek"
                playback.mode = PlaybackMode.SEEK

            # 處理幾何滑鼠拖曳，即時適配流場
            if getattr(session, "active_polygons", None) is not None:
                if handle_mouse_geometry(window, session):
                    _apply_geometry_update(
                        session,
                        rasterizer,
                        gpu_fields,
                        session.active_polygons,
                        session.wind_angle,
                        session.re_level,
                        "adapt",
                        "analyze",
                    )
                    geom_dirty = True

            _sync_playback_from_session(playback, session)
            update_playback(playback, dt_sec=1.0 / 60.0)
            _sync_playback_to_session(playback, session)

        # ── 每 1 秒自動儲存一個 PNG 幀到 _temp 資料夾 ──
        current_time = time.time()
        if current_time - last_png_time >= 1.0:
            png_path = os.path.join(_temp_dir, f"frame_{png_count:06d}.png")
            window.save_image(png_path)
            last_png_time = current_time
            png_count += 1

        with prof.stage("present"):
            window.show()

        # 錄影：抓 colorized rgb (H,W,3) → mp4（flipud 對齊 origin-lower 顯示，RGB→BGR）
        if recorder is not None and rec_count < rec_max_frames:
            fr = (np.clip(gpu_fields.rgb.to_numpy(), 0.0, 1.0) * 255).astype(np.uint8)
            frame_bgr = np.flipud(fr)[:, :, ::-1]
            recorder.write(frame_bgr)
            rec_count += 1
            if png_every > 0 and rec_count % png_every == 0:
                cv2.imwrite(f"{png_dir}/f_{rec_count:06d}.png", frame_bgr)
            if rec_count >= rec_max_frames:
                recorder.release()
                recorder = None
                print(f"[record] 完成 {rec_count} frames")
                if png_every > 0:
                    print(f"[record] PNG 序列 → {png_dir}")
                if rec_exit_on_complete:
                    print("[record] exit_on_complete → 自動關閉視窗")
                    break

        prof.tick()
        _frame_idx += 1

        if chain_finished:
            break

        # render frame cap → render_fps（30）；與 model step 節流解耦
        _elapsed = time.perf_counter() - _loop_t0
        if _elapsed < _frame_interval:
            time.sleep(_frame_interval - _elapsed)

    if recorder is not None:
        recorder.release()
        print(f"[record] 視窗關閉，已存 {rec_count} frames")


# ─────────────────────────────────────────────────────────────────────────────
# 內部輔助函式
# ─────────────────────────────────────────────────────────────────────────────

_PHASE_LABEL = {
    SimPhase.MEAN_CONVERGING: "Phase A: Mean Flow Converging... (Do not drag)",
    SimPhase.TURB_LIVE: "Phase B: Turb Live (Safe to drag)",
    SimPhase.COMPLETED: "Converged / Completed (Safe to drag)",
    SimPhase.WARM_START: "Warm Starting...",
    SimPhase.COLD_START: "Cold Starting...",
    SimPhase.IDLE: "Idle",
}


def _draw_status_overlay(window, session) -> None:
    """decompose phase 標示 + 分布守門/G_vec 警示 banner（GGUI sub-window）。"""
    from lbmnca.bridge.core.session import SimPhase
    from lbmnca.bridge.render.colorize import comfort_scale_warning
    import re

    warnings = list(getattr(session, "warnings", []) or [])
    # ★DEEP_AUDIT R4-G9：comfort 圖層的速度換算係數未校準時（lbm_to_ms=1.0 哨兵值），
    #   Lawson/Beaufort 分級恆綠而毫無物理意義。畫面照畫（行為不變），但不再靜默誤導。
    if session.viz_mode == "comfort":
        _cal_warn = comfort_scale_warning(session.config["physics"]["lbm_to_ms"])
        if _cal_warn is not None:
            warnings.append(_cal_warn)
    phase_label = _PHASE_LABEL.get(session.sim_phase)

    if phase_label and session.sim_phase == SimPhase.MEAN_CONVERGING:
        mean_steps = getattr(session.backend, "_mean_steps", None)
        if mean_steps is not None:
            phase_label = f"{phase_label} (Step: {mean_steps})"
        if hasattr(session.backend, "_auto_stop") and not session.backend._auto_stop:
            phase_label = f"{phase_label} [FREE-RUN]"

    # Get mouse coordinates
    mx, my = window.get_cursor_pos()
    W = session.config["resolution"]["W"]
    H = session.config["resolution"]["H"]
    px, py = int(mx * W), int(my * H)
    mouse_str = f"Mouse: ({mx:.3f}, {my:.3f}) | Pixel: ({px}, {py})"

    if not warnings and phase_label is None and not mouse_str:
        return
    n_lines = len(warnings) + (1 if phase_label else 0) + 1  # +1 for mouse_str
    height = min(0.30, 0.05 + 0.035 * n_lines)
    gui = window.get_gui()
    with gui.sub_window("Status", 0.55, 0.02, 0.43, height):
        if phase_label:
            gui.text(phase_label)
        gui.text(mouse_str)
        for w in warnings:
            # Remove non-ASCII characters (like Chinese) for Taichi GUI compatibility
            w_ascii = re.sub(r"[^\x00-\x7F]+", "", w).strip()
            gui.text(f"WARN: {w_ascii}")


def _draw_inlet_fix_slider(window, session) -> None:
    """底部左側小面板：拖曳調整 inlet 固定填補寬度（-/= 鍵的 GUI 版本，更直覺）。

    純顯示層參數，見 window._border_fix()；即時寫回 session.inlet_fix_width，
    下一幀渲染立即生效，不影響 backend/warm chain 狀態。

    位置刻意避開右下 Status 面板（x 0.55+）與頂部時間軸（y 0.9-1.0，見
    input_handler.handle_mouse_timeline）；但本專案的 handle_mouse_geometry()
    未對 GUI sub_window 區域做輸入排除，在滑桿上拖曳理論上仍可能同時命中底下的
    建物拖曳判定（既有 Status 面板本身無此問題是因為它純文字、無需點擊）。
    """
    gui = window.get_gui()
    with gui.sub_window("Inlet Fix", 0.02, 0.02, 0.32, 0.08):
        val = gui.slider_float(
            "col width (px)", float(session.inlet_fix_width), 0.0, 80.0
        )
        session.inlet_fix_width = int(round(val))


def _apply_geometry_update(
    session,
    rasterizer,
    gpu_fields,
    polygons: list,
    wind_angle: float,
    wind_level: int,
    mode: str,
    action: str = "analyze",
) -> bool:
    """就地更新光柵化幾何與模型後端推理引擎，回傳 True 代表成功。"""
    import torch
    from lbmnca.bridge.geometry.gvec_bridge import GVecUnavailableError, get_gvec
    from lbmnca.bridge.geometry.regime_features import compute_cond_raw
    from lbmnca.bridge.core.static_builder import build_static_raw
    from lbmnca.bridge.integration.model_backend import GeometryInputs
    from lbmnca.bridge.core.session import SimPhase

    use_warm = mode in ("warm", "adapt") and action != "cold_start"

    # ★polygons 進來就已是 canonical 方向（風沿 +x）——旋轉由
    #   geometry_ingest.ingest_analyze 在 world 座標系一次完成。此處再轉一次會是
    #   雙重旋轉（座標契約見 src/geometry/angle_transform.py 模組 docstring）。
    session.wind_angle = wind_angle

    # rasterize -> mask -> sync to Taichi
    edges_np, n_edges = _polygons_to_edges(polygons)
    rasterizer.rasterize(edges_np, n_edges)
    mask_np = rasterizer.get_mask_numpy()
    gpu_fields.mask.from_numpy(mask_np)

    # build model-agnostic GeometryInputs
    session.warnings = []
    session._steady_saved = False
    if session.backend is not None:
        device = getattr(session.backend, "device", "cuda")
        static_raw = build_static_raw(mask_np)  # [4,H,W]
        allow_fb = bool(
            (session.config.get("geometry") or {}).get("allow_gvec_fallback", False)
        )
        try:
            g_vec = get_gvec(mask_np, allow_fallback=allow_fb)
        except GVecUnavailableError as e:
            # UI-09：不可 silent 用錯條件推論 → 中止本次分析並顯示錯誤
            session.warnings = [f"G_vec 不可用，分析中止：{e}"]
            session.sim_phase = SimPhase.IDLE
            print(f"\033[91m[window] FAIL {session.warnings[0]}\033[0m")
            return False
        cond_raw = compute_cond_raw(
            mask_np, wind_level, wind_angle, session.config, g_vec=g_vec
        )
        # 訓練分布守門警示（軟性，不擋推論）
        session.warnings = list(cond_raw.get("warnings") or [])
        for w in session.warnings:
            print(f"\033[91m[window] WARN {w}\033[0m")
        geo = GeometryInputs(
            mask=torch.from_numpy(mask_np).to(device),
            static_raw=torch.from_numpy(static_raw).to(device),
            cond_raw=cond_raw,
            H=mask_np.shape[0],
            W=mask_np.shape[1],
        )
        session.backend.prepare(geo, warm=use_warm)

    if session.trajectory is not None:
        session.trajectory.reset("warm" if use_warm else "cold")
        session.generation_head = 0
        session.total_frames = 0
    if session.guard is not None:
        session.guard.reset()

    # decompose backend → 兩階段 phase；其餘維持 cold/warm
    if getattr(session.backend, "phase", None) is not None:
        session.sim_phase = SimPhase.MEAN_CONVERGING
    else:
        session.sim_phase = SimPhase.WARM_START if use_warm else SimPhase.COLD_START
    session.re_level = wind_level
    return True


def _process_command(cmd, session, rasterizer, gpu_fields, playback) -> None:
    """處理單一 WS 命令。

    Args:
        cmd: Command 物件，含 action(str) 與 payload(dict) 欄位。
        session: NCA session 物件（就地修改）。
        rasterizer: MaskRasterizer 實例。
        gpu_fields: GPU field 集合物件。
        playback: PlaybackState 物件（就地修改）。
    """
    import torch

    from lbmnca.bridge.geometry.gvec_bridge import GVecUnavailableError, get_gvec
    from lbmnca.bridge.geometry.regime_features import compute_cond_raw
    from lbmnca.bridge.core.static_builder import build_static_raw
    from lbmnca.bridge.integration.model_backend import GeometryInputs
    from lbmnca.bridge.ggui.playback import PlaybackMode
    from lbmnca.bridge.core.session import SimPhase

    action = cmd.action
    payload = cmd.payload

    # zarr case 直餵（C0-3 chain 驗證）：用 zarr 真實 mask/g_vec/p_vec，
    # root 注入 mean_flow_post_conv 當 cold-start IC（真實場、非零），children warm 接鏈。
    if action == "analyze_zarr":
        import os
        import cv2
        import zarr

        H = session.config["resolution"]["H"]
        W = session.config["resolution"]["W"]
        ds = payload["dataset"]
        cid = payload["case_id"]
        inject = bool(payload.get("inject_ic", False))
        wind_level = int(payload.get("wind_level", 2))
        session.wind_angle = 0.0

        # Resolve path relative to this file to be robust against different working directories
        # ★v4 佈局：dataset_zarr_v4/{ds}_unified.zarr（單檔格式 tes_unified_v1，graph 內嵌），
        #   node 條件向量走 02 端 GraphIndex；v3 的 dataset_zarr/{ds}.zarr 已刪，不再相容。
        ggui_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.abspath(os.path.join(ggui_dir, "../../.."))
        zarr_dir = os.path.join(workspace_root, "dataset_zarr_v4")
        zarr_path = os.path.join(zarr_dir, f"{ds}_unified.zarr")

        # 嘗試從 cases_json 載入 buildings 並轉換為 active_polygons，供 GGUI 視窗滑鼠拖曳
        try:
            import json

            cases_json_path = os.path.join(zarr_dir, f"{ds}_cases.json")
            if os.path.exists(cases_json_path):
                with open(cases_json_path, "r", encoding="utf-8") as f:
                    cases_data = json.load(f)
                node = None
                for n in cases_data.get("nodes", []):
                    if n["case_id"] == cid:
                        node = n
                        break
                if node and "buildings" in node:
                    from lbmnca.bridge.core.geometry_ingest import _rect_to_polygon

                    polygons = []
                    for b in node["buildings"]:
                        if isinstance(b, dict):
                            polygons.append(_rect_to_polygon(b))
                        elif isinstance(b, list) and len(b) == 4:
                            b_dict = {"cx": b[0], "cy": b[1], "w": b[2], "h": b[3]}
                            polygons.append(_rect_to_polygon(b_dict))
                    session.active_polygons = polygons
            else:
                session.active_polygons = None
        except Exception as e:
            session.active_polygons = None
            print(
                f"\033[91m[window] Warning: Could not load buildings for dragging from cases JSON: {e}\033[0m"
            )

        try:
            # v4：nodes/{id}/{mean_flow, static}
            z_root = zarr.open(zarr_path, mode="r")
            z_cases = z_root["nodes"]
            if cid in z_cases:
                zc = z_cases[cid]
            else:
                # Fuzzy match (prefix matching) for suffixes like _re404_cold
                matched_key = None
                for key in z_cases.keys():
                    if key.startswith(cid):
                        matched_key = key
                        break
                if matched_key:
                    cid = matched_key
                    zc = z_cases[matched_key]
                else:
                    raise KeyError(cid)
        except Exception as e:  # noqa: BLE001 — 載入失敗即中止本次（不 silent）
            session.warnings = [f"zarr case 載入失敗：{ds}/{cid}：{e}"]
            session.sim_phase = SimPhase.IDLE
            print(
                f"\033[91m[window] FAIL {session.warnings[0]} (path: {zarr_path})\033[0m"
            )
            return

        mask_id = dict(zc.attrs).get("mask_id", cid)
        s0 = zc["static"][0]
        fluid_frac = float((s0 > 0).mean())
        img_path = os.path.join(
            workspace_root, "dataset", ds, "masks", f"{mask_id}.png"
        )
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            mask_np = (s0 > 0).astype(np.float32)  # fallback：近似自 SDF_norm
        else:
            if img.shape != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
            a = (img > 127).astype(np.float32)
            # 極性自動對齊（PNG solid/fluid 慣例未知）→ 取 fluid_frac 較接近者
            mask_np = (
                a
                if abs(a.mean() - fluid_frac) <= abs((1.0 - a).mean() - fluid_frac)
                else (1.0 - a)
            )
        gpu_fields.mask.from_numpy(mask_np)

        session.warnings = []
        session._steady_saved = False
        if session.backend is not None:
            device = getattr(session.backend, "device", "cuda")
            static_raw = build_static_raw(mask_np)
            # v4 的 g_vec / p_vec 存在 graph（單檔格式內嵌），不在 node array 裡 →
            # 走 02 端 GraphIndex（與訓練/評估同一個讀取實作）。
            from lbmnca.nca.tes.data.graph_index import GraphIndex

            graph = GraphIndex.for_zarr(zarr_path)
            g_vec = torch.from_numpy(np.asarray(graph.g_vec(cid), dtype=np.float32))
            p_gt = [float(v) for v in graph.p_vec(cid)]  # [Re, U_inlet_lu, L_char_px]

            cond_raw = compute_cond_raw(
                mask_np, wind_level, 0.0, session.config, g_vec=g_vec
            )
            # ★analyze_zarr 是「重播某一個已模擬的 GT case」，不是使用者畫的新幾何：
            #   條件必須用該 case 的真值，而非 wind_level 離散化的 Re 與 mask 量測的 L_char，
            #   否則注入的 GT mean_flow 與送進模型的條件不是同一個 case。
            #   payload 的 wind_level 對本 action 因此只影響顯示，不影響推論。
            cond_raw["Re"], cond_raw["U_inlet"] = p_gt[0], p_gt[1]
            cond_raw["u_inlet"], cond_raw["L_char"] = p_gt[1], p_gt[2]
            cond_raw["p_vec"] = p_gt
            print(
                f"[window] analyze_zarr {ds}/{cid} 用 GT 條件 p_vec={p_gt}"
                f"（wind_level={wind_level} 僅供顯示）"
            )
            # L_char 已換成 GT 值 → 原本針對「mask 量測值」發的守門警示要重算，
            # 否則警示描述的是一個沒有被送進模型的數字。
            from lbmnca.bridge.geometry.regime_features import L_CHAR_TRAIN_RANGE

            warns = [w for w in (cond_raw.get("warnings") or []) if "L_char" not in w]
            if not (L_CHAR_TRAIN_RANGE[0] <= p_gt[2] <= L_CHAR_TRAIN_RANGE[1]):
                warns.append(
                    f"L_char={p_gt[2]:.0f}px（GT）超出訓練分布 {L_CHAR_TRAIN_RANGE}"
                )
            cond_raw["warnings"] = warns
            session.warnings = list(warns)
            ic = None
            if inject:
                ic = torch.from_numpy(
                    np.asarray(zc["mean_flow"][:], dtype=np.float32)
                ).to(device)
            geo = GeometryInputs(
                mask=torch.from_numpy(mask_np).to(device),
                static_raw=torch.from_numpy(static_raw).to(device),
                cond_raw=cond_raw,
                H=H,
                W=W,
            )
            try:
                session.backend.prepare(
                    geo, warm=True, ic_field=ic
                )  # decompose 支援 ic_field
            except TypeError:
                session.backend.prepare(geo, warm=True)  # 其他 backend 無此參數
        if session.trajectory is not None:
            session.trajectory.reset("warm")
            session.generation_head = 0
            session.total_frames = 0
        if session.guard is not None:
            session.guard.reset()
        session.sim_phase = (
            SimPhase.MEAN_CONVERGING
            if getattr(session.backend, "phase", None) is not None
            else SimPhase.WARM_START
        )
        session.re_level = wind_level
        print(
            f"[window] analyze_zarr {ds}/{cid} inject_ic={inject} fluid={mask_np.mean():.3f}"
        )
        return

    # "analyze" (slim protocol) + legacy aliases cold_start/warm_start
    if action in ("analyze", "cold_start", "warm_start"):
        polygons = payload.get("polygons", [])
        wind_angle = float(payload.get("wind_angle", 0.0))
        wind_level = int(payload.get("wind_level", 2))
        mode = payload.get("mode", "fresh")
        use_warm = mode in ("warm", "adapt") and action != "cold_start"

        # ★polygons 進來就已是 canonical 方向（風沿 +x），見 _apply_geometry_update 同段註解。
        session.wind_angle = wind_angle

        # rasterize -> mask -> sync to Taichi
        edges_np, n_edges = _polygons_to_edges(polygons)
        rasterizer.rasterize(edges_np, n_edges)
        mask_np = rasterizer.get_mask_numpy()
        gpu_fields.mask.from_numpy(mask_np)

        # build model-agnostic GeometryInputs
        session.warnings = []
        session._steady_saved = False
        if session.backend is not None:
            device = getattr(session.backend, "device", "cuda")
            static_raw = build_static_raw(mask_np)  # [4,H,W]
            allow_fb = bool(
                (session.config.get("geometry") or {}).get("allow_gvec_fallback", False)
            )
            try:
                g_vec = get_gvec(mask_np, allow_fallback=allow_fb)
            except GVecUnavailableError as e:
                # UI-09：不可 silent 用錯條件推論 → 中止本次分析並顯示錯誤
                session.warnings = [f"G_vec 不可用，分析中止：{e}"]
                session.sim_phase = SimPhase.IDLE
                print(f"\033[91m[window] FAIL {session.warnings[0]}\033[0m")
                return
            cond_raw = compute_cond_raw(
                mask_np, wind_level, wind_angle, session.config, g_vec=g_vec
            )
            # 訓練分布守門警示（軟性，不擋推論）
            session.warnings = list(cond_raw.get("warnings") or [])
            for w in session.warnings:
                print(f"\033[91m[window] WARN {w}\033[0m")
            geo = GeometryInputs(
                mask=torch.from_numpy(mask_np).to(device),
                static_raw=torch.from_numpy(static_raw).to(device),
                cond_raw=cond_raw,
                H=mask_np.shape[0],
                W=mask_np.shape[1],
            )
            session.backend.prepare(geo, warm=use_warm)

        if session.trajectory is not None:
            session.trajectory.reset("warm" if use_warm else "cold")
            session.generation_head = 0
            session.total_frames = 0
        if session.guard is not None:
            session.guard.reset()

        # decompose backend → 兩階段 phase；其餘維持 cold/warm
        if getattr(session.backend, "phase", None) is not None:
            session.sim_phase = SimPhase.MEAN_CONVERGING
        else:
            session.sim_phase = SimPhase.WARM_START if use_warm else SimPhase.COLD_START
        session.re_level = wind_level
        playback.mode = PlaybackMode.LIVE
        print(
            f"[window] {action}: mode={mode} wind_angle={wind_angle:.1f}° level={wind_level}"
        )

    elif action == "seek":
        frame = int(payload.get("frame", 0))
        if session.trajectory is not None:
            frame = max(0, min(frame, session.trajectory.n_frames - 1))
        playback.playback_head = float(frame)
        from lbmnca.bridge.ggui.playback import PlaybackMode

        playback.mode = PlaybackMode.SEEK
        session.playback_mode = "seek"

    elif action == "set_viz_mode":
        mode = payload.get("viz_mode", "velocity")
        session.viz_mode = mode

    elif action == "set_auto_stop":
        # free-run 開關：關閉停步守門讓模型持續推進（觀察固定點之後的長時行為）。
        enabled = bool(payload["enabled"])
        if hasattr(session.backend, "set_auto_stop"):
            session.backend.set_auto_stop(enabled)
            if not enabled and session.sim_phase == SimPhase.COMPLETED:
                session.sim_phase = SimPhase.MEAN_CONVERGING  # 解凍後恢復推進
        else:
            print(
                f"[window] WARN backend 不支援 set_auto_stop（backend="
                f"{type(session.backend).__name__}）"
            )

    elif action == "set_speed":
        speed = float(payload.get("speed", 1.0))
        playback.playback_speed = max(0.1, min(speed, 2.0))
        session.playback_speed = playback.playback_speed
        from lbmnca.bridge.ggui.playback import PlaybackMode

        if speed < 0.99:
            playback.mode = PlaybackMode.SLOW_MO
            session.playback_mode = "slow_mo"

    elif action == "save_version":
        label = payload.get("label", "")
        if hasattr(session, "save_version"):
            idx = session.save_version(label)
            print(f"[window] 版本已儲存: {idx} — {label}")


def _polygons_to_edges(polygons: list) -> tuple:
    """將多邊形頂點列表轉換為 MaskRasterizer 需要的 edges numpy 格式。

    Args:
        polygons: 多邊形列表，每個多邊形為 [[x, y], ...] 格式（歸一化座標）。

    Returns:
        (edges_np, n_edges) — edges_np shape [MAX_EDGES, 2, 2] float32，
        n_edges 為實際邊數。
    """
    MAX_EDGES = 2000
    edges = np.zeros((MAX_EDGES, 2, 2), dtype=np.float32)
    n = 0
    for poly in polygons:
        if len(poly) < 3:
            continue
        for i in range(len(poly)):
            if n >= MAX_EDGES:
                break
            v0 = poly[i]
            v1 = poly[(i + 1) % len(poly)]
            edges[n, 0, 0] = float(v0[0])
            edges[n, 0, 1] = float(v0[1])
            edges[n, 1, 0] = float(v1[0])
            edges[n, 1, 1] = float(v1[1])
            n += 1
    return edges, n


def _step_nca_if_running(session, gpu_fields, playback) -> None:
    """若 sim_phase == COLD_START/WARM_START，執行一步 NCA 並推入 trajectory。

    Args:
        session: NCA session 物件（就地修改 generation_head、total_frames、sim_phase）。
        gpu_fields: GPU field 集合物件（未直接使用，保留供未來擴充）。
        playback: PlaybackState 物件（LIVE 模式時跟隨最新幀）。
    """
    from lbmnca.bridge.core.session import SimPhase
    from lbmnca.bridge.ggui.playback import PlaybackMode

    _running_phases = (
        SimPhase.COLD_START,
        SimPhase.WARM_START,
        SimPhase.MEAN_CONVERGING,
        SimPhase.TURB_LIVE,
    )
    if session.sim_phase not in _running_phases:
        return
    if session.backend is None or session.trajectory is None:
        return

    # 幀上限：cold/warm 維持既有快照數；MEAN_CONVERGING 由 adapter 內部
    # mean_max_steps 控制、TURB_LIVE 無上限（UI-04，滑窗緩衝承接）
    if session.sim_phase in (SimPhase.COLD_START, SimPhase.WARM_START):
        max_frames = (
            session.trajectory.MAX_FRAMES_WARM
            if session.sim_phase == SimPhase.WARM_START
            else session.trajectory.MAX_FRAMES_COLD
        )
        if session.generation_head >= max_frames:
            session.sim_phase = SimPhase.COMPLETED
            if hasattr(session.backend, "save_steady"):
                session.backend.save_steady()
            print(f"[window] simulation complete: {session.generation_head} frames")
            return

    # one backend step
    session.backend.step(1)
    phy = session.backend.phy_field()  # [9,H,W] CUDA, raw physical
    conv = session.backend.convergence_field()  # [1,H,W] or None

    # ConvergenceGuard: trip -> stop stepping, never push NaN
    if session.guard is not None and not session.guard.check(
        session.backend, phy, conv
    ):
        session.sim_phase = SimPhase.COMPLETED
        session.tier = max(session.tier, 1)
        print(f"[window] guard tripped ({session.guard.reason}) -> tier {session.tier}")
        return

    session.trajectory.push(phy)
    if conv is not None:
        gpu_fields.convergence.from_torch(conv[0].contiguous())
    session.generation_head += 1
    session.total_frames = session.trajectory.n_frames

    # decompose backend：phase 由 adapter 內部狀態機推進（mean 收斂 → turb），
    # session.sim_phase 跟隨同步（驅動 overlay/UI 顯示）
    backend_phase = getattr(session.backend, "phase", None)
    if backend_phase is not None:
        if backend_phase == "turb":
            session.sim_phase = SimPhase.TURB_LIVE
        else:
            if getattr(session.backend, "_mean_done", False):
                session.sim_phase = SimPhase.COMPLETED
            else:
                session.sim_phase = SimPhase.MEAN_CONVERGING

        # 凍結後存 steady（adapt 模式 warm-transfer 素材）
        if (
            backend_phase == "turb" or getattr(session.backend, "_mean_done", False)
        ) and not getattr(session, "_steady_saved", False):
            if hasattr(session.backend, "save_steady"):
                session.backend.save_steady()
            session._steady_saved = True

    if playback.mode == PlaybackMode.LIVE:
        playback.playback_head = float(session.trajectory.n_frames - 1)


def _render_ibfv(
    gpu_fields,
    director,
    ibfv_pass,
    ibfv_params,
    derive_fields,
    compute_vmax,
    colorize_ibfv,
    *,
    seeded: bool,
    H: int,
    W: int,
    prof,
    field_colorizer=None,
    backend=None,
    tex_strength: float = 0.5,
    tex_vel_knee: float = 0.35,
    vis_vel_max: float | None = None,
    inlet_fix_width: int = 10,
) -> None:
    """IBFV mean 層渲染（smooth_transition Phase A，每幀 animated 推進）。

    資料流：director.advance() → director.current() 取內插 mean moments →
    from_torch 零拷貝入 physics → derive_fields 得驅動速度 → ibfv_pass.render()
    平流一步 → colorize_ibfv 著色至 rgb。序列為空時填底色。

    Args:
        gpu_fields: GPU field 集合物件。
        director: PlaybackDirector（mean 收斂序列）。
        ibfv_pass: mean 層 IBFVPass 實例。
        ibfv_params: mean 層 IBFVParams。
        derive_fields: field_ops.derive_fields kernel。
        compute_vmax: field_ops.compute_vmax kernel。
        colorize_ibfv: colorize.colorize_ibfv kernel。
        seeded: IBFV 紋理是否已對當前幾何 seed（False 則本幀重 seed）。
        H: 場高。
        W: 場寬。
        prof: profiler。
        inlet_fix_width: inlet 邊界固定填補寬度（px），見 window._border_fix()。
    """
    director.advance(1.0)
    frame = director.current()  # 內插 mean moments [9,H,W] 或 None
    if frame is None and backend is not None:
        # director 空（剛清/連續幾何串流）→ 直接取 backend 當前 mean，避免黑幀
        mf = getattr(backend, "mean_field", None)
        if callable(mf):
            frame = mf()
    if frame is None:
        gpu_fields.rgb.fill(0.1)
        return
    frame = _border_fix(frame, inlet_fix_width)  # 邊界偽影修正（inlet 邊界亮線）

    with prof.stage("dlpack"):
        gpu_fields.physics.from_torch(frame.contiguous())
    with prof.stage("derive"):
        derive_fields(
            gpu_fields.physics,
            gpu_fields.vel_x,
            gpu_fields.vel_y,
            gpu_fields.vel_mag,
            gpu_fields.pressure,
        )
    # 顯示尺度：優先 vis_range SSOT 固定值（與主渲染路徑同尺度、不閃）。
    # 取不到（無 FieldNormalizer 的測試 backend）才退回逐幀自適應 vmax。
    if vis_vel_max is not None:
        vmax_val = vis_vel_max + 1e-8
    else:
        with prof.stage("vmax"):
            compute_vmax(gpu_fields.vel_mag, gpu_fields.mask, gpu_fields._vmax)
            vmax_val = float(gpu_fields._vmax[None]) + 1e-8

    with prof.stage("ibfv"):
        if not seeded:
            ibfv_pass.init(gpu_fields.mask, ibfv_params)
        tex = ibfv_pass.render(
            gpu_fields.vel_x,
            gpu_fields.vel_y,
            gpu_fields.mask,
            ibfv_params,
        )
    with prof.stage("colorize"):
        if field_colorizer is not None:
            # 02 FieldColorizer：torch 著色 vel_mag（plasma 固定範圍）+ IBFV 紋理疊流動感
            dev = frame.device
            rho = frame[0].clamp_min(1e-6)
            vmag = torch.sqrt((frame[3] / rho) ** 2 + (frame[5] / rho) ** 2)[
                None
            ]  # [1,H,W] lattice
            mask_t = gpu_fields.mask.to_torch(device=dev)  # [H,W] 0/1
            rgb_t = field_colorizer.getVelColor(
                vmag, mask_t[None], use_fixed_range=True
            )  # [3,H,W]
            tex_t = tex.to_torch(device=dev)  # [H,W] IBFV 流動紋理
            # 零均值 + 速度加權調變：低速背景不疊紋理（乾淨），有流動處才有線條；
            # solid 區 w≈0 → mod≈1，建物灰保持乾淨。
            v_ref = vmax_val  # vis_range SSOT（或測試 backend 的動態值）
            w = (vmag[0] / max(tex_vel_knee * v_ref, 1e-8)).clamp(0.0, 1.0) * mask_t
            mod = 1.0 + tex_strength * (tex_t * 2.0 - 1.0) * w
            rgb_t = (rgb_t * mod[None]).clamp(0.0, 1.0)
            gpu_fields.rgb.from_torch(rgb_t.permute(1, 2, 0).contiguous())
        else:
            colorize_ibfv(
                tex,
                gpu_fields.vel_mag,
                gpu_fields.lut_plasma,
                gpu_fields.mask,
                vmax_val,
                0.7,
                gpu_fields.rgb,
            )


def _border_fix(frame, inlet_width: int = 10):
    """消除邊界 1-px 偽影：最外一圈用內鄰複製；inlet 額外 soft-edge 填補。

    conv / NCA 算子在域邊界無完整鄰居 → 最外圈（尤其 inlet col0）易出尖峰假值，
    vel / vorticity 顯示成邊界亮線（實測 col0 亮度 173 vs 內鄰 144 vs 內部 110）。

    純 render-layer 修正：只動傳給 GGUI 顯示用的複本，不寫回 backend/trajectory，
    不影響 warm chain 狀態（frame 已是 .clone() 起點，模型真實 state 不受影響）。

    inlet（左緣）修正分兩段：
      1. [0, inlet_width) 固定填補：以 col=inlet_width 的值向左複製（近均勻自由流，
         物理影響極小，去除硬邊界的暗條紋/垂直線偽影）。
      2. [inlet_width, inlet_width+feather) soft edge：以 smootherstep 權重（3t²-2t³，
         兩端一階導數為 0）從「固定值」漸變回模型原始輸出。★2026-07-23：原本用線性權重，
         線性在兩端斜率不連續，對渦度（速度的一階空間導數）而言等於在
         col=inlet_width 與 col=inlet_width+feather 兩處產生真實的導數尖峰，肉眼呈一條
         垂直亮線（GT 沒有此 pattern）；換 smootherstep 使一階導數在兩端為 0，消除該尖峰。
         feather 寬度隨 inlet_width 等比例導出（約一半，最少 4px、最多 20px），
         不另開使用者可調參數。

    Args:
        frame: [9, H, W] torch 物理場。
        inlet_width: inlet 固定填補寬度（px），0 = 停用該修正。GGUI 內按 -/= 調整
            （見 src/ggui/input_handler.py、window.py::_handle_kb_command）。
    Returns:
        邊界修正後的複本 [9, H, W]。
    """
    f = frame.clone()
    if inlet_width > 0:
        ref = f[:, :, inlet_width : inlet_width + 1]  # 固定值來源：inlet_width 那一列
        f[:, :, :inlet_width] = ref
        feather = max(4, min(inlet_width // 2, 20))
        feather = min(feather, f.shape[-1] - inlet_width - 1)
        if feather > 0:
            # (0,1) 開區間 smootherstep 權重，兩端不含 0/1 避免與固定段/原始段重疊處產生階梯，
            # 且兩端一階導數為 0（C¹ 連續）→ 渦度（一階空間導數）在接縫處不出現尖峰/亮線。
            t = torch.linspace(0.0, 1.0, feather + 2, device=f.device, dtype=f.dtype)[
                1:-1
            ]
            t = t * t * (3.0 - 2.0 * t)
            t = t.view(1, 1, -1)
            orig_slice = frame[:, :, inlet_width : inlet_width + feather]
            f[:, :, inlet_width : inlet_width + feather] = (
                ref * (1.0 - t) + orig_slice * t
            )
    # 其餘三邊 outlet/top/bottom 各 1 列內鄰複製（去邊界尖峰）
    f[:, 0, :] = f[:, 1, :]
    f[:, -1, :] = f[:, -2, :]
    f[:, :, -1] = f[:, :, -2]
    return f


def _colorize(
    viz_mode: str,
    gpu_fields,
    vmax: float,
    lbm_to_ms: float,
    vor_range: float = None,
    p_center: float = None,
    p_range: float = None,
) -> None:
    """依 viz_mode 呼叫對應 colorize kernel。

    Args:
        viz_mode: 視覺化模式字串，可為 "velocity"/"vorticity"/"pressure"/"streamline"/"comfort"。
        gpu_fields: GPU field 集合物件。
        vmax: 速度量值上界（vis_range SSOT 固定值，或測試 backend 的動態值）。
        lbm_to_ms: LBM 格子速度 → m/s 換算係數。
        vor_range: 渦度對稱範圍（vis_range SSOT）。None → 退回 vmax*5 啟發（測試 backend）。
        p_center / p_range: 壓力 colormap 中心/半寬（vis_range SSOT）。None → 預設 1/3, 0.01。
    """
    from lbmnca.bridge.render.colorize import (
        colorize_velocity,
        colorize_vorticity,
        colorize_pressure,
        colorize_comfort,
    )

    if viz_mode == "velocity":
        colorize_velocity(
            gpu_fields.vel_mag,
            gpu_fields.lut_plasma,
            gpu_fields.mask,
            vmax,
            gpu_fields.rgb,
        )
    elif viz_mode == "vorticity":
        vrange = vor_range if vor_range is not None else vmax * 5.0
        colorize_vorticity(
            gpu_fields.vorticity,
            gpu_fields.lut_vorticity,
            gpu_fields.mask,
            vrange,
            gpu_fields.rgb,
        )
    elif viz_mode == "pressure":
        colorize_pressure(
            gpu_fields.pressure,
            gpu_fields.lut_rdbu,
            gpu_fields.mask,
            p_center=p_center if p_center is not None else 1.0 / 3.0,
            p_range=p_range if p_range is not None else 0.01,
            rgb=gpu_fields.rgb,
        )
    elif viz_mode == "comfort":
        colorize_comfort(
            gpu_fields.vel_mag,
            gpu_fields.mask,
            lbm_to_ms,
            gpu_fields.lut_comfort,
            gpu_fields.rgb,
        )
    elif viz_mode == "streamline":
        # 先畫 velocity 底圖，流線在外面畫
        colorize_velocity(
            gpu_fields.vel_mag,
            gpu_fields.lut_plasma,
            gpu_fields.mask,
            vmax,
            gpu_fields.rgb,
        )
    else:
        colorize_velocity(
            gpu_fields.vel_mag,
            gpu_fields.lut_plasma,
            gpu_fields.mask,
            vmax,
            gpu_fields.rgb,
        )


def _handle_kb_command(cmd: dict, session, playback) -> None:
    """鍵盤命令 → session + playback 狀態更新。

    Args:
        cmd: handle_keyboard() 回傳的命令字典。
        session: NCA session 物件（就地修改）。
        playback: PlaybackState 物件（就地修改）。
    """
    from lbmnca.bridge.ggui.playback import PlaybackMode

    action = cmd.get("action")

    if action == "toggle_pause":
        if playback.mode == PlaybackMode.LIVE:
            playback.mode = PlaybackMode.SEEK
            session.playback_mode = "seek"
        else:
            playback.mode = PlaybackMode.LIVE
            session.playback_mode = "live"

    elif action == "step":
        playback.mode = PlaybackMode.SEEK
        session.playback_mode = "seek"
        n = session.trajectory.n_frames if session.trajectory else 0
        playback.playback_head = max(
            0.0, min(playback.playback_head + cmd.get("delta", 0), float(max(0, n - 1)))
        )

    elif action == "set_mode" and cmd.get("mode") == "live":
        playback.mode = PlaybackMode.LIVE
        session.playback_mode = "live"

    elif action == "reset":
        playback.playback_head = 0.0
        playback.mode = PlaybackMode.SEEK
        session.playback_mode = "seek"

    elif action == "set_viz_mode":
        session.viz_mode = cmd.get("mode", "velocity")

    elif action == "director_rate":
        # IBFV 慢放：以 ×1.5 / ÷1.5 步進調 director rate，clamp [0.05, 4.0]。
        base = session.director_rate if session.director_rate is not None else 1.0
        factor = 1.5 if cmd.get("delta", 0) > 0 else (1.0 / 1.5)
        session.director_rate = max(0.05, min(base * factor, 4.0))
        print(f"[window] director rate → {session.director_rate:.3f}")

    elif action == "toggle_auto_stop":
        # F 鍵：停步守門 on/off（free-run 觀察長時迭代行為）。
        from lbmnca.bridge.core.session import SimPhase

        backend = session.backend
        if hasattr(backend, "set_auto_stop"):
            enabled = not backend._auto_stop
            backend.set_auto_stop(enabled)
            if not enabled and session.sim_phase == SimPhase.COMPLETED:
                session.sim_phase = SimPhase.MEAN_CONVERGING
        else:
            print(
                f"[window] WARN backend 不支援 set_auto_stop（backend="
                f"{type(backend).__name__}）"
            )

    elif action == "inlet_fix_width":
        # inlet 邊界固定值填補寬度，clamp [0, 80]（W=640，80px≈12.5% 已是合理上限）。
        session.inlet_fix_width = max(
            0, min(session.inlet_fix_width + cmd.get("delta", 0), 80)
        )
        print(f"[window] inlet fix width → {session.inlet_fix_width}px")


def _sync_playback_from_session(playback, session) -> None:
    """把 session 的 playback 欄位同步至 local PlaybackState。

    Args:
        playback: PlaybackState 物件（就地修改）。
        session: NCA session 物件（唯讀）。
    """
    from lbmnca.bridge.ggui.playback import PlaybackMode

    mode_map = {
        "live": PlaybackMode.LIVE,
        "seek": PlaybackMode.SEEK,
        "slow_mo": PlaybackMode.SLOW_MO,
    }
    playback.mode = mode_map.get(session.playback_mode, PlaybackMode.LIVE)
    playback.playback_speed = session.playback_speed


def _sync_playback_to_session(playback, session) -> None:
    """把 local PlaybackState 同步回 session。

    Args:
        playback: PlaybackState 物件（唯讀）。
        session: NCA session 物件（就地修改）。
    """
    from lbmnca.bridge.ggui.playback import PlaybackMode

    session.playback_head = playback.playback_head
    session.playback_mode = {
        PlaybackMode.LIVE: "live",
        PlaybackMode.SEEK: "seek",
        PlaybackMode.SLOW_MO: "slow_mo",
    }.get(playback.mode, "live")
