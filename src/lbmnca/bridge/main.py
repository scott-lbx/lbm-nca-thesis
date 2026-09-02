"""
main.py — lbmnca.bridge entrypoint.

Startup order:
1. load config.yaml
2. ti.init(arch=ti.cuda)
3. GPUFields (Taichi field allocation)
4. build ModelBackend (config.model.backend: dummy|gcnca|tes|fieldstore)
5. TrajectoryBuffer + CommandQueue
6. WebSocket server (daemon thread)
7. enter GGUI main loop (main thread, blocks until window closes)

Usage:
    .venv\\Scripts\\python.exe main.py
"""
from __future__ import annotations

import threading

from lbmnca.paths import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "bridge" / "config.yaml"


def main() -> None:
    import yaml
    import taichi as ti

    # ── 1. config ────────────────────────────────────────────────────────
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    H = config["resolution"]["H"]
    W = config["resolution"]["W"]

    # ── 2. Taichi (before any ti.field) ──────────────────────────────────
    ti.init(arch=ti.cuda, log_level=ti.WARN)

    # ── 3. GPU fields ────────────────────────────────────────────────────
    from lbmnca.bridge.render.gpu_fields import GPUFields
    gpu_fields = GPUFields(H, W)
    print(f"[main] GPUFields OK ({H}x{W})")

    # ── 4. model backend ─────────────────────────────────────────────────
    from lbmnca.bridge.integration.adapters import build_backend
    backend = build_backend(config)
    print(f"[main] backend = {config['model']['backend']}")

    # ── 5. trajectory + command queue ────────────────────────────────────
    from lbmnca.bridge.simulation.trajectory import TrajectoryBuffer
    trajectory = TrajectoryBuffer(
        H, W, device=config["model"].get("device", "cuda"),
        # TURB_LIVE 滑窗容量（UI-04；256 幀 ≈ 2.1 GB VRAM）
        capacity=int(config["simulation"].get("live_window_frames", 256)),
    )

    from lbmnca.bridge.core.command_queue import CommandQueue
    cmd_queue = CommandQueue()

    # ── 6. WebSocket server (daemon) ─────────────────────────────────────
    from lbmnca.bridge.core.server import run_ws_server
    ws_thread = threading.Thread(
        target=run_ws_server,
        args=(config["server"]["host"], config["server"]["port"], cmd_queue),
        daemon=True,
    )
    ws_thread.start()
    print(f"[main] WS server ws://{config['server']['host']}:{config['server']['port']}")

    # ── 7. session ───────────────────────────────────────────────────────
    from lbmnca.bridge.core.session import SimulationSession
    session = SimulationSession()
    session.backend = backend
    session.trajectory = trajectory
    session.config = config

    # ── Inject Default Root Case（demo_chain 啟用時改由 GGUI 內 chain driver 推送）──
    chain_cfg = config.get("demo_chain") or {}
    if not chain_cfg.get("enabled"):
        # ★2026-07-13 p5/GH 模式：不注入預設 case（舊 C0-1_Cold_M30C30 已不存在），靜候 WebSocket analyze。
        print("[main] demo_chain disabled — 等待 WebSocket analyze 指令（p5/GH）。")
    else:
        print(f"[main] demo_chain enabled: {len(chain_cfg['case_ids'])} cases，由 GGUI chain driver 推送")

    # ── 8. GGUI main loop (blocks) ───────────────────────────────────────
    from lbmnca.bridge.ggui.window import run_ggui
    print("[main] starting GGUI...")
    run_ggui(config, session, cmd_queue, gpu_fields)

    print("[main] window closed.")


if __name__ == "__main__":
    main()
