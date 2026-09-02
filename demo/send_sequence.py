"""send_sequence.py — headless 固定時間序列 WebSocket 送出器（假冒 Grasshopper）。

讀 demo/demo.config.json 的 sequences.<name>，連 lbmnca bridge 的 WS server，
依每個 event 的絕對時間 t（秒）送出。與網頁前端（demo/index.html + demo/app.js）
讀**同一份 config**、事件內容一致；供 CI / 錄影 / 無頭復現。

協定（server 端分流見 lbmnca.bridge.core.server::_parse_command）：
  analyze  → 頂層扁平 {action, buildings, wind_level, mode}
  control  → payload 包裹 {action, payload:{...}}

用法：
  .venv/Scripts/python.exe demo/send_sequence.py --sequence basic
  .venv/Scripts/python.exe demo/send_sequence.py --sequence d2-1 --port 8765
  --list 列出 config 內可用序列名稱。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

# 預設 config：與本檔同目錄的 demo/demo.config.json
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "demo.config.json"

_CONTROL_ACTIONS = {"set_viz_mode", "seek", "set_speed", "save_version", "analyze_zarr"}
_CONTROL_KEYS = {
    "set_viz_mode": ["viz_mode"],
    "seek": ["frame"],
    "set_speed": ["speed"],
    "save_version": ["label"],
    "analyze_zarr": ["dataset", "case_id", "inject_ic", "wind_level"],
}


def _build_message(ev: dict, default_wind: int, sim_h: int, flip_y: bool) -> dict:
    """把一個 sequence event 轉成 WS 訊息（analyze 扁平 / control 包裹）。

    flip_y=True 時對 analyze 的 buildings 做 cy' = sim_h - cy，與網頁前端一致
    （p5 canvas y 向下、GGUI/Taichi y 向上）。預設 False＝維持既有 headless 行為。
    """
    action = ev["action"]
    if action == "analyze":
        buildings = ev.get("buildings", [])
        if flip_y:
            buildings = [
                {"cx": b["cx"], "cy": sim_h - b["cy"], "w": b["w"], "h": b["h"]}
                for b in buildings
            ]
        return {
            "action": "analyze",
            "buildings": buildings,
            "wind_level": int(ev.get("wind_level", default_wind)),
            "mode": ev.get("mode", "warm"),
        }
    if action in _CONTROL_ACTIONS:
        payload = {k: ev[k] for k in _CONTROL_KEYS[action] if k in ev}
        return {"action": action, "payload": payload}
    raise ValueError(f"未知 sequence action: {action!r}")


def _load(config_path: Path, seq_name: str) -> tuple[dict, int, int]:
    """載入 config，取出指定序列。缺鍵直接 KeyError（禁止靜默 fallback）。"""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    sequences = cfg["sequences"]
    if seq_name not in sequences:
        raise SystemExit(
            f"[send_sequence] 序列 {seq_name!r} 不存在。可用：{', '.join(sequences)}"
        )
    sim_h = int(cfg["sim"]["height"])
    default_wind = next(w["value"] for w in cfg["wind_levels"] if w.get("default") is True)
    return sequences[seq_name], int(default_wind), sim_h


async def run(host: str, port: int, seq: dict, default_wind: int, sim_h: int, flip_y: bool) -> None:
    try:
        from websockets.asyncio.client import connect
    except ImportError:  # 舊版 websockets
        from websockets import connect  # type: ignore

    events = sorted(seq.get("events", []), key=lambda e: float(e.get("t", 0.0)))
    wind = int(seq.get("wind_level", default_wind))
    uri = f"ws://{host}:{port}"
    print(f"[send_sequence] 連線 {uri}（序列 {seq.get('name','?')}，{len(events)} events）")

    async with connect(uri) as ws:
        print("[send_sequence] 已連線")
        t0 = time.monotonic()
        for ev in events:
            target = float(ev.get("t", 0.0))
            dt = target - (time.monotonic() - t0)
            if dt > 0:
                await asyncio.sleep(dt)
            msg = _build_message(ev, wind, sim_h, flip_y)
            await ws.send(json.dumps(msg))
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                status = json.loads(resp).get("status", "?")
            except asyncio.TimeoutError:
                status = "no-ack"
            print(f"[t={target:6.1f}s] → {msg['action']:12s}  ({status})")
        print("[send_sequence] 序列送畢")
        if seq.get("loop"):
            print("[send_sequence] loop=true：重跑")
            await run(host, port, seq, default_wind, sim_h, flip_y)


def main() -> None:
    ap = argparse.ArgumentParser(description="Headless fixed-sequence WS sender")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                    help="demo.config.json 路徑（預設 demo/demo.config.json）")
    ap.add_argument("--sequence", default="basic", help="config.sequences 內的序列名稱")
    ap.add_argument("--list", action="store_true", help="列出可用序列後結束")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--flip-y", action="store_true",
                    help="對 analyze buildings 做 cy'=sim.height-cy（與網頁前端一致）")
    args = ap.parse_args()

    if args.list:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        for k, v in cfg["sequences"].items():
            print(f"  {k:10s}  {v['name']}  ({len(v['events'])} events)")
        return

    seq, default_wind, sim_h = _load(args.config, args.sequence)
    try:
        asyncio.run(run(args.host, args.port, seq, default_wind, sim_h, args.flip_y))
    except KeyboardInterrupt:
        print("\n[send_sequence] 中斷")


if __name__ == "__main__":
    main()
