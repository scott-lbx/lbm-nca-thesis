"""WebSocket 伺服器（背景執行緒，asyncio）。
GH 送來 JSON 命令 → 解析 → 推入 CommandQueue → 回傳確認。

相容 websockets >= 16.0（asyncio-based API）。
handler 簽名只接受 websocket（無 path），使用 websockets.asyncio.server.serve。
"""
from __future__ import annotations
import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lbmnca.bridge.core.command_queue import CommandQueue

logger = logging.getLogger(__name__)


def _parse_command(data: dict) -> "Command | None":
    """解析 JSON → Command（slim 協定：主要 action = "analyze"）。

    "analyze" 帶 world 座標幾何，server 端正規化 + 抽風向（geometry_ingest）。
    其餘 GGUI-local 控制（seek/viz/speed）不再經 WS；若收到則原樣轉送（相容）。
    """
    from lbmnca.bridge.core.command_queue import Command
    from lbmnca.bridge.core.geometry_ingest import ingest_analyze

    action = data.get("action")
    if not action:
        return None

    if action == "analyze":
        return Command(action="analyze", payload=ingest_analyze(data))

    # 相容：舊版 payload-wrapped 或 GGUI-local 控制
    return Command(action=action, payload=data.get("payload", data))


async def _handle_client(websocket, cmd_queue: "CommandQueue") -> None:
    """單一 WebSocket 連線的 handler（websockets 16.x 簽名：只有 websocket）。"""
    logger.info("GH connected: %s", websocket.remote_address)
    try:
        async for raw in websocket:
            try:
                data = json.loads(raw)
                cmd = _parse_command(data)
                if cmd is not None:
                    cmd_queue.push(cmd)
                    msg = "已送出分析" if cmd.action == "analyze" else f"已收到 {cmd.action}"
                    await websocket.send(json.dumps({
                        "status": "ok",
                        "action": cmd.action,
                        "msg": msg,
                    }))
                else:
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "Unknown action",
                    }))
            except json.JSONDecodeError as e:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": f"JSON parse error: {e}",
                }))
    except Exception:
        logger.info("GH disconnected")


def run_ws_server(host: str, port: int, cmd_queue: "CommandQueue") -> None:
    """在背景執行緒中啟動 asyncio WebSocket 伺服器。

    呼叫此函式的執行緒會永遠阻塞（直到主程式結束）。
    建議以 daemon=True 的 threading.Thread 呼叫。

    使用 websockets 16.x asyncio API（websockets.asyncio.server.serve）。
    handler 簽名為單參數（websocket），path 已移除。
    """
    from websockets.asyncio.server import serve

    async def _serve() -> None:
        handler = lambda ws: _handle_client(ws, cmd_queue)
        async with serve(handler, host, port):
            logger.info("WebSocket server listening on ws://%s:%d", host, port)
            await asyncio.Future()  # 永遠不結束

    asyncio.run(_serve())
