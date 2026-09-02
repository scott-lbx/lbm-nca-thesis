"""Thread-safe command queue：WebSocket thread → GGUI main thread。"""
from __future__ import annotations
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Any


@dataclass
class Command:
    """從 GH/WS 傳來的命令（slim 協定）。

    主要 action：
        "analyze"  - 幾何分析（mode=fresh/adapt + 已正規化 polygons + wind_angle + wind_level）
                     由 server geometry_ingest 從 world 幾何轉出
    時間軸 / viz / 速度 / 版本控制改為 GGUI-local（鍵盤/滑鼠），不再經 WS。
    舊 action（cold_start/warm_start/seek/set_viz_mode/set_speed）仍相容處理。
    """

    action: str
    payload: dict[str, Any] = field(default_factory=dict)


class CommandQueue:
    """GGUI 主循環每幀從此取命令。"""

    def __init__(self) -> None:
        self._q: Queue[Command] = Queue()

    def push(self, cmd: Command) -> None:
        """WebSocket handler 呼叫（背景 asyncio 執行緒）。"""
        self._q.put_nowait(cmd)

    def drain(self) -> list[Command]:
        """GGUI 主循環每幀呼叫，取出所有待處理命令。"""
        cmds: list[Command] = []
        while True:
            try:
                cmds.append(self._q.get_nowait())
            except Empty:
                break
        return cmds

    def size(self) -> int:
        return self._q.qsize()
