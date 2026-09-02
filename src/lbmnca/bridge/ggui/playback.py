from enum import Enum
from dataclasses import dataclass, field


class PlaybackMode(Enum):
    LIVE = "live"       # 跟隨最新生成幀
    SEEK = "seek"       # 暫停，手動拖拽時間軸
    SLOW_MO = "slow_mo" # 慢速播放（playback_speed < 1.0）


@dataclass
class PlaybackState:
    mode: PlaybackMode = PlaybackMode.LIVE
    generation_head: int = 0      # NCA 已生成幀數（by simulation layer update）
    playback_head: float = 0.0    # 顯示位置（小數，用於插值）
    playback_speed: float = 1.0   # 播放速度係數（0.1 ~ 1.0）
    total_frames: int = 0         # buffer 中總幀數


def update_playback(state: PlaybackState, dt_sec: float) -> None:
    """每幀呼叫，依 mode 推進 playback_head。

    LIVE: playback_head 跟隨 generation_head（最新幀）
    SEEK: playback_head 不變（等使用者拖曳）
    SLOW_MO: playback_head 以 playback_speed 速率推進，到達 total_frames-1 後停止

    Args:
        state: 目前播放狀態，會就地修改。
        dt_sec: 距上一幀的時間（秒）。
    """
    if state.total_frames <= 0:
        return

    max_frame = float(state.total_frames - 1)

    if state.mode == PlaybackMode.LIVE:
        state.playback_head = min(float(state.generation_head), max_frame)

    elif state.mode == PlaybackMode.SEEK:
        # 不更新；等使用者拖曳
        pass

    elif state.mode == PlaybackMode.SLOW_MO:
        # 60fps base：每秒推進 playback_speed * 60 幀
        state.playback_head += state.playback_speed * 60.0 * dt_sec
        state.playback_head = max(0.0, min(state.playback_head, max_frame))
