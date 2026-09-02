import taichi as ti
import numpy as np
from lbmnca.bridge.ggui.playback import PlaybackState, PlaybackMode


# 時間軸水平範圍（歸一化）
_TIMELINE_X_START = 0.05
_TIMELINE_X_END = 0.95
_TIMELINE_X_RANGE = _TIMELINE_X_END - _TIMELINE_X_START

# 時間軸垂直位置（歸一化）
_TIMELINE_Y = 0.93

# 背景點數（用於模擬橫條）
_BG_POINT_COUNT = 200


def draw_timeline(canvas: ti.ui.Canvas, state: PlaybackState, gf) -> None:
    """在視窗底部畫時間軸（使用 GPUFields 預配置 field，零每幀配置）。

    元素（歸一化座標）：背景灰點串、白色 playhead、綠色 generation head（LIVE）。

    Args:
        canvas: Taichi GGUI canvas。
        state: 播放狀態。
        gf: GPUFields（提供 tl_bg / tl_ph / tl_gen 預配置 field）。
    """
    if state.total_frames <= 0:
        return

    max_frame = float(state.total_frames - 1)

    # --- 1. 背景橫條（預配置靜態 field，直接畫）---
    canvas.circles(gf.tl_bg, radius=0.003, color=(0.4, 0.4, 0.4))

    # --- 2. Playhead（只更新單點值，不配置 field）---
    pr = max(0.0, min(state.playback_head / max_frame if max_frame > 0 else 0.0, 1.0))
    gf.tl_ph[0] = [_TIMELINE_X_START + pr * _TIMELINE_X_RANGE, _TIMELINE_Y]
    canvas.circles(gf.tl_ph, radius=0.010, color=(1.0, 1.0, 1.0))

    # --- 3. Generation head（LIVE 模式）---
    if state.mode == PlaybackMode.LIVE:
        gr = max(0.0, min(float(state.generation_head) / max_frame if max_frame > 0 else 0.0, 1.0))
        gf.tl_gen[0] = [_TIMELINE_X_START + gr * _TIMELINE_X_RANGE, _TIMELINE_Y]
        canvas.circles(gf.tl_gen, radius=0.007, color=(0.2, 1.0, 0.3))


def draw_status_overlay(canvas: ti.ui.Canvas, mode_str: str, frame_str: str) -> None:
    """左上角狀態資訊（模式、幀數）。

    Phase 2 先留空，Phase 8 整合時再補完整文字渲染。

    Args:
        canvas: Taichi GGUI canvas 物件。
        mode_str: 顯示用模式字串（如 "LIVE"、"SEEK"）。
        frame_str: 顯示用幀數字串（如 "42 / 100"）。
    """
    pass  # Phase 2 先略過，Phase 8 整合時再補
