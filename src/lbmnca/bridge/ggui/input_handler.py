import taichi as ti


def handle_keyboard(window: ti.ui.Window) -> dict:
    """解析鍵盤輸入，回傳命令字典（每幀呼叫）。

    回傳格式：{"action": str, ...} 或 {} 若無輸入

    鍵位對應：
        空白鍵 (space): toggle pause（LIVE ↔ SEEK）
        方向鍵: "arrow_left", "arrow_right", "arrow_up", "arrow_down"
        1~6: set_viz_mode
        L/R: set_mode / reset
        F: toggle auto_stop（停步守門 on/off；off = free-run 持續推進）
        [/]: director_rate
        -/=: inlet_fix_width（inlet 邊界固定值填補寬度，soft edge）
    """
    # 使用 get_event 偵測單次按下（避免重複觸發）
    if window.get_event(ti.ui.PRESS):
        key = window.event.key

        if key == ti.ui.SPACE:
            return {"action": "toggle_pause"}
        elif key == ti.ui.LEFT:
            return {"action": "arrow_left"}
        elif key == ti.ui.RIGHT:
            return {"action": "arrow_right"}
        elif key == ti.ui.UP:
            return {"action": "arrow_up"}
        elif key == ti.ui.DOWN:
            return {"action": "arrow_down"}
        elif key == "1":
            return {"action": "set_viz_mode", "mode": "velocity"}
        elif key == "2":
            return {"action": "set_viz_mode", "mode": "vorticity"}
        elif key == "3":
            return {"action": "set_viz_mode", "mode": "pressure"}
        elif key == "4":
            return {"action": "set_viz_mode", "mode": "streamline"}
        elif key == "5":
            return {"action": "set_viz_mode", "mode": "comfort"}
        elif key == "6":
            return {"action": "set_viz_mode", "mode": "ibfv"}
        elif key in ("l", "L"):
            return {"action": "set_mode", "mode": "live"}
        elif key in ("r", "R"):
            return {"action": "reset"}
        elif key in ("f", "F"):
            return {"action": "toggle_auto_stop"}
        elif key == "[":
            return {"action": "director_rate", "delta": -1}
        elif key == "]":
            return {"action": "director_rate", "delta": +1}
        elif key == "-":
            return {"action": "inlet_fix_width", "delta": -2}
        elif key == "=":
            return {"action": "inlet_fix_width", "delta": +2}

    return {}


def handle_mouse_timeline(window: ti.ui.Window, total_frames: int) -> float | None:
    """偵測時間軸區域左鍵拖曳，回傳 seek 目標 frame index（float）。

    時間軸區域：螢幕頂部 10%（ui_y ∈ [0.9, 1.0] 歸一化坐標，Taichi Y 軸由下往上）
    左鍵按住拖曳：回傳 seek_target = x_norm * total_frames
    其他情況：回傳 None

    Args:
        window: Taichi GGUI 視窗物件。
        total_frames: 目前 buffer 中的總幀數。

    Returns:
        目標幀索引（float），若未在時間軸區域按下左鍵則回傳 None。
    """
    if total_frames <= 0:
        return None

    if window.is_pressed(ti.ui.LMB):
        x_norm, ui_y = window.get_cursor_pos()
        # 時間軸區域：ui_y ∈ [0.9, 1.0] (bottom-up, so this is the top)
        if 0.9 <= ui_y <= 1.0:
            # clamp x_norm 至 [0, 1] 避免超出邊界
            x_clamped = max(0.0, min(x_norm, 1.0))
            seek_target = x_clamped * float(total_frames - 1)
            return seek_target

    return None


def point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    """射線法 (Ray Casting) 判定點 (x, y) 是否在多邊形 poly 內。"""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def handle_mouse_geometry(window: ti.ui.Window, session) -> bool:
    """處理滑鼠點擊拖曳幾何，就地更新 session.active_polygons 並在發生拖曳時回傳 True。"""
    if getattr(session, "active_polygons", None) is None:
        return False

    if window.get_event(ti.ui.PRESS):
        if window.event.key == ti.ui.LMB:
            x, ui_y = window.get_cursor_pos()
            y = ui_y  # Both Taichi UI and our polygons are bottom-up (origin="lower")
            # 避開頂部的時間軸區域 (ui_y ∈ [0.9, 1.0])
            if ui_y < 0.9:
                hit_idx = -1
                for idx in reversed(range(len(session.active_polygons))):
                    poly = session.active_polygons[idx]
                    if point_in_polygon(x, y, poly):
                        hit_idx = idx
                        break
                session.selected_building_idx = hit_idx

    if window.is_pressed(ti.ui.LMB):
        x, ui_y = window.get_cursor_pos()
        y = ui_y  # Both Taichi UI and our polygons are bottom-up
        # 避開頂部的時間軸區域 (ui_y ∈ [0.9, 1.0])
        if ui_y >= 0.9:
            return False

        drag_state = getattr(session, "drag_state", None)
        if drag_state is None:
            # 尚未開始拖曳，尋找滑鼠點中了哪個多邊形
            hit_idx = getattr(session, "selected_building_idx", -1)
            if hit_idx != -1:
                session.drag_state = {"idx": hit_idx, "last_x": x, "last_y": y}
                # 開始拖曳
        else:
            # 正在拖曳中，計算移動的增量 dx, dy
            idx = drag_state["idx"]
            last_x = drag_state["last_x"]
            last_y = drag_state["last_y"]
            dx = x - last_x
            dy = y - last_y

            # 更新選中多邊形的所有點
            poly = session.active_polygons[idx]
            for pt in poly:
                pt[0] = max(0.0, min(1.0, pt[0] + dx))
                pt[1] = max(0.0, min(1.0, pt[1] + dy))

            # 更新 drag_state 中的 last_x, last_y
            session.drag_state["last_x"] = x
            session.drag_state["last_y"] = y
            return True  # 幾何發生了改變
    else:
        # 滑鼠左鍵放開，清空拖曳狀態
        if getattr(session, "drag_state", None) is not None:
            session.drag_state = None

    return False
