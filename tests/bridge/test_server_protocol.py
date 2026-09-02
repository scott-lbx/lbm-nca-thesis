"""server._parse_command：analyze 走 ingest、其餘 action 原樣轉送。

不起 WS server、不建 backend——只驗協定分流。
"""
from __future__ import annotations

import math

import pytest

from lbmnca.bridge.core.server import _parse_command


def rot(x, y, deg):
    a = math.radians(deg)
    return [x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)]


def rect_world(cx, cy, w, h, deg=0.0):
    return [rot(cx - w / 2, cy - h / 2, deg), rot(cx + w / 2, cy - h / 2, deg),
            rot(cx + w / 2, cy + h / 2, deg), rot(cx - w / 2, cy + h / 2, deg)]


def test_missing_action_returns_none():
    assert _parse_command({}) is None
    assert _parse_command({"payload": {"x": 1}}) is None


def test_analyze_web_dialect_goes_through_ingest():
    cmd = _parse_command({
        "action": "analyze",
        "buildings": [{"cx": 960, "cy": 540, "w": 100, "h": 60}],
        "wind_level": 3, "mode": "warm",
    })
    assert cmd.action == "analyze"
    assert set(cmd.payload) == {"polygons", "wind_angle", "wind_level", "mode"}
    assert cmd.payload["wind_angle"] == 0.0
    assert cmd.payload["wind_level"] == 3


def test_analyze_gh_dialect_extracts_wind_angle():
    cmd = _parse_command({
        "action": "analyze", "mode": "fresh", "wind_level": 2,
        "boundary_world": rect_world(96, 54, 192, 108, 33.0),
        "buildings_world": [rect_world(60, 40, 10, 10, 33.0)],
    })
    assert cmd.payload["wind_angle"] == pytest.approx(33.0, abs=1e-6)
    assert len(cmd.payload["polygons"]) == 1


def test_analyze_reads_top_level_not_payload():
    """analyze 讀頂層欄位；誤包進 payload 的內容不得被採用。"""
    cmd = _parse_command({
        "action": "analyze",
        "payload": {"buildings": [{"cx": 1, "cy": 1, "w": 1, "h": 1}]},
    })
    assert cmd.payload["polygons"] == []


@pytest.mark.parametrize("action,payload", [
    ("set_viz_mode", {"viz_mode": "velocity"}),
    ("seek", {"frame": 12}),
    ("set_speed", {"speed": 0.3}),
    ("save_version", {"label": "x"}),
    ("analyze_zarr", {"dataset": "D", "case_id": "C", "inject_ic": True}),
])
def test_control_actions_pass_payload_through(action, payload):
    cmd = _parse_command({"action": action, "payload": payload})
    assert cmd.action == action
    assert cmd.payload == payload


def test_control_without_payload_falls_back_to_whole_message():
    """相容分支：沒有 payload 鍵時整包當 payload。"""
    msg = {"action": "set_viz_mode", "viz_mode": "ibfv"}
    cmd = _parse_command(msg)
    assert cmd.payload == msg


def test_unknown_action_still_forwarded():
    """未知 action 不在此擋下（GGUI 端才決定要不要處理），但必須有 Command。"""
    cmd = _parse_command({"action": "future_feature", "payload": {}})
    assert cmd.action == "future_feature"
