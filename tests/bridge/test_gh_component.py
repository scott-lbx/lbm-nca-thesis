"""gh/analyze_component_standalone.py：以假 Rhino 物件全文 exec，不需 Rhino、不需 server。

覆蓋：幾何抽取、長寬比守門（旋轉不得誤報）、payload 方言、_MiniWS RFC6455 framing、
以及 run=True 的送出流程（注入假 websocket 模組攔截）。
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import pytest

GH_SCRIPT = (
    Path(__file__).resolve().parents[2] / "demo" / "gh" / "analyze_component_standalone.py"
)
SRC = GH_SCRIPT.read_text(encoding="utf-8")


# ── Rhino 替身 ───────────────────────────────────────────────────────────────

class _Pt:
    def __init__(self, x, y):
        self.X, self.Y = float(x), float(y)


class _FakeCurve:
    """只實作腳本用到的 TryGetPolyline。"""

    def __init__(self, pts):
        self._pl = [_Pt(x, y) for x, y in pts]

    def TryGetPolyline(self):
        return True, self._pl


class _FakeRect(_FakeCurve):
    """Rectangle3d 路徑（有 .Corner）。"""

    def Corner(self, i):
        return self._pl[i]


class _FakeDataTree:
    """GH DataTree 路徑（有 .Branches）。"""

    def __init__(self, branches):
        self.Branches = branches


def rot(x, y, deg):
    a = math.radians(deg)
    return (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a))


def rect(cx, cy, w, h, deg=0.0):
    return [rot(cx - w / 2, cy - h / 2, deg), rot(cx + w / 2, cy - h / 2, deg),
            rot(cx + w / 2, cy + h / 2, deg), rot(cx - w / 2, cy + h / 2, deg)]


def run_component(**inputs):
    """執行元件全文，回傳其全域命名空間（含 status / payload）。"""
    g = {"__name__": "__gh_component__"}
    g.update(inputs)
    exec(compile(SRC, str(GH_SCRIPT), "exec"), g)
    return g


BASE = dict(run=False, wind_level=2, mode="fresh", viz_mode="",
            url="ws://localhost:8765")


def make_scene(deg=0.0, bw=192.0, bh=108.0):
    return _FakeRect(rect(bw / 2, bh / 2, bw, bh, deg)), [
        _FakeCurve(rect(60.0, 54.0, 8.0, 12.0, deg)),
        _FakeCurve(rect(105.0, 66.0, 6.0, 10.0, deg)),
    ]


# ── 幾何抽取 ─────────────────────────────────────────────────────────────────

def test_preview_builds_world_dialect_payload():
    bnd, blds = make_scene()
    g = run_component(**BASE, boundary=bnd, buildings=blds)
    assert "ready" in g["status"]
    p = json.loads(g["payload"])
    assert p["action"] == "analyze"
    assert p["mode"] == "fresh"
    assert len(p["boundary_world"]) == 4
    assert len(p["buildings_world"]) == 2
    assert "buildings" not in p, "GH 必須送 world 多邊形方言，不是 web 扁平方言"


def test_datatree_input_flattened():
    bnd, blds = make_scene()
    g = run_component(**BASE, boundary=bnd, buildings=_FakeDataTree([blds]))
    assert len(json.loads(g["payload"])["buildings_world"]) == 2


def test_single_curve_input_accepted():
    bnd, blds = make_scene()
    g = run_component(**BASE, boundary=bnd, buildings=blds[0])
    assert len(json.loads(g["payload"])["buildings_world"]) == 1


def test_degenerate_building_dropped():
    bnd, blds = make_scene()
    thin = _FakeCurve([(0.0, 0.0), (1.0, 1.0)])          # 只有 2 點
    g = run_component(**BASE, boundary=bnd, buildings=[thin] + blds)
    assert len(json.loads(g["payload"])["buildings_world"]) == 2


def test_invalid_boundary_blocks_send():
    _, blds = make_scene()
    g = run_component(**BASE, boundary=None, buildings=blds)
    assert "boundary" in g["status"]
    assert g["payload"] == ""


def test_no_valid_buildings_blocks_send():
    bnd, _ = make_scene()
    g = run_component(**BASE, boundary=bnd, buildings=[])
    assert "建物" in g["status"]
    assert g["payload"] == ""


# ── 長寬比守門（★旋轉不得誤報）────────────────────────────────────────────────

@pytest.mark.parametrize("deg", [0.0, 12.5, 30.0, 45.0, 77.0, -25.0])
def test_aspect_guard_silent_for_rotated_16_9(deg):
    """16:9 的 boundary 不論轉幾度都不該出拉伸警示。

    舊版量的是 world AABB：一個正好 16:9 的 boundary 只要一轉，AABB 長寬比就偏離
    16:9 → 每次旋轉都誤報。server 端其實是在 boundary 自身座標系正規化的。
    """
    bnd, blds = make_scene(deg=deg)
    g = run_component(**BASE, boundary=bnd, buildings=blds)
    assert "WARN boundary 長寬比" not in g["status"], g["status"]


@pytest.mark.parametrize("deg", [0.0, 35.0])
def test_aspect_guard_fires_for_genuinely_wrong_aspect(deg):
    """真的不是 16:9（此處 4:3）就要報，且同樣不受旋轉影響。"""
    bnd, blds = make_scene(deg=deg, bw=160.0, bh=120.0)
    g = run_component(**BASE, boundary=bnd, buildings=blds)
    assert "WARN boundary 長寬比" in g["status"]
    assert "1.333" in g["status"]


# ── 輸入正規化 ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad,expect", [(0, 2), (9, 2), (1, 1), (3, 3)])
def test_wind_level_normalized(bad, expect):
    bnd, blds = make_scene()
    g = run_component(**{**BASE, "wind_level": bad}, boundary=bnd, buildings=blds)
    assert json.loads(g["payload"])["wind_level"] == expect


@pytest.mark.parametrize("bad,expect",
                         [("FRESH", "fresh"), ("adapt", "adapt"),
                          ("warm", "warm"), ("nonsense", "fresh")])
def test_mode_normalized(bad, expect):
    bnd, blds = make_scene()
    g = run_component(**{**BASE, "mode": bad}, boundary=bnd, buildings=blds)
    assert json.loads(g["payload"])["mode"] == expect


# ── _MiniWS RFC6455 framing（不開 socket）─────────────────────────────────────

class _FakeSock:
    def __init__(self, handshake=b"HTTP/1.1 101 Switching Protocols\r\n\r\n"):
        self.sent = bytearray()
        self._inbox = bytearray(handshake)

    def sendall(self, b):
        self.sent.extend(b)

    def recv(self, n):
        out = bytes(self._inbox[:n])
        del self._inbox[:n]
        return out

    def settimeout(self, _):
        pass

    def close(self):
        pass


def _mini_ws(monkeypatch, payload_bytes=b""):
    """建一個接上假 socket 的 _MiniWS。"""
    import socket as _socket

    g = {"__name__": "__gh_probe__", "run": False, "boundary": None,
         "buildings": None, "wind_level": 2, "mode": "fresh",
         "viz_mode": "", "url": "ws://localhost:8765"}
    exec(compile(SRC, str(GH_SCRIPT), "exec"), g)
    sock = _FakeSock(b"HTTP/1.1 101 Switching Protocols\r\n\r\n" + payload_bytes)
    monkeypatch.setattr(_socket, "create_connection", lambda *a, **k: sock)
    return g["_MiniWS"]("ws://h:1/p"), sock


@pytest.mark.parametrize("n", [5, 200, 70000])
def test_miniws_frame_encoding(monkeypatch, n):
    """FIN+text opcode、MASK 位元、三段長度編碼、payload 正確 XOR。"""
    ws, sock = _mini_ws(monkeypatch)
    sock.sent.clear()
    text = "x" * n
    ws.send_text(text)
    f = bytes(sock.sent)
    assert f[0] == 0x81                                   # FIN + text
    assert f[1] & 0x80                                    # MASK set
    ln = f[1] & 0x7F
    if n < 126:
        assert ln == n
        off = 2
    elif n < 65536:
        assert ln == 126
        assert struct.unpack(">H", f[2:4])[0] == n
        off = 4
    else:
        assert ln == 127
        assert struct.unpack(">Q", f[2:10])[0] == n
        off = 10
    mask, body = f[off:off + 4], f[off + 4:]
    assert len(body) == n
    assert bytes(body[i] ^ mask[i % 4] for i in range(n)).decode() == text


def test_miniws_recv_unmasked_text(monkeypatch):
    """server→client 幀不帶 mask；recv_text 應正確解碼。"""
    msg = json.dumps({"status": "ok", "msg": "hi"}).encode()
    frame = bytes([0x81, len(msg)]) + msg
    ws, _ = _mini_ws(monkeypatch, payload_bytes=frame)
    assert json.loads(ws.recv_text())["status"] == "ok"


def test_miniws_close_frame_raises(monkeypatch):
    ws, _ = _mini_ws(monkeypatch, payload_bytes=bytes([0x88, 0x00]))
    with pytest.raises(IOError):
        ws.recv_text()


def test_miniws_rejects_failed_handshake(monkeypatch):
    import socket as _socket

    g = {"__name__": "__gh_probe__", "run": False, "boundary": None,
         "buildings": None, "wind_level": 2, "mode": "fresh",
         "viz_mode": "", "url": "ws://localhost:8765"}
    exec(compile(SRC, str(GH_SCRIPT), "exec"), g)
    sock = _FakeSock(b"HTTP/1.1 400 Bad Request\r\n\r\n")
    monkeypatch.setattr(_socket, "create_connection", lambda *a, **k: sock)
    with pytest.raises(IOError):
        g["_MiniWS"]("ws://h:1/p")


@pytest.mark.parametrize("url,expect", [
    ("ws://localhost:8765", ("localhost", 8765, "/")),
    ("ws://1.2.3.4:9/path/x", ("1.2.3.4", 9, "/path/x")),
    ("ws://host", ("host", 80, "/")),
])
def test_parse_ws_url(url, expect):
    g = run_component(**BASE, boundary=None, buildings=None)
    assert g["_parse_ws_url"](url) == expect


def test_wss_rejected_by_fallback():
    g = run_component(**BASE, boundary=None, buildings=None)
    with pytest.raises(ValueError):
        g["_parse_ws_url"]("wss://secure:443")


# ── run=True 送出流程（注入假 websocket 模組）─────────────────────────────────

class _FakeWSConn:
    def __init__(self, sink):
        self._sink = sink

    def send(self, s):
        self._sink.append(s)

    def recv(self):
        return json.dumps({"status": "ok", "msg": "ack"})

    def close(self):
        pass


@pytest.fixture
def fake_websocket(monkeypatch):
    import types

    sink: list[str] = []
    mod = types.ModuleType("websocket")
    mod.create_connection = lambda *a, **k: _FakeWSConn(sink)
    monkeypatch.setitem(sys.modules, "websocket", mod)
    return sink


def test_run_true_sends_analyze_and_reports_ack(fake_websocket):
    bnd, blds = make_scene()
    g = run_component(**{**BASE, "run": True}, boundary=bnd, buildings=blds)
    assert g["status"].startswith("[ok]")
    assert len(fake_websocket) == 1
    assert json.loads(fake_websocket[0])["action"] == "analyze"


def test_viz_mode_sent_as_wrapped_control(fake_websocket):
    """控制指令走 payload 包裹；analyze 走頂層扁平。"""
    bnd, blds = make_scene()
    run_component(**{**BASE, "run": True, "viz_mode": "vorticity"},
                  boundary=bnd, buildings=blds)
    assert len(fake_websocket) == 2
    viz = json.loads(fake_websocket[0])
    assert viz == {"action": "set_viz_mode", "payload": {"viz_mode": "vorticity"}}
    assert "payload" not in json.loads(fake_websocket[1])


def test_unknown_viz_mode_warns_and_is_not_sent(fake_websocket):
    bnd, blds = make_scene()
    g = run_component(**{**BASE, "run": True, "viz_mode": "bogus"},
                      boundary=bnd, buildings=blds)
    assert "WARN 未知 viz_mode" in g["status"]
    assert len(fake_websocket) == 1                       # 只送了 analyze


def test_connection_failure_reported_not_raised(monkeypatch):
    import types

    mod = types.ModuleType("websocket")

    def boom(*a, **k):
        raise OSError("connection refused")

    mod.create_connection = boom
    monkeypatch.setitem(sys.modules, "websocket", mod)
    bnd, blds = make_scene()
    g = run_component(**{**BASE, "run": True}, boundary=bnd, buildings=blds)
    assert "連線失敗" in g["status"]


# ── Rhino 7 / IronPython 相容性 ──────────────────────────────────────────────

def test_no_python3_only_syntax():
    """Rhino 7 GHPython 是 IronPython 2.7：不得有 f-string / `X | None` 註解。"""
    assert 'f"' not in SRC and "f'" not in SRC, "含 f-string，IronPython 2.7 會語法錯誤"
    assert "-> str | None" not in SRC
    assert "from __future__" not in SRC


def test_script_is_self_contained():
    """不得 import 03 repo 任何模組（GH 端沒有 sys.path）。"""
    for bad in ("from src.", "import src.", "websocket_client"):
        assert bad not in SRC, f"不自足：出現 {bad}"
