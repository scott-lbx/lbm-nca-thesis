# analyze_component_standalone.py
# ─────────────────────────────────────────────────────────────────────────────
# Grasshopper Python 元件「完整單檔」腳本 — 把 GH 使用者輸入送到 03 的 WebSocket。
#
# ★自足：不 import 03 repo 任何模組、不需 `pip install websocket-client`。
#   內建 RFC6455 client（純 socket），偵測到 websocket-client 存在時優先用它。
#
# ★相容：Rhino 8 ScriptEditor（CPython 3.9）與 Rhino 7 GHPython（IronPython 2.7）
#   皆可跑（無 f-string、無 `X | None` 型別註解、無 dataclass）。
#
# 使用方式：把本檔全文貼進 GH Python 元件，並依下表建立 Input/Output（名稱需一致）。
#
# ── Inputs ───────────────────────────────────────────────────────────────────
#   run         bool    Item    Button/Toggle；True 才送出
#   boundary    Curve   Item    分析域外框（Rectangle3d 或封閉曲線）
#                               ★矩形「最長邊」方向 = 風向（server 端抽取）
#   buildings   Curve   List    建物封閉外框曲線（可多棟；少於 3 點的會被丟棄）
#   wind_level  int     Item    1=微風(Re150) 2=中風(Re300) 3=強風(Re500)；預設 2
#   mode        str     Item    "fresh"（冷啟重算）| "warm"（沿用前次穩態；"adapt" 同義）；預設 fresh
#   viz_mode    str     Item    可選：velocity|vorticity|pressure|streamline|comfort|ibfv
#                               留空＝不切換（GGUI 端鍵盤仍可切）
#   auto_stop   bool    Item    可選；停步守門開關（不接＝不改 server 現況）
#                               True  = 判準宣告收斂即凍結（部署行為，預設）
#                               False = free-run 持續推進，觀察固定點之後的長時行為
#                               ★實測：free-run 過閘後約 10-30 步開始出條紋，
#                                 100 步以上結構瓦解，而 KE 判準不會察覺 → 展示請用 True
#   url         str     Item    可選；預設 "ws://localhost:8765"
#
# ── Outputs ──────────────────────────────────────────────────────────────────
#   status      str             server ack / 警示 / 錯誤訊息
#   payload     str             實際送出的 JSON（除錯用；run=False 時為預覽）
#   metrics     str             送出前的分布守門自檢：風向角、L_char(px)、BCR、長寬比
#
# ── 注意 ─────────────────────────────────────────────────────────────────────
# * 流場「不會」回傳到 GH。流場顯示在 03 啟動的 Taichi GGUI 獨立視窗。
#   本元件的 status 只是 server 的 ack 字串。
# * server 端先把幾何旋轉進 boundary 自身的座標系（最長邊 → +x，即風向），再把
#   boundary 正規化到 [0,1]×[0,1]、光柵化成 360×640 mask。boundary 轉幾度都不會
#   造成變形；只有「boundary 本身長寬比 != 16:9」才會被拉伸 → status 出 WARN。
# * GGUI 顯示的是 canonical frame（風一律沿 +x），不是你在 Rhino 裡的方位。
# * mask 慣例：建物內部 = solid，其餘 = fluid（server 端處理，GH 不需管）。
# ─────────────────────────────────────────────────────────────────────────────

import json

DEFAULT_URL = "ws://localhost:8765"
TARGET_ASPECT = 1920.0 / 1080.0  # server 端 sim 域長寬比
ASPECT_TOL = 0.08  # 8% 內視為可接受
# C-series 訓練分布守門（與 03 src/geometry/regime_features.py 同值）
L_CHAR_TRAIN_RANGE = (32.0, 128.0)  # sim px（1920x1080 基準）
BCR_TRAIN_RANGE = (0.01, 0.30)


# ═══════════════════════════════════════════════════════════════════════════
# 1. 幾何抽取（Rhino → world 座標點列）
# ═══════════════════════════════════════════════════════════════════════════


def _curve_to_pts(curve):
    """Rhino 曲線 → [[x, y], ...] world 座標頂點列（失敗回 []）。"""
    if curve is None:
        return []
    polyline = None
    try:
        ok, pl = curve.TryGetPolyline()
        if ok:
            polyline = pl
    except Exception:
        pass
    if polyline is None:
        try:
            pc = curve.ToPolyline(0.01, 0.1, 0.01, 1.0)
            if pc is not None:
                polyline = pc.ToPolyline() if hasattr(pc, "ToPolyline") else pc
        except Exception:
            return []
    if polyline is None:
        return []
    return [[float(p.X), float(p.Y)] for p in polyline]


def _boundary_corners(boundary_obj):
    """boundary → 4+ 個 world 角點。支援 Rectangle3d（有 .Corner）或任意封閉曲線。"""
    if boundary_obj is None:
        return []
    # Rectangle3d 路徑
    if hasattr(boundary_obj, "Corner"):
        try:
            return [
                [float(boundary_obj.Corner(i).X), float(boundary_obj.Corner(i).Y)]
                for i in range(4)
            ]
        except Exception:
            pass
    # 曲線路徑：取頂點；若首尾重合則去掉重複的尾點
    pts = _curve_to_pts(boundary_obj)
    if (
        len(pts) >= 2
        and abs(pts[0][0] - pts[-1][0]) < 1e-9
        and abs(pts[0][1] - pts[-1][1]) < 1e-9
    ):
        pts = pts[:-1]
    return pts


def _boundary_wh(corners):
    """boundary 自身的（長邊, 短邊）長度——不是 world axis-aligned bounding box。

    server 端把幾何旋轉進 boundary 自身的座標系（最長邊 → +x）之後才做 bbox 正規化，
    所以會不會被拉伸取決於「boundary 本身的長寬比」，與它在 world 裡轉了幾度無關。
    先前此處量的是 AABB：一個正好 16:9 的 boundary 只要一轉，AABB 長寬比就會偏離
    16:9，於是每次旋轉都誤報一次拉伸警示；而一個真的不是 16:9 的 boundary 若剛好轉到
    某個角度，AABB 反而可能接近 16:9 而漏報。

    取最長邊與其相鄰邊（矩形的兩個邊長）。非矩形則退回 AABB，並由呼叫端的
    容差吸收——本元件的 boundary 契約本來就是矩形。
    """
    n = len(corners)
    if n < 3:
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return (max(xs) - min(xs)), (max(ys) - min(ys))

    def _len(i):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % n]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    lengths = [_len(i) for i in range(n)]
    i_max = lengths.index(max(lengths))
    long_e = lengths[i_max]
    short_e = lengths[(i_max + 1) % n]
    if short_e <= 1e-9:
        short_e = min(l for l in lengths if l > 1e-9)
    return long_e, short_e


def _poly_area(pts):
    """多邊形面積（shoelace，取絕對值，world 單位平方）。"""
    n = len(pts)
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        acc += x0 * y1 - x1 * y0
    return abs(acc) * 0.5


def _rotate(pts, deg):
    """把點列旋轉 -deg（進入 boundary 自身座標系，最長邊 → +x）。"""
    rad = -deg * 3.141592653589793 / 180.0
    ca = _cos(rad)
    sa = _sin(rad)
    return [[x * ca - y * sa, x * sa + y * ca] for x, y in pts]


def _cos(a):
    import math

    return math.cos(a)


def _sin(a):
    import math

    return math.sin(a)


def _wind_angle_from_corners(corners):
    """風向 = boundary 最長邊方向（度，自 +X 逆時針）——與 server 端同式。"""
    import math

    n = len(corners)
    if n < 2:
        return 0.0
    best_len = -1.0
    best_ang = 0.0
    for i in range(n):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        ln = dx * dx + dy * dy
        if ln > best_len:
            best_len = ln
            best_ang = math.degrees(math.atan2(dy, dx))
    return best_ang


def _preflight_metrics(corners, polys):
    """送出前自檢 server 端會算的三個分布守門量（風向、L_char、BCR）。

    server 端在 GGUI 視窗印警示，GH 這側看不到；此處以同一套定義先算一次，
    讓使用者在 Rhino 內就知道幾何是否落在訓練分布內。

    L_char 定義（與 03 src/geometry/regime_features.py 一致）：
        迎風橫向（canonical frame 的 y 方向）投影寬度，全體建物取最大，
        單位為 1920x1080 sim px。此處以 boundary 短邊映射至 1080 px 換算。
    BCR = 建物總面積 / boundary 面積。

    Returns:
        (wind_angle_deg, l_char_px, bcr, aspect_ratio)
    """
    ang = _wind_angle_from_corners(corners)
    c_can = _rotate(corners, ang)
    xs = [p[0] for p in c_can]
    ys = [p[1] for p in c_can]
    bw = max(1e-9, max(xs) - min(xs))
    bh = max(1e-9, max(ys) - min(ys))

    l_char = 0.0
    area_sum = 0.0
    for poly in polys:
        p_can = _rotate(poly, ang)
        pys = [p[1] for p in p_can]
        y_ext = max(pys) - min(pys)
        l_char = max(l_char, y_ext / bh * 1080.0)
        area_sum += _poly_area(poly)

    bcr = area_sum / (bw * bh)
    return ang, l_char, bcr, (bw / bh)


def _flatten_buildings(buildings_input):
    """buildings 輸入（List / DataTree / 單一曲線）→ [[[x,y],...], ...]。"""
    if buildings_input is None:
        return []
    crvs = []
    if hasattr(buildings_input, "Branches"):  # GH DataTree
        for branch in buildings_input.Branches:
            for c in branch:
                crvs.append(c)
    elif isinstance(buildings_input, (list, tuple)):  # List access
        crvs = list(buildings_input)
    else:  # 單一物件
        crvs = [buildings_input]

    out = []
    for c in crvs:
        pts = _curve_to_pts(c)
        if len(pts) >= 3:
            out.append(pts)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. WebSocket transport
#    優先用 websocket-client；沒裝就用內建純 socket RFC6455 實作。
# ═══════════════════════════════════════════════════════════════════════════


def _parse_ws_url(url):
    """ws://host:port/path → (host, port, path)。"""
    u = url.strip()
    if u.startswith("ws://"):
        u = u[5:]
    elif u.startswith("wss://"):
        raise ValueError(
            "內建 fallback 不支援 wss://，請安裝 websocket-client 或改用 ws://"
        )
    if "/" in u:
        hostport, path = u.split("/", 1)
        path = "/" + path
    else:
        hostport, path = u, "/"
    if ":" in hostport:
        host, port_s = hostport.rsplit(":", 1)
        port = int(port_s)
    else:
        host, port = hostport, 80
    return host, port, path


class _MiniWS(object):
    """最小 RFC6455 text-frame client（僅供本元件短連線用）。"""

    def __init__(self, url, timeout=3.0):
        import base64
        import os
        import socket

        host, port, path = _parse_ws_url(url)
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(timeout)

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            "GET %s HTTP/1.1\r\n"
            "Host: %s:%d\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: %s\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n" % (path, host, port, key)
        )
        self._sock.sendall(req.encode("ascii"))

        # 讀 handshake 回應直到 \r\n\r\n
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise IOError("handshake 期間連線被關閉")
            buf += chunk
        head = buf.split(b"\r\n\r\n", 1)[0].decode("latin-1")
        if "101" not in head.split("\r\n")[0]:
            raise IOError("WebSocket handshake 失敗：%s" % head.split("\r\n")[0])
        self._rest = buf.split(b"\r\n\r\n", 1)[1]

    def send_text(self, text):
        import os
        import struct

        data = text.encode("utf-8")
        n = len(data)
        frame = bytearray()
        frame.append(0x81)  # FIN + opcode text
        if n < 126:
            frame.append(0x80 | n)  # MASK + len
        elif n < 65536:
            frame.append(0x80 | 126)
            frame.extend(struct.pack(">H", n))
        else:
            frame.append(0x80 | 127)
            frame.extend(struct.pack(">Q", n))
        mask = bytearray(os.urandom(4))
        frame.extend(mask)
        frame.extend(bytearray(data[i] ^ mask[i % 4] for i in range(n)))
        self._sock.sendall(bytes(frame))

    def _read(self, n):
        out = self._rest[:n]
        self._rest = self._rest[n:]
        while len(out) < n:
            chunk = self._sock.recv(n - len(out))
            if not chunk:
                raise IOError("連線在讀取中關閉")
            out += chunk
        return out

    def recv_text(self):
        import struct

        b0 = self._read(1)[0]
        b1 = self._read(1)[0]
        if not isinstance(b0, int):  # IronPython 2.7: bytes 索引出 str
            b0, b1 = ord(b0), ord(b1)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        ln = b1 & 0x7F
        if ln == 126:
            ln = struct.unpack(">H", self._read(2))[0]
        elif ln == 127:
            ln = struct.unpack(">Q", self._read(8))[0]
        mask = self._read(4) if masked else None
        payload = self._read(ln)
        if mask:
            payload = bytes(
                bytearray(payload[i] ^ mask[i % 4] for i in range(len(payload)))
            )
        if opcode == 0x08:
            raise IOError("server 送出 close frame")
        return payload.decode("utf-8", "replace")

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass


def _send_and_receive(payload_str, url, timeout=3.0):
    """送出一筆 JSON、讀一筆回應。回傳 dict。"""
    try:
        import websocket  # websocket-client（若 Rhino 環境有裝）

        ws = websocket.create_connection(url, timeout=timeout)
        try:
            ws.send(payload_str)
            raw = ws.recv()
        finally:
            try:
                ws.close()
            except Exception:
                pass
    except ImportError:
        ws = _MiniWS(url, timeout=timeout)
        try:
            ws.send_text(payload_str)
            raw = ws.recv_text()
        finally:
            ws.close()
    try:
        return json.loads(raw)
    except Exception:
        return {"status": "?", "message": str(raw)}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Payload builder
#    走 03 的 legacy/world 方言（src/core/geometry_ingest.py::ingest_analyze
#    的 boundary_world 分支）——這是 GH 唯一能表達「任意風向」的路徑。
#    p5 網頁走的是另一支扁平 sim-px 方言 {buildings:[{cx,cy,w,h}]}，
#    兩者 server 端同一入口自動分流，不衝突。
# ═══════════════════════════════════════════════════════════════════════════


def _build_analyze_payload(corners, polygons, wind_level_v, mode_v):
    return json.dumps(
        {
            "action": "analyze",
            "mode": mode_v,
            "wind_level": wind_level_v,
            "boundary_world": corners,
            "buildings_world": polygons,
        }
    )


def _build_auto_stop_payload(enabled):
    # 停步守門開關：True = 判準宣告收斂即凍結；False = free-run 持續推進。
    return json.dumps(
        {"action": "set_auto_stop", "payload": {"enabled": bool(enabled)}}
    )


def _build_viz_payload(viz):
    # 控制指令走 payload 包裹（server _parse_command 的相容分支）
    return json.dumps({"action": "set_viz_mode", "payload": {"viz_mode": viz}})


# ═══════════════════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════════════════

VALID_VIZ = ("velocity", "vorticity", "pressure", "streamline", "comfort", "ibfv")

_url = url if ("url" in dir() and url) else DEFAULT_URL
_wind = int(wind_level) if ("wind_level" in dir() and wind_level) else 2
_mode = (mode if ("mode" in dir() and mode) else "fresh").strip().lower()
_viz = viz_mode.strip().lower() if ("viz_mode" in dir() and viz_mode) else ""
# auto_stop 未接線（None）＝不改 server 現況；接了才送 set_auto_stop。
_auto_stop = auto_stop if ("auto_stop" in dir() and auto_stop is not None) else None

if _wind not in (1, 2, 3):
    _wind = 2
if _mode not in ("fresh", "adapt", "warm"):
    _mode = "fresh"

_notes = []
status = ""
payload = ""
metrics = ""

_corners = _boundary_corners(boundary)
_polys = _flatten_buildings(buildings)

if len(_corners) < 3:
    status = "boundary 無效：需 Rectangle3d 或封閉平面曲線（取得 %d 個角點）" % len(
        _corners
    )
elif not _polys:
    status = "無有效建物：buildings 需為封閉曲線（頂點 >= 3）"
else:
    _bw, _bh = _boundary_wh(_corners)
    if _bh > 1e-9:
        _ar = _bw / _bh
        if abs(_ar - TARGET_ASPECT) / TARGET_ASPECT > ASPECT_TOL:
            _notes.append(
                "WARN boundary 長寬比 %.3f 偏離 sim 域 16:9 (%.3f)；"
                "server 依 boundary 自身 bbox 正規化 → 幾何會被拉伸。"
                "建議把 boundary 畫成 16:9（轉幾度沒關係，只看長寬比）。"
                % (_ar, TARGET_ASPECT)
            )

    # ── 分布守門自檢（server 端只在 GGUI 印，GH 這側看不到）──────────────
    _ang, _lchar, _bcr, _ar2 = _preflight_metrics(_corners, _polys)
    metrics = (
        "wind_angle=%.1f deg | L_char=%.0f px | BCR=%.3f | aspect=%.3f (16:9=1.778)"
        % (_ang, _lchar, _bcr, _ar2)
    )
    if not (L_CHAR_TRAIN_RANGE[0] <= _lchar <= L_CHAR_TRAIN_RANGE[1]):
        _notes.append(
            "WARN L_char=%.0f px 超出訓練分布 %s；L_char = 建物迎風橫向投影寬度"
            "（垂直風向的那一邊），換算到 1920x1080 sim px。"
            "調法：讓該邊長 = boundary 短邊的 %.1f%%~%.1f%%。"
            % (
                _lchar,
                str(L_CHAR_TRAIN_RANGE),
                L_CHAR_TRAIN_RANGE[0] / 1080.0 * 100.0,
                L_CHAR_TRAIN_RANGE[1] / 1080.0 * 100.0,
            )
        )
    if not (BCR_TRAIN_RANGE[0] <= _bcr <= BCR_TRAIN_RANGE[1]):
        _notes.append(
            "WARN BCR=%.3f 超出訓練分布 %s（建物總面積 / boundary 面積）。"
            % (_bcr, str(BCR_TRAIN_RANGE))
        )

    payload = _build_analyze_payload(_corners, _polys, _wind, _mode)

    if not run:
        status = "ready（run=True 送出）｜%d 棟建物、wind_level=%d、mode=%s｜%s" % (
            len(_polys),
            _wind,
            _mode,
            metrics,
        )
    else:
        try:
            if _auto_stop is not None:
                _send_and_receive(_build_auto_stop_payload(_auto_stop), _url)
                _notes.append(
                    "auto_stop → %s%s"
                    % (
                        bool(_auto_stop),
                        (
                            ""
                            if _auto_stop
                            else "（free-run：過閘後持續推進，結構會逐步瓦解）"
                        ),
                    )
                )
            if _viz:
                if _viz in VALID_VIZ:
                    _send_and_receive(_build_viz_payload(_viz), _url)
                    _notes.append("viz_mode → %s" % _viz)
                else:
                    _notes.append(
                        "WARN 未知 viz_mode '%s'，已略過（合法值：%s）"
                        % (_viz, "|".join(VALID_VIZ))
                    )

            _resp = _send_and_receive(payload, _url)
            _ack = _resp.get("msg") or _resp.get("message") or str(_resp)
            status = "[%s] %s ｜%d 棟、wind=%d、mode=%s｜%s" % (
                _resp.get("status", "?"),
                _ack,
                len(_polys),
                _wind,
                _mode,
                metrics,
            )
        except Exception as _e:
            status = (
                "連線失敗：%s\n"
                "請確認 03 server 已啟動（lbmnca.bridge 下 `python main.py`）"
                "且 GGUI 視窗已出現、log 顯示 WebSocket listening on %s" % (_e, _url)
            )

if _notes:
    status = status + "\n" + "\n".join(_notes)
