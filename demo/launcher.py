"""launcher.py — 常駐本機服務，橋接瀏覽器按鈕與後端程序啟動。

瀏覽器 JS 無法直接 spawn 本機程序（沙箱限制）。此服務固定監聽 8090，
提供 /status /start 兩個 GET endpoint（GET 避免 CORS 預檢），由網頁按鈕
fetch 呼叫，實際由本程序 subprocess.Popen 起 main.py + http.server。

WS port 可調（?ws_port=N）：main.py 本身無 CLI/env port 覆寫機制（見
docs/DEPLOYMENT.md 33 檔執行鏈，不能改 main.py），故用針對性文字替換
只改 config.yaml 的 server.port 那一行，其餘內容（含大量重要註解）原樣保留，
不走 yaml 全量 load/dump。http.server(8080)／launcher 自身(8090) 維持固定。

啟動：雙擊 launcher.bat，或設開機自動啟動（見 install_autostart.ps1）。
"""

import http.server
import json
import re
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PORT = 8090
HTTP_PORT = 8080
MIN_PORT = 1024
MAX_PORT = 65535

REPO = Path(__file__).resolve().parent
WORKSPACE = REPO.parent
PYEXE = WORKSPACE / ".venv" / "Scripts" / "python.exe"
CONFIG_PATH = WORKSPACE / "configs" / "bridge" / "config.yaml"

CREATE_NEW_CONSOLE = 0x00000010

_children = {}

_SERVER_PORT_RE = re.compile(
    r"(?m)^(server:\n(?:^  .*\n)*?^  port:\s*)(\d+)",
)


def _port_open(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def _read_configured_ws_port():
    text = CONFIG_PATH.read_text(encoding="utf-8")
    m = _SERVER_PORT_RE.search(text)
    if not m:
        raise RuntimeError(f"config.yaml: 找不到 server.port（regex 沒配到，檔案格式是否變了？）")
    return int(m.group(2))


def _set_configured_ws_port(new_port):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    m = _SERVER_PORT_RE.search(text)
    if not m:
        raise RuntimeError(f"config.yaml: 找不到 server.port（regex 沒配到，檔案格式是否變了？）")
    new_text = text[: m.start()] + m.group(1) + str(new_port) + text[m.end():]
    CONFIG_PATH.write_text(new_text, encoding="utf-8")
    print(f"[launcher] config.yaml server.port: {m.group(2)} -> {new_port}")


def _valid_port(port):
    return MIN_PORT <= port <= MAX_PORT and port not in (PORT, HTTP_PORT)


def _spawn(key, argv, cwd):
    proc = subprocess.Popen(argv, cwd=str(cwd), creationflags=CREATE_NEW_CONSOLE)
    _children[key] = proc
    print(f"[launcher] spawned {key}: {argv} (pid={proc.pid}, cwd={cwd})")


class Handler(http.server.BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query_ws_port(self, qs):
        if "ws_port" not in qs:
            return _read_configured_ws_port(), None
        raw = qs["ws_port"][0]
        try:
            port = int(raw)
        except ValueError:
            return None, f"ws_port 不是整數: {raw!r}"
        if not _valid_port(port):
            return None, f"ws_port 超出範圍或與固定 port 衝突: {port}（需 {MIN_PORT}-{MAX_PORT}，不可為 {PORT}/{HTTP_PORT}）"
        return port, None

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/status":
            ws_port, err = self._query_ws_port(qs)
            if err:
                self._json({"error": err}, code=400)
                return
            self._json({
                "launcher": True,
                "ws": _port_open(ws_port),
                "http": _port_open(HTTP_PORT),
                "ws_port": ws_port,
            })
            return

        if parsed.path == "/start":
            if not PYEXE.exists():
                self._json({"error": f"venv python not found: {PYEXE}"}, code=500)
                return

            ws_port, err = self._query_ws_port(qs)
            if err:
                self._json({"error": err}, code=400)
                return

            started = []
            already = []

            old_port = _read_configured_ws_port()
            if _port_open(ws_port):
                already.append("main.py")
            elif old_port != ws_port and _port_open(old_port):
                self._json({
                    "error": f"main.py already running on port {old_port}",
                    "current_port": old_port,
                }, code=409)
                return
            else:
                if ws_port != old_port:
                    _set_configured_ws_port(ws_port)
                _spawn("main.py", [str(PYEXE), "-m", "lbmnca.bridge.main"], WORKSPACE)
                started.append("main.py")

            if _port_open(HTTP_PORT):
                already.append("http.server")
            else:
                _spawn("http.server", [str(PYEXE), "-m", "http.server", str(HTTP_PORT)], WORKSPACE)
                started.append("http.server")

            self._json({"started": started, "already": already, "ws_port": ws_port})
            return

        if parsed.path == "/stop":
            stopped = []
            for key, proc in list(_children.items()):
                if proc.poll() is None:
                    proc.terminate()
                    stopped.append(key)
                del _children[key]
            self._json({"stopped": stopped})
            return

        self._json({"error": "not found"}, code=404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def log_message(self, fmt, *args):
        print(f"[launcher] {self.address_string()} " + (fmt % args))


def main():
    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[launcher] listening on http://localhost:{PORT}")
    print(f"[launcher] repo={REPO}")
    print(f"[launcher] workspace={WORKSPACE}")
    print(f"[launcher] pyexe={PYEXE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
