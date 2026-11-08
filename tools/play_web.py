#!/usr/bin/env python3
"""Chơi Wrongway với bot qua trình duyệt (bản chạy cục bộ).

    python tools/play_web.py                 # mở http://localhost:8000
    python tools/play_web.py --bot greedy --port 8080 --no-browser

Dùng chung giao diện (public/index.html) và logic (api/_core.py) với bản
deploy trên Vercel. Luật do environment.py của Arena kiểm tra; bot mặc định là
submissions/MSSV.py. Chỉ dùng thư viện chuẩn.
"""

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
from _core import ARENA_DIR, BadRequest, handle  # noqa: E402

INDEX = ROOT / "public" / "index.html"
BOT_PATH = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/move":
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, handle(req, BOT_PATH))
        except (BadRequest, ValueError) as exc:
            self._send(400, {"error": str(exc) or "Request không hợp lệ"})


def main():
    global BOT_PATH
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", default="MSSV", help="tên file trong submissions/ hoặc đường dẫn .py")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    path = Path(a.bot)
    BOT_PATH = path if path.suffix == ".py" and path.exists() else ARENA_DIR / "submissions" / (a.bot + ".py")
    if not BOT_PATH.exists():
        sys.exit("Không tìm thấy bot: {}".format(BOT_PATH))
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = "http://localhost:%d" % a.port
    print("Đang chạy {} - bot: {} (Ctrl+C để dừng)".format(url, BOT_PATH.name), flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng.")


if __name__ == "__main__":
    main()
