"""Vercel serverless function: POST /api/move (xem api/_core.py)."""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core import BadRequest, handle  # noqa: E402

MAX_BODY = 64 * 1024


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._send(413, {"error": "Request quá lớn"})
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, handle(req))
        except (BadRequest, ValueError) as exc:
            self._send(400, {"error": str(exc) or "Request không hợp lệ"})

    def do_GET(self):
        self._send(405, {"error": "Dùng POST"})
