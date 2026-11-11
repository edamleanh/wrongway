"""Vercel serverless function: thư viện bot (xem api/_bots.py).

GET  /api/bots            -> danh sách bot
GET  /api/bots?id=...     -> mã nguồn bot (id=env: environment.py của Arena)

Chỉ đọc: bot được thêm bằng cách push file vào submissions/, không có upload qua web.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bots import handle_get  # noqa: E402
from _core import BadRequest  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, cache=False):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "public, max-age=60" if cache else "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        try:
            self._send(200, handle_get(query), cache=query.get("id") == "env")
        except BadRequest as exc:
            self._send(404, {"error": str(exc)})
        except Exception as exc:
            self._send(500, {"error": "Lỗi máy chủ: " + str(exc)})
