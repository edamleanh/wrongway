"""Vercel serverless function: thư viện bot (xem api/_bots.py).

GET  /api/bots            -> danh sách bot
GET  /api/bots?id=...     -> mã nguồn bot (id=env: environment.py của Arena)
POST /api/bots {"op": "upload", "name", "author", "description", "code"}
POST /api/bots {"op": "delete", "id", "password"}
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bots import handle_get, handle_post  # noqa: E402
from _core import BadRequest  # noqa: E402

MAX_BODY = 256 * 1024


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

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._send(413, {"error": "File quá lớn"})
        try:
            self._send(200, handle_post(json.loads(self.rfile.read(length) or b"{}")))
        except (BadRequest, ValueError) as exc:
            self._send(400, {"error": str(exc) or "Request không hợp lệ"})
        except Exception as exc:
            self._send(500, {"error": "Lỗi máy chủ: " + str(exc)})
