#!/usr/bin/env python3
"""Chơi Wrongway với bot qua trình duyệt.

    python tools/play_web.py                 # mở http://localhost:8000
    python tools/play_web.py --bot greedy --port 8080 --no-browser

Luật do environment.py của Arena kiểm tra; bot là file trong submissions/
(mặc định MSSV.py). Chỉ dùng thư viện chuẩn.
"""

import argparse
import copy
import importlib.util
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import _arena_path  # noqa: F401  (thiết lập sys.path)
from _arena_path import SUBMISSIONS
from environment import Environment

WEB_DIR = Path(__file__).resolve().parent / "web"
LEVELS = {"easy": 1, "medium": 3, "hard": None}   # None = cấu hình gốc của bot


def load_bot(path):
    """Nạp mới module bot cho mỗi ván (xóa biến toàn cục của ván trước)."""
    spec = importlib.util.spec_from_file_location("web_bot_%d" % time.time_ns(), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Game:
    def __init__(self, bot_path):
        self.bot_path = Path(bot_path)
        self.lock = threading.Lock()
        self.new(human=0, level="hard")

    def new(self, human, level):
        self.bot = load_bot(self.bot_path)
        if LEVELS.get(level):
            self.bot.MAX_DEPTH = LEVELS[level]
        self.env = Environment()
        self.human = human
        self.level = level
        self.history = []          # (env, log, bot_info) trước mỗi nước của người
        self.log = []
        self.bot_info = None

    # ------------------------------------------------------------ trạng thái
    def state(self):
        env = self.env
        s = env.snapshot()
        mine = not env.done and env.current_player == self.human
        s.update({
            "human": self.human,
            "level": self.level,
            "bot_name": self.bot_path.stem,
            "log": self.log,
            "bot_info": self.bot_info,
            "can_undo": bool(self.history),
            "legal_moves": env.legal_moves(self.human) if mine else [],
            "legal_walls": env.legal_wall_actions(self.human) if mine else [],
            "distances": [env.distance_map(i)[r][c] for i, (r, c) in enumerate(env.positions)],
            "wall_owners": [{"o": e["action"]["orientation"], "r": e["action"]["row"],
                             "c": e["action"]["col"], "owner": e["player"]}
                            for e in self.log if e["action"] and e["action"]["type"] == "wall"],
        })
        return s

    def _record(self, player, result, extra=None):
        entry = {"turn": self.env.turn_count, "player": player,
                 "who": "Bạn" if player == self.human else "Bot",
                 "action": result.get("action"), "status": result["status"]}
        if extra:
            entry.update(extra)
        self.log.append(entry)

    # ------------------------------------------------------------ hành động
    def human_action(self, action):
        env = self.env
        if env.done:
            return "Ván đã kết thúc"
        if env.current_player != self.human:
            return "Chưa tới lượt bạn"
        kind = action.get("type")
        if kind == "move":
            if list(action.get("to", [])) not in env.legal_moves():
                return "Nước đi không hợp lệ"
        elif kind == "wall":
            wall = {k: action.get(k) for k in ("orientation", "row", "col")}
            if wall not in env.legal_wall_actions():
                return "Không thể đặt tường ở đây"
        elif kind != "pass":
            return "Hành động không hợp lệ"
        self.history.append((copy.deepcopy(env), list(self.log), self.bot_info))
        self._record(self.human, env.step(action))
        return None

    def bot_move(self):
        env = self.env
        if env.done or env.current_player == self.human:
            return
        obs = env.get_observation()
        t0 = time.perf_counter()
        try:
            action = self.bot.choose_action(obs)
        except Exception as exc:          # bot lỗi -> mất lượt, giống Arena
            action = {"type": "pass", "error": str(exc)}
        elapsed = time.perf_counter() - t0
        info = dict(getattr(self.bot, "LAST_INFO", {}) or {})
        info["time"] = round(elapsed, 3)
        self.bot_info = info
        self._record(1 - self.human, env.step(action), {"time": info["time"],
                                                        "depth": info.get("depth")})

    def undo(self):
        if self.history:
            self.env, self.log, self.bot_info = self.history.pop()


GAME = None


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
            self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            with GAME.lock:
                self._send(200, GAME.state())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._send(400, {"error": "JSON không hợp lệ"})
        with GAME.lock:
            error = None
            if self.path == "/api/new":
                GAME.new(human=1 if body.get("human") == 1 else 0,
                         level=body.get("level") if body.get("level") in LEVELS else "hard")
            elif self.path == "/api/action":
                error = GAME.human_action(body.get("action") or {})
            elif self.path == "/api/bot":
                GAME.bot_move()
            elif self.path == "/api/undo":
                GAME.undo()
            else:
                return self._send(404, {"error": "not found"})
            state = GAME.state()
            state["error"] = error
            self._send(400 if error else 200, state)


def main():
    global GAME
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", default="MSSV", help="tên file trong submissions/ hoặc đường dẫn .py")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    path = Path(a.bot)
    if not (path.suffix == ".py" and path.exists()):
        path = SUBMISSIONS / (a.bot + ".py")
    GAME = Game(path)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = "http://localhost:%d" % a.port
    print("Đang chạy {} - bot: {} (Ctrl+C để dừng)".format(url, path.name))
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng.")


if __name__ == "__main__":
    main()
