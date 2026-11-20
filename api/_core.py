"""Xử lý ván cờ không lưu trạng thái (stateless), dùng chung cho Vercel và bản chạy cục bộ.

Trình duyệt giữ trạng thái ván và gửi kèm mỗi request; server dựng lại
Environment của Arena từ trạng thái đó, kiểm tra/áp dụng nước đi, cho bot đáp,
rồi trả về trạng thái mới. Nhờ vậy chạy được trên serverless (Vercel) và
nhiều người chơi cùng lúc mà không lẫn ván.

Request (JSON):
    {"op": "new",    "human": 0|1}
    {"op": "action", "state": {...}, "human": 0|1, "action": {...}}
    {"op": "bot",    "state": {...}, "human": 0|1, "level": "easy"|"medium"|"hard"}
Response: {"state": snapshot, "view": {...}, "event": {...} | null, "bot_name": str}
"""

import importlib.util
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARENA_DIR = ROOT / "wrongway-student" / "wrongway"
DEFAULT_BOT = ARENA_DIR / "submissions" / "MSSV.py"
if str(ARENA_DIR) not in sys.path:
    sys.path.insert(0, str(ARENA_DIR))

from environment import Environment  # noqa: E402

LEVELS = {"easy": 1, "medium": 3, "hard": None}   # None = cấu hình gốc của bot
WALLS_PER_PLAYER = 10

_lock = threading.Lock()
_bots = {}


class BadRequest(Exception):
    pass


def load_bot(path):
    path = Path(path)
    if path not in _bots:
        spec = importlib.util.spec_from_file_location("web_bot_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _bots[path] = (module, getattr(module, "MAX_DEPTH", None))
    return _bots[path]


def load_bot_from_code(code_str, name="custom"):
    """Nạp bot tạm thời từ chuỗi mã nguồn Python trong RAM (không lưu file ra ổ đĩa)."""
    spec = importlib.util.spec_from_loader("web_bot_" + name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(code_str, module.__dict__)
    if not callable(getattr(module, "choose_action", None)):
        raise BadRequest("Bot thiếu hàm choose_action(observation)")
    return module, getattr(module, "MAX_DEPTH", None)


# ------------------------------------------------------------ kiểm tra đầu vào
def _int(value, lo, hi, name):
    if type(value) is not int or not lo <= value <= hi:
        raise BadRequest("Trường '{}' không hợp lệ".format(name))
    return value


def _pair(value, lo, hi, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise BadRequest("Trường '{}' không hợp lệ".format(name))
    return (_int(value[0], lo, hi, name), _int(value[1], lo, hi, name))


def env_from_state(s):
    """Dựng lại Environment từ snapshot do trình duyệt gửi lên (có kiểm tra)."""
    if not isinstance(s, dict):
        raise BadRequest("Thiếu trạng thái ván")
    env = Environment(max_turns=_int(s.get("max_turns", 500), 1, 1000, "max_turns"))
    positions = s.get("positions")
    if not isinstance(positions, list) or len(positions) != 2:
        raise BadRequest("Trường 'positions' không hợp lệ")
    env.positions = [_pair(p, 0, 8, "positions") for p in positions]
    if env.positions[0] == env.positions[1]:
        raise BadRequest("Hai quân không thể cùng một ô")
    walls = s.get("walls")
    if not isinstance(walls, dict):
        raise BadRequest("Trường 'walls' không hợp lệ")
    env.horizontal_walls = {_pair(w, 0, 7, "walls") for w in walls.get("horizontal", [])}
    env.vertical_walls = {_pair(w, 0, 7, "walls") for w in walls.get("vertical", [])}
    remaining = s.get("walls_remaining")
    if not isinstance(remaining, list) or len(remaining) != 2:
        raise BadRequest("Trường 'walls_remaining' không hợp lệ")
    env.walls_remaining = [_int(v, 0, WALLS_PER_PLAYER, "walls_remaining") for v in remaining]
    placed = len(env.horizontal_walls) + len(env.vertical_walls)
    if placed != 2 * WALLS_PER_PLAYER - sum(env.walls_remaining):
        raise BadRequest("Số tường không khớp")
    env.current_player = _int(s.get("current_player"), 0, 1, "current_player")
    env.turn_count = _int(s.get("turn"), 0, env.max_turns, "turn")
    env.done = bool(s.get("done"))
    winner = s.get("winner")
    env.winner = None if winner is None else _int(winner, 0, 1, "winner")
    return env


# ------------------------------------------------------------ phản hồi
def view(env, human):
    mine = not env.done and env.current_player == human
    return {
        "legal_moves": env.legal_moves(human) if mine else [],
        "legal_walls": env.legal_wall_actions(human) if mine else [],
        "distances": [env.distance_map(i)[r][c] for i, (r, c) in enumerate(env.positions)],
    }


def respond(env, human, event=None, bot_path=DEFAULT_BOT, custom_name=None):
    res = {"state": env.snapshot(), "view": view(env, human), "event": event,
           "bot_name": custom_name or Path(bot_path).stem}
    if not env.done:
        res["observation"] = env.get_observation(env.current_player)
    return res


def _event(env, player, human, result, extra=None):
    e = {"turn": env.turn_count, "player": player, "who": "Bạn" if player == human else "Bot",
         "action": result.get("action"), "status": result["status"]}
    if extra:
        e.update(extra)
    return e


def _resolve_bot(name, default_path):
    if name:
        clean = name.replace("builtin:", "").strip()
        candidate = ARENA_DIR / "submissions" / (Path(clean).stem + ".py")
        if candidate.is_file():
            return candidate
    return default_path


# ------------------------------------------------------------ xử lý
def handle(req, bot_path=DEFAULT_BOT):
    if not isinstance(req, dict):
        raise BadRequest("Request phải là JSON object")
    op = req.get("op")
    human = _int(req.get("human", 0), 0, 1, "human")
    bot_path = _resolve_bot(req.get("bot"), bot_path)
    custom_name = req.get("bot_name")

    if op == "new":
        return respond(Environment(), human, bot_path=bot_path, custom_name=custom_name)

    env = env_from_state(req.get("state"))
    if env.done:
        raise BadRequest("Ván đã kết thúc")

    if op == "action":
        player = _int(req.get("player", human), 0, 1, "player")
        if env.current_player != player:
            raise BadRequest("Chưa tới lượt bạn" if player == human else "Chưa tới lượt bot")
        action = req.get("action")
        if not isinstance(action, dict):
            raise BadRequest("Thiếu nước đi")
        if player == human:
            kind = action.get("type")
            if kind == "move":
                if list(action.get("to") or []) not in env.legal_moves():
                    raise BadRequest("Nước đi không hợp lệ")
            elif kind == "wall":
                wall = {k: action.get(k) for k in ("orientation", "row", "col")}
                if wall not in env.legal_wall_actions():
                    raise BadRequest("Không thể đặt tường ở đây")
            elif kind == "pass":
                pass
            else:
                raise BadRequest("Hành động không hợp lệ")
        result = env.step(action)
        extra = {}
        if req.get("elapsed") is not None:
            extra["time"] = req["elapsed"]
        return respond(env, human, _event(env, player, human, result, extra), bot_path, custom_name)

    if op == "bot":
        if env.current_player == human:
            raise BadRequest("Đang là lượt của bạn")
        level = req.get("level") if req.get("level") in LEVELS else "hard"
        player = env.current_player
        obs = env.get_observation()
        bot_code = req.get("bot_code")
        with _lock:                       # MAX_DEPTH là biến toàn cục của module bot
            if bot_code:
                bot, default_depth = load_bot_from_code(bot_code, custom_name or "custom")
            else:
                bot, default_depth = load_bot(bot_path)
            if hasattr(bot, "MAX_DEPTH"):
                bot.MAX_DEPTH = LEVELS[level] or default_depth
            t0 = time.perf_counter()
            try:
                action = bot.choose_action(obs)
            except Exception:             # bot lỗi -> mất lượt, giống Arena
                action = {"type": "pass"}
            elapsed = round(time.perf_counter() - t0, 3)
            info = dict(getattr(bot, "LAST_INFO", {}) or {})
        result = env.step(action)
        return respond(env, human, _event(env, player, human, result,
                                          {"time": elapsed, "depth": info.get("depth")}), bot_path, custom_name)

    raise BadRequest("op không hợp lệ")
