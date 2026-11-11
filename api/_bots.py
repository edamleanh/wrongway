"""Thư viện bot: bot có sẵn (mọi file .py hợp lệ trong submissions/) + bot người dùng upload.

Server KHÔNG chạy code bot nào cả: chỉ lưu và trả mã nguồn. Trận đấu bot vs bot
chạy trong trình duyệt (Pyodide/WebAssembly, xem public/match-worker.js).

Kho lưu bot upload:
- Upstash Redis qua REST API nếu có biến môi trường KV_REST_API_URL/KV_REST_API_TOKEN
  (hoặc UPSTASH_REDIS_REST_URL/UPSTASH_REDIS_REST_TOKEN) - dùng khi deploy Vercel.
- Thư mục uploads/ khi chạy cục bộ (tools/play_web.py).

Xóa bot: cần ADMIN_PASSWORD (biến môi trường) khi chạy trên Vercel; cục bộ thì không cần.
"""

import ast
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.request

from _core import ARENA_DIR, ROOT, BadRequest

SUBMISSIONS = ARENA_DIR / "submissions"
# Bot có sẵn = MỌI file .py trong submissions/ có hàm choose_action (tự nhận, không cần đăng ký).
# Muốn thêm bot: bỏ file <tên>.py vào submissions/ rồi push lên git. Mô tả lấy từ dòng đầu docstring.
HIDDEN = {
    "TEMPLATE",      # file mẫu của giảng viên
    "dev",           # bản thử nghiệm đang sửa
    "bot",           # bản port AI của wrongway.app: chỉ kiểm thử riêng, không được công khai
}
DESCRIPTIONS = {     # mô tả cho file không có docstring phù hợp (ưu tiên hơn docstring)
    "MSSV": "Bot của mình (v4): alpha-beta + trọng số học bằng tiến hóa + tường phòng thủ",
    "HuynhDuy_test2": "Bot của HuynhDuy: dựng hàng tường + cột tường để nhốt đối thủ",
    "wall_builder": "Bot mẫu: đặt tường theo BFS",
    "greedy": "Bot mẫu: đi theo đường ngắn nhất",
    "random_agent": "Bot mẫu: ngẫu nhiên",
}
FIRST, LAST = ["MSSV"], ["wall_builder", "greedy", "random_agent"]   # thứ tự hiển thị
MAX_SIZE = 100 * 1024
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{2,32}$")
KEY_META, KEY_CODE = "wrongway:bots:meta", "wrongway:bots:code"


# ------------------------------------------------------------ kho lưu
class RedisStore:
    kind = "redis"

    def __init__(self, url, token):
        self.url, self.token = url.rstrip("/"), token

    def _cmd(self, *args):
        req = urllib.request.Request(
            self.url, data=json.dumps(list(args)).encode("utf-8"), method="POST",
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=8) as res:
                data = json.load(res)
        except urllib.error.HTTPError as exc:
            raise RuntimeError("Redis lỗi {}: {}".format(exc.code, exc.read()[:200]))
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError("Redis lỗi: " + str(data["error"]))
        return data.get("result")

    def list(self):
        flat = self._cmd("HGETALL", KEY_META) or []
        return [json.loads(flat[i + 1]) for i in range(0, len(flat), 2)]

    def get(self, name):
        return self._cmd("HGET", KEY_CODE, name)

    def add(self, meta, code):
        if self._cmd("HSETNX", KEY_META, meta["name"], json.dumps(meta, ensure_ascii=False)) != 1:
            return False
        self._cmd("HSET", KEY_CODE, meta["name"], code)
        return True

    def delete(self, name):
        self._cmd("HDEL", KEY_META, name)
        self._cmd("HDEL", KEY_CODE, name)


class DirStore:
    kind = "local"

    def __init__(self, folder):
        self.folder = folder
        self.index = folder / "index.json"

    def _load(self):
        try:
            return json.loads(self.index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self, data):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.index.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def list(self):
        return list(self._load().values())

    def get(self, name):
        path = self.folder / (name + ".py")
        return path.read_text(encoding="utf-8") if path.exists() else None

    def add(self, meta, code):
        data = self._load()
        if meta["name"] in data:
            return False
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / (meta["name"] + ".py")).write_text(code, encoding="utf-8")
        data[meta["name"]] = meta
        self._save(data)
        return True

    def delete(self, name):
        data = self._load()
        data.pop(name, None)
        self._save(data)
        path = self.folder / (name + ".py")
        if path.exists():
            path.unlink()


def get_store():
    url = os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    if url and token:
        return RedisStore(url, token)
    if os.environ.get("VERCEL"):
        return None                     # Vercel: ổ đĩa chỉ đọc, cần Redis
    return DirStore(ROOT / "uploads")


def on_vercel():
    return bool(os.environ.get("VERCEL"))


def can_delete():
    return bool(os.environ.get("ADMIN_PASSWORD")) or not on_vercel()


# ------------------------------------------------------------ bot có sẵn
def _describe(name, tree):
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    doc = (ast.get_docstring(tree) or "").strip()
    return doc.splitlines()[0][:120] if doc else "Bot trong submissions/" + name + ".py"


def builtin_bots():
    """{tên: mô tả} của các file .py hợp lệ trong submissions/ (chỉ đọc cú pháp, không chạy)."""
    found = {}
    for path in sorted(SUBMISSIONS.glob("*.py"), key=lambda p: p.stem.lower()):
        name = path.stem
        if name in HIDDEN or name.startswith("_") or not NAME_RE.match(name):
            continue
        try:
            code = path.read_text(encoding="utf-8")
            validate_code(code)
        except (OSError, UnicodeDecodeError, BadRequest):
            continue                      # file hỏng / không phải bot: bỏ qua
        found[name] = _describe(name, ast.parse(code))
    rank = lambda n: (0, FIRST.index(n)) if n in FIRST else (2, LAST.index(n)) if n in LAST else (1, n.lower())
    return {n: found[n] for n in sorted(found, key=rank)}


# ------------------------------------------------------------ API
def list_bots():
    store = get_store()
    bots = [{"id": "builtin:" + n, "name": n, "description": d, "builtin": True}
            for n, d in builtin_bots().items()]
    uploaded = []
    error = None
    if store is not None:
        try:
            uploaded = sorted(store.list(), key=lambda m: -m.get("uploaded_at", 0))
        except Exception as exc:          # Redis lỗi: vẫn trả bot có sẵn
            error = str(exc)
    for m in uploaded:
        bots.append({"id": "user:" + m["name"], "name": m["name"], "author": m.get("author", ""),
                     "description": m.get("description", ""), "size": m.get("size"),
                     "uploaded_at": m.get("uploaded_at"), "builtin": False})
    return {"bots": bots, "storage": store.kind if store else None,
            "can_delete": can_delete(), "error": error}


def get_source(bot_id):
    """Mã nguồn một bot, hoặc environment.py của Arena (id = 'env')."""
    if bot_id == "env":
        return (ARENA_DIR / "environment.py").read_text(encoding="utf-8")
    kind, _, name = (bot_id or "").partition(":")
    if kind == "builtin" and name in builtin_bots():
        return (SUBMISSIONS / (name + ".py")).read_text(encoding="utf-8")
    if kind == "user" and NAME_RE.match(name):
        store = get_store()
        code = store.get(name) if store else None
        if code is not None:
            return code
    raise BadRequest("Không tìm thấy bot '{}'".format(bot_id))


def validate_code(code):
    """Chỉ phân tích cú pháp (không chạy): phải có hàm top-level choose_action."""
    if not isinstance(code, str) or not code.strip():
        raise BadRequest("File rỗng")
    if len(code.encode("utf-8")) > MAX_SIZE:
        raise BadRequest("File quá lớn (tối đa {} KB)".format(MAX_SIZE // 1024))
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise BadRequest("Lỗi cú pháp dòng {}: {}".format(exc.lineno, exc.msg))
    if not any(isinstance(n, ast.FunctionDef) and n.name == "choose_action" for n in tree.body):
        raise BadRequest("File phải có hàm choose_action(observation) ở cấp ngoài cùng")


def upload(body):
    store = get_store()
    if store is None:
        raise BadRequest("Chưa cấu hình kho lưu bot (Upstash Redis) trên Vercel - xem README")
    name = str(body.get("name", "")).strip()
    if not NAME_RE.match(name):
        raise BadRequest("Tên bot 2-32 ký tự, chỉ gồm chữ không dấu, số, _ hoặc -")
    if name in builtin_bots():
        raise BadRequest("Tên này trùng bot có sẵn")
    code = body.get("code")
    validate_code(code)
    meta = {"name": name, "author": str(body.get("author", "")).strip()[:40],
            "description": str(body.get("description", "")).strip()[:120],
            "size": len(code.encode("utf-8")), "uploaded_at": int(time.time())}
    if not store.add(meta, code):
        raise BadRequest("Đã có bot tên '{}' - chọn tên khác".format(name))
    return {"ok": True, "id": "user:" + name}


def delete(body):
    store = get_store()
    if store is None:
        raise BadRequest("Chưa cấu hình kho lưu bot")
    secret = os.environ.get("ADMIN_PASSWORD")
    if on_vercel():
        if not secret:
            raise BadRequest("Chưa đặt ADMIN_PASSWORD nên không thể xóa")
        if not hmac.compare_digest(str(body.get("password", "")), secret):
            raise BadRequest("Sai mật khẩu")
    kind, _, name = str(body.get("id", "")).partition(":")
    if kind != "user" or not NAME_RE.match(name):
        raise BadRequest("Chỉ xóa được bot do người dùng upload")
    store.delete(name)
    return {"ok": True}


def handle_get(query):
    """query: dict tham số URL (giá trị đầu tiên)."""
    if query.get("id"):
        return {"id": query["id"], "code": get_source(query["id"])}
    return list_bots()


def handle_post(body):
    if not isinstance(body, dict):
        raise BadRequest("Request phải là JSON object")
    op = body.get("op")
    if op == "upload":
        return upload(body)
    if op == "delete":
        return delete(body)
    raise BadRequest("op không hợp lệ")
