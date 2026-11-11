"""Thư viện bot cho trang đấu bot vs bot.

Bot = MỌI file .py trong wrongway-student/wrongway/submissions/ có hàm choose_action
ở cấp ngoài cùng. Muốn thêm bot: bỏ file vào thư mục đó rồi push lên git (Vercel tự
deploy lại). Không có upload qua web.

Server KHÔNG chạy code bot nào cả: chỉ đọc cú pháp và trả mã nguồn. Trận đấu chạy
trong trình duyệt (Pyodide/WebAssembly, xem public/match-worker.js).
"""

import ast
import re

from _core import ARENA_DIR, BadRequest

SUBMISSIONS = ARENA_DIR / "submissions"
# Tự nhận bot, không cần đăng ký. Mô tả lấy từ dòng đầu docstring (hoặc DESCRIPTIONS).
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
    return {"bots": [{"id": "builtin:" + n, "name": n, "description": d, "builtin": True}
                     for n, d in builtin_bots().items()]}


def get_source(bot_id):
    """Mã nguồn một bot, hoặc environment.py của Arena (id = 'env')."""
    if bot_id == "env":
        return (ARENA_DIR / "environment.py").read_text(encoding="utf-8")
    kind, _, name = (bot_id or "").partition(":")
    if kind == "builtin" and name in builtin_bots():
        return (SUBMISSIONS / (name + ".py")).read_text(encoding="utf-8")
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


def handle_get(query):
    """query: dict tham số URL (giá trị đầu tiên)."""
    if query.get("id"):
        return {"id": query["id"], "code": get_source(query["id"])}
    return list_bots()
