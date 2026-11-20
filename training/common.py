"""Phần dùng chung cho training: đường dẫn, nạp bot, ghi trọng số vào file nộp."""

import importlib.util
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
ARENA_DIR = PROJECT / "wrongway-student" / "wrongway"
SUBMISSIONS = ARENA_DIR / "submissions"
BOT_FILE = SUBMISSIONS / "MSSV.py"
TRAIN_DIR = PROJECT / "training"
DATA_DIR = TRAIN_DIR / "data"
CKPT_DIR = TRAIN_DIR / "checkpoints"
TMP_DIR = TRAIN_DIR / "tmp"

if str(ARENA_DIR) not in sys.path:
    sys.path.insert(0, str(ARENA_DIR))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

_BLOCK = re.compile(r"(# =+ WEIGHTS BEGIN[^\n]*\n)(.*?)(\n# =+ WEIGHTS END)", re.S)


def load_module(path, name=None):
    path = Path(path)
    name = name or "bot_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weights_to_source(weights, digits=4):
    """weights (dict/list số) -> mã nguồn 'WEIGHTS = ...' (JSON hợp lệ cũng là Python hợp lệ).

    Trọng số phẳng (chỉ có số) viết trên một dòng. Trọng số dạng bảng (hàm đánh giá tra bảng,
    training/td_learn.py) viết mỗi khóa một dòng, bảng 2 chiều mỗi hàng một dòng cho dễ đọc."""
    def rnd(x):
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rnd(v) for v in x]
        return round(float(x), digits)

    def dump(x):
        return json.dumps(x, separators=(", ", ": "))

    w = rnd(weights)
    if not any(isinstance(v, (list, dict)) for v in w.values()):
        return "WEIGHTS = " + dump(w)
    lines = ["WEIGHTS = {"]
    for key, value in w.items():
        if isinstance(value, list) and value and isinstance(value[0], list):
            lines.append('    "%s": [' % key)
            lines += ["        %s," % dump(row) for row in value]
            lines.append("    ],")
        else:
            lines.append('    "%s": %s,' % (key, dump(value)))
    lines.append("}")
    return "\n".join(lines)


def write_weights(bot_text, weights):
    """Trả về nội dung file bot với khối WEIGHTS đã thay."""
    src = "WEIGHTS = None" if weights is None else weights_to_source(weights)
    new, n = _BLOCK.subn(lambda m: m.group(1) + src + m.group(3), bot_text)
    if n != 1:
        raise RuntimeError("Không tìm thấy khối WEIGHTS BEGIN/END trong file bot")
    return new


def make_variant(weights, name):
    """Tạo bản sao bot với trọng số cho trước trong training/tmp/, trả về đường dẫn."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / (name + ".py")
    path.write_text(write_weights(BOT_FILE.read_text(encoding="utf-8"), weights),
                    encoding="utf-8")
    return path
