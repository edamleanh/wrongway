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
    """weights (dict/list số) -> dòng 'WEIGHTS = ...' (JSON hợp lệ cũng là Python hợp lệ)."""
    def rnd(x):
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rnd(v) for v in x]
        return round(float(x), digits)
    return "WEIGHTS = " + json.dumps(rnd(weights), separators=(", ", ": "))


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
