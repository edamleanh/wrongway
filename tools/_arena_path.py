"""Thêm thư mục Arena vào sys.path để các tool import được arena.py."""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
ARENA_DIR = PROJECT / "wrongway-student" / "wrongway"
SUBMISSIONS = ARENA_DIR / "submissions"
LOGS = PROJECT / "logs"

for _stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

if str(ARENA_DIR) not in sys.path:
    sys.path.insert(0, str(ARENA_DIR))
