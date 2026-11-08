#!/usr/bin/env python3
"""Kiểm tra một bài nộp trước khi lên Moodle.

    py -3 tools/check_submission.py wrongway-student/wrongway/submissions/MSSV.py

Kiểm tra: tên file là MSSV dạng số, chỉ import thư viện chuẩn, có
choose_action, thời gian import, đấu cả hai bên với các bot mẫu không
có timeout/invalid/error, và thời gian lớn nhất mỗi nước.
"""

import ast
import re
import sys
import time
from pathlib import Path

import _arena_path  # noqa: F401
from _arena_path import SUBMISSIONS
from arena import run_match

SAMPLE_BOTS = ["greedy", "random_agent", "wall_builder"]
WARN_ELAPSED = 0.7   # bot dùng TIME_BUDGET 0.55 s theo đồng hồ thật


def stdlib_names():
    names = set(getattr(sys, "stdlib_module_names", ()))
    if not names:  # Python < 3.10: danh sách tối thiểu
        names = {"collections", "heapq", "itertools", "math", "random", "time",
                 "functools", "typing", "copy", "dataclasses", "array", "bisect"}
    return names


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print(__doc__)
        return 2
    path = Path(argv[0]).resolve()
    problems, warnings = [], []

    if not re.fullmatch(r"\d+", path.stem):
        warnings.append("Tên file '{}' chưa phải MSSV dạng số (vd 26127000.py).".format(path.name))

    tree = ast.parse(path.read_text(encoding="utf-8"))
    std = stdlib_names()
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append("Import tương đối không được phép (dòng {}).".format(node.lineno))
            mods = [node.module or ""]
        for m in mods:
            top = m.split(".")[0]
            if top and top not in std:
                problems.append("Import ngoài thư viện chuẩn: '{}' (dòng {}).".format(m, node.lineno))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("input", "open"):
            warnings.append("Gọi {}() ở dòng {}: tránh đọc bàn phím/file.".format(
                node.func.id, node.lineno))
    if not any(isinstance(n, ast.FunctionDef) and n.name == "choose_action" for n in tree.body):
        problems.append("Thiếu hàm top-level choose_action(observation).")

    if problems:
        print("LỖI:\n  " + "\n  ".join(problems))
        return 1

    print("Đấu thử với bot mẫu, cả hai bên...")
    max_elapsed = 0.0
    for bot in SAMPLE_BOTS:
        for side in (0, 1):
            pair = (path, SUBMISSIONS / (bot + ".py"))
            if side == 1:
                pair = pair[::-1]
            t0 = time.time()
            res = run_match(*pair, seed=side)
            st = res["stats"][side]
            mine = [t["elapsed"] for t in res["turns"] if t["player"] == side]
            max_elapsed = max([max_elapsed] + mine)
            result = ("hòa" if res["winner"] is None
                      else "THẮNG" if res["winner"] == side else "thua")
            print("  P{} vs {:<13} {:<6} {:>3} lượt  {}  ({:.1f}s)".format(
                side, bot, result, res["total_turns"], st, time.time() - t0))
            for k in ("timeout", "invalid", "error"):
                if st[k]:
                    problems.append("{} lần {} khi P{} vs {}.".format(st[k], k, side, bot))
            for t in res["turns"]:
                if t["player"] == side and t["error"]:
                    problems.append("Lượt {}: {}".format(t["turn"], t["error"]))
                    break

    if max_elapsed > WARN_ELAPSED:
        warnings.append("Nước chậm nhất {:.3f}s (> {}s) - Colab có thể chậm hơn máy bạn.".format(
            max_elapsed, WARN_ELAPSED))
    print("Thời gian lớn nhất mỗi nước: {:.3f}s".format(max_elapsed))
    for w in warnings:
        print("CẢNH BÁO: " + w)
    if problems:
        print("LỖI:\n  " + "\n  ".join(problems))
        return 1
    print("OK: sẵn sàng nộp" + (" (xem cảnh báo)" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
