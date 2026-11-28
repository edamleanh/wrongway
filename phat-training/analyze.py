"""In diễn biến một replay: mỗi lượt, nước đi và khoảng cách tới đích của hai bên.

Cách đọc: d0 / d1 là số bước ngắn nhất còn lại của người chơi 0 / 1 tới hàng đích
của chính họ (bỏ qua quân đối thủ). w0 / w1 là số tường còn lại.
"""
import json
import sys
from collections import deque

SIZE = 9


def can_step(r, c, nr, nc, h, v):
    if nr == r + 1:                       # xuống: tường ngang (r, c-1) hoặc (r, c) chắn
        return not ((r, c - 1) in h or (r, c) in h)
    if nr == r - 1:                       # lên: tường ngang (r-1, c-1) hoặc (r-1, c)
        return not ((r - 1, c - 1) in h or (r - 1, c) in h)
    if nc == c + 1:                       # sang phải: tường dọc (r-1, c) hoặc (r, c)
        return not ((r - 1, c) in v or (r, c) in v)
    return not ((r - 1, c - 1) in v or (r, c - 1) in v)   # sang trái


def dist(pos, goal, h, v):
    seen = {tuple(pos): 0}
    q = deque([tuple(pos)])
    while q:
        r, c = q.popleft()
        if r == goal:
            return seen[(r, c)]
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if not (0 <= nr < SIZE and 0 <= nc < SIZE):
                continue
            if (nr, nc) in seen or not can_step(r, c, nr, nc, h, v):
                continue
            seen[(nr, nc)] = seen[(r, c)] + 1
            q.append((nr, nc))
    return 99


def fmt(action):
    t = action.get("type")
    if t == "move":
        return "M({},{})".format(*action["to"])
    if t == "wall":
        return "W{}({},{})".format(action["orientation"], action["row"], action["col"])
    return t or "?"


def main(path, every=1):
    data = json.load(open(path, encoding="utf-8"))
    print("winner:", data.get("winner_name"), data.get("reason"), "turns:", data.get("total_turns"))
    for item in data["turns"]:
        st = item["state"]
        h = set(tuple(w) for w in st["walls"]["horizontal"])
        v = set(tuple(w) for w in st["walls"]["vertical"])
        d0 = dist(st["positions"][0], st["goal_rows"][0], h, v)
        d1 = dist(st["positions"][1], st["goal_rows"][1], h, v)
        w0, w1 = st["walls_remaining"]
        if item["turn"] % every == 0:
            print("{:>3} P{} {:<12} d0 {:>2} d1 {:>2}  w {} {}  h{} v{}".format(
                item["turn"], item["player"], fmt(item["action"]), d0, d1, w0, w1, len(h), len(v)))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
