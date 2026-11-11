"""Bot "bot": bản Python của AI 'hard' trong bot.js (game gốc kiểu Quoridor).

Port từ hàm aiHard() của bot.js sang luật Wrongway:
- Mô hình tường giống hệt: H (r, c) chặn giữa hàng r và r+1 ở cột c, c+1;
  V (r, c) chặn giữa cột c và c+1 ở hàng r, r+1. Luật nhảy qua đối thủ cũng giống.
- Bỏ các phần chỉ có trong game gốc: hộp Chaos, búa phá tường, tường rơi, replay.

Chiến thuật (giữ nguyên tinh thần aiHard):
1. Thắng ngay nếu một bước tới đích.
2. "Nước rút cuối": hết tường, hoặc đường ngắn và đang dẫn -> chạy thẳng về đích.
3. Còn lại: minimax alpha-beta 3 nửa-nước (mình -> đối thủ -> mình -> đánh giá).
   Ứng viên tường chỉ lấy dọc 6 ô đầu đường đi ngắn nhất của đối thủ, giữ tường có lợi
   (làm đối thủ chậm hơn mình) và chỉ giữ vài tường tốt nhất mỗi tầng.
   Ở gốc cộng thêm: thưởng bước theo đường BFS, thưởng tiến/phạt lùi, phạt quay lại ô cũ.

Khác bot.js: có hạn giờ TIME_BUDGET; nước gốc nào chưa tính xong khi hết giờ thì bỏ
(bot.js gốc không giới hạn thời gian, JS cũng nhanh hơn Python nhiều).

Chỉ dùng thư viện chuẩn, chạy được chỉ từ observation (biến toàn cục _history chỉ
dùng để phạt đi vòng lặp, mất cũng không sao).
"""

from collections import deque
from time import perf_counter

N = 9                    # kích thước bàn
TIME_BUDGET = 0.5        # giây cho mỗi nước (giới hạn thật 1 s)
WIN = 50000              # điểm thắng/thua tuyệt đối

# Lịch sử vị trí của mình theo player_id (để phạt đi qua lại). Có thể mất khi restart.
_history = {}


class TimeUp(Exception):
    """Hết giờ giữa chừng một nhánh tìm kiếm."""


# ------------------------------------------------------------------ luật bàn cờ
def edge_open(walls, r, c, nr, nc):
    """Hai ô kề nhau (r,c) và (nr,nc) có bị tường chắn không (giống environment.py)."""
    if not (0 <= nr < N and 0 <= nc < N):
        return False
    hw, vw = walls
    if nr != r:                              # đi dọc: xem tường ngang
        er = min(r, nr)
        return (er, c) not in hw and (er, c - 1) not in hw
    ec = min(c, nc)                          # đi ngang: xem tường dọc
    return (r, ec) not in vw and (r - 1, ec) not in vw


def bfs_path(pos, walls, goal_row):
    """Đường ngắn nhất (bỏ qua quân) từ pos tới hàng đích, dạng list ô; None nếu bị nhốt.

    Thứ tự duyệt: tiến về đích trước, rồi sang ngang, cuối cùng lùi -> đường đi thẳng.
    """
    if pos[0] == goal_row:
        return [pos]
    fwd = 1 if goal_row > pos[0] else -1
    dirs = ((fwd, 0), (0, -1), (0, 1), (-fwd, 0))
    parent = {pos: None}
    queue = deque([pos])
    while queue:
        cur = queue.popleft()
        r, c = cur
        for dr, dc in dirs:
            nxt = (r + dr, c + dc)
            if nxt in parent or not edge_open(walls, r, c, nxt[0], nxt[1]):
                continue
            parent[nxt] = cur
            if nxt[0] == goal_row:           # tới đích: lần ngược lại đường đi
                path = [nxt]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(nxt)
    return None


def path_len(pos, walls, goal_row):
    path = bfs_path(pos, walls, goal_row)
    return None if path is None else len(path) - 1


def get_moves(pos, opp, walls):
    """Các ô đi được (gồm nhảy thẳng qua đối thủ / nhảy chéo khi sau lưng có tường)."""
    moves = []
    r, c = pos
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if not edge_open(walls, r, c, nr, nc):
            continue
        if (nr, nc) != opp:
            moves.append((nr, nc))
            continue
        jr, jc = nr + dr, nc + dc            # nhảy thẳng qua đối thủ
        if edge_open(walls, nr, nc, jr, jc):
            moves.append((jr, jc))
            continue
        for sr, sc in ((dc, dr), (-dc, -dr)):  # sau lưng bị chặn -> sang hai bên đối thủ
            kr, kc = nr + sr, nc + sc
            if edge_open(walls, nr, nc, kr, kc):
                moves.append((kr, kc))
    return moves


def place_wall(walls, orient, r, c):
    """Tường mới (chưa kiểm tra đường đi) hoặc None nếu chồng/cắt tường cũ."""
    if not (0 <= r < N - 1 and 0 <= c < N - 1):
        return None
    hw, vw = walls
    if orient == "H":
        if (r, c - 1) in hw or (r, c) in hw or (r, c + 1) in hw or (r, c) in vw:
            return None
        return (hw | {(r, c)}, vw)
    if (r - 1, c) in vw or (r, c) in vw or (r + 1, c) in vw or (r, c) in hw:
        return None
    return (hw, vw | {(r, c)})


def wall_candidates(target, walls, depth, goal_row):
    """Tường quanh `depth` ô đầu tiên trên đường ngắn nhất của target (như bot.js)."""
    path = bfs_path(target, walls, goal_row)
    if path is None:
        return []
    result = {}                              # dict giữ thứ tự chèn, không trùng
    for (pr, pc) in path[:depth]:
        for dr in (-1, 0):
            for dc in (-1, 0):
                r, c = pr + dr, pc + dc
                if 0 <= r < N - 1 and 0 <= c < N - 1:
                    result[("H", r, c)] = True
                    result[("V", r, c)] = True
    return list(result)


def is_forward(frm, to, goal_row):
    return to[0] > frm[0] if goal_row > frm[0] else to[0] < frm[0]


def is_backward(frm, to, goal_row):
    return to[0] < frm[0] if goal_row > frm[0] else to[0] > frm[0]


# ------------------------------------------------------------------ bộ tìm kiếm
class Searcher:
    """Minimax alpha-beta của aiHard. 'ai' = mình, 'hum' = đối thủ."""

    def __init__(self, ai_goal, hum_goal, deadline):
        self.ai_goal = ai_goal
        self.hum_goal = hum_goal
        self.deadline = deadline

    def evaluate(self, ap, hp, w, ab, hb):
        """Điểm tĩnh, dương = tốt cho mình (công thức evalState của bot.js)."""
        ai_d = path_len(ap, w, self.ai_goal)
        hum_d = path_len(hp, w, self.hum_goal)
        if ai_d is None:
            return -99999
        if hum_d is None:
            return 99999
        if ai_d == 0:
            return WIN
        if hum_d == 0:
            return -WIN
        # Khoảng cách tương đối là chính; thưởng mạnh tiến độ của mình; tường dự trữ là phụ.
        return (hum_d - ai_d) * 25 - ai_d * 3.5 + hum_d * 0.8 + (ab - hb) * 3

    def gen_actions(self, ap, hp, w, ab, hb, for_ai, max_walls):
        """Nước đi + tối đa max_walls tường có lợi, sắp theo độ ưu tiên (cho alpha-beta)."""
        me, opp = (ap, hp) if for_ai else (hp, ap)
        my_barr = ab if for_ai else hb
        me_goal, opp_goal = ((self.ai_goal, self.hum_goal) if for_ai
                             else (self.hum_goal, self.ai_goal))
        acts = []
        my_path = bfs_path(me, w, me_goal)
        for m in get_moves(me, opp, w):
            pri = 0
            if my_path and len(my_path) > 1 and my_path[1] == m:
                pri = 30                       # đúng bước trên đường ngắn nhất
            if is_forward(me, m, me_goal):
                pri += 8
            acts.append((pri, "move", m))
        if my_barr > 0 and my_path is not None:
            old_opp = path_len(opp, w, opp_goal) or 0
            my_old = len(my_path) - 1
            scored = []
            for orient, r, c in wall_candidates(opp, w, 6, opp_goal):
                nxt = place_wall(w, orient, r, c)
                if nxt is None:
                    continue
                # Tường hợp lệ khi cả hai còn đường tới đích (tryWall của bot.js).
                opp_new = path_len(opp, nxt, opp_goal)
                if opp_new is None:
                    continue
                my_new = path_len(me, nxt, me_goal)
                if my_new is None:
                    continue
                gain = (opp_new - old_opp) - (my_new - my_old)
                if gain > 0:                   # chỉ giữ tường thực sự có lợi
                    scored.append((gain, nxt, (orient, r, c)))
            scored.sort(key=lambda x: -x[0])
            for gain, nxt, key in scored[:max_walls]:
                acts.append((gain * 8, "wall", (nxt, key)))
        acts.sort(key=lambda a: -a[0])         # sort ổn định như JS
        return acts

    def search(self, ap, hp, w, ab, hb, depth, maximizing, alpha, beta):
        if ap[0] == self.ai_goal:
            return WIN + depth                 # thắng càng sớm càng tốt
        if hp[0] == self.hum_goal:
            return -WIN - depth
        if perf_counter() > self.deadline:
            raise TimeUp()
        if depth == 0:
            return self.evaluate(ap, hp, w, ab, hb)
        max_w = 6 if depth >= 2 else 4
        acts = self.gen_actions(ap, hp, w, ab, hb, maximizing, max_w)
        if not acts:
            return self.evaluate(ap, hp, w, ab, hb)
        if maximizing:
            best = float("-inf")
            for _, kind, data in acts:
                if kind == "move":
                    s = self.search(data, hp, w, ab, hb, depth - 1, False, alpha, beta)
                else:
                    s = self.search(ap, hp, data[0], ab - 1, hb, depth - 1, False, alpha, beta)
                best = max(best, s)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best
        best = float("inf")
        for _, kind, data in acts:
            if kind == "move":
                s = self.search(ap, data, w, ab, hb, depth - 1, True, alpha, beta)
            else:
                s = self.search(ap, hp, data[0], ab, hb - 1, depth - 1, True, alpha, beta)
            best = min(best, s)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


# ------------------------------------------------------------------ quyết định
def best_step(me, opp, walls, goal_row):
    """Bước hợp lệ làm đường tới đích ngắn nhất (thay moveTowardGoal của bot.js)."""
    best, best_d = None, None
    for m in get_moves(me, opp, walls):
        d = path_len(m, walls, goal_row)
        if d is not None and (best_d is None or d < best_d):
            best, best_d = m, d
    return best


def recent_positions(observation, me):
    """recentAi của bot.js: ô hiện tại trước, rồi các ô cũ của mình (mới nhất trước)."""
    pid = observation.get("player_id", 0)
    turn = observation.get("turn", 0)
    hist = _history.get(pid)
    if hist is None or turn <= hist["turn"]:   # ván mới (hoặc bot bị restart)
        hist = {"turn": turn, "cells": []}
        _history[pid] = hist
    hist["turn"] = turn
    if not hist["cells"] or hist["cells"][-1] != me:
        hist["cells"].append(me)
    del hist["cells"][:-6]
    return list(reversed(hist["cells"]))


def decide(observation):
    start = perf_counter()
    you, opp_info = observation["you"], observation["opponent"]
    ai_pos, hum_pos = tuple(you["position"]), tuple(opp_info["position"])
    ai_goal, hum_goal = you["goal_row"], opp_info["goal_row"]
    ai_barr, hum_barr = you["walls_remaining"], opp_info["walls_remaining"]
    walls = (frozenset(tuple(x) for x in observation["walls"]["horizontal"]),
             frozenset(tuple(x) for x in observation["walls"]["vertical"]))
    recent = recent_positions(observation, ai_pos)

    moves = get_moves(ai_pos, hum_pos, walls)
    # 1) Thắng ngay.
    for m in moves:
        if m[0] == ai_goal:
            return ("move", m)

    # 2) Nước rút cuối: hết tường, hoặc đường ngắn và đang dẫn rõ.
    ai_path = bfs_path(ai_pos, walls, ai_goal)
    hum_d = path_len(hum_pos, walls, hum_goal)
    hum_d = 999 if hum_d is None else hum_d
    if ai_path and len(ai_path) > 1:
        ai_d = len(ai_path) - 1
        if ai_barr <= 0 or (ai_d <= 3 and hum_d > ai_d + 1) or hum_d > ai_d + 2:
            m = best_step(ai_pos, hum_pos, walls, ai_goal)
            if m is not None:
                return ("move", m)

    # 3) Minimax: gốc xét mọi nước + 10 tường tốt nhất, mỗi nhánh sâu thêm 2 nửa-nước.
    searcher = Searcher(ai_goal, hum_goal, start + TIME_BUDGET)
    best_score, best_action = float("-inf"), None
    for _, kind, data in searcher.gen_actions(ai_pos, hum_pos, walls, ai_barr, hum_barr, True, 10):
        try:
            if kind == "move":
                s = searcher.search(data, hum_pos, walls, ai_barr, hum_barr, 2, False,
                                    float("-inf"), float("inf"))
            else:
                s = searcher.search(ai_pos, hum_pos, data[0], ai_barr - 1, hum_barr, 2, False,
                                    float("-inf"), float("inf"))
        except TimeUp:
            break                              # nước này chưa tính xong -> bỏ, dùng nước tốt nhất đã có
        if kind == "move":
            # Thưởng bước theo đường BFS chính (chống đi zíc-zắc giữa các đường bằng nhau).
            if ai_path and len(ai_path) > 1 and ai_path[1] == data:
                s += 20
            if is_forward(ai_pos, data, ai_goal):
                s += 6
            elif is_backward(ai_pos, data, ai_goal):
                s -= 4
            # Phạt quay lại ô vừa đi qua: -60 / -48 / -36 / -24.
            for i in range(1, min(len(recent), 5)):
                if recent[i] == data:
                    s -= (6 - i) * 12
                    break
        if s > best_score:
            best_score, best_action = s, (kind, data)

    if best_action is None:                    # không tính kịp gì: đi theo đường ngắn nhất
        m = best_step(ai_pos, hum_pos, walls, ai_goal)
        return ("move", m) if m is not None else None
    if best_action[0] == "move":
        return best_action
    return ("wall", best_action[1][1])


def choose_action(observation):
    legal_moves = observation.get("legal_moves") or []
    try:
        decision = decide(observation)
        if decision is not None:
            kind, data = decision
            if kind == "move" and list(data) in legal_moves:
                return {"type": "move", "to": list(data)}
            if kind == "wall":
                orient, r, c = data
                wall = {"orientation": orient, "row": r, "col": c}
                if wall in (observation.get("legal_walls") or []):
                    return {"type": "wall", "orientation": orient, "row": r, "col": c}
    except Exception:
        pass
    # Dự phòng: ô hợp lệ gần đích nhất theo bảng distances của Arena.
    dist = observation.get("distances")
    if legal_moves:
        best = legal_moves[0]
        if dist:
            reachable = [m for m in legal_moves if dist[m[0]][m[1]] >= 0]
            if reachable:
                best = min(reachable, key=lambda m: dist[m[0]][m[1]])
        return {"type": "move", "to": list(best)}
    return {"type": "pass"}
