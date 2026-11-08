"""Agent Wrongway: tìm kiếm alpha-beta + hàm đánh giá có trọng số học được.

Đổi tên file thành <MSSV>.py trước khi nộp.

Cách hoạt động:
1. Tự mô phỏng luật chơi (đi, nhảy, đi chéo, tường hợp lệ) để nhìn trước.
2. Tìm kiếm negamax alpha-beta, đào sâu dần (1, 2, 3... nước) tới khi hết giờ.
   Ứng viên mỗi nút: mọi nước đi + tối đa WALL_CANDIDATES tường chắn trên
   đường ngắn nhất của đối thủ (xếp theo lợi ích) để giữ tốc độ.
3. Hàm đánh giá = tổng có trọng số các đặc trưng (khoảng cách, số tường...).
   Trọng số WEIGHTS được học bằng thuật toán tiến hóa qua tự đấu
   (training/evolve.py) rồi ghi thẳng vào file này.

Ràng buộc nộp: một file, Python 3.9+, thư viện chuẩn, < 1 s mỗi nước,
import < 5 s, RAM <= 128 MB.
"""

import time
from collections import deque

# ---------------------------------------------------------------- 1. Cấu hình
SIZE = 9
TIME_BUDGET = 0.55       # giây cho mỗi nước (giới hạn thật 1 s, Colab chậm hơn máy)
MAX_DEPTH = 4            # độ sâu tối đa của đào sâu dần
WALL_CANDIDATES = 8      # số tường tốt nhất được xét ở mỗi nút
WIN = 10000.0
DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))

# Trọng số hàm đánh giá (nhìn từ người sắp đi):
#   race          : khoảng cách đối thủ - khoảng cách mình
#   walls         : số tường mình còn - số tường đối thủ còn
#   wall_urgency  : tường quý hơn khi đối thủ gần đích: wl_me/(d_opp+1) - wl_opp/(d_me+1)
#   safe_race     : +1 nếu đối thủ hết tường và mình về trước, -1 nếu ngược lại
#   my_dist       : khoảng cách của mình (tuyệt đối)
#   opp_dist      : khoảng cách của đối thủ (tuyệt đối)
# ======================= WEIGHTS BEGIN (tự sinh bởi training/evolve.py) =======================
WEIGHTS = {"race": 0.686, "walls": 1.7783, "wall_urgency": -0.2722, "safe_race": 4.4264, "my_dist": -0.1467, "opp_dist": -0.2988}
# ======================= WEIGHTS END =========================================================


class Timeout(Exception):
    pass


# ------------------------------------------------------- 2. Luật chơi
# Tường ngang H (r, c) chặn [r,c]-[r+1,c] và [r,c+1]-[r+1,c+1].
# Tường dọc  V (r, c) chặn [r,c]-[r,c+1] và [r+1,c]-[r+1,c+1].
def edge_open(a, b, hw, vw):
    """Hai ô kề nhau a -> b có đi qua được không (b phải nằm trong bàn)."""
    r, c = a
    r2, c2 = b
    if not (0 <= r2 < SIZE and 0 <= c2 < SIZE):
        return False
    if r != r2:
        er = r if r < r2 else r2
        return (er, c) not in hw and (er, c - 1) not in hw
    ec = c if c < c2 else c2
    return (r, ec) not in vw and (r - 1, ec) not in vw


def distance(start, goal_row, hw, vw):
    """BFS: số bước ít nhất tới goal_row, bỏ qua quân. -1 nếu bị bít."""
    if start[0] == goal_row:
        return 0
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        cell, d = queue.popleft()
        r, c = cell
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if nxt in seen or not edge_open(cell, nxt, hw, vw):
                continue
            if nxt[0] == goal_row:
                return d + 1
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return -1


def distance_map(goal_row, hw, vw):
    """Khoảng cách từ mọi ô tới goal_row (BFS ngược từ hàng đích)."""
    dist = [[-1] * SIZE for _ in range(SIZE)]
    queue = deque()
    for c in range(SIZE):
        dist[goal_row][c] = 0
        queue.append((goal_row, c))
    while queue:
        cell = queue.popleft()
        r, c = cell
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if edge_open(cell, nxt, hw, vw) and dist[nxt[0]][nxt[1]] < 0:
                dist[nxt[0]][nxt[1]] = dist[r][c] + 1
                queue.append(nxt)
    return dist


def shortest_path(start, goal_row, hw, vw):
    """Một đường ngắn nhất (list ô) từ start tới goal_row."""
    parent = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell[0] == goal_row:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            return path[::-1]
        r, c = cell
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if nxt not in parent and edge_open(cell, nxt, hw, vw):
                parent[nxt] = cell
                queue.append(nxt)
    return [start]


def legal_moves(me, opp, hw, vw):
    """Giống Arena: đi 1 ô; nhảy qua đối thủ; đi chéo khi phía sau bị chặn."""
    r, c = me
    out = set()
    for dr, dc in DIRS:
        adj = (r + dr, c + dc)
        if not edge_open(me, adj, hw, vw):
            continue
        if adj != opp:
            out.add(adj)
            continue
        beyond = (adj[0] + dr, adj[1] + dc)
        if edge_open(adj, beyond, hw, vw):
            out.add(beyond)
        else:
            for sdr, sdc in ((dc, dr), (-dc, -dr)):
                beside = (adj[0] + sdr, adj[1] + sdc)
                if edge_open(adj, beside, hw, vw):
                    out.add(beside)
    return sorted(out)


def wall_fits(o, r, c, hw, vw):
    """Không chồng lấn/cắt tường cũ (chưa kiểm tra việc bít đường)."""
    if not (0 <= r < SIZE - 1 and 0 <= c < SIZE - 1):
        return False
    if o == "H":
        return ((r, c - 1) not in hw and (r, c) not in hw and (r, c + 1) not in hw
                and (r, c) not in vw)
    return ((r - 1, c) not in vw and (r, c) not in vw and (r + 1, c) not in vw
            and (r, c) not in hw)


def walls_on_path(path):
    """Các tường (o, r, c) chặn ít nhất một bước của đường đi."""
    out = set()
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r1 != r2:                       # bước dọc -> tường ngang
            er = min(r1, r2)
            out.add(("H", er, c1))
            out.add(("H", er, c1 - 1))
        else:                              # bước ngang -> tường dọc
            ec = min(c1, c2)
            out.add(("V", r1, ec))
            out.add(("V", r1 - 1, ec))
    return out


# ------------------------------------------------------- 3. Đánh giá
def evaluate(p, dist, wl):
    """Điểm trạng thái nhìn từ người chơi p (người sắp đi)."""
    q = 1 - p
    dp, dq = dist[p], dist[q]
    w = WEIGHTS
    safe = 0
    if wl[q] == 0 and dp <= dq:
        safe = 1
    elif wl[p] == 0 and dq < dp:
        safe = -1
    return (w["race"] * (dq - dp)
            + w["walls"] * (wl[p] - wl[q])
            + w["wall_urgency"] * (wl[p] / (dq + 1.0) - wl[q] / (dp + 1.0))
            + w["safe_race"] * safe
            + w["my_dist"] * dp
            + w["opp_dist"] * dq)


# ------------------------------------------------------- 4. Tìm kiếm
class Search:
    def __init__(self, goals, deadline):
        self.goals = goals
        self.deadline = deadline

    def children(self, p, pos, hw, vw, wl, dist):
        """Danh sách (thứ tự, action, pos, hw, vw, wl, dist) hoặc 'WIN' nếu thắng ngay."""
        q = 1 - p
        goal = self.goals[p]
        dmap = distance_map(goal, hw, vw)
        out = []
        for cell in legal_moves(pos[p], pos[q], hw, vw):
            if cell[0] == goal:
                return "WIN", cell
            nd = list(dist)
            nd[p] = dmap[cell[0]][cell[1]]
            npos = list(pos)
            npos[p] = cell
            out.append((dist[q] - nd[p], ("move", cell), npos, hw, vw, wl, nd))

        if wl[p] > 0:
            walls = []
            for o, r, c in walls_on_path(shortest_path(pos[q], self.goals[q], hw, vw)):
                if not wall_fits(o, r, c, hw, vw):
                    continue
                nh = hw | {(r, c)} if o == "H" else hw
                nv = vw | {(r, c)} if o == "V" else vw
                dq2 = distance(pos[q], self.goals[q], nh, nv)
                if dq2 <= dist[q]:          # không làm chậm đối thủ (hoặc bít đường: -1)
                    continue
                dp2 = distance(pos[p], goal, nh, nv)
                if dp2 < 0:
                    continue
                nd = list(dist)
                nd[p], nd[q] = dp2, dq2
                nwl = list(wl)
                nwl[p] -= 1
                walls.append((dq2 - dp2, ("wall", (o, r, c)), pos, nh, nv, nwl, nd))
            walls.sort(key=lambda x: -x[0])
            out.extend(walls[:WALL_CANDIDATES])
        out.sort(key=lambda x: -x[0])
        return out, None

    def negamax(self, p, pos, hw, vw, wl, dist, depth, alpha, beta):
        if time.perf_counter() > self.deadline:
            raise Timeout()
        if depth == 0:
            return evaluate(p, dist, wl)
        kids, _ = self.children(p, pos, hw, vw, wl, dist)
        if kids == "WIN":
            return WIN + depth               # thắng sớm hơn thì tốt hơn
        if not kids:
            return evaluate(p, dist, wl)
        best = -float("inf")
        for _, _, npos, nh, nv, nwl, nd in kids:
            v = -self.negamax(1 - p, npos, nh, nv, nwl, nd, depth - 1, -beta, -alpha)
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        return best


def to_dict(action):
    kind, data = action
    if kind == "move":
        return {"type": "move", "to": [data[0], data[1]]}
    o, r, c = data
    return {"type": "wall", "orientation": o, "row": r, "col": c}


def choose_action(observation):
    deadline = time.perf_counter() + TIME_BUDGET
    try:
        me = observation["player_id"]
        you, opp = observation["you"], observation["opponent"]
        pos = [None, None]
        goals = [0, 0]
        wl = [0, 0]
        pos[me], pos[1 - me] = tuple(you["position"]), tuple(opp["position"])
        goals[me], goals[1 - me] = you["goal_row"], opp["goal_row"]
        wl[me], wl[1 - me] = you["walls_remaining"], opp["walls_remaining"]
        hw = frozenset(tuple(w) for w in observation["walls"]["horizontal"])
        vw = frozenset(tuple(w) for w in observation["walls"]["vertical"])
        dist = [distance(pos[i], goals[i], hw, vw) for i in (0, 1)]

        search = Search(goals, deadline)
        kids, win_cell = search.children(me, pos, hw, vw, wl, dist)
        if kids == "WIN":
            return to_dict(("move", win_cell))

        # Chỉ giữ ứng viên mà Arena xác nhận hợp lệ (an toàn tuyệt đối).
        legal_m = {tuple(m) for m in observation["legal_moves"]}
        legal_w = {(w["orientation"], w["row"], w["col"]) for w in observation["legal_walls"]}
        kids = [k for k in kids if (k[1][1] in legal_m if k[1][0] == "move" else k[1][1] in legal_w)]
        if kids:
            best = kids[0]
            for depth in range(1, MAX_DEPTH + 1):
                try:
                    alpha, cur_best, cur_val = -float("inf"), None, -float("inf")
                    for kid in kids:
                        _, _, npos, nh, nv, nwl, nd = kid
                        v = -search.negamax(1 - me, npos, nh, nv, nwl, nd, depth - 1,
                                            -float("inf"), -alpha)
                        if v > cur_val:
                            cur_val, cur_best = v, kid
                            alpha = max(alpha, v)
                except Timeout:
                    break
                best = cur_best
                kids.remove(best)
                kids.insert(0, best)         # xét nước tốt nhất trước ở độ sâu kế tiếp
                if cur_val >= WIN:
                    break                    # đã thấy đường thắng chắc
            return to_dict(best[1])
    except Exception:
        pass  # Không để lỗi làm mất lượt; rơi về nước an toàn.

    moves = observation.get("legal_moves") or []
    if moves:
        d = observation["distances"]
        r, c = min(moves, key=lambda m: d[m[0]][m[1]] if d[m[0]][m[1]] >= 0 else 99)
        return {"type": "move", "to": [r, c]}
    return {"type": "pass"}
