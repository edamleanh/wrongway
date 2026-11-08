"""Agent Wrongway: tìm kiếm alpha-beta + hàm đánh giá có trọng số học được.

Đổi tên file thành <MSSV>.py trước khi nộp.

Cách hoạt động:
1. Tự mô phỏng luật chơi (đi, nhảy, đi chéo, tường hợp lệ) để nhìn trước.
   Ô cờ được đánh số 0..80 (= hàng*9 + cột); cạnh bị tường chặn lưu trong
   một frozenset số nguyên -> BFS nhanh, tìm kiếm được sâu hơn.
2. Tìm kiếm negamax alpha-beta, đào sâu dần (1, 2, 3... nước) tới khi hết giờ,
   có bảng chuyển vị (transposition table) để không tính lại thế cờ trùng.
   Ứng viên mỗi nút: mọi nước đi + tối đa WALL_CANDIDATES tường chắn trên
   đường ngắn nhất của đối thủ (xếp theo lợi ích).
3. Hàm đánh giá = tổng có trọng số các đặc trưng (khoảng cách, số tường...).
   Trọng số WEIGHTS được học bằng thuật toán tiến hóa qua tự đấu
   (training/evolve.py) rồi ghi thẳng vào file này.

Ràng buộc nộp: một file, Python 3.9+, thư viện chuẩn, < 1 s mỗi nước,
import < 5 s, RAM <= 128 MB.
"""

import time

# ---------------------------------------------------------------- 1. Cấu hình
SIZE = 9
N = SIZE * SIZE
TIME_BUDGET = 0.55       # giây cho mỗi nước (giới hạn thật 1 s, Colab chậm hơn máy)
MAX_DEPTH = 6            # độ sâu tối đa của đào sâu dần
WALL_CANDIDATES = 8      # số tường tốt nhất được xét ở mỗi nút
ALLOW_PASS = True        # xét cả nước đứng yên {"type": "pass"} (hợp lệ theo luật)
WIN = 10000.0
DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))

# Trọng số hàm đánh giá (nhìn từ người sắp đi):
#   race          : khoảng cách đối thủ - khoảng cách mình
#   walls         : số tường mình còn - số tường đối thủ còn
#   wall_urgency  : tường quý hơn khi đối thủ gần đích: wl_me/(d_opp+1) - wl_opp/(d_me+1)
#   safe_race     : +1 nếu đối thủ hết tường và mình về trước, -1 nếu ngược lại
#   my_dist       : khoảng cách của mình (tuyệt đối)
#   opp_dist      : khoảng cách của đối thủ (tuyệt đối)
#   pass_penalty  : điểm trừ khi chọn đứng yên ở gốc (chống chờ quá nhiều)
# ======================= WEIGHTS BEGIN (tự sinh bởi training/evolve.py) =======================
WEIGHTS = {"race": 1.4142, "walls": 1.8144, "wall_urgency": -0.3837, "safe_race": 5.2611, "my_dist": 0.0484, "opp_dist": -0.5199, "pass_penalty": 0.4689}
# ======================= WEIGHTS END =========================================================


class Timeout(Exception):
    pass


# ------------------------------------------------------- 2. Bảng tính sẵn
ROW = [i // SIZE for i in range(N)]
COL = [i % SIZE for i in range(N)]
# STEP[cell][k] = ô kế bên theo hướng DIRS[k], -1 nếu ra ngoài bàn.
STEP = []
for _i in range(N):
    _r, _c = divmod(_i, SIZE)
    STEP.append([(_r + dr) * SIZE + _c + dc if 0 <= _r + dr < SIZE and 0 <= _c + dc < SIZE else -1
                 for dr, dc in DIRS])
NEI = [[b for b in STEP[i] if b >= 0] for i in range(N)]
# Hai hướng vuông góc với mỗi hướng (dùng cho đi chéo quanh đối thủ).
SIDE = {0: (1, 2), 3: (1, 2), 1: (0, 3), 2: (0, 3)}


def cell(r, c):
    return r * SIZE + c


def wall_keys(o, r, c):
    """4 khóa cạnh (a*81+b, cả hai chiều) mà tường (o, r, c) chặn."""
    if o == "H":
        pairs = ((cell(r, c), cell(r + 1, c)), (cell(r, c + 1), cell(r + 1, c + 1)))
    else:
        pairs = ((cell(r, c), cell(r, c + 1)), (cell(r + 1, c), cell(r + 1, c + 1)))
    return [k for a, b in pairs for k in (a * N + b, b * N + a)]


def blocked_from(hw, vw):
    keys = []
    for r, c in hw:
        keys.extend(wall_keys("H", r, c))
    for r, c in vw:
        keys.extend(wall_keys("V", r, c))
    return frozenset(keys)


# ------------------------------------------------------- 3. Luật chơi
def distance(start, goal_row, blocked):
    """BFS theo tầng: số bước ít nhất tới goal_row, bỏ qua quân. -1 nếu bị bít."""
    if ROW[start] == goal_row:
        return 0
    seen = bytearray(N)
    seen[start] = 1
    frontier = [start]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for a in frontier:
            base = a * N
            for b in NEI[a]:
                if not seen[b] and base + b not in blocked:
                    if ROW[b] == goal_row:
                        return d
                    seen[b] = 1
                    nxt.append(b)
        frontier = nxt
    return -1


def distance_map(goal_row, blocked):
    """Khoảng cách từ mọi ô tới goal_row (list 81 phần tử, -1 = không tới)."""
    dist = [-1] * N
    frontier = [cell(goal_row, c) for c in range(SIZE)]
    for a in frontier:
        dist[a] = 0
    d = 0
    while frontier:
        d += 1
        nxt = []
        for a in frontier:
            for b in NEI[a]:
                if dist[b] < 0 and b * N + a not in blocked:
                    dist[b] = d
                    nxt.append(b)
        frontier = nxt
    return dist


def shortest_path(start, goal_row, blocked):
    """Một đường ngắn nhất (list ô) từ start tới goal_row."""
    parent = {start: -1}
    frontier = [start]
    while frontier:
        nxt = []
        for a in frontier:
            if ROW[a] == goal_row:
                path = []
                while a >= 0:
                    path.append(a)
                    a = parent[a]
                return path[::-1]
            base = a * N
            for b in NEI[a]:
                if b not in parent and base + b not in blocked:
                    parent[b] = a
                    nxt.append(b)
        frontier = nxt
    return [start]


def legal_moves(me, opp, blocked):
    """Giống Arena: đi 1 ô; nhảy qua đối thủ; đi chéo khi phía sau bị chặn."""
    out = set()
    for k in range(4):
        adj = STEP[me][k]
        if adj < 0 or me * N + adj in blocked:
            continue
        if adj != opp:
            out.add(adj)
            continue
        beyond = STEP[adj][k]
        if beyond >= 0 and adj * N + beyond not in blocked:
            out.add(beyond)
        else:
            for s in SIDE[k]:
                beside = STEP[adj][s]
                if beside >= 0 and adj * N + beside not in blocked:
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
    for a, b in zip(path, path[1:]):
        r1, c1, r2, c2 = ROW[a], COL[a], ROW[b], COL[b]
        if r1 != r2:                       # bước dọc -> tường ngang
            er = min(r1, r2)
            out.add(("H", er, c1))
            out.add(("H", er, c1 - 1))
        else:                              # bước ngang -> tường dọc
            ec = min(c1, c2)
            out.add(("V", r1, ec))
            out.add(("V", r1 - 1, ec))
    return out


# ------------------------------------------------------- 4. Đánh giá
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


# ------------------------------------------------------- 5. Tìm kiếm
EXACT, LOWER, UPPER = 0, 1, 2


class Search:
    def __init__(self, goals, deadline):
        self.goals = goals
        self.deadline = deadline
        self.tt = {}
        self.nodes = 0

    def children(self, p, pos, hw, vw, blocked, wl, dist):
        """List (thứ tự, action, pos, hw, vw, blocked, wl, dist) hoặc ('WIN', ô)."""
        q = 1 - p
        goal = self.goals[p]
        dmap = distance_map(goal, blocked)
        out = []
        for c in legal_moves(pos[p], pos[q], blocked):
            if ROW[c] == goal:
                return "WIN", c
            nd = list(dist)
            nd[p] = dmap[c]
            npos = list(pos)
            npos[p] = c
            out.append((dist[q] - nd[p], ("move", c), npos, hw, vw, blocked, wl, nd))

        if wl[p] > 0:
            walls = []
            goal_q = self.goals[q]
            for o, r, c in walls_on_path(shortest_path(pos[q], goal_q, blocked)):
                if not wall_fits(o, r, c, hw, vw):
                    continue
                nb = blocked.union(wall_keys(o, r, c))
                dq2 = distance(pos[q], goal_q, nb)
                if dq2 <= dist[q]:          # không làm chậm đối thủ (hoặc bít đường: -1)
                    continue
                dp2 = distance(pos[p], goal, nb)
                if dp2 < 0:
                    continue
                nh = hw | {(r, c)} if o == "H" else hw
                nv = vw | {(r, c)} if o == "V" else vw
                nd = list(dist)
                nd[p], nd[q] = dp2, dq2
                nwl = list(wl)
                nwl[p] -= 1
                walls.append((dq2 - dp2, ("wall", (o, r, c)), pos, nh, nv, nb, nwl, nd))
            walls.sort(key=lambda x: -x[0])
            out.extend(walls[:WALL_CANDIDATES])
        out.sort(key=lambda x: -x[0])
        return out, None

    def negamax(self, p, pos, hw, vw, blocked, wl, dist, depth, alpha, beta):
        self.nodes += 1
        if (self.nodes & 63) == 0 and time.perf_counter() > self.deadline:
            raise Timeout()
        if depth == 0:
            return evaluate(p, dist, wl)

        key = (p, pos[0], pos[1], wl[0], wl[1], blocked)
        entry = self.tt.get(key)
        best_action = None
        if entry is not None:
            e_depth, e_val, e_flag, best_action = entry
            if e_depth >= depth:
                if e_flag == EXACT:
                    return e_val
                if e_flag == LOWER and e_val >= beta:
                    return e_val
                if e_flag == UPPER and e_val <= alpha:
                    return e_val

        kids, _ = self.children(p, pos, hw, vw, blocked, wl, dist)
        if kids == "WIN":
            return WIN + depth               # thắng sớm hơn thì tốt hơn
        if not kids:
            return evaluate(p, dist, wl)
        if best_action is not None:          # nước tốt nhất lần trước xét đầu tiên
            for i, k in enumerate(kids):
                if k[1] == best_action:
                    kids.insert(0, kids.pop(i))
                    break

        alpha0 = alpha
        best, best_act = -float("inf"), None
        for _, act, npos, nh, nv, nb, nwl, nd in kids:
            v = -self.negamax(1 - p, npos, nh, nv, nb, nwl, nd, depth - 1, -beta, -alpha)
            if v > best:
                best, best_act = v, act
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        flag = UPPER if best <= alpha0 else LOWER if best >= beta else EXACT
        self.tt[key] = (depth, best, flag, best_act)
        return best


def to_dict(action):
    kind, data = action
    if kind == "pass":
        return {"type": "pass"}
    if kind == "move":
        return {"type": "move", "to": [ROW[data], COL[data]]}
    o, r, c = data
    return {"type": "wall", "orientation": o, "row": r, "col": c}


LAST_INFO = {}   # thông tin lần tìm kiếm gần nhất (độ sâu, số nút) để debug


def choose_action(observation):
    deadline = time.perf_counter() + TIME_BUDGET
    try:
        me = observation["player_id"]
        you, opp = observation["you"], observation["opponent"]
        pos, goals, wl = [0, 0], [0, 0], [0, 0]
        pos[me], pos[1 - me] = cell(*you["position"]), cell(*opp["position"])
        goals[me], goals[1 - me] = you["goal_row"], opp["goal_row"]
        wl[me], wl[1 - me] = you["walls_remaining"], opp["walls_remaining"]
        hw = frozenset(tuple(w) for w in observation["walls"]["horizontal"])
        vw = frozenset(tuple(w) for w in observation["walls"]["vertical"])
        blocked = blocked_from(hw, vw)
        dist = [distance(pos[i], goals[i], blocked) for i in (0, 1)]

        search = Search(goals, deadline)
        kids, win_cell = search.children(me, pos, hw, vw, blocked, wl, dist)
        if kids == "WIN":
            return to_dict(("move", win_cell))
        if ALLOW_PASS:   # đứng yên: chỉ xét cho mình ở gốc (không làm phình cây tìm kiếm)
            kids.append((-999, ("pass", None), pos, hw, vw, blocked, wl, dist))

        # Chỉ giữ ứng viên mà Arena xác nhận hợp lệ (an toàn tuyệt đối).
        legal_m = {cell(*m) for m in observation["legal_moves"]}
        legal_w = {(w["orientation"], w["row"], w["col"]) for w in observation["legal_walls"]}
        kids = [k for k in kids if k[1][0] == "pass" or
                (k[1][1] in legal_m if k[1][0] == "move" else k[1][1] in legal_w)]
        if kids:
            best, depth_done = kids[0], 0
            for depth in range(1, MAX_DEPTH + 1):
                try:
                    alpha, cur_best = -float("inf"), None
                    for kid in kids:
                        _, _, npos, nh, nv, nb, nwl, nd = kid
                        v = -search.negamax(1 - me, npos, nh, nv, nb, nwl, nd, depth - 1,
                                            -float("inf"), -alpha)
                        if kid[1][0] == "pass":
                            v -= WEIGHTS.get("pass_penalty", 0.0)
                        if v > alpha:
                            alpha, cur_best = v, kid
                except Timeout:
                    break
                best, depth_done = cur_best, depth
                kids.remove(best)
                kids.insert(0, best)         # xét nước tốt nhất trước ở độ sâu kế tiếp
                if alpha >= WIN:
                    break                    # đã thấy đường thắng chắc
            LAST_INFO.update(depth=depth_done, nodes=search.nodes)
            return to_dict(best[1])
    except Exception:
        pass  # Không để lỗi làm mất lượt; rơi về nước an toàn.

    moves = observation.get("legal_moves") or []
    if moves:
        d = observation["distances"]
        r, c = min(moves, key=lambda m: d[m[0]][m[1]] if d[m[0]][m[1]] >= 0 else 99)
        return {"type": "move", "to": [r, c]}
    return {"type": "pass"}
