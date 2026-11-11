"""Agent Wrongway: tìm kiếm alpha-beta + hàm đánh giá có trọng số học được.

Đổi tên file thành <MSSV>.py trước khi nộp.

Cách hoạt động:
1. Tự mô phỏng luật chơi (đi, nhảy, đi chéo, tường hợp lệ) để nhìn trước.
   Ô cờ được đánh số 0..80 (= hàng*9 + cột); cạnh bị tường chặn lưu trong
   một frozenset số nguyên. BFS tính khoảng cách dùng BITBOARD (như Stockfish):
   cả bàn là một số nguyên 81 bit, mỗi bước loang ra mọi ô cùng lúc bằng phép
   dịch bit -> nhanh gấp ~2 lần duyệt từng ô, nên nhìn sâu hơn trong cùng thời gian.
2. Tìm kiếm negamax alpha-beta, đào sâu dần (1, 2, 3... nước) tới khi hết giờ,
   có bảng chuyển vị (transposition table) để không tính lại thế cờ trùng.
   Hết giờ giữa một độ sâu: vẫn dùng nước đã được chứng minh tốt hơn nước cũ.
   Ứng viên mỗi nút: mọi nước đi + tối đa WALL_CANDIDATES tường chắn trên
   đường ngắn nhất của đối thủ (xếp theo lợi ích). Ở gốc thêm "tường phòng thủ":
   chiếm chỗ các bức tường nguy hiểm nhất mà đối thủ có thể đặt để nhốt mình.
3. Hàm đánh giá = tổng có trọng số các đặc trưng (khoảng cách, số tường...).
   Có đặc trưng "độ dễ bị chặn" (vuln): một bức tường của đối thủ làm mình đi
   thêm tối đa bao nhiêu bước.
   Trọng số WEIGHTS được học bằng thuật toán tiến hóa qua tự đấu
   (training/evolve.py) rồi ghi thẳng vào file này.

Ràng buộc nộp: một file, Python 3.9+, thư viện chuẩn, < 1 s mỗi nước,
import < 5 s, RAM <= 128 MB.
"""

import time
from itertools import chain

# ---------------------------------------------------------------- 1. Cấu hình
SIZE = 9
N = SIZE * SIZE
TIME_BUDGET = 0.55       # giây cho mỗi nước (giới hạn thật 1 s, Colab chậm hơn máy)
MAX_DEPTH = 6            # độ sâu tối đa của đào sâu dần
WALL_CANDIDATES = 8      # số tường tốt nhất được xét ở mỗi nút
DEFENSE_THREATS = 3      # số tường nguy hiểm nhất của đối thủ cần tìm cách chặn trước (ở gốc)
VULN_STEPS = 4           # độ dễ bị chặn: chỉ thử tường trên VULN_STEPS bước đầu của đường đi
WIN = 10000.0
DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))

# Trọng số hàm đánh giá (nhìn từ người sắp đi):
#   race          : khoảng cách đối thủ - khoảng cách mình
#   walls         : số tường mình còn - số tường đối thủ còn
#   wall_urgency  : tường quý hơn khi đối thủ gần đích: wl_me/(d_opp+1) - wl_opp/(d_me+1)
#   safe_race     : +1 nếu đối thủ hết tường và mình về trước, -1 nếu ngược lại
#   my_dist       : khoảng cách của mình (tuyệt đối)
#   opp_dist      : khoảng cách của đối thủ (tuyệt đối)
#   vuln          : độ "dễ bị chặn": số bước bức tường tốt nhất của mình làm đối thủ
#                   chậm đi trừ số bước bức tường tốt nhất của đối thủ làm mình chậm đi
# ======================= WEIGHTS BEGIN (tự sinh bởi training/evolve.py) =======================
WEIGHTS = {"race": 1.3805, "walls": 2.0643, "wall_urgency": -0.0789, "safe_race": 6.8451, "my_dist": -0.0117, "opp_dist": -0.4182, "vuln": 1.0}
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
# NEIK[a] = [(ô kề b, khóa cạnh a->b)]: tính sẵn để BFS khỏi phải nhân/cộng mỗi lần.
NEIK = [[(b, a * N + b) for b in NEI[a]] for a in range(N)]
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


# Bitboard (giống Stockfish): cả bàn cờ là một số nguyên 81 bit, bit thứ i = ô i.
# Một bước BFS cho MỌI ô cùng lúc = vài phép dịch bit, thay vì duyệt từng ô.
#   dịch >> 9 = lên một hàng, << 9 = xuống một hàng, >> 1 = sang trái, << 1 = sang phải.
ROW_BITS = [sum(1 << (r * SIZE + c) for c in range(SIZE)) for r in range(SIZE)]
# 4 bitboard "đi được" khi chưa có tường: bit i bật nếu từ ô i đi hướng đó không ra ngoài bàn.
OPEN0 = (sum(1 << i for i in range(N) if ROW[i] > 0),          # lên
         sum(1 << i for i in range(N) if ROW[i] < SIZE - 1),   # xuống
         sum(1 << i for i in range(N) if COL[i] > 0),          # trái
         sum(1 << i for i in range(N) if COL[i] < SIZE - 1))   # phải


class Blocked(frozenset):
    """Tập khóa cạnh bị chặn (như frozenset: dùng làm khóa bảng chuyển vị, kiểm tra `in`)
    kèm 4 bitboard up/down/left/right: bit i bật nếu từ ô i đi hướng đó được."""
    __slots__ = ("up", "down", "left", "right")


def wall_masks(blocked, o, r, c):
    """4 bitboard 'đi được' sau khi thêm tường (o, r, c) vào blocked."""
    a = r * SIZE + c
    if o == "H":     # chặn đi xuống từ (r,c),(r,c+1) và đi lên từ hai ô ngay dưới
        bits = (1 << a) | (1 << (a + 1))
        return blocked.up & ~(bits << SIZE), blocked.down & ~bits, blocked.left, blocked.right
    bits = (1 << a) | (1 << (a + SIZE))   # V: chặn sang phải từ (r,c),(r+1,c), sang trái từ hai ô bên phải
    return blocked.up, blocked.down, blocked.left & ~(bits << 1), blocked.right & ~bits


def with_wall(blocked, o, r, c, masks=None):
    """Blocked mới = blocked + tường (o, r, c)."""
    nb = Blocked(chain(blocked, wall_keys(o, r, c)))
    nb.up, nb.down, nb.left, nb.right = masks or wall_masks(blocked, o, r, c)
    return nb


def blocked_from(hw, vw):
    nb = Blocked()
    nb.up, nb.down, nb.left, nb.right = OPEN0
    for r, c in hw:
        nb = with_wall(nb, "H", r, c)
    for r, c in vw:
        nb = with_wall(nb, "V", r, c)
    return nb


# ------------------------------------------------------- 3. Luật chơi
def bfs_dist(start, goal_row, up, down, left, right):
    """BFS bằng bitboard: số bước ít nhất từ start tới goal_row (bỏ qua quân), -1 nếu bị bít.
    frontier = tập ô vừa tới ở bước trước; mỗi vòng mở rộng cả tập theo 4 hướng."""
    reach = frontier = 1 << start
    goal = ROW_BITS[goal_row]
    if reach & goal:
        return 0
    d = 0
    while True:
        nxt = (((frontier & up) >> SIZE) | ((frontier & down) << SIZE)
               | ((frontier & left) >> 1) | ((frontier & right) << 1)) & ~reach
        if not nxt:
            return -1
        d += 1
        if nxt & goal:
            return d
        reach |= nxt
        frontier = nxt


def distance(start, goal_row, blocked):
    """Số bước ít nhất tới goal_row với tập tường blocked (kiểu Blocked)."""
    return bfs_dist(start, goal_row, blocked.up, blocked.down, blocked.left, blocked.right)


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
            for b, k in NEIK[a]:
                if b not in parent and k not in blocked:
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


def vulnerability(start, goal_row, blocked, d0):
    """Số bước tối đa một bức tường (trên vài bước đầu của đường ngắn nhất)
    làm người đứng ở start phải đi thêm. Không tính tường bít đường."""
    path = shortest_path(start, goal_row, blocked)[:VULN_STEPS + 1]
    worst = 0
    for o, r, c in walls_on_path(path):
        if not (0 <= r < SIZE - 1 and 0 <= c < SIZE - 1):
            continue
        a = r * SIZE + c                  # trùng cạnh đã chặn (gần đúng cho wall_fits):
        if o == "H":                      # bit "đi xuống" của 2 ô phía trên tường đã tắt
            if (blocked.down >> a) & 3 != 3:
                continue
        elif (blocked.right >> a) & 1 == 0 or (blocked.right >> (a + SIZE)) & 1 == 0:
            continue
        d = bfs_dist(start, goal_row, *wall_masks(blocked, o, r, c))
        if d - d0 > worst:
            worst = d - d0
    return worst


def defensive_walls(me, pos, hw, vw, blocked, wl, dist, goals):
    """Tường phòng thủ (chỉ xét ở gốc): tìm các bức tường đối thủ có thể đặt làm
    mình đi thêm >= 2 bước, rồi đề xuất tường của mình chiếm chỗ/cắt ngang vị trí đó
    (tường chồng lấn hoặc giao nhau thì không đặt được) mà không làm mình chậm đi."""
    q = 1 - me
    if wl[me] == 0 or wl[q] == 0:
        return []
    threats = []
    for o, r, c in walls_on_path(shortest_path(pos[me], goals[me], blocked)):
        if not wall_fits(o, r, c, hw, vw):
            continue
        m2 = wall_masks(blocked, o, r, c)
        d = bfs_dist(pos[me], goals[me], *m2)
        if d - dist[me] >= 2 and bfs_dist(pos[q], goals[q], *m2) >= 0:
            threats.append((d - dist[me], o, r, c))
    threats.sort(reverse=True)
    out, seen = [], set()
    for gain, o, r, c in threats[:DEFENSE_THREATS]:
        if o == "H":   # chặn được bằng tường dọc cắt ngang hoặc tường ngang chồng lấn
            blockers = (("V", r, c), ("H", r, c - 1), ("H", r, c + 1))
        else:
            blockers = (("H", r, c), ("V", r - 1, c), ("V", r + 1, c))
        for b in blockers:
            if b in seen or not wall_fits(b[0], b[1], b[2], hw, vw):
                continue
            seen.add(b)
            nb = with_wall(blocked, *b)
            dp2, dq2 = distance(pos[me], goals[me], nb), distance(pos[q], goals[q], nb)
            if dp2 < 0 or dq2 < 0 or dp2 > dist[me]:
                continue                   # không bít đường, không tự làm chậm mình
            nh = hw | {(b[1], b[2])} if b[0] == "H" else hw
            nv = vw | {(b[1], b[2])} if b[0] == "V" else vw
            nd = list(dist)
            nd[me], nd[q] = dp2, dq2
            nwl = list(wl)
            nwl[me] -= 1
            out.append((gain + dq2 - dp2, ("wall", b), pos, nh, nv, nb, nwl, nd))
    return out


# ------------------------------------------------------- 4. Đánh giá
def evaluate(p, dist, wl, vuln=(0, 0)):
    """Điểm trạng thái nhìn từ người chơi p (người sắp đi).
    vuln[i] = độ dễ bị chặn của người chơi i (0 nếu đối phương hết tường)."""
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
            + w["opp_dist"] * dq
            + w.get("vuln", 0.0) * (vuln[q] - vuln[p]))


# ------------------------------------------------------- 5. Tìm kiếm
EXACT, LOWER, UPPER = 0, 1, 2


class Search:
    def __init__(self, goals, deadline):
        self.goals = goals
        self.deadline = deadline
        self.tt = {}
        self.nodes = 0
        self.vcache = {}

    def leaf(self, p, pos, blocked, wl, dist):
        """Đánh giá lá có tính độ dễ bị chặn (lưu cache theo ô + tường)."""
        if not WEIGHTS.get("vuln"):
            return evaluate(p, dist, wl)
        vul = [0, 0]
        for i in (0, 1):
            if wl[1 - i] > 0 and dist[i] > 0:
                key = (pos[i], self.goals[i], blocked)
                v = self.vcache.get(key)
                if v is None:
                    v = vulnerability(pos[i], self.goals[i], blocked, dist[i])
                    self.vcache[key] = v
                vul[i] = v
        return evaluate(p, dist, wl, vul)

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
                m2 = wall_masks(blocked, o, r, c)
                dq2 = bfs_dist(pos[q], goal_q, *m2)
                if dq2 <= dist[q]:          # không làm chậm đối thủ (hoặc bít đường: -1)
                    continue
                dp2 = bfs_dist(pos[p], goal, *m2)
                if dp2 < 0:
                    continue
                nb = with_wall(blocked, o, r, c, m2)
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
            return self.leaf(p, pos, blocked, wl, dist)

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
            return self.leaf(p, pos, blocked, wl, dist)
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

        have = {k[1] for k in kids}
        kids += [k for k in defensive_walls(me, pos, hw, vw, blocked, wl, dist, goals)
                 if k[1] not in have]

        # Chỉ giữ ứng viên mà Arena xác nhận hợp lệ (an toàn tuyệt đối).
        legal_m = {cell(*m) for m in observation["legal_moves"]}
        legal_w = {(w["orientation"], w["row"], w["col"]) for w in observation["legal_walls"]}
        kids = [k for k in kids if (k[1][1] in legal_m if k[1][0] == "move" else k[1][1] in legal_w)]
        if kids:
            best, depth_done = kids[0], 0
            for depth in range(1, MAX_DEPTH + 1):
                try:
                    alpha, cur_best = -float("inf"), None
                    for kid in kids:
                        _, _, npos, nh, nv, nb, nwl, nd = kid
                        v = -search.negamax(1 - me, npos, nh, nv, nb, nwl, nd, depth - 1,
                                            -float("inf"), -alpha)
                        if v > alpha:
                            alpha, cur_best = v, kid
                except Timeout:
                    # Hết giờ giữa chừng: kids[0] là nước tốt nhất của độ sâu trước và được xét
                    # trước tiên. Nếu đã có nước khác vượt qua nó ở độ sâu mới này thì nước đó
                    # tốt hơn thật (đã được chứng minh) -> dùng luôn, không bỏ phí (như Stockfish).
                    if cur_best is not None and cur_best is not kids[0]:
                        best = cur_best
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
