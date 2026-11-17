"""Agent Wrongway: tìm kiếm alpha-beta + hàm đánh giá có trọng số học được.

Bản v7 (hàm đánh giá tuyến tính 7 trọng số, tiến hóa). Mạnh ngang MSSV.py (bản TD, #37).
Muốn nộp bản này: đổi tên file thành <MSSV>.py.

Cách hoạt động:
1. Tự mô phỏng luật chơi (đi, nhảy, đi chéo, tường hợp lệ) để nhìn trước.
   Ô cờ được đánh số 0..80 (= hàng*9 + cột). Tường và khoảng cách đều dùng
   BITBOARD (như Stockfish): cả bàn là một số nguyên 81 bit; tập tường là 4
   bitboard "ô nào đi được lên/xuống/trái/phải"; BFS loang ra mọi ô cùng lúc
   bằng phép dịch bit. BFS là phần tốn thời gian nhất nên có 3 mẹo để làm ít đi:
   - loang ngược từ đích MỘT lần cho mỗi tập tường (goal_layers, có cache) là
     biết khoảng cách và đường ngắn nhất của mọi ô;
   - thử một bức tường thì chỉ loang lại vài bước quanh bức tường đó
     (dist_after_wall) thay vì BFS lại từ đầu;
   - ở nút lá, "độ dễ bị chặn" chỉ được tính khi còn ảnh hưởng tới việc cắt tỉa.
   -> cùng độ sâu nhanh gấp ~2.6 lần BFS thường (chọn cùng tập nước), nên nhìn
   sâu hơn trong cùng thời gian.
2. Tìm kiếm negamax alpha-beta, đào sâu dần (1, 2, 3... nước) tới khi hết giờ,
   có bảng chuyển vị (transposition table) để không tính lại thế cờ trùng.
   Hết giờ giữa một độ sâu: vẫn dùng nước đã được chứng minh tốt hơn nước cũ.
   Sắp thứ tự nước để alpha-beta cắt tỉa nhiều (kỹ thuật kiểu Stockfish):
   nước trong bảng TT được thử TRƯỚC khi sinh các nước còn lại (thường cắt tỉa
   ngay, khỏi tốn BFS cho mọi tường); rồi 2 "killer move" của tầng; hòa điểm thì
   xét "history"; các nước sau nước đầu dò bằng cửa sổ hẹp (PVS).
   -> nhanh hơn ~1.3 lần ở độ sâu 5-6, chọn nước y hệt.
   Ở gốc, hai nước bằng điểm thì chọn nước tiến gần đích hơn (tránh đi qua đi lại
   chờ đối thủ chặn trong khi đối thủ cứ thế đi về đích).
   Ứng viên mỗi nút: mọi nước đi + tối đa WALL_CANDIDATES tường chắn trên
   đường ngắn nhất của đối thủ (xếp theo lợi ích). Ở gốc thêm "tường phòng thủ":
   chiếm chỗ các bức tường nguy hiểm nhất mà đối thủ có thể đặt để nhốt mình.
3. Hàm đánh giá = tổng có trọng số các đặc trưng (khoảng cách, số tường...).
   Có đặc trưng "độ dễ bị chặn" (vuln): một bức tường của đối thủ làm mình đi
   thêm tối đa bao nhiêu bước. Đặc trưng "giữ tường khi dẫn xa" (reserve): đang
   dẫn càng nhiều bước thì mỗi bức tường còn giữ càng quý (để chống lại tường
   của đối thủ), nên bot không xài hết tường khi đã thắng chắc cuộc đua.
   Hàm đánh giá ĐỐI XỨNG: đổi vai hai người chơi thì điểm đổi dấu (chỉ cộng thêm
   hằng số "tempo" cho người sắp đi). Nhờ vậy một thế cờ được chấm như nhau dù
   tìm tới độ sâu chẵn hay lẻ; bản cũ lệch ~9 điểm giữa hai trường hợp nên bot
   đổi phong cách theo độ sâu đạt được ở mỗi lượt.
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
MAX_DEPTH = 10           # trần độ sâu của đào sâu dần (thế cờ ít tường tìm rất nhanh nên tới được 8-10)
WALL_CANDIDATES = 8      # số tường tốt nhất được xét ở mỗi nút
DEFENSE_THREATS = 3      # số tường nguy hiểm nhất của đối thủ cần tìm cách chặn trước (ở gốc)
VULN_STEPS = 8           # độ dễ bị chặn: chỉ thử tường trên VULN_STEPS bước đầu của đường đi
KILLERS = True           # nhớ 2 nước gây cắt tỉa ở mỗi tầng, thử sớm ở nút anh em
HISTORY = True           # điểm lịch sử: nước hay gây cắt tỉa được ưu tiên khi hòa thứ tự
PVS = True               # tìm cửa sổ hẹp cho các nước sau nước đầu
LAZY_TT = True           # thử nước trong bảng TT trước khi sinh mọi nước con
EPS = 1e-6
RUSH_WALLS = 0           # còn <= RUSH_WALLS tường: ở gốc chỉ xét nước tiến gần đích (+ tường)
PROGRESS_BONUS = 0.001   # ở gốc: khi hòa điểm, ưu tiên nước đưa mình gần đích hơn (phá thế hòa)
WIN = 10000.0
RESERVE_CAP = 6          # reserve: mức dẫn tối đa được tính (bước)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))   # lên, xuống, trái, phải (cùng thứ tự với 4 bitboard)

# Trọng số hàm đánh giá (nhìn từ người sắp đi):
#   race          : khoảng cách đối thủ - khoảng cách mình
#   walls         : số tường mình còn - số tường đối thủ còn
#   wall_urgency  : tường quý hơn khi đối thủ gần đích: wl_me/(d_opp+1) - wl_opp/(d_me+1)
#   safe_race     : +1 nếu đối thủ hết tường và mình về trước, -1 nếu ngược lại
#   tempo         : hằng số cộng cho người sắp đi (được đi trước). Mọi đặc trưng khác đều
#                   ĐỐI XỨNG (đổi vai hai người chơi thì đổi dấu), nên cùng một thế cờ được
#                   chấm như nhau dù nút lá ở độ sâu chẵn hay lẻ.
#   reserve       : giữ tường khi dẫn xa: wl_me*min(lead_me,CAP) - wl_opp*min(lead_opp,CAP)
#                   (lead = số bước dẫn trước, 0 nếu đang thua) -> dẫn càng xa, tường càng quý
#   vuln          : độ "dễ bị chặn": số bước bức tường tốt nhất của mình làm đối thủ
#                   chậm đi trừ số bước bức tường tốt nhất của đối thủ làm mình chậm đi
# ======================= WEIGHTS BEGIN (tự sinh bởi training/evolve.py) =======================
WEIGHTS = {"race": 1.3805, "walls": 2.0643, "wall_urgency": -0.0789, "safe_race": 6.8451, "vuln": 1.0, "reserve": 0.1, "tempo": 1.0}
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
# Hai hướng vuông góc với mỗi hướng (dùng cho đi chéo quanh đối thủ).
SIDE = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}


# EDGE_WALLS[a, b] = các tường (o, r, c) nằm trong bàn chặn bước đi giữa hai ô kề a -> b.
EDGE_WALLS = {}
for _i in range(N):
    for _k, _b in enumerate(STEP[_i]):
        if _b < 0:
            continue
        _r, _c = min(ROW[_i], ROW[_b]), min(COL[_i], COL[_b])
        if _k < 2:     # bước dọc -> tường ngang ở mép dưới hàng _r, bắt đầu từ cột _c hoặc _c - 1
            _pair = (("H", _r, _c), ("H", _r, _c - 1))
        else:          # bước ngang -> tường dọc ở mép phải cột _c, bắt đầu từ hàng _r hoặc _r - 1
            _pair = (("V", _r, _c), ("V", _r - 1, _c))
        EDGE_WALLS[_i, _b] = tuple(w for w in _pair if 0 <= w[1] < SIZE - 1 and 0 <= w[2] < SIZE - 1)


def cell(r, c):
    return r * SIZE + c


# Bitboard (giống Stockfish): cả bàn cờ là một số nguyên 81 bit, bit thứ i = ô i.
# Một bước BFS cho MỌI ô cùng lúc = vài phép dịch bit, thay vì duyệt từng ô.
#   dịch >> 9 = lên một hàng, << 9 = xuống một hàng, >> 1 = sang trái, << 1 = sang phải.
ROW_BITS = [sum(1 << (r * SIZE + c) for c in range(SIZE)) for r in range(SIZE)]
# 4 bitboard "đi được" khi chưa có tường: bit i bật nếu từ ô i đi hướng đó không ra ngoài bàn.
OPEN0 = (sum(1 << i for i in range(N) if ROW[i] > 0),          # lên
         sum(1 << i for i in range(N) if ROW[i] < SIZE - 1),   # xuống
         sum(1 << i for i in range(N) if COL[i] > 0),          # trái
         sum(1 << i for i in range(N) if COL[i] < SIZE - 1))   # phải


# Tập tường của một thế cờ ("blocked") = bộ 4 bitboard (lên, xuống, trái, phải): bit i của
# blocked[k] bật nếu từ ô i đi được theo hướng DIRS[k] (không ra ngoài bàn, không bị tường chặn).
# Là tuple số nguyên nên thêm một bức tường chỉ tốn vài phép bit, và dùng được ngay làm khóa
# bảng chuyển vị.
def with_wall(blocked, o, r, c):
    """blocked sau khi thêm tường (o, r, c)."""
    up, down, left, right = blocked
    a = r * SIZE + c
    if o == "H":     # chặn đi xuống từ (r,c),(r,c+1) và đi lên từ hai ô ngay dưới
        bits = 3 << a
        return up & ~(bits << SIZE), down & ~bits, left, right
    bits = (1 << a) | (1 << (a + SIZE))   # V: chặn sang phải từ (r,c),(r+1,c), sang trái từ hai ô bên phải
    return up, down, left & ~(bits << 1), right & ~bits


def blocked_from(hw, vw):
    blocked = OPEN0
    for r, c in hw:
        blocked = with_wall(blocked, "H", r, c)
    for r, c in vw:
        blocked = with_wall(blocked, "V", r, c)
    return blocked


# ------------------------------------------------------- 3. Luật chơi
def flood(reach, d, target, up, down, left, right):
    """BFS bằng bitboard. reach = tập ô đã tới được sau d bước; mỗi vòng loang cả tập thêm một
    bước theo 4 hướng. Trả số bước ít nhất để chạm một ô của target, -1 nếu bị bít."""
    while not reach & target:
        nxt = (reach | ((reach & up) >> 9) | ((reach & down) << 9)
               | ((reach & left) >> 1) | ((reach & right) << 1))
        if nxt == reach:
            return -1
        reach = nxt
        d += 1
    return d


def bfs_dist(start, goal_row, up, down, left, right):
    """Số bước ít nhất từ ô start tới goal_row (bỏ qua quân), -1 nếu bị bít."""
    return flood(1 << start, 0, ROW_BITS[goal_row], up, down, left, right)


def reach_steps(start, blocked, steps):
    """reach[k] = bitboard các ô tới được từ start trong <= k bước (k = 0..steps)."""
    up, down, left, right = blocked
    cur = 1 << start
    reach = [cur]
    for _ in range(steps):
        cur |= ((cur & up) >> 9) | ((cur & down) << 9) | ((cur & left) >> 1) | ((cur & right) << 1)
        reach.append(cur)
    return reach


def dist_after_wall(reach, i, a, layers, walled):
    """Khoảng cách mới tới đích sau khi thêm một bức tường chặn bước thứ i của đường đi, mà
    KHÔNG phải BFS lại từ đầu. Tường nào cũng nằm giữa 4 ô a, a+1, a+9, a+10 (a = ô góc trên
    trái) nên chỉ làm đổi khoảng cách ở "phía sau" 4 ô đó:
    - Phía xuất phát: reach = reach_steps khi chưa có tường. Nếu cả 4 ô đều cách điểm xuất
      phát >= i bước thì i bước loang đầu không đổi -> loang tiếp từ reach[i].
    - Phía đích: layers = goal_layers khi chưa có tường. Gọi k = khoảng cách tới đích nhỏ nhất
      trong 4 ô; các ô ở lớp k trở xuống giữ nguyên khoảng cách -> chỉ cần loang tới khi chạm
      lớp k rồi cộng k.
    Nhờ vậy mỗi lần thử tường chỉ loang vài bước quanh bức tường. walled = tập tường mới."""
    square = 1539 << a                         # 1539 = 0b11_0000000_11: bit của 4 ô quanh tường
    if i and reach[i - 1] & square:
        i = 0
    k = 0
    while not layers[k] & square:
        k += 1
    d = flood(reach[i], i, layers[k], *walled)
    return d + k if d >= 0 else -1


def distance(start, goal_row, blocked):
    """Số bước ít nhất tới goal_row với tập tường blocked."""
    return bfs_dist(start, goal_row, *blocked)


def goal_layers(goal_row, blocked):
    """Loang NGƯỢC từ hàng đích: layers[d] = bitboard các ô cách goal_row đúng d bước.
    Tính một lần cho mỗi (đích, tập tường) là biết khoảng cách của MỌI ô."""
    up, down, left, right = blocked
    reach = frontier = ROW_BITS[goal_row]
    layers = [frontier]
    while True:
        # Ô x thuộc lớp kế tiếp nếu từ x đi một bước tới được một ô của frontier.
        nxt = (((frontier << SIZE) & up) | ((frontier >> SIZE) & down)
               | ((frontier << 1) & left) | ((frontier >> 1) & right)) & ~reach
        if not nxt:
            return layers
        layers.append(nxt)
        reach |= nxt
        frontier = nxt


def layer_index(layers, c):
    """Khoảng cách từ ô c tới đích theo goal_layers (-1 nếu không tới được)."""
    bit = 1 << c
    for d, layer in enumerate(layers):
        if layer & bit:
            return d
    return -1


def shortest_path(start, d, layers, blocked, max_steps=N):
    """Một đường ngắn nhất (list ô, tối đa max_steps bước) từ start (đang cách đích d bước):
    mỗi bước sang một ô kề thuộc lớp gần đích hơn một bậc (layers = goal_layers)."""
    path = [start]
    a = start
    for i in range(d - 1, max(d - 1 - max_steps, -1), -1):
        nxt = layers[i]
        for k in range(4):
            if (blocked[k] >> a) & 1 and (nxt >> STEP[a][k]) & 1:
                a = STEP[a][k]
                break
        path.append(a)
    return path


def legal_moves(me, opp, blocked):
    """Giống Arena: đi 1 ô; nhảy qua đối thủ; đi chéo khi phía sau bị chặn."""
    out = set()
    for k in range(4):
        if not (blocked[k] >> me) & 1:        # ra ngoài bàn hoặc bị tường chặn
            continue
        adj = STEP[me][k]
        if adj != opp:
            out.add(adj)
        elif (blocked[k] >> adj) & 1:         # nhảy thẳng qua đầu đối thủ
            out.add(STEP[adj][k])
        else:
            for s in SIDE[k]:
                if (blocked[s] >> adj) & 1:
                    out.add(STEP[adj][s])
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
    """Các tường chặn ít nhất một bước của đường đi: dict {(o, r, c): i} với i = bước đầu tiên
    của đường bị tường đó chặn (dict giữ thứ tự thêm vào: tường gần đầu đường trước)."""
    out = {}
    for i in range(len(path) - 1):
        for w in EDGE_WALLS[path[i], path[i + 1]]:
            if w not in out:
                out[w] = i
    return out


def vulnerability(start, goal_row, blocked, d0, layers):
    """Số bước tối đa một bức tường (trên vài bước đầu của đường ngắn nhất)
    làm người đứng ở start phải đi thêm. Không tính tường bít đường.
    layers = goal_layers(goal_row, blocked); d0 = khoảng cách hiện tại."""
    path = shortest_path(start, d0, layers, blocked, VULN_STEPS)
    reach = reach_steps(start, blocked, len(path) - 1)
    down, right = blocked[1], blocked[3]
    worst = 0
    for (o, r, c), i in walls_on_path(path).items():
        a = r * SIZE + c                  # trùng cạnh đã chặn (gần đúng cho wall_fits):
        if o == "H":                      # bit "đi xuống" của 2 ô phía trên tường đã tắt
            if (down >> a) & 3 != 3:
                continue
        elif (right >> a) & 1 == 0 or (right >> (a + SIZE)) & 1 == 0:
            continue
        d = dist_after_wall(reach, i, a, layers, with_wall(blocked, o, r, c))
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
    path = shortest_path(pos[me], dist[me], goal_layers(goals[me], blocked), blocked)
    for o, r, c in walls_on_path(path):
        if not wall_fits(o, r, c, hw, vw):
            continue
        m2 = with_wall(blocked, o, r, c)
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
    return (w.get("tempo", 0.0)
            + w["race"] * (dq - dp)
            + w["walls"] * (wl[p] - wl[q])
            + w["wall_urgency"] * (wl[p] / (dq + 1.0) - wl[q] / (dp + 1.0))
            + w["safe_race"] * safe
            + w.get("vuln", 0.0) * (vuln[q] - vuln[p])
            + w.get("reserve", 0.0) * (wl[p] * min(max(dq - dp, 0), RESERVE_CAP)
                                       - wl[q] * min(max(dp - dq, 0), RESERVE_CAP)))


# ------------------------------------------------------- 5. Tìm kiếm
EXACT, LOWER, UPPER = 0, 1, 2


class Search:
    def __init__(self, goals, deadline):
        self.goals = goals
        self.deadline = deadline
        self.tt = {}
        self.nodes = 0
        self.vcache = {}
        self.lcache = {}    # (đích, tập tường) -> goal_layers
        self.killers = {}   # tầng -> 2 nước gây cắt tỉa gần nhất
        self.history = {}   # (người chơi, nước) -> tổng depth^2 các lần gây cắt tỉa

    def layers_to(self, goal_row, blocked):
        """goal_layers có cache: rất nhiều nút trong cây chung một tập tường."""
        key = (goal_row, blocked)
        layers = self.lcache.get(key)
        if layers is None:
            layers = self.lcache[key] = goal_layers(goal_row, blocked)
        return layers

    def vuln(self, i, pos, blocked, wl, dist):
        """Độ dễ bị chặn của người chơi i (lưu cache theo ô + tường)."""
        if wl[1 - i] == 0 or dist[i] == 0:
            return 0
        key = (pos[i], i, blocked)
        v = self.vcache.get(key)
        if v is None:
            goal = self.goals[i]
            v = self.vcache[key] = vulnerability(pos[i], goal, blocked, dist[i],
                                                 self.layers_to(goal, blocked))
        return v

    def leaf(self, p, pos, blocked, wl, dist, alpha, beta):
        """Đánh giá lá: điểm = base + w * (vuln[q] - vuln[p]), hai vuln đều >= 0.
        vuln là phần đắt nhất (mỗi cái ~10 lần BFS) nên tính LƯỜI: nếu base đã <= alpha thì
        chỉ vuln[q] mới cứu được -> tính nó trước, vẫn <= alpha thì trả luôn (đó là chặn trên
        của điểm thật, đủ để alpha-beta cắt tỉa) khỏi tính vuln[p]. Tương tự khi base >= beta."""
        base = evaluate(p, dist, wl)
        w = WEIGHTS.get("vuln", 0.0)
        if not w:
            return base
        q = 1 - p
        if w > 0:
            if base <= alpha:
                high = base + w * self.vuln(q, pos, blocked, wl, dist)
                if high <= alpha:
                    return high
                return high - w * self.vuln(p, pos, blocked, wl, dist)
            if base >= beta:
                low = base - w * self.vuln(p, pos, blocked, wl, dist)
                if low >= beta:
                    return low
                return low + w * self.vuln(q, pos, blocked, wl, dist)
        return base + w * (self.vuln(q, pos, blocked, wl, dist) - self.vuln(p, pos, blocked, wl, dist))

    def children(self, p, pos, hw, vw, blocked, wl, dist):
        """List (thứ tự, action, pos, hw, vw, blocked, wl, dist) hoặc ('WIN', ô)."""
        q = 1 - p
        goal = self.goals[p]
        layers = self.layers_to(goal, blocked)
        out = []
        for c in legal_moves(pos[p], pos[q], blocked):
            if ROW[c] == goal:
                return "WIN", c
            nd = list(dist)
            nd[p] = layer_index(layers, c)
            npos = list(pos)
            npos[p] = c
            out.append((dist[q] - nd[p], ("move", c), npos, hw, vw, blocked, wl, nd))

        if wl[p] > 0:
            walls = []
            goal_q = self.goals[q]
            layers_q = self.layers_to(goal_q, blocked)
            path = shortest_path(pos[q], dist[q], layers_q, blocked)
            reach = reach_steps(pos[q], blocked, len(path) - 1)
            # Tường chặn một đường ngắn nhất của chính mình: chỉ những tường này mới có thể làm
            # mình chậm đi / bít đường mình -> các tường khác khỏi cần BFS cho mình.
            own = walls_on_path(shortest_path(pos[p], dist[p], layers, blocked))
            for (o, r, c), i in walls_on_path(path).items():
                if not wall_fits(o, r, c, hw, vw):
                    continue
                nb = with_wall(blocked, o, r, c)
                dq2 = dist_after_wall(reach, i, r * SIZE + c, layers_q, nb)
                if dq2 <= dist[q]:          # không làm chậm đối thủ (hoặc bít đường: -1)
                    continue
                dp2 = dist[p]
                if (o, r, c) in own:
                    dp2 = bfs_dist(pos[p], goal, *nb)
                    if dp2 < 0:
                        continue
                walls.append((dq2 - dp2, o, r, c, nb, dp2, dq2))
            walls.sort(key=lambda x: -x[0])
            nwl = list(wl)
            nwl[p] -= 1
            for score, o, r, c, nb, dp2, dq2 in walls[:WALL_CANDIDATES]:
                nh = hw | {(r, c)} if o == "H" else hw
                nv = vw | {(r, c)} if o == "V" else vw
                nd = list(dist)
                nd[p], nd[q] = dp2, dq2
                out.append((score, ("wall", (o, r, c)), pos, nh, nv, nb, nwl, nd))
        out.sort(key=lambda x: -x[0])
        return out, None

    def make_child(self, p, pos, hw, vw, blocked, wl, dist, act):
        """Dựng đúng một nút con cho nước act (nước TT) mà không sinh cả danh sách.
        Trả None nếu nước không còn hợp lệ / không thuộc tập ứng viên."""
        q = 1 - p
        kind, data = act
        if kind == "move":
            if data not in legal_moves(pos[p], pos[q], blocked):
                return None
            nd = list(dist)
            nd[p] = distance(data, self.goals[p], blocked)
            npos = list(pos)
            npos[p] = data
            return npos, hw, vw, blocked, wl, nd
        if wl[p] <= 0:
            return None
        o, r, c = data
        if not wall_fits(o, r, c, hw, vw):
            return None
        nb = with_wall(blocked, o, r, c)
        dq2 = bfs_dist(pos[q], self.goals[q], *nb)
        if dq2 <= dist[q]:
            return None
        dp2 = bfs_dist(pos[p], self.goals[p], *nb)
        if dp2 < 0:
            return None
        nh = hw | {(r, c)} if o == "H" else hw
        nv = vw | {(r, c)} if o == "V" else vw
        nd = list(dist)
        nd[p], nd[q] = dp2, dq2
        nwl = list(wl)
        nwl[p] -= 1
        return pos, nh, nv, nb, nwl, nd

    def negamax(self, p, pos, hw, vw, blocked, wl, dist, depth, alpha, beta, ply=1):
        self.nodes += 1
        if (self.nodes & 63) == 0 and time.perf_counter() > self.deadline:
            raise Timeout()
        if depth == 0:
            return self.leaf(p, pos, blocked, wl, dist, alpha, beta)

        key = (p, pos[0], pos[1], wl[0], wl[1], blocked)
        entry = self.tt.get(key)
        tt_act = None
        if entry is not None:
            e_depth, e_val, e_flag, tt_act = entry
            if e_depth >= depth:
                if e_flag == EXACT:
                    return e_val
                if e_flag == LOWER and e_val >= beta:
                    return e_val
                if e_flag == UPPER and e_val <= alpha:
                    return e_val

        alpha0 = alpha
        best, best_act = -float("inf"), None
        searched = None
        # Giai đoạn 1: nước TT thường đủ để cắt tỉa -> khỏi tốn công sinh mọi nước con.
        if LAZY_TT and tt_act is not None:
            child = self.make_child(p, pos, hw, vw, blocked, wl, dist, tt_act)
            if child is not None:
                best = -self.negamax(1 - p, *child, depth - 1, -beta, -alpha, ply + 1)
                best_act = searched = tt_act
                if best > alpha:
                    alpha = best

        if alpha < beta:
            kids, _ = self.children(p, pos, hw, vw, blocked, wl, dist)
            if kids == "WIN":
                return WIN + depth               # thắng sớm hơn thì tốt hơn
            if not kids and searched is None:
                return self.leaf(p, pos, blocked, wl, dist, alpha, beta)
            # Thứ tự: nước TT, 2 nước killer của tầng này, rồi theo điểm tĩnh (+ lịch sử khi hòa).
            if HISTORY:
                h = self.history
                kids.sort(key=lambda k: (-k[0], -h.get((p, k[1]), 0)))
            front = []
            if tt_act is not None and searched is None:
                front.append(tt_act)
            if KILLERS:
                front += [a for a in self.killers.get(ply, ()) if a not in front]
            for a in reversed(front):
                for i, k in enumerate(kids):
                    if k[1] == a:
                        kids.insert(0, kids.pop(i))
                        break

            for _, act, npos, nh, nv, nb, nwl, nd in kids:
                if act == searched:
                    continue
                if PVS and best_act is not None:
                    # Cửa sổ hẹp: chỉ hỏi "có hơn alpha không?"; nếu có mới tìm lại đầy đủ.
                    v = -self.negamax(1 - p, npos, nh, nv, nb, nwl, nd, depth - 1,
                                      -alpha - EPS, -alpha, ply + 1)
                    if alpha < v < beta:
                        v = -self.negamax(1 - p, npos, nh, nv, nb, nwl, nd, depth - 1,
                                          -beta, -v, ply + 1)
                else:
                    v = -self.negamax(1 - p, npos, nh, nv, nb, nwl, nd, depth - 1,
                                      -beta, -alpha, ply + 1)
                if v > best:
                    best, best_act = v, act
                    if v > alpha:
                        alpha = v
                        if alpha >= beta:
                            break

        if best >= beta:                         # nước gây cắt tỉa: ghi nhớ cho các nút sau
            if KILLERS:
                ks = self.killers.setdefault(ply, [])
                if best_act not in ks:
                    ks.insert(0, best_act)
                    del ks[2:]
            if HISTORY:
                hk = (p, best_act)
                self.history[hk] = self.history.get(hk, 0) + depth * depth
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
        # Hết tường thì chỉ còn đua về đích: lùi/đi ngang không làm tường của đối thủ bớt
        # nguy hiểm mà mất nhịp. Tìm kiếm có hiệu ứng chẵn/lẻ (độ sâu chẵn thấy lùi tốt hơn,
        # lẻ thấy bằng nhau) -> không chặn thì bot tiến một bước lùi một bước hàng chục lượt.
        if wl[me] <= RUSH_WALLS:
            ahead = [k for k in kids if k[1][0] == "move" and k[7][me] < dist[me]]
            if ahead:
                kids = ahead + [k for k in kids if k[1][0] == "wall"]
        if kids:
            best, depth_done = kids[0], 0
            for depth in range(1, MAX_DEPTH + 1):
                try:
                    alpha, cur_best = -float("inf"), None
                    for kid in kids:
                        _, _, npos, nh, nv, nb, nwl, nd = kid
                        # Điểm ở gốc = giá trị tìm kiếm + thưởng tiến bộ. Khi đối thủ còn nhiều
                        # tường, mọi nước có thể hòa điểm (tiến lên kiểu gì cũng "sẽ bị chặn")
                        # -> không có thưởng thì bot đứng đi qua đi lại, mất nhịp.
                        bonus = PROGRESS_BONUS * (dist[me] - nd[me])
                        target = alpha - bonus           # giá trị tìm kiếm cần vượt
                        if PVS and cur_best is not None:
                            v = -search.negamax(1 - me, npos, nh, nv, nb, nwl, nd, depth - 1,
                                                -target - EPS, -target)
                            if v <= target:
                                continue
                        v = -search.negamax(1 - me, npos, nh, nv, nb, nwl, nd, depth - 1,
                                            -float("inf"), -target)
                        if v > target:
                            alpha, cur_best = v + bonus, kid
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
