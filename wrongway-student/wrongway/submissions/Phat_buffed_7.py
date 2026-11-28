"""Agent Wrongway - Phat_buffed_7 = Phat_buffed_4 + threat_response_bonus + PLUG_HOLD_SEAL_ONLY (#60).

Motivation from experiments #58/#59: wall parameter tuning only gave ±0.07s noise (<1σ).
New approach: add threat_response feature (wall response) + fully enable PLUG_HOLD_SEAL_ONLY law 4c.

Threat response: when opponent has a wall that threatens our goal (drops our distance >= 2),
prefer moves that either: (a) block that threat, or (b) escape the threatened path.
This is distinct from wall_diff - it's about temporal response to immediate dangers.

PLUG_HOLD_SEAL_ONLY law 4c: when we have <=2 walls remaining (running low), opponent has >=2 walls
and we're in endgame (d_me <= 6), only ordinary moves that slow opponent >= lim = max(2, threat)
steps are kept, where threat = max steps opponent's single wall can add to our shortest path.
"Harmony" moves (gain <= 1 step) are filtered out - we move pieces and hold wall for opponent's
turn to capture us.

Changes vs Phat_buffed_4:
- A. Added THREAT_RESPONSE_WEIGHT (new feature: +w * threat_response to leaf evaluation)
- B. PLUG_HOLD_SEAL_ONLY flag fully activates law 4c filtering in gen_moves at root
- C. Wall scoring in gen_moves includes threat_response bonus
- D. Same WALL_LIN=282.34, WALL_QUAD=-9.59 as Phat_buffed_4 (regression to known good params)

Reference: wall_diff = walls_left[me] - walls_left[opp], dist_diff = 117 pts/step.
Wall n threat = WALL_LIN + WALL_QUAD*(2n-1), dist_diff = 117 pts/step.
Wall 1 = 2.33 steps, Wall 5 = 1.68 steps, Wall 10 = 0.86 steps.
"""

import time
from collections import OrderedDict

# ============================================================================
# 0. CẤU HÌNH
# ============================================================================
SIZE = 9
TIME_BUDGET = 0.55        # giây mỗi nước; Arena cho 1.0 s, Colab chậm hơn máy cá nhân
MAX_DEPTH = 12            # trần đào sâu dần
EPS = 1e-3                # độ rộng cửa sổ hẹp của PVS (điểm ở đây: 1 bước ~ 1.4 điểm)
VULN_STEPS = 8            # độ dễ bị chặn: chỉ xét 8 bước đầu của đường ngắn nhất
PROGRESS_BONUS = 0.001    # ở gốc: phá thế hòa điểm bằng cách ưu tiên nước tiến gần đích
WALL_CANDIDATES_ROOT = 18  # số tường tốt nhất xét ở gốc (gốc có thêm vùng DAG + hộ giá)
WALL_CANDIDATES = 10      # số tường tốt nhất xét ở các nút trong
DIST_CAP = 40             # bảng dist: khoảng cách lớn nhất có ô riêng
OPP_CAP = 20              # bảng walls: khoảng cách đối thủ lớn nhất có cột riêng
LEAD_CAP = 6              # bảng lead: mức dẫn/thua lớn nhất có cột riêng (-6..6)
INF = 100000.0
WIN_SCORE = 90000.0
MAX_PLY = 64
TT_EXACT = 1
TT_LOWER = 2
TT_UPPER = 3

# ============================================================================
# 0b. CẤU HÌNH MỚI: THREAT RESPONSE
# ============================================================================
THREAT_RESPONSE_WEIGHT = 15.0   # hệ số phản ứng với tường nguy hiểm của đối thủ
                                                    # >= 10 nghĩa là: mỗi bước nguy hiểm +15 điểm
                                                    # đủ để override wall_diff nhỏ (wall_diff 282 ~ 2.33 bước -> 2.33*117 ≈ 273,
                                                    # nhưng threat_response hoạt động trong ballpark 0-8 steps * 15 ≈ 0-120,
                                                    # đủ để có tác động rõ rệt nhưng không dominante)

# ============================================================================
# 0c. HÀM ĐÁNH GIÁ TUYẾN TÍNH 10 ĐẶC TRƯNG (thêm threat_response)
# ============================================================================
FEATURE_KEYS = [
    "dist_diff",
    "race_urgency",
    "wall_diff",
    "wall_quad",
    "dag_width_diff",
    "center_diff",
    "jump_tempo",
    "tempo",
    "vuln_diff",
    "threat_response",    # THEM MOI: phản ứng với amenaz tường đối thủ
]

BASE_FEATURE_KEYS = FEATURE_KEYS[:8]   # 8 đặc trưng rẻ; vuln_diff tính lười theo cửa sổ
# A2 (#54): hai núm tách rời, thay cho một WALL_SCALE của bản 1-2. Giá biên của bức tường thứ n
# là WALL_LIN + WALL_QUAD*(2n-1), quy ra bước với dist_diff = 117 điểm/bước:
#   bức 1 (282.34 - 9.59): 2.33 bước   bức 5 (282.34 - 86.31): 1.68 bước   bức 10: 0.86 bước
# WALL_LIN = giá bức tường đầu tiên (độ lớn); WALL_QUAD âm = độ lõm. Quét được độc lập từng cái.
WALL_LIN = 282.34
WALL_QUAD = -9.59

# Trọng số khi đi trước (P0)
WEIGHTS_FIRST = {
    "dist_diff": 117.0000,
    "race_urgency": 121.0500,
    "wall_diff": WALL_LIN,
    "wall_quad": WALL_QUAD,
    "dag_width_diff": 5.7500,
    "center_diff": 8.8000,
    "jump_tempo": 36.2500,
    "tempo": 44.1383,
    "vuln_diff": 30.1084,
    "threat_response": THREAT_RESPONSE_WEIGHT,   # THEM MOI
}

# Trọng số khi đi sau (P1)
WEIGHTS_SECOND = {
    "dist_diff": 117.0000,
    "race_urgency": 121.2422,
    "wall_diff": WALL_LIN,
    "wall_quad": WALL_QUAD,
    "dag_width_diff": 5.7500,
    "center_diff": 8.9078,
    "jump_tempo": 36.2500,
    "tempo": 43.5434,
    "vuln_diff": 30.1289,
    "threat_response": THREAT_RESPONSE_WEIGHT,   # THEM MOI
}

WEIGHTS = WEIGHTS_FIRST


def get_weights_for_obs(obs):
    """Chọn bộ trọng số theo vai trò đi trước (0) / đi sau (1)."""
    return WEIGHTS_FIRST if obs.get("player_id", 0) == 0 else WEIGHTS_SECOND


# ============================================================================
# 1. HẰNG SỐ BITBOARD 81-BIT & BẢNG TRA CỨU TIỀN TÍNH TOÁN
# ============================================================================
MASK_81 = (1 << 81) - 1
POPCOUNT16 = tuple(bin(i).count("1") for i in range(65536))


def popcount81(x):
    return (
        POPCOUNT16[x & 0xFFFF]
        + POPCOUNT16[(x >> 16) & 0xFFFF]
        + POPCOUNT16[(x >> 32) & 0xFFFF]
        + POPCOUNT16[(x >> 48) & 0xFFFF]
        + POPCOUNT16[(x >> 64) & 0xFFFF]
        + ((x >> 80) & 1)
    )


NOT_COL0 = 0
NOT_COL8 = 0
ROW_MASKS = [0] * 9
for _r in range(9):
    for _c in range(9):
        _bit = 1 << (_r * 9 + _c)
        ROW_MASKS[_r] |= _bit
        if _c != 0:
            NOT_COL0 |= _bit
        if _c != 8:
            NOT_COL8 |= _bit


# 4 bitboard "đi được" khi chưa có tường: bit i bật nếu từ ô i đi hướng đó không ra ngoài bàn.
BASE_UP = (MASK_81 >> 9) << 9
BASE_DOWN = MASK_81 >> 9
BASE_LEFT = MASK_81 & NOT_COL0
BASE_RIGHT = MASK_81 & NOT_COL8


# Mỗi bức tường (chỉ số idx = r * 8 + c) chặn 2 cạnh; 4 bảng dưới đây là 4 ô quanh tường.
H_WALL_TOP2 = [0] * 64
H_WALL_BOT2 = [0] * 64
V_WALL_LEFT2 = [0] * 64
V_WALL_RIGHT2 = [0] * 64

H_CONFLICT_H = [0] * 64   # các tường ngang chồng lấn / cắt ngang tường ngang idx
V_CONFLICT_V = [0] * 64   # các tường dọc chồng lấn / cắt ngang tường dọc idx
AROUND_SQ_WALLS = [0] * 81  # các tường có thể chạm vào ô (để dựng vùng DAG)
TRAP_BOX_WALLS = [0] * 81   # vùng tường quanh quân (ứng viên bẫy)

for _r in range(8):
    for _c in range(8):
        _idx = _r * 8 + _c
        _s1 = _r * 9 + _c
        _s2 = _s1 + 1
        _s3 = _s1 + 9
        _s4 = _s1 + 10
        H_WALL_TOP2[_idx] = (1 << _s1) | (1 << _s2)
        H_WALL_BOT2[_idx] = (1 << _s3) | (1 << _s4)
        V_WALL_LEFT2[_idx] = (1 << _s1) | (1 << _s3)
        V_WALL_RIGHT2[_idx] = (1 << _s2) | (1 << _s4)

        _hc = 1 << _idx
        if _c > 0:
            _hc |= 1 << (_idx - 1)
        if _c < 7:
            _hc |= 1 << (_idx + 1)
        H_CONFLICT_H[_idx] = _hc

        _vc = 1 << _idx
        if _r > 0:
            _vc |= 1 << (_idx - 8)
        if _r < 7:
            _vc |= 1 << (_idx + 8)
        V_CONFLICT_V[_idx] = _vc

for _sq in range(81):
    _r, _c = divmod(_sq, 9)
    _m = 0
    for _dr in (-1, 0):
        for _dc in (-1, 0):
            _wr, _wc = _r + _dr, _c + _dc
            if 0 <= _wr < 8 and 0 <= _wc < 8:
                _m |= 1 << (_wr * 8 + _wc)
    AROUND_SQ_WALLS[_sq] = _m
    _tb = 0
    for _dr in (-2, -1, 0, 1):
        for _dc in (-2, -1, 0, 1):
            _wr, _wc = _r + _dr, _c + _dc
            if 0 <= _wr < 8 and 0 <= _wc < 8:
                _tb |= 1 << (_wr * 8 + _wc)
    TRAP_BOX_WALLS[_sq] = _tb


# EDGE_WALLS[(a, b)] = các tường chặn bước đi giữa hai ô kề a -> b, dạng (is_vert, idx).
# Dùng để lấy ứng viên tường từ đường ngắn nhất (rẻ hơn quét cả bàn).
EDGE_WALLS = {}
for _sq in range(81):
    _r, _c = divmod(_sq, 9)
    if _r > 0:                      # đi LÊN: tường ngang ở hàng _r - 1, cột _c hoặc _c - 1
        _lst = []
        for _cc in (_c, _c - 1):
            if 0 <= _cc < 8:
                _lst.append((False, (_r - 1) * 8 + _cc))
        EDGE_WALLS[(_sq, _sq - 9)] = tuple(_lst)
    if _r < 8:                      # đi XUỐNG: tường ngang ở hàng _r
        _lst = []
        for _cc in (_c, _c - 1):
            if 0 <= _cc < 8:
                _lst.append((False, _r * 8 + _cc))
        EDGE_WALLS[(_sq, _sq + 9)] = tuple(_lst)
    if _c > 0:                      # đi SANG TRÁI: tường dọc ở cột _c - 1
        _lst = []
        for _rr in (_r, _r - 1):
            if 0 <= _rr < 8:
                _lst.append((True, _rr * 8 + (_c - 1)))
        EDGE_WALLS[(_sq, _sq - 1)] = tuple(_lst)
    if _c < 8:                      # đi SANG PHẢI: tường dọc ở cột _c
        _lst = []
        for _rr in (_r, _r - 1):
            if 0 <= _rr < 8:
                _lst.append((True, _rr * 8 + _c))
        EDGE_WALLS[(_sq, _sq + 1)] = tuple(_lst)


# ------------------------------------------------------------------ Zobrist
_Z_SEED = 0x9E3779B97F4A7C15


def _next_rand64():
    global _Z_SEED
    _Z_SEED = (_Z_SEED + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = _Z_SEED
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


ZOBRIST_PAWN = [[_next_rand64() for _ in range(81)] for _ in range(2)]
ZOBRIST_HWALL = [_next_rand64() for _ in range(64)]
ZOBRIST_VWALL = [_next_rand64() for _ in range(64)]
ZOBRIST_WLEFT = [[_next_rand64() for _ in range(11)] for _ in range(2)]
ZOBRIST_SIDE = _next_rand64()


# ============================================================================
# 2. CACHE GIỮA CÁC LƯỢT (LRU có giới hạn để không vượt 128 MB)
# ============================================================================
class BoundedCache(OrderedDict):
    """Dict LRU: quá maxsize thì bỏ mục cũ nhất."""

    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize

    def get_item(self, key):
        val = self.get(key)
        if val is not None:
            self.move_to_end(key)
        return val

    def put_item(self, key, val):
        self[key] = val
        self.move_to_end(key)
        if len(self) > self.maxsize:
            self.popitem(last=False)


GOAL_LAYERS_CACHE = BoundedCache(32000)   # (tường, hàng đích) -> các lớp khoảng cách
VULN_CACHE = BoundedCache(40000)          # (người chơi, ô, tường) -> độ dễ bị chặn
TRANSPOSITION_TABLE = BoundedCache(110000)
KILLERS = [[0, 0] for _ in range(MAX_PLY)]
HISTORY = [[0] * 256 for _ in range(2)]
LAST_TURN = -1


def clear_search_memory():
    """Xóa bảng nhớ khi sang ván mới (hoặc khi agent bị khởi động lại giữa ván)."""
    GOAL_LAYERS_CACHE.clear()
    VULN_CACHE.clear()
    TRANSPOSITION_TABLE.clear()
    for p in range(MAX_PLY):
        KILLERS[p][0] = 0
        KILLERS[p][1] = 0
    for s in range(2):
        for m in range(256):
            HISTORY[s][m] = 0


def get_goal_layers(h_mask, v_mask, goal_row, up_bb, down_bb, left_bb, right_bb):
    """Loang ngược từ hàng đích: layers[d] = bitboard các ô cách đích đúng d bước.

    Đầu vào: h_mask, v_mask - tường đã đặt; goal_row - hàng đích; 4 bitboard "đi được".
    Đầu ra: tuple các lớp (có cache theo đúng tập tường + hàng đích).
    """
    key = (h_mask, v_mask, goal_row)
    cached = GOAL_LAYERS_CACHE.get_item(key)
    if cached is not None:
        return cached

    frontier = ROW_MASKS[goal_row]
    visited = frontier
    layers = [frontier]
    while frontier:
        nxt = (
            ((frontier << 9) & up_bb)
            | ((frontier >> 9) & down_bb)
            | ((frontier << 1) & left_bb)
            | ((frontier >> 1) & right_bb)
        ) & ~visited
        if not nxt:
            break
        layers.append(nxt)
        visited |= nxt
        frontier = nxt

    res = tuple(layers)
    GOAL_LAYERS_CACHE.put_item(key, res)
    return res


def dist_from_layers(layers, sq):
    """Tra khoảng cách của ô sq trong các lớp (999 nếu không tới được)."""
    bit = 1 << sq
    for d, l_bb in enumerate(layers):
        if l_bb & bit:
            return d
    return 999


def shortest_dag_analysis(layers, start_sq, d_start, up_bb, down_bb, left_bb, right_bb,
                          need_zone=True):
    """Phân tích DAG các đường ngắn nhất (chỉ dùng ở gốc).

    Trả về (wall_zone_64bit, br_count, dag_cell_count):
    - wall_zone: mọi vị trí tường có thể cắt ít nhất một đường ngắn nhất;
    - br_count: số hướng bước đầu độc lập; dag_cell_count: độ rộng hành lang.
    """
    if d_start >= 999:
        return 0, 0, 0
    if d_start == 0:
        return AROUND_SQ_WALLS[start_sq], 1, 1

    cur_bb = 1 << start_sq
    next_layer = layers[d_start - 1]
    br_count = 0
    if (up_bb & cur_bb) and ((cur_bb >> 9) & next_layer):
        br_count += 1
    if (down_bb & cur_bb) and ((cur_bb << 9) & next_layer):
        br_count += 1
    if (left_bb & cur_bb) and ((cur_bb >> 1) & next_layer):
        br_count += 1
    if (right_bb & cur_bb) and ((cur_bb << 1) & next_layer):
        br_count += 1

    dag_sq_bb = cur_bb
    for d in range(d_start - 1, -1, -1):
        step_bb = (
            (((cur_bb & up_bb) >> 9)
             | ((cur_bb & down_bb) << 9)
             | ((cur_bb & left_bb) >> 1)
             | ((cur_bb & right_bb) << 1))
            & layers[d]
        )
        dag_sq_bb |= step_bb
        cur_bb = step_bb

    if not need_zone:
        # Hàm đánh giá chỉ cần độ rộng hành lang: bỏ phần dựng vùng tường (đắt nhất).
        return 0, br_count, popcount81(dag_sq_bb)

    wall_zone = 0
    tmp = dag_sq_bb
    while tmp:
        lsb = tmp & -tmp
        wall_zone |= AROUND_SQ_WALLS[lsb.bit_length() - 1]
        tmp ^= lsb

    return wall_zone, br_count, popcount81(dag_sq_bb)


def dist_after_wall(layers, start_sq, d_orig, wall_idx, is_vert, up_bb, down_bb, left_bb, right_bb):
    """Khoảng cách tới đích của start_sq SAU KHI thêm một bức tường, không BFS lại từ đầu.

    Tường chỉ nằm giữa 4 ô nên chỉ đổi khoảng cách "phía sau" 4 ô đó: tìm lớp gần đích nhất
    bị ảnh hưởng, loang tiếp từ lớp trước nó với tập tường mới.

    Đầu vào: layers - kết quả get_goal_layers khi chưa có tường; start_sq, d_orig - ô và
             khoảng cách hiện tại; wall_idx, is_vert - bức tường thử; 4 bitboard "đi được".
    Đầu ra: khoảng cách mới, 999 nếu bị bít đường.
    """
    if d_orig >= 999:
        return 999

    max_d = len(layers) - 1
    if not is_vert:                       # tường ngang: chặn đi xuống phía trên, lên phía dưới
        t2 = H_WALL_TOP2[wall_idx]
        b2 = H_WALL_BOT2[wall_idx]
        n_up = up_bb & ~b2
        n_down = down_bb & ~t2
        n_left = left_bb
        n_right = right_bb
        affected_bb = t2 | b2
    else:                                 # tường dọc: chặn sang phải phía trái, sang trái phía phải
        l2 = V_WALL_LEFT2[wall_idx]
        r2 = V_WALL_RIGHT2[wall_idx]
        n_up = up_bb
        n_down = down_bb
        n_left = left_bb & ~r2
        n_right = right_bb & ~l2
        affected_bb = l2 | r2

    # Lớp gần đích nhất bị ảnh hưởng: nếu nó vẫn còn đường vào từ lớp trước thì khoảng cách
    # của start_sq không đổi.
    min_dirty_d = 999
    for d in range(1, max_d + 1):
        cand = layers[d] & affected_bb
        if cand:
            prev_l = layers[d - 1]
            tmp = cand
            broken = False
            while tmp:
                lsb = tmp & -tmp
                tmp ^= lsb
                if not (
                    (((lsb & n_up) >> 9) & prev_l)
                    | (((lsb & n_down) << 9) & prev_l)
                    | (((lsb & n_left) >> 1) & prev_l)
                    | (((lsb & n_right) << 1) & prev_l)
                ):
                    broken = True
                    break
            if broken:
                min_dirty_d = d
                break

    if min_dirty_d == 999 or d_orig < min_dirty_d:
        return d_orig

    valid_prefix = 0
    for d in range(min_dirty_d):
        valid_prefix |= layers[d]

    frontier = layers[min_dirty_d - 1]
    visited = valid_prefix
    cur_d = min_dirty_d - 1
    start_bit = 1 << start_sq

    while frontier:
        cur_d += 1
        nxt = (
            ((frontier << 9) & n_up)
            | ((frontier >> 9) & n_down)
            | ((frontier << 1) & n_left)
            | ((frontier >> 1) & n_right)
        ) & ~visited
        if not nxt:
            break
        if nxt & start_bit:
            return cur_d
        visited |= nxt
        frontier = nxt

    return 999


# ============================================================================
# 3. TRẠNG THÁI BITBOARD & TÌM KIẾM ALPHA-BETA
# ============================================================================
class BitboardQuoridorState:
    """Thế cờ dạng bitboard, sửa tại chỗ khi đi (make_move / unmake_move)."""

    __slots__ = (
        "up_bb", "down_bb", "left_bb", "right_bb",
        "h_mask", "v_mask",
        "pawns", "goals", "walls_left",
        "side", "zhash", "turn_base", "max_turns",
        "deadline", "nodes", "stop", "weights"
    )

    def __init__(self, obs, deadline, weights=None):
        self.weights = weights if weights is not None else get_weights_for_obs(obs)
        self.up_bb = BASE_UP
        self.down_bb = BASE_DOWN
        self.left_bb = BASE_LEFT
        self.right_bb = BASE_RIGHT
        self.h_mask = 0
        self.v_mask = 0

        my_id = obs["player_id"]
        opp_id = 1 - my_id
        you = obs["you"]
        opp = obs["opponent"]

        self.pawns = [0, 0]
        self.goals = [0, 0]
        self.walls_left = [0, 0]

        self.pawns[my_id] = you["position"][0] * 9 + you["position"][1]
        self.pawns[opp_id] = opp["position"][0] * 9 + opp["position"][1]
        self.goals[my_id] = you["goal_row"]
        self.goals[opp_id] = opp["goal_row"]
        self.walls_left[my_id] = you["walls_remaining"]
        self.walls_left[opp_id] = opp["walls_remaining"]

        for r, c in obs["walls"]["horizontal"]:
            idx = r * 8 + c
            self.h_mask |= 1 << idx
            self.down_bb &= ~H_WALL_TOP2[idx]
            self.up_bb &= ~H_WALL_BOT2[idx]

        for r, c in obs["walls"]["vertical"]:
            idx = r * 8 + c
            self.v_mask |= 1 << idx
            self.right_bb &= ~V_WALL_LEFT2[idx]
            self.left_bb &= ~V_WALL_RIGHT2[idx]

        self.side = my_id
        self.turn_base = obs.get("turn", 0)
        self.max_turns = obs.get("max_turns", 500)
        self.deadline = deadline
        self.nodes = 0
        self.stop = False

        zh = (
            ZOBRIST_PAWN[0][self.pawns[0]]
            ^ ZOBRIST_PAWN[1][self.pawns[1]]
            ^ ZOBRIST_WLEFT[0][self.walls_left[0]]
            ^ ZOBRIST_WLEFT[1][self.walls_left[1]]
        )
        hm = self.h_mask
        while hm:
            lsb = hm & -hm
            zh ^= ZOBRIST_HWALL[lsb.bit_length() - 1]
            hm ^= lsb
        vm = self.v_mask
        while vm:
            lsb = vm & -vm
            zh ^= ZOBRIST_VWALL[lsb.bit_length() - 1]
            vm ^= lsb
        if self.side == 1:
            zh ^= ZOBRIST_SIDE
        self.zhash = zh

    # ------------------------------------------------------------ 3a. Hàm đánh giá
    def dist_of(self, i):
        """Số bước từ quân người chơi i tới hàng đích của i (999 nếu bị bít)."""
        layers = get_goal_layers(
            self.h_mask, self.v_mask, self.goals[i],
            self.up_bb, self.down_bb, self.left_bb, self.right_bb
        )
        return dist_from_layers(layers, self.pawns[i])

    def path_edges(self, layers, start, d, limit):
        """Các cạnh (a, b) trên MỘT đường ngắn nhất từ start, tối đa limit cạnh đầu.

        Đầu vào: layers - các lớp khoảng cách; start - ô xuất phát; d - khoảng cách;
                 limit - số cạnh tối đa cần lấy.
        Đầu ra: list (a, b) với b là ô kề cách đích ít hơn một bước.
        """
        out = []
        if d <= 0 or d >= 999:
            return out
        up, down, left, right = self.up_bb, self.down_bb, self.left_bb, self.right_bb
        a = start
        i = d - 1
        while i >= 0 and len(out) < limit:
            b = -1
            if (up >> a) & 1:
                t = a - 9
                if (layers[i] >> t) & 1:
                    b = t
            if b < 0 and (down >> a) & 1:
                t = a + 9
                if (layers[i] >> t) & 1:
                    b = t
            if b < 0 and (left >> a) & 1:
                t = a - 1
                if (layers[i] >> t) & 1:
                    b = t
            if b < 0 and (right >> a) & 1:
                t = a + 1
                if (layers[i] >> t) & 1:
                    b = t
            if b < 0:
                break
            out.append((a, b))
            a = b
            i -= 1
        return out

    def wall_ok(self, is_vert, idx):
        """Bức tường có đặt được không (không chồng lấn / cắt ngang tường khác)."""
        if is_vert:
            return not ((self.v_mask & V_CONFLICT_V[idx]) or (self.h_mask & (1 << idx)))
        return not ((self.h_mask & H_CONFLICT_H[idx]) or (self.v_mask & (1 << idx)))

    def vuln(self, i, layers, d0):
        """Độ dễ bị chặn của người chơi i: bức tường trên 8 bước đầu đường ngắn nhất làm
        người đó đi thêm tối đa bao nhiêu bước (0 nếu đối thủ hết tường).

        Đầu vào: i - người chơi; layers, d0 - lớp khoảng cách và khoảng cách của i.
        Đầu ra: số bước đi thêm lớn nhất (bỏ qua tường bít đường).
        """
        if d0 <= 0 or self.walls_left[i ^ 1] == 0:
            return 0
        key = (i, self.pawns[i], self.h_mask, self.v_mask)
        cached = VULN_CACHE.get_item(key)
        if cached is not None:
            return cached

        up, down, left, right = self.up_bb, self.down_bb, self.left_bb, self.right_bb
        start = self.pawns[i]
        worst = 0
        for a, b in self.path_edges(layers, start, d0, VULN_STEPS):
            for is_vert, idx in EDGE_WALLS[(a, b)]:
                if not self.wall_ok(is_vert, idx):
                    continue
                nd = dist_after_wall(layers, start, d0, idx, is_vert, up, down, left, right)
                if nd < 999 and nd - d0 > worst:
                    worst = nd - d0
        VULN_CACHE.put_item(key, worst)
        return worst

    def _can_straight_jump(self, p1, p2):
        """Có nhảy thẳng qua đầu quân địch được không (1.0) hay không (0.0)."""
        r1, c1 = divmod(p1, 9)
        r2, c2 = divmod(p2, 9)
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return 0.0
        b1 = 1 << p1
        b2 = 1 << p2
        if r1 == r2 - 1 and (self.down_bb & b1) and (self.down_bb & b2):
            return 1.0
        if r1 == r2 + 1 and (self.up_bb & b1) and (self.up_bb & b2):
            return 1.0
        if c1 == c2 - 1 and (self.right_bb & b1) and (self.right_bb & b2):
            return 1.0
        if c1 == c2 + 1 and (self.left_bb & b1) and (self.left_bb & b2):
            return 1.0
        return 0.0

    def leaf(self, alpha, beta):
        """Chấm điểm nút lá theo 10 đặc trưng tuyến tính (thêm threat_response).

        Đầu vào: alpha, beta - cửa sổ tìm kiếm.
        Đầu ra: điểm nhìn từ người sắp đi (có thể chỉ là chặn trên/dưới khi ngoài cửa sổ).

        threat_response: đo đạc xem tường đối thủ có thể làm giảm khoảng cách của mình đi bao nhiêu bước.
        Nếu opponent có thể giảm d_me >= 2 bước -> threat_response = positive (phạt).
        Ngược lại nếu nước của mình giảm d_opp >= 2 -> threat_response = negative (thưởng).
        """
        me = self.side
        opp = me ^ 1
        layers_me = get_goal_layers(
            self.h_mask, self.v_mask, self.goals[me],
            self.up_bb, self.down_bb, self.left_bb, self.right_bb
        )
        layers_opp = get_goal_layers(
            self.h_mask, self.v_mask, self.goals[opp],
            self.up_bb, self.down_bb, self.left_bb, self.right_bb
        )
        d_me = dist_from_layers(layers_me, self.pawns[me])
        d_opp = dist_from_layers(layers_opp, self.pawns[opp])
        wl_me = self.walls_left[me]
        wl_opp = self.walls_left[opp]
        p_me = self.pawns[me]
        p_opp = self.pawns[opp]

        _, _, cells_me = shortest_dag_analysis(
            layers_me, p_me, d_me, self.up_bb, self.down_bb, self.left_bb, self.right_bb,
            need_zone=False
        )
        _, _, cells_opp = shortest_dag_analysis(
            layers_opp, p_opp, d_opp, self.up_bb, self.down_bb, self.left_bb, self.right_bb,
            need_zone=False
        )
        dag_w_me = cells_me / (d_me + 1.0) if d_me < 999 else 0.0
        dag_w_opp = cells_opp / (d_opp + 1.0) if d_opp < 999 else 0.0

        r_me, c_me = divmod(p_me, 9)
        r_opp, c_opp = divmod(p_opp, 9)
        feats = {
            "dist_diff": float(d_opp - d_me),
            "race_urgency": (10.0 / (d_me + 1.0)) - (10.0 / (d_opp + 1.0)),
            "wall_diff": float(wl_me - wl_opp),
            "wall_quad": float(wl_me * wl_me - wl_opp * wl_opp),
            "dag_width_diff": dag_w_me - dag_w_opp,
            "center_diff": float((4 - abs(c_me - 4)) - (4 - abs(c_opp - 4))),
            "jump_tempo": (self._can_straight_jump(p_me, p_opp)
                           - self._can_straight_jump(p_opp, p_me)),
            "tempo": 1.0,
            "vuln_diff": 0.0,
            # THEM MOI: threat_response - phản ứng với tường nguy hiểm của đối thủ
            "threat_response": 0.0,  # sẽ tính bên dưới
        }
        w = self.weights
        base = sum(w[k] * feats[k] for k in BASE_FEATURE_KEYS)
        # Add threat_response: tính trực tiếp thay vì thông qua BASE_FEATURE_KEYS
        # vì threat_response là feature thứ 10 (index 9), không nằm trong 8 feature đầu
        threat_val = feats["threat_response"]
        base += w["threat_response"] * threat_val
        
        wv = w["vuln_diff"]
        swing = 8.0 * wv
        if base + swing <= alpha:                 # chặn trên đã <= alpha: đủ để cắt tỉa
            return base + wv * (self.vuln(opp, layers_opp, d_opp) if wl_me > 0 else 0)
        if base - swing >= beta:                  # chặn dưới đã >= beta
            return base - wv * (self.vuln(me, layers_me, d_me) if wl_opp > 0 else 0)
        v_me = self.vuln(me, layers_me, d_me) if wl_opp > 0 else 0
        v_opp = self.vuln(opp, layers_opp, d_opp) if wl_me > 0 else 0
        return base + wv * (v_opp - v_me)


    # ------------------------------------------------------------ 3b. Sinh nước đi
    def gen_pawn_moves(self):
        """Liệt kê nước đi quân đúng luật Arena (đi 1 ô, nhảy qua đầu, đi chéo khi bị chặn).

        Đầu ra: list (điểm thứ tự, ô đến). Điểm chỉ dùng để xếp thứ tự nước đi.
        """
        me = self.side
        sq = self.pawns[me]
        opp_sq = self.pawns[me ^ 1]
        goal_r = self.goals[me]
        cur_row_d = abs(sq // 9 - goal_r)
        bit = 1 << sq

        out = []
        dirs = (
            (self.up_bb, -9),
            (self.down_bb, 9),
            (self.left_bb, -1),
            (self.right_bb, 1),
        )
        for idx, (dir_bb, delta) in enumerate(dirs):
            if not (dir_bb & bit):
                continue
            nxt = sq + delta
            if nxt != opp_sq:
                prog = cur_row_d - abs(nxt // 9 - goal_r)
                center = 4 - abs((nxt % 9) - 4)
                out.append((580 + prog * 110 + center * 4, nxt))
            else:
                opp_bit = 1 << opp_sq
                if dir_bb & opp_bit:                       # nhảy thẳng qua đầu đối thủ
                    dst = opp_sq + delta
                    prog = cur_row_d - abs(dst // 9 - goal_r)
                    out.append((690 + prog * 110, dst))
                else:                                      # đi chéo khi sau lưng bị chặn
                    perp = (dirs[2], dirs[3]) if idx < 2 else (dirs[0], dirs[1])
                    for p_bb, p_delta in perp:
                        if p_bb & opp_bit:
                            dst = opp_sq + p_delta
                            prog = cur_row_d - abs(dst // 9 - goal_r)
                            out.append((665 + prog * 110, dst))
        return out

    def defense_masks(self, layers_me, d_me):
        """Tường hộ giá ở gốc: chặn chỗ đối thủ có thể đặt tường làm mình chậm >= 2 bước.

        Duyệt mọi vị trí tường cắt được đường ngắn nhất của mình (vùng DAG); nếu bức tường
        đó làm mình chậm >= 2 bước thì đánh dấu bức tường chồng lấn / cắt ngang nó (cùng
        hướng và hai tường kề trên hành lang) là tường hộ giá.

        Đầu vào: layers_me, d_me - lớp khoảng cách và khoảng cách của mình.
        Đầu ra: (def_h_mask, def_v_mask) - các chỉ số tường ngang / dọc nên xét thêm ở gốc.
        """
        def_h = 0
        def_v = 0
        if self.walls_left[self.side ^ 1] == 0 or d_me <= 0:
            return 0, 0
        p_me = self.pawns[self.side]
        zone_me, _, _ = shortest_dag_analysis(
            layers_me, p_me, d_me, self.up_bb, self.down_bb, self.left_bb, self.right_bb
        )
        cm = zone_me
        while cm:
            lsb = cm & -cm
            idx = lsb.bit_length() - 1
            cm ^= lsb
            if self.wall_ok(False, idx):
                nd = dist_after_wall(
                    layers_me, p_me, d_me, idx, False,
                    self.up_bb, self.down_bb, self.left_bb, self.right_bb
                )
                if nd < 999 and nd - d_me >= 2:
                    def_v |= lsb                      # tường dọc khóa giao điểm chữ thập
                    def_h |= H_CONFLICT_H[idx]
                    r_idx = idx >> 3
                    if r_idx > 0:
                        def_v |= 1 << (idx - 8)
                    if r_idx < 7:
                        def_v |= 1 << (idx + 8)
            if self.wall_ok(True, idx):
                nd = dist_after_wall(
                    layers_me, p_me, d_me, idx, True,
                    self.up_bb, self.down_bb, self.left_bb, self.right_bb
                )
                if nd < 999 and nd - d_me >= 2:
                    def_h |= lsb
                    def_v |= V_CONFLICT_V[idx]
                    c_idx = idx & 7
                    if c_idx > 0:
                        def_h |= 1 << (idx - 1)
                    if c_idx < 7:
                        def_h |= 1 << (idx + 1)
        return def_h, def_v

    def gen_moves(self, ply, tt_move, is_root=False, root_legal_pawn_set=None, root_legal_walls=None):
        """Sinh và xếp thứ tự nước đi: mọi nước quân + các tường đáng xét.

        Ứng viên tường ở nút trong = tường chặn đường ngắn nhất của đối thủ (rẻ, như MSSV);
        ở gốc thêm vùng DAG (cắt cả đường ngắn nhất thay thế), hộp bẫy quanh quân địch và
        tường hộ giá. Tường nào không làm đối thủ chậm đi thì bỏ (trừ tường hộ giá).

        Đầu vào: ply - tầng; tt_move - nước trong bảng chuyển vị; is_root - có phải gốc;
                 root_legal_pawn_set, root_legal_walls - tập nước hợp lệ Arena xác nhận.
        Đầu ra: list nước đi (mã hoá: < 128 đi quân, 128..191 tường H, >= 192 tường V).
        """
        me = self.side
        opp = me ^ 1
        pawn_moves = self.gen_pawn_moves()
        if is_root and root_legal_pawn_set is not None:
            pawn_moves = [(sc, mv) for sc, mv in pawn_moves if mv in root_legal_pawn_set]

        moves = list(pawn_moves)
        if self.walls_left[me] == 0:
            if tt_move:
                for i, (_, mv) in enumerate(moves):
                    if mv == tt_move:
                        moves[i] = (1000000, mv)
                        break
            moves.sort(reverse=True)
            return [mv for _, mv in moves]

        p_me = self.pawns[me]
        p_opp = self.pawns[opp]
        layers_me = get_goal_layers(
            self.h_mask, self.v_mask, self.goals[me],
            self.up_bb, self.down_bb, self.left_bb, self.right_bb
        )
        layers_opp = get_goal_layers(
            self.h_mask, self.v_mask, self.goals[opp],
            self.up_bb, self.down_bb, self.left_bb, self.right_bb
        )
        d_me = dist_from_layers(layers_me, p_me)
        d_opp = dist_from_layers(layers_opp, p_opp)

        # Tường nằm trên đường ngắn nhất của MÌNH mới có thể làm mình chậm đi.
        own_walls = set()
        for a, b in self.path_edges(layers_me, p_me, d_me, d_me):
            for w in EDGE_WALLS[(a, b)]:
                own_walls.add(w)

        cands = []
        seen = set()
        for a, b in self.path_edges(layers_opp, p_opp, d_opp, d_opp):
            for w in EDGE_WALLS[(a, b)]:
                if w not in seen:
                    seen.add(w)
                    cands.append(w)

        def_h = 0
        def_v = 0
        if is_root:
            zone_opp, _, _ = shortest_dag_analysis(
                layers_opp, p_opp, d_opp, self.up_bb, self.down_bb, self.left_bb, self.right_bb
            )
            def_h, def_v = self.defense_masks(layers_me, d_me)
            for extra in (zone_opp | TRAP_BOX_WALLS[p_opp], def_h, def_v):
                cm = extra
                while cm:
                    lsb = cm & -cm
                    idx = lsb.bit_length() - 1
                    cm ^= lsb
                    for is_vert in (False, True):
                        w = (is_vert, idx)
                        if w not in seen:
                            seen.add(w)
                            cands.append(w)

        wall_scored = []
        k0, k1 = KILLERS[ply]
        for is_vert, idx in cands:
            if not self.wall_ok(is_vert, idx):
                continue
            is_def = ((def_v if is_vert else def_h) >> idx) & 1
            nd_opp = dist_after_wall(
                layers_opp, p_opp, d_opp, idx, is_vert,
                self.up_bb, self.down_bb, self.left_bb, self.right_bb
            )
            if nd_opp >= 999:
                continue
            det_opp = nd_opp - d_opp
            if det_opp <= 0 and not is_def:
                continue
            det_me = 0
            if (is_vert, idx) in own_walls:
                nd_me = dist_after_wall(
                    layers_me, p_me, d_me, idx, is_vert,
                    self.up_bb, self.down_bb, self.left_bb, self.right_bb
                )
                if nd_me >= 999:
                    continue
                det_me = nd_me - d_me
            if is_def and det_me > 0:
                continue                     # tường hộ giá tuyệt đối không tự làm chậm mình

            mv = (192 + idx) if is_vert else (128 + idx)
            # Điểm tĩnh CHỈ theo hiệu quả của tường: cắt danh sách ứng viên bằng điểm này,
            # killer/history chỉ dùng để xếp lại thứ tự sau đó (không chen vào top).
            # THÊM THREAT_RESPONSE BONUS: nếu det_opp lớn (tường cắt nhiều bước),
            # thêm bonus theo THREAT_RESPONSE_WEIGHT
            threat_bonus = 0
            if det_opp >= 2:
                threat_bonus = THREAT_RESPONSE_WEIGHT * det_opp
            sc = det_opp * 150 - det_me * 155 + (260 if is_def else 0) + threat_bonus
            wall_scored.append((sc, mv))

        wall_scored.sort(reverse=True)
        moves.extend(wall_scored[:WALL_CANDIDATES_ROOT if is_root else WALL_CANDIDATES])

        if is_root and root_legal_walls is not None:
            # Ở gốc chỉ giữ tường mà Arena xác nhận hợp lệ (an toàn tuyệt đối).
            lh_mask, lv_mask = root_legal_walls
            keep = []
            for sc, mv in moves:
                if mv < 128:
                    keep.append((sc, mv))
                elif mv < 192:
                    if (lh_mask >> (mv - 128)) & 1:
                        keep.append((sc, mv))
                elif (lv_mask >> (mv - 192)) & 1:
                    keep.append((sc, mv))
            moves = keep

        # Thứ tự cuối cùng: nước trong bảng chuyển vị, rồi 2 killer của tầng, rồi theo
        # điểm tĩnh giảm dần (hoà điểm: nước tường có history cao hơn đứng trước).
        h = HISTORY[me]
        moves.sort(key=lambda it: (it[0], h[it[1]] if it[1] >= 128 else 0), reverse=True)
        for a in (k1, k0, tt_move):
            if a:
                for i, (_, mv) in enumerate(moves):
                    if mv == a:
                        moves.insert(0, moves.pop(i))
                        break
        return [mv for _, mv in moves]

    # ------------------------------------------------------------ 3c. Đi / lùi nước
    def make_move(self, mv):
        """Đi một nước tại chỗ; trả về dữ liệu để unmake_move khôi phục."""

        me = self.side
        opp = me ^ 1
        zh = self.zhash

        if mv < 128:                                   # đi quân
            prev_sq = self.pawns[me]
            self.pawns[me] = mv
            self.zhash = zh ^ ZOBRIST_PAWN[me][prev_sq] ^ ZOBRIST_PAWN[me][mv] ^ ZOBRIST_SIDE
            self.side = opp
            return (prev_sq, 0, 0)

        w_left = self.walls_left[me]
        self.walls_left[me] = w_left - 1
        self.side = opp

        if mv < 192:                                   # tường ngang
            idx = mv - 128
            self.h_mask |= 1 << idx
            prev_u, prev_d = self.up_bb, self.down_bb
            self.down_bb &= ~H_WALL_TOP2[idx]
            self.up_bb &= ~H_WALL_BOT2[idx]
            self.zhash = (
                zh ^ ZOBRIST_WLEFT[me][w_left] ^ ZOBRIST_WLEFT[me][w_left - 1]
                ^ ZOBRIST_HWALL[idx] ^ ZOBRIST_SIDE
            )
            return (w_left, prev_u, prev_d)
        idx = mv - 192                                 # tường dọc
        self.v_mask |= 1 << idx
        prev_l, prev_r = self.left_bb, self.right_bb
        self.right_bb &= ~V_WALL_LEFT2[idx]
        self.left_bb &= ~V_WALL_RIGHT2[idx]
        self.zhash = (
            zh ^ ZOBRIST_WLEFT[me][w_left] ^ ZOBRIST_WLEFT[me][w_left - 1]
            ^ ZOBRIST_VWALL[idx] ^ ZOBRIST_SIDE
        )
        return (w_left, prev_l, prev_r)

    def unmake_move(self, mv, prev_zh, undo):
        """Lùi nước vừa đi (khôi phục đúng trạng thái trước make_move)."""
        opp = self.side
        me = opp ^ 1
        self.side = me
        self.zhash = prev_zh

        if mv < 128:
            self.pawns[me] = undo[0]
            return

        w_left, b1, b2 = undo
        self.walls_left[me] = w_left
        if mv < 192:
            idx = mv - 128
            self.h_mask ^= 1 << idx
            self.up_bb, self.down_bb = b1, b2
        else:
            idx = mv - 192
            self.v_mask ^= 1 << idx
            self.left_bb, self.right_bb = b1, b2

    # ------------------------------------------------------------ 3d. Negamax
    def negamax(self, depth, alpha, beta, ply):
        """Negamax alpha-beta có bảng chuyển vị, killer, history và PVS.

        Đầu vào: depth - số tầng còn lại; alpha, beta - cửa sổ; ply - tầng tính từ gốc.
        Đầu ra: điểm nhìn từ người sắp đi (0 nếu đã hết giờ).
        """
        if (self.nodes & 63) == 0 and time.perf_counter() >= self.deadline:
            self.stop = True
        if self.stop:
            return 0.0

        self.nodes += 1
        me = self.side
        opp = me ^ 1

        if self.pawns[me] // 9 == self.goals[me]:
            return WIN_SCORE - ply
        if self.pawns[opp] // 9 == self.goals[opp]:
            return -WIN_SCORE + ply
        if self.turn_base + ply >= self.max_turns:
            return 0.0

        if depth <= 0 or ply >= MAX_PLY - 2:
            return self.leaf(alpha, beta)

        orig_alpha = alpha
        zh = self.zhash
        tt_move = 0
        tt_entry = TRANSPOSITION_TABLE.get_item(zh)

        if tt_entry is not None:
            tt_depth, tt_flag, tt_score, tt_move = tt_entry
            if tt_score > 80000:
                tt_score -= ply
            elif tt_score < -80000:
                tt_score += ply
            if tt_depth >= depth:
                if tt_flag == TT_EXACT:
                    return tt_score
                if tt_flag == TT_LOWER and tt_score >= beta:
                    return tt_score
                if tt_flag == TT_UPPER and tt_score <= alpha:
                    return tt_score

        best_score = -INF
        best_move = 0
        searched_tt = False

        if tt_move:
            searched_tt = True
            undo = self.make_move(tt_move)
            score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            self.unmake_move(tt_move, zh, undo)
            if self.stop:
                return 0.0
            best_score = score
            best_move = tt_move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    TRANSPOSITION_TABLE.put_item(zh, (depth, TT_LOWER, best_score, best_move))
                    return best_score

        moves = self.gen_moves(ply, tt_move)
        if not moves:
            return self.leaf(alpha, beta)
        if best_move == 0:
            best_move = moves[0]

        k0, k1 = KILLERS[ply]
        quiets = []

        for i, mv in enumerate(moves):
            if searched_tt and mv == tt_move:
                continue
            undo = self.make_move(mv)
            if not searched_tt and i == 0:
                score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            else:
                score = -self.negamax(depth - 1, -alpha - EPS, -alpha, ply + 1)
                if not self.stop and alpha < score < beta:
                    score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            self.unmake_move(mv, zh, undo)
            if self.stop:
                return 0.0

            if score > best_score:
                best_score = score
                best_move = mv
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if mv >= 128:                 # chỉ tường mới ghi killer/history
                            if k0 != mv:
                                KILLERS[ply][1] = k0
                                KILLERS[ply][0] = mv
                            bonus = depth * depth * 4
                            HISTORY[me][mv] += bonus
                            for qm in quiets:
                                HISTORY[me][qm] -= bonus
                        break
            if mv >= 128:
                quiets.append(mv)

        if not self.stop:
            store_score = best_score
            if store_score > 80000:
                store_score += ply
            elif store_score < -80000:
                store_score -= ply
            flag = TT_UPPER if best_score <= orig_alpha else (TT_LOWER if best_score >= beta else TT_EXACT)
            TRANSPOSITION_TABLE.put_item(zh, (depth, flag, store_score, best_move))

        return best_score

# ============================================================================
# 4. CUỘC ĐUA CHÍNH XÁC KHI HẾT TƯỜNG + LUẬT "GIỮ CHỐT" + LUẬT "GIỮ BỨC CUỐI"
# ============================================================================
# Hai chiến thuật lấy từ khảo sát các bot top vòng 1 (xem phat-training/EXPERIMENTS.md #53).
# 
# 4a. Khi cả hai đã hết tường thì bàn cờ ĐỨNG YÊN: ván chỉ còn hai quân đi trên một đồ thị
#     cố định, nên giải được CHÍNH XÁC bằng duyệt ngược (retrograde) toàn bộ 2*81*81 thế cờ,
#     kể cả nước "pass" hợp lệ. Nhờ vậy cuối ván bot không đoán theo hàm đánh giá nữa:
#     thắng về nhanh nhất, thua câu giờ lâu nhất, hòa thì giữ hòa và tiến lên.
#     Bảng chỉ phụ thuộc tập tường + hai hàng đích nên dựng một lần cho cả ván.
#
# 4b. "Giữ chốt" (law 4c PLUG_HOLD_SEAL_ONLY): khi ta còn <= 2 tường (sắp cạn), đối thủ còn >= 2 tường
#     và ta đã vào tàn cuộc (d_me <= 6) thì mỗi nước TƯỜNG ở gốc chỉ được giữ nếu nó làm đối thủ chậm
#     >= lim bước, với lim = max(2, threat) (chặn trên 4) và threat = số bước lớn nhất rằng MỘT
#     bức tường của đối thủ có thể cộng vào đường ngắn nhất của ta. Tường "hoà nhịp" (lãi <= 1 bước)
#     bị loại khỏi danh sách nước gốc, nên bot đi quân và giữ tường cho lượt đối thủ bít mình.
#
# 4c. "Giữ bức cuối" (từ Phat_buffed_4): giữ tường để đáp trả bít hàng đích, đặc biệt quan trọng
#     khi cả hai hết tường và hàm đánh giá không còn có tác dụng.