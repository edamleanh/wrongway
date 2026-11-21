"""Bot của Phát (phiên bản Phat_test2 nâng cấp toàn diện - Hành lang hộ giá & Bitboard đỉnh cao):
1. Động cơ Bitboard 81-bit in-place & Zobrist Hashing O(1):
   - Mô hình hóa bàn cờ bằng bitboard song song, cập nhật trạng thái in-place không cấp phát bộ nhớ.
   - Bảng chuyển vị (Transposition Table) và Goal Layers Cache kiểu LRU tăng tốc độ sâu lên 10-14 ply.
2. Chiến thuật Hành lang hộ giá (Escort Corridor & Flank Shielding):
   - Phân tích toàn bộ DAG các đường ngắn nhất (shortest_dag_analysis) để phát hiện điểm nóng chí tử (Threat Chokepoints).
   - Khi đối thủ có nguy cơ chặn ngang (detour >= 2), bot chủ động sinh tường hộ giá dọc sườn (V-wall cho đường chạy dọc,
     H-wall cho đường chạy ngang) nhằm chiếm giữ giao điểm chữ thập, vĩnh viễn tước đoạt quyền chặn đầu của đối thủ.
   - Thưởng nóng Move Ordering (+260) cho các tường hộ giá bảo vệ hành lang mà không làm chậm quân mình.
3. Bộ đôi trọng số tối ưu tách biệt Đi Trước (P0) & Đi Sau (P1):
   - Chi phí tường bậc hai (wall_quad): Đặt tường đầu trận cực rẻ -> tự tin dùng tường hộ giá sớm.
   - Đo độ rộng hành lang (dag_width_diff): Tránh bị dồn vào khe hẹp 1 ô, tích cực bóp nghẹt hành lang đối thủ.
   - Nhạy cảm phòng thủ (vuln_diff) & Khóa nhịp nhảy đầu (jump_tempo).
4. Ngân sách thời gian an toàn chuẩn mực:
   - TIME_BUDGET = 0.56s (an toàn tuyệt đối dưới ngưỡng 0.70s và 1.0s của Arena).
"""

import time
from collections import OrderedDict

# ============================================================================
# BỘ TRỌNG SỐ TÁCH BIỆT ĐI TRƯỚC (P0) VÀ ĐI SAU (P1)
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
]

# Trọng số khi đi trước (P0): Chủ động tiến công, chiếm trung tâm
WEIGHTS_FIRST = {
    "dist_diff": 117.0000,
    "race_urgency": 121.0500,
    "wall_diff": 205.7036,
    "wall_quad": -3.6750,
    "dag_width_diff": 5.7500,
    "center_diff": 8.8000,
    "jump_tempo": 36.2500,
    "tempo": 44.1383,
    "vuln_diff": 30.1084,
}

# Trọng số khi đi sau (P1): Chi phí tường rẻ hơn, nhạy bén hộ giá & phòng ngự phản công
WEIGHTS_SECOND = {
    "dist_diff": 117.0000,
    "race_urgency": 121.2422,
    "wall_diff": 193.8156,
    "wall_quad": -3.4492,
    "dag_width_diff": 5.7500,
    "center_diff": 8.9078,
    "jump_tempo": 36.2500,
    "tempo": 43.5434,
    "vuln_diff": 30.1289,
}

WEIGHTS = WEIGHTS_FIRST
BASE_FEATURE_KEYS = FEATURE_KEYS[:8]
TIME_BUDGET = 0.56  # Ngân sách thời gian theo yêu cầu: 0.56s


def get_weights_for_obs(obs, weights=None):
    if weights is not None:
        return weights
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

BASE_UP = (MASK_81 >> 9) << 9
BASE_DOWN = MASK_81 >> 9
BASE_LEFT = MASK_81 & NOT_COL0
BASE_RIGHT = MASK_81 & NOT_COL8

H_WALL_SQUARES = []
V_WALL_SQUARES = []
H_WALL_TOP2 = [0] * 64
H_WALL_BOT2 = [0] * 64
V_WALL_LEFT2 = [0] * 64
V_WALL_RIGHT2 = [0] * 64

H_CONFLICT_H = [0] * 64
V_CONFLICT_V = [0] * 64
AROUND_SQ_WALLS = [0] * 81
TRAP_BOX_WALLS = [0] * 81

for _r in range(8):
    for _c in range(8):
        _idx = _r * 8 + _c
        _s1 = _r * 9 + _c
        _s2 = _s1 + 1
        _s3 = _s1 + 9
        _s4 = _s1 + 10
        H_WALL_SQUARES.append((_s1, _s2, _s3, _s4))
        V_WALL_SQUARES.append((_s1, _s3, _s2, _s4))

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
    _tb = 0
    for _dr in (-1, 0):
        for _dc in (-1, 0):
            _wr, _wc = _r + _dr, _c + _dc
            if 0 <= _wr < 8 and 0 <= _wc < 8:
                _m |= 1 << (_wr * 8 + _wc)
    AROUND_SQ_WALLS[_sq] = _m
    for _dr in (-2, -1, 0, 1):
        for _dc in (-2, -1, 0, 1):
            _wr, _wc = _r + _dr, _c + _dc
            if 0 <= _wr < 8 and 0 <= _wc < 8:
                _tb |= 1 << (_wr * 8 + _wc)
    TRAP_BOX_WALLS[_sq] = _tb

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

INF = 100000
WIN_SCORE = 90000
MAX_PLY = 64
TT_EXACT = 1
TT_LOWER = 2
TT_UPPER = 3


# ============================================================================
# 2. BOUNDED LRU CACHE & REVERSE GOAL LAYERS + LOCAL RE-BFS
# ============================================================================
class BoundedCache(OrderedDict):
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


GOAL_LAYERS_CACHE = BoundedCache(32000)
TRANSPOSITION_TABLE = BoundedCache(110000)
KILLERS = [[0, 0] for _ in range(MAX_PLY)]
HISTORY = [[0] * 256 for _ in range(2)]
LAST_TURN = -1


def clear_search_memory():
    GOAL_LAYERS_CACHE.clear()
    TRANSPOSITION_TABLE.clear()
    for p in range(MAX_PLY):
        KILLERS[p][0] = 0
        KILLERS[p][1] = 0
    for s in range(2):
        for m in range(256):
            HISTORY[s][m] = 0


def get_goal_layers(h_mask, v_mask, goal_row, up_bb, down_bb, left_bb, right_bb):
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
    bit = 1 << sq
    for d, l_bb in enumerate(layers):
        if l_bb & bit:
            return d
    return 999


def shortest_dag_analysis(layers, start_sq, d_start, up_bb, down_bb, left_bb, right_bb):
    """Phân tích toàn bộ DAG đường ngắn nhất:
    Trả về (wall_zone_64bit, br_count, dag_cell_count).
    - wall_zone_64bit: toàn bộ các vị trí tường có thể cắt ít nhất 1 đường ngắn nhất.
    - br_count: số hướng bước đầu độc lập.
    - dag_cell_count: tổng số ô trong DAG (độ rộng hành lang).
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

    wall_zone = 0
    tmp = dag_sq_bb
    while tmp:
        lsb = tmp & -tmp
        wall_zone |= AROUND_SQ_WALLS[lsb.bit_length() - 1]
        tmp ^= lsb

    return wall_zone, br_count, popcount81(dag_sq_bb)


def dist_after_wall(layers, start_sq, d_orig, wall_idx, is_vert, up_bb, down_bb, left_bb, right_bb):
    if d_orig >= 999:
        return 999

    max_d = len(layers) - 1
    if not is_vert:
        t2 = H_WALL_TOP2[wall_idx]
        b2 = H_WALL_BOT2[wall_idx]
        n_up = up_bb & ~b2
        n_down = down_bb & ~t2
        n_left = left_bb
        n_right = right_bb
        affected_bb = t2 | b2
    else:
        l2 = V_WALL_LEFT2[wall_idx]
        r2 = V_WALL_RIGHT2[wall_idx]
        n_up = up_bb
        n_down = down_bb
        n_left = left_bb & ~r2
        n_right = right_bb & ~l2
        affected_bb = l2 | r2

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
# 3. TRẠNG THÁI BITBOARD & TÌM KIẾM ALPHA-BETA VỚI HÀNH LANG HỘ GIÁ
# ============================================================================
class BitboardQuoridorState:
    __slots__ = (
        "up_bb", "down_bb", "left_bb", "right_bb",
        "h_mask", "v_mask",
        "pawns", "goals", "walls_left",
        "side", "zhash", "turn_base", "max_turns",
        "deadline", "nodes", "stop", "weights"
    )

    def __init__(self, obs, deadline, weights=None):
        self.weights = get_weights_for_obs(obs, weights)
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

    def _pocket_size_3(self, sq):
        b0 = 1 << sq
        u, d, l, r = self.up_bb, self.down_bb, self.left_bb, self.right_bb
        b1 = b0 | ((b0 & u) >> 9) | ((b0 & d) << 9) | ((b0 & l) >> 1) | ((b0 & r) << 1)
        b2 = b1 | ((b1 & u) >> 9) | ((b1 & d) << 9) | ((b1 & l) >> 1) | ((b1 & r) << 1)
        b3 = b2 | ((b2 & u) >> 9) | ((b2 & d) << 9) | ((b2 & l) >> 1) | ((b2 & r) << 1)
        return popcount81(b3)

    def _can_straight_jump(self, p1, p2):
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

    def compute_vulnerability(self, layers_target, sq_target, d_target, zone_mask):
        max_detour = 0
        hm = self.h_mask
        vm = self.v_mask
        cm = zone_mask
        while cm:
            lsb = cm & -cm
            idx = lsb.bit_length() - 1
            cm ^= lsb
            if not ((hm & H_CONFLICT_H[idx]) or (vm & lsb)):
                nd = dist_after_wall(
                    layers_target, sq_target, d_target, idx, False,
                    self.up_bb, self.down_bb, self.left_bb, self.right_bb
                )
                if nd < 999 and nd - d_target > max_detour:
                    max_detour = nd - d_target
            if not ((vm & V_CONFLICT_V[idx]) or (hm & lsb)):
                nd = dist_after_wall(
                    layers_target, sq_target, d_target, idx, True,
                    self.up_bb, self.down_bb, self.left_bb, self.right_bb
                )
                if nd < 999 and nd - d_target > max_detour:
                    max_detour = nd - d_target
        return float(min(max_detour, 8))

    def extract_features(self, alpha=-INF, beta=INF, force_full=False):
        me = self.side
        opp = me ^ 1
        p_me = self.pawns[me]
        p_opp = self.pawns[opp]
        g_me = self.goals[me]
        g_opp = self.goals[opp]

        layers_me = get_goal_layers(
            self.h_mask, self.v_mask, g_me,
            self.up_bb, self.down_bb, self.left_bb, self.right_bb
        )
        layers_opp = get_goal_layers(
            self.h_mask, self.v_mask, g_opp,
            self.up_bb, self.down_bb, self.left_bb, self.right_bb
        )

        d_me = dist_from_layers(layers_me, p_me)
        d_opp = dist_from_layers(layers_opp, p_opp)

        zone_me, br_me, dag_cells_me = shortest_dag_analysis(
            layers_me, p_me, d_me, self.up_bb, self.down_bb, self.left_bb, self.right_bb
        )
        zone_opp, br_opp, dag_cells_opp = shortest_dag_analysis(
            layers_opp, p_opp, d_opp, self.up_bb, self.down_bb, self.left_bb, self.right_bb
        )

        w_me = float(self.walls_left[me])
        w_opp = float(self.walls_left[opp])
        lead = float(d_opp - d_me)

        reserve_lead = (lead * w_me) if lead > 0 else ((lead * w_opp) if lead < 0 else 0.0)
        wall_starve = (w_me if (w_opp == 0 and w_me > 0) else 0.0) - (w_opp if (w_me == 0 and w_opp > 0) else 0.0)

        dag_w_me = dag_cells_me / (d_me + 1.0) if d_me < 999 else 0.0
        dag_w_opp = dag_cells_opp / (d_opp + 1.0) if d_opp < 999 else 0.0

        pocket_me = self._pocket_size_3(p_me)
        pocket_opp = self._pocket_size_3(p_opp)

        r_me, c_me = divmod(p_me, 9)
        r_opp, c_opp = divmod(p_opp, 9)

        feats = {
            "dist_diff": lead,
            "race_urgency": (10.0 / (d_me + 1.0)) - (10.0 / (d_opp + 1.0)),
            "wall_diff": w_me - w_opp,
            "wall_quad": (w_me * w_me) - (w_opp * w_opp),
            "reserve_lead": reserve_lead,
            "wall_starve": wall_starve,
            "branch_diff": float(br_me - br_opp),
            "dag_width_diff": dag_w_me - dag_w_opp,
            "pocket_diff": (pocket_me - pocket_opp) / 10.0,
            "center_diff": float((4 - abs(c_me - 4)) - (4 - abs(c_opp - 4))),
            "progress_diff": float((8 - abs(r_me - g_me)) - (8 - abs(r_opp - g_opp))),
            "jump_tempo": self._can_straight_jump(p_me, p_opp) - self._can_straight_jump(p_opp, p_me),
            "tempo": 1.0,
            "vuln_diff": 0.0,
            "vuln_exploit": 0.0,
        }

        w = self.weights
        base_score = sum(w[k] * feats[k] for k in BASE_FEATURE_KEYS)

        max_vuln_swing = 8.0 * abs(w["vuln_diff"])
        if force_full or ((base_score + max_vuln_swing > alpha) and (base_score - max_vuln_swing < beta)):
            vuln_me = self.compute_vulnerability(layers_me, p_me, d_me, zone_me) if w_opp > 0 else 0.0
            vuln_opp = self.compute_vulnerability(layers_opp, p_opp, d_opp, zone_opp) if w_me > 0 else 0.0
            feats["vuln_diff"] = vuln_opp - vuln_me

        return feats

    def evaluate(self, alpha=-INF, beta=INF):
        me = self.side
        opp = me ^ 1
        if self.pawns[me] // 9 == self.goals[me]:
            return WIN_SCORE
        if self.pawns[opp] // 9 == self.goals[opp]:
            return -WIN_SCORE

        feats = self.extract_features(alpha=alpha, beta=beta, force_full=False)
        w = self.weights
        score = sum(w[k] * feats[k] for k in FEATURE_KEYS)
        return int(round(score))

    def gen_pawn_moves(self):
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
                if dir_bb & opp_bit:
                    dst = opp_sq + delta
                    prog = cur_row_d - abs(dst // 9 - goal_r)
                    out.append((690 + prog * 110, dst))
                else:
                    perp = (dirs[2], dirs[3]) if idx < 2 else (dirs[0], dirs[1])
                    for p_bb, p_delta in perp:
                        if p_bb & opp_bit:
                            dst = opp_sq + p_delta
                            prog = cur_row_d - abs(dst // 9 - goal_r)
                            out.append((665 + prog * 110, dst))
        return out

    def gen_moves(self, ply, tt_move, is_root=False, root_legal_pawn_set=None, root_legal_walls=None):
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

        zone_opp, _, _ = shortest_dag_analysis(
            layers_opp, p_opp, d_opp, self.up_bb, self.down_bb, self.left_bb, self.right_bb
        )
        zone_me, _, _ = shortest_dag_analysis(
            layers_me, p_me, d_me, self.up_bb, self.down_bb, self.left_bb, self.right_bb
        )

        def_h_mask = 0
        def_v_mask = 0
        # ====================================================================
        # CHIẾN THUẬT HÀNH LANG HỘ GIÁ (ESCORT CORRIDOR) & THREAT CHOKEPOINTS
        # Phát hiện mọi điểm nóng mà đối thủ có thể chặn ta trễ >= 2 bước
        # Tự động sinh tường hộ giá chiếm giữ giao điểm chữ thập và che sườn
        # ====================================================================
        if self.walls_left[opp] > 0 and (is_root or ply <= 1):
            cm_def = zone_me
            while cm_def:
                lsb = cm_def & -cm_def
                idx = lsb.bit_length() - 1
                cm_def ^= lsb

                # 1. Đối thủ dọa đặt tường ngang H tại idx
                if not ((self.h_mask & H_CONFLICT_H[idx]) or (self.v_mask & lsb)):
                    nd_m = dist_after_wall(
                        layers_me, p_me, d_me, idx, False,
                        self.up_bb, self.down_bb, self.left_bb, self.right_bb
                    )
                    if nd_m < 999 and nd_m - d_me >= 2:
                        # Hộ giá chí mạng: Tường dọc V tại idx khóa giao điểm chữ thập
                        def_v_mask |= lsb
                        def_h_mask |= H_CONFLICT_H[idx]
                        # Mở rộng hộ giá dọc sườn hành lang (flank escort)
                        _r_idx, _c_idx = divmod(idx, 8)
                        if _r_idx > 0:
                            def_v_mask |= 1 << (idx - 8)
                        if _r_idx < 7:
                            def_v_mask |= 1 << (idx + 8)

                # 2. Đối thủ dọa đặt tường dọc V tại idx
                if not ((self.v_mask & V_CONFLICT_V[idx]) or (self.h_mask & lsb)):
                    nd_m = dist_after_wall(
                        layers_me, p_me, d_me, idx, True,
                        self.up_bb, self.down_bb, self.left_bb, self.right_bb
                    )
                    if nd_m < 999 and nd_m - d_me >= 2:
                        # Hộ giá chí mạng: Tường ngang H tại idx khóa giao điểm chữ thập
                        def_h_mask |= lsb
                        def_v_mask |= V_CONFLICT_V[idx]
                        # Mở rộng hộ giá ngang sườn hành lang (flank escort)
                        _r_idx, _c_idx = divmod(idx, 8)
                        if _c_idx > 0:
                            def_h_mask |= 1 << (idx - 1)
                        if _c_idx < 7:
                            def_h_mask |= 1 << (idx + 1)

        cand_h = zone_opp | def_h_mask | TRAP_BOX_WALLS[p_opp]
        cand_v = zone_opp | def_v_mask | TRAP_BOX_WALLS[p_opp]

        if is_root and root_legal_walls is not None:
            lh_mask, lv_mask = root_legal_walls
            cand_h &= lh_mask
            cand_v &= lv_mask

        wall_candidates = []
        k0, k1 = KILLERS[ply]

        for is_vert, cand_mask in ((False, cand_h), (True, cand_v)):
            cm = cand_mask
            while cm:
                lsb = cm & -cm
                idx = lsb.bit_length() - 1
                cm ^= lsb

                if not is_vert:
                    if (self.h_mask & H_CONFLICT_H[idx]) or (self.v_mask & lsb):
                        continue
                else:
                    if (self.v_mask & V_CONFLICT_V[idx]) or (self.h_mask & lsb):
                        continue

                nd_opp = dist_after_wall(
                    layers_opp, p_opp, d_opp, idx, is_vert,
                    self.up_bb, self.down_bb, self.left_bb, self.right_bb
                )
                if nd_opp >= 999:
                    continue

                nd_me = dist_after_wall(
                    layers_me, p_me, d_me, idx, is_vert,
                    self.up_bb, self.down_bb, self.left_bb, self.right_bb
                )
                if nd_me >= 999:
                    continue

                det_opp = nd_opp - d_opp
                det_me = nd_me - d_me
                is_def = ((def_v_mask if is_vert else def_h_mask) & lsb) != 0

                # Lọc tường vô hại hoặc tự hại
                if det_opp <= 0 and not is_def:
                    continue
                if det_me > det_opp and not is_def:
                    continue
                if is_def and det_me > 0:
                    # Tường hộ giá tuyệt đối không được tự làm chậm mình
                    continue

                mv = (192 + idx) if is_vert else (128 + idx)
                # Thưởng lớn (+260) cho tường hộ giá / phòng thủ bảo vệ lối đi an toàn
                sc = det_opp * 150 - det_me * 155 + (260 if is_def and det_me <= 0 else 0)
                sc += HISTORY[me][mv] >> 4
                if mv == k0:
                    sc += 420
                elif mv == k1:
                    sc += 360
                wall_candidates.append((sc, mv))

        wall_candidates.sort(reverse=True)
        max_walls = 18 if is_root else 12
        moves.extend(wall_candidates[:max_walls])

        if tt_move:
            for i, (_, mv) in enumerate(moves):
                if mv == tt_move:
                    moves[i] = (1000000, mv)
                    break

        moves.sort(reverse=True)
        return [mv for _, mv in moves]

    def make_move(self, mv):
        me = self.side
        opp = me ^ 1
        zh = self.zhash

        if mv < 128:
            prev_sq = self.pawns[me]
            self.pawns[me] = mv
            self.zhash = zh ^ ZOBRIST_PAWN[me][prev_sq] ^ ZOBRIST_PAWN[me][mv] ^ ZOBRIST_SIDE
            self.side = opp
            return (prev_sq, 0, 0)

        w_left = self.walls_left[me]
        self.walls_left[me] = w_left - 1
        self.side = opp

        if mv < 192:
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
        else:
            idx = mv - 192
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

    def negamax(self, depth, alpha, beta, ply):
        if (self.nodes & 63) == 0:
            if time.perf_counter() >= self.deadline:
                self.stop = True
        if self.stop:
            return 0

        self.nodes += 1
        me = self.side
        opp = me ^ 1

        if self.pawns[me] // 9 == self.goals[me]:
            return WIN_SCORE - ply
        if self.pawns[opp] // 9 == self.goals[opp]:
            return -WIN_SCORE + ply
        if self.turn_base + ply >= self.max_turns:
            return 0

        if depth <= 0 or ply >= MAX_PLY - 2:
            return self.evaluate(alpha, beta)

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

            if tt_depth >= depth and (beta - alpha <= 1):
                if tt_flag == TT_EXACT:
                    return tt_score
                if tt_flag == TT_LOWER and tt_score >= beta:
                    return tt_score
                if tt_flag == TT_UPPER and tt_score <= alpha:
                    return tt_score

        best_score = -INF
        best_move = 0
        searched_tt = False

        if tt_move != 0:
            searched_tt = True
            undo = self.make_move(tt_move)
            score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            self.unmake_move(tt_move, zh, undo)
            if self.stop:
                return 0

            best_score = score
            best_move = tt_move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    TRANSPOSITION_TABLE.put_item(zh, (depth, TT_LOWER, best_score, best_move))
                    return best_score

        moves = self.gen_moves(ply, tt_move, is_root=False)
        if not moves:
            return self.evaluate(alpha, beta)
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
                score = -self.negamax(depth - 1, -alpha - 1, -alpha, ply + 1)
                if not self.stop and alpha < score < beta:
                    score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)

            self.unmake_move(mv, zh, undo)
            if self.stop:
                return 0

            if score > best_score:
                best_score = score
                best_move = mv
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if mv >= 128:
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
# 4. GIAO TIẾP VỚI ARENA
# ============================================================================
def _encode_move(mv):
    if mv < 128:
        return {"type": "move", "to": [mv // 9, mv % 9]}
    if mv < 192:
        idx = mv - 128
        return {"type": "wall", "orientation": "H", "row": idx // 8, "col": idx % 8}
    idx = mv - 192
    return {"type": "wall", "orientation": "V", "row": idx // 8, "col": idx % 8}


def choose_action_with_info(observation, weights=None, time_budget=TIME_BUDGET):
    global LAST_TURN
    t0 = time.perf_counter()
    deadline = t0 + time_budget

    legal_moves = observation.get("legal_moves") or []
    legal_walls = observation.get("legal_walls") or []
    if not legal_moves and not legal_walls:
        return {"type": "pass"}, 0

    goal_row = observation["you"]["goal_row"]
    for r, c in legal_moves:
        if r == goal_row:
            return {"type": "move", "to": [r, c]}, WIN_SCORE

    if legal_moves:
        best_fb = min(legal_moves, key=lambda rc: (abs(rc[0] - goal_row), abs(rc[1] - 4)))
        fallback = {"type": "move", "to": best_fb}
    else:
        w0 = legal_walls[0]
        fallback = {"type": "wall", "orientation": w0["orientation"], "row": w0["row"], "col": w0["col"]}

    turn = observation.get("turn", 0)
    if turn <= 1 or turn < LAST_TURN:
        clear_search_memory()
    LAST_TURN = turn

    try:
        active_weights = get_weights_for_obs(observation, weights)
        state = BitboardQuoridorState(observation, deadline, weights=active_weights)
        root_legal_pawn_set = {r * 9 + c for r, c in legal_moves}

        lh_mask = 0
        lv_mask = 0
        for w in legal_walls:
            idx = w["row"] * 8 + w["col"]
            if w["orientation"] == "V":
                lv_mask |= 1 << idx
            else:
                lh_mask |= 1 << idx

        tt_entry = TRANSPOSITION_TABLE.get_item(state.zhash)
        tt_move = tt_entry[3] if tt_entry is not None else 0

        root_moves = state.gen_moves(
            0, tt_move, is_root=True,
            root_legal_pawn_set=root_legal_pawn_set,
            root_legal_walls=(lh_mask, lv_mask)
        )
        for sq in root_legal_pawn_set:
            if sq not in root_moves:
                root_moves.append(sq)

        if not root_moves:
            return fallback, 0
        if len(root_moves) == 1:
            return _encode_move(root_moves[0]), state.evaluate()

        best_move = root_moves[0]
        best_score = -INF
        zh = state.zhash

        for depth in range(1, 14):
            if time.perf_counter() >= deadline:
                break

            iter_best_move = best_move
            iter_best_score = -INF
            alpha = -INF
            beta = INF
            completed_any = False

            for i, mv in enumerate(root_moves):
                if time.perf_counter() >= deadline:
                    state.stop = True
                    break

                undo = state.make_move(mv)
                if i == 0:
                    sc = -state.negamax(depth - 1, -beta, -alpha, 1)
                else:
                    sc = -state.negamax(depth - 1, -alpha - 1, -alpha, 1)
                    if not state.stop and alpha < sc < beta:
                        sc = -state.negamax(depth - 1, -beta, -alpha, 1)
                state.unmake_move(mv, zh, undo)

                if state.stop:
                    break

                completed_any = True
                is_better = False
                if sc > iter_best_score:
                    is_better = True
                elif sc == iter_best_score and mv < 128 and iter_best_move < 128:
                    if abs(mv // 9 - goal_row) < abs(iter_best_move // 9 - goal_row):
                        is_better = True

                if is_better:
                    iter_best_score = sc
                    iter_best_move = mv
                    if sc > alpha:
                        alpha = sc

            if completed_any and (not state.stop or iter_best_score > best_score):
                best_move = iter_best_move
                best_score = iter_best_score
                root_moves.remove(best_move)
                root_moves.insert(0, best_move)

            if state.stop or best_score > WIN_SCORE - 50:
                break

        return _encode_move(best_move), best_score
    except Exception:
        return fallback, 0


def choose_action(observation):
    action, _ = choose_action_with_info(observation, weights=None, time_budget=TIME_BUDGET)
    return action
