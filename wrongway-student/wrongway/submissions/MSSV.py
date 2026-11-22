"""Agent Wrongway: tìm kiếm alpha-beta + hàm đánh giá tra bảng học bằng TD-leaf(lambda).

Tổng quan (lý do và số liệu thử nghiệm nằm trong báo cáo PDF):
1. Luật chơi được mô phỏng bằng BITBOARD: cả bàn cờ là một số nguyên 81 bit (bit i = ô i,
   ô i = hàng * 9 + cột); BFS tính khoảng cách loang mọi ô cùng lúc bằng phép dịch bit.
2. Tìm kiếm negamax alpha-beta, đào sâu dần tới khi hết TIME_BUDGET, có bảng chuyển vị
   và sắp thứ tự nước (nước trong bảng chuyển vị, killer, history, PVS). Ứng viên mỗi nút:
   mọi nước đi + tối đa WALL_CANDIDATES tường trên đường ngắn nhất của đối thủ; riêng ở gốc
   thêm các tường phòng thủ.
3. Hàm đánh giá = tổng các ô tra bảng (khoảng cách, số tường, mức dẫn, độ dễ bị chặn).
   Các ô được học qua tự đấu (training/td_learn.py) rồi ghi thẳng vào khối WEIGHTS bên dưới,
   vì file nộp không được đọc file ngoài.

Ký hiệu dùng xuyên suốt:
- p: người chơi sắp đi ở nút đang xét; q = 1 - p: đối thủ của p; me: mình (ở gốc).
- pos[i], goals[i], wl[i], dist[i]: ô đang đứng, hàng đích, số tường còn lại và số bước
  tới đích của người chơi i.
- (o, r, c): bức tường hướng o ("H" ngang, "V" dọc) có góc trên trái ở hàng r, cột c.
- hw, vw: tập tường ngang / dọc dạng {(r, c)}; blocked: bộ 4 bitboard "đi được".

Ràng buộc nộp: một file, Python 3.9+, chỉ thư viện chuẩn, < 1 s mỗi nước,
import < 5 s, RAM <= 128 MB.
"""

import time

# ---------------------------------------------------------------- 1. Cấu hình
SIZE = 9
N = SIZE * SIZE
TIME_BUDGET = 0.55       # giây cho mỗi nước (giới hạn thật 1 s, Colab chậm hơn máy)
MAX_DEPTH = 10           # trần độ sâu của đào sâu dần (thế cờ ít tường tới được 8-10)
WALL_CANDIDATES = 8      # số tường tốt nhất được xét ở mỗi nút
DEFENSE_THREATS = 3      # số tường nguy hiểm nhất của đối thủ cần tìm cách chặn trước (ở gốc)
VULN_STEPS = 8           # độ dễ bị chặn: chỉ thử tường trên VULN_STEPS bước đầu của đường đi
KILLERS = True           # nhớ 2 nước gây cắt tỉa ở mỗi tầng, thử sớm ở nút anh em
HISTORY = True           # điểm lịch sử: nước hay gây cắt tỉa được ưu tiên khi hòa thứ tự
PVS = True               # tìm cửa sổ hẹp cho các nước sau nước đầu
LAZY_TT = True           # thử nước trong bảng TT trước khi sinh mọi nước con
EPS = 1e-6               # độ rộng cửa sổ hẹp của PVS
RUSH_WALLS = 0           # còn <= RUSH_WALLS tường: ở gốc chỉ xét nước tiến gần đích (+ tường)
PROGRESS_BONUS = 0.001   # ở gốc: khi hòa điểm, ưu tiên nước đưa mình gần đích hơn (phá thế hòa)
WIN = 10000.0            # điểm của thế thắng (lớn hơn mọi điểm đánh giá)
DIST_CAP = 40            # bảng dist: khoảng cách lớn nhất có ô riêng
OPP_CAP = 20             # bảng walls: khoảng cách đối thủ lớn nhất có cột riêng
LEAD_CAP = 6             # bảng lead: mức dẫn/thua lớn nhất có cột riêng (-LEAD_CAP..LEAD_CAP)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))   # lên, xuống, trái, phải (cùng thứ tự với 4 bitboard)

# Hàm đánh giá = tổng các ô TRA BẢNG (đặc trưng thưa), nhìn từ người sắp đi:
#   điểm = tempo + safe_race * safe + giá_trị(mình) - giá_trị(đối thủ)
#          + vuln[tường mình] * vuln_đối_thủ - vuln[tường đối thủ] * vuln_mình
#   giá_trị(X) = dist[d_X] + walls[tường_X][d_Y] + lead[tường_X][d_Y - d_X]   (Y = đối phương)
#   dist[d]       : giá trị của việc còn cách đích d bước (càng xa càng âm)
#   walls[w][d]   : giá trị của việc còn w tường khi đối phương còn cách đích d bước
#   lead[w][l]    : thưởng/phạt theo mức dẫn l (-6..6 bước) khi còn w tường (giữ tường khi dẫn xa)
#   vuln[w]       : mỗi bước "dễ bị chặn" của đối phương đáng bao nhiêu khi mình còn w tường
#   safe_race     : +1 nếu đối thủ hết tường và mình về trước, -1 nếu ngược lại
#   tempo         : hằng số cộng cho người sắp đi (được đi trước)
# Đổi vai hai người chơi thì điểm đổi dấu (trừ tempo) -> chấm như nhau ở độ sâu chẵn và lẻ.
# Các ô được học bằng TD-leaf(lambda) qua tự đấu (training/td_learn.py).
# Sau khi học, phần chênh walls[w][d] - walls[0][d] được nhân 1.2 (tường còn trong tay đáng giá
# hơn): tự đấu chỉ học lối chơi của chính bot nên đánh giá thấp tường trước đối thủ biết giữ tường.
# ======================= WEIGHTS BEGIN (tự sinh bởi training/td_learn.py) =======================
WEIGHTS = {
    "tempo": 1.0091,
    "safe_race": 6.8991,
    "dist": [0.0, -0.5374, -2.5276, -4.2, -5.6729, -7.0722, -8.4326, -9.7688, -11.1486, -12.4701, -13.8578, -15.2127, -16.5472, -17.8974, -19.2694, -20.5943, -21.932, -23.3368, -24.7457, -26.1292, -27.5081, -28.8939, -30.309, -31.6937, -33.1009, -34.508, -35.9169, -37.2822, -38.6638, -40.0111, -41.3883, -42.7534, -44.176, -45.5556, -46.937, -48.3175, -49.698, -51.0785, -52.459, -53.8395, -55.22],
    "walls": [
        [0.0, -0.0149, 0.1793, 0.3276, 0.3757, 0.3182, 0.2471, 0.1686, 0.1139, 0.0597, 0.0194, -0.0277, -0.0811, -0.1325, -0.1493, -0.1625, -0.1985, -0.2299, -0.1814, -0.183, -0.2347],
        [2.3825, 1.1124, 1.7815, 2.2916, 2.433, 2.5113, 2.5091, 2.4556, 2.4311, 2.4653, 2.4253, 2.3838, 2.4244, 2.4012, 2.3497, 2.3539, 2.3315, 2.3205, 2.3852, 2.44, 2.4381],
        [4.765, 4.2812, 4.623, 4.816, 4.8834, 4.996, 4.9854, 4.9532, 4.9344, 4.9899, 5.0281, 4.9326, 4.9552, 4.8944, 4.7988, 4.9123, 4.8915, 4.9243, 4.9328, 4.9489, 5.0056],
        [7.1474, 6.8003, 7.0769, 7.2107, 7.3241, 7.4731, 7.4962, 7.4873, 7.4413, 7.526, 7.4467, 7.4281, 7.4152, 7.4113, 7.3494, 7.4153, 7.4427, 7.4554, 7.4445, 7.4609, 7.5004],
        [9.5299, 9.326, 9.5523, 9.7373, 9.7989, 9.8851, 9.8861, 9.9894, 9.9781, 9.8997, 9.8626, 9.8893, 9.907, 10.0157, 9.8599, 9.8514, 9.9241, 9.9671, 9.896, 9.9174, 9.8763],
        [11.9124, 11.8728, 12.0906, 12.2178, 12.2637, 12.3462, 12.3927, 12.517, 12.3795, 12.4378, 12.5134, 12.3949, 12.3684, 12.371, 12.4237, 12.3548, 12.3835, 12.3561, 12.4577, 12.404, 12.4133],
        [14.2949, 14.2566, 14.4973, 14.5667, 14.6948, 14.9262, 14.9394, 15.0944, 14.8783, 14.7656, 14.7316, 15.0087, 14.8953, 14.8127, 15.2733, 14.8856, 14.6196, 14.9569, 14.8596, 14.8702, 14.8829],
        [16.6774, 16.818, 17.0025, 17.0983, 17.2799, 17.481, 17.4733, 17.3005, 17.3985, 17.2391, 17.4382, 17.3691, 17.2977, 17.2596, 17.313, 17.3231, 17.3372, 17.3499, 17.3415, 17.3436, 17.3555],
        [19.0598, 19.2534, 19.4845, 19.6184, 19.7479, 19.8855, 19.5827, 19.5584, 19.9843, 19.7762, 19.7058, 19.7475, 19.7518, 19.8006, 19.8009, 19.8158, 19.8123, 19.8215, 19.8137, 19.816, 19.8281],
        [21.4423, 21.8417, 22.0181, 22.1435, 22.0595, 22.188, 22.1854, 21.9536, 22.0995, 22.1203, 22.223, 22.2273, 22.2445, 22.2601, 22.2675, 22.2737, 22.284, 22.293, 22.2858, 22.2884, 22.3008],
        [23.8248, 24.309, 24.5161, 24.5104, 24.6085, 24.2799, 24.3877, 24.5458, 24.327, 24.5611, 24.6232, 24.6982, 24.715, 24.7304, 24.7383, 24.7449, 24.7556, 24.765, 24.7581, 24.7608, 24.7734],
    ],
    "lead": [
        [-0.0016, -0.0057, -0.0118, -0.0241, -0.0445, -0.0866, -0.005, 0.1183, 0.2145, 0.2132, 0.1943, 0.1125, -0.0366],
        [-0.1231, -0.255, -0.3614, -0.4134, -0.4387, -0.4065, -0.2, 0.2031, 0.2994, 0.3236, 0.4381, 0.4728, 0.5108],
        [-0.0536, -0.0478, -0.087, -0.0993, -0.0652, -0.0137, 0.0271, 0.2526, 0.4354, 0.608, 0.7847, 0.9542, 1.1413],
        [0.0169, 0.0669, 0.0467, 0.032, 0.0092, 0.0128, 0.0513, 0.3583, 0.5978, 0.866, 1.191, 1.429, 1.7442],
        [0.0647, 0.0699, 0.0678, 0.1049, 0.0775, 0.0533, 0.0724, 0.4154, 0.7658, 1.211, 1.4295, 1.8568, 2.3392],
        [0.0884, 0.1291, 0.1593, 0.0438, 0.1405, 0.1332, 0.0136, 0.4772, 0.9856, 1.4838, 1.9729, 2.4859, 2.9678],
        [0.1464, 0.1673, 0.1681, 0.1086, 0.2759, 0.0297, 0.1582, 0.6762, 1.1118, 1.5685, 2.2936, 2.9293, 3.564],
        [0.2168, 0.1306, 0.1043, 0.1911, 0.2625, 0.2144, 0.085, 0.6972, 1.2415, 1.9564, 2.7575, 3.517, 4.1703],
        [0.1612, 0.0582, 0.0628, 0.0918, 0.2329, 0.0447, 0.0525, 0.7087, 1.3888, 2.1595, 3.1514, 3.9875, 4.8],
        [0.086, 0.0541, 0.0501, 0.3302, -0.1796, 0.0444, 0.0946, 0.7745, 1.6705, 2.6271, 3.5927, 4.4999, 5.4],
        [0.0367, -0.0255, 0.02, 0.1345, 0.0031, 0.1631, -0.3719, 0.6607, 1.8638, 2.9882, 3.9996, 5.0, 6.0],
    ],
    "vuln": [0.0, 0.748, 0.9375, 1.0004, 1.0213, 1.1077, 1.1798, 1.1826, 1.1597, 0.9718, 0.7345],
}
# ======================= WEIGHTS END =========================================================


class Timeout(Exception):
    """Báo hết thời gian suy nghĩ; ném ra giữa cây tìm kiếm để dừng ngay."""


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
        EDGE_WALLS[_i, _b] = tuple(w for w in _pair
                                   if 0 <= w[1] < SIZE - 1 and 0 <= w[2] < SIZE - 1)


def cell(r, c):
    """Đổi tọa độ (hàng, cột) thành số thứ tự ô.

    Đầu vào: r, c - hàng và cột (0..8).
    Đầu ra: số thứ tự ô r * 9 + c (0..80).
    """
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
    """Thêm một bức tường vào tập tường dạng bitboard.

    Đầu vào: blocked - bộ 4 bitboard "đi được"; (o, r, c) - bức tường cần thêm.
    Đầu ra: bộ 4 bitboard mới (không sửa blocked cũ).
    """
    up, down, left, right = blocked
    a = r * SIZE + c
    if o == "H":     # chặn đi xuống từ (r,c),(r,c+1) và đi lên từ hai ô ngay dưới
        bits = 3 << a
        return up & ~(bits << SIZE), down & ~bits, left, right
    # V: chặn sang phải từ (r, c), (r + 1, c) và sang trái từ hai ô bên phải
    bits = (1 << a) | (1 << (a + SIZE))
    return up, down, left & ~(bits << 1), right & ~bits


def blocked_from(hw, vw):
    """Dựng tập tường dạng bitboard từ danh sách tường của observation.

    Đầu vào: hw, vw - tập tường ngang / dọc dạng {(r, c)}.
    Đầu ra: bộ 4 bitboard "đi được" (lên, xuống, trái, phải).
    """
    blocked = OPEN0
    for r, c in hw:
        blocked = with_wall(blocked, "H", r, c)
    for r, c in vw:
        blocked = with_wall(blocked, "V", r, c)
    return blocked


# ------------------------------------------------------- 3. Luật chơi
def flood(reach, d, target, up, down, left, right):
    """BFS bằng bitboard: mỗi vòng loang cả tập ô thêm một bước theo 4 hướng.

    Đầu vào: reach - bitboard các ô đã tới được sau d bước; target - bitboard ô đích;
             up, down, left, right - 4 bitboard "đi được".
    Đầu ra: số bước ít nhất để chạm một ô của target, -1 nếu bị bít.
    """
    while not reach & target:
        nxt = (reach | ((reach & up) >> 9) | ((reach & down) << 9)
               | ((reach & left) >> 1) | ((reach & right) << 1))
        if nxt == reach:
            return -1
        reach = nxt
        d += 1
    return d


def bfs_dist(start, goal_row, up, down, left, right):
    """Tính số bước ít nhất từ một ô tới hàng đích (bỏ qua quân cờ).

    Đầu vào: start - ô xuất phát; goal_row - hàng đích; up, down, left, right - tập tường.
    Đầu ra: số bước, -1 nếu bị bít đường.
    """
    return flood(1 << start, 0, ROW_BITS[goal_row], up, down, left, right)


def reach_steps(start, blocked, steps):
    """Loang từ một ô ra một số bước cố định.

    Đầu vào: start - ô xuất phát; blocked - tập tường; steps - số bước cần loang.
    Đầu ra: list reach, reach[k] = bitboard các ô tới được trong <= k bước (k = 0..steps).
    """
    up, down, left, right = blocked
    cur = 1 << start
    reach = [cur]
    for _ in range(steps):
        cur |= ((cur & up) >> 9) | ((cur & down) << 9) | ((cur & left) >> 1) | ((cur & right) << 1)
        reach.append(cur)
    return reach


def dist_after_wall(reach, i, a, layers, walled):
    """Tính khoảng cách tới đích sau khi thêm một bức tường mà không BFS lại từ đầu.

    Tường nào cũng nằm giữa 4 ô a, a+1, a+9, a+10 nên chỉ đổi khoảng cách "phía sau" 4 ô đó:
    phía xuất phát, i bước loang đầu không đổi (loang tiếp từ reach[i]); phía đích, các ô ở
    lớp k trở xuống giữ nguyên (loang tới lớp k rồi cộng k, k = lớp gần đích nhất của 4 ô).

    Đầu vào: reach - kết quả reach_steps khi chưa có tường; i - bước đầu tiên của đường đi bị
             tường chặn; a - ô góc trên trái của tường; layers - goal_layers khi chưa có tường;
             walled - tập tường đã thêm bức tường đó.
    Đầu ra: khoảng cách mới, -1 nếu bị bít đường.
    """
    square = 1539 << a                         # 1539 = 0b11_0000000_11: bit của 4 ô quanh tường
    if i and reach[i - 1] & square:
        i = 0
    k = 0
    while not layers[k] & square:
        k += 1
    d = flood(reach[i], i, layers[k], *walled)
    return d + k if d >= 0 else -1


def distance(start, goal_row, blocked):
    """Tính số bước ít nhất từ start tới goal_row.

    Đầu vào: start - ô xuất phát; goal_row - hàng đích; blocked - tập tường.
    Đầu ra: số bước, -1 nếu bị bít đường.
    """
    return bfs_dist(start, goal_row, *blocked)


def goal_layers(goal_row, blocked):
    """Loang ngược từ hàng đích để biết khoảng cách của mọi ô cùng lúc.

    Đầu vào: goal_row - hàng đích; blocked - tập tường.
    Đầu ra: list layers, layers[d] = bitboard các ô cách goal_row đúng d bước.
    """
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
    """Tra khoảng cách của một ô trong kết quả goal_layers.

    Đầu vào: layers - kết quả goal_layers; c - ô cần tra.
    Đầu ra: khoảng cách từ c tới đích, -1 nếu không tới được.
    """
    bit = 1 << c
    for d, layer in enumerate(layers):
        if layer & bit:
            return d
    return -1


def shortest_path(start, d, layers, blocked, max_steps=N):
    """Dựng một đường ngắn nhất: mỗi bước sang ô kề thuộc lớp gần đích hơn một bậc.

    Đầu vào: start - ô xuất phát (cách đích d bước); layers - kết quả goal_layers;
             blocked - tập tường; max_steps - số bước tối đa cần dựng.
    Đầu ra: list các ô trên đường đi, bắt đầu từ start.
    """
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
    """Liệt kê nước đi hợp lệ như Arena: đi 1 ô, nhảy qua đối thủ, đi chéo khi sau lưng
    đối thủ bị chặn.

    Đầu vào: me - ô của người đi; opp - ô của đối thủ; blocked - tập tường.
    Đầu ra: list các ô đến được (đã sắp xếp).
    """
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
    """Kiểm tra bức tường có nằm trong bàn và không chồng lấn / cắt ngang tường cũ.

    Đầu vào: (o, r, c) - bức tường cần đặt; hw, vw - tập tường ngang / dọc hiện có.
    Đầu ra: True nếu đặt được (chưa kiểm tra việc bít đường).
    """
    if not (0 <= r < SIZE - 1 and 0 <= c < SIZE - 1):
        return False
    if o == "H":
        return ((r, c - 1) not in hw and (r, c) not in hw and (r, c + 1) not in hw
                and (r, c) not in vw)
    return ((r - 1, c) not in vw and (r, c) not in vw and (r + 1, c) not in vw
            and (r, c) not in hw)


def walls_on_path(path):
    """Liệt kê các bức tường chặn ít nhất một bước của đường đi.

    Đầu vào: path - list ô của đường đi.
    Đầu ra: dict {(o, r, c): i}, i = bước đầu tiên bị tường đó chặn; tường gần đầu đường
            đứng trước.
    """
    out = {}
    for i in range(len(path) - 1):
        for w in EDGE_WALLS[path[i], path[i + 1]]:
            if w not in out:
                out[w] = i
    return out


def vulnerability(start, goal_row, blocked, d0, layers):
    """Tính "độ dễ bị chặn": một bức tường đặt trên VULN_STEPS bước đầu của đường ngắn nhất
    làm người đứng ở start phải đi thêm tối đa bao nhiêu bước.

    Đầu vào: start - ô của người chơi; goal_row - hàng đích; blocked - tập tường;
             d0 - khoảng cách hiện tại; layers - goal_layers(goal_row, blocked).
    Đầu ra: số bước đi thêm lớn nhất (0 nếu không tường nào làm chậm; bỏ qua tường bít đường).
    """
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
    """Tìm tường phòng thủ (chỉ dùng ở gốc): chiếm chỗ hoặc cắt ngang vị trí của những bức
    tường đối thủ có thể đặt để làm mình đi thêm >= 2 bước.

    Đầu vào: me - mình; pos, wl, dist, goals - trạng thái hai người chơi; hw, vw, blocked -
             tập tường.
    Đầu ra: list nút con dạng (điểm thứ tự, action, pos, hw, vw, blocked, wl, dist) cho các
            tường phòng thủ không làm mình chậm đi.
    """
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
def side_value(d_me, d_opp, walls_me):
    """Tính giá trị thế cờ của một bên bằng tổng 3 ô tra bảng dist, walls, lead.

    Đầu vào: d_me, d_opp - số bước tới đích của bên đó và của đối phương; walls_me - số tường
             bên đó còn.
    Đầu ra: giá trị (chưa tính độ dễ bị chặn).
    """
    w = WEIGHTS
    lead = d_opp - d_me
    if lead > LEAD_CAP:
        lead = LEAD_CAP
    elif lead < -LEAD_CAP:
        lead = -LEAD_CAP
    return (w["dist"][d_me if d_me < DIST_CAP else DIST_CAP]
            + w["walls"][walls_me][d_opp if d_opp < OPP_CAP else OPP_CAP]
            + w["lead"][walls_me][lead + LEAD_CAP])


def evaluate(p, dist, wl, vuln=(0, 0)):
    """Chấm điểm thế cờ nhìn từ người sắp đi.

    Đầu vào: p - người sắp đi; dist, wl - số bước tới đích và số tường còn của hai người;
             vuln - độ dễ bị chặn của hai người (0 nếu đối phương hết tường).
    Đầu ra: điểm (dương = có lợi cho p).
    """
    q = 1 - p
    dp, dq = dist[p], dist[q]
    w = WEIGHTS
    safe = 0
    if wl[q] == 0 and dp <= dq:
        safe = 1
    elif wl[p] == 0 and dq < dp:
        safe = -1
    return (w["tempo"] + w["safe_race"] * safe
            + side_value(dp, dq, wl[p]) - side_value(dq, dp, wl[q])
            + w["vuln"][wl[p]] * vuln[q] - w["vuln"][wl[q]] * vuln[p])


# ------------------------------------------------------- 5. Tìm kiếm
EXACT, LOWER, UPPER = 0, 1, 2    # loại giá trị trong bảng chuyển vị: đúng / chặn dưới / chặn trên


class Search:
    """Một lần tìm kiếm alpha-beta cho một nước đi, kèm các bảng nhớ dùng chung trong lần đó."""

    def __init__(self, goals, deadline):
        """Khởi tạo bảng chuyển vị, cache và bảng killer/history rỗng.

        Đầu vào: goals - hàng đích của hai người; deadline - thời điểm (perf_counter) phải dừng.
        """
        self.goals = goals
        self.deadline = deadline
        self.tt = {}        # bảng chuyển vị: thế cờ -> (độ sâu, giá trị, loại, nước tốt nhất)
        self.nodes = 0
        self.vcache = {}    # (ô, người chơi, tập tường) -> độ dễ bị chặn
        self.lcache = {}    # (đích, tập tường) -> goal_layers
        self.killers = {}   # tầng -> 2 nước gây cắt tỉa gần nhất
        self.history = {}   # (người chơi, nước) -> tổng depth^2 các lần gây cắt tỉa

    def layers_to(self, goal_row, blocked):
        """Lấy goal_layers có cache (rất nhiều nút trong cây chung một tập tường).

        Đầu vào: goal_row - hàng đích; blocked - tập tường.
        Đầu ra: kết quả goal_layers(goal_row, blocked).
        """
        key = (goal_row, blocked)
        layers = self.lcache.get(key)
        if layers is None:
            layers = self.lcache[key] = goal_layers(goal_row, blocked)
        return layers

    def vuln(self, i, pos, blocked, wl, dist):
        """Lấy độ dễ bị chặn của người chơi i, có cache theo ô + tập tường.

        Đầu vào: i - người chơi; pos, blocked, wl, dist - trạng thái thế cờ.
        Đầu ra: độ dễ bị chặn (0 nếu đối phương hết tường hoặc i đã về đích).
        """
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
        """Đánh giá nút lá: điểm = base + wq * vuln[q] - wp * vuln[p] (vuln, wq, wp >= 0).

        vuln đắt nhất (~10 lần BFS mỗi cái) nên tính lười: nếu base <= alpha thì chỉ vuln[q]
        mới kéo điểm lên được -> tính nó trước; vẫn <= alpha thì trả luôn (chặn trên của điểm
        thật, đủ để cắt tỉa) mà khỏi tính vuln[p]. Tương tự khi base >= beta.

        Đầu vào: p - người sắp đi; pos, blocked, wl, dist - trạng thái; alpha, beta - cửa sổ.
        Đầu ra: điểm nhìn từ p (có thể chỉ là chặn trên/dưới khi nằm ngoài cửa sổ).
        """
        base = evaluate(p, dist, wl)
        q = 1 - p
        wq, wp = WEIGHTS["vuln"][wl[p]], WEIGHTS["vuln"][wl[q]]   # hệ số của vuln[q], vuln[p]
        if base <= alpha:
            high = base + wq * self.vuln(q, pos, blocked, wl, dist)
            if high <= alpha:
                return high
            return high - wp * self.vuln(p, pos, blocked, wl, dist)
        if base >= beta:
            low = base - wp * self.vuln(p, pos, blocked, wl, dist)
            if low >= beta:
                return low
            return low + wq * self.vuln(q, pos, blocked, wl, dist)
        return (base + wq * self.vuln(q, pos, blocked, wl, dist)
                - wp * self.vuln(p, pos, blocked, wl, dist))

    def children(self, p, pos, hw, vw, blocked, wl, dist):
        """Sinh các nút con: mọi nước đi + tối đa WALL_CANDIDATES tường làm đối thủ chậm đi.

        Đầu vào: p - người sắp đi; pos, hw, vw, blocked, wl, dist - trạng thái.
        Đầu ra: (list nút con (điểm thứ tự, action, pos, hw, vw, blocked, wl, dist) đã xếp
                giảm dần, None), hoặc ("WIN", ô) nếu đi một bước là về đích.
        """
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
            # Chỉ tường chặn đường ngắn nhất của chính mình mới có thể làm mình chậm đi
            # -> các tường khác khỏi cần BFS cho mình.
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
        """Dựng đúng một nút con cho nước act (nước lấy từ bảng chuyển vị) mà không sinh cả
        danh sách nút con.

        Đầu vào: p - người sắp đi; pos, hw, vw, blocked, wl, dist - trạng thái; act - nước đi.
        Đầu ra: (pos, hw, vw, blocked, wl, dist) sau nước đi, hoặc None nếu nước không còn
                hợp lệ / không thuộc tập ứng viên.
        """
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
        """Tìm kiếm negamax alpha-beta có bảng chuyển vị, killer, history và PVS.

        Đầu vào: p - người sắp đi; pos, hw, vw, blocked, wl, dist - trạng thái; depth - số tầng
                 còn phải tìm; alpha, beta - cửa sổ; ply - tầng hiện tại tính từ gốc.
        Đầu ra: điểm của thế cờ nhìn từ p (ném Timeout nếu quá deadline).
        """
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
    """Đổi nước đi nội bộ sang định dạng hành động của Arena.

    Đầu vào: action - ("move", ô) hoặc ("wall", (o, r, c)).
    Đầu ra: dict {"type": "move", "to": [r, c]} hoặc {"type": "wall", ...}.
    """
    kind, data = action
    if kind == "move":
        return {"type": "move", "to": [ROW[data], COL[data]]}
    o, r, c = data
    return {"type": "wall", "orientation": o, "row": r, "col": c}


LAST_INFO = {}   # thông tin lần tìm kiếm gần nhất (độ sâu, số nút) để debug


def choose_action(observation):
    """Chọn nước đi cho lượt hiện tại (hàm Arena gọi).

    Đào sâu dần alpha-beta tới khi hết TIME_BUDGET rồi trả nước tốt nhất của độ sâu đã xong;
    gặp lỗi bất ngờ thì trả nước đi gần đích nhất để không mất lượt.

    Đầu vào: observation - dict trạng thái ván do Arena gửi.
    Đầu ra: dict hành động: move, wall hoặc pass.
    """
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
        kids = [k for k in kids
                if (k[1][1] in legal_m if k[1][0] == "move" else k[1][1] in legal_w)]
        # Hết tường thì chỉ còn đua về đích: lùi hay đi ngang chỉ mất nhịp (không chặn thì
        # bot có thể tiến một bước lùi một bước hàng chục lượt).
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
                        # Điểm gốc = giá trị tìm kiếm + thưởng tiến bộ nhỏ: khi mọi nước hòa
                        # điểm thì chọn nước tiến gần đích, không đi qua đi lại.
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
                    # Hết giờ giữa chừng: kids[0] là nước tốt nhất của độ sâu trước, xét đầu
                    # tiên; nước nào đã vượt nó ở độ sâu mới thì tốt hơn thật -> dùng luôn.
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
