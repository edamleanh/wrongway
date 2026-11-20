"""Agent Wrongway: tìm kiếm alpha-beta + hàm đánh giá có trọng số học được.

Đổi tên file thành <MSSV>.py trước khi nộp.

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
3. Hàm đánh giá = tổng các ô TRA BẢNG (đặc trưng thưa): giá trị của "còn cách
   đích d bước", của "còn w tường khi đối thủ cách đích d bước", của "dẫn l bước
   khi còn w tường" (dẫn càng xa thì mỗi bức tường còn giữ càng quý, nên bot
   không xài hết tường khi đã thắng chắc cuộc đua), và hệ số của "độ dễ bị chặn"
   (vuln: một bức tường của đối thủ làm mình đi thêm tối đa bao nhiêu bước) theo
   số tường đối thủ còn.
   Hàm đánh giá ĐỐI XỨNG: đổi vai hai người chơi thì điểm đổi dấu (chỉ cộng thêm
   hằng số "tempo" cho người sắp đi). Nhờ vậy một thế cờ được chấm như nhau dù
   tìm tới độ sâu chẵn hay lẻ; bản cũ lệch ~9 điểm giữa hai trường hợp nên bot
   đổi phong cách theo độ sâu đạt được ở mỗi lượt.
   Các ô trong bảng được học bằng HỌC TĂNG CƯỜNG, thuật toán TD-leaf(lambda)
   (training/td_learn.py): bot tự đấu với chính nó; sau mỗi ván, mỗi ô được chỉnh
   một chút để dự đoán thắng/thua ở thế cờ trước khớp hơn với dự đoán ở thế cờ
   sau và với kết quả ván. Bảng khởi đầu = công thức tuyến tính cũ (trọng số tìm
   bằng thuật toán tiến hóa, training/evolve.py). Bảng đã học được ghi thẳng vào
   file này (file nộp không được đọc file ngoài).

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
DIST_CAP = 40            # bảng dist: khoảng cách lớn nhất có ô riêng
OPP_CAP = 20             # bảng walls: khoảng cách đối thủ lớn nhất có cột riêng
LEAD_CAP = 6             # bảng lead: mức dẫn/thua lớn nhất có cột riêng (-LEAD_CAP..LEAD_CAP)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))   # lên, xuống, trái, phải (cùng thứ tự với 4 bitboard)

# Hàm đánh giá = tổng các ô TRA BẢNG (đặc trưng thưa), nhìn từ người sắp đi:
#   điểm = tempo + safe_race * safe + giá_trị(mình) - giá_trị(đối thủ)
#          + vuln[tường mình] * vuln_đối_thủ - vuln[tường đối thủ] * vuln_mình
#   giá_trị(X) = dist[d_X] + walls[tường_X][d_Y] + lead[tường_X][d_Y - d_X]     (Y = đối phương của X)
#   dist[d]       : giá trị của việc còn cách đích d bước (càng xa càng âm)
#   walls[w][d]   : giá trị của việc còn w tường khi đối phương còn cách đích d bước
#   lead[w][l]    : thưởng/phạt theo mức dẫn l (-6..6 bước) khi còn w tường (giữ tường khi dẫn xa)
#   vuln[w]       : mỗi bước "dễ bị chặn" của đối phương đáng bao nhiêu khi mình còn w tường
#   safe_race     : +1 nếu đối thủ hết tường và mình về trước, -1 nếu ngược lại
#   tempo         : hằng số cộng cho người sắp đi (được đi trước)
# Đổi vai hai người chơi thì điểm đổi dấu (trừ tempo) -> chấm như nhau ở độ sâu chẵn và lẻ.
# Các ô trong bảng được HỌC bằng TD-leaf(lambda) qua tự đấu (training/td_learn.py): sau mỗi ván,
# mỗi ô được chỉnh để dự đoán của thế cờ trước khớp hơn với dự đoán của thế cờ sau / kết quả ván.
# ======================= WEIGHTS BEGIN (tự sinh bởi training/td_learn.py) =======================
WEIGHTS = {
    "tempo": 1.0091,
    "safe_race": 6.8991,
    "dist": [0.0, -0.5374, -2.5276, -4.2, -5.6729, -7.0722, -8.4326, -9.7688, -11.1486, -12.4701, -13.8578, -15.2127, -16.5472, -17.8974, -19.2694, -20.5943, -21.932, -23.3368, -24.7457, -26.1292, -27.5081, -28.8939, -30.309, -31.6937, -33.1009, -34.508, -35.9169, -37.2822, -38.6638, -40.0111, -41.3883, -42.7534, -44.176, -45.5556, -46.937, -48.3175, -49.698, -51.0785, -52.459, -53.8395, -55.22],
    "walls": [
        [0.0, -0.0149, 0.1793, 0.3276, 0.3757, 0.3182, 0.2471, 0.1686, 0.1139, 0.0597, 0.0194, -0.0277, -0.0811, -0.1325, -0.1493, -0.1625, -0.1985, -0.2299, -0.1814, -0.183, -0.2347],
        [1.9854, 0.9245, 1.5145, 1.9643, 2.0901, 2.1458, 2.1321, 2.0744, 2.0449, 2.0644, 2.0243, 1.9819, 2.0068, 1.9789, 1.9332, 1.9345, 1.9098, 1.8954, 1.9574, 2.0028, 1.9926],
        [3.9708, 3.5652, 3.8824, 4.0679, 4.1321, 4.2164, 4.1957, 4.1558, 4.131, 4.1682, 4.1933, 4.1059, 4.1158, 4.0566, 3.9741, 4.0665, 4.0432, 4.0653, 4.0804, 4.0936, 4.1322],
        [5.9562, 5.6644, 5.9273, 6.0635, 6.166, 6.2806, 6.288, 6.2675, 6.2201, 6.2816, 6.2088, 6.1855, 6.1658, 6.154, 6.0996, 6.1523, 6.1692, 6.1745, 6.1735, 6.1869, 6.2112],
        [7.9416, 7.7692, 7.9901, 8.169, 8.2284, 8.2906, 8.2796, 8.3526, 8.3341, 8.2597, 8.2221, 8.2365, 8.2423, 8.3243, 8.1917, 8.1824, 8.237, 8.2676, 8.2164, 8.234, 8.1911],
        [9.927, 9.8915, 10.1054, 10.2361, 10.2824, 10.3415, 10.3684, 10.4589, 10.3352, 10.3748, 10.4311, 10.3245, 10.2935, 10.2871, 10.3282, 10.2686, 10.2865, 10.2584, 10.3512, 10.3062, 10.3053],
        [11.9124, 11.878, 12.111, 12.1935, 12.3083, 12.4915, 12.4907, 12.6068, 12.4176, 12.3146, 12.2796, 12.5026, 12.3992, 12.3218, 12.7029, 12.3776, 12.1499, 12.4258, 12.3528, 12.3613, 12.3633],
        [13.8978, 14.0125, 14.1986, 14.3032, 14.4625, 14.6205, 14.6023, 14.4452, 14.5177, 14.3759, 14.5351, 14.4696, 14.4012, 14.3609, 14.4026, 14.4088, 14.4146, 14.4199, 14.421, 14.4225, 14.4238],
        [15.8832, 16.042, 16.267, 16.4033, 16.5192, 16.6243, 16.3601, 16.3268, 16.6726, 16.4901, 16.4247, 16.4516, 16.4463, 16.4784, 16.4759, 16.4861, 16.4772, 16.4796, 16.4812, 16.4828, 16.4843],
        [17.8686, 18.1989, 18.3783, 18.5075, 18.4455, 18.543, 18.529, 18.3228, 18.4352, 18.4435, 18.5224, 18.5181, 18.5236, 18.528, 18.5314, 18.5343, 18.5369, 18.5392, 18.5413, 18.5432, 18.5449],
        [19.854, 20.255, 20.46, 20.4799, 20.5697, 20.2863, 20.3643, 20.4829, 20.2915, 20.4775, 20.5226, 20.5772, 20.5823, 20.5866, 20.5904, 20.5937, 20.5966, 20.5992, 20.6015, 20.6035, 20.6054],
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
def side_value(d_me, d_opp, walls_me):
    """Giá trị thế cờ của một bên = tổng 3 ô tra bảng (chưa tính vuln)."""
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
    return (w["tempo"] + w["safe_race"] * safe
            + side_value(dp, dq, wl[p]) - side_value(dq, dp, wl[q])
            + w["vuln"][wl[p]] * vuln[q] - w["vuln"][wl[q]] * vuln[p])


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
        """Đánh giá lá: điểm = base + wq * vuln[q] - wp * vuln[p], với vuln và hai hệ số đều >= 0.
        vuln là phần đắt nhất (mỗi cái ~10 lần BFS) nên tính LƯỜI: nếu base đã <= alpha thì
        chỉ vuln[q] mới cứu được -> tính nó trước, vẫn <= alpha thì trả luôn (đó là chặn trên
        của điểm thật, đủ để alpha-beta cắt tỉa) khỏi tính vuln[p]. Tương tự khi base >= beta."""
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
        return base + wq * self.vuln(q, pos, blocked, wl, dist) - wp * self.vuln(p, pos, blocked, wl, dist)

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
