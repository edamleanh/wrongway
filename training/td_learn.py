#!/usr/bin/env python3
"""Học các bảng của hàm đánh giá bằng TD-leaf(lambda) qua tự đấu (học tăng cường).

Ý tưởng (Temporal-Difference learning):
- Hàm đánh giá E(s) của bot là tổng các ô tra bảng (xem khối WEIGHTS trong file bot).
  Đổi sang "xác suất người chơi 0 thắng": v(s) = sigmoid(±E(s) / K).
- Bot tự đấu với chính nó. Ở mỗi lượt t, tìm kiếm alpha-beta độ sâu D (chính negamax của bot)
  cho ra nước đi, điểm gốc và THẾ CỜ LÁ của biến chính (principal variation, lấy từ bảng chuyển
  vị); v_t = v(lá đó) = v(điểm gốc) là dự đoán của bot ở lượt t.
- Dự đoán tốt thì phải nhất quán theo thời gian: v_t ~ v_{t+1} ~ ... ~ kết quả ván z.
  Sai số TD: d_t = v_{t+1} - v_t (lượt cuối: z - v_t). Mỗi ô bảng có mặt ở lá của lượt t được
  chỉnh theo tổng sai số về sau có giảm dần: e_t = d_t + lambda * d_{t+1} + lambda^2 * d_{t+2}...
  (lambda = 1: học thẳng từ kết quả ván; lambda = 0: chỉ học từ lượt kế tiếp.)
- Không có nhãn do người gán và không có đối thủ mẫu: tín hiệu duy nhất là thắng/thua của tự đấu.

Khác tiến hóa (evolve.py): tiến hóa thử ngẫu nhiên cả bộ trọng số rồi giữ bộ thắng; TD chỉnh
TỪNG ô theo hướng giảm sai số dự đoán, nên học được hàng trăm tham số.

    python -u training/td_learn.py --bot wrongway-student/wrongway/submissions/dev.py \
        --iterations 30 --games 240 --workers 10 > logs/td.log 2>&1
"""

import argparse
import json
import math
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from common import BOT_FILE, CKPT_DIR, load_module, write_weights

INF = float("inf")
_bot = None          # module bot trong tiến trình con


def _init(bot_path):
    global _bot
    _bot = load_module(bot_path, "td_bot")


# ------------------------------------------------------------------ bố trí tham số
class Layout:
    """Vị trí của từng bảng trong vector tham số phẳng."""

    def __init__(self, m):
        self.dc, self.oc, self.lc = m.DIST_CAP, m.OPP_CAP, m.LEAD_CAP
        self.n_lead = 2 * self.lc + 1
        self.TEMPO, self.SAFE, self.DIST = 0, 1, 2
        self.WALLS = self.DIST + self.dc + 1
        self.LEAD = self.WALLS + 11 * (self.oc + 1)
        self.VULN = self.LEAD + 11 * self.n_lead
        self.size = self.VULN + 11

    def to_vector(self, w):
        return np.array([w["tempo"], w["safe_race"]] + list(w["dist"])
                        + [x for row in w["walls"] for x in row]
                        + [x for row in w["lead"] for x in row] + list(w["vuln"]), dtype=float)

    def to_dict(self, v):
        v = [float(x) for x in v]
        oc1 = self.oc + 1
        return {"tempo": v[self.TEMPO], "safe_race": v[self.SAFE],
                "dist": v[self.DIST:self.WALLS],
                "walls": [v[self.WALLS + w * oc1:self.WALLS + (w + 1) * oc1] for w in range(11)],
                "lead": [v[self.LEAD + w * self.n_lead:self.LEAD + (w + 1) * self.n_lead] for w in range(11)],
                "vuln": v[self.VULN:self.VULN + 11]}

    def features(self, p, dist, wl, vuln):
        """Các ô bảng có mặt trong E(thế cờ, người sắp đi p): dict {chỉ số: hệ số}.
        Phải khớp evaluate() của bot (có kiểm tra ở selftest)."""
        f = {self.TEMPO: 1.0}
        q = 1 - p
        if wl[q] == 0 and dist[p] <= dist[q]:
            f[self.SAFE] = 1.0
        elif wl[p] == 0 and dist[q] < dist[p]:
            f[self.SAFE] = -1.0
        for x, y, sign in ((p, q, 1.0), (q, p, -1.0)):
            dx, dy = dist[x], dist[y]
            lead = max(-self.lc, min(self.lc, dy - dx))
            keys = [(self.DIST + min(dx, self.dc), sign),
                    (self.WALLS + wl[x] * (self.oc + 1) + min(dy, self.oc), sign),
                    (self.LEAD + wl[x] * self.n_lead + lead + self.lc, sign)]
            if vuln[y]:
                keys.append((self.VULN + wl[x], sign * vuln[y]))
            for i, c in keys:
                f[i] = f.get(i, 0.0) + c
        return {i: c for i, c in f.items() if c}


def selftest(m, lay, w):
    rng = random.Random(1)
    vec = lay.to_vector(w)
    for _ in range(5000):
        dist = [rng.randint(1, 45), rng.randint(1, 45)]
        wl = [rng.randint(0, 10), rng.randint(0, 10)]
        vu = (rng.randint(0, 7) if wl[1] else 0, rng.randint(0, 7) if wl[0] else 0)
        p = rng.randint(0, 1)
        e = sum(vec[i] * c for i, c in lay.features(p, dist, wl, vu).items())
        assert abs(e - m.evaluate(p, dist, wl, vu)) < 1e-9, (dist, wl, vu, e, m.evaluate(p, dist, wl, vu))


# ------------------------------------------------------------------ tự đấu (tiến trình con)
def pv_leaf(S, p, pos, hw, vw, blocked, wl, dist, depth):
    """Lá của biến chính: từ một nút đã tìm xong, đi theo nước tốt nhất lưu trong bảng chuyển vị
    đủ depth nước. Trả (người sắp đi, pos, blocked, wl, dist) hoặc None nếu đứt giữa chừng."""
    for _ in range(depth):
        entry = S.tt.get((p, pos[0], pos[1], wl[0], wl[1], blocked))
        if entry is None or entry[3] is None:
            return None
        child = S.make_child(p, pos, hw, vw, blocked, wl, dist, entry[3])
        if child is None:
            return None
        pos, hw, vw, blocked, wl, dist = child
        p = 1 - p
    return p, pos, blocked, wl, dist


def start_state(rng, opening):
    m = _bot
    pos = [m.cell(8, 4), m.cell(0, 4)]
    goals = [0, 8]
    hw = vw = frozenset()
    blocked = m.blocked_from(hw, vw)
    p = 0
    for _ in range(opening):                      # mở đầu ngẫu nhiên, chỉ nước đi
        pos[p] = rng.choice(m.legal_moves(pos[p], pos[1 - p], blocked))
        p = 1 - p
    dist = [m.distance(pos[i], goals[i], blocked) for i in (0, 1)]
    return p, pos, goals, hw, vw, blocked, [10, 10], dist


def choose(goals, state, depth):
    """Tìm kiếm ở gốc bằng CHÍNH negamax của bot (đào sâu dần, bảng chuyển vị, PVS) tới đúng độ sâu
    depth -> thế cờ lá giống lúc thi đấu thật. Trả (nút con tốt nhất, điểm nhìn từ người sắp đi,
    lá biến chính hoặc None, danh sách nút con, Search); nút con = 'WIN' nếu thắng ngay."""
    p, pos, hw, vw, blocked, wl, dist = state
    S = _bot.Search(goals, INF)                   # Search mới cho mỗi nước, như bot thật
    kids, _ = S.children(p, pos, hw, vw, blocked, wl, dist)
    if kids == "WIN":
        return "WIN", None, None, None, S
    if not kids:
        return None, None, None, kids, S
    order = list(kids)
    alpha, best = -INF, None
    for d in range(1, depth + 1):
        alpha, best = -INF, None
        for kid in order:
            _, _, npos, nh, nv, nb, nwl, nd = kid
            v = -S.negamax(1 - p, npos, nh, nv, nb, nwl, nd, d - 1, -INF, -alpha)
            if v > alpha:
                alpha, best = v, kid
        order.remove(best)
        order.insert(0, best)                     # nước tốt nhất được xét trước ở độ sâu kế tiếp
    _, _, npos, nh, nv, nb, nwl, nd = best
    leaf = None
    if abs(alpha) < _bot.WIN / 2:
        leaf = pv_leaf(S, 1 - p, npos, nh, nv, nb, nwl, nd, depth - 1)
    return best, alpha, leaf, kids, S


def play_game(job):
    """Một ván tự đấu. Trả (bản ghi từng lượt, z) với z = 1 / 0 / 0.5 (người chơi 0 thắng/thua/hòa).
    Bản ghi: (sgn, điểm gốc, đặc trưng lá hoặc None, thăm_dò); sgn = +1 nếu người sắp đi ở gốc là
    người chơi 0 -> v = sigmoid(sgn * điểm / K). Đặc trưng lá đã đổi dấu về phía người chơi 0, và
    chỉ được ghi khi điểm của lá đúng bằng điểm gốc (lá đúng là lá của biến chính)."""
    weights, seed, depth, eps, opening, max_plies = job
    m = _bot
    m.WEIGHTS = weights
    lay = Layout(m)
    rng = random.Random(seed)
    p, pos, goals, hw, vw, blocked, wl, dist = start_state(rng, opening)
    records, z = [], 0.5
    for _ in range(max_plies):
        kid, value, leaf, kids, S = choose(goals, (p, pos, hw, vw, blocked, wl, dist), depth)
        if kid == "WIN":
            z = 1.0 if p == 0 else 0.0
            break
        if kid is None:                           # không có nước nào (không xảy ra trong thực tế)
            break
        explore = len(kids) > 1 and rng.random() < eps
        if explore:                               # thăm dò: một nước khác trong nhóm đầu theo thứ tự tĩnh
            kid = rng.choice([k for k in kids[:4] if k is not kid])
        feats = None
        if leaf is not None:
            lp, lpos, lblocked, lwl, ldist = leaf
            vu = (S.vuln(0, lpos, lblocked, lwl, ldist), S.vuln(1, lpos, lblocked, lwl, ldist))
            e = m.evaluate(lp, ldist, lwl, vu)
            if abs((e if lp == p else -e) - value) < 1e-6:
                side = 1.0 if lp == 0 else -1.0
                feats = [(i, c * side) for i, c in lay.features(lp, ldist, lwl, vu).items()]
        records.append((1.0 if p == 0 else -1.0, value, feats, explore))
        _, _, pos, hw, vw, blocked, wl, dist = kid
        p = 1 - p
    return records, z


def play_match(job):
    """Ván đối kháng giữa hai bộ bảng ở độ sâu cố định (để theo dõi tiến bộ). Trả người thắng/None."""
    w0, w1, seed, depth, opening, max_plies = job
    m = _bot
    rng = random.Random(seed)
    p, pos, goals, hw, vw, blocked, wl, dist = start_state(rng, opening)
    for _ in range(max_plies):
        m.WEIGHTS = w0 if p == 0 else w1
        kid = choose(goals, (p, pos, hw, vw, blocked, wl, dist), depth)[0]
        if kid == "WIN":
            return p
        if kid is None:
            return None
        _, _, pos, hw, vw, blocked, wl, dist = kid
        p = 1 - p
    return None


# ------------------------------------------------------------------ cập nhật TD (tiến trình chính)
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def values_of(records, z, K):
    return [sigmoid(r[0] * r[1] / K) for r in records] + [z]


def fit_k(games):
    """Chọn K để sigmoid(E / K) khớp nhất với kết quả ván (log-loss nhỏ nhất)."""
    best = None
    for K in [x / 4.0 for x in range(4, 161)]:
        loss = n = 0
        for records, z in games:
            for v in values_of(records, z, K)[:-1]:
                v = min(1 - 1e-6, max(1e-6, v))
                loss -= z * math.log(v) + (1 - z) * math.log(1 - v)
                n += 1
        if best is None or loss / n < best[0]:
            best = (loss / n, K)
    return best[1], best[0]


def td_update(games, K, lam, size):
    """Trả (tử số, mẫu số, thống kê): bước chỉnh của ô i = eta * K * tử[i] / (mẫu[i] + shrink)."""
    num, den = np.zeros(size), np.zeros(size)
    sq = cnt = used = 0
    logloss = 0.0
    for records, z in games:
        v = values_of(records, z, K)
        e = 0.0
        for t in range(len(records) - 1, -1, -1):
            _, _, feats, explore = records[t]
            d = v[t + 1] - v[t]
            e = 0.0 if explore else d + lam * e    # sau nước thăm dò thì không học ngược về trước
            sq += d * d
            cnt += 1
            vt = min(1 - 1e-6, max(1e-6, v[t]))
            logloss -= z * math.log(vt) + (1 - z) * math.log(1 - vt)
            if feats is None or e == 0.0:
                continue
            used += 1
            g = e * 4.0 * v[t] * (1.0 - v[t])     # đạo hàm sigmoid (chuẩn hóa: = 1 khi v = 0.5)
            for i, c in feats:
                num[i] += g * c
                den[i] += abs(c)
    return num, den, {"td_rms": math.sqrt(sq / max(1, cnt)), "logloss": logloss / max(1, cnt), "n": cnt,
                      "used": used / max(1, cnt)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", default=str(BOT_FILE), help="file bot có hàm đánh giá dạng bảng")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--games", type=int, default=240, help="số ván tự đấu mỗi vòng")
    ap.add_argument("--depth", default="5,6", help="độ sâu tìm kiếm khi tự đấu (xen kẽ theo ván); nên gần "
                    "độ sâu lúc thi đấu: học ở độ sâu 2-3 thắng ở độ sâu 3-4 nhưng thua ở độ sâu 6-7")
    ap.add_argument("--lam", type=float, default=0.7, help="lambda của TD(lambda)")
    ap.add_argument("--eta", type=float, default=0.2, help="cỡ bước")
    ap.add_argument("--shrink", type=float, default=100.0, help="ô ít xuất hiện thì chỉnh ít")
    ap.add_argument("--eps", type=float, default=0.05, help="xác suất đi nước thăm dò")
    ap.add_argument("--opening", type=int, default=4, help="số nước mở đầu ngẫu nhiên tối đa")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--k", type=float, default=None, help="thang sigmoid (mặc định: tự khớp ở vòng đầu)")
    ap.add_argument("--eval-every", type=int, default=5, help="mỗi N vòng đấu thử với bảng ban đầu")
    ap.add_argument("--eval-pairs", type=int, default=40)
    ap.add_argument("--eval-depth", default="5,6")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="td")
    a = ap.parse_args()

    bot_file = Path(a.bot).resolve()
    m = load_module(bot_file, "td_bot_main")
    lay = Layout(m)
    selftest(m, lay, m.WEIGHTS)
    vec = lay.to_vector(m.WEIGHTS)
    init = vec.copy()
    depths = [int(d) for d in a.depth.split(",")]
    eval_depths = [int(d) for d in a.eval_depth.split(",")]
    out_dir = CKPT_DIR / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    K = a.k
    print("Bot: %s | %d tham số | độ sâu %s | lambda %.2f | eta %.2f" % (bot_file.name, lay.size, depths, a.lam, a.eta),
          flush=True)

    with Pool(a.workers, initializer=_init, initargs=(str(bot_file),)) as pool:
        for it in range(a.iterations):
            t0 = time.time()
            w = lay.to_dict(vec)
            base = rng.randrange(10 ** 9)
            jobs = [(w, base + g, depths[g % len(depths)], a.eps, rng.randint(0, a.opening), a.max_plies)
                    for g in range(a.games)]
            games = pool.map(play_game, jobs, chunksize=1)
            if K is None:
                K, ll = fit_k(games)
                print("Thang sigmoid K = %.2f (log-loss %.4f)" % (K, ll), flush=True)
            num, den, st = td_update(games, K, a.lam, lay.size)
            step = a.eta * K * num / (den + a.shrink)
            vec = vec + step
            vec[lay.VULN:lay.VULN + 11] = np.maximum(vec[lay.VULN:lay.VULN + 11], 0.0)   # hệ số vuln phải >= 0
            zs = [z for _, z in games]
            line = ("Vòng %2d: %d ván, %.0f lượt/ván, P0 thắng %.2f, hòa %.2f | TD rms %.4f | log-loss %.4f "
                    "| lá dùng được %.2f | bước lớn nhất %.3f | lệch so với đầu %.2f" % (
                        it, len(games), st["n"] / len(games), sum(z == 1.0 for z in zs) / len(zs),
                        sum(z == 0.5 for z in zs) / len(zs), st["td_rms"], st["logloss"], st["used"],
                        float(np.max(np.abs(step))), float(np.max(np.abs(vec - init)))))
            if a.eval_every and (it + 1) % a.eval_every == 0:
                cur, old = lay.to_dict(vec), lay.to_dict(init)
                ej = []
                for g in range(a.eval_pairs):
                    d = eval_depths[g % len(eval_depths)]
                    ej.append((cur, old, 777000 + g, d, 2, a.max_plies))      # bảng mới cầm quân 0
                    ej.append((old, cur, 777000 + g, d, 2, a.max_plies))      # bảng mới cầm quân 1
                res = pool.map(play_match, ej, chunksize=1)
                pts = sum((0.5 if r is None else float(r == i % 2)) for i, r in enumerate(res))
                line += " | vs bảng đầu %.3f" % (pts / len(res))
            print(line + " | %.0fs" % (time.time() - t0), flush=True)
            (out_dir / ("iter_%03d.json" % it)).write_text(json.dumps(lay.to_dict(vec)), encoding="utf-8")
            bot_file.write_text(write_weights(bot_file.read_text(encoding="utf-8"), lay.to_dict(vec)),
                                encoding="utf-8")


if __name__ == "__main__":
    main()
