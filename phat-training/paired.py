#!/usr/bin/env python3
"""So hai bot theo TỪNG CẶP VÁN (cùng seed, đổi bên) thay vì chỉ so tổng.

    python phat-training/paired.py Phat_buffed_3 Phat_buffed_2 OPP1 OPP2 --pairs 20 --budget 0.275

Vì sao cần: `gauntlet.py` chỉ in tổng W-D-L, nên chênh lệch giữa hai bản bị phương sai do nước mở
đầu ngẫu nhiên nuốt mất. Ở đây mỗi (đối thủ, seed) chạy 2 ván cho bot A và 2 ván cho bot B với
CÙNG seed (cùng nước mở đầu, đổi bên) -> mỗi cặp là một đơn vị đo điểm, rồi so hiệu điểm theo cặp
(có sai số chuẩn) và đếm số cặp A hơn / bằng / kém.

Chấm điểm giống mùa giải (`tools/tournament.py`): **thắng 3, hoà 1, thua 0**, tức mỗi ván được
1 / 1/3 / 0 điểm và mỗi cặp (2 ván) là trung bình của hai ván. Hoà KHÔNG phải 0.5 — vì hoà chỉ
đáng 1/3 trận thắng, nên một bản chơi hoà nhiều bị chấm thấp hơn cách đếm 0.5.

--budget đặt cho CẢ HAI bên (mô phỏng máy chấm chậm hơn, như #51/#54).
"""

import argparse
import math
import time
from multiprocessing import Pool
from pathlib import Path

import evolve
from common import SUBMISSIONS, game_pt, rate


def resolve(name):
    p = Path(name)
    if p.suffix == ".py" and p.exists():
        return str(p.resolve())
    return str((SUBMISSIONS / (name + ".py")).resolve())


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / max(1, n - 1)
    return n, m, math.sqrt(var / n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bot_a")
    ap.add_argument("bot_b")
    ap.add_argument("opponents", nargs="+")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--opening", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=500,
                    help="số lượt tối đa, hết lượt = hoà (arena để 500)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--budget", type=float, default=None,
                    help="giây/nước cho cả hai bên (mặc định: giữ TIME_BUDGET trong file)")
    a = ap.parse_args()

    la = "A:" + Path(a.bot_a).stem
    lb = "B:" + Path(a.bot_b).stem
    bots = [(la, resolve(a.bot_a)), (lb, resolve(a.bot_b))]
    opps = [(Path(o).stem, resolve(o)) for o in a.opponents]
    side_budget = None if a.budget is None else (a.budget, a.budget)
    cfg = (a.opening, None, None, a.max_turns, None, side_budget)

    jobs, meta = [], []
    for label, bot in bots:
        for oname, opp in opps:
            for job, side in evolve.match_jobs(bot, opp, a.pairs, a.seed, cfg):
                jobs.append(job)
                meta.append((label, oname, job[2], side))

    t0 = time.time()
    with Pool(a.workers) as pool:
        winners = pool.map(evolve.play, jobs, chunksize=1)

    pts = {}
    wdl = {}
    for (label, oname, seed, side), w in zip(meta, winners):
        pts[(label, oname, seed, side)] = game_pt(w, side)
        key = (label, oname)
        wdl.setdefault(key, [0, 0, 0])
        wdl[key][0 if w == side else 1 if w is None else 2] += 1

    print("{} vs {}  ({} cặp/ô, mở đầu {} nước, ngân sách {})".format(
        la, lb, a.pairs, a.opening, a.budget if a.budget is not None else "theo file"))
    print("\n{:<26} {:<24} {:>3} {:>3} {:>3}  {:>6}".format("bot", "đối thủ", "W", "D", "L", "điểm"))
    for label, oname in [(x, o) for x, _ in bots for o, _ in opps]:
        w, d, l = wdl[(label, oname)]
        print("{:<26} {:<24} {:>3} {:>3} {:>3}  {:>6.3f}".format(
            label, oname, w, d, l, rate(w, d, l)))

    def pair_score(label, oname, seed):
        return (pts[(label, oname, seed, 0)] + pts[(label, oname, seed, 1)]) / 2.0

    print("\nso từng cặp ván (A − B), mỗi cặp = 2 ván cùng seed đổi bên:")
    pooled = []
    for oname, _ in opps:
        ds = [pair_score(la, oname, s) - pair_score(lb, oname, s)
              for s in range(a.seed, a.seed + a.pairs)]
        pooled += ds
        n, m, se = mean_se(ds)
        win = sum(1 for x in ds if x > 0)
        tie = sum(1 for x in ds if x == 0)
        loss = sum(1 for x in ds if x < 0)
        print("  {:<24} n {:>3}  hiệu {:>+6.3f} ± {:.3f}   hơn {:<3} bằng {:<3} kém {:<3}".format(
            oname, n, m, se, win, tie, loss))
    n, m, se = mean_se(pooled)
    win = sum(1 for x in pooled if x > 0)
    tie = sum(1 for x in pooled if x == 0)
    loss = sum(1 for x in pooled if x < 0)
    print("  {:<24} n {:>3}  hiệu {:>+6.3f} ± {:.3f}   hơn {:<3} bằng {:<3} kém {:<3}".format(
        "TỔNG", n, m, se, win, tie, loss))
    if se > 0:
        print("  (hiệu/sai số = {:+.2f}σ)".format(m / se))
    print("({:.0f}s)".format(time.time() - t0))


if __name__ == "__main__":
    main()
