#!/usr/bin/env python3
"""So sức mạnh: một bot đấu với nhiều đối thủ, mỗi cặp ván cùng seed đổi bên,
mở đầu bằng vài nước ngẫu nhiên để các ván khác nhau (bot tất định).

    python phat-training/gauntlet.py dev MSSV wall_builder greedy --pairs 10
    python phat-training/gauntlet.py dev MSSV --pairs 20 --budget 0.55

Điểm chấm như mùa giải (tools/tournament.py): thắng 3, hoà 1, thua 0 — tức một ván hoà
đáng 1/3 trận thắng (`common.DRAW_PT`), không phải 0.5.

Đối số đầu là bot cần đo; các đối số sau là đối thủ (tên trong submissions/
hoặc đường dẫn .py). Mặc định bot dùng TIME_BUDGET/MAX_DEPTH của chính file đó;
--budget/--opp-budget ghi đè để đo ở tốc độ máy chấm (ví dụ --budget 0.275 cho
máy chậm gấp đôi, cả hai bên cùng ngân sách).
"""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import evolve
from common import SUBMISSIONS, rate


def resolve(name):
    p = Path(name)
    if p.suffix == ".py" and p.exists():
        return str(p.resolve())
    return str((SUBMISSIONS / (name + ".py")).resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bot")
    ap.add_argument("opponents", nargs="+")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--opening", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--budget", type=float, default=None,
                    help="giây/nước cho bot cần đo (mặc định: giữ TIME_BUDGET trong file)")
    ap.add_argument("--opp-budget", type=float, default=None,
                    help="giây/nước cho đối thủ dạng file (mặc định: bằng --budget, "
                         "vì máy chấm chậm hơn thì cả hai bên cùng chậm)")
    a = ap.parse_args()

    bot = resolve(a.bot)
    opp_budget = a.opp_budget if a.opp_budget is not None else a.budget
    # depth = None -> giữ MAX_DEPTH trong file. Ngân sách đặt thẳng vào module của từng bên
    # (side_budget), vì tham số budget của evolve.play chỉ áp cho bot dạng dict trọng số.
    side_budget = None if a.budget is None and opp_budget is None else (a.budget, opp_budget)
    cfg = (a.opening, None, None, a.max_turns, None, side_budget)
    batches = {opp: evolve.match_jobs(bot, resolve(opp), a.pairs, a.seed, cfg)
               for opp in a.opponents}
    t0 = time.time()
    with Pool(a.workers) as pool:
        flat = [(opp, job, side) for opp, items in batches.items() for job, side in items]
        winners = pool.map(evolve.play, [j for _, j, _ in flat], chunksize=1)
    res = {opp: [0, 0, 0] for opp in a.opponents}           # thắng, hòa, thua
    for (opp, _, side), w in zip(flat, winners):
        res[opp][0 if w == side else 1 if w is None else 2] += 1
    print("{} ({} cặp ván/đối thủ, mở đầu {} nước ngẫu nhiên)".format(Path(bot).stem, a.pairs, a.opening))
    if side_budget is not None:
        print("  ngân sách/nước: bot {}  đối thủ {}".format(
            a.budget if a.budget is not None else "theo file",
            opp_budget if opp_budget is not None else "theo file"))
    for opp, (w, d, l) in res.items():
        print("  vs {:<14} W {:>3}  D {:>3}  L {:>3}  điểm {:.2f}".format(
            opp, w, d, l, rate(w, d, l)))
    print("({:.0f}s)".format(time.time() - t0))


if __name__ == "__main__":
    main()
