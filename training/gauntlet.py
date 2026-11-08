#!/usr/bin/env python3
"""So sức mạnh: một bot đấu với nhiều đối thủ, mỗi cặp ván cùng seed đổi bên,
mở đầu bằng vài nước ngẫu nhiên để các ván khác nhau (bot tất định).

    python training/gauntlet.py dev MSSV wall_builder greedy --pairs 10
    python training/gauntlet.py dev MSSV --pairs 20 --budget 0.55

Đối số đầu là bot cần đo; các đối số sau là đối thủ (tên trong submissions/
hoặc đường dẫn .py). Bot dùng TIME_BUDGET/MAX_DEPTH của chính file đó.
"""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import evolve
from common import SUBMISSIONS


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
    a = ap.parse_args()

    bot = resolve(a.bot)
    # budget/depth = None -> giữ nguyên cấu hình trong file (evolve.play chỉ đặt cho dict)
    cfg = (a.opening, None, None, a.max_turns)
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
    for opp, (w, d, l) in res.items():
        print("  vs {:<14} W {:>3}  D {:>3}  L {:>3}  điểm {:.2f}".format(
            opp, w, d, l, (w + 0.5 * d) / max(1, w + d + l)))
    print("({:.0f}s)".format(time.time() - t0))


if __name__ == "__main__":
    main()
