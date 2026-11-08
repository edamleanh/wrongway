#!/usr/bin/env python3
"""Round-robin giống mùa giải: mỗi cặp đấu 2 ván, đổi bên. Thắng 3, hòa 1.

Ví dụ:
    py -3 tools/tournament.py MSSV greedy random_agent wall_builder --seeds 3
    py -3 tools/tournament.py MSSV --vs-all --seeds 2
"""

import argparse
import itertools
import json
import sys
import time

import _arena_path  # noqa: F401  (thiết lập sys.path)
from _arena_path import LOGS, SUBMISSIONS
from arena import resolve_player, run_match, save_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("players", nargs="+", help="tên file trong submissions/ hoặc đường dẫn .py")
    p.add_argument("--vs-all", action="store_true",
                   help="chỉ đấu player đầu tiên với mọi bot khác trong submissions/")
    p.add_argument("--seeds", type=int, default=1, help="số seed cho mỗi cặp (mỗi seed = 2 ván)")
    p.add_argument("--timeout", type=float, default=1.0)
    p.add_argument("--max-turns", type=int, default=500)
    p.add_argument("--save-logs", action="store_true", help="lưu replay từng ván vào logs/")
    args = p.parse_args(argv)

    names = list(args.players)
    if args.vs_all:
        others = sorted(f.stem for f in SUBMISSIONS.glob("*.py")
                        if f.stem not in ("TEMPLATE", names[0]))
        names = [names[0]] + others
        pairs = [(names[0], o) for o in others]
    else:
        pairs = list(itertools.combinations(names, 2))
    paths = {n: resolve_player(n, SUBMISSIONS) for n in names}

    table = {n: {"W": 0, "D": 0, "L": 0, "timeout": 0, "invalid": 0, "error": 0,
                 "max_elapsed": 0.0} for n in names}
    start = time.time()
    for a, b in pairs:
        for seed in range(args.seeds):
            for p0, p1 in ((a, b), (b, a)):
                res = run_match(paths[p0], paths[p1], timeout=args.timeout,
                                max_turns=args.max_turns, seed=seed)
                w = res["winner"]
                for idx, name in enumerate((p0, p1)):
                    row = table[name]
                    if w is None:
                        row["D"] += 1
                    elif w == idx:
                        row["W"] += 1
                    else:
                        row["L"] += 1
                    for k in ("timeout", "invalid", "error"):
                        row[k] += res["stats"][idx][k]
                for t in res["turns"]:
                    who = (p0, p1)[t["player"]]
                    table[who]["max_elapsed"] = max(table[who]["max_elapsed"], t["elapsed"])
                outcome = "draw" if w is None else "P{} {}".format(w, (p0, p1)[w])
                print("{:>14} vs {:<14} seed={} -> {} ({} turns)".format(
                    p0, p1, seed, outcome, res["total_turns"]))
                if args.save_logs:
                    save_json(LOGS / "{}_vs_{}_s{}.json".format(p0, p1, seed), res)

    print("\n{:<16}{:>4}{:>4}{:>4}{:>6}{:>8}{:>6}{:>6}{:>6}{:>9}".format(
        "Agent", "W", "D", "L", "Pts", "Ratio", "TO", "Inv", "Err", "MaxT(s)"))
    ranked = sorted(names, key=lambda n: (-(3 * table[n]["W"] + table[n]["D"]), -table[n]["W"]))
    for n in ranked:
        r = table[n]
        games = r["W"] + r["D"] + r["L"]
        pts = 3 * r["W"] + r["D"]
        ratio = pts / (3 * games) if games else 0.0
        print("{:<16}{:>4}{:>4}{:>4}{:>6}{:>8.2f}{:>6}{:>6}{:>6}{:>9.3f}".format(
            n, r["W"], r["D"], r["L"], pts, ratio, r["timeout"], r["invalid"],
            r["error"], r["max_elapsed"]))
    print("\nTổng thời gian: {:.1f}s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
