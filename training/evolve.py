#!/usr/bin/env python3
"""Học trọng số hàm đánh giá bằng chiến lược tiến hóa (1+lambda) qua tự đấu.

Mỗi thế hệ:
1. Sinh LAMBDA bộ trọng số đột biến từ nhà vô địch: w' = w + sigma * scale * N(0, 1).
2. Mỗi đột biến đấu với nhà vô địch (đổi bên, mở đầu ngẫu nhiên vài nước để
   các ván khác nhau) và với "nhóm chuẩn": các nhà vô địch cũ (hall of fame),
   các bot trong training/opponents/ và wall_builder.
3. Đột biến tốt nhất phải thắng nhà vô địch >= --gate, qua một loạt ván xác
   nhận mới, và không kém nhà vô địch quá --pool-tolerance trước nhóm chuẩn
   -> trở thành nhà vô địch, được ghi vào khối WEIGHTS của MSSV.py.
4. sigma tự điều chỉnh: tăng khi tìm được bản tốt hơn, giảm khi không.

Khi train, bot tìm kiếm ở độ sâu cố định (--depth) thay vì theo thời gian
-> nhanh và kết quả lặp lại được. Khi nộp, bot đào sâu hơn theo TIME_BUDGET.

    python -u training/evolve.py --generations 30 --workers 10 > logs/evolve.log 2>&1
"""

import argparse
import json
import random
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

from common import BOT_FILE, CKPT_DIR, SUBMISSIONS, TRAIN_DIR, load_module, write_weights

# Độ lớn đột biến riêng cho từng trọng số (nhân với sigma). Khóa mới -> mặc định 0.3.
SCALES = {"race": 0.4, "walls": 0.3, "wall_urgency": 0.6, "safe_race": 1.0,
          "my_dist": 0.15, "opp_dist": 0.15, "pass_penalty": 0.5, "vuln": 0.3}
OPPONENTS_DIR = TRAIN_DIR / "opponents"
SAMPLE_BOTS = ["wall_builder"]

_modules = {}


def _agent(spec):
    """spec: dict trọng số (khóa "__bot__" = file bot dùng trọng số đó, mặc định MSSV.py),
    đường dẫn .py, hoặc tên bot mẫu."""
    if isinstance(spec, dict):
        path = spec.get("__bot__", str(BOT_FILE))
        if ("w", path) not in _modules:
            _modules[("w", path)] = load_module(path, "bot_evolve_%d" % len(_modules))
        return _modules[("w", path)], {k: v for k, v in spec.items() if k != "__bot__"}
    path = spec if spec.endswith(".py") else str(SUBMISSIONS / (spec + ".py"))
    if path not in _modules:
        _modules[path] = load_module(path, "opp_%d" % len(_modules))
    return _modules[path], None


def play(job):
    """job = (spec_p0, spec_p1, seed, opening_plies, budget, depth, max_turns) -> winner.

    budget/depth = None: giữ cấu hình trong file bot.
    """
    from environment import Environment
    s0, s1, seed, opening, budget, depth, max_turns = job
    rng = random.Random(seed)
    random.seed(seed)
    env = Environment(max_turns=max_turns)
    for _ in range(opening):                      # mở đầu ngẫu nhiên, chỉ nước đi
        env.step({"type": "move", "to": rng.choice(env.legal_moves())})
    specs = (s0, s1)
    while not env.done:
        p = env.current_player
        module, weights = _agent(specs[p])
        if weights is not None:
            module.WEIGHTS = weights
            if budget is not None:
                module.TIME_BUDGET = budget
            if depth is not None:
                module.MAX_DEPTH = depth
        try:
            action = module.choose_action(env.get_observation(p))
        except Exception:
            action = {"type": "pass"}
        env.step(action)
    return env.winner


def match_jobs(a, b, n_pairs, seed0, cfg):
    """n_pairs cặp ván, mỗi cặp cùng seed nhưng đổi bên -> công bằng."""
    jobs = []
    for i in range(n_pairs):
        seed = seed0 + i
        jobs.append(((a, b, seed) + cfg, 0))     # a cầm quân 0
        jobs.append(((b, a, seed) + cfg, 1))     # a cầm quân 1
    return jobs


def score_of(results):
    total = 0.0
    for winner, side in results:
        total += 0.5 if winner is None else float(winner == side)
    return total / max(1, len(results))


def run_batches(pool, batches):
    """batches: dict key -> list (job, side). Chạy song song tất cả, trả score theo key."""
    flat = [(key, job, side) for key, items in batches.items() for job, side in items]
    winners = pool.map(play, [job for _, job, _ in flat], chunksize=1)
    grouped = {key: [] for key in batches}
    for (key, _, side), winner in zip(flat, winners):
        grouped[key].append((winner, side))
    return {key: score_of(v) for key, v in grouped.items()}


def mutate(weights, sigma, rng):
    return {k: round(v + sigma * SCALES.get(k, 0.3) * rng.gauss(0, 1), 4)
            for k, v in weights.items()}


def pool_jobs(spec, pool_specs, pairs, seed0, cfg):
    return [j for opp in pool_specs for j in match_jobs(spec, opp, pairs, seed0, cfg)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", default=str(BOT_FILE), help="file bot cần tiến hóa (mặc định MSSV.py)")
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--lam", type=int, default=6, help="số đột biến mỗi thế hệ")
    ap.add_argument("--pairs", type=int, default=10, help="số cặp ván vs nhà vô địch / đột biến")
    ap.add_argument("--pool-pairs", type=int, default=3, help="số cặp ván vs mỗi đối thủ chuẩn")
    ap.add_argument("--hof", type=int, default=3, help="số nhà vô địch cũ trong nhóm chuẩn")
    ap.add_argument("--confirm-pairs", type=int, default=12)
    ap.add_argument("--gate", type=float, default=0.55)
    ap.add_argument("--pool-tolerance", type=float, default=0.05)
    ap.add_argument("--target", action="append", default=[],
                    help="bot .py làm mục tiêu chính (lặp lại được). Khi có: chọn/nhận đột biến "
                         "theo điểm trước mục tiêu, so với nhà vô địch trên cùng seed")
    ap.add_argument("--target-pairs", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.06, help="mức hơn tối thiểu trước mục tiêu")
    ap.add_argument("--min-vs-champ", type=float, default=0.45,
                    help="khi có --target: điểm tối thiểu khi đấu trực tiếp nhà vô địch")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=10.0, help="trần thời gian/nước khi train")
    ap.add_argument("--depth", type=int, default=4, help="độ sâu cố định khi train")
    ap.add_argument("--opening", type=int, default=2, help="số nước mở đầu ngẫu nhiên")
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    bot_file = Path(a.bot).resolve()

    def S(w):                                     # gắn file bot vào bộ trọng số
        return dict(w, __bot__=str(bot_file))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    history = CKPT_DIR / "evolve_history.jsonl"
    rng = random.Random(a.seed if a.seed is not None else time.time_ns())
    champion = dict(load_module(bot_file, "bot_init").WEIGHTS)
    src = bot_file.read_text(encoding="utf-8")
    for k in SCALES:                              # đặc trưng bot có dùng mà chưa có trọng số -> 0
        if '"%s"' % k in src:
            champion.setdefault(k, 0.0)
    hall = []                                     # các nhà vô địch cũ
    fixed = SAMPLE_BOTS + [str(p) for p in sorted(OPPONENTS_DIR.glob("*.py"))]
    targets = [str(Path(t).resolve()) for t in a.target]
    sigma = a.sigma
    cfg = (a.opening, a.budget, a.depth, a.max_turns)
    print("Bot:", bot_file.name, "| nhà vô địch ban đầu:", champion)
    print("Đối thủ chuẩn:", [Path(f).name for f in fixed], "| mục tiêu:", [Path(t).name for t in targets],
          flush=True)

    def target_jobs(w, seed):
        return [j for t in targets for j in match_jobs(S(w), t, a.target_pairs, seed, cfg)]

    with Pool(a.workers) as pool:
        for gen in range(a.generations):
            t0 = time.time()
            seed0 = rng.randrange(10 ** 9)
            mutants = [mutate(champion, sigma, rng) for _ in range(a.lam)]
            pool_specs = fixed + [S(h) for h in hall[-a.hof:]]

            batches = {("pool", "champ"): pool_jobs(S(champion), pool_specs, a.pool_pairs, seed0 + 7777, cfg)}
            if targets:
                batches[("target", "champ")] = target_jobs(champion, seed0 + 4242)
            for i, m in enumerate(mutants):
                batches[("champ", i)] = match_jobs(S(m), S(champion), a.pairs, seed0, cfg)
                batches[("pool", i)] = pool_jobs(S(m), pool_specs, a.pool_pairs, seed0 + 7777, cfg)
                if targets:
                    batches[("target", i)] = target_jobs(m, seed0 + 4242)
            scores = run_batches(pool, batches)

            base_pool = scores[("pool", "champ")]
            accepted, confirm, extra = False, None, ""
            if targets:
                base_t = scores[("target", "champ")]
                best = max(range(a.lam), key=lambda i: (scores[("target", i)], scores[("champ", i)]))
                vs_champ, vs_pool = scores[("champ", best)], scores[("pool", best)]
                t_m = scores[("target", best)]
                extra = " | vs mục tiêu {:.2f} (vô địch {:.2f})".format(t_m, base_t)
                if (t_m >= base_t + a.margin and vs_champ >= a.min_vs_champ
                        and vs_pool >= base_pool - a.pool_tolerance):
                    c = run_batches(pool, {"m": target_jobs(mutants[best], seed0 + 99991),
                                           "c": target_jobs(champion, seed0 + 99991)})
                    confirm = c["m"] - c["c"]
                    accepted = ((t_m - base_t) + confirm) / 2 >= a.margin
            else:
                best = max(range(a.lam), key=lambda i: (scores[("champ", i)], scores[("pool", i)]))
                vs_champ, vs_pool = scores[("champ", best)], scores[("pool", best)]
                if vs_champ >= a.gate and vs_pool >= base_pool - a.pool_tolerance:
                    confirm = run_batches(pool, {"c": match_jobs(
                        S(mutants[best]), S(champion), a.confirm_pairs, seed0 + 99991, cfg)})["c"]
                    n1, n2 = 2 * a.pairs, 2 * a.confirm_pairs
                    accepted = (vs_champ * n1 + confirm * n2) / (n1 + n2) >= a.gate

            if accepted:
                shutil.copy(bot_file, CKPT_DIR / "{}_before_gen_{:03d}.py".format(bot_file.stem, gen))
                hall.append(champion)
                champion = mutants[best]
                bot_file.write_text(write_weights(bot_file.read_text(encoding="utf-8"), champion),
                                    encoding="utf-8")
                sigma = min(2.0, sigma * 1.3)
            else:
                sigma = max(0.15, sigma * 0.8)

            print("Thế hệ {:>2}: vs vô địch {:.2f} | vs nhóm chuẩn {:.2f} (vô địch {:.2f}){}{} "
                  "| {} | sigma {:.2f} | {:.0f}s".format(
                      gen, vs_champ, vs_pool, base_pool, extra,
                      "" if confirm is None else " | xác nhận {:+.2f}".format(confirm),
                      "CHẤP NHẬN" if accepted else "giữ nguyên", sigma, time.time() - t0), flush=True)
            if accepted:
                print("   ->", champion, flush=True)
            with open(history, "a", encoding="utf-8") as f:
                f.write(json.dumps({"gen": gen, "bot": bot_file.name, "vs_champ": vs_champ,
                                    "confirm": confirm, "vs_pool": vs_pool, "champ_vs_pool": base_pool,
                                    "target": scores.get(("target", best)),
                                    "champ_target": scores.get(("target", "champ")),
                                    "accepted": accepted, "sigma": sigma, "weights": champion}) + "\n")


if __name__ == "__main__":
    main()
