"""Phase 1 - bootstrap data: the existing alpha-beta bot plays itself.

Needs: nothing.  Writes: data/bootstrap.npz (one-hot teacher moves + game results).
  uv run python -m training.alphazero.phases.p01_bootstrap --run main --preset quick"""
import sys
import time

from ..data import save_shard
from ..games import bootstrap_game, game_summary, run_parallel

PHASE, REQUIRES, TITLE = "p01", None, "alpha-beta teacher games -> data/bootstrap.npz"


def run(ctx):
    cfg = ctx.cfg
    jobs = [(cfg.seed * 1_000_003 + i, cfg.boot_budget, cfg.wall_limit, cfg.opening_plies, cfg.ply_cap)
            for i in range(cfg.boot_games)]
    samples, stats, started = [], [], time.time()
    for done, (records, st) in enumerate(run_parallel(bootstrap_game, jobs, cfg.workers), 1):
        samples.extend(records)
        stats.append(st)
        if done % 50 == 0 or done == len(jobs):
            print("  %d/%d games, %d positions, %.0fs" % (done, len(jobs), len(samples), time.time() - started), flush=True)
    save_shard(ctx.path("data", "bootstrap.npz"), samples)
    writer = ctx.writer()
    for name, value in game_summary(stats).items():
        writer.add_scalar("bootstrap/" + name, value, 0)
    writer.add_scalar("bootstrap/positions", len(samples), 0)
    writer.close()


if __name__ == "__main__":
    from ._cli import main
    main(sys.modules[__name__])
