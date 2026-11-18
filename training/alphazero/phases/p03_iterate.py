"""Phase 3 - AlphaZero loop, cfg.iterations times:
  self-play (MCTS with the best net) -> train a candidate on the replay buffer
  -> gate (candidate vs best, alpha-beta with each net) -> accept if score >= accept_score,
  else retry once with lr * retry_lr_factor on the same data.

Needs: p02.  Writes: data/selfplay_iterNNN.npz, checkpoints/iterNNN.pt, manifest best/history.
Resumable: re-running continues at the next iteration. More iterations later:
  uv run python -m training.alphazero.phases.p03_iterate --run main --set iterations=30 --force"""
import sys
import time

from ..config import pick_device
from ..data import load_buffer, save_shard
from ..games import eval_game, game_summary, run_parallel, selfplay_game
from ..net import load_checkpoint, save_checkpoint
from ..trainer import train_policy_value

PHASE, REQUIRES, TITLE = "p03", "p02", "self-play / train / gate iterations"
SP_KEYS = ("c_puct", "dirichlet_alpha", "dirichlet_eps", "opening_plies", "ply_cap", "simulations", "temp_moves")


def selfplay(ctx, iteration, writer):
    cfg = ctx.cfg
    out = ctx.path("data", "selfplay_iter%03d.npz" % iteration)
    if out.exists():
        print("  self-play data for iteration %d exists - reuse" % iteration)
        return
    params = {k: getattr(cfg, k) for k in SP_KEYS}
    jobs = [(cfg.seed * 1_000_003 + iteration * 10_007 + g, params) for g in range(cfg.sp_games)]
    samples, stats, started = [], [], time.time()
    best = str(ctx.path(ctx.manifest["best"]))
    for done, (records, st) in enumerate(run_parallel(selfplay_game, jobs, cfg.workers, {"model": best}), 1):
        samples.extend(records)
        stats.append(st)
        if done % 10 == 0 or done == len(jobs):
            print("  self-play %d/%d games, %d positions, %.0fs" % (done, len(jobs), len(samples), time.time() - started), flush=True)
    save_shard(out, samples)
    for name, value in game_summary(stats).items():
        writer.add_scalar("selfplay/" + name, value, iteration)
    writer.add_scalar("selfplay/positions", len(samples), iteration)
    writer.add_scalar("selfplay/sec_per_game", (time.time() - started) * max(1, cfg.workers) / max(1, len(jobs)), iteration)


def train(ctx, iteration, lr, writer):
    cfg = ctx.cfg
    data = load_buffer(ctx.path("data"), cfg.buffer_size)
    net = load_checkpoint(ctx.path(ctx.manifest["best"]))
    step = train_policy_value(net, data, cfg, pick_device(cfg.device), writer,
                              ctx.manifest["steps"].get("train", 0), cfg.epochs_per_iter, lr, "train")
    ctx.manifest["steps"]["train"] = step
    candidate = "checkpoints/iter%03d.pt" % iteration
    save_checkpoint(ctx.path(candidate), net.cpu(), iteration)
    ctx.save()
    return candidate


def gate(ctx, candidate):
    cfg = ctx.cfg
    jobs = [(cfg.seed + 7919 * (g // 2), g % 2, cfg.eval_budget, cfg.wall_limit, cfg.opening_plies, cfg.ply_cap)
            for g in range(cfg.eval_games)]                     # same opening twice, sides swapped
    nets = {"new": str(ctx.path(candidate)), "old": str(ctx.path(ctx.manifest["best"]))}
    scores = list(run_parallel(eval_game, jobs, cfg.workers, nets))
    return sum(scores) / max(1, len(scores))


def run(ctx):
    cfg = ctx.cfg
    writer = ctx.writer()
    while ctx.manifest["iteration"] < cfg.iterations:
        iteration = ctx.manifest["iteration"] + 1
        print("  == iteration %d/%d (best: %s)" % (iteration, cfg.iterations, ctx.manifest["best"]), flush=True)
        selfplay(ctx, iteration, writer)
        lr = cfg.lr
        for attempt in range(2):
            candidate = train(ctx, iteration, lr, writer)
            score = gate(ctx, candidate)
            accepted = score >= cfg.accept_score
            ctx.manifest["history"].append({"iter": iteration, "attempt": attempt, "lr": lr,
                                            "score": score, "accepted": accepted})
            writer.add_scalar("gate/score", score, iteration)
            writer.add_scalar("gate/accepted", int(accepted), iteration)
            print("  gate: %s scores %.3f vs %s -> %s" % (candidate, score, ctx.manifest["best"],
                                                          "ACCEPT" if accepted else "reject"), flush=True)
            if accepted:
                ctx.manifest["best"] = candidate
                break
            lr *= cfg.retry_lr_factor
        ctx.manifest["iteration"] = iteration
        ctx.save()
    writer.close()


if __name__ == "__main__":
    from ._cli import main
    main(sys.modules[__name__])
