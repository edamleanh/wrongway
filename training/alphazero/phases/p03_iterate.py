"""Phase 3 - AlphaZero loop, cfg.iterations times:
  self-play (MCTS with the best net) -> train a candidate on the replay buffer
  -> gate: candidate vs best (alpha-beta with each net) AND candidate vs the no-network engine;
  accept if score vs best >= accept_score and score vs no-network >= anchor_min,
  else retry once with lr * retry_lr_factor on the same data.

Needs: p02.  Writes: data/selfplay_iterNNN.npz, checkpoints/iterNNN.pt, manifest best/history.
Resumable: re-running continues at the next iteration. With stop_after_hours > 0 the phase
pauses (exit code 3, phase not marked done) before an iteration that would not finish in time.
More iterations later:
  uv run python -m training.alphazero.phases.p03_iterate --run main --set iterations=30 --force"""
import sys
import time

from ..config import pick_device
from ..data import load_buffer, save_shard
from ..hf_sync import safe_push
from ..games import game_summary, run_parallel, selfplay_game, tagged_eval_game
from ..net import load_checkpoint, save_checkpoint
from ..run_state import PROCESS_STARTED
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
    """-> (score vs best net, score vs no-network engine). Same opening twice, sides swapped."""
    cfg = ctx.cfg

    def jobs(n, opponent, salt):
        return [(cfg.seed + salt + 7919 * (g // 2), g % 2, cfg.eval_budget, cfg.wall_limit,
                 cfg.opening_plies, cfg.ply_cap, opponent) for g in range(n)]

    all_jobs = jobs(cfg.eval_games, "old", 0) + jobs(cfg.anchor_games, "base", 104729)
    nets = {"new": str(ctx.path(candidate)), "old": str(ctx.path(ctx.manifest["best"]))}
    scores = {"old": [], "base": []}
    for opponent, score in run_parallel(tagged_eval_game, all_jobs, cfg.workers, nets):
        scores[opponent].append(score)
    mean = {k: sum(v) / len(v) if v else 1.0 for k, v in scores.items()}   # no games -> no constraint
    return mean["old"], mean["base"]


def run(ctx):
    cfg = ctx.cfg
    writer = ctx.writer()
    durations = []
    while ctx.manifest["iteration"] < cfg.iterations:
        iteration = ctx.manifest["iteration"] + 1
        if cfg.stop_after_hours > 0 and durations:
            elapsed = time.time() - PROCESS_STARTED
            if elapsed + max(durations) > cfg.stop_after_hours * 3600:
                writer.close()
                print("  pausing before iteration %d: %.1f h used, an iteration takes up to %.1f h, limit %.1f h"
                      % (iteration, elapsed / 3600, max(durations) / 3600, cfg.stop_after_hours), flush=True)
                raise SystemExit(3)
        iter_started = time.time()
        print("  == iteration %d/%d (best: %s)" % (iteration, cfg.iterations, ctx.manifest["best"]), flush=True)
        selfplay(ctx, iteration, writer)
        lr = cfg.lr
        for attempt in range(2):
            candidate = train(ctx, iteration, lr, writer)
            score, anchor = gate(ctx, candidate)
            accepted = score >= cfg.accept_score and anchor >= cfg.anchor_min
            ctx.manifest["history"].append({"iter": iteration, "attempt": attempt, "lr": lr,
                                            "score": score, "anchor": anchor, "accepted": accepted})
            writer.add_scalar("gate/score", score, iteration)
            writer.add_scalar("gate/anchor_score", anchor, iteration)
            writer.add_scalar("gate/accepted", int(accepted), iteration)
            print("  gate: %s scores %.3f vs %s, %.3f vs no-network -> %s"
                  % (candidate, score, ctx.manifest["best"], anchor, "ACCEPT" if accepted else "reject"), flush=True)
            if accepted:
                ctx.manifest["best"] = candidate
                break
            lr *= cfg.retry_lr_factor
        ctx.manifest["iteration"] = iteration
        ctx.save()
        writer.flush()
        safe_push(ctx)                       # no-op unless AZ_HF_REPO is set
        durations.append(time.time() - iter_started)
    writer.close()


if __name__ == "__main__":
    from ._cli import main
    main(sys.modules[__name__])
