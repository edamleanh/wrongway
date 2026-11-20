"""Run the AlphaZero pipeline (all phases or a slice) for one run folder.

  uv run python -m training.alphazero.pipeline --run main --preset quick        # phases 1..7
  uv run python -m training.alphazero.pipeline --run main --from 3 --to 3 --set iterations=30 --force
  uv run python -m training.alphazero.pipeline --run main --status
  uv run tensorboard --logdir training/alphazero/runs

Phases (each needs the one before):
  1 bootstrap -> 2 pretrain -> 3 iterate (self-play/train/gate) -> 4 distill
  -> 5 export .pth -> 6 embed into agent .py -> 7 benchmark"""
from .phases import (p01_bootstrap, p02_pretrain, p03_iterate, p04_distill, p05_export_pth,
                     p06_embed, p07_benchmark)
from .phases._cli import build_parser, run_phase
from .run_state import RunContext

PHASES = [p01_bootstrap, p02_pretrain, p03_iterate, p04_distill, p05_export_pth, p06_embed, p07_benchmark]


def status(ctx):
    m = ctx.manifest
    print("run %s (%s)  preset=%s  iteration %d/%d  best=%s" % (ctx.name, ctx.dir, m["preset"], m["iteration"],
                                                              ctx.cfg.iterations, m["best"]))
    for module in PHASES:
        print("  %-4s %-9s %s" % (module.PHASE, "done" if ctx.is_done(module.PHASE) else "pending", module.TITLE))
    for row in m["history"][-5:]:
        print("  iter %d attempt %d lr %.1e score %.3f vs-no-net %s accepted %s"
              % (row["iter"], row["attempt"], row["lr"], row["score"],
                 "%.3f" % row["anchor"] if "anchor" in row else "-", row["accepted"]))
    if "benchmark" in m:
        print("  benchmark:", m["benchmark"])


def main():
    ap = build_parser(__doc__)
    ap.add_argument("--from", dest="start", type=int, default=1, help="first phase number (1-7)")
    ap.add_argument("--to", dest="stop", type=int, default=7, help="last phase number (1-7)")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    args = ap.parse_args()
    ctx = RunContext(args.run, args.preset, args.set)
    if args.status:
        status(ctx)
        return
    for module in PHASES[args.start - 1:args.stop]:
        run_phase(ctx, module, args.force)
    print("TensorBoard: uv run tensorboard --logdir %s" % ctx.path("tb"))


if __name__ == "__main__":
    main()
