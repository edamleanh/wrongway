"""Shared command line for every phase: --run NAME [--preset P] [--set key=value ...] [--force]."""
import argparse
import time

from ..run_state import RunContext


def build_parser(description):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main", help="run folder under training/alphazero/runs/ (default: main)")
    ap.add_argument("--preset", help="only when creating a run: smoke | quick | standard | spec")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config field, repeatable (see config.py)")
    ap.add_argument("--force", action="store_true", help="redo even if marked done (later phases become stale)")
    return ap


def run_phase(ctx, module, force=False):
    if module.REQUIRES:
        ctx.require(module.REQUIRES)
    if ctx.is_done(module.PHASE) and not force:
        print("[%s] %s already done - skip (--force to redo)" % (ctx.name, module.PHASE))
        return
    print("[%s] %s: %s" % (ctx.name, module.PHASE, module.TITLE), flush=True)
    started = time.time()
    module.run(ctx)
    ctx.mark_done(module.PHASE)
    print("[%s] %s done in %.0fs" % (ctx.name, module.PHASE, time.time() - started), flush=True)


def main(module):
    args = build_parser(module.__doc__).parse_args()
    run_phase(RunContext(args.run, args.preset, args.set), module, args.force)
