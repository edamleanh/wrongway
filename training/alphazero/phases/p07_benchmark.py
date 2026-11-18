"""Phase 7 - acceptance checks for the embedded agent (pure Python, as graded).

1. pytest (rules parity + inference parity), fast tests only
2. tools/check_submission.py <agent>: legality vs sample bots, per-move time
   (its file-name check expects an MSSV number - rename the file when submitting)
3. training/gauntlet.py <agent> <opponents> --pairs N: strength (spec: accept if >= 55% vs MSSV)
Needs: p06.  Writes: export/benchmark.txt, manifest["benchmark"], TensorBoard bench/*."""
import re
import subprocess
import sys
from pathlib import Path

from ..engine import PROJECT

PHASE, REQUIRES, TITLE = "p07", "p06", "benchmark the embedded agent"
_LINE = re.compile(r"vs (\S+)\s+W\s+\d+\s+D\s+\d+\s+L\s+\d+\s+\S+\s+([0-9.]+)")


def parse_gauntlet(text):
    return {m.group(1): float(m.group(2)) for m in _LINE.finditer(text)}


def run(ctx):
    cfg = ctx.cfg
    agent = str(ctx.agent_path)
    report = []

    def sh(cmd):
        print("  $", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True)
        report.append("$ %s\n(exit %d)\n%s\n%s" % (" ".join(cmd), result.returncode, result.stdout, result.stderr))
        print(result.stdout[-2500:], result.stderr[-1500:], sep="\n")
        return result

    tests = sh([sys.executable, "-m", "pytest", "-q", "-m", "not slow", "training/alphazero/tests"])
    check = sh([sys.executable, "tools/check_submission.py", agent])
    opponents = [o for o in cfg.bench_opponents.split(",") if o]
    gauntlet = sh([sys.executable, "training/gauntlet.py", agent, *opponents,
                   "--pairs", str(cfg.bench_pairs), "--workers", str(cfg.bench_workers)])
    scores = parse_gauntlet(gauntlet.stdout)
    writer = ctx.writer()
    for opponent, score in scores.items():
        writer.add_scalar("bench/score_vs_" + Path(opponent).stem, score, ctx.manifest["iteration"])
    writer.close()
    accepted = tests.returncode == 0 and scores.get("MSSV", 0.0) >= cfg.accept_score
    ctx.manifest["benchmark"] = {"tests_ok": tests.returncode == 0, "check_exit": check.returncode,
                                 "scores": scores, "accepted": accepted}
    ctx.save()
    ctx.path("export", "benchmark.txt").write_text("\n\n".join(report), encoding="utf-8")
    print("  verdict: %s (scores %s)" % ("ACCEPT" if accepted else "REJECT", scores))


if __name__ == "__main__":
    from ._cli import main
    main(sys.modules[__name__])
