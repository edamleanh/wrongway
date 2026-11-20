"""Self-play training for the DEPLOYED az_agent (expert iteration), continuing from a trained net.

Phase 3 of the pipeline (p03_iterate) generates data with net-only MCTS, which plays weaker than
the agent itself (alpha-beta + net at the root). Here the agent's own search is the teacher:

  each generation:
    1. self-play   the agent (alpha-beta + best net at the root, same code as az_agent.py) plays
                   itself; league_frac of the games are against the no-network engine (= MSSV.py),
                   learner on each side half the time. Target = the move the search played
                   (one-hot), value target = game result. sp_budget > eval_budget makes the
                   teacher stronger than the agent at play time.
    2. train       candidate = best + epochs on the replay buffer (newest buffer_size positions)
    3. gate        candidate vs best and candidate vs no-network engine (= MSSV.py), same
                   opening twice with sides swapped; scores split by side (first / second).
                   Accept if score vs best >= accept_score and vs no-network >= anchor_min.
    4. embed       accepted net -> MODEL_BLOB of the output agent file (VALUE_BLOB kept).

Start (net = MODEL_BLOB of an agent file, or a .pt checkpoint; default az_agent_std1.py):
  uv run python -m training.alphazero.selfplay_train --run sp1
  uv run python -m training.alphazero.selfplay_train --run sp1 --init training/alphazero/runs/std1/checkpoints/iter000.pt
  uv run python -m training.alphazero.selfplay_train --run try --preset smoke     # wiring test, ~1 min
Continue / more generations (config is saved in the run; change with --set):
  uv run python -m training.alphazero.selfplay_train --run sp1 --set generations=20
  uv run python -m training.alphazero.selfplay_train --run sp1 --status
Benchmark the result like any agent:
  python training/gauntlet.py training/alphazero/runs/sp1/export/az_agent_sp.py MSSV --pairs 40 --workers 6

Kaggle: training/alphazero/kaggle_selfplay.ipynb (stop_after_hours -> exit code 3, AZ_HF_REPO
pushes the run to Hugging Face after start and after every generation, the next session resumes).

Outputs in training/alphazero/runs/<run>/: selfplay.json (config + progress), data/gen*.npz,
checkpoints/gen*.pt, tb/ (TensorBoard: sp/*, gate/*, train/*), export/az_agent_sp.py."""
import argparse
import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch

from .adapters import TorchPolicy
from .blob import embed, pack_layers
from .config import _parse, pick_device
from .data import load_buffer, save_shard
from .engine import AGENT_FILE, SUBMISSIONS, load_agent
from .export import layers_from_linears
from .hf_sync import safe_push_selfplay
from .games import agent_selfplay_game, game_summary, random_opening, run_parallel, side_eval_game
from .net import AZNet, load_checkpoint, policy_layers, save_checkpoint
from .run_state import PROCESS_STARTED, RUNS_DIR
from .trainer import train_policy_value

STATE_FILE = "selfplay.json"
DEFAULT_INIT = SUBMISSIONS / "az_agent_std1.py"


@dataclass
class SPConfig:
    seed: int = 0
    device: str = "auto"              # training only; games run on CPU worker processes
    data_parallel: bool = True
    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))
    generations: int = 10
    stop_after_hours: float = 0.0     # >0: exit 3 before a generation that would end past this many
                                      # hours since the process started (Kaggle 12 h sessions)
    # self-play
    sp_games: int = 200
    sp_budget: float = 0.25           # seconds per move for the teacher search
    league_frac: float = 0.3          # share of games vs the no-network engine (= MSSV.py)
    opening_plies: int = 4            # 0..N random pawn moves first
    explore_plies: int = 12           # exploration window (plies after the opening)
    explore_eps: float = 0.15         # chance to play a policy-sampled move inside the window
    ply_cap: int = 200
    wall_limit: int = 16
    # training (fields read by trainer.train_policy_value)
    buffer_size: int = 200_000
    epochs_per_gen: int = 4
    batch_size: int = 512
    lr: float = 5e-4                  # fine-tuning a trained net: lower than pretraining
    lr_min: float = 5e-5
    weight_decay: float = 1e-4
    mirror_augment: bool = True
    log_every: int = 50
    # gate
    eval_budget: float = 0.10
    eval_games: int = 40              # candidate vs best (even: pairs with sides swapped)
    anchor_games: int = 40            # candidate vs no-network engine
    accept_score: float = 0.55
    anchor_min: float = 0.5


PRESETS = {
    "default": {},
    "smoke": {"workers": 2, "generations": 1, "sp_games": 2, "sp_budget": 0.02, "ply_cap": 30,
              "epochs_per_gen": 1, "batch_size": 32, "eval_budget": 0.02, "eval_games": 2,
              "anchor_games": 2, "log_every": 1},
}


# ---------------------------------------------------------------- run folder + progress
class SPRun:
    def __init__(self, name, preset=None, overrides=(), runs_dir=None):
        self.dir = Path(runs_dir or RUNS_DIR) / name
        self.name = name
        self.state_path = self.dir / STATE_FILE
        if self.state_path.exists():
            if preset:
                raise SystemExit("run '%s' exists; drop --preset (use --set) or pick a new --run" % name)
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            values = self.state["config"]
        else:
            self.state = {"gen": 0, "best": None, "best_base": None, "baseline": None,
                          "history": [], "train_step": 0, "init": None}
            values = dict(PRESETS[preset or "default"])
        cfg = SPConfig(**values)
        names = {f.name for f in fields(SPConfig)}
        for item in overrides:
            key, sep, raw = item.partition("=")
            key = key.strip()
            if not sep or key not in names:
                raise SystemExit("bad --set %r (fields: %s)" % (item, ", ".join(sorted(names))))
            setattr(cfg, key, _parse(getattr(cfg, key), raw))
        self.cfg = cfg
        self.state["config"] = asdict(cfg)
        for sub in ("data", "checkpoints", "tb", "export"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.save()

    def path(self, *parts):
        return self.dir.joinpath(*parts)

    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)


# ---------------------------------------------------------------- initial net
def net_from_agent(agent_path):
    """AZNet whose backbone + policy head come from an agent's MODEL_BLOB (value head: fresh)."""
    agent = load_agent(agent_path, name="sp_init_%d" % time.time_ns())
    if not agent.MODEL_BLOB:
        raise SystemExit("%s has no MODEL_BLOB - pass an agent with embedded weights or a .pt" % agent_path)
    flat = agent.unpack_blob(agent.MODEL_BLOB, agent.POLICY_SHAPES)
    net, cursor = AZNet(), 0
    with torch.no_grad():
        for layer, (inputs, outputs) in zip(policy_layers(net), agent.POLICY_SHAPES):
            n = inputs * outputs
            layer.weight.copy_(torch.tensor(flat[cursor:cursor + n]).view(outputs, inputs))
            cursor += n
            layer.bias.copy_(torch.tensor(flat[cursor:cursor + outputs]))
            cursor += outputs
    return net.eval()


def initial_net(init):
    init = Path(init)
    return load_checkpoint(init) if init.suffix == ".pt" else net_from_agent(init)


# ---------------------------------------------------------------- the four steps
def selfplay(run, gen, writer):
    cfg = run.cfg
    out = run.path("data", "gen%03d.npz" % gen)
    if out.exists():
        print("  self-play data for generation %d exists - reuse" % gen)
        return
    rng = random.Random(cfg.seed * 1_000_003 + gen)
    jobs = []
    for g in range(cfg.sp_games):
        league = rng.random() < cfg.league_frac
        params = {"opponent": "base" if league else "self", "learner_side": g % 2, "budget": cfg.sp_budget,
                  "wall_limit": cfg.wall_limit, "opening_plies": cfg.opening_plies, "ply_cap": cfg.ply_cap,
                  "explore_plies": cfg.explore_plies, "explore_eps": cfg.explore_eps}
        jobs.append((cfg.seed * 1_000_003 + gen * 10_007 + g, params))
    samples, stats, league_scores, started = [], [], {0: [], 1: []}, time.time()
    best = str(run.path(run.state["best"]))
    for done, (records, st) in enumerate(run_parallel(agent_selfplay_game, jobs, cfg.workers, {"model": best}), 1):
        samples.extend(records)
        stats.append(st)
        if st["opponent"] == "base":
            side = st["learner_side"]
            league_scores[side].append(0.5 if st["winner"] not in (0, 1) else float(st["winner"] == side))
        if done % 10 == 0 or done == len(jobs):
            print("  self-play %d/%d games, %d positions, %.0fs"
                  % (done, len(jobs), len(samples), time.time() - started), flush=True)
    save_shard(out, samples)
    for name, value in game_summary(stats).items():
        writer.add_scalar("sp/" + name, value, gen)
    writer.add_scalar("sp/positions", len(samples), gen)
    for side, scores in league_scores.items():
        if scores:
            writer.add_scalar("sp/league_score_as_p%d" % side, sum(scores) / len(scores), gen)


def train(run, gen, writer):
    cfg = run.cfg
    data = load_buffer(run.path("data"), cfg.buffer_size, "gen*.npz")
    print("  training on %d positions" % len(data[2]), flush=True)
    net = load_checkpoint(run.path(run.state["best"]))
    run.state["train_step"] = train_policy_value(net, data, cfg, pick_device(cfg.device), writer,
                                                 run.state["train_step"], cfg.epochs_per_gen, cfg.lr, "train")
    candidate = "checkpoints/gen%03d.pt" % gen
    save_checkpoint(run.path(candidate), net.cpu(), gen)
    run.save()
    return candidate


def side_scores(rows):
    """[(side, score)] -> {"all", "p0" (new moved first), "p1" (new moved second)} means."""
    def mean(xs):
        return sum(xs) / len(xs) if xs else None
    return {"all": mean([s for _, s in rows]), "p0": mean([s for side, s in rows if side == 0]),
            "p1": mean([s for side, s in rows if side == 1])}


def play_matches(run, new, matches):
    """matches: [(opponent tag, games, seed salt)]; opponent "old" = best net, "base" = no network.
    -> {tag: side_scores}. Game g and g+1 share an opening, sides swapped."""
    cfg = run.cfg
    jobs = []
    for opponent, games, salt in matches:
        jobs += [(cfg.seed + salt + 7919 * (g // 2), g % 2, cfg.eval_budget, cfg.wall_limit,
                  cfg.opening_plies, cfg.ply_cap, opponent) for g in range(games)]
    nets = {"new": str(run.path(new)), "old": str(run.path(run.state["best"]))}
    rows = {tag: [] for tag, _, _ in matches}
    for opponent, side, score in run_parallel(side_eval_game, jobs, cfg.workers, nets):
        rows[opponent].append((side, score))
    return {tag: side_scores(r) for tag, r in rows.items()}


def fmt(s):
    if s is None or s["all"] is None:
        return "-"
    return "%.3f (first %s, second %s)" % (s["all"], "%.2f" % s["p0"] if s["p0"] is not None else "-",
                                           "%.2f" % s["p1"] if s["p1"] is not None else "-")


def export_agent(run, checkpoint, template):
    """Write the net into export/az_agent_sp.py (created from `template`), check pure-Python parity."""
    out = run.path("export", "az_agent_sp.py")
    if not out.exists():
        shutil.copyfile(template, out)
    value_blob = load_agent(out, name="sp_out_%d" % time.time_ns()).VALUE_BLOB
    net = load_checkpoint(run.path(checkpoint))
    embed(out, pack_layers(layers_from_linears(policy_layers(net))), value_blob)
    agent = load_agent(out, name="sp_check_%d" % time.time_ns())
    torch_policy, rng = TorchPolicy(net), random.Random(0)
    diff = 0.0
    for _ in range(10):
        state = random_opening(rng, 6)
        diff = max(diff, max(abs(a - b) for a, b in zip(agent._POLICY.logits(state), torch_policy.logits(state))))
    if diff > 5e-2:
        raise SystemExit("pure-Python policy differs from torch by %.4f in %s" % (diff, out))
    print("  embedded %s into %s (max |logit diff| %.4f)" % (checkpoint, out, diff), flush=True)
    return out


# ---------------------------------------------------------------- main loop
def start(run, init):
    """Generation 0: the initial net becomes best; measure it vs the no-network engine."""
    init = Path(init)
    net = initial_net(init)
    save_checkpoint(run.path("checkpoints", "gen000.pt"), net, 0)
    run.state.update(best="checkpoints/gen000.pt", init=str(init))
    run.save()
    template = init if init.suffix == ".py" else AGENT_FILE
    export_agent(run, "checkpoints/gen000.pt", template)
    if run.cfg.anchor_games:
        print("  baseline: initial net vs no-network engine (%d games)..." % run.cfg.anchor_games, flush=True)
        base = play_matches(run, "checkpoints/gen000.pt", [("base", run.cfg.anchor_games, 104729)])["base"]
        run.state.update(baseline=base, best_base=base)
        run.save()
        print("  baseline vs no-network (= MSSV.py): %s" % fmt(base), flush=True)
    safe_push_selfplay(run, "%s: start from %s" % (run.name, init.name))   # no-op unless AZ_HF_REPO is set


def status(run):
    s = run.state
    print("run %s (%s)  generation %d/%d  best=%s  init=%s"
          % (run.name, run.dir, s["gen"], run.cfg.generations, s["best"], s["init"]))
    print("  initial net vs no-network: %s" % fmt(s["baseline"]))
    print("  best net    vs no-network: %s" % fmt(s["best_base"]))
    for row in s["history"][-10:]:
        print("  gen %d  vs best %s | vs no-network %s -> %s"
              % (row["gen"], fmt(row["vs_best"]), fmt(row["vs_base"]), "ACCEPT" if row["accepted"] else "reject"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="sp1", help="run folder under training/alphazero/runs/ (default: sp1)")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="only when creating a run")
    ap.add_argument("--init", default=str(DEFAULT_INIT),
                    help="starting net: agent .py with MODEL_BLOB or AZNet .pt (default: az_agent_std1.py)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    args = ap.parse_args()

    run = SPRun(args.run, args.preset, args.set)
    if args.status:
        status(run)
        return
    cfg = run.cfg
    if run.state["best"] is None:
        start(run, args.init)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run.path("tb")))
    if run.state["baseline"]:
        for side in ("p0", "p1"):
            if run.state["baseline"][side] is not None:
                writer.add_scalar("gate/base_score_%s" % side, run.state["baseline"][side], 0)
    durations = []
    while run.state["gen"] < cfg.generations:
        gen = run.state["gen"] + 1
        if cfg.stop_after_hours > 0 and durations:
            elapsed = time.time() - PROCESS_STARTED
            if elapsed + max(durations) > cfg.stop_after_hours * 3600:
                writer.close()
                print("pausing before generation %d: %.1f h used, a generation takes up to %.1f h, limit %.1f h"
                      % (gen, elapsed / 3600, max(durations) / 3600, cfg.stop_after_hours), flush=True)
                sys.exit(3)
        began = time.time()
        print("== generation %d/%d (best: %s)" % (gen, cfg.generations, run.state["best"]), flush=True)
        selfplay(run, gen, writer)
        candidate = train(run, gen, writer)
        result = play_matches(run, candidate, [("old", cfg.eval_games, 0), ("base", cfg.anchor_games, 104729)])
        vs_best, vs_base = result["old"], result["base"]
        accepted = ((vs_best["all"] if vs_best["all"] is not None else 1.0) >= cfg.accept_score
                    and (vs_base["all"] if vs_base["all"] is not None else 1.0) >= cfg.anchor_min)
        print("  gate: vs best %s | vs no-network %s -> %s"
              % (fmt(vs_best), fmt(vs_base), "ACCEPT" if accepted else "reject"), flush=True)
        for tag, s in (("best", vs_best), ("base", vs_base)):
            for key, value in s.items():
                if value is not None:
                    writer.add_scalar("gate/%s_score_%s" % (tag, key), value, gen)
        writer.add_scalar("gate/accepted", int(accepted), gen)
        run.state["history"].append({"gen": gen, "vs_best": vs_best, "vs_base": vs_base, "accepted": accepted,
                                     "minutes": round((time.time() - began) / 60, 1)})
        if accepted:
            run.state.update(best=candidate, best_base=vs_base)
            export_agent(run, candidate, AGENT_FILE)   # template unused: file exists since start()
        run.state["gen"] = gen
        run.save()
        writer.flush()
        safe_push_selfplay(run)                       # no-op unless AZ_HF_REPO is set
        durations.append(time.time() - began)
    writer.close()
    status(run)


if __name__ == "__main__":
    main()
