"""Whole games for data generation and gating, plus a process pool to run many at once.

Every game function takes ONE tuple (pool-friendly) and runs on CPU with torch limited to
one thread: the pure-Python rules dominate the cost, so GPU/MPS is used only in trainer.py."""
import math
import multiprocessing as mp
import random

import torch

from .adapters import TorchPolicy, TorchValue
from .engine import E
from .mcts import MCTS, NetEvaluator, pick_action
from .net import load_checkpoint
from .symmetry import N_ACTIONS, canonical_action, canonical_policy

_NETS = {}   # per worker process: name -> AZNet (filled by _init_worker)


def _init_worker(checkpoints):
    torch.set_num_threads(1)
    _NETS.clear()
    for key, path in (checkpoints or {}).items():
        _NETS[key] = load_checkpoint(path, "cpu")


def run_parallel(fn, jobs, workers, checkpoints=None):
    """Yield fn(job) for every job (unordered when parallel). checkpoints: {name: .pt path}."""
    if workers <= 1:
        _init_worker(checkpoints)
        for job in jobs:
            yield fn(job)
        return
    ctx = mp.get_context("spawn")                   # same behaviour on macOS, Linux, Windows
    with ctx.Pool(workers, initializer=_init_worker, initargs=(checkpoints,)) as pool:
        yield from pool.imap_unordered(fn, jobs, chunksize=1)


def random_opening(rng, max_plies):
    state = E.State()
    for _ in range(rng.randint(0, max_plies)):
        state = E.play(state, rng.choice(E.pawn_moves(state)), False)
    return state


def finished(state, ply_cap):
    return E.outcome(state) is not None or state.turn >= ply_cap


def score_records(records, winner):
    """(features, pi, player) -> (features, pi, z), z from that player's point of view."""
    out = []
    for features, pi, player in records:
        z = 0.0 if winner is None or winner == 2 else (1.0 if winner == player else -1.0)
        out.append((features, pi, z))
    return out


def game_summary(stats):
    n = max(1, len(stats))
    return {"games": len(stats),
            "avg_plies": sum(s["plies"] for s in stats) / n,
            "p0_win": sum(s["winner"] == 0 for s in stats) / n,
            "p1_win": sum(s["winner"] == 1 for s in stats) / n,
            "draw": sum(s["winner"] not in (0, 1) for s in stats) / n}


def bootstrap_game(job):
    """(seed, budget, wall_limit, opening_plies, ply_cap) -> (samples, stats). Teacher = alpha-beta."""
    seed, budget, wall_limit, opening_plies, ply_cap = job
    rng = random.Random(seed)
    state = random_opening(rng, opening_plies)
    records = []
    while not finished(state, ply_cap):
        action = E.Search(budget=budget, wall_limit=wall_limit, mode="alphabeta").choose(state)
        pi = [0.0] * N_ACTIONS
        pi[canonical_action(state, action)] = 1.0
        records.append((E.encode(state), pi, state.turn % 2))
        state = E.play(state, action, False)
    winner = E.outcome(state)
    return score_records(records, winner), {"plies": state.turn, "winner": winner}


def selfplay_game(job):
    """(seed, params dict) -> (samples, stats). Uses the worker's "model" net."""
    seed, p = job
    rng = random.Random(seed)
    mcts = MCTS(NetEvaluator(_NETS["model"]), p["c_puct"], p["dirichlet_alpha"], p["dirichlet_eps"], rng)
    state = random_opening(rng, p["opening_plies"])
    records = []
    while not finished(state, p["ply_cap"]):
        visits = mcts.search(state, p["simulations"], noise=True)
        records.append((E.encode(state), canonical_policy(state, visits), state.turn % 2))
        temperature = 1.0 if len(records) <= p["temp_moves"] else 0.0
        state = E.play(state, pick_action(visits, temperature, rng), False)
    winner = E.outcome(state)
    return score_records(records, winner), {"plies": state.turn, "winner": winner}


def eval_game(job):
    """(seed, new_side, budget, wall_limit, opening_plies, ply_cap, opponent) -> score of "new" (1 / 0.5 / 0).

    Both sides run the deployment alpha-beta (MSSV.py search); nets act only at the root.
    opponent "old" = the worker's "old" net; "base" = no network = plays exactly like submissions/MSSV.py."""
    seed, new_side, budget, wall_limit, opening_plies, ply_cap, opponent = job
    rng = random.Random(seed)
    players = {"new": (TorchPolicy(_NETS["new"]), TorchValue(_NETS["new"]))}
    players["opp"] = (None, None) if opponent == "base" else (TorchPolicy(_NETS[opponent]), TorchValue(_NETS[opponent]))
    state = random_opening(rng, opening_plies)
    while not finished(state, ply_cap):
        policy, value = players["new" if state.turn % 2 == new_side else "opp"]
        action = E.Search(budget=budget, wall_limit=wall_limit, policy=policy, value_net=value,
                          mode="learned_alphabeta").choose(state)
        state = E.play(state, action, False)
    winner = E.outcome(state)
    return 0.5 if winner in (None, 2) else float(winner == new_side)


def tagged_eval_game(job):
    """eval_game, returning (opponent, score) so unordered pool results can be grouped."""
    return job[6], eval_game(job)


def side_eval_game(job):
    """eval_game, returning (opponent, side of "new", score): per-side stats (first vs second)."""
    return job[6], job[1], eval_game(job)


def sample_policy(policy, state, rng, temperature=1.0):
    """A legal action drawn from the net's softmax (exploration move)."""
    logits = policy.logits(state)
    actions = E.legal_actions(state)
    top = max(logits[a] for a in actions)
    weights = [math.exp((logits[a] - top) / temperature) for a in actions]
    return rng.choices(actions, weights)[0]


def agent_selfplay_game(job):
    """(seed, params dict) -> (samples, stats). Expert iteration with the DEPLOYED search.

    Every move is chosen by the agent's own alpha-beta with the worker's "model" net at the
    root (exactly what submissions/az_agent.py plays); the target is that move (one-hot),
    the value target the game result. params["opponent"]: "self" = the same agent on both
    sides; "base" = the no-network engine (plays like submissions/MSSV.py) on the side that is
    not params["learner_side"]. Exploration: random pawn opening, then in the first
    explore_plies plies the played move is, with probability explore_eps, sampled from the
    policy instead (the search move stays the recorded target)."""
    seed, p = job
    rng = random.Random(seed)
    learner = (TorchPolicy(_NETS["model"]), TorchValue(_NETS["model"]))
    sides = [learner, learner]
    if p["opponent"] == "base":
        sides[1 - p["learner_side"]] = (None, None)
    state = random_opening(rng, p["opening_plies"])
    records = []
    while not finished(state, p["ply_cap"]):
        policy, value = sides[state.turn % 2]
        action = E.Search(budget=p["budget"], wall_limit=p["wall_limit"], policy=policy, value_net=value,
                          mode="learned_alphabeta").choose(state)
        pi = [0.0] * N_ACTIONS
        pi[canonical_action(state, action)] = 1.0
        records.append((E.encode(state), pi, state.turn % 2))
        if len(records) <= p["explore_plies"] and rng.random() < p["explore_eps"]:
            action = sample_policy(learner[0], state, rng)
        state = E.play(state, action, False)
    winner = E.outcome(state)
    return score_records(records, winner), {"plies": state.turn, "winner": winner,
                                            "opponent": p["opponent"], "learner_side": p["learner_side"]}
