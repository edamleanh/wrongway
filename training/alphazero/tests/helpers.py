"""Shared test helpers: random positions from the engine and from the instructor Arena."""
import random
import sys

from training.alphazero.engine import ARENA_DIR, E

if str(ARENA_DIR) not in sys.path:
    sys.path.insert(0, str(ARENA_DIR))
from environment import Environment  # noqa: E402


def random_states(n_games=10, seed=1, max_plies=80, wall_rate=0.3):
    """Engine States from random games with plenty of walls."""
    rng = random.Random(seed)
    for _ in range(n_games):
        state = E.State()
        while E.outcome(state) is None and state.turn < max_plies:
            yield state
            actions = E.legal_actions(state)
            walls = [a for a in actions if 81 <= a < E.PASS]
            moves = [a for a in actions if a < 81]
            pick = rng.choice(walls) if walls and rng.random() < wall_rate else rng.choice(moves)
            state = E.play(state, pick, False)


def arena_envs(n_games=25, seed=7, starting_player=0, max_turns=120):
    """Live Arena Environments at every turn of random games."""
    rng = random.Random(seed)
    for _ in range(n_games):
        env = Environment(max_turns=max_turns, starting_player=starting_player)
        while not env.done:
            yield env
            walls = env.legal_wall_actions()
            if walls and rng.random() < 0.35:
                env.step({"type": "wall", **rng.choice(walls)})
            else:
                env.step({"type": "move", "to": rng.choice(env.legal_moves())})


def arena_actions(obs):
    """Arena observation -> set of engine action indices (moves, walls, pass)."""
    actions = {E.action_index({"type": "move", "to": m}) for m in obs["legal_moves"]}
    actions |= {E.action_index(dict(w, type="wall")) for w in obs["legal_walls"]}
    return actions | {E.PASS}


def is_legal(obs, action):
    if action["type"] == "pass":
        return True
    if action["type"] == "move":
        return list(action["to"]) in obs["legal_moves"]
    wall = {k: action[k] for k in ("orientation", "row", "col")}
    return wall in obs["legal_walls"]
