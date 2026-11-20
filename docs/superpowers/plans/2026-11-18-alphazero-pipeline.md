# AlphaZero Training + Agent Extraction Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End-to-end, resumable pipeline that trains a policy+value network by AlphaZero self-play (CPU / CUDA / Apple MPS), distills the value head into a tiny net, exports a `.pth`, and embeds it into a single pure-Python agent file `wrongway-student/wrongway/submissions/az_agent.py` (fork of `agents/MSSV.py`).

**Architecture:** The agent file is the single source of truth for rules + `encode()`; training imports it, so train/inference features can never drift. Training code lives in `training/alphazero/` as small modules plus seven numbered phase scripts (`p01_bootstrap` … `p07_benchmark`), each refusing to run until the previous phase is marked done in the run's `manifest.json`. Game generation runs in CPU worker processes (pure-Python rules dominate cost); gradient steps run on the best device (`cuda` > `mps` > `cpu`). Everything is logged to TensorBoard under the run folder.

**Tech Stack:** Python 3.12 (training) / 3.9+ stdlib-only (agent), uv, PyTorch ≥ 2.3, TensorBoard, NumPy, pytest.

**Spec:** `docs/alphazero_pipeline.md` (copied into the worktree in Task 1 from the main checkout).

---

## Deviations from the spec (decided while planning — read first)

| # | Spec says | Plan does | Why |
|---|---|---|---|
| 1 | Policy head 211 logits | 210 logits (`ACTION_COUNT = 210`, actions 0–209) | The old `Student` packed 210 logits + 1 value into 211 outputs; the new value head is separate. |
| 2 | Policy target indexed by real action | Indexed in the **canonical frame** `transform_action(a, rotate)` | `encode()` rotates the board so the mover always heads for row 0; the policy output must use the same frame (exactly like the old `Student.predict`). |
| 3 | `pickle` of Python lists | `.npz` shards (float16 features/policy, float32 result) | 500k × 861 floats as pickled lists is several GB. |
| 4 | Policy “called once per turn at root” | Enforced: `Search.root_logits` computed once in `choose()`; non-root `ordered()` never calls the net | Old code called the model at every node. |
| 5 | “Any GPU” | Self-play / bootstrap / gate in CPU worker processes (torch single-thread); training + distill on `--set device=auto|cpu|cuda|mps` | Per-leaf batch-1 inference is faster on CPU than shipping to a GPU; rules are pure Python anyway. |
| 6 | Bootstrap budget 0.55 s, 5000 games | Preset `standard`: 0.10 s, 2000 games. Preset `spec`: spec values | 5000 × ~60 plies × 0.55 s ≈ 46 CPU-hours. |
| 7 | Distill on 10k fresh alpha-beta positions | Distill on the replay buffer (bootstrap alpha-beta games + MCTS games), up to `distill_positions` | Same distribution, zero extra generation cost, far more positions. |
| 8 | Gate with `Search(..., model=new_net)` | Gate with `Search(policy=TorchPolicy(net), value_net=TorchValue(net))` — same search code as deployment, full value head | Tiny net does not exist yet during the loop. |
| 9 | — | MCTS adds Dirichlet root noise (α 0.3, ε 0.25) and uses a fresh tree per move | Standard AlphaZero exploration; fresh tree keeps memory bounded and noise simple. |
| 10 | — | Games longer than `ply_cap` (200) are stopped and scored as draws | Avoids pass/shuffle games eating self-play time. |
| 11 | — | Blob stores float16 by default (`blob_dtype=e`), ~0.5 MB string | Halves submission size; parity test bounds the error. |
| 12 | — | Fork fixes the Arena decoder (walls come as `{"horizontal": [[r,c]], "vertical": [...]}`; player 1 may start) and drops `Student`/`puct` | `agents/MSSV.py` crashes on real Arena observations (`'str' object has no attribute 'get'`). |

---

## File structure

```
pyproject.toml                         # uv project for training only (torch, tensorboard, numpy, pytest)
.python-version                        # 3.12
docs/alphazero_pipeline.md             # spec (copied from main checkout)
training/__init__.py                   # makes training.alphazero importable
training/alphazero/
  README.md                            # how to run, phases table, TB tags, presets
  __init__.py
  engine.py        # loads submissions/az_agent.py as module E (rules + encode, single source of truth)
  config.py        # RunConfig dataclass, PRESETS, make_config(), pick_device()
  run_state.py     # RunContext: run folder, manifest.json, require()/mark_done(), TensorBoard writer
  symmetry.py      # canonical policy frame, left/right mirror permutations
  blob.py          # (stdlib) weights <-> base85 text blob, embed() into agent file
  net.py           # AZNet (651->256->128 | policy 210 | value 32->1), TinyValue (651->32->1), checkpoints
  adapters.py      # TorchPolicy / TorchValue: torch nets behind the agent's Search interface
  export.py        # .pth export, .pth -> blobs, embed_pth()
  embed_pth.py     # standalone CLI: any .pth -> any agent file
  data.py          # .npz shards, replay buffer
  mcts.py          # PUCT MCTS for self-play
  games.py         # bootstrap / self-play / gate games + process pool
  trainer.py       # policy+value training, value distillation, pearson
  pipeline.py      # run phases 1..7 (or a slice), --status
  phases/
    __init__.py
    _cli.py            # shared --run/--preset/--set/--force, dependency check
    p01_bootstrap.py   # alpha-beta teacher games          -> data/bootstrap.npz
    p02_pretrain.py    # needs p01                         -> checkpoints/iter000.pt
    p03_iterate.py     # needs p02: selfplay/train/gate xN -> data/selfplay_iterNNN.npz, checkpoints/iterNNN.pt
    p04_distill.py     # needs p03                         -> checkpoints/tiny_value.pt
    p05_export_pth.py  # needs p04                         -> export/az_agent.pth
    p06_embed.py       # needs p05                         -> submissions/az_agent.py (blobs embedded)
    p07_benchmark.py   # needs p06                         -> export/benchmark.txt
  tests/
    __init__.py
    helpers.py
    test_engine_parity.py  test_agent_inference.py  test_config.py  test_symmetry.py
    test_net.py  test_data.py  test_mcts.py  test_games.py  test_trainer.py  test_pipeline.py
  runs/<run>/      # gitignored: manifest.json, data/, checkpoints/, tb/, export/
wrongway-student/wrongway/submissions/az_agent.py   # fork of agents/MSSV.py (the deliverable)
```

Run any phase with `uv run python -m training.alphazero.phases.pNN_xxx --run NAME`, or all of them with `uv run python -m training.alphazero.pipeline --run NAME --preset quick`.

---

### Task 1: Project setup (fork, spec, uv, packages)

**Files:**
- Create: `wrongway-student/wrongway/submissions/az_agent.py` (copy of `/Users/henrytang/Projects/wrongway/agents/MSSV.py`)
- Create: `docs/alphazero_pipeline.md` (copy of `/Users/henrytang/Projects/wrongway/docs/alphazero_pipeline.md`)
- Create: `pyproject.toml`, `.python-version`, `training/__init__.py`, `training/alphazero/__init__.py`, `training/alphazero/phases/__init__.py`, `training/alphazero/tests/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Copy the fork base and the spec into the worktree**

```bash
cp /Users/henrytang/Projects/wrongway/agents/MSSV.py wrongway-student/wrongway/submissions/az_agent.py
mkdir -p docs training/alphazero/phases training/alphazero/tests
cp /Users/henrytang/Projects/wrongway/docs/alphazero_pipeline.md docs/alphazero_pipeline.md
touch training/__init__.py training/alphazero/__init__.py training/alphazero/phases/__init__.py training/alphazero/tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "wrongway-training"
version = "0.1.0"
description = "Offline training tools for the Wrongway agents (never imported by a submission)"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "torch>=2.3",
    "tensorboard>=2.16",
]

[dependency-groups]
dev = ["pytest>=8"]

[tool.uv]
package = false

[tool.pytest.ini_options]
testpaths = ["training/alphazero/tests"]
pythonpath = ["."]
markers = ["slow: end-to-end runs (deselect with -m 'not slow')"]
```

- [ ] **Step 3: Pin Python and ignore run folders**

```bash
echo "3.12" > .python-version
printf '\n# AlphaZero runs (data, checkpoints, tensorboard) and uv env\ntraining/alphazero/runs/\n.venv/\n' >> .gitignore
```

- [ ] **Step 4: Create the environment and check devices**

Run: `uv sync`
Then: `uv run python -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())"`
Expected (on this Mac): `2.x.x cuda False mps True`. On Linux+NVIDIA the PyPI wheel is a CUDA build (`cuda True`). On Windows+NVIDIA, PyPI torch is CPU-only — see README (Task 15) for the CUDA index.

- [ ] **Step 5: Verify the fork still plays a legal game (baseline, before any change)**

Run: `uv run python wrongway-student/wrongway/arena.py -p1 az_agent -p2 greedy`
Expected: the match runs; az_agent likely loses on time-outs/invalid or plays the fallback every move (decoder bug, fixed in Task 2). Note what you see; do not fix yet.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore docs/alphazero_pipeline.md training/__init__.py training/alphazero wrongway-student/wrongway/submissions/az_agent.py
git commit -m "chore: scaffold AlphaZero pipeline (fork az_agent from agents/MSSV.py, uv project)"
```

---

### Task 2: Engine loader + Arena rules parity (fix decoder)

**Files:**
- Create: `training/alphazero/engine.py`
- Create: `training/alphazero/tests/helpers.py`
- Create: `training/alphazero/tests/test_engine_parity.py`
- Modify: `wrongway-student/wrongway/submissions/az_agent.py` (`state_from_observation`)

- [ ] **Step 1: Write `engine.py`**

```python
"""Rules + feature encoding for training = the agent file itself (single source of truth).

Training never re-implements the game: it imports submissions/az_agent.py, so encode(),
legal_actions() and play() are byte-for-byte what the submitted agent runs."""
import importlib.util
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ARENA_DIR = PROJECT / "wrongway-student" / "wrongway"
SUBMISSIONS = ARENA_DIR / "submissions"
AGENT_NAME = "az_agent"
AGENT_FILE = SUBMISSIONS / (AGENT_NAME + ".py")


def load_agent(path=AGENT_FILE, name="az_engine"):
    """Import an agent .py file as a module (cached in sys.modules under `name`)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = load_agent()
```

- [ ] **Step 2: Write test helpers**

```python
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
```

- [ ] **Step 3: Write the failing parity tests**

```python
"""The forked agent's rules must match the instructor Arena exactly."""
import random

from training.alphazero.engine import E
from training.alphazero.tests.helpers import Environment, arena_actions, arena_envs


def test_legal_actions_match_arena():
    checked = 0
    for env in arena_envs():
        obs = env.get_observation(env.current_player)
        state = E.state_from_observation(obs)
        assert set(E.legal_actions(state)) == arena_actions(obs)
        checked += 1
    assert checked > 300


def test_distances_match_arena():
    for env in arena_envs(n_games=10):
        obs = env.get_observation(env.current_player)
        state = E.state_from_observation(obs)
        ours = E.distances(state.h, state.v, obs["you"]["goal_row"])
        assert list(ours) == [d for row in obs["distances"] for d in row]


def test_decoder_handles_player1_starting():
    for env in arena_envs(n_games=5, starting_player=1):
        obs = env.get_observation(env.current_player)
        state = E.state_from_observation(obs)
        me = state.p1 if state.turn % 2 else state.p0
        assert [me // 9, me % 9] == obs["you"]["position"]
        assert set(E.legal_actions(state)) == arena_actions(obs)


def test_play_matches_arena_step():
    rng = random.Random(3)
    for _ in range(10):
        env = Environment(max_turns=120)
        state = E.state_from_observation(env.get_observation(env.current_player))
        while not env.done:
            action = rng.choice(sorted(E.legal_actions(state)))
            env.step(E.action_dict(action))
            state = E.play(state, action, validate=False)
            if env.done:
                assert E.outcome(state) == (2 if env.winner is None else env.winner)
                break
            assert state == E.state_from_observation(env.get_observation(env.current_player))
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest training/alphazero/tests/test_engine_parity.py -v`
Expected: FAIL with `AttributeError: 'str' object has no attribute 'get'` (decoder iterates the walls dict's keys).

- [ ] **Step 5: Fix the decoder in `az_agent.py`**

Replace the whole `state_from_observation` function with:

```python
def _placed_walls(walls):
    """Arena gives {"horizontal": [[r, c], ...], "vertical": [...]}; a list of wall dicts also works."""
    if isinstance(walls, dict):
        for key, orientation in (("horizontal", "H"), ("vertical", "V")):
            for row, col in walls.get(key, ()):
                yield wall_action(orientation, row, col)
    else:
        for wall in walls:
            yield action_index(dict(wall, type="wall"))


def state_from_observation(observation):
    """Arena observation -> State. Player 0 is whoever heads for row 0."""
    you, opponent = observation["you"], observation["opponent"]
    own_goal, other_goal = you["goal_row"], opponent["goal_row"]
    if {own_goal, other_goal} != {0, 8}:
        raise ValueError("goal rows must be opposite board edges")
    player = 0 if own_goal == 0 else 1
    turn = observation.get("turn", observation.get("turn_count"))
    maximum = observation.get("max_turns", 500)
    if type(turn) is not int or turn < 0:
        raise ValueError("invalid turn counter")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("invalid turn limit")
    if turn % 2 != player:
        # Arena may let player 1 move first (starting_player=1). State assumes the side to
        # move is turn % 2, so shift by one ply and keep the same number of plies left.
        turn += 1
        maximum += 1
    h = v = 0
    for action in _placed_walls(observation["walls"]):
        bit = 1 << ((action - 81) % 64)
        if action < 145:
            h |= bit
        else:
            v |= bit
    first, second = (opponent, you) if player else (you, opponent)
    return State(_position(first), _position(second), _walls_left(first),
                 _walls_left(second), h, v, turn, maximum)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest training/alphazero/tests/test_engine_parity.py -v`
Expected: 4 passed. If `test_legal_actions_match_arena` still fails, the wall/jump rules differ from `environment.py` — use superpowers:systematic-debugging on the first mismatching observation before continuing.

- [ ] **Step 7: Verify in a real Arena match**

Run: `uv run python wrongway-student/wrongway/arena.py -p1 az_agent -p2 wall_builder`
Expected: no `invalid`/`error` turns for az_agent.

- [ ] **Step 8: Commit**

```bash
git add training/alphazero/engine.py training/alphazero/tests wrongway-student/wrongway/submissions/az_agent.py
git commit -m "fix(az_agent): decode Arena walls dict and player-1 start; add rules parity tests"
```

---

### Task 3: Pure-Python inference in the agent + weight blob format

**Files:**
- Create: `training/alphazero/blob.py`
- Create: `training/alphazero/tests/test_agent_inference.py`
- Modify: `wrongway-student/wrongway/submissions/az_agent.py` (constants, `Student` → `MLP`/`PolicyNet`/`TinyValueNet`, `Search`, `choose_action`)

- [ ] **Step 1: Write `blob.py` (stdlib only)**

```python
"""Network weights <-> a text blob that lives inside the single-file agent.

blob = base85( zlib( dtype byte + little-endian floats ) ),  dtype 'e' = float16, 'f' = float32.
Float order, layer by layer: weight rows in PyTorch layout [out][in] (row-major), then bias.
The agent's unpack_blob()/MLP read exactly this order."""
import base64
import re
import struct
import zlib
from pathlib import Path

BLOCK = re.compile(r"(# =+ MODEL BLOBS BEGIN[^\n]*\n)(.*?)(# =+ MODEL BLOBS END)", re.S)


def pack_layers(layers, dtype="e"):
    """layers: [(weight [out][in] nested lists, bias list), ...] -> blob string."""
    flat = []
    for weight, bias in layers:
        for row in weight:
            flat.extend(row)
        flat.extend(bias)
    if dtype == "e" and max(abs(x) for x in flat) > 60000.0:
        dtype = "f"                                    # float16 would overflow
    raw = dtype.encode("ascii") + struct.pack("<%d%s" % (len(flat), dtype), *flat)
    return base64.b85encode(zlib.compress(raw, 9)).decode("ascii")


def _literal(name, blob, width=4096):
    """Source for `name = "..."`, split in 4 KB chunks so editors cope (base85 has no quotes/backslashes)."""
    if not blob:
        return '%s = ""\n' % name
    chunks = "".join('    "%s"\n' % blob[i:i + width] for i in range(0, len(blob), width))
    return "%s = (\n%s)\n" % (name, chunks)


def embed(agent_path, model_blob, value_blob, mode="learned_alphabeta"):
    """Rewrite MODEL_BLOB / VALUE_BLOB / DEPLOY_MODE between the MODEL BLOBS markers."""
    path = Path(agent_path)
    text = path.read_text(encoding="utf-8")
    body = _literal("MODEL_BLOB", model_blob) + _literal("VALUE_BLOB", value_blob) + 'DEPLOY_MODE = "%s"\n' % mode
    new, count = BLOCK.subn(lambda m: m.group(1) + body + m.group(3), text)
    if count != 1:
        raise ValueError("MODEL BLOBS markers must appear exactly once in %s" % path)
    path.write_text(new, encoding="utf-8")
```

- [ ] **Step 2: Write the failing tests**

```python
"""Pure-Python inference inside the agent: blob decode, MLP math, Search wiring, embed."""
import math
import random
import shutil
import time

import pytest

from training.alphazero.blob import embed, pack_layers
from training.alphazero.engine import AGENT_FILE, E, load_agent
from training.alphazero.tests.helpers import Environment, is_legal, random_states


def random_layers(shapes, scale=0.05, seed=0):
    rng = random.Random(seed)
    return [([[rng.uniform(-scale, scale) for _ in range(i)] for _ in range(o)],
             [rng.uniform(-scale, scale) for _ in range(o)]) for i, o in shapes]


def dense_forward(layers, x):
    for index, (weight, bias) in enumerate(layers):
        x = [b + sum(w * v for w, v in zip(row, x)) for row, b in zip(weight, bias)]
        if index < len(layers) - 1:
            x = [max(0.0, v) for v in x]
    return x


def test_value_net_matches_dense_reference():
    layers = random_layers(E.VALUE_SHAPES, scale=0.1)
    net = E.TinyValueNet(pack_layers(layers, "f"))
    for state in list(random_states(n_games=3))[:30]:
        x = E.encode(state)
        expected = dense_forward(layers, x)[0]
        assert abs(net.forward(x)[0] - expected) < 1e-4
        assert abs(net.value(state) - math.tanh(expected)) < 1e-4


def test_policy_logits_are_in_real_action_frame():
    layers = random_layers(E.POLICY_SHAPES, scale=0.05, seed=1)
    net = E.PolicyNet(pack_layers(layers, "e"))
    state = E.State(turn=1)                       # player 1 to move -> canonical frame is rotated
    out = dense_forward(layers, E.encode(state))
    logits = net.logits(state)
    assert len(logits) == E.ACTION_COUNT
    for action in range(E.ACTION_COUNT):
        assert abs(logits[action] - out[E.transform_action(action, True)]) < 5e-2


def test_blob_with_wrong_shape_is_rejected():
    blob = pack_layers(random_layers(E.VALUE_SHAPES), "f")
    with pytest.raises(ValueError):
        E.PolicyNet(blob)


class CountingPolicy:
    def __init__(self):
        self.calls = 0

    def logits(self, state):
        self.calls += 1
        return [0.0] * E.ACTION_COUNT


class CountingValue:
    def __init__(self):
        self.calls = 0

    def value(self, state):
        self.calls += 1
        return 0.0


def test_search_uses_policy_once_and_value_at_leaves():
    policy, value = CountingPolicy(), CountingValue()
    state = E.State()
    action = E.Search(budget=0.3, policy=policy, value_net=value, mode="learned_alphabeta").choose(state)
    assert action in E.legal_actions(state)
    assert policy.calls == 1
    assert value.calls > 0


def test_embed_then_play_in_arena(tmp_path):
    agent = tmp_path / "agent_copy.py"
    shutil.copy(AGENT_FILE, agent)
    embed(agent, pack_layers(random_layers(E.POLICY_SHAPES, 0.05, 2), "e"),
          pack_layers(random_layers(E.VALUE_SHAPES, 0.1, 3), "e"))
    module = load_agent(agent, name="agent_copy_embedded")
    assert module.DEPLOY_MODE == "learned_alphabeta"
    assert module._POLICY is not None and module._VALUE is not None
    env = Environment()
    for _ in range(6):
        obs = env.get_observation(env.current_player)
        started = time.perf_counter()
        action = module.choose_action(obs)
        assert time.perf_counter() - started < 1.0
        assert is_legal(obs, action)
        env.step(action)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest training/alphazero/tests/test_agent_inference.py -v`
Expected: FAIL with `AttributeError: module 'az_engine' has no attribute 'VALUE_SHAPES'`.

- [ ] **Step 4: Replace the agent's header constants**

In `az_agent.py` replace the block from `DEFAULT_BUDGET = 0.70` through `MODEL_BLOB = ""  # Filled by tools/export_agent.py; no external model file is needed.` with:

```python
DEFAULT_BUDGET = 0.55       # seconds per move (limit 1 s; grading Colab CPU is ~2x slower than a laptop)
DEPLOY_WALL_LIMIT = 16
EVAL_WEIGHTS = (0.35, 0.08, 0.10, 0.025, 0.15, 0.12)
POLICY_SHAPES = ((651, 256), (256, 128), (128, 210))   # backbone + policy head (AlphaZero net)
VALUE_SHAPES = ((651, 32), (32, 1))                   # tiny value net distilled from the value head
POLICY_WEIGHT = 0.20        # how much root logits (clipped to +-10) shift root move ordering
VALUE_SCALE = 12.0          # tanh value (-1..1) -> pawn-ply units, the scale heuristic() uses

# ==== MODEL BLOBS BEGIN (written by training/alphazero/phases/p06_embed.py - do not edit by hand) ====
MODEL_BLOB = ""
VALUE_BLOB = ""
DEPLOY_MODE = "alphabeta"
# ==== MODEL BLOBS END ====
```

- [ ] **Step 5: Replace `masked_policy` and `class Student` with the pure-Python nets**

Delete the functions/classes `masked_policy` and `Student` and insert:

```python
def unpack_blob(blob, shapes):
    """Text blob -> flat float tuple (format: training/alphazero/blob.py).

    base85(zlib(dtype byte 'e' float16 | 'f' float32, then little-endian floats));
    per layer: weight rows [out][in], then bias."""
    raw = zlib.decompress(base64.b85decode(blob.encode("ascii")))
    code = chr(raw[0])
    if code not in ("e", "f"):
        raise ValueError("unknown weight dtype")
    count = sum(inputs * outputs + outputs for inputs, outputs in shapes)
    if len(raw) - 1 != count * struct.calcsize("<" + code):
        raise ValueError("weight count does not match architecture")
    return struct.unpack("<%d%s" % (count, code), raw[1:])


class MLP:
    """Fully connected net in plain Python: ReLU after every layer except the last.

    The first layer is stored column by column, so a forward pass only visits the
    non-zero inputs: encode() sets roughly 190 of its 651 features."""
    SHAPES = ()

    def __init__(self, blob):
        weights = unpack_blob(blob, self.SHAPES)
        cursor, self.layers = 0, []
        for inputs, outputs in self.SHAPES:
            rows = [weights[cursor + o * inputs:cursor + (o + 1) * inputs] for o in range(outputs)]
            cursor += inputs * outputs
            self.layers.append((rows, weights[cursor:cursor + outputs]))
            cursor += outputs
        first_rows, first_bias = self.layers[0]
        self.columns = list(zip(*first_rows))   # columns[i]: weights from input i to every hidden unit
        self.first_bias = list(first_bias)
        self.layers[0] = (None, None)           # rows no longer needed

    def forward(self, inputs):
        hidden = self.first_bias
        for x, column in zip(inputs, self.columns):
            if x:
                hidden = [h + x * w for h, w in zip(hidden, column)]
        values = [h if h > 0.0 else 0.0 for h in hidden]
        last = len(self.layers) - 1
        for index in range(1, last + 1):
            rows, bias = self.layers[index]
            values = [b + sum(w * v for w, v in zip(row, values)) for row, b in zip(rows, bias)]
            if index < last:
                values = [v if v > 0.0 else 0.0 for v in values]
        return values


class PolicyNet(MLP):
    """Backbone + policy head. Outputs live in the canonical (rotated) frame of encode()."""
    SHAPES = POLICY_SHAPES

    def logits(self, state):
        out = self.forward(encode(state))
        rotate = (state.goal1 if state.turn % 2 else state.goal0) == 8
        return [out[transform_action(a, rotate)] for a in range(ACTION_COUNT)]


class TinyValueNet(MLP):
    """651 -> 32 -> 1 value net; +1 = side to move wins."""
    SHAPES = VALUE_SHAPES

    def value(self, state):
        return math.tanh(self.forward(encode(state))[0])
```

- [ ] **Step 6: Rewire `Search` to `policy` (root only) + `value_net` (leaves)**

Make these exact edits inside `class Search`:

(a) Replace `__init__`:

```python
    def __init__(self, budget=DEFAULT_BUDGET, wall_limit=16, policy=None, value_net=None, mode="alphabeta"):
        self.budget = budget
        self.wall_limit = wall_limit
        self.policy = policy          # .logits(state) -> 210 floats; called once per turn at the root
        self.value_net = value_net    # .value(state) -> -1..1 for the side to move; called at leaves
        self.mode = mode
        self.root_logits = None
        self.nodes = 0
        self.depth = 0
        self.deadline = 0.0
        self.completed = False
```

(b) Replace `evaluate` and delete `prediction`:

```python
    def evaluate(self, state):
        if outcome(state) is not None:
            return heuristic(state)
        goal = state.goal1 if state.turn % 2 else state.goal0
        if any(a // 9 == goal for a in pawn_moves(state)):
            return 9999.0
        if self.value_net is None:
            return heuristic(state)
        key = ("v", id(self.value_net), state)
        value = CACHE.get(key)
        if value is None:
            value = self.value_net.value(state) * VALUE_SCALE
            CACHE.put(key, value)
        return value
```

(c) In `ordered`, replace `logits = self.prediction(state)[0] if self.model else None` with:

```python
        logits = self.root_logits if root_actions is not None else None   # policy only at the root
```

and replace both `0.20 * max(-10.0, min(10.0, logits[action]))` and `0.2 * max(-10.0, min(10.0, logits[action]))` with `POLICY_WEIGHT * max(-10.0, min(10.0, logits[action]))`.

(d) In `negamax`, replace `key = ("t", self.mode, self.wall_limit, id(self.model), state)` with:

```python
        key = ("t", self.mode, self.wall_limit, id(self.value_net), state)
```

(e) In `choose`, right after `if outcome(state) is not None: return PASS` add:

```python
        self.root_logits = None
        if root_actions is None:
            root_actions = legal_actions(state)   # training calls choose(state) without Arena actions
```

and inside the `try:` block, immediately before `ordered = self.ordered(state, root_actions, best)`, add:

```python
            if self.policy is not None:
                self.root_logits = self.policy.logits(state)
                self.check()
```

and delete the two lines `if self.mode in ("puct", "heuristic_puct"):` / `return self.puct(state, ordered)`.

(f) Delete the whole `puct` method.

- [ ] **Step 7: Load the nets at import and pass them to `Search`**

Replace `_MODEL = None` and the body of `choose_action` with:

```python
def _load(cls, blob):
    """Decode an embedded net once at import (import limit 5 s; the 1 s move clock is untouched)."""
    try:
        return cls(blob) if blob else None
    except (ValueError, zlib.error, struct.error):
        return None


_POLICY = _load(PolicyNet, MODEL_BLOB)
_VALUE = _load(TinyValueNet, VALUE_BLOB)


def choose_action(observation):
    """Arena entry point. Only return actions supplied as legal by Arena."""
    started = time.monotonic()
    moves = observation.get("legal_moves", [])
    fallback = {"type": "pass"}
    if moves:
        ds = observation.get("distances")
        def move_key(move):
            row, col = move
            distance = ds[row][col] if ds is not None else abs(row - observation["you"]["goal_row"])
            return (distance if distance >= 0 else 1000, abs(col - 4), row, col)
        fallback = {"type": "move", "to": list(min(moves, key=move_key))}
    try:
        actions = [action_index({"type": "move", "to": move}) for move in moves]
        actions.extend(action_index(dict(wall, type="wall")) for wall in observation.get("legal_walls", []))
        actions.append(PASS)
        state = state_from_observation(observation)
        budget = max(0.0, DEFAULT_BUDGET - (time.monotonic() - started))
        learned = DEPLOY_MODE == "learned_alphabeta"
        search = Search(budget, DEPLOY_WALL_LIMIT, _POLICY if learned else None,
                        _VALUE if learned else None, DEPLOY_MODE)
        action = search.choose(state, actions)
        return action_dict(action) if action in actions else fallback
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OverflowError, zlib.error, struct.error):
        # An unexpected observation loses strategic strength, not the turn.
        return fallback
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest training/alphazero/tests/test_agent_inference.py training/alphazero/tests/test_engine_parity.py -v`
Expected: all pass.

- [ ] **Step 9: Measure pure-Python inference cost (record in commit message)**

```bash
uv run python -c "
import time
from training.alphazero.engine import E
from training.alphazero.blob import pack_layers
from training.alphazero.tests.test_agent_inference import random_layers
p = E.PolicyNet(pack_layers(random_layers(E.POLICY_SHAPES), 'e'))
v = E.TinyValueNet(pack_layers(random_layers(E.VALUE_SHAPES), 'e'))
s = E.State()
t = time.perf_counter(); [p.logits(s) for _ in range(20)]; print('policy ms', (time.perf_counter()-t)/20*1000)
t = time.perf_counter(); [v.value(s) for _ in range(200)]; print('value ms', (time.perf_counter()-t)/200*1000)
"
```
Expected on an M-series laptop: policy roughly 5–30 ms, value roughly 0.5–2 ms. If value > 3 ms, the leaf budget is too small — stop and report before continuing.

- [ ] **Step 10: Commit**

```bash
git add training/alphazero/blob.py training/alphazero/tests/test_agent_inference.py wrongway-student/wrongway/submissions/az_agent.py
git commit -m "feat(az_agent): pure-Python policy/value nets from embedded blobs; policy at root only"
```

---

### Task 4: Run config, presets, device choice, run folder + manifest

**Files:**
- Create: `training/alphazero/config.py`
- Create: `training/alphazero/run_state.py`
- Test: `training/alphazero/tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest

from training.alphazero.config import make_config, pick_device
from training.alphazero.run_state import RunContext


def test_presets_and_overrides():
    cfg = make_config("smoke", overrides=["simulations=7", "mirror_augment=false", "boot_budget=0.5"])
    assert cfg.simulations == 7
    assert cfg.mirror_augment is False
    assert cfg.boot_budget == 0.5
    assert cfg.iterations == 1                     # from the smoke preset


def test_unknown_override_key_is_an_error():
    with pytest.raises(ValueError, match="nope"):
        make_config(overrides=["nope=1"])


def test_pick_device():
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in ("cpu", "cuda", "mps")


def test_run_context_persists_and_enforces_order(tmp_path):
    ctx = RunContext("t", preset="smoke", runs_dir=tmp_path)
    for sub in ("data", "checkpoints", "tb", "export"):
        assert (tmp_path / "t" / sub).is_dir()
    with pytest.raises(SystemExit, match="p01_bootstrap"):
        ctx.require("p01")
    ctx.mark_done("p01")
    ctx.mark_done("p02")
    again = RunContext("t", runs_dir=tmp_path, overrides=["simulations=9"])
    assert again.is_done("p02") and again.cfg.simulations == 9 and again.cfg.iterations == 1
    again.mark_done("p01")                         # redoing an early phase makes later ones stale
    assert not again.is_done("p02")


def test_existing_run_rejects_new_preset(tmp_path):
    RunContext("t", preset="smoke", runs_dir=tmp_path)
    with pytest.raises(SystemExit, match="already exists"):
        RunContext("t", preset="quick", runs_dir=tmp_path)


def test_agent_path_default_and_override(tmp_path):
    ctx = RunContext("t", preset="smoke", runs_dir=tmp_path)
    assert ctx.agent_path.name == "az_agent.py"
    other = RunContext("u", preset="smoke", runs_dir=tmp_path, overrides=["agent_path=%s" % (tmp_path / "x.py")])
    assert other.agent_path == Path(tmp_path / "x.py")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest training/alphazero/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.alphazero.config'`.

- [ ] **Step 3: Write `config.py`**

```python
"""Every knob of the pipeline in one dataclass, plus named presets.

A run stores its resolved config in manifest.json; change it later with --set key=value."""
import os
from dataclasses import dataclass, field, fields


def _default_workers():
    return max(1, (os.cpu_count() or 2) - 1)


@dataclass
class RunConfig:
    seed: int = 0
    device: str = "auto"              # auto | cpu | cuda | mps  (training + distill only)
    workers: int = field(default_factory=_default_workers)   # CPU processes for games
    agent_path: str = ""              # "" -> wrongway-student/wrongway/submissions/az_agent.py
    # p01 bootstrap: the alpha-beta bot plays itself
    boot_games: int = 2000
    boot_budget: float = 0.10
    opening_plies: int = 4            # 0..N random pawn moves at the start of every game
    ply_cap: int = 200                # longer games stop and count as draws
    wall_limit: int = 16
    # p02 / p03 training
    batch_size: int = 512
    lr: float = 1e-3
    lr_min: float = 1e-4
    weight_decay: float = 1e-4
    pretrain_epochs: int = 5
    epochs_per_iter: int = 10
    buffer_size: int = 500_000
    mirror_augment: bool = True
    log_every: int = 50
    # p03 self-play
    iterations: int = 20
    sp_games: int = 200
    simulations: int = 100
    c_puct: float = 1.5
    temp_moves: int = 10
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    # p03 gate
    eval_games: int = 40
    eval_budget: float = 0.10
    accept_score: float = 0.55
    retry_lr_factor: float = 0.5
    # p04 distill
    distill_positions: int = 200_000
    distill_epochs: int = 50
    distill_val_frac: float = 0.05
    distill_min_pearson: float = 0.90
    # p06 embed
    blob_dtype: str = "e"             # e = float16 (half size), f = float32
    # p07 benchmark
    bench_pairs: int = 40
    bench_opponents: str = "MSSV,training/opponents/v2_trained.py"
    bench_workers: int = 4


PRESETS = {
    "standard": {},
    "spec": {"boot_games": 5000, "boot_budget": 0.55, "eval_budget": 0.55, "distill_positions": 500_000},
    "quick": {"boot_games": 400, "iterations": 5, "sp_games": 60, "simulations": 48,
              "epochs_per_iter": 4, "eval_games": 20, "distill_epochs": 15, "bench_pairs": 10},
    "smoke": {"workers": 2, "boot_games": 2, "boot_budget": 0.02, "ply_cap": 30, "batch_size": 32,
              "pretrain_epochs": 1, "epochs_per_iter": 1, "iterations": 1, "sp_games": 2,
              "simulations": 4, "eval_games": 2, "eval_budget": 0.02, "distill_positions": 2000,
              "distill_epochs": 1, "distill_min_pearson": 0.0, "bench_pairs": 1, "bench_workers": 2,
              "log_every": 1},
}


def _parse(current, raw):
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return type(current)(raw)


def make_config(preset="standard", base=None, overrides=()):
    """base: dict saved in a manifest (wins over preset). overrides: ["key=value", ...]."""
    if preset not in PRESETS:
        raise ValueError("unknown preset %r (choose: %s)" % (preset, ", ".join(PRESETS)))
    values = dict(base) if base is not None else dict(PRESETS[preset])
    cfg = RunConfig(**values)
    names = {f.name for f in fields(RunConfig)}
    for item in overrides:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or key not in names:
            raise ValueError("bad override %r (unknown key %r)" % (item, key))
        setattr(cfg, key, _parse(getattr(cfg, key), raw))
    return cfg


def pick_device(name="auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- [ ] **Step 4: Write `run_state.py`**

```python
"""One training run = one folder under training/alphazero/runs/<name>/.

manifest.json  config + progress (phases done, current iteration, best checkpoint, history)
data/          .npz sample shards        checkpoints/  .pt files
tb/            TensorBoard events        export/       .pth + benchmark report"""
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from .config import make_config
from .engine import AGENT_FILE

RUNS_DIR = Path(os.environ.get("AZ_RUNS_DIR", Path(__file__).resolve().parent / "runs"))
PHASE_MODULES = {
    "p01": "p01_bootstrap", "p02": "p02_pretrain", "p03": "p03_iterate", "p04": "p04_distill",
    "p05": "p05_export_pth", "p06": "p06_embed", "p07": "p07_benchmark",
}
PHASE_ORDER = list(PHASE_MODULES)


class RunContext:
    def __init__(self, name, preset=None, overrides=(), runs_dir=None):
        self.name = name
        self.dir = Path(runs_dir) if runs_dir else RUNS_DIR
        self.dir = self.dir / name
        self.manifest_path = self.dir / "manifest.json"
        if self.manifest_path.exists():
            if preset:
                raise SystemExit("run '%s' already exists; drop --preset (use --set) or pick a new --run" % name)
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.cfg = make_config(base=self.manifest["config"], overrides=overrides)
        else:
            self.cfg = make_config(preset or "standard", overrides=overrides)
            self.manifest = {"preset": preset or "standard", "done": {}, "iteration": 0,
                             "best": None, "history": [], "steps": {}}
        self.manifest["config"] = asdict(self.cfg)
        for sub in ("data", "checkpoints", "tb", "export"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.save()

    def path(self, *parts):
        return self.dir.joinpath(*parts)

    @property
    def agent_path(self):
        return Path(self.cfg.agent_path) if self.cfg.agent_path else AGENT_FILE

    def save(self):
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    def is_done(self, phase):
        return phase in self.manifest["done"]

    def require(self, phase):
        if phase and not self.is_done(phase):
            raise SystemExit("[%s] phase %s must finish first:\n  uv run python -m training.alphazero.phases.%s --run %s"
                             % (self.name, phase, PHASE_MODULES[phase], self.name))

    def mark_done(self, phase):
        for later in PHASE_ORDER[PHASE_ORDER.index(phase) + 1:]:
            self.manifest["done"].pop(later, None)       # built on old outputs -> stale
        self.manifest["done"][phase] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def writer(self):
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(self.path("tb")))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest training/alphazero/tests/test_config.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add training/alphazero/config.py training/alphazero/run_state.py training/alphazero/tests/test_config.py
git commit -m "feat(alphazero): run config presets, device pick, manifest with phase ordering"
```

---

### Task 5: Canonical frame + mirror symmetry

**Files:**
- Create: `training/alphazero/symmetry.py`
- Test: `training/alphazero/tests/test_symmetry.py`

- [ ] **Step 1: Write the failing tests**

```python
from training.alphazero.engine import E
from training.alphazero.symmetry import (FEATURE_MIRROR, N_ACTIONS, POLICY_MIRROR,
                                         canonical_action, canonical_policy, rotate_for)
from training.alphazero.tests.helpers import random_states


def test_feature_mirror_matches_encode_mirror():
    for state in list(random_states(n_games=4, seed=5))[:120]:
        plain = E.encode(state)
        assert [plain[i] for i in FEATURE_MIRROR] == E.encode(state, mirror=True)


def test_policy_mirror_is_an_involution():
    assert sorted(POLICY_MIRROR) == list(range(N_ACTIONS))
    assert all(POLICY_MIRROR[POLICY_MIRROR[a]] == a for a in range(N_ACTIONS))


def test_canonical_policy_puts_mass_on_canonical_index():
    state = E.State(turn=1)                  # player 1 -> rotated
    assert rotate_for(state)
    pi = canonical_policy(state, {4: 3, E.PASS: 1})
    assert abs(sum(pi) - 1.0) < 1e-9
    assert pi[canonical_action(state, 4)] == 0.75
    assert pi[E.PASS] == 0.25


def test_mirrored_policy_lines_up_with_mirrored_encoding():
    state = E.State(turn=1)
    for action in E.legal_actions(state):
        j = canonical_action(state, action)
        assert POLICY_MIRROR[j] == E.transform_action(action, True, True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest training/alphazero/tests/test_symmetry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.alphazero.symmetry'`.

- [ ] **Step 3: Write `symmetry.py`**

```python
"""Canonical action frame and left/right mirror symmetry.

encode() rotates the board so the side to move always heads for row 0. Network policy
outputs live in that frame: output j <-> real action transform_action(j, rotate).
Mirroring columns is an exact symmetry of Wrongway, so every sample can be flipped for
free data augmentation: x_mirror = x[FEATURE_MIRROR], pi_mirror = pi[POLICY_MIRROR]."""
from .engine import E

N_FEATURES = 651
N_ACTIONS = E.ACTION_COUNT           # 210 = 81 moves + 64 H walls + 64 V walls + pass


def rotate_for(state):
    return (state.goal1 if state.turn % 2 else state.goal0) == 8


def canonical_action(state, action):
    return E.transform_action(action, rotate_for(state))


def canonical_policy(state, visits):
    """{real action: count} -> length-210 distribution in the canonical frame."""
    pi = [0.0] * N_ACTIONS
    total = float(sum(visits.values()))
    if total > 0:
        for action, count in visits.items():
            pi[canonical_action(state, action)] = count / total
    return pi


def _feature_mirror():
    perm = list(range(N_FEATURES))
    for offset in (0, 81, 324, 405, 486, 567):       # 9x9 cell planes: pawns, distances, goal rows
        for cell in range(81):
            perm[offset + cell] = offset + E.transform_cell(cell, False, True)
    for offset in (162, 243):                         # wall planes: slot (row, col) stored at row*9+col
        for row in range(8):
            for col in range(8):
                perm[offset + row * 9 + col] = offset + row * 9 + (7 - col)
    return perm                                       # 648..650 (scalars) stay put


FEATURE_MIRROR = _feature_mirror()
POLICY_MIRROR = [E.transform_action(a, False, True) for a in range(N_ACTIONS)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest training/alphazero/tests/test_symmetry.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add training/alphazero/symmetry.py training/alphazero/tests/test_symmetry.py
git commit -m "feat(alphazero): canonical policy frame and mirror permutations"
```

---

### Task 6: Torch networks, adapters, `.pth` export, torch ↔ pure-Python parity

**Files:**
- Create: `training/alphazero/net.py`, `training/alphazero/adapters.py`, `training/alphazero/export.py`, `training/alphazero/embed_pth.py`
- Test: `training/alphazero/tests/test_net.py`

- [ ] **Step 1: Write the failing tests**

```python
import shutil

import torch

from training.alphazero.adapters import TorchPolicy, TorchValue
from training.alphazero.blob import pack_layers
from training.alphazero.engine import AGENT_FILE, E, load_agent
from training.alphazero.export import blobs_from_pth, embed_pth, export_pth, layers_from_linears
from training.alphazero.net import AZNet, TinyValue, load_checkpoint, policy_layers, save_checkpoint, tiny_layers
from training.alphazero.tests.helpers import random_states

STATES = None


def states():
    global STATES
    if STATES is None:
        STATES = list(random_states(n_games=3, seed=11))[:25]
    return STATES


def test_forward_shapes():
    logits, value = AZNet()(torch.zeros(4, 651))
    assert logits.shape == (4, 210) and value.shape == (4,)
    assert TinyValue()(torch.zeros(3, 651)).shape == (3,)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = AZNet()
    save_checkpoint(tmp_path / "a.pt", net, 3)
    back = load_checkpoint(tmp_path / "a.pt")
    x = torch.rand(2, 651)
    assert torch.allclose(net(x)[0], back(x)[0]) and torch.allclose(net(x)[1], back(x)[1])
    assert torch.load(tmp_path / "a.pt", weights_only=True)["iter"] == 3


def test_pure_python_policy_matches_torch():
    torch.manual_seed(1)
    net = AZNet()
    py = E.PolicyNet(pack_layers(layers_from_linears(policy_layers(net)), "f"))
    tp = TorchPolicy(net)
    for state in states():
        assert max(abs(a - b) for a, b in zip(py.logits(state), tp.logits(state))) < 1e-4


def test_pure_python_tiny_value_matches_torch():
    torch.manual_seed(2)
    tiny = TinyValue()
    py = E.TinyValueNet(pack_layers(layers_from_linears(tiny_layers(tiny)), "f"))
    tv = TorchValue(tiny)
    for state in states():
        assert abs(py.value(state) - tv.value(state)) < 1e-5


def test_pth_export_and_embed(tmp_path):
    torch.manual_seed(3)
    net, tiny = AZNet(), TinyValue()
    pth = tmp_path / "w.pth"
    export_pth(pth, net, tiny, {"note": "test"})
    model_blob, value_blob = blobs_from_pth(pth, "e")
    assert E.PolicyNet(model_blob) and E.TinyValueNet(value_blob)
    agent = tmp_path / "agent.py"
    shutil.copy(AGENT_FILE, agent)
    embed_pth(pth, agent, "e")
    module = load_agent(agent, name="agent_pth_embedded")
    tp, tv = TorchPolicy(net), TorchValue(tiny)
    for state in states()[:10]:
        assert max(abs(a - b) for a, b in zip(module._POLICY.logits(state), tp.logits(state))) < 5e-2
        assert abs(module._VALUE.value(state) - tv.value(state)) < 2e-2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest training/alphazero/tests/test_net.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.alphazero.adapters'`.

- [ ] **Step 3: Write `net.py`**

```python
"""PyTorch networks (training only; the agent runs the same math in plain Python).

AZNet:     651 -> 256 -> 128 backbone, policy head 128 -> 210 logits,
           value head 128 -> 32 -> 1 tanh (+1 = side to move wins)
TinyValue: 651 -> 32 -> 1 tanh, distilled from AZNet's value head for fast leaf eval."""
import torch
from torch import nn

N_FEATURES, N_ACTIONS = 651, 210


class AZNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(N_FEATURES, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.policy = nn.Linear(128, N_ACTIONS)
        self.value = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x):
        h = self.backbone(x)
        return self.policy(h), self.value(h).squeeze(-1)


class TinyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEATURES, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x):
        return self.net(x).squeeze(-1)


def policy_layers(net):
    """Linear layers the agent's PolicyNet runs, in order (matches POLICY_SHAPES)."""
    return [net.backbone[0], net.backbone[2], net.policy]


def tiny_layers(tiny):
    """Linear layers the agent's TinyValueNet runs, in order (matches VALUE_SHAPES)."""
    return [tiny.net[0], tiny.net[2]]


def save_checkpoint(path, net, iteration):
    torch.save({"backbone": net.backbone.state_dict(), "policy": net.policy.state_dict(),
                "value": net.value.state_dict(), "iter": iteration}, str(path))


def load_checkpoint(path, device="cpu"):
    data = torch.load(str(path), map_location=device, weights_only=True)
    net = AZNet()
    net.backbone.load_state_dict(data["backbone"])
    net.policy.load_state_dict(data["policy"])
    net.value.load_state_dict(data["value"])
    return net.to(device).eval()
```

- [ ] **Step 4: Write `adapters.py`**

```python
"""Torch nets behind the agent's Search interface (.logits(state) / .value(state)),
so gating games run the exact deployment search code with full-precision nets."""
import torch

from .engine import E
from .symmetry import N_ACTIONS, rotate_for


def _features(state):
    return torch.tensor([E.encode(state)], dtype=torch.float32)


class TorchPolicy:
    def __init__(self, net):
        self.net = net.eval()

    @torch.no_grad()
    def logits(self, state):
        out = self.net(_features(state))[0][0].tolist()
        rotate = rotate_for(state)
        return [out[E.transform_action(a, rotate)] for a in range(N_ACTIONS)]


class TorchValue:
    """Works with AZNet (uses its value head) or TinyValue."""
    def __init__(self, module):
        self.module = module.eval()

    @torch.no_grad()
    def value(self, state):
        out = self.module(_features(state))
        if isinstance(out, tuple):
            out = out[1]
        return float(out[0])
```

- [ ] **Step 5: Write `export.py` and `embed_pth.py`**

`training/alphazero/export.py`:

```python
"""Inference weights: torch nets -> .pth -> text blobs embedded in the agent file.

The .pth is the hand-off artifact: plain tensors in agent order, no torch module code needed."""
import torch

from .blob import embed, pack_layers
from .engine import E
from .net import policy_layers, tiny_layers

PTH_FORMAT = "wrongway-az-v1"


def layers_from_linears(linears):
    return [(l.weight.detach().cpu().float().tolist(), l.bias.detach().cpu().float().tolist()) for l in linears]


def _tensors(linears):
    out = []
    for layer in linears:
        out.append(layer.weight.detach().cpu().float().clone())
        out.append(layer.bias.detach().cpu().float().clone())
    return out


def export_pth(path, net, tiny, meta):
    torch.save({"format": PTH_FORMAT,
                "policy_shapes": [list(s) for s in E.POLICY_SHAPES],
                "value_shapes": [list(s) for s in E.VALUE_SHAPES],
                "policy": _tensors(policy_layers(net)),
                "value": _tensors(tiny_layers(tiny)),
                "meta": meta}, str(path))


def blobs_from_pth(path, dtype="e"):
    data = torch.load(str(path), map_location="cpu", weights_only=True)
    if data.get("format") != PTH_FORMAT:
        raise ValueError("%s is not a %s file" % (path, PTH_FORMAT))
    if [tuple(s) for s in data["policy_shapes"]] != list(E.POLICY_SHAPES) or \
            [tuple(s) for s in data["value_shapes"]] != list(E.VALUE_SHAPES):
        raise ValueError("layer shapes in %s do not match the agent" % path)

    def layers(tensors):
        return [(tensors[i].tolist(), tensors[i + 1].tolist()) for i in range(0, len(tensors), 2)]

    return pack_layers(layers(data["policy"]), dtype), pack_layers(layers(data["value"]), dtype)


def embed_pth(pth_path, agent_path, dtype="e"):
    model_blob, value_blob = blobs_from_pth(pth_path, dtype)
    embed(agent_path, model_blob, value_blob)
    return len(model_blob) + len(value_blob)
```

`training/alphazero/embed_pth.py`:

```python
"""Embed any exported .pth into any agent file (outside the phase pipeline).

  uv run python -m training.alphazero.embed_pth --pth runs/main/export/az_agent.pth \
      --agent wrongway-student/wrongway/submissions/az_agent.py"""
import argparse

from .export import embed_pth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--dtype", default="e", choices=("e", "f"))
    a = ap.parse_args()
    size = embed_pth(a.pth, a.agent, a.dtype)
    print("embedded %d blob chars into %s" % (size, a.agent))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest training/alphazero/tests/test_net.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add training/alphazero/net.py training/alphazero/adapters.py training/alphazero/export.py training/alphazero/embed_pth.py training/alphazero/tests/test_net.py
git commit -m "feat(alphazero): torch nets, .pth export, embed CLI, torch/pure-Python parity tests"
```

---

### Task 7: Sample shards + replay buffer

**Files:**
- Create: `training/alphazero/data.py`
- Test: `training/alphazero/tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

```python
import os

import pytest

from training.alphazero.data import load_buffer, load_shard, save_shard


def sample(z):
    return ([0.0] * 650 + [1.0], [1.0] + [0.0] * 209, z)


def test_shard_roundtrip(tmp_path):
    save_shard(tmp_path / "a.npz", [sample(1.0), sample(-1.0)])
    x, pi, z = load_shard(tmp_path / "a.npz")
    assert x.shape == (2, 651) and pi.shape == (2, 210) and list(z) == [1.0, -1.0]


def test_buffer_keeps_newest_positions(tmp_path):
    save_shard(tmp_path / "old.npz", [sample(1.0), sample(1.0)])
    save_shard(tmp_path / "new.npz", [sample(-1.0), sample(-1.0)])
    os.utime(tmp_path / "old.npz", (1, 1))
    x, pi, z = load_buffer(tmp_path, 3)
    assert len(z) == 3 and sorted(z.tolist()) == [-1.0, -1.0, 1.0]


def test_buffer_pattern_and_empty(tmp_path):
    save_shard(tmp_path / "bootstrap.npz", [sample(0.0)])
    save_shard(tmp_path / "selfplay_iter001.npz", [sample(1.0)])
    assert load_buffer(tmp_path, 10, "bootstrap*.npz")[2].tolist() == [0.0]
    with pytest.raises(FileNotFoundError):
        load_buffer(tmp_path / "missing", 10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest training/alphazero/tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.alphazero.data'`.

- [ ] **Step 3: Write `data.py`**

```python
"""Training samples on disk: one compressed .npz shard per data-producing step.

x  float16 (N, 651)  encode(state) of the position (canonical frame)
pi float16 (N, 210)  target move distribution, canonical frame (one-hot from the alpha-beta
                     teacher, visit counts from MCTS)
z  float32 (N,)      final result for the side to move: +1 win, -1 loss, 0 draw"""
from pathlib import Path

import numpy as np
import torch

from .symmetry import N_ACTIONS, N_FEATURES


def save_shard(path, samples):
    path = Path(path)
    x = np.asarray([s[0] for s in samples], dtype=np.float16).reshape(-1, N_FEATURES)
    pi = np.asarray([s[1] for s in samples], dtype=np.float16).reshape(-1, N_ACTIONS)
    z = np.asarray([s[2] for s in samples], dtype=np.float32)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, x=x, pi=pi, z=z)
    tmp.replace(path)


def load_shard(path):
    with np.load(path) as data:
        return data["x"], data["pi"], data["z"]


def shard_paths(data_dir, pattern="*.npz"):
    paths = [p for p in Path(data_dir).glob(pattern) if not p.name.endswith(".tmp.npz")]
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def load_buffer(data_dir, max_positions, pattern="*.npz"):
    """Replay buffer: newest positions first, up to max_positions, as CPU tensors (x, pi, z)."""
    xs, pis, zs, total = [], [], [], 0
    for path in reversed(shard_paths(data_dir, pattern)):
        take = max_positions - total
        if take <= 0:
            break
        x, pi, z = load_shard(path)
        take = min(take, len(z))
        xs.append(x[len(z) - take:])
        pis.append(pi[len(z) - take:])
        zs.append(z[len(z) - take:])
        total += take
    if not xs:
        raise FileNotFoundError("no .npz shards matching %s in %s" % (pattern, data_dir))
    return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(pis)),
            torch.from_numpy(np.concatenate(zs)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest training/alphazero/tests/test_data.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add training/alphazero/data.py training/alphazero/tests/test_data.py
git commit -m "feat(alphazero): npz sample shards and replay buffer"
```

---

### Task 8: PUCT MCTS

**Files:**
- Create: `training/alphazero/mcts.py`
- Test: `training/alphazero/tests/test_mcts.py`

- [ ] **Step 1: Write the failing tests**

```python
import random

import torch

from training.alphazero.engine import E
from training.alphazero.mcts import MCTS, NetEvaluator, pick_action, terminal_value
from training.alphazero.net import AZNet


def uniform(state, actions):
    return {a: 1.0 / len(actions) for a in actions}, 0.0


def test_visits_sum_to_simulations():
    visits = MCTS(uniform, rng=random.Random(0)).search(E.State(), 20)
    assert sum(visits.values()) == 20
    assert set(visits) == set(E.legal_actions(E.State()))


def test_finds_immediate_win():
    state = E.State(p0=13, p1=40, turn=0)        # player 0 one step from row 0, player 1 in the centre
    visits = MCTS(uniform, rng=random.Random(0)).search(state, 60)
    assert pick_action(visits, 0.0, random.Random(0)) == 4


def test_terminal_value_is_from_side_to_move():
    lost = E.State(p0=4, p1=40, turn=1)           # player 0 already on row 0, player 1 to move
    assert terminal_value(lost) == -1.0
    draw = E.State(turn=500, max_turns=500)
    assert terminal_value(draw) == 0.0


def test_noise_and_net_evaluator_run():
    torch.manual_seed(0)
    mcts = MCTS(NetEvaluator(AZNet()), rng=random.Random(1))
    visits = mcts.search(E.State(), 8, noise=True)
    assert sum(visits.values()) == 8


def test_pick_action_temperature():
    visits = {1: 10, 2: 0, 3: 5}
    assert pick_action(visits, 0.0, random.Random(0)) == 1
    assert pick_action(visits, 1.0, random.Random(0)) in (1, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest training/alphazero/tests/test_mcts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.alphazero.mcts'`.

- [ ] **Step 3: Write `mcts.py`**

```python
"""Monte-Carlo tree search guided by the network (AlphaZero PUCT), used for self-play data.

tree[state] = {action: [prior, visits, total_value]}; total_value is seen from the player
who picks `action` at `state`. A fresh tree is built for every move (bounded memory,
and Dirichlet noise can be written straight into the root priors)."""
import math

import torch

from .engine import E
from .symmetry import rotate_for


class NetEvaluator:
    """(state, legal actions) -> ({action: prior}, value for the side to move)."""
    def __init__(self, net):
        self.net = net.eval()

    @torch.no_grad()
    def __call__(self, state, actions):
        logits, value = self.net(torch.tensor([E.encode(state)], dtype=torch.float32))
        out = logits[0].tolist()
        rotate = rotate_for(state)
        raw = {a: out[E.transform_action(a, rotate)] for a in actions}
        top = max(raw.values())
        weights = {a: math.exp(l - top) for a, l in raw.items()}
        total = sum(weights.values())
        return {a: w / total for a, w in weights.items()}, float(value[0])


def terminal_value(state):
    winner = E.outcome(state)
    if winner == 2:
        return 0.0
    return 1.0 if winner == state.turn % 2 else -1.0


def pick_action(visits, temperature, rng):
    """temperature 0 -> most visited; otherwise sample proportional to visits^(1/T)."""
    if temperature <= 0:
        return max(visits, key=lambda a: (visits[a], -a))
    actions = list(visits)
    weights = [visits[a] ** (1.0 / temperature) for a in actions]
    if sum(weights) <= 0:
        return rng.choice(actions)
    return rng.choices(actions, weights)[0]


class MCTS:
    def __init__(self, evaluator, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25, rng=None):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = rng

    def _score(self, edge, sqrt_total):
        prior, visits, total = edge
        q = total / visits if visits else 0.0
        return q + self.c_puct * prior * sqrt_total / (1 + visits)

    def search(self, root, simulations, noise=False):
        """Run `simulations` playouts from a non-terminal root; return {action: visits}."""
        tree = {}

        def expand(state):
            actions = E.legal_actions(state)
            priors, value = self.evaluator(state, actions)
            tree[state] = {a: [priors[a], 0, 0.0] for a in actions}
            return value

        expand(root)
        if noise and self.dirichlet_eps > 0:
            edges = list(tree[root].values())
            sample = [self.rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in edges]
            total = sum(sample) or 1.0
            for edge, g in zip(edges, sample):
                edge[0] = (1 - self.dirichlet_eps) * edge[0] + self.dirichlet_eps * g / total
        for _ in range(simulations):
            state, path = root, []
            while state in tree and E.outcome(state) is None:          # SELECT
                node = tree[state]
                sqrt_total = math.sqrt(max(1, sum(edge[1] for edge in node.values())))
                action = max(node, key=lambda a: self._score(node[a], sqrt_total))
                path.append(node[action])
                state = E.play(state, action, False)
            if E.outcome(state) is not None:
                value = terminal_value(state)
            else:
                value = expand(state)                                    # EXPAND + EVALUATE
            for edge in reversed(path):                                  # BACKPROPAGATE
                value = -value                                           # parent's point of view
                edge[1] += 1
                edge[2] += value
        return {a: edge[1] for a, edge in tree[root].items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest training/alphazero/tests/test_mcts.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add training/alphazero/mcts.py training/alphazero/tests/test_mcts.py
git commit -m "feat(alphazero): PUCT MCTS with Dirichlet root noise"
```

---

### Task 9: Games (bootstrap, self-play, gate) + process pool

**Files:**
- Create: `training/alphazero/games.py`
- Test: `training/alphazero/tests/test_games.py`

- [ ] **Step 1: Write the failing tests**

```python
from training.alphazero.games import (_init_worker, bootstrap_game, eval_game, game_summary,
                                      run_parallel, selfplay_game)
from training.alphazero.net import AZNet, save_checkpoint
from training.alphazero.symmetry import canonical_action
from training.alphazero.engine import E

SP_PARAMS = {"c_puct": 1.5, "dirichlet_alpha": 0.3, "dirichlet_eps": 0.25, "opening_plies": 2,
             "ply_cap": 12, "simulations": 4, "temp_moves": 4}


def check_records(records):
    assert records
    for features, pi, z in records:
        assert len(features) == 651 and len(pi) == 210
        assert abs(sum(pi) - 1.0) < 1e-6
        assert z in (-1.0, 0.0, 1.0)


def test_bootstrap_game():
    records, stats = bootstrap_game((1, 0.01, 8, 2, 16))
    check_records(records)
    assert stats["plies"] <= 16


def test_selfplay_and_eval_games(tmp_path):
    ckpt = tmp_path / "m.pt"
    save_checkpoint(ckpt, AZNet(), 0)
    _init_worker({"model": str(ckpt), "new": str(ckpt), "old": str(ckpt)})
    records, stats = selfplay_game((3, SP_PARAMS))
    check_records(records)
    assert eval_game((5, 0, 0.02, 8, 2, 12)) in (0.0, 0.5, 1.0)


def test_run_parallel_uses_processes():
    jobs = [(seed, 0.01, 8, 2, 10) for seed in range(2)]
    results = list(run_parallel(bootstrap_game, jobs, workers=2))
    assert len(results) == 2


def test_game_summary():
    s = game_summary([{"plies": 10, "winner": 0}, {"plies": 20, "winner": None}])
    assert s == {"games": 2, "avg_plies": 15.0, "p0_win": 0.5, "p1_win": 0.0, "draw": 0.5}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest training/alphazero/tests/test_games.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.alphazero.games'`.

- [ ] **Step 3: Write `games.py`**

```python
"""Whole games for data generation and gating, plus a process pool to run many at once.

Every game function takes ONE tuple (pool-friendly) and runs on CPU with torch limited to
one thread: the pure-Python rules dominate the cost, so GPU/MPS is used only in trainer.py."""
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
    """(seed, new_side, budget, wall_limit, opening_plies, ply_cap) -> score of "new" (1 / 0.5 / 0).

    Both sides run the deployment alpha-beta: their own policy at the root, own value head at leaves."""
    seed, new_side, budget, wall_limit, opening_plies, ply_cap = job
    rng = random.Random(seed)
    players = {key: (TorchPolicy(_NETS[key]), TorchValue(_NETS[key])) for key in ("new", "old")}
    state = random_opening(rng, opening_plies)
    while not finished(state, ply_cap):
        policy, value = players["new" if state.turn % 2 == new_side else "old"]
        action = E.Search(budget=budget, wall_limit=wall_limit, policy=policy, value_net=value,
                          mode="learned_alphabeta").choose(state)
        state = E.play(state, action, False)
    winner = E.outcome(state)
    return 0.5 if winner in (None, 2) else float(winner == new_side)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest training/alphazero/tests/test_games.py -v`
Expected: 4 passed (the pool test takes a few seconds: spawned workers import torch).

- [ ] **Step 5: Commit**

```bash
git add training/alphazero/games.py training/alphazero/tests/test_games.py
git commit -m "feat(alphazero): bootstrap/self-play/gate games and spawn process pool"
```

---

### Task 10: Trainer (policy+value, distillation)

**Files:**
- Create: `training/alphazero/trainer.py`
- Test: `training/alphazero/tests/test_trainer.py`

- [ ] **Step 1: Write the failing tests**

```python
import torch
import torch.nn.functional as F

from training.alphazero.config import make_config
from training.alphazero.net import AZNet, TinyValue
from training.alphazero.trainer import distill_value, pearson, teacher_values, train_policy_value


class Recorder:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars.setdefault(tag, []).append((step, value))


def toy_data(n=256):
    rows = torch.arange(n)
    x = torch.zeros(n, 651)
    x[rows, rows % 10] = 1.0
    pi = torch.zeros(n, 210)
    pi[rows, rows % 5] = 1.0
    z = torch.where(rows % 2 == 1, 1.0, -1.0)
    return x.half(), pi.half(), z.float()


def loss_of(net, data):
    x, pi, z = data
    with torch.no_grad():
        logits, value = net(x.float())
        return float(-(pi.float() * F.log_softmax(logits, 1)).sum(1).mean() + F.mse_loss(value, z))


def test_training_reduces_loss_and_logs():
    torch.manual_seed(0)
    cfg = make_config("smoke", overrides=["mirror_augment=false", "batch_size=32", "log_every=1"])
    net, data, rec = AZNet(), toy_data(), Recorder()
    before = loss_of(net, data)
    step = train_policy_value(net, data, cfg, torch.device("cpu"), rec, 0, 30, 1e-2, "t")
    assert loss_of(net, data) < 0.5 * before
    assert step == 30 * 8
    assert {"t/policy_loss", "t/value_loss", "t/lr", "t/policy_top1"} <= set(rec.scalars)


def test_mirror_augmentation_runs():
    cfg = make_config("smoke", overrides=["batch_size=32"])
    train_policy_value(AZNet(), toy_data(64), cfg, torch.device("cpu"), Recorder(), 0, 1, 1e-3, "t")


def test_pearson():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert abs(pearson(a, a) - 1.0) < 1e-6 and abs(pearson(a, -a) + 1.0) < 1e-6


def test_distill_recovers_a_tiny_teacher():
    torch.manual_seed(1)
    teacher = TinyValue()
    x = (torch.rand(1024, 651) < 0.3).half()

    class Wrap(torch.nn.Module):          # teacher_values expects (logits, value) like AZNet
        def forward(self, inp):
            return None, teacher(inp)

    targets = teacher_values(Wrap(), x, torch.device("cpu"))
    cfg = make_config("smoke", overrides=["distill_epochs=60", "batch_size=64", "lr=0.003",
                                          "distill_val_frac=0.1", "mirror_augment=false"])
    metrics = distill_value(TinyValue(), x, targets, cfg, torch.device("cpu"), Recorder())
    assert metrics["val_pearson"] > 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest training/alphazero/tests/test_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.alphazero.trainer'`.

- [ ] **Step 3: Write `trainer.py`**

```python
"""Gradient steps. train_policy_value: p02 pretrain + p03 iterations. distill_value: p04.

Data stays on the CPU as float16; each batch is mirrored (50%), moved to the device and
cast to float32 there (MPS has no float64; float32 everywhere is safe on cpu/cuda/mps)."""
import math

import torch
import torch.nn.functional as F

from .symmetry import FEATURE_MIRROR, POLICY_MIRROR

_FEAT = torch.tensor(FEATURE_MIRROR)
_POL = torch.tensor(POLICY_MIRROR)


def _maybe_mirror(xb, pib, enabled, generator):
    if not enabled:
        return xb, pib
    flip = (torch.rand(xb.shape[0], generator=generator) < 0.5)[:, None]
    xb = torch.where(flip, xb[:, _FEAT], xb)
    if pib is not None:
        pib = torch.where(flip, pib[:, _POL], pib)
    return xb, pib


def train_policy_value(net, data, cfg, device, writer, step, epochs, lr, tag):
    """Adam + cosine lr decay to cfg.lr_min. Loss = soft cross-entropy(policy) + MSE(value).
    Returns the next global step (TensorBoard x-axis continues across iterations)."""
    x, pi, z = data
    n = len(z)
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, epochs * math.ceil(n / cfg.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=min(cfg.lr_min, lr))
    gen = torch.Generator().manual_seed(cfg.seed + step)
    for _ in range(epochs):
        for idx in torch.randperm(n, generator=gen).split(cfg.batch_size):
            xb, pib = _maybe_mirror(x[idx], pi[idx], cfg.mirror_augment, gen)
            xb, pib, zb = xb.to(device).float(), pib.to(device).float(), z[idx].to(device).float()
            logits, value = net(xb)
            policy_loss = -(pib * F.log_softmax(logits, dim=1)).sum(1).mean()
            value_loss = F.mse_loss(value, zb)
            loss = policy_loss + value_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf"))  # measure only
            opt.step()
            sched.step()
            if step % cfg.log_every == 0:
                with torch.no_grad():
                    decided = zb != 0
                    sign = ((value > 0) == (zb > 0))[decided].float().mean() if decided.any() else None
                    stats = {"policy_loss": policy_loss, "value_loss": value_loss, "loss": loss,
                             "policy_top1": (logits.argmax(1) == pib.argmax(1)).float().mean(),
                             "lr": sched.get_last_lr()[0], "grad_norm": grad_norm}
                    if sign is not None:
                        stats["value_sign_acc"] = sign
                for name, val in stats.items():
                    writer.add_scalar("%s/%s" % (tag, name), float(val), step)
            step += 1
    net.eval()
    return step


@torch.no_grad()
def teacher_values(net, x, device, batch=4096):
    """Value-head outputs of an AZNet-like module for every row of x (CPU float32 tensor)."""
    net.to(device).eval()
    out = [net(x[i:i + batch].to(device).float())[1].float().cpu() for i in range(0, len(x), batch)]
    return torch.cat(out)


def pearson(a, b):
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def distill_value(tiny, x, targets, cfg, device, writer):
    """Fit tiny(x) to the teacher's values (MSE). Held-out split -> val MSE + Pearson per epoch."""
    gen = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(targets), generator=gen)
    n_val = max(1, int(len(targets) * cfg.distill_val_frac))
    val, train = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(tiny.parameters(), lr=cfg.lr)
    tiny.to(device)
    metrics = {}
    for epoch in range(cfg.distill_epochs):
        tiny.train()
        losses = []
        for idx in train[torch.randperm(len(train), generator=gen)].split(cfg.batch_size):
            xb, _ = _maybe_mirror(x[idx], None, cfg.mirror_augment, gen)   # value is mirror-invariant
            loss = F.mse_loss(tiny(xb.to(device).float()), targets[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        tiny.eval()
        with torch.no_grad():
            pred = tiny(x[val].to(device).float()).float().cpu()
        metrics = {"train_mse": sum(losses) / max(1, len(losses)),
                   "val_mse": float(F.mse_loss(pred, targets[val])),
                   "val_pearson": pearson(pred, targets[val])}
        for name, val_ in metrics.items():
            writer.add_scalar("distill/" + name, val_, epoch)
    return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest training/alphazero/tests/test_trainer.py -v`
Expected: 4 passed. If `test_distill_recovers_a_tiny_teacher` lands just under 0.9, raise its `distill_epochs` override to 120 (it is a realizable target; low correlation means too few steps, not a bug).

- [ ] **Step 5: Commit**

```bash
git add training/alphazero/trainer.py training/alphazero/tests/test_trainer.py
git commit -m "feat(alphazero): policy/value trainer with mirror augmentation and value distillation"
```

---

### Task 11: Phase CLI + phases 1–2 (bootstrap, pretrain)

**Files:**
- Create: `training/alphazero/phases/_cli.py`, `training/alphazero/phases/p01_bootstrap.py`, `training/alphazero/phases/p02_pretrain.py`
- Test: `training/alphazero/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test (dependency enforcement through the real CLI)**

```python
import json
import os
import shutil
import subprocess
import sys

import pytest

from training.alphazero.engine import AGENT_FILE, PROJECT, load_agent


def run_cli(args, runs_dir, timeout=900):
    env = dict(os.environ, AZ_RUNS_DIR=str(runs_dir))
    return subprocess.run([sys.executable, "-m"] + args, cwd=str(PROJECT), env=env,
                          capture_output=True, text=True, timeout=timeout)


def test_phase_refuses_to_run_before_its_dependency(tmp_path):
    r = run_cli(["training.alphazero.phases.p02_pretrain", "--run", "fresh", "--preset", "smoke"], tmp_path)
    assert r.returncode != 0
    assert "p01_bootstrap" in (r.stderr + r.stdout)


def test_bootstrap_then_pretrain(tmp_path):
    r = run_cli(["training.alphazero.phases.p01_bootstrap", "--run", "s", "--preset", "smoke"], tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    r = run_cli(["training.alphazero.phases.p02_pretrain", "--run", "s"], tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    run = tmp_path / "s"
    manifest = json.loads((run / "manifest.json").read_text())
    assert set(manifest["done"]) == {"p01", "p02"}
    assert manifest["best"] == "checkpoints/iter000.pt"
    assert (run / "data" / "bootstrap.npz").exists() and (run / "checkpoints" / "iter000.pt").exists()
    assert list((run / "tb").glob("events.out.tfevents.*"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest training/alphazero/tests/test_pipeline.py -v`
Expected: FAIL (`No module named training.alphazero.phases.p02_pretrain`).

- [ ] **Step 3: Write `phases/_cli.py`**

```python
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
```

- [ ] **Step 4: Write `phases/p01_bootstrap.py`**

```python
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
```

- [ ] **Step 5: Write `phases/p02_pretrain.py`**

```python
"""Phase 2 - pretrain the AlphaZero net on the bootstrap data (imitate the alpha-beta bot).

Needs: p01.  Writes: checkpoints/iter000.pt (becomes "best"), resets the iteration loop.
  uv run python -m training.alphazero.phases.p02_pretrain --run main"""
import sys

import torch

from ..config import pick_device
from ..data import load_buffer
from ..net import AZNet, save_checkpoint
from ..trainer import train_policy_value

PHASE, REQUIRES, TITLE = "p02", "p01", "pretrain on bootstrap data -> checkpoints/iter000.pt"


def run(ctx):
    cfg = ctx.cfg
    # Redoing pretraining restarts the loop: drop data/checkpoints of the old iterations.
    for stale in list(ctx.path("data").glob("selfplay_iter*.npz")) + list(ctx.path("checkpoints").glob("iter*.pt")):
        print("  removing stale", stale.relative_to(ctx.dir))
        stale.unlink()
    device = pick_device(cfg.device)
    print("  device:", device)
    torch.manual_seed(cfg.seed)
    data = load_buffer(ctx.path("data"), cfg.buffer_size, "bootstrap*.npz")
    net = AZNet()
    writer = ctx.writer()
    step = train_policy_value(net, data, cfg, device, writer, 0, cfg.pretrain_epochs, cfg.lr, "train")
    writer.close()
    save_checkpoint(ctx.path("checkpoints", "iter000.pt"), net.cpu(), 0)
    ctx.manifest.update({"best": "checkpoints/iter000.pt", "iteration": 0, "history": [],
                         "steps": {"train": step}})
    ctx.save()


if __name__ == "__main__":
    from ._cli import main
    main(sys.modules[__name__])
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest training/alphazero/tests/test_pipeline.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add training/alphazero/phases training/alphazero/tests/test_pipeline.py
git commit -m "feat(alphazero): phase CLI with dependency checks; p01 bootstrap, p02 pretrain"
```

---

### Task 12: Phase 3 — self-play / train / gate loop

**Files:**
- Create: `training/alphazero/phases/p03_iterate.py`
- Modify: `training/alphazero/tests/test_pipeline.py` (add test)

- [ ] **Step 1: Add the failing test**

Append to `test_pipeline.py`:

```python
def test_iterate_after_pretrain(tmp_path):
    for args in (["training.alphazero.phases.p01_bootstrap", "--run", "s", "--preset", "smoke"],
                 ["training.alphazero.phases.p02_pretrain", "--run", "s"],
                 ["training.alphazero.phases.p03_iterate", "--run", "s"]):
        r = run_cli(args, tmp_path)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    run = tmp_path / "s"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["iteration"] == 1 and manifest["history"]
    assert (run / "data" / "selfplay_iter001.npz").exists()
    assert (run / "checkpoints" / "iter001.pt").exists()
    assert manifest["best"] in ("checkpoints/iter000.pt", "checkpoints/iter001.pt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest training/alphazero/tests/test_pipeline.py::test_iterate_after_pretrain -v`
Expected: FAIL (`No module named training.alphazero.phases.p03_iterate`).

- [ ] **Step 3: Write `phases/p03_iterate.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest training/alphazero/tests/test_pipeline.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add training/alphazero/phases/p03_iterate.py training/alphazero/tests/test_pipeline.py
git commit -m "feat(alphazero): p03 resumable self-play/train/gate loop"
```

---

### Task 13: Phases 4–6 (distill, export .pth, embed)

**Files:**
- Create: `training/alphazero/phases/p04_distill.py`, `training/alphazero/phases/p05_export_pth.py`, `training/alphazero/phases/p06_embed.py`
- Modify: `training/alphazero/tests/test_pipeline.py` (add test)

- [ ] **Step 1: Add the failing test**

```python
def test_distill_export_embed(tmp_path):
    agent = tmp_path / "az_smoke.py"
    shutil.copy(AGENT_FILE, agent)
    steps = [["training.alphazero.phases.p01_bootstrap", "--run", "s", "--preset", "smoke", "--set", "agent_path=%s" % agent]]
    steps += [["training.alphazero.phases.%s" % m, "--run", "s"]
              for m in ("p02_pretrain", "p03_iterate", "p04_distill", "p05_export_pth", "p06_embed")]
    for args in steps:
        r = run_cli(args, tmp_path)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    run = tmp_path / "s"
    manifest = json.loads((run / "manifest.json").read_text())
    assert {"p04", "p05", "p06"} <= set(manifest["done"])
    assert (run / "checkpoints" / "tiny_value.pt").exists()
    assert (run / "export" / "az_smoke.pth").exists()
    module = load_agent(agent, name="az_smoke_embedded")
    assert module.DEPLOY_MODE == "learned_alphabeta" and module._POLICY and module._VALUE
    assert manifest["embed"]["max_policy_diff"] < 5e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest training/alphazero/tests/test_pipeline.py::test_distill_export_embed -v`
Expected: FAIL (`No module named training.alphazero.phases.p04_distill`).

- [ ] **Step 3: Write `phases/p04_distill.py`**

```python
"""Phase 4 - distill the best net's value head into TinyValue (651 -> 32 -> 1).

Needs: p03.  Writes: checkpoints/tiny_value.pt, manifest["distill"] (val MSE, Pearson).
Spec target: Pearson >= 0.90 on held-out positions (warning only; p07 is the real gate)."""
import sys

import torch

from ..config import pick_device
from ..data import load_buffer
from ..net import TinyValue, load_checkpoint
from ..trainer import distill_value, teacher_values

PHASE, REQUIRES, TITLE = "p04", "p03", "distill value head -> checkpoints/tiny_value.pt"


def run(ctx):
    cfg = ctx.cfg
    device = pick_device(cfg.device)
    x, _, _ = load_buffer(ctx.path("data"), cfg.distill_positions)
    teacher = load_checkpoint(ctx.path(ctx.manifest["best"]))
    targets = teacher_values(teacher, x, device)
    torch.manual_seed(cfg.seed)
    tiny = TinyValue()
    writer = ctx.writer()
    metrics = distill_value(tiny, x, targets, cfg, device, writer)
    writer.close()
    torch.save({"tiny": tiny.cpu().state_dict(), "teacher": ctx.manifest["best"], **metrics},
               str(ctx.path("checkpoints", "tiny_value.pt")))
    ctx.manifest["distill"] = metrics
    ctx.save()
    print("  distill: val_mse %.4f, val_pearson %.3f" % (metrics["val_mse"], metrics["val_pearson"]))
    if metrics["val_pearson"] < cfg.distill_min_pearson:
        print("  WARNING: Pearson below %.2f - try --set distill_epochs=100 --force" % cfg.distill_min_pearson)


if __name__ == "__main__":
    from ._cli import main
    main(sys.modules[__name__])
```

- [ ] **Step 4: Write `phases/p05_export_pth.py`**

```python
"""Phase 5 - export inference weights (best policy net + tiny value net) to one .pth.

Needs: p04.  Writes: export/<agent name>.pth  (plain tensors in agent layer order + meta).
This file is the hand-off artifact: embed it anywhere with training.alphazero.embed_pth."""
import sys

import torch

from ..export import export_pth
from ..net import TinyValue, load_checkpoint

PHASE, REQUIRES, TITLE = "p05", "p04", "export weights -> export/<agent>.pth"


def run(ctx):
    net = load_checkpoint(ctx.path(ctx.manifest["best"]))
    tiny = TinyValue()
    tiny.load_state_dict(torch.load(str(ctx.path("checkpoints", "tiny_value.pt")), weights_only=True)["tiny"])
    out = ctx.path("export", ctx.agent_path.stem + ".pth")
    export_pth(out, net, tiny, {"run": ctx.name, "best": ctx.manifest["best"],
                                "iteration": ctx.manifest["iteration"], "distill": ctx.manifest.get("distill")})
    ctx.manifest["export_pth"] = str(out.relative_to(ctx.dir))
    ctx.save()
    print("  wrote", out)


if __name__ == "__main__":
    from ._cli import main
    main(sys.modules[__name__])
```

- [ ] **Step 5: Write `phases/p06_embed.py`**

```python
"""Phase 6 - embed the .pth into the single-file agent (pure-Python inference, stdlib only).

Needs: p05.  Rewrites MODEL_BLOB / VALUE_BLOB / DEPLOY_MODE between the MODEL BLOBS markers
of the agent (default submissions/az_agent.py; change with --set agent_path=...), re-imports
it and checks the pure-Python outputs against the torch nets."""
import sys
import time

import torch

from ..adapters import TorchPolicy, TorchValue
from ..engine import load_agent
from ..export import embed_pth
from ..net import TinyValue, load_checkpoint
from ..games import random_opening

PHASE, REQUIRES, TITLE = "p06", "p05", "embed weights into the agent .py"


def _probe_states(n=20, seed=0):
    import random
    rng = random.Random(seed)
    return [random_opening(rng, 6) for _ in range(n)]


def run(ctx):
    agent_path = ctx.agent_path
    size = embed_pth(ctx.path(ctx.manifest["export_pth"]), agent_path, ctx.cfg.blob_dtype)
    agent = load_agent(agent_path, name="embedded_%d" % time.time_ns())
    if agent._POLICY is None or agent._VALUE is None:
        raise SystemExit("agent could not decode the embedded blobs: " + str(agent_path))
    policy = TorchPolicy(load_checkpoint(ctx.path(ctx.manifest["best"])))
    tiny = TinyValue()
    tiny.load_state_dict(torch.load(str(ctx.path("checkpoints", "tiny_value.pt")), weights_only=True)["tiny"])
    value = TorchValue(tiny)
    states = _probe_states()
    max_policy = max(max(abs(a - b) for a, b in zip(agent._POLICY.logits(s), policy.logits(s))) for s in states)
    max_value = max(abs(agent._VALUE.value(s) - value.value(s)) for s in states)
    ctx.manifest["embed"] = {"agent": str(agent_path), "blob_chars": size, "file_bytes": agent_path.stat().st_size,
                             "max_policy_diff": max_policy, "max_value_diff": max_value}
    ctx.save()
    print("  embedded into %s (%.0f KB), max |diff| policy %.4f value %.4f"
          % (agent_path, agent_path.stat().st_size / 1024, max_policy, max_value))
    if max_policy > 5e-2 or max_value > 2e-2:
        raise SystemExit("pure-Python inference differs from torch - try --set blob_dtype=f --force")


if __name__ == "__main__":
    from ._cli import main
    main(sys.modules[__name__])
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest training/alphazero/tests/test_pipeline.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add training/alphazero/phases/p04_distill.py training/alphazero/phases/p05_export_pth.py training/alphazero/phases/p06_embed.py training/alphazero/tests/test_pipeline.py
git commit -m "feat(alphazero): p04 distill, p05 .pth export, p06 embed with parity check"
```

---

### Task 14: Phase 7 benchmark + `pipeline.py` orchestrator

**Files:**
- Create: `training/alphazero/phases/p07_benchmark.py`, `training/alphazero/pipeline.py`
- Modify: `training/alphazero/tests/test_pipeline.py` (add tests)

- [ ] **Step 1: Add the failing tests**

Add `from training.alphazero.phases.p07_benchmark import parse_gauntlet` to the imports at the top of `test_pipeline.py`, then append:

```python
def test_parse_gauntlet():
    out = ("az (40 cặp ván/đối thủ, mở đầu 2 nước ngẫu nhiên)\n"
           "  vs MSSV           W  45  D   2  L  33  điểm 0.58\n"
           "  vs training/opponents/v2_trained.py W  60  D   0  L  20  điểm 0.75\n")
    assert parse_gauntlet(out) == {"MSSV": 0.58, "training/opponents/v2_trained.py": 0.75}


@pytest.mark.slow
def test_pipeline_smoke_end_to_end(tmp_path):
    agent = tmp_path / "az_e2e.py"
    shutil.copy(AGENT_FILE, agent)
    r = run_cli(["training.alphazero.pipeline", "--run", "e2e", "--preset", "smoke",
                 "--set", "agent_path=%s" % agent, "--to", "6"], tmp_path)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    manifest = json.loads((tmp_path / "e2e" / "manifest.json").read_text())
    assert set(manifest["done"]) == {"p01", "p02", "p03", "p04", "p05", "p06"}
    r = run_cli(["training.alphazero.pipeline", "--run", "e2e", "--status"], tmp_path)
    assert "p06" in r.stdout and "pending" in r.stdout          # p07 still pending
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest training/alphazero/tests/test_pipeline.py -v -k "gauntlet or end_to_end"`
Expected: FAIL (`No module named 'training.alphazero.phases.p07_benchmark'`).

- [ ] **Step 3: Write `phases/p07_benchmark.py`**

```python
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
```

- [ ] **Step 4: Write `pipeline.py`**

```python
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
        print("  iter %(iter)d attempt %(attempt)d lr %(lr).1e score %(score).3f accepted %(accepted)s" % row)
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
```

- [ ] **Step 5: Run the whole test suite**

Run: `uv run pytest -v`
Expected: all pass (the slow end-to-end test takes a few minutes).

- [ ] **Step 6: Commit**

```bash
git add training/alphazero/phases/p07_benchmark.py training/alphazero/pipeline.py training/alphazero/tests/test_pipeline.py
git commit -m "feat(alphazero): p07 benchmark and pipeline orchestrator with --status"
```

---

### Task 15: Docs (README, CLAUDE.md)

**Files:**
- Create: `training/alphazero/README.md`
- Modify: `CLAUDE.md` (structure tree + commands)

- [ ] **Step 1: Write `training/alphazero/README.md`**

````markdown
# AlphaZero pipeline for the Wrongway agent

Trains a policy + value net by self-play and ships it inside one pure-Python file:
`wrongway-student/wrongway/submissions/az_agent.py` (fork of `agents/MSSV.py`).
Spec: `docs/alphazero_pipeline.md`. Plan + deviations: `docs/superpowers/plans/2026-11-18-alphazero-pipeline.md`.

## Setup

```bash
uv sync                    # torch, tensorboard, numpy, pytest into .venv
uv run pytest -m "not slow"
```

Devices (training/distill only; games always run on CPU worker processes):
`--set device=auto` (default) picks `cuda` > `mps` > `cpu`. Linux + NVIDIA: the default PyPI
wheel already has CUDA. Windows + NVIDIA: install a CUDA wheel, e.g.
`uv pip install torch --index-url https://download.pytorch.org/whl/cu124`.

## Phases (each needs the previous one)

| # | Script (`uv run python -m training.alphazero.phases.…`) | Produces | TensorBoard |
|---|---|---|---|
| 1 | `p01_bootstrap` | `data/bootstrap.npz` | `bootstrap/*` |
| 2 | `p02_pretrain` | `checkpoints/iter000.pt` | `train/*` |
| 3 | `p03_iterate` | `data/selfplay_iterNNN.npz`, `checkpoints/iterNNN.pt` | `selfplay/*`, `train/*`, `gate/*` |
| 4 | `p04_distill` | `checkpoints/tiny_value.pt` | `distill/*` |
| 5 | `p05_export_pth` | `export/az_agent.pth` | – |
| 6 | `p06_embed` | blobs inside `submissions/az_agent.py` | – |
| 7 | `p07_benchmark` | `export/benchmark.txt` | `bench/*` |

All outputs live in `training/alphazero/runs/<run>/` (gitignored), progress in its `manifest.json`.

```bash
uv run python -m training.alphazero.pipeline --run try1 --preset smoke     # ~minutes, checks wiring
uv run python -m training.alphazero.pipeline --run main --preset quick     # first real run
uv run python -m training.alphazero.pipeline --run main --status
uv run tensorboard --logdir training/alphazero/runs
```

Presets: `smoke` (wiring test), `quick`, `standard` (default), `spec` (exact spec numbers, slow).
Change anything with `--set key=value` (fields in `config.py`). Redoing a phase (`--force`)
marks every later phase as pending. Phase 3 resumes where it stopped.

## Using the result

- Embed any exported weights into any agent file:
  `uv run python -m training.alphazero.embed_pth --pth runs/main/export/az_agent.pth --agent path/to/agent.py`
- Submit: copy `az_agent.py` to `<MSSV>.py`. It imports only the standard library.
- Benchmark rule (spec): accept only if ≥ 55% vs current `MSSV` over 80 games, no move > 0.55 s,
  tests pass. Record the result in `training/EXPERIMENTS.md`.
````

- [ ] **Step 2: Update `CLAUDE.md`**

In the structure tree, under `training/`, add:

```
│   ├── alphazero/             # AlphaZero: self-play MCTS -> net -> .pth -> nhúng vào submissions/az_agent.py (uv + torch)
│   │   ├── phases/p01..p07    # mỗi phase cần phase trước; pipeline.py chạy tất cả; xem README.md
```

and under “Lệnh”, add:

```
uv run python -m training.alphazero.pipeline --run main --preset quick   # train AlphaZero end-to-end
uv run tensorboard --logdir training/alphazero/runs
uv run pytest -m "not slow"
```

- [ ] **Step 3: Commit**

```bash
git add training/alphazero/README.md CLAUDE.md
git commit -m "docs(alphazero): README with phases, presets, devices; CLAUDE.md entries"
```

---

### Task 16: First real run + record result (operational)

- [ ] **Step 1: Wiring run on this machine**

Run: `uv run python -m training.alphazero.pipeline --run try1 --preset smoke --to 6`
Expected: all six phases `done`; `p02` prints `device: mps`.

- [ ] **Step 2: Quick real run (background; watch TensorBoard)**

Run: `uv run python -m training.alphazero.pipeline --run quick1 --preset quick`
Watch: `uv run tensorboard --logdir training/alphazero/runs` → `train/policy_loss` falling, `train/value_sign_acc` > 0.6, `gate/score` per iteration, `selfplay/avg_plies`. Use the measured `selfplay/sec_per_game` to size the `standard` run.

- [ ] **Step 3: Record in `training/EXPERIMENTS.md`**

Add an entry: preset, device, wall time per phase, final `gate/score` history, distill Pearson, embed size, p07 scores vs `MSSV` and `v2_trained`, verdict (keep/reject, why).

- [ ] **Step 4: Commit only if p07 says ACCEPT**

```bash
git add wrongway-student/wrongway/submissions/az_agent.py training/EXPERIMENTS.md
git commit -m "feat(az_agent): embed AlphaZero weights from run quick1 (gauntlet vs MSSV: <score>)"
```

If REJECT: commit only `EXPERIMENTS.md`, then restore the agent with `git checkout -- wrongway-student/wrongway/submissions/az_agent.py`.
