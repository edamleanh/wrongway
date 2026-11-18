# AlphaZero Training Pipeline for Wrongway Agent

## Goal

Train a policy + value network to improve the existing negamax alpha-beta agent in
`agents/MSSV.py`. The policy network guides move ordering inside alpha-beta search
(called once per turn at root). The value network is distilled into a tiny 2-layer net
(651→32→1) that replaces `heuristic()` at leaf nodes.

Both networks share a backbone trained via AlphaZero-style MCTS self-play.

---

## Constraints (non-negotiable)

- **Inference environment**: Google Colab free-tier CPU, pure Python 3.9+, standard
  library only. No numpy, no PyTorch at inference.
- **Time budget per move**: 0.55s real-time, simulated as 0.275s during benchmarking
  (Colab CPU is ~2× slower than a modern laptop).
- **Submission format**: single `.py` file. Network weights embedded as a
  base85+zlib-compressed string constant (`MODEL_BLOB`, `VALUE_BLOB`).
- **Training environment**: any GPU (PyTorch allowed during training only).

---

## Existing Infrastructure (do not rewrite)

All game logic lives in `agents/MSSV.py`. Use it directly.

| Symbol | Type | Description |
|---|---|---|
| `State` | NamedTuple | Full board state: `p0,p1` (cell index 0–80), `w0,w1` (walls 0–10), `h,v` (wall bitmasks, 64-bit int), `turn`, `max_turns`, `goal0,goal1` |
| `encode(state)` | `state → list[float]` | 651-dim feature vector (8 planes × 9×9 + 3 scalars) |
| `legal_actions(state)` | `state → list[int]` | Action indices 0–210. 0–80 = moves, 81–144 = H walls, 145–208 = V walls, 209 = pass |
| `play(state, action, validate)` | `state,int,bool → state` | Returns next state. Use `validate=False` for speed |
| `outcome(state)` | `state → int\|None` | `0`=player0 wins, `1`=player1 wins, `2`=draw, `None`=ongoing |
| `action_dict(action)` | `int → dict` | Converts action index to Arena dict |
| `Student` | class | Existing pure-Python MLP inference engine. Loads from base85+zlib blob. Reuse for both policy and tiny value net |
| `CACHE` | `ByteCache` | Shared 16MB LRU cache. `graph()` and `distances()` results are cached here |

Import from `agents/MSSV.py`:
```python
import sys
sys.path.insert(0, "agents")
from MSSV import State, encode, legal_actions, play, outcome, action_dict, ACTION_COUNT, PASS
```

---

## Network Architecture

### Shared Backbone (PyTorch, training only)
```
Input:  651 floats  (output of encode())
Linear(651, 256) → ReLU
Linear(256, 128) → ReLU
Output: 128-dim embedding
```

### Policy Head (PyTorch, training only)
```
Linear(128, 211)  → raw logits (no softmax; loss uses cross_entropy which applies it)
Output: 211 logits, one per action index (0–210)
```

### Value Head (PyTorch, training only)
```
Linear(128, 32) → ReLU
Linear(32, 1)   → Tanh
Output: scalar in (-1, +1). +1 = current player wins, -1 = loses, 0 = draw
```

### Tiny Value Net (PyTorch distilled, then exported to pure Python)
```
Input:  651 floats
Linear(651, 32) → ReLU
Linear(32, 1)   → Tanh
Output: scalar in (-1, +1)
```
This is trained to match the full value head's outputs, not game outcomes directly.

---

## Files to Create

```
training/
  net.py           # PyTorch model definition
  selfplay.py      # MCTS self-play data generation
  train_net.py     # training loop
  distill.py       # distill value head → tiny 651→32→1
  alphazero.py     # orchestration: runs all phases in order
  export_net.py    # serialize weights → base85+zlib string
```

---

## Phase 1: Bootstrap from Existing Alpha-Beta Bot

**Purpose**: avoid cold-start (random net produces useless training signal).

**Script**: `training/selfplay.py --mode bootstrap --games 5000 --out data/bootstrap.pkl`

**Process**:
1. Import `agents/MSSV.py`. Instantiate `Search(budget=0.55, wall_limit=16, model=None, mode="alphabeta")`.
2. Play complete games: `search.choose(state)` for both players.
3. For each position in the game, record:
   - `features = encode(state)` — 651 floats
   - `policy_target` — one-hot vector of length 211, 1.0 at the chosen action index
   - `value_target` — game outcome from current player's perspective: `+1.0` if current player's side won, `-1.0` if lost, `0.0` if draw. Assign retrospectively after game ends.
4. Store as list of `(features, policy_target, value_target)` tuples. Serialize with `pickle`.

**Output**: `data/bootstrap.pkl`

---

## Phase 2: AlphaZero Self-Play Loop

### 2a. MCTS Self-Play

**Script**: `training/selfplay.py --mode selfplay --games 200 --simulations 100 --out data/selfplay_iter{N}.pkl --model checkpoints/iter{N}.pt`

**MCTS algorithm** (implement from scratch using `agents/MSSV.py` primitives):

```
Tree node: dict mapping action_index → [prior_prob, visit_count, total_value]
           one node per unique State reached

For each simulation (repeat M=100 times per move):

  SELECT:
    start at root state
    while state is in tree and not terminal:
      node = tree[state]
      N_total = sum of all visit_counts in node
      for each action in node:
        Q = total_value / visit_count  if visit_count > 0  else 0.0
        U = C_puct * prior * sqrt(N_total) / (1 + visit_count)
        score = Q + U
      action = argmax(score)
      state = play(state, action, False)

  EXPAND:
    if state not in tree and not terminal:
      features = encode(state)
      policy_logits, value = net.forward(features)
      priors = softmax(policy_logits)  masked to legal_actions(state)
      tree[state] = {a: [priors[a], 0, 0.0] for a in legal_actions(state)}
      leaf_value = value  (scalar from value head, range -1..+1)
    else if terminal:
      winner = outcome(state)
      player = state.turn % 2
      leaf_value = +1.0 if winner == player else (-1.0 if winner != 2 else 0.0)

  BACKPROPAGATE:
    walk back up the path of (state, action) pairs
    for each node on path (alternating perspective):
      visit_count += 1
      total_value += leaf_value
      leaf_value = -leaf_value  (flip perspective each level)

After M simulations:
  policy_target = visit_counts / M  (length-211 vector, 0 for illegal actions)
  record (encode(root_state), policy_target, placeholder_value)
  action = argmax(visit_counts)  during evaluation; sample during training
  root_state = play(root_state, action, False)

After game ends:
  fill value_target for all recorded positions:
    +1.0 if that position's player won, -1.0 if lost, 0.0 if draw
```

**C_puct**: 1.5 (UCT exploration constant).

**Temperature**: sample actions proportional to visit_count^(1/temp) for first 10 moves (temp=1.0), then argmax (temp→0).

**Output**: `data/selfplay_iter{N}.pkl` — same format as bootstrap data.

### 2b. Training

**Script**: `training/train_net.py --data data/ --checkpoint checkpoints/iter{N}.pt --out checkpoints/iter{N+1}.pt`

**Data**: merge all `.pkl` files in `data/`, keep last 500,000 positions (replay buffer).

**Loss**:
```python
policy_loss = F.cross_entropy(policy_logits, policy_target)   # policy_target is soft distribution
value_loss  = F.mse_loss(value_out.squeeze(), value_target)
loss = policy_loss + value_loss
```

**Hyperparameters**:
- Optimizer: Adam, lr=1e-3, weight decay=1e-4
- Batch size: 512
- Epochs per iteration: 10 passes over current replay buffer
- lr schedule: cosine decay to 1e-4 over all epochs

**Checkpoint format**: `torch.save({"backbone": ..., "policy": ..., "value": ..., "iter": N}, path)`

### 2c. Evaluation

**Script**: `training/alphazero.py` calls this internally after each train step.

Play 40 games: new checkpoint vs previous checkpoint.
- Use `Search(budget=0.55, wall_limit=16, model=new_net, mode="learned_alphabeta")` vs same with old net.
- If new net wins ≥ 22/40 (55%): accept, replace checkpoint. Else: discard.

### 2d. Loop

```
load bootstrap data
pretrain on bootstrap data (5 epochs)
save as checkpoints/iter0.pt

for N in range(20):
    selfplay(model=iter{N}.pt, games=200, simulations=100) → iter{N}.pkl
    train(data=all pkl files, in=iter{N}.pt) → iter{N+1}.pt
    evaluate(iter{N+1} vs iter{N})
    if accepted: continue
    else: reduce lr, retry training with same data
```

---

## Phase 3: Distill Value Head → Tiny Net

**Script**: `training/distill.py --model checkpoints/final.pt --games 10000 --out checkpoints/tiny_value.pt`

**Process**:
1. Generate 10,000 positions via alpha-beta self-play (diverse positions, not just opening).
2. For each position: run `full_net.backbone + full_net.value_head` → get target scalar.
3. Train tiny net `(651→32→1)`:
   ```python
   loss = F.mse_loss(tiny_net(features), full_value_targets)
   ```
4. Optimizer: Adam lr=1e-3, 50 epochs, batch 512.
5. Validate: Pearson correlation between tiny and full net outputs on held-out 1000 positions. Target ≥ 0.90.

---

## Phase 4: Export Weights to Pure Python

**Script**: `training/export_net.py --policy checkpoints/final.pt --value checkpoints/tiny_value.pt --out agents/MSSV.py`

**Policy net export** (backbone + policy head, 651→256→128→211):
```python
weights = []
for layer in [backbone_linear1, backbone_linear2, policy_linear]:
    weights.extend(layer.weight.detach().numpy().flatten().tolist())
    weights.extend(layer.bias.detach().numpy().flatten().tolist())
raw = struct.pack("<%df" % len(weights), *weights)
compressed = base64.b85encode(zlib.compress(raw, level=9)).decode("ascii")
# write into MODEL_BLOB = "..." in agents/MSSV.py
```

**Tiny value net export** (651→32→1):
Same process, different weight shapes. Write into `VALUE_BLOB = "..."` in `agents/MSSV.py`.

**`Student` class shape config**: `Student.SHAPES` currently hardcoded as `((651,64),(64,32),(32,211))`.
Export script must patch `SHAPES` or create a second loader class `TinyValueNet` with `SHAPES = ((651,32),(32,1))`.

**After export, patch `agents/MSSV.py`**:
```python
DEPLOY_MODE = "learned_alphabeta"   # was "alphabeta"
MODEL_BLOB  = "<policy blob>"       # was ""
VALUE_BLOB  = "<value blob>"        # new constant, add after MODEL_BLOB
```

---

## Inference Integration (changes to `agents/MSSV.py`)

Add `VALUE_BLOB = ""` constant after `MODEL_BLOB`.

Add `TinyValueNet` class (identical to `Student` but `SHAPES = ((651,32),(32,1))`):
```python
class TinyValueNet(Student):
    SHAPES = ((651, 32), (32, 1))
```

Add global `_VALUE_NET = None`.

Patch `choose_action()` to load `_VALUE_NET` from `VALUE_BLOB` same as `_MODEL` from `MODEL_BLOB`.

Pass `_VALUE_NET` into `Search.__init__` as `value_net` parameter.

Patch `Search.evaluate()`:
```python
def evaluate(self, state):
    if outcome(state) is not None:
        return heuristic(state)
    goal = state.goal1 if state.turn % 2 else state.goal0
    if any(a // 9 == goal for a in pawn_moves(state)):
        return 9999.0
    if self.value_net is not None:
        return self.value_net.predict(state)[1] * 12.0
    if self.model is not None:
        return self.prediction(state)[1] * 12.0 + 0.10 * heuristic(state)
    return heuristic(state)
```

`TinyValueNet.predict(state)` returns `(None, scalar)` to match `Student.predict` interface:
```python
def predict(self, state):
    values = encode(state)
    for index, (rows, bias) in enumerate(self.layers):
        values = [b + sum(w * x for w, x in zip(row, values)) for row, b in zip(rows, bias)]
        if index < len(self.layers) - 1:
            values = [max(0.0, x) for x in values]
    return None, math.tanh(values[0])
```

---

## Benchmarking After Each Phase

Run after bootstrap pretraining, after each self-play iteration, and after final export:

```bash
# vs existing bots (40 games each, Colab-speed budget 0.275s):
python training/gauntlet.py MSSV dev training/opponents/v2_trained.py --pairs 20 --workers 1

# vs AI Hard (80 games, standard benchmark):
# (use existing gauntlet.py infrastructure)

# timing check — must not exceed 0.55s per move on target hardware:
python tools/check_submission.py wrongway-student/wrongway/submissions/MSSV.py
```

**Accept new model only if**:
- Wins ≥ 55% vs current `submissions/MSSV.py` over 80 games
- No single move exceeds 0.55s in `check_submission.py`
- Passes `python -m unittest discover tests`

---

## Directory Layout After Pipeline

```
training/
  net.py
  selfplay.py
  train_net.py
  distill.py
  alphazero.py
  export_net.py
  data/
    bootstrap.pkl
    selfplay_iter0.pkl
    selfplay_iter1.pkl
    ...
  checkpoints/
    iter0.pt
    iter1.pt
    ...
    final.pt
    tiny_value.pt
agents/
  MSSV.py          ← MODEL_BLOB and VALUE_BLOB filled, DEPLOY_MODE patched
```

---

## Summary of What Changes vs Current Bot

| Component | Before | After |
|---|---|---|
| Move ordering | hand-crafted wall formula | policy logits from AlphaZero net (once/turn) |
| Leaf evaluation | linear 6-weight formula | tiny 651→32→1 net distilled from value head |
| Search backbone | negamax alpha-beta | unchanged |
| Inference cost | table lookups | +~1ms/turn (policy) +~1ms/leaf (value, ~50 leaves) |
| Training | TD self-play | AlphaZero MCTS self-play |
