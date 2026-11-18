"""Wrongway agent and reusable rules engine. Python 3.9+, standard library only.

The observation decoder is provisional until checked against the instructor Arena.
Run `python -m unittest discover -s tests` and read README.md before submission.
"""
import base64
import math
import struct
import sys
import time
import zlib
from collections import OrderedDict, deque
from typing import NamedTuple

RULES_VERSION = "wrongway-md-v1"
MOVE_COUNT, ACTION_COUNT, PASS = 81, 210, 209
DEFAULT_BUDGET = 0.55       # seconds per move (limit 1 s; grading Colab CPU is ~2x slower than a laptop)
DEPLOY_WALL_LIMIT = 16
EVAL_WEIGHTS = (0.35, 0.08, 0.10, 0.025, 0.15, 0.12)
POLICY_SHAPES = ((651, 256), (256, 128), (128, 210))   # backbone + policy head (AlphaZero net)
VALUE_SHAPES = ((651, 32), (32, 1))                   # tiny value net distilled from the value head
POLICY_WEIGHT = 0.20        # how much root logits (clipped to +-10) shift root move ordering
VALUE_SCALE = 12.0          # tanh value (-1..1) -> pawn-ply units, the scale heuristic() uses
HEURISTIC_WEIGHT = 0.10     # leaf = VALUE_SCALE * value + HEURISTIC_WEIGHT * heuristic(): keeps the
                            # hand-written distance/wall knowledge the net may not have learned

# ==== MODEL BLOBS BEGIN (written by training/alphazero/phases/p06_embed.py - do not edit by hand) ====
MODEL_BLOB = ""
VALUE_BLOB = ""
DEPLOY_MODE = "alphabeta"
# ==== MODEL BLOBS END ====


class State(NamedTuple):
    p0: int = 76
    p1: int = 4
    w0: int = 10
    w1: int = 10
    h: int = 0
    v: int = 0
    turn: int = 0
    max_turns: int = 500
    goal0: int = 0
    goal1: int = 8


def _size(value, seen=None):
    """Conservative owned-object accounting (shared objects count per entry)."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, (tuple, list)):
        size += sum(_size(x, seen) for x in value)
    elif isinstance(value, dict):
        size += sum(_size(k, seen) + _size(v, seen) for k, v in value.items())
    return size


class ByteCache:
    """One shared cache budget for graphs, distances, evaluations and search."""
    def __init__(self, limit=16 * 1024 * 1024):
        self.limit = limit
        self.used = 0
        self.data = OrderedDict()

    def get(self, key):
        item = self.data.get(key)
        if item is None:
            return None
        self.data.move_to_end(key)
        return item[0]

    def put(self, key, value):
        # 256 bytes covers the OrderedDict link, entry tuple and table capacity.
        cost = _size(key) + _size(value) + 256
        previous = self.data.pop(key, None)
        if previous is not None:
            self.used -= previous[1]
        if cost > self.limit:
            return
        while self.data and self.used + cost > self.limit:
            _, (_, old_cost) = self.data.popitem(last=False)
            self.used -= old_cost
        self.data[key] = (value, cost)
        self.used += cost

    def clear(self):
        self.data.clear()
        self.used = 0


CACHE = ByteCache()


def wall_action(orientation, row, col):
    if orientation not in ("H", "V") or not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError("invalid wall")
    return 81 + (64 if orientation == "V" else 0) + row * 8 + col


def action_dict(action):
    if not isinstance(action, int) or not 0 <= action < ACTION_COUNT:
        raise ValueError("invalid action index")
    if action < 81:
        return {"type": "move", "to": [action // 9, action % 9]}
    if action == PASS:
        return {"type": "pass"}
    slot = (action - 81) % 64
    return {"type": "wall", "orientation": "H" if action < 145 else "V",
            "row": slot // 8, "col": slot % 8}


def action_index(action):
    kind = action.get("type", "wall" if "orientation" in action else None)
    if kind == "pass":
        return PASS
    if kind == "move":
        row, col = action["to"]
        if type(row) is not int or type(col) is not int or not (0 <= row < 9 and 0 <= col < 9):
            raise ValueError("invalid move coordinates")
        return row * 9 + col
    if kind == "wall":
        if type(action["row"]) is not int or type(action["col"]) is not int:
            raise ValueError("wall coordinates must be integers")
        return wall_action(action["orientation"], action["row"], action["col"])
    raise ValueError("unknown action type")


def graph(h, v):
    key = ("g", h, v)
    cached = CACHE.get(key)
    if cached is not None:
        return cached
    adj = [[] for _ in range(81)]
    for row in range(9):
        for col in range(9):
            cell = row * 9 + col
            if row < 8:
                blocked = ((col < 8 and h & (1 << (row * 8 + col))) or
                           (col > 0 and h & (1 << (row * 8 + col - 1))))
                if not blocked:
                    adj[cell].append(cell + 9)
                    adj[cell + 9].append(cell)
            if col < 8:
                blocked = ((row < 8 and v & (1 << (row * 8 + col))) or
                           (row > 0 and v & (1 << ((row - 1) * 8 + col))))
                if not blocked:
                    adj[cell].append(cell + 1)
                    adj[cell + 1].append(cell)
    result = tuple(tuple(neighbors) for neighbors in adj)
    CACHE.put(key, result)
    return result


def distances(h, v, goal):
    key = ("d", h, v, goal)
    cached = CACHE.get(key)
    if cached is not None:
        return cached
    adj = graph(h, v)
    distance = [-1] * 81
    queue = deque(range(goal * 9, goal * 9 + 9))
    for cell in queue:
        distance[cell] = 0
    while queue:
        cell = queue.popleft()
        next_distance = distance[cell] + 1
        for neighbor in adj[cell]:
            if distance[neighbor] < 0:
                distance[neighbor] = next_distance
                queue.append(neighbor)
    result = tuple(distance)
    CACHE.put(key, result)
    return result


def pawn_moves(state, player=None):
    player = state.turn % 2 if player is None else player
    pawn, other = (state.p1, state.p0) if player else (state.p0, state.p1)
    adj = graph(state.h, state.v)
    moves = []
    for target in adj[pawn]:
        if target != other:
            moves.append(target)
            continue
        dr = other // 9 - pawn // 9
        dc = other % 9 - pawn % 9
        row, col = other // 9 + dr, other % 9 + dc
        behind = row * 9 + col
        if 0 <= row < 9 and 0 <= col < 9 and behind in adj[other]:
            moves.append(behind)
        else:
            for side in adj[other]:
                if ((dr and side // 9 == other // 9) or
                        (dc and side % 9 == other % 9)):
                    moves.append(side)
    return sorted(set(moves))


def wall_geometry_ok(state, action):
    if not 81 <= action < PASS:
        return False
    slot = (action - 81) % 64
    row, col = divmod(slot, 8)
    bit = 1 << slot
    if (state.h | state.v) & bit:
        return False
    if action < 145:
        return not (state.h & ((bit << 1 if col < 7 else 0) |
                               (bit >> 1 if col > 0 else 0)))
    return not (state.v & ((bit << 8 if row < 7 else 0) |
                           (bit >> 8 if row > 0 else 0)))


def add_wall(state, action):
    bit = 1 << ((action - 81) % 64)
    return state._replace(h=state.h | bit) if action < 145 else state._replace(v=state.v | bit)


def wall_legal(state, action):
    if not wall_geometry_ok(state, action):
        return False
    trial = add_wall(state, action)
    return (distances(trial.h, trial.v, state.goal0)[state.p0] >= 0 and
            distances(trial.h, trial.v, state.goal1)[state.p1] >= 0)


def outcome(state):
    """Winner 0/1, draw 2, or None. A win on the last ply beats the draw."""
    if state.p0 // 9 == state.goal0:
        return 0
    if state.p1 // 9 == state.goal1:
        return 1
    return 2 if state.turn >= state.max_turns else None


def legal_actions(state):
    if outcome(state) is not None:
        return []
    actions = pawn_moves(state) + [PASS]
    if (state.w1 if state.turn % 2 else state.w0) > 0:
        actions.extend(a for a in range(81, PASS) if wall_legal(state, a))
    return actions


def play(state, action, validate=True):
    if validate and action not in legal_actions(state):
        raise ValueError("illegal action")
    player = state.turn % 2
    if action < 81:
        state = state._replace(p1=action) if player else state._replace(p0=action)
    elif action != PASS:
        state = add_wall(state, action)
        state = state._replace(w1=state.w1 - 1) if player else state._replace(w0=state.w0 - 1)
    return state._replace(turn=state.turn + 1)


def transform_cell(cell, rotate=False, mirror=False):
    row, col = divmod(cell, 9)
    if rotate:
        row, col = 8 - row, 8 - col
    if mirror:
        col = 8 - col
    return row * 9 + col


def transform_action(action, rotate=False, mirror=False):
    if action < 81:
        return transform_cell(action, rotate, mirror)
    if action == PASS:
        return action
    row, col = divmod((action - 81) % 64, 8)
    if rotate:
        row, col = 7 - row, 7 - col
    if mirror:
        col = 7 - col
    return wall_action("H" if action < 145 else "V", row, col)


def encode(state, mirror=False):
    """651 inputs: eight 9x9 planes and three normalized scalar features."""
    player = state.turn % 2
    own, other = (state.p1, state.p0) if player else (state.p0, state.p1)
    own_goal, other_goal = (state.goal1, state.goal0) if player else (state.goal0, state.goal1)
    rotate = own_goal == 8
    features = [0.0] * 651
    features[transform_cell(own, rotate, mirror)] = 1.0
    features[81 + transform_cell(other, rotate, mirror)] = 1.0
    for orientation, bits, offset in (("H", state.h, 162), ("V", state.v, 243)):
        while bits:
            bit = bits & -bits
            slot = bit.bit_length() - 1
            action = transform_action(81 + (64 if orientation == "V" else 0) + slot, rotate, mirror)
            row, col = divmod((action - 81) % 64, 8)
            features[offset + row * 9 + col] = 1.0
            bits -= bit
    for offset, goal in ((324, own_goal), (405, other_goal)):
        for cell, distance in enumerate(distances(state.h, state.v, goal)):
            features[offset + transform_cell(cell, rotate, mirror)] = distance / 80.0
    for offset, goal in ((486, own_goal), (567, other_goal)):
        for col in range(9):
            features[offset + transform_cell(goal * 9 + col, rotate, mirror)] = 1.0
    features[648] = (state.w1 if player else state.w0) / 10.0
    features[649] = (state.w0 if player else state.w1) / 10.0
    features[650] = max(0, state.max_turns - state.turn) / float(state.max_turns)
    return features


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


def heuristic(state):
    """Value from the side-to-move perspective, measured roughly in pawn plies."""
    player = state.turn % 2
    winner = outcome(state)
    if winner is not None:
        return 0.0 if winner == 2 else (10000.0 if winner == player else -10000.0)
    key = ("e", state)
    cached = CACHE.get(key)
    if cached is not None:
        return cached
    d0 = distances(state.h, state.v, state.goal0)
    d1 = distances(state.h, state.v, state.goal1)
    adj = graph(state.h, state.v)
    score = []
    for who, pawn, distance, walls in ((0, state.p0, d0, state.w0), (1, state.p1, d1, state.w1)):
        moves = pawn_moves(state, who)
        # Count genuinely forward alternatives, and account for immediate jumps.
        alternatives = sum(distance[n] < distance[pawn] for n in moves)
        effective = min([distance[pawn]] + [1 + distance[n] for n in moves])
        bottleneck = max(0, 2 - len(adj[pawn]))
        reserve = EVAL_WEIGHTS[0] * walls + EVAL_WEIGHTS[1] * min(walls, 3) * min(effective, 6)
        score.append(-effective + reserve + EVAL_WEIGHTS[2] * alternatives + EVAL_WEIGHTS[3] * len(moves) - EVAL_WEIGHTS[4] * bottleneck)
    result = score[player] - score[1 - player] + EVAL_WEIGHTS[5]
    CACHE.put(key, result)
    return result


class SearchTimeout(Exception):
    pass


class Search:
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

    def check(self):
        if time.monotonic() >= self.deadline:
            raise SearchTimeout()

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
            value = self.value_net.value(state) * VALUE_SCALE + HEURISTIC_WEIGHT * heuristic(state)
            CACHE.put(key, value)
        return value

    def fallback(self, state, actions):
        player = state.turn % 2
        goal = state.goal1 if player else state.goal0
        ds = distances(state.h, state.v, goal)
        moves = [a for a in actions if a < 81]
        return min(moves, key=lambda a: (ds[a], abs(a % 9 - 4), a)) if moves else PASS

    def ordered(self, state, root_actions=None, preferred=None):
        self.check()
        player = state.turn % 2
        own = state.p1 if player else state.p0
        other = state.p0 if player else state.p1
        goal = state.goal1 if player else state.goal0
        other_goal = state.goal0 if player else state.goal1
        moves = [a for a in (root_actions if root_actions is not None else pawn_moves(state)) if a < 81]
        if root_actions is None:
            candidates = range(81, PASS) if (state.w1 if player else state.w0) else []
        else:
            candidates = [a for a in root_actions if 81 <= a < PASS]
        d_own = distances(state.h, state.v, goal)
        d_other = distances(state.h, state.v, other_goal)
        enemy_threat = any(a // 9 == other_goal for a in pawn_moves(state, 1 - player))
        logits = self.root_logits if root_actions is not None else None   # policy only at the root
        ranked_walls, defensive = [], []
        for action in candidates:
            self.check()
            if root_actions is None and not wall_legal(state, action):
                continue
            child = play(state, action, False)
            do = distances(child.h, child.v, goal)[own] - d_own[own]
            de = distances(child.h, child.v, other_goal)[other] - d_other[other]
            slot = (action - 81) % 64
            row, col = divmod(slot, 8)
            near = min(abs(row - own // 9) + abs(col - own % 9),
                       abs(row - other // 9) + abs(col - other % 9))
            # Existing walls reward possible combinations, including zero-immediate-gain walls.
            bits = state.h | state.v
            connected = any(0 <= r < 8 and 0 <= c < 8 and bits & (1 << (r * 8 + c))
                            for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)))
            value = de - 1.15 * do - 0.04 * near + 0.15 * connected
            if logits:
                value += POLICY_WEIGHT * max(-10.0, min(10.0, logits[action]))
            ranked_walls.append((value, action))
            if enemy_threat and not any(a // 9 == other_goal for a in pawn_moves(child, 1 - player)):
                defensive.append(action)
        ranked_walls.sort(reverse=True)
        wall_actions = [a for _, a in ranked_walls[:self.wall_limit]]
        actions = list(dict.fromkeys(moves + [PASS] + defensive + wall_actions))
        if root_actions is not None:
            allowed = set(root_actions)
            actions = [a for a in actions if a in allowed]
        scored = []
        for action in actions:
            self.check()
            child = play(state, action, False)
            score = -heuristic(child)
            if action == preferred:
                score += 100000.0
            if logits:
                score += POLICY_WEIGHT * max(-10.0, min(10.0, logits[action]))
            if outcome(child) == player:
                score += 1000000.0
            # Immediate losing replies are never hidden by a selective wall list.
            if enemy_threat and outcome(child) is None:
                if any(a // 9 == other_goal for a in pawn_moves(child)):
                    score -= 100000.0
            scored.append((score, action))
        scored.sort(reverse=True)
        return [action for _, action in scored]

    def negamax(self, state, depth, alpha, beta):
        self.check()
        self.nodes += 1
        if depth == 0 or outcome(state) is not None:
            return self.evaluate(state)
        key = ("t", self.mode, self.wall_limit, id(self.value_net), state)
        entry = CACHE.get(key)
        preferred = entry[3] if entry else None
        original_alpha, original_beta = alpha, beta
        if entry and entry[0] >= depth:
            _, value, flag, _ = entry
            if flag == 0:
                return value
            if flag == 1:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value
        value, best = -float("inf"), None
        for action in self.ordered(state, preferred=preferred):
            score = -self.negamax(play(state, action, False), depth - 1, -beta, -alpha)
            if score > value:
                value, best = score, action
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        flag = -1 if value <= original_alpha else (1 if value >= original_beta else 0)
        CACHE.put(key, (depth, value, flag, best))
        return value

    def choose(self, state, root_actions=None):
        started = time.monotonic()
        self.deadline = started + max(0.0, self.budget)
        self.nodes, self.depth, self.completed = 0, 0, False
        if outcome(state) is not None:
            return PASS
        self.root_logits = None
        if root_actions is None:
            root_actions = legal_actions(state)   # training calls choose(state) without Arena actions
        # Avoid expensive wall generation before a fallback exists.
        actions = root_actions if root_actions is not None else pawn_moves(state) + [PASS]
        best = self.fallback(state, actions)
        goal = state.goal1 if state.turn % 2 else state.goal0
        wins = [a for a in actions if a < 81 and a // 9 == goal]
        if wins:
            self.completed = True
            return wins[0]
        try:
            enemy_goal = state.goal0 if state.turn % 2 else state.goal1
            if any(a // 9 == enemy_goal for a in pawn_moves(state, 1 - state.turn % 2)):
                # Establish a safe fallback BEFORE expensive wall ranking. When
                # the caller supplies Arena actions no speculative legality is used.
                responses = root_actions if root_actions is not None else legal_actions(state)
                safe = []
                for action in responses:
                    self.check()
                    child = play(state, action, False)
                    if outcome(child) is not None or not any(a // 9 == enemy_goal for a in pawn_moves(child)):
                        safe.append(action)
                        best = action
                if safe:
                    root_actions = safe
                    best = self.fallback(state, safe) if any(a < 81 for a in safe) else safe[0]
            if self.policy is not None:
                self.root_logits = self.policy.logits(state)
                self.check()
            ordered = self.ordered(state, root_actions, best)
            if not ordered:
                return best
            best = ordered[0]
            for depth in range(1, state.max_turns - state.turn + 1):
                self.check()
                value, iteration_best, alpha = -float("inf"), best, -float("inf")
                for action in ordered:
                    score = -self.negamax(play(state, action, False), depth - 1, -float("inf"), -alpha)
                    if score > value:
                        value, iteration_best = score, action
                    alpha = max(alpha, value)
                best, self.depth, self.completed = iteration_best, depth, True
                ordered.remove(best)
                ordered.insert(0, best)
                if abs(value) >= 9999:
                    break
        except SearchTimeout:
            pass
        return best


def _position(player):
    value = player.get("position", player.get("pos"))
    if value is None and "row" in player and "col" in player:
        value = (player["row"], player["col"])
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("position must be [row, col]; official schema not yet verified")
    row, col = value
    if type(row) is not int or type(col) is not int or not (0 <= row < 9 and 0 <= col < 9):
        raise ValueError("invalid pawn coordinates")
    return row * 9 + col


def _walls_left(player):
    for key in ("walls_remaining", "walls_left", "remaining_walls"):
        if key in player:
            value = player[key]
            if type(value) is int and 0 <= value <= 10:
                return value
    raise ValueError("missing wall inventory; official schema not yet verified")


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


def observation_for(state):
    """Documented development schema, not a claim about the official schema."""
    players = [{"position": [p // 9, p % 9], "goal_row": g, "walls_remaining": w}
               for p, g, w in ((state.p0, state.goal0, state.w0), (state.p1, state.goal1, state.w1))]
    walls = [action_dict(a) for a in range(81, PASS)
             if (state.h if a < 145 else state.v) & (1 << ((a - 81) % 64))]
    for wall in walls:
        del wall["type"]
    actions = legal_actions(state)
    goal = state.goal1 if state.turn % 2 else state.goal0
    ds = distances(state.h, state.v, goal)
    return {"you": players[state.turn % 2], "opponent": players[1 - state.turn % 2],
            "walls": walls, "turn": state.turn, "max_turns": state.max_turns,
            "legal_moves": [[a // 9, a % 9] for a in actions if a < 81],
            "legal_walls": [dict((k, v) for k, v in action_dict(a).items() if k != "type")
                            for a in actions if 81 <= a < PASS],
            "distances": [list(ds[row * 9:row * 9 + 9]) for row in range(9)]}


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
