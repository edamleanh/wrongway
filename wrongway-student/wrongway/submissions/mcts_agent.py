"""WrongWay MCTS, standalone Python standard library.

The local MCTS survey motivates short stochastic heavy playouts, implicit
minimax, progressive widening/bias and tree reuse. Sample means, heuristic
minimax estimates and exact solver certificates are deliberately separate.
Tree values always use the ROOT player's perspective.
"""

from collections import OrderedDict, deque, namedtuple
from math import log, sqrt
import random
from time import monotonic

BOARD_SIZE = 9
TIME_BUDGET = .70
WALL_WEIGHT = 2.0
JUMP_WEIGHT = .5
ROLLOUT_PLIES = 4
EXPLORATION = .65
IMPLICIT_WEIGHT = .35
PROGRESSIVE_BIAS = .35
WIDENING_SCALE = 2.5
WIDENING_EXPONENT = .55
NODE_LIMIT = 4000
BOARD_CACHE_LIMIT = 768
TREE_DEPTH_LIMIT = 80
PASS = 209
LAST_SEARCH_INFO = {}
State = namedtuple("State", "board_size positions goal_rows walls_remaining "
                   "horizontal_walls vertical_walls current_player turn max_turns")

# Internal cells: row*9+col. Walls: H=81..144, V=145..208.
BASE_GRAPH = tuple(tuple(nr * 9 + nc for dr, dc in
                        ((-1, 0), (0, -1), (0, 1), (1, 0))
                        for nr, nc in ((r + dr, c + dc),)
                        if 0 <= nr < 9 and 0 <= nc < 9)
                   for r in range(9) for c in range(9))
WALL_EDGES, OVERLAP_MASKS, EDGE_WALLS = [], [], {}
for orientation in ("H", "V"):
    for row in range(8):
        for col in range(8):
            cell = row * 9 + col
            edges = (((cell, cell + 9), (cell + 1, cell + 10))
                     if orientation == "H" else
                     ((cell, cell + 1), (cell + 9, cell + 10)))
            code = 81 + len(WALL_EDGES)
            WALL_EDGES.append(edges)
            mask = 0
            for delta in (-1, 0, 1):
                r, c = (row, col + delta) if orientation == "H" else (row + delta, col)
                if 0 <= r < 8 and 0 <= c < 8:
                    mask |= 1 << (r * 8 + c)
            OVERLAP_MASKS.append(mask)
            for a, b in edges:
                EDGE_WALLS.setdefault((min(a, b), max(a, b)), []).append(code)
WALL_EDGES = tuple(WALL_EDGES)
OVERLAP_MASKS = tuple(OVERLAP_MASKS)
EDGE_WALLS = {edge: tuple(codes) for edge, codes in EDGE_WALLS.items()}
_BOARD_CACHE = OrderedDict()
_SAVED_ROOT = None
_SAVED_PLAYER = None


class SearchTimeout(Exception):
    """Internal search deadline, before the external one-second deadline."""


class SearchContext:
    def __init__(self, deadline, root_player):
        self.deadline = deadline
        self.root_player = root_player
        self.nodes = 0
        self.cache_hits = 0

    def check_time(self):
        if monotonic() >= self.deadline:
            raise SearchTimeout()

    def board_data(self, state, parent_board=None, added_wall=None):
        """Bounded LRU: graph and COMPLETE BFS maps, independent of pawn positions."""
        self.check_time()
        key = (state.horizontal_walls, state.vertical_walls, state.goal_rows)
        data = _BOARD_CACHE.get(key)
        if data is not None:
            _BOARD_CACHE.move_to_end(key)
            self.cache_hits += 1
            return data
        graph = (add_wall(parent_board[0], added_wall) if parent_board is not None
                 and added_wall is not None else build_graph(state))
        data = (graph, tuple(distance_map(graph, goal) for goal in state.goal_rows))
        _BOARD_CACHE[key] = data
        if len(_BOARD_CACHE) > BOARD_CACHE_LIMIT:
            _BOARD_CACHE.popitem(last=False)
        return data


def state_from_observation(observation):
    positions, goals, inventories = [0, 0], [0, 0], [0, 0]
    player = observation["player_id"]
    for who, info in ((player, observation["you"]), (1 - player, observation["opponent"])):
        row, col = info["position"]
        positions[who] = row * 9 + col
        goals[who] = info["goal_row"]
        inventories[who] = info["walls_remaining"]
    masks = [sum(1 << (r * 8 + c) for r, c in observation["walls"][kind])
             for kind in ("horizontal", "vertical")]
    return State(observation["board_size"], tuple(positions), tuple(goals),
                 tuple(inventories), *masks, observation["current_player"],
                 observation["turn"], observation["max_turns"])


def encode_action(action):
    if action["type"] == "move":
        return action["to"][0] * 9 + action["to"][1]
    if action["type"] == "wall":
        return 81 + (64 if action["orientation"] == "V" else 0) + action["row"] * 8 + action["col"]
    return PASS


def decode_action(code):
    if code < 81:
        return {"type": "move", "to": list(divmod(code, 9))}
    if code == PASS:
        return {"type": "pass"}
    index = code - 81
    row, col = divmod(index % 64, 8)
    return {"type": "wall", "orientation": "H" if index < 64 else "V", "row": row, "col": col}


def add_wall(graph, code):
    """A wall changes four adjacency entries; reuse the other 77."""
    child = list(graph)
    for a, b in WALL_EDGES[code - 81]:
        child[a] = tuple(cell for cell in child[a] if cell != b)
        child[b] = tuple(cell for cell in child[b] if cell != a)
    return tuple(child)


def build_graph(state):
    graph = BASE_GRAPH
    for offset, mask in ((0, state.horizontal_walls), (64, state.vertical_walls)):
        while mask:
            bit = mask & -mask
            graph = add_wall(graph, 81 + offset + bit.bit_length() - 1)
            mask ^= bit
    return graph


def distance_map(graph, goal_row, size=9):
    """Exact multi-source BFS on the wall graph, ignoring pawn obstruction."""
    distances = [-1] * 81
    pending = deque(range(goal_row * 9, goal_row * 9 + size))
    for cell in pending:
        distances[cell] = 0
    while pending:
        cell = pending.popleft()
        next_distance = distances[cell] + 1
        for neighbor in graph[cell]:
            if distances[neighbor] == -1:
                distances[neighbor] = next_distance
                pending.append(neighbor)
    return tuple(distances)


def terminal_value(state):
    """Exact result for SIDE TO MOVE; reaching goal precedes the turn-limit draw."""
    for player in (0, 1):
        if state.positions[player] // 9 == state.goal_rows[player]:
            return 1. if player == state.current_player else -1.
    return 0. if state.turn >= state.max_turns else None


def legal_moves(state, player=None, graph=None):
    if terminal_value(state) is not None:
        return []
    player = state.current_player if player is None else player
    graph = build_graph(state) if graph is None else graph
    position, opponent = state.positions[player], state.positions[1 - player]
    destinations = set()
    for adjacent in graph[position]:
        if adjacent != opponent:
            destinations.add(adjacent)
        else:
            beyond = opponent + opponent - position
            if beyond in graph[opponent]:
                destinations.add(beyond)
            else:
                delta = opponent - position
                for neighbor in graph[opponent]:
                    if (abs(delta) == 9 and neighbor // 9 == opponent // 9
                            or abs(delta) == 1 and neighbor % 9 == opponent % 9):
                        destinations.add(neighbor)
    return sorted(destinations)


def wall_fits(state, orientation, row, col):
    """Geometry only; two BFS maps separately check routes."""
    if orientation not in ("H", "V") or type(row) is not int or type(col) is not int:
        return False
    if not (0 <= row < 8 and 0 <= col < 8):
        return False
    index = row * 8 + col
    same, other, offset = ((state.horizontal_walls, state.vertical_walls, 0)
                           if orientation == "H" else
                           (state.vertical_walls, state.horizontal_walls, 64))
    return not (same & OVERLAP_MASKS[offset + index] or other & (1 << index))


def apply_action(state, code):
    """Apply an ALREADY LEGAL action; preserve the parent immutable state."""
    player = state.current_player
    changes = {"turn": state.turn + 1, "current_player": 1 - player}
    if code < 81:
        positions = list(state.positions)
        positions[player] = code
        changes["positions"] = tuple(positions)
    elif code != PASS:
        index = code - 81
        name = "horizontal_walls" if index < 64 else "vertical_walls"
        changes[name] = getattr(state, name) | (1 << (index % 64))
        inventories = list(state.walls_remaining)
        inventories[player] -= 1
        changes["walls_remaining"] = tuple(inventories)
    return state._replace(**changes)


def child_board(state, action, context, parent_board):
    return (context.board_data(state, parent_board, action)
            if 81 <= action < PASS else parent_board)


def has_routes(state, board):
    return all(board[1][p][state.positions[p]] >= 0 for p in (0, 1))


def legal_actions(state, context=None):
    """Full legal set for rule checks; tree search validates wall routes lazily."""
    if terminal_value(state) is not None:
        return []
    context = context or SearchContext(float("inf"), state.current_player)
    board = context.board_data(state)
    actions = legal_moves(state, graph=board[0]) + [PASS]
    if state.walls_remaining[state.current_player]:
        for code in range(81, PASS):
            info = decode_action(code)
            if wall_fits(state, info["orientation"], info["row"], info["col"]):
                child = apply_action(state, code)
                if has_routes(child, child_board(child, code, context, board)):
                    actions.append(code)
    return sorted(actions)


def winning_moves(state, graph, player=None):
    player = state.current_player if player is None else player
    if abs(state.positions[player] // 9 - state.goal_rows[player]) > 2:
        return []
    return [move for move in legal_moves(state, player, graph)
            if move // 9 == state.goal_rows[player]]


def root_value(state, board, root_player):
    """Bounded heuristic; strictly between exact loss -1 and exact win +1."""
    exact = terminal_value(state)
    if exact is not None:
        return exact if state.current_player == root_player else -exact
    other = 1 - root_player
    distances = [board[1][p][state.positions[p]] for p in (0, 1)]
    jumps = []
    for p in (0, 1):
        moves = legal_moves(state, p, board[0])
        best = min((board[1][p][cell] for cell in moves), default=distances[p])
        jumps.append(distances[p] - 1 - best)
    score = (distances[other] - distances[root_player]
             + WALL_WEIGHT * (state.walls_remaining[root_player] - state.walls_remaining[other])
             + JUMP_WEIGHT * (jumps[root_player] - jumps[other]))
    return score / (4. + abs(score))


def path_wall_candidates(state, board, player, steps=4):
    """Small rollout set: walls touching one shortest-path witness."""
    current = state.positions[player]
    distances = board[1][player]
    proposals = set()
    for _ in range(steps):
        if distances[current] <= 0:
            break
        neighbor = next((cell for cell in board[0][current]
                         if distances[cell] == distances[current] - 1), None)
        if neighbor is None:
            break
        edge = (min(current, neighbor), max(current, neighbor))
        proposals.update(EDGE_WALLS[edge])
        current = neighbor
    return proposals


def ordered_proposals(state, board, context):
    """Keep EVERY geometric action; this ordering is not a top-k restriction."""
    player = state.current_player
    sign = 1 if player == context.root_player else -1
    proposals = []
    for action in legal_moves(state, graph=board[0]):
        child = apply_action(state, action)
        proposals.append((action, sign * root_value(child, board, context.root_player)))
    important = path_wall_candidates(state, board, 1 - player)
    if state.walls_remaining[player]:
        for action in range(81, PASS):
            index = action - 81
            row, col = divmod(index % 64, 8)
            if wall_fits(state, "H" if index < 64 else "V", row, col):
                proposals.append((action, .05 if action in important else -.45))
    proposals.append((PASS, -1.5))
    return sorted(proposals, key=lambda pair: (-pair[1], pair[0]))


class Node:
    __slots__ = ("state", "action", "prior", "children", "pending", "next_index",
                 "visits", "value_sum", "heuristic", "minimax", "solved", "coverage")

    def __init__(self, state, action, prior, heuristic):
        self.state, self.action, self.prior = state, action, prior
        self.children, self.pending = [], None
        self.next_index, self.visits, self.value_sum = 0, 0, 0.
        self.heuristic = self.minimax = heuristic
        self.solved = None if terminal_value(state) is None else heuristic
        self.coverage = True


def prepare_node(node, board, context):
    if node.pending is not None or node.solved is not None:
        return
    if winning_moves(node.state, board[0]):
        node.solved = 1. if node.state.current_player == context.root_player else -1.
        node.minimax = node.solved
        return
    node.pending = ordered_proposals(node.state, board, context)


def update_node(node, context):
    """Partial-tree minimax is a heuristic; only full coverage certifies loss."""
    if not node.children:
        return
    maximize = node.state.current_player == context.root_player
    choose = max if maximize else min
    node.minimax = choose(child.minimax for child in node.children)
    winning = 1. if maximize else -1.
    if any(child.solved == winning for child in node.children):
        node.solved = winning
    elif (node.coverage and node.pending is not None
          and node.next_index == len(node.pending)
          and all(child.solved is not None for child in node.children)):
        node.solved = choose(child.solved for child in node.children)
    if node.solved is not None:
        node.minimax = node.solved


def select_child(node, context):
    sign = 1 if node.state.current_player == context.root_player else -1
    available = [child for child in node.children if child.solved is None]
    if not available:
        available = node.children
    exploration = EXPLORATION * sqrt(log(node.visits + 2))

    def score(child):
        if not child.visits:
            return float("inf")
        mean = child.value_sum / child.visits
        estimate = (1 - IMPLICIT_WEIGHT) * mean + IMPLICIT_WEIGHT * child.minimax
        return (sign * estimate + exploration / sqrt(child.visits)
                + PROGRESSIVE_BIAS * child.prior / (child.visits + 1))
    return max(available, key=score)


def expand(node, board, context):
    while node.next_index < len(node.pending) and context.nodes < NODE_LIMIT:
        context.check_time()
        action, prior = node.pending[node.next_index]
        child = apply_action(node.state, action)
        data = child_board(child, action, context, board)
        # Do not consume an action if validation timed out before completion.
        node.next_index += 1
        if not has_routes(child, data):
            continue
        child_node = Node(child, action, prior, root_value(child, data, context.root_player))
        node.children.append(child_node)
        context.nodes += 1
        return child_node, data
    return None, None


def rollout(state, board, context):
    """Short stochastic heavy playout; avoid scanning 128 walls every ply."""
    for _ in range(ROLLOUT_PLIES):
        context.check_time()
        if terminal_value(state) is not None:
            break
        player = state.current_player
        sign = 1 if player == context.root_player else -1
        moves = legal_moves(state, graph=board[0])
        wins = [move for move in moves if move // 9 == state.goal_rows[player]]
        if wins:
            state = apply_action(state, wins[0])
            break
        actions = list(moves)
        if state.walls_remaining[player] and random.random() < .30:
            proposals = sorted(path_wall_candidates(state, board, 1 - player))
            random.shuffle(proposals)
            actions.extend(proposals[:6])
        choices = []
        for action in actions or [PASS]:
            context.check_time()
            if 81 <= action < PASS:
                info = decode_action(action)
                if not wall_fits(state, info["orientation"], info["row"], info["col"]):
                    continue
            child = apply_action(state, action)
            data = child_board(child, action, context, board)
            if not has_routes(child, data):
                continue
            value = sign * root_value(child, data, context.root_player)
            # Test after our move: moving a pawn can CREATE an enemy goal jump.
            if winning_moves(child, data[0]):
                value = -2.
            choices.append((value, action, child, data))
        if not choices:
            state = apply_action(state, PASS)
            continue
        choices.sort(key=lambda item: (-item[0], item[1]))
        choice = choices[0] if random.random() < .85 else random.choice(choices[:3])
        state, board = choice[2], choice[3]
    return root_value(state, board, context.root_player)


def simulate(root, context):
    node = root
    board = context.board_data(node.state)
    path = [node]
    for _ in range(TREE_DEPTH_LIMIT):
        context.check_time()
        prepare_node(node, board, context)
        if node.solved is not None:
            value = node.solved
            break
        width = max(1, int(WIDENING_SCALE * (node.visits + 1) ** WIDENING_EXPONENT))
        if len(node.children) < width and context.nodes < NODE_LIMIT:
            child, data = expand(node, board, context)
            if child is not None:
                path.append(child)
                prepare_node(child, data, context)
                value = child.solved if child.solved is not None else rollout(child.state, data, context)
                break
        if not node.children:
            value = rollout(node.state, board, context)
            break
        node = select_child(node, context)
        path.append(node)
        board = context.board_data(node.state)
    else:
        value = root_value(node.state, board, context.root_player)
    for ancestor in reversed(path):
        ancestor.visits += 1
        ancestor.value_sum += value
        update_node(ancestor, context)


def root_actions(observation):
    actions = [row * 9 + col for row, col in observation["legal_moves"]]
    for wall in observation["legal_walls"]:
        actions.append(encode_action(dict(wall, type="wall")))
    return actions + [PASS]


def reuse_root(state, player):
    """Match the exact state after our selected move and the opponent reply."""
    if _SAVED_ROOT is None or _SAVED_PLAYER != player or state.turn == 0:
        return None
    return next((node for node in [_SAVED_ROOT] + _SAVED_ROOT.children
                 if node.state == state), None)


def count_nodes(root):
    stack, count = [root], 0
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(node.children)
    return count


def choose_action(observation):
    """Exact root tactics -> timed MCTS -> robust legal action, with fallback."""
    global LAST_SEARCH_INFO, _SAVED_ROOT, _SAVED_PLAYER
    started = monotonic()
    actions = root_actions(observation)
    fallback = actions[0] if actions else PASS
    LAST_SEARCH_INFO = {"simulations": 0, "reused": False, "nodes": 0}
    if observation.get("board_size", 9) != 9:
        return decode_action(fallback)
    state = state_from_observation(observation)
    player = observation["player_id"]
    context = SearchContext(started + TIME_BUDGET, player)
    root, reused, old_visits = None, False, 0
    try:
        board = context.board_data(state)
        wins = winning_moves(state, board[0])
        if wins:
            fallback = wins[0]
            LAST_SEARCH_INFO["immediate_win"] = True
            _SAVED_ROOT = None
            return decode_action(fallback)
        ranked = []
        danger = abs(state.positions[1 - player] // 9 - state.goal_rows[1 - player]) <= 2
        for action in actions:
            context.check_time()
            child = apply_action(state, action)
            data = child_board(child, action, context, board)
            value = root_value(child, data, player)
            unsafe = danger and terminal_value(child) is None and bool(winning_moves(child, data[0]))
            ranked.append((action, value, unsafe))
        safe = [item for item in ranked if not item[2]]
        usable = safe or ranked
        usable.sort(key=lambda item: (-item[1], item[0]))
        fallback = usable[0][0]
        root = reuse_root(state, player)
        # Release the previous root and its siblings before growing a new tree.
        # Nodes have no parent pointers, so only the reached subtree survives.
        _SAVED_ROOT = None
        reused = root is not None
        if root is None:
            root = Node(state, None, 0., root_value(state, board, player))
        old_visits = root.visits
        allowed = {item[0] for item in usable}
        root.children = [child for child in root.children if child.action in allowed]
        expanded = {child.action for child in root.children}
        root.pending = [(action, value) for action, value, _ in usable if action not in expanded]
        root.next_index = 0
        root.coverage = len(usable) == len(actions)
        context.nodes = count_nodes(root)
        while root.solved is None:
            context.check_time()
            simulate(root, context)
    except SearchTimeout:
        pass
    finally:
        if root is not None:
            children = root.children
            proven = [child for child in children if child.solved == 1.]
            available = proven or [child for child in children if child.solved != -1.]
            if available:
                selected = max(available, key=lambda child: (child.visits,
                    (1 - IMPLICIT_WEIGHT) * child.value_sum / max(1, child.visits)
                    + IMPLICIT_WEIGHT * child.minimax, child.prior))
                fallback = selected.action
                _SAVED_ROOT, _SAVED_PLAYER = selected, player
            else:
                _SAVED_ROOT = None
            LAST_SEARCH_INFO.update(nodes=context.nodes, root_children=len(children),
                root_actions=len(actions), root_visits=root.visits, cache_hits=context.cache_hits,
                cache_boards=len(_BOARD_CACHE), reused=reused, solved=root.solved,
                simulations=root.visits - old_visits)
        LAST_SEARCH_INFO["seconds"] = monotonic() - started
    return decode_action(fallback)
