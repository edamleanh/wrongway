"""MCTS Agent with Adversarial Opponent Perspective Check.

Based on monte_carlo.py, with an adversarial evaluation policy:
When performing Monte Carlo Tree Search at the opponent's turn, it reuses
the current heuristic perspective (HeuristicEngine) to check if the opponent
has an advantageous counter-move (e.g. winning advance, path takeover, or massive wall).
If an advantageous move exists with score R_opp, the simulation reward
is assigned as -R_opp.
"""

import heapq
import math
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOARD_SIZE = 9
DIRECTIONS = ((-1, 0), (0, -1), (0, 1), (1, 0))
CPUCT = 1.41         # Exploration constant for PUCT
EPSILON = 0.15       # Random move probability during rollout
ROLLOUT_DEPTH = 60   # Max rollout steps before cutoff
TIME_FRACTION = 0.55 # Fraction of 1-second budget to ensure zero timeouts
ROOT_WALL_DEPTH = 4  # Minimax plies used to judge a wall at the search root
INNER_WALL_DEPTH = 1 # Cheap A* race check for walls inside the tree
WALL_DELTA_MIN = 1   # Minimum (delta_opp - delta_self) for a wall to be considered

Wall = Tuple[str, int, int]

# ===================================================================
# Lightweight Game State for Fast Simulation
# ===================================================================
class FastState:
    """Minimal Quoridor state supporting cloning, fast edge checks, and wall checks."""

    __slots__ = (
        "positions", "goal_rows", "walls_remaining",
        "h_walls", "v_walls", "current_player", "done", "winner",
    )

    def __init__(self):
        self.positions: List[List[int]] = [[8, 4], [0, 4]]
        self.goal_rows: List[int] = [0, 8]
        self.walls_remaining: List[int] = [10, 10]
        self.h_walls: Set[Tuple[int, int]] = set()
        self.v_walls: Set[Tuple[int, int]] = set()
        self.current_player: int = 0
        self.done: bool = False
        self.winner: Optional[int] = None

    def clone(self) -> "FastState":
        s = FastState.__new__(FastState)
        s.positions = [[self.positions[0][0], self.positions[0][1]],
                       [self.positions[1][0], self.positions[1][1]]]
        s.goal_rows = self.goal_rows
        s.walls_remaining = [self.walls_remaining[0], self.walls_remaining[1]]
        s.h_walls = set(self.h_walls)
        s.v_walls = set(self.v_walls)
        s.current_player = self.current_player
        s.done = self.done
        s.winner = self.winner
        return s

    @staticmethod
    def from_observation(obs: Dict[str, Any]) -> "FastState":
        s = FastState.__new__(FastState)
        pid = obs["player_id"]
        opp = 1 - pid
        s.positions = [None, None]
        s.positions[pid] = list(obs["you"]["position"])
        s.positions[opp] = list(obs["opponent"]["position"])
        s.goal_rows = [None, None]
        s.goal_rows[pid] = obs["you"]["goal_row"]
        s.goal_rows[opp] = obs["opponent"]["goal_row"]
        s.walls_remaining = [None, None]
        s.walls_remaining[pid] = obs["you"]["walls_remaining"]
        s.walls_remaining[opp] = obs["opponent"]["walls_remaining"]
        s.h_walls = {tuple(w) for w in obs["walls"]["horizontal"]}
        s.v_walls = {tuple(w) for w in obs["walls"]["vertical"]}
        s.current_player = obs["current_player"]
        s.done = obs["done"]
        s.winner = obs["winner"]
        return s

    # ---- Edge & adjacency helpers ----

    def _edge_open(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        if not (0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE):
            return False
        if r1 != r2:  # vertical movement -> blocked by horizontal walls
            er = r1 if r1 < r2 else r2
            return (er, c1) not in self.h_walls and (er, c1 - 1) not in self.h_walls
        # horizontal movement -> blocked by vertical walls
        ec = c1 if c1 < c2 else c2
        return (r1, ec) not in self.v_walls and (r1 - 1, ec) not in self.v_walls

    def _neighbours(self, r: int, c: int):
        """Yield grid-adjacent cells reachable through open edges."""
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if self._edge_open(r, c, nr, nc):
                yield nr, nc

    # ---- Legal pawn moves (with jump logic) ----

    def legal_moves(self, player: int) -> List[Tuple[int, int]]:
        r, c = self.positions[player]
        opp_r, opp_c = self.positions[1 - player]
        dests: Set[Tuple[int, int]] = set()
        for dr, dc in DIRECTIONS:
            ar, ac = r + dr, c + dc
            if not self._edge_open(r, c, ar, ac):
                continue
            if (ar, ac) != (opp_r, opp_c):
                dests.add((ar, ac))
                continue
            br, bc = ar + dr, ac + dc
            if self._edge_open(ar, ac, br, bc):
                dests.add((br, bc))
            else:
                for sdr, sdc in ((dc, dr), (-dc, -dr)):
                    sr, sc = ar + sdr, ac + sdc
                    if self._edge_open(ar, ac, sr, sc):
                        dests.add((sr, sc))
        return sorted(dests)

    # ---- Fast wall legality check ----

    def _wall_overlap_ok(self, orient: str, wr: int, wc: int) -> bool:
        """Check overlap / crossing without path check."""
        if orient == "H":
            if any((wr, wc + d) in self.h_walls for d in (-1, 0, 1)):
                return False
            if (wr, wc) in self.v_walls:
                return False
        else:
            if any((wr + d, wc) in self.v_walls for d in (-1, 0, 1)):
                return False
            if (wr, wc) in self.h_walls:
                return False
        return True

    def _has_path(self, start_r: int, start_c: int, goal_row: int) -> bool:
        """BFS reachability check to goal_row."""
        if start_r == goal_row:
            return True
        q = deque([(start_r, start_c)])
        visited = {(start_r, start_c)}
        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbours(r, c):
                if (nr, nc) not in visited:
                    if nr == goal_row:
                        return True
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return False

    def is_wall_legal(self, player: int, orient: str, wr: int, wc: int) -> bool:
        """Full legality check for a single wall placement."""
        if self.walls_remaining[player] <= 0:
            return False
        if not (0 <= wr < BOARD_SIZE - 1 and 0 <= wc < BOARD_SIZE - 1):
            return False
        if not self._wall_overlap_ok(orient, wr, wc):
            return False
        walls_set = self.h_walls if orient == "H" else self.v_walls
        walls_set.add((wr, wc))
        ok = True
        for p in (0, 1):
            if not self._has_path(self.positions[p][0], self.positions[p][1],
                                  self.goal_rows[p]):
                ok = False
                break
        walls_set.discard((wr, wc))
        return ok

    # ---- Apply action ----

    def apply_move(self, player: int, dest: Tuple[int, int]):
        self.positions[player][0] = dest[0]
        self.positions[player][1] = dest[1]
        if dest[0] == self.goal_rows[player]:
            self.done = True
            self.winner = player
        self.current_player = 1 - player

    def apply_wall(self, player: int, orient: str, wr: int, wc: int):
        walls_set = self.h_walls if orient == "H" else self.v_walls
        walls_set.add((wr, wc))
        self.walls_remaining[player] -= 1
        self.current_player = 1 - player


# ===================================================================
# HeuristicEngine
# ===================================================================
class HeuristicEngine:

    @staticmethod
    def astar(state: FastState, player: int, want_path: bool = True
              ) -> Tuple[int, List[Tuple[int, int]]]:
        """A* shortest path from player's position to their goal row.

        Manhattan distance is an admissible/consistent heuristic because walls
        can only add moves. Parents are recorded while searching, so the path
        is rebuilt directly when the goal is popped.
        """
        start_r, start_c = state.positions[player]
        start = (start_r, start_c)
        goal_row = state.goal_rows[player]
        if start_r == goal_row:
            return 0, [start] if want_path else []

        open_heap: List[Tuple[int, int, Tuple[int, int]]] = [
            (abs(start_r - goal_row), 0, start)
        ]
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        closed: Set[Tuple[int, int]] = set()

        while open_heap:
            _, g, cell = heapq.heappop(open_heap)
            if cell in closed:
                continue
            closed.add(cell)
            if cell[0] == goal_row:
                if not want_path:
                    return g, []
                path = []
                cur: Optional[Tuple[int, int]] = cell
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return g, path

            r, c = cell
            ng = g + 1
            for nr, nc in state._neighbours(r, c):
                nxt = (nr, nc)
                if nxt in closed:
                    continue
                if ng < g_score.get(nxt, 999):
                    g_score[nxt] = ng
                    if want_path:
                        parent[nxt] = cell
                    heapq.heappush(open_heap, (ng + abs(nr - goal_row), ng, nxt))

        return 999, []

    @staticmethod
    def astar_distance(state: FastState, player: int) -> int:
        """A* distance only (skip parent/path reconstruction)."""
        return HeuristicEngine.astar(state, player, want_path=False)[0]

    @staticmethod
    def bfs_distance_map(state: FastState, player: int) -> List[List[int]]:
        """BFS distance from every cell to player's goal row."""
        goal_row = state.goal_rows[player]
        dist = [[-1] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        q = deque()
        for c in range(BOARD_SIZE):
            dist[goal_row][c] = 0
            q.append((goal_row, c))
        while q:
            r, c = q.popleft()
            for nr, nc in state._neighbours(r, c):
                if dist[nr][nc] == -1:
                    dist[nr][nc] = dist[r][c] + 1
                    q.append((nr, nc))
        return dist

    @staticmethod
    def shortest_path_edges(state: FastState, player: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Rebuild the A* shortest path and return its consecutive edges."""
        _, path = HeuristicEngine.astar(state, player, want_path=True)
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    @staticmethod
    def probe_wall(state: FastState, orient: str, wr: int, wc: int,
                   player: int) -> Tuple[int, int]:
        """Temporarily add a wall and return A* distances for both players.

        The state is mutated in place and restored immediately, so no clone is
        needed. A distance of 999 means that player's route was cut off, which
        also means the wall is illegal.
        """
        walls_set = state.h_walls if orient == "H" else state.v_walls
        walls_set.add((wr, wc))
        try:
            d_player = HeuristicEngine.astar_distance(state, player)
            d_other = HeuristicEngine.astar_distance(state, 1 - player)
        finally:
            walls_set.discard((wr, wc))
        return d_player, d_other

    @staticmethod
    def race_value(state: FastState, root_player: int, depth: int) -> float:
        """Minimax race value (opponent distance - own distance) for root_player.

        Only pawn moves are searched, which keeps the branching factor tiny and
        turns the wall decision into a real adversarial search rather than a
        pile of hand-tuned bonuses. The state is mutated in place and restored.
        """
        if state.done:
            if state.winner == root_player:
                return 10000.0
            if state.winner is not None:
                return -10000.0
            return 0.0
        if depth <= 0:
            return float(
                HeuristicEngine.astar_distance(state, 1 - root_player)
                - HeuristicEngine.astar_distance(state, root_player)
            )

        player = state.current_player
        moves = state.legal_moves(player)
        if not moves:
            state.current_player = 1 - player
            try:
                return HeuristicEngine.race_value(state, root_player, depth - 1)
            finally:
                state.current_player = player

        maximize = (player == root_player)
        best = -1e9 if maximize else 1e9
        pos = state.positions[player]
        orig_r, orig_c = pos[0], pos[1]
        goal_row = state.goal_rows[player]
        for dest in moves:
            pos[0], pos[1] = dest
            if dest[0] == goal_row:
                value = 10000.0 if maximize else -10000.0
            else:
                state.current_player = 1 - player
                try:
                    value = HeuristicEngine.race_value(state, root_player, depth - 1)
                finally:
                    state.current_player = player
            pos[0], pos[1] = orig_r, orig_c
            if maximize:
                if value > best:
                    best = value
            elif value < best:
                best = value
        pos[0], pos[1] = orig_r, orig_c
        return best

    @staticmethod
    def search_action_value(state: FastState, player: int, action: tuple,
                            depth: int) -> float:
        """Value of taking `action` for `player`, searching `depth - 1` replies."""
        old_player = state.current_player
        if action[0] == "move":
            pos = state.positions[player]
            orig_r, orig_c = pos[0], pos[1]
            pos[0], pos[1] = action[1], action[2]
            if action[1] == state.goal_rows[player]:
                pos[0], pos[1] = orig_r, orig_c
                return 10000.0
            state.current_player = 1 - player
            try:
                value = HeuristicEngine.race_value(state, player, depth - 1)
            finally:
                state.current_player = old_player
                pos[0], pos[1] = orig_r, orig_c
            return value

        if action[0] == "wall":
            walls_set = state.h_walls if action[1] == "H" else state.v_walls
            walls_set.add((action[2], action[3]))
            state.current_player = 1 - player
            try:
                value = HeuristicEngine.race_value(state, player, depth - 1)
            finally:
                state.current_player = old_player
                walls_set.discard((action[2], action[3]))
            return value

        return 0.0

    @staticmethod
    def walls_blocking_edge(a: Tuple[int, int], b: Tuple[int, int]) -> List[Wall]:
        """Các wall (orient, row, col) chặn cạnh (a, b)."""
        r1, c1 = a
        r2, c2 = b
        result: List[Wall] = []
        if r1 != r2:
            er = r1 if r1 < r2 else r2
            for wc in (c1 - 1, c1):
                if 0 <= wc < BOARD_SIZE - 1 and 0 <= er < BOARD_SIZE - 1:
                    result.append(("H", er, wc))
        else:
            ec = c1 if c1 < c2 else c2
            for wr in (r1 - 1, r1):
                if 0 <= wr < BOARD_SIZE - 1 and 0 <= ec < BOARD_SIZE - 1:
                    result.append(("V", wr, ec))
        return result

    @staticmethod
    def counter_wall(orient: str, r: int, c: int) -> Wall:
        """Wall cắt chéo: cùng (r, c), hướng ngược lại."""
        return ("V" if orient == "H" else "H", r, c)

    @staticmethod
    def wall_preventers(orient: str, r: int, c: int) -> List[Wall]:
        """Hình học thuần, không cần state: các wall làm (orient, r, c) không thể đặt
        (cắt chéo + chồng lên ±1). Chỉ lọc biên, CHƯA lọc hợp lệ."""
        dr, dc = (0, 1) if orient == "H" else (1, 0)
        cands = (
            ("V" if orient == "H" else "H", r, c),   # cắt chéo
            (orient, r - dr, c - dc),                # chồng lên, lệch -1
            (orient, r + dr, c + dc),                # chồng lên, lệch +1
        )
        lim = BOARD_SIZE - 1
        return [w for w in cands if 0 <= w[1] < lim and 0 <= w[2] < lim]

    @staticmethod
    def walls_preventing_wall(state: FastState, player: int,
                              orient: str, r: int, c: int) -> List[Wall]:
        """wall_preventers đã lọc hợp lệ cho `player` (dùng ngoài vòng MCTS)."""
        return [w for w in HeuristicEngine.wall_preventers(orient, r, c)
                if state.is_wall_legal(player, *w)]

    @staticmethod
    def walls_protecting_edge(state: FastState, player: int,
                              a: Tuple[int, int], b: Tuple[int, int]
                              ) -> Dict[Wall, List[Wall]]:
        """{wall đối phương dùng để chặn (a, b): [wall của `player` làm nó không đặt được]}."""
        opp = 1 - player
        blockers = HeuristicEngine.walls_blocking_edge(a, b)
        result: Dict[Wall, List[Wall]] = {}
        for w in blockers:
            if not state.is_wall_legal(opp, *w):
                continue
            result[w] = [p for p in HeuristicEngine.walls_preventing_wall(state, player, *w)
                         if p not in blockers]
        return result

    @staticmethod
    def wall_score(state: FastState, player: int, d_self_before: int, d_opp_before: int,
                   d_self_after: int, d_opp_after: int,
                   phase_mask: float, wall_bonus: float) -> Tuple[float, int, int]:
        """Score a wall from pre-computed before/after A* distances."""
        opp = 1 - player
        if d_self_after >= 999 or d_opp_after >= 999:
            return -1e9, 0, 0

        delta_opp = d_opp_after - d_opp_before
        delta_self = d_self_after - d_self_before

        w_self = state.walls_remaining[player]
        w_opp = state.walls_remaining[opp]

        ratio_dist = (d_self_before + 1.0) / (d_opp_before + 1.0)

        is_oke_wall = (w_self >= 8 and w_opp >= 9) and (delta_opp - delta_self) >= 1
        wasting_walls = ((
            (w_opp + w_self >= 12) and d_opp_before >= 4
        ) or (w_opp - w_self) >= 2)
        wasting_walls *= (1 - is_oke_wall)

        wasting_walls_exp = int(w_opp - 7) if w_opp >= 9 and d_opp_before >= 6 else 1
        little_remaining_walls = (w_opp <= 1)
        wall_cost = 1.0 / max(w_self, 1)

        score = (
            (delta_opp - delta_self) * ratio_dist
            + (1 - int(little_remaining_walls)) * (
                + wall_bonus * (w_self - w_opp)
                - wasting_walls * (2 ** ((w_opp - w_self) ** wasting_walls_exp))
                - wall_cost * phase_mask)
        )
        return score, delta_opp, delta_self

    @staticmethod
    def evaluate_move(state: FastState, dest: Tuple[int, int], player: int,
                      dist_map: List[List[int]]) -> float:
        """U(m): prefer moves that reduce distance to goal."""
        cur_r, cur_c = state.positions[player]
        cur_dist = dist_map[cur_r][cur_c]
        new_dist = dist_map[dest[0]][dest[1]]
        if new_dist < 0:
            return -1.0
        improvement = cur_dist - new_dist
        return 1.0 + improvement * 0.5

    @staticmethod
    def walls_intersect(w1_orient: str, w1_r: int, w1_c: int, 
                        w2_orient: str, w2_r: int, w2_c: int) -> bool:
        """Kiểm tra xem hai bức tường có đè lên nhau hoặc giao cắt chéo nhau không."""
        if w1_orient == w2_orient:
            if w1_orient == "H":
                return w1_r == w2_r and abs(w1_c - w2_c) <= 1
            else: # "V"
                return w1_c == w2_c and abs(w1_r - w2_r) <= 1
        else:
            return w1_r == w2_r and w1_c == w2_c


# ===================================================================
# MCTSNode
# ===================================================================
class MCTSNode:
    __slots__ = (
        "state", "parent", "action", "children",
        "visits", "value_sum", "prior", "player_who_moved",
        "_expanded",
    )

    def __init__(self, state: FastState, parent=None, action=None, prior=0.0,
                 player_who_moved=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[tuple, "MCTSNode"] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.player_who_moved = player_who_moved
        self._expanded = False

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def expand(self, root_player: int, deep: bool = False):
        if self._expanded or self.state.done:
            self._expanded = True
            return
        self._expanded = True

        player = self.state.current_player
        opp = 1 - player
        actions = []
        scores = []

        d_self = HeuristicEngine.astar_distance(self.state, player)
        d_opp = HeuristicEngine.astar_distance(self.state, opp)

        # =================================================================
        # 1. DI CHUYỂN PAWN - ĐIỂM LÀ KẾT QUẢ MINIMAX RACE SEARCH
        # =================================================================
        search_depth = ROOT_WALL_DEPTH if deep else INNER_WALL_DEPTH
        moves = self.state.legal_moves(player)
        for dest in moves:
            action = ("move", dest[0], dest[1])
            value = HeuristicEngine.search_action_value(
                self.state, player, action, search_depth
            )
            actions.append(action)
            scores.append(value)
        best_move_value = max(scores) if scores else -1e9

        # =================================================================
        # 2. ĐẶT TƯỜNG - CHỈ GIỮ WALL THẮNG NƯỚC ĐI TỐT NHẤT
        # =================================================================
        if self.state.walls_remaining[player] > 0:
            candidates = set()
            opp_edges = HeuristicEngine.shortest_path_edges(self.state, opp)
            for ea, eb in opp_edges:
                candidates.update(HeuristicEngine.walls_blocking_edge(ea, eb))

            for orient, wr, wc in candidates:
                if not self.state._wall_overlap_ok(orient, wr, wc):
                    continue
                d_self_after, d_opp_after = HeuristicEngine.probe_wall(
                    self.state, orient, wr, wc, player
                )
                if d_self_after >= 999 or d_opp_after >= 999:
                    continue

                # A wall that does not lengthen the opponent's A* route is a
                # pure tempo loss; drop it before running the race search.
                delta_gain = (d_opp_after - d_opp) - (d_self_after - d_self)
                if delta_gain < WALL_DELTA_MIN:
                    continue

                action = ("wall", orient, wr, wc)
                value = HeuristicEngine.search_action_value(
                    self.state, player, action, search_depth
                )
                if value > best_move_value:
                    actions.append(action)
                    scores.append(value)

        # =================================================================
        # 3. KHỞI TẠO NODE CON TỪ SOFTMAX PRIORS
        # =================================================================
        priors = _softmax(scores, temperature=1.0)
        for a, p in zip(actions, priors):
            child_state = self.state.clone()
            _apply_action_tuple(child_state, player, a)
            self.children[a] = MCTSNode(child_state, parent=self, action=a, prior=p, player_who_moved=player)
    
    def best_child_puct(self, root_player: int) -> "MCTSNode":
        """Select child with highest PUCT score according to the actor's perspective."""
        sqrt_total = math.sqrt(self.visits) if self.visits > 0 else 1.0

        best_score = -1e9
        best_child = None
        is_root_turn = (self.state.current_player == root_player)

        for child in self.children.values():
            if is_root_turn:
                q = child.q_value
            else:
                q = -child.q_value

            exploration = CPUCT * child.prior * (sqrt_total / (1.0 + child.visits))
            score = q + exploration
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


# ===================================================================
# HeuristicMCTS with Adversarial Opponent Check
# ===================================================================
class HeuristicMCTS:

    def __init__(self, root_player: int):
        self.root_player = root_player

    def search(self, root_state: FastState, time_limit: float) -> Dict[str, Any]:
        root = MCTSNode(root_state)
        root.expand(self.root_player, deep=True)

        if not root.children:
            return {"type": "pass"}

        deadline = time.monotonic() + time_limit
        iterations = 0

        while time.monotonic() < deadline:
            node = self._select(root)
            result = self._simulate(node.state)
            self._backpropagate(node, result)
            iterations += 1

        # Pick child with highest visit count
        best_action = max(root.children, key=lambda a: root.children[a].visits)
        return _action_tuple_to_dict(best_action)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.state.done:
            if not node._expanded:
                node.expand(self.root_player)
                if node.children:
                    unvisited = [c for c in node.children.values() if c.visits == 0]
                    if unvisited:
                        return random.choice(unvisited)
                return node
            if not node.children:
                return node
            node = node.best_child_puct(self.root_player)
        return node

    def _check_opponent_advantage(self, state: FastState) -> Tuple[bool, float]:
        """Reuse the current heuristic perspective to check if the opponent has an advantageous move.

        Returns: (has_advantage, reward_opp)
        If opponent has an advantage, the MCTS reward for root is -reward_opp.
        """
        opp = state.current_player
        root = 1 - opp

        # 1. Immediate win check for opponent
        goal_row = state.goal_rows[opp]
        for dest in state.legal_moves(opp):
            if dest[0] == goal_row:
                return True, 1.0

        d_opp = HeuristicEngine.astar_distance(state, opp)
        d_root = HeuristicEngine.astar_distance(state, root)
        max_opp_adv = 0.0

        # 2. Check advantageous walls blocking root's shortest path.
        #    Probe each wall in place (add/remove) using A*, no state clone.
        if state.walls_remaining[opp] > 0:
            root_edges = HeuristicEngine.shortest_path_edges(state, root)
            phase_mask = max(0.0, min(1.0, (14.5 - (d_opp + d_root) * 0.5)))
            wall_bonus = 0.5

            seen_walls = set()
            for edge_a, edge_b in root_edges:
                candidates = HeuristicEngine.walls_blocking_edge(edge_a, edge_b)
                for orient, wr, wc in candidates:
                    if (orient, wr, wc) in seen_walls:
                        continue
                    seen_walls.add((orient, wr, wc))
                    if not state._wall_overlap_ok(orient, wr, wc):
                        continue

                    d_opp_after, d_root_after = HeuristicEngine.probe_wall(
                        state, orient, wr, wc, opp
                    )
                    if d_opp_after >= 999 or d_root_after >= 999:
                        continue

                    h, delta_root, delta_opp = HeuristicEngine.wall_score(
                        state, opp, d_opp, d_root,
                        d_opp_after, d_root_after, phase_mask, wall_bonus
                    )
                    lead = d_root_after - d_opp_after

                    # An advantageous wall puts the opponent ahead in the race
                    # or imposes a major bottleneck (delta_root >= 3)
                    if lead > 0 and h > 0.0:
                        adv = min(0.95, 0.3 + 0.15 * lead + 0.05 * h)
                        if adv > max_opp_adv:
                            max_opp_adv = adv
                    elif delta_root >= 3 and delta_root > delta_opp + 1:
                        adv = min(0.85, 0.2 + 0.15 * (delta_root - delta_opp))
                        if adv > max_opp_adv:
                            max_opp_adv = adv

        # 3. Check positional pawn moves for opponent
        opp_moves = state.legal_moves(opp)
        dist_map_opp = HeuristicEngine.bfs_distance_map(state, opp)
        for dest in opp_moves:
            d_dest = dist_map_opp[dest[0]][dest[1]]
            if d_dest >= 0 and d_dest < d_opp:
                lead = d_root - d_dest
                if lead >= 2 and d_dest <= 4:
                    adv = min(0.9, 0.35 + 0.12 * lead)
                    if adv > max_opp_adv:
                        max_opp_adv = adv

        if max_opp_adv > 0.0:
            return True, max_opp_adv

        return False, 0.0

    def _simulate(self, state: FastState) -> float:
        """Simulation rollout.

        At opponent's turn: uses current perspective to check if opponent
        has an advantageous move; if so, reward = - reward_opp.
        """
        s = state.clone()
        if s.done:
            if s.winner == self.root_player:
                return 1.0
            elif s.winner is not None:
                return -1.0
            return 0.0

        for _ in range(ROLLOUT_DEPTH):
            if s.done:
                break
            player = s.current_player

            # Check if it is the opponent's turn
            if player != self.root_player:
                has_adv, opp_reward = self._check_opponent_advantage(s)
                if has_adv:
                    # If opponent has an advantageous move, reward is - reward đó
                    return -opp_reward

            # Pawn advancement
            moves = s.legal_moves(player)
            if not moves:
                s.current_player = 1 - player
                continue

            if random.random() < EPSILON:
                dest = random.choice(moves)
            else:
                dist_map = HeuristicEngine.bfs_distance_map(s, player)
                best_dest = moves[0]
                best_d = 999
                for m in moves:
                    d = dist_map[m[0]][m[1]]
                    if 0 <= d < best_d:
                        best_d = d
                        best_dest = m
                dest = best_dest

            s.apply_move(player, dest)

        if s.winner == self.root_player:
            return 1.0
        elif s.winner is not None:
            return -1.0

        # Heuristic draw evaluation
        d_self = HeuristicEngine.astar_distance(s, self.root_player)
        d_opp = HeuristicEngine.astar_distance(s, 1 - self.root_player)
        if d_self < d_opp:
            return 0.3
        elif d_self > d_opp:
            return -0.3
        return 0.0

    def _backpropagate(self, node: MCTSNode, result: float):
        """Backpropagate simulation result (+ for root win, - for opp advantage/win)."""
        while node is not None:
            node.visits += 1
            node.value_sum += result
            node = node.parent


# ===================================================================
# Helper Functions
# ===================================================================
def _apply_action_tuple(state: FastState, player: int, action: tuple):
    if action[0] == "move":
        state.apply_move(player, (action[1], action[2]))
    elif action[0] == "wall":
        state.apply_wall(player, action[1], action[2], action[3])
    else:
        state.current_player = 1 - player


def _action_tuple_to_dict(action: tuple) -> Dict[str, Any]:
    if action[0] == "move":
        return {"type": "move", "to": [action[1], action[2]]}
    elif action[0] == "wall":
        return {"type": "wall", "orientation": action[1],
                "row": action[2], "col": action[3]}
    return {"type": "pass"}


def _softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    if temperature <= 0:
        temperature = 0.01
    scaled = [s / temperature for s in scores]
    max_s = max(scaled)
    exps = [math.exp(s - max_s) for s in scaled]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / total for e in exps]


# ===================================================================
# Entry Point
# ===================================================================
def choose_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point called by the arena runner."""
    if observation.get("done"):
        return {"type": "pass"}

    player_id = observation["player_id"]
    state = FastState.from_observation(observation)

    # Quick check: if on goal or 1 step away, greedy move immediately wins
    my_dist = HeuristicEngine.astar_distance(state, player_id)
    if my_dist <= 1:
        moves = observation["legal_moves"]
        distances = observation["distances"]
        best = min(moves, key=lambda m: distances[m[0]][m[1]]
                   if distances[m[0]][m[1]] >= 0 else 999)
        return {"type": "move", "to": list(best)}

    # Allocate search budget
    time_budget = TIME_FRACTION * 1.0

    mcts = HeuristicMCTS(root_player=player_id)
    action = mcts.search(state, time_budget)
    return action
