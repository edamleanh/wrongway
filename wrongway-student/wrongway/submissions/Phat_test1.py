"""Bot của Phát: Minimax 8 tầng kết hợp Iterative Deepening, Alpha-Beta Pruning, Bộ lọc tường hiệu quả (Gain-based) và Chống bẫy phong tỏa hàng đích (Anti-Funnel Trap).
"""

"""
Chiến thuật:
- Tầm nhìn 8 tầng (Depth 8): Nhìn trước 4 nước của ta và 4 nước phản đòn của địch.
- Iterative Deepening (IDS):
    1. Chạy tầng 2 (Depth 2) trước để luôn có nước đi an toàn chỉ sau ~2ms.
    2. Mở rộng lên tầng 4 (Depth 4) sau ~5ms.
    3. Đào sâu lên tầng 6 (Depth 6) sau ~20ms.
    4. Vươn xa đến tầng 8 (Depth 8) sau ~40-60ms để làm chủ toàn diện cuộc đua về đích.
    5. Luôn bảo vệ an toàn thời gian (< 0.50s), tuyệt đối không bao giờ bị timeout trên Colab.
- Alpha-Beta Pruning: Cắt tỉa nhánh cờ kém hiệu quả ở mọi tầng.
- Heuristic:
    Score = 100 * (Dist_địch - Dist_ta) + 15 * (Tường_ta - Tường_địch)
            + 5 * Mobility_ta - Door_Penalty_ta + Door_Bonus_địch
"""

from collections import deque
import time

RULES_VERSION = "wrongway-md-v1"


class SearchTimeout(Exception):
    """Ngoại lệ ném ra khi sắp hết ngân sách thời gian."""
    pass


def add_wall_edges(blocked, orientation, row, col):
    """Thêm 2 cạnh bị tường chặn vào tập hợp blocked."""
    if orientation == "H":
        pairs = [
            ((row, col), (row + 1, col)),
            ((row, col + 1), (row + 1, col + 1)),
        ]
    else:
        pairs = [
            ((row, col), (row, col + 1)),
            ((row + 1, col), (row + 1, col + 1)),
        ]
    for start, end in pairs:
        blocked.add((start, end))
        blocked.add((end, start))


def get_blocked_edges(walls):
    """Lấy tất cả các cạnh bị chặn bởi tường đã đặt trên bàn cờ."""
    blocked = set()
    for row, col in walls["horizontal"]:
        add_wall_edges(blocked, "H", row, col)
    for row, col in walls["vertical"]:
        add_wall_edges(blocked, "V", row, col)
    return blocked


def count_blocked_goal_doors(goal_row, blocked, size=9):
    """
    Đếm số lượng cánh cửa dẫn vào hàng đích đã bị tường chặn.
    - Với Người chơi 0 (goal_row = 0): cửa nối giữa Hàng 1 và Hàng 0.
    - Với Người chơi 1 (goal_row = 8): cửa nối giữa Hàng 7 và Hàng 8.
    """
    adjacent_row = 1 if goal_row == 0 else size - 2
    blocked_count = 0
    for col in range(size):
        edge = ((adjacent_row, col), (goal_row, col))
        if edge in blocked:
            blocked_count += 1
    return blocked_count


def bfs_path(start, goal_row, size, blocked):
    """
    Tìm đường đi ngắn nhất từ start đến goal_row.
    Trả về: (khoảng cách, danh sách các ô trên đường đi).
    """
    start = tuple(start)
    queue = deque([start])
    parent = {start: None}

    reached = None
    while queue:
        curr = queue.popleft()
        if curr[0] == goal_row:
            reached = curr
            break

        row, col = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            neighbor = (nr, nc)
            if 0 <= nr < size and 0 <= nc < size:
                if neighbor not in parent and (curr, neighbor) not in blocked:
                    parent[neighbor] = curr
                    queue.append(neighbor)

    if reached is None:
        return size * size, []

    path = []
    curr = reached
    while curr is not None:
        path.append(curr)
        curr = parent[curr]
    path.reverse()
    return len(path) - 1, path


def evaluate_state(my_pos, opp_pos, my_goal, opp_goal, my_walls, opp_walls, size, blocked, my_mobility):
    """Hàm Heuristic chấm điểm thế cờ tại nút lá."""
    my_dist, _ = bfs_path(my_pos, my_goal, size, blocked)
    opp_dist, _ = bfs_path(opp_pos, opp_goal, size, blocked)

    if my_dist == 0:
        return 10000
    if opp_dist == 0:
        return -10000

    # Điều chỉnh khoảng cách nếu có cơ hội nhảy qua đầu đối thủ (Pawn Jump Adjustment)
    dr_my = 1 if my_goal > my_pos[0] else -1
    jump_my = (my_pos[0] + 2 * dr_my, my_pos[1])
    if opp_pos == (my_pos[0] + dr_my, my_pos[1]):
        if 0 <= jump_my[0] < size and (my_pos, opp_pos) not in blocked and (opp_pos, jump_my) not in blocked:
            my_dist = max(1, my_dist - 1)

    dr_opp = 1 if opp_goal > opp_pos[0] else -1
    jump_opp = (opp_pos[0] + 2 * dr_opp, opp_pos[1])
    if my_pos == (opp_pos[0] + dr_opp, opp_pos[1]):
        if 0 <= jump_opp[0] < size and (opp_pos, my_pos) not in blocked and (my_pos, jump_opp) not in blocked:
            opp_dist = max(1, opp_dist - 1)

    # Tính độ mở cửa đích
    my_blocked = count_blocked_goal_doors(my_goal, blocked, size)
    opp_blocked = count_blocked_goal_doors(opp_goal, blocked, size)

    # Phạt lũy tiến ngay từ khi cửa đích bị bịt >= 2 cửa để chống bẫy phong tỏa từ sớm
    door_penalty = 0
    if my_blocked >= 2:
        door_penalty = 70 * (my_blocked - 1)
    if my_blocked >= 5:
        door_penalty += 160 * (my_blocked - 4)

    door_bonus = 0
    if opp_blocked >= 2:
        door_bonus = 50 * (opp_blocked - 1)

    # Thưởng thế trận khi bẫy hành lang mép biên (cột 7 hoặc cột 0) bị khóa chặt lối thoát hiểm
    chokepoint_bonus = 0
    has_v7 = any(((r, 7), (r, 8)) in blocked for r in range(size - 1))
    has_h7 = any(((r, 8), (r + 1, 8)) in blocked for r in range(size - 1))
    if has_v7 and has_h7:
        chokepoint_bonus += 140

    has_v0 = any(((r, 0), (r, 1)) in blocked for r in range(size - 1))
    has_h0 = any(((r, 0), (r + 1, 0)) in blocked for r in range(size - 1))
    if has_v0 and has_h0:
        chokepoint_bonus += 140

    # Thưởng thế trận khi cửa đích của ta được bảo vệ bằng Door Jammer (tường dọc V ở hàng đích)
    my_target_h_row = 0 if my_goal == 0 else size - 2
    has_door_jammer = any(((my_target_h_row, c), (my_target_h_row, c + 1)) in blocked for c in (3, 4))
    if has_door_jammer:
        chokepoint_bonus += 120

    score = (
        100 * (opp_dist - my_dist)
        + 15 * (my_walls - opp_walls)
        + 5 * my_mobility
        - door_penalty
        + door_bonus
        + chokepoint_bonus
    )
    return score


def get_effective_walls(legal_walls, opp_pos, opp_goal, my_pos, my_goal, size, blocked,
                        opp_path, my_path, observation_walls, max_candidates=7):
    """
    Lọc các bức tường ứng viên chất lượng cao:
    1. Tấn công (Offensive): Tìm các bức tường cắt vào đường đi của địch mang lại Net Gain cao nhất.
    2. Chokepoint Trap (Phản bẫy mê cung):
       - Nếu địch xây hàng rào dọc (như cột 7), chặn ngay hành lang thoát hiểm hẹp (cột 8) để biến mê cung thành cái bẫy tự nhốt địch.
       - Cắt chữ thập phá thế liên kết tường của địch.
    3. Hộ tống hành lang (Corridor Shield):
       - Che chắn các vị trí đối thủ có thể đặt tường ngang chặn đứng bước tiến của ta.
    """
    scored_walls = []
    seen_walls = set()

    def add_candidate(wall, score):
        key = (wall["orientation"], wall["row"], wall["col"])
        if key not in seen_walls:
            seen_walls.add(key)
            scored_walls.append((score, wall))

    opp_dist, _ = bfs_path(opp_pos, opp_goal, size, blocked)
    my_dist, _ = bfs_path(my_pos, my_goal, size, blocked)

    # --- 1. TẤN CÔNG (OFFENSIVE): Cắt vào các cạnh trên đường đi ngắn nhất của địch ---
    if opp_path and len(opp_path) >= 2:
        # Chỉ xét các cạnh gần vị trí hiện tại của địch (trong vòng 3 bước),
        # trừ khi địch đã áp sát hàng đích (opp_dist <= 3).
        # Tuyệt đối không phí lượt đặt tường ở tít hàng đích địch khi địch đang ở đầu bên kia bàn cờ!
        max_lookahead = len(opp_path) - 1 if opp_dist <= 3 else min(3, len(opp_path) - 1)
        path_edges = set()
        for i in range(max_lookahead):
            u, v = opp_path[i], opp_path[i + 1]
            path_edges.add((u, v))
            path_edges.add((v, u))

        for wall in legal_walls:
            w_ori, w_r, w_c = wall["orientation"], wall["row"], wall["col"]
            test_blocked = set()
            add_wall_edges(test_blocked, w_ori, w_r, w_c)

            if any(edge in test_blocked for edge in path_edges):
                cand_blocked = blocked | test_blocked
                opp_after, _ = bfs_path(opp_pos, opp_goal, size, cand_blocked)
                my_after, _ = bfs_path(my_pos, my_goal, size, cand_blocked)
                gain = (opp_after - opp_dist) - (my_after - my_dist)

                min_gain = 1 if opp_dist <= my_dist + 1 else 2
                if gain >= min_gain:
                    add_candidate(wall, gain * 35)

    # --- 2. PHẢN BẪY MÊ CUNG (COUNTER-MAZE & CHOKEPOINT TRAP) ---
    # (a) Chokepoint Trap: Nếu địch xây tường dọc ở cột 7 (hoặc cột 0) để dùng cột biên làm đại lộ:
    v_walls_7 = [w for w in observation_walls.get("vertical", []) if w[1] == size - 2]
    if len(v_walls_7) >= 1:
        # Nếu địch xuất phát từ hàng 0 (đi xuống): đặt bẫy ở hàng 3 hoặc 4
        # Nếu địch xuất phát từ hàng 8 (đi lên): đặt bẫy ở hàng 5 hoặc 4
        opp_start_row = size - 1 if opp_goal == 0 else 0
        trap_rows = (3, 4) if opp_start_row == 0 else (5, 4)
        has_trap = any(w[1] == size - 2 and w[0] in (3, 4, 5) for w in observation_walls.get("horizontal", []))
        if not has_trap:
            for wall in legal_walls:
                if wall["orientation"] == "H" and wall["col"] == size - 2 and wall["row"] in trap_rows:
                    cand_blocked = blocked.copy()
                    add_wall_edges(cand_blocked, "H", wall["row"], wall["col"])
                    my_after, _ = bfs_path(my_pos, my_goal, size, cand_blocked)
                    if my_after == my_dist:  # Hoàn toàn không làm tăng đường đi của ta
                        add_candidate(wall, 160)

    # (b) Door Jammer: Chỉ can thiệp khi địch đã đặt từ 2 tường ngang trở lên ở hàng đích
    # và có nguy cơ phong tỏa thực sự (tránh tự bóp chân mình bằng tường dọc sớm)
    my_target_h_row = 0 if my_goal == 0 else size - 2
    enemy_h_on_target = [w for w in observation_walls.get("horizontal", []) if w[0] == my_target_h_row]
    if len(enemy_h_on_target) >= 2:
        has_door_jammer = any(w[0] == my_target_h_row and w[1] in (3, 4) for w in observation_walls.get("vertical", []))
        if not has_door_jammer:
            v_cand = [w for w in legal_walls if w["orientation"] == "V" and w["row"] == my_target_h_row and w["col"] in (4, 3)]
            v_cand.sort(key=lambda w: abs(w["col"] - my_pos[1]))
            for wall in v_cand:
                cand_blocked = blocked.copy()
                add_wall_edges(cand_blocked, "V", wall["row"], wall["col"])
                my_after, _ = bfs_path(my_pos, my_goal, size, cand_blocked)
                if my_after == my_dist:
                    add_candidate(wall, 150)
                    break

    scored_walls.sort(key=lambda item: item[0], reverse=True)
    return [wall for score, wall in scored_walls[:max_candidates]]


def get_blocking_walls_for_path(path):
    """Tìm tất cả các vị trí tường có thể chặn trực tiếp vào một con đường."""
    candidates = []
    seen = set()
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        walls = []
        if r1 != r2:
            r = min(r1, r2)
            c = c1
            if 0 <= r < 8 and 0 <= c < 8: walls.append(('H', r, c))
            if 0 <= r < 8 and 0 <= c - 1 < 8: walls.append(('H', r, c - 1))
        else:
            c = min(c1, c2)
            r = r1
            if 0 <= r < 8 and 0 <= c < 8: walls.append(('V', r, c))
            if 0 <= r - 1 < 8 and 0 <= c < 8: walls.append(('V', r - 1, c))
        for w in walls:
            if w not in seen:
                seen.add(w)
                candidates.append(w)
    return candidates


def get_pawn_moves(pos, other_pos, goal_row, size, blocked, max_moves=2):
    """Lấy các nước đi quân hợp lệ kề bên, tính cả luật nhảy quân (Jump rule), sắp xếp theo độ gần đích nhất."""
    r, c = pos
    valid = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        n_pos = (nr, nc)
        if 0 <= nr < size and 0 <= nc < size and (pos, n_pos) not in blocked:
            if n_pos != other_pos:
                valid.append(n_pos)
            else:
                # Luật nhảy thẳng qua đầu đối phương
                jr, jc = r + 2 * dr, c + 2 * dc
                j_pos = (jr, jc)
                if 0 <= jr < size and 0 <= jc < size and (other_pos, j_pos) not in blocked:
                    valid.append(j_pos)
                else:
                    # Nhảy chéo nếu ô phía sau bị tường chặn hoặc chạm mép bàn cờ
                    side_deltas = [(0, -1), (0, 1)] if dr != 0 else [(-1, 0), (1, 0)]
                    for sdr, sdc in side_deltas:
                        dr_diag, dc_diag = nr + sdr, nc + sdc
                        diag_pos = (dr_diag, dc_diag)
                        if 0 <= dr_diag < size and 0 <= dc_diag < size and (other_pos, diag_pos) not in blocked:
                            valid.append(diag_pos)

    if not valid:
        return [pos]

    valid.sort(key=lambda p: bfs_path(p, goal_row, size, blocked)[0])
    return valid[:max_moves]


def minimax(depth, is_max, my_pos, opp_pos, my_walls, opp_walls,
            my_goal, opp_goal, size, blocked, alpha, beta,
            start_time, time_limit, my_mobility):
    """Thuật toán Minimax đệ quy với cắt tỉa Alpha-Beta."""
    if time.time() - start_time > time_limit:
        raise SearchTimeout()

    if my_pos[0] == my_goal:
        return 10000 + depth
    if opp_pos[0] == opp_goal:
        return -10000 - depth
    if depth == 0:
        return evaluate_state(my_pos, opp_pos, my_goal, opp_goal,
                              my_walls, opp_walls, size, blocked, my_mobility)

    if is_max:
        max_eval = -float("inf")
        next_moves = get_pawn_moves(my_pos, opp_pos, my_goal, size, blocked, max_moves=2)
        for move in next_moves:
            val = minimax(depth - 1, False, move, opp_pos, my_walls, opp_walls,
                          my_goal, opp_goal, size, blocked, alpha, beta,
                          start_time, time_limit, my_mobility)
            max_eval = max(max_eval, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break

        # Ở các tầng trung gian sâu (depth == 6 hoặc depth == 4):
        # Tính thêm nhánh đặt tường cho ta nếu còn tường để nhìn trước nước chặn đường địch
        if depth in (4, 6) and my_walls > 0:
            opp_dist, opp_path = bfs_path(opp_pos, opp_goal, size, blocked)
            if opp_path and len(opp_path) >= 2:
                candidate_walls = get_blocking_walls_for_path(opp_path)
                valid_wall_count = 0
                for ori, r, c in candidate_walls:
                    test_edges = set()
                    add_wall_edges(test_edges, ori, r, c)
                    if not test_edges.intersection(blocked):
                        cand_blocked = blocked | test_edges
                        opp_after, _ = bfs_path(opp_pos, opp_goal, size, cand_blocked)
                        if opp_after < size * size and opp_after > opp_dist:
                            my_after, _ = bfs_path(my_pos, my_goal, size, cand_blocked)
                            if my_after < size * size:
                                val = minimax(depth - 1, False, my_pos, opp_pos, my_walls - 1, opp_walls,
                                              my_goal, opp_goal, size, cand_blocked, alpha, beta,
                                              start_time, time_limit, my_mobility)
                                max_eval = max(max_eval, val)
                                alpha = max(alpha, val)
                                valid_wall_count += 1
                                if beta <= alpha or valid_wall_count >= 2:
                                    break

        return max_eval
    else:
        min_eval = float("inf")
        next_moves = get_pawn_moves(opp_pos, my_pos, opp_goal, size, blocked, max_moves=2)
        for move in next_moves:
            val = minimax(depth - 1, True, my_pos, move, my_walls, opp_walls,
                          my_goal, opp_goal, size, blocked, alpha, beta,
                          start_time, time_limit, my_mobility)
            min_eval = min(min_eval, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_eval


def choose_action(observation):
    start_time = time.time()
    time_limit = 0.50  # Giới hạn an toàn tuyệt đối trên Colab (Arena cho 1.0s)

    size = observation["board_size"]
    you = observation["you"]
    opponent = observation["opponent"]
    legal_moves = observation["legal_moves"]
    legal_walls = observation["legal_walls"]

    # 1. Thắng ngay lập tức nếu bước được vào hàng đích
    for move in legal_moves:
        if move[0] == you["goal_row"]:
            return {"type": "move", "to": list(move)}

    my_pos = tuple(you["position"])
    opp_pos = tuple(opponent["position"])
    my_goal = you["goal_row"]
    opp_goal = opponent["goal_row"]

    blocked = get_blocked_edges(observation["walls"])
    opp_dist, opp_path = bfs_path(opp_pos, opp_goal, size, blocked)
    my_dist, my_path = bfs_path(my_pos, my_goal, size, blocked)

    # 2. Tạo danh sách hành động ứng viên ở Tầng 1 (Root)
    candidate_actions = []

    # (a) Nước đi quân: Lọc 2 nước đưa ta gần đích nhất
    ranked_moves = sorted(
        legal_moves,
        key=lambda m: bfs_path(tuple(m), my_goal, size, blocked)[0]
    )
    for m in ranked_moves[:2]:
        candidate_actions.append({"type": "move", "to": list(m)})

    # (b) Nước đặt tường:
    # Luôn xem xét tường nếu còn tường, đặc biệt khi địch dựng bẫy hoặc có nguy cơ phong tỏa
    my_blocked = count_blocked_goal_doors(my_goal, blocked, size)
    should_consider_walls = (
        you["walls_remaining"] > 0 and (
            opp_dist <= my_dist + 2 or
            my_blocked >= 1 or
            opponent["walls_remaining"] < 10 or
            you["walls_remaining"] >= 8
        )
    )

    if should_consider_walls:
        effective_walls = get_effective_walls(
            legal_walls, opp_pos, opp_goal, my_pos, my_goal, size, blocked,
            opp_path, my_path, observation["walls"], max_candidates=6
        )
        for w in effective_walls:
            candidate_actions.append({
                "type": "wall",
                "orientation": w["orientation"],
                "row": w["row"],
                "col": w["col"],
            })

    if not candidate_actions:
        return {"type": "move", "to": list(ranked_moves[0])} if ranked_moves else {"type": "pass"}

    best_action = candidate_actions[0]
    my_mobility = len(legal_moves)

    # 3. Iterative Deepening: Chạy Depth 2 -> Depth 4 -> Depth 6 -> Depth 8
    for target_depth in (2, 4, 6, 8):
        depth_best_action = best_action
        best_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        interrupted = False
        for act in candidate_actions:
            if time.time() - start_time > time_limit:
                interrupted = True
                break

            sim_blocked = set(blocked)
            sim_my_pos = my_pos
            sim_my_walls = you["walls_remaining"]

            if act["type"] == "move":
                sim_my_pos = tuple(act["to"])
            elif act["type"] == "wall":
                add_wall_edges(sim_blocked, act["orientation"], act["row"], act["col"])
                sim_my_walls -= 1

            if sim_my_pos[0] == my_goal:
                return act

            try:
                score = minimax(
                    depth=target_depth - 1,
                    is_max=False,
                    my_pos=sim_my_pos,
                    opp_pos=opp_pos,
                    my_walls=sim_my_walls,
                    opp_walls=opponent["walls_remaining"],
                    my_goal=my_goal,
                    opp_goal=opp_goal,
                    size=size,
                    blocked=sim_blocked,
                    alpha=alpha,
                    beta=beta,
                    start_time=start_time,
                    time_limit=time_limit,
                    my_mobility=my_mobility,
                )
            except SearchTimeout:
                interrupted = True
                break

            if score > best_score:
                best_score = score
                depth_best_action = act

            alpha = max(alpha, best_score)

        if not interrupted:
            best_action = depth_best_action

        # Nếu đã dùng quá 0.35s thì không mở rộng sang tầng sâu hơn nữa
        if time.time() - start_time > 0.35:
            break

    return best_action
