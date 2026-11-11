from collections import deque


def add_wall_edges(blocked, orientation, row, col):
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


def blocked_edges(walls):
    blocked = set()
    for row, col in walls["horizontal"]:
        add_wall_edges(blocked, "H", row, col)
    for row, col in walls["vertical"]:
        add_wall_edges(blocked, "V", row, col)
    return blocked


def distance_to_goal(position, goal_row, size, blocked):
    start = tuple(position)
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current, distance = queue.popleft()
        row, col = current
        if row == goal_row:
            return distance

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            if neighbor in visited or (current, neighbor) in blocked:
                continue
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))

    return size * size


def wall_priority_key(wall, op_pos):
    dist = abs(wall["row"] - op_pos[0]) + abs(wall["col"] - op_pos[1])
    if dist == 1:
        return (2, 1 if wall["orientation"] == "H" else 0, 0)
    return (1, 0, dist)


def policy(observation):
    you = observation["you"]
    opponent = observation["opponent"]
    my_pos = you["position"]
    op_pos = opponent["position"]
    goal_row = you["goal_row"]
    board_size = observation.get("board_size", 9)
    moves = observation["legal_moves"]
    walls = observation["legal_walls"]
    distances = observation.get("distances")

    target_h_row = goal_row - 1 if goal_row > 0 else 0
    first_wall_placed = any(
        w[0] == target_h_row and w[1] == 6
        for w in observation["walls"]["horizontal"]
    )

    if not first_wall_placed:
        goal_h_walls = [
            w for w in walls
            if w["orientation"] == "H" and w["row"] == target_h_row and w["col"] == 6
        ]
        if goal_h_walls:
            return goal_h_walls[0], None

    start_wall_row = 0 if goal_row == board_size - 1 else board_size - 2
    v_wall_col = 7

    placed_h = [w for w in observation["walls"]["horizontal"] if w[0] == start_wall_row]
    placed_v = [w for w in observation["walls"]["vertical"] if w[1] == v_wall_col]

    start_h_walls = [
        w for w in walls
        if w["orientation"] == "H" 
        and w["row"] == start_wall_row
        and w["col"] < board_size - 2
    ]
    start_v_walls = [
        w for w in walls
        if w["orientation"] == "V"
        and w["col"] == v_wall_col
        and (w["row"] % 2 != start_wall_row % 2)
        and (w["row"] != goal_row - (goal_row == 8))
    ]

    start_h_walls.sort(key=lambda w: wall_priority_key(w, op_pos), reverse=True)
    start_v_walls.sort(key=lambda w: wall_priority_key(w, op_pos), reverse=True)

    dist_to_h = abs(op_pos[0] - start_wall_row)
    dist_to_v = abs(op_pos[1] - v_wall_col)

    if dist_to_h <= dist_to_v:
        primary_walls, secondary_walls = start_h_walls, start_v_walls
        primary_count, secondary_count = len(placed_h), len(placed_v)
    else:
        primary_walls, secondary_walls = start_v_walls, start_h_walls
        primary_count, secondary_count = len(placed_v), len(placed_h)

    if primary_count < 4 and primary_walls:
        return primary_walls[0], None
    if secondary_count < 4 and secondary_walls:
        return secondary_walls[0], None

    size = observation["board_size"]
    blocked = blocked_edges(observation["walls"])
    own_distance = distance_to_goal(you["position"], you["goal_row"], size, blocked)
    enemy_distance = distance_to_goal(opponent["position"], opponent["goal_row"], size, blocked)

    best_wall = None

    if enemy_distance <= own_distance + 1:
        candidate_walls = []
        for wall in observation["legal_walls"]:
            candidate = blocked.copy()
            add_wall_edges(candidate, wall["orientation"], wall["row"], wall["col"])
            own_after = distance_to_goal(you["position"], you["goal_row"], size, candidate)
            enemy_after = distance_to_goal(opponent["position"], opponent["goal_row"], size, candidate)
            gain = (enemy_after - enemy_distance) - (own_after - own_distance)

            if gain > 0:
                candidate_walls.append((gain, wall_priority_key(wall, op_pos), wall))

        if candidate_walls:
            candidate_walls.sort(key=lambda item: (item[0], item[1]), reverse=True)
            best_wall = candidate_walls[0][2]

    if best_wall is not None:
        return best_wall, None

    best_m = None
    if distances and moves:
        best_m = min(
            moves,
            key=lambda m: distances[m[0]][m[1]] if distances[m[0]][m[1]] >= 0 else float("inf"),
            default=None,
        )

    if best_m:
        return None, best_m
    if moves:
        return None, moves[0]

    return None, None


def choose_action(observation):
    moves = observation["legal_moves"]
    you = observation["you"]

    for move in moves:
        if move[0] == you["goal_row"]:
            return {"type": "move", "to": list(move)}

    best_wall, best_move = policy(observation)

    if best_wall is not None:
        return {
            "type": "wall",
            "orientation": best_wall["orientation"],
            "row": best_wall["row"],
            "col": best_wall["col"],
        }
    if best_move is not None:
        return {"type": "move", "to": list(best_move)}
    return {"type": "pass"}