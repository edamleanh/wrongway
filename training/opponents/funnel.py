"""Bot luyện tập "phễu + hành lang riêng" (mô phỏng chiến thuật trong video thầy gửi).

Không phải bài nộp, chỉ để MSSV tập đấu (training/gauntlet.py).

Kế hoạch 8 tường (lấy cột ngoài cùng gần mình làm hành lang, ví dụ cột 8):
- HÀNG RÀO: 4 tường ngang sát đích của đối thủ, che 8 cột còn lại -> đối thủ chỉ có thể
  tới đích qua cột hành lang.
- HÀNH LANG: 4 tường dọc ngăn cột hành lang với phần còn lại của bàn, chừa một lối vào ở
  phía đích của mình -> đối thủ phải đi vòng lên tận đầu bên kia mới vào được.
Mình chạy trong hành lang: đối thủ không thể chặn ngang (hành lang là đường duy nhất của
mình, tường bít đường bị cấm), nên đường của mình gần như miễn nhiễm với tường.

Mỗi lượt:
1. Thắng ngay nếu được.
2. Chưa ở cột hành lang và (đối thủ còn xa >= 6 bước, hoặc hàng rào đã xong) -> đi sang
   cột hành lang.
3. Còn tường và còn tường kế hoạch đặt được mà không làm mình chậm đi -> đặt tường làm
   đối thủ chậm nhất (hòa thì theo thứ tự kế hoạch: hàng rào trước, rồi hành lang từ phía
   đích đối thủ). Không kịp vào hành lang thì phễu vẫn bắt đối thủ đi vòng.
4. Còn lại: đi theo đường ngắn nhất.

"Phễu đúng lúc" (TRIGGER = k): giống người chơi trong video, không để lộ ý đồ sớm. Trước khi
kích hoạt chỉ đi (vào hành lang nếu còn dư thời gian, rồi về đích), không mất nhịp nào.
Khi đối thủ còn cách đích <= k bước thì mỗi lượt đặt tường kế hoạch làm đối thủ chậm nhất,
nên hàng rào đóng đúng chỗ đối thủ đang lao tới. Hành lang chọn ở mép xa đối thủ.
"""
from collections import deque

SIZE = 9
# None = phễu dựng sẵn từ đầu (bản gốc). Số k = "phễu đúng lúc": không đặt tường nào cho tới
# khi đối thủ còn cách đích <= k bước, rồi đặt liên tiếp (xem timed()). Thử 2/3/4 với MSSV:
# chỉ k = 4 thắng được vài ván (MSSV đứng chờ trước khe hở, xem EXPERIMENTS.md).
TRIGGER = 4


def blocked_edges(h_walls, v_walls):
    """Tập cạnh bị chặn ((r1, c1), (r2, c2)) cả hai chiều."""
    out = set()
    for o, walls in (("H", h_walls), ("V", v_walls)):
        for r, c in walls:
            if o == "H":
                pairs = (((r, c), (r + 1, c)), ((r, c + 1), (r + 1, c + 1)))
            else:
                pairs = (((r, c), (r, c + 1)), ((r + 1, c), (r + 1, c + 1)))
            for a, b in pairs:
                out.add((a, b))
                out.add((b, a))
    return out


def distance(start, goal_row, blocked):
    """BFS số bước ít nhất tới goal_row (bỏ qua quân), -1 nếu bị bít."""
    start = tuple(start)
    seen = {start}
    q = deque([(start, 0)])
    while q:
        (r, c), d = q.popleft()
        if r == goal_row:
            return d
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < SIZE and 0 <= n[1] < SIZE and n not in seen and ((r, c), n) not in blocked:
                seen.add(n)
                q.append((n, d + 1))
    return -1


def plan(my_goal, opp_goal, lane_col):
    """8 tường kế hoạch theo thứ tự ưu tiên: hàng rào, rồi hành lang từ phía đích đối thủ."""
    fence_row = 7 if opp_goal == SIZE - 1 else 0            # ranh giới sát đích đối thủ
    fence_cols = (0, 2, 4, 6) if lane_col == SIZE - 1 else (1, 3, 5, 7)
    fence = [("H", fence_row, c) for c in fence_cols]
    if lane_col == SIZE - 1:                                 # thứ tự: gần khe hở trước
        fence.reverse()
    v_col = SIZE - 2 if lane_col == SIZE - 1 else 0         # tường dọc giữa cột hành lang và bàn
    # Che các hàng trừ hàng đích của mình (lối vào hành lang), bắt đầu từ phía đích đối thủ.
    v_rows = (7, 5, 3, 1) if my_goal == 0 else (0, 2, 4, 6)
    lane = [("V", r, v_col) for r in v_rows]
    return fence + lane


def best_plan_wall(observation, me, op, my_goal, opp_goal, lane_col, hw, vw, blocked, d_opp,
                   min_gain):
    """Tường kế hoạch hợp lệ làm đối thủ chậm nhất (>= min_gain bước), không tự làm mình
    chậm (tường rào cho phép chậm <= 2). Hòa thì theo thứ tự kế hoạch. None nếu không có."""
    legal = {(w["orientation"], w["row"], w["col"]) for w in observation["legal_walls"]}
    d_me = distance(me, my_goal, blocked)
    best, best_key = None, None
    for i, w in enumerate(plan(my_goal, opp_goal, lane_col)):
        if w not in legal:
            continue
        nb = blocked_edges(hw + [w[1:]] if w[0] == "H" else hw,
                           vw + [w[1:]] if w[0] == "V" else vw)
        gain = distance(op, opp_goal, nb) - d_opp
        d_me2 = distance(me, my_goal, nb)
        if gain < min_gain or d_me2 < 0 or d_me2 - d_me > (2 if w[0] == "H" else 0):
            continue
        key = (gain, -i)
        if best_key is None or key > best_key:
            best, best_key = w, key
    return best


def timed(observation, me, op, my_goal, opp_goal, moves, dist_map, hw, vw, blocked, d_opp):
    """Phễu đúng lúc: giấu bài tới khi đối thủ còn cách đích <= TRIGGER bước, rồi đặt
    liên tiếp các tường kế hoạch (mỗi lượt bức làm đối thủ chậm nhất)."""
    walls = set(("H",) + w for w in hw) | set(("V",) + w for w in vw)
    # Cột hành lang: giữ mép đã bắt đầu xây; chưa xây thì chọn mép xa cột đối thủ
    # (đối thủ phải vòng xa nhất), hòa thì mép gần mình, rồi mép phải.
    started = [c for c in (SIZE - 1, 0) if walls & set(plan(my_goal, opp_goal, c))]
    if started:
        lane_col = started[0]
    else:
        lane_col = max((SIZE - 1, 0), key=lambda c: (abs(c - op[1]), -abs(c - me[1]), c))
    triggered = bool(started) or d_opp <= TRIGGER

    if triggered and observation["you"]["walls_remaining"] > 0:
        w = best_plan_wall(observation, me, op, my_goal, opp_goal, lane_col, hw, vw, blocked,
                           d_opp, min_gain=1)
        if w is not None:
            return {"type": "wall", "orientation": w[0], "row": w[1], "col": w[2]}
    if not triggered and me[1] != lane_col and d_opp >= TRIGGER + 3:
        side = (me[0], me[1] + (1 if lane_col > me[1] else -1))   # còn dư thời gian: vào hành lang
        if side in moves:
            return {"type": "move", "to": list(side)}
    m = min(moves, key=lambda x: dist_map[x[0]][x[1]] if dist_map[x[0]][x[1]] >= 0 else 99)
    return {"type": "move", "to": list(m)}


def choose_action(observation):
    try:
        you, opp = observation["you"], observation["opponent"]
        me, op = tuple(you["position"]), tuple(opp["position"])
        my_goal, opp_goal = you["goal_row"], opp["goal_row"]
        moves = [tuple(m) for m in observation["legal_moves"]]
        dist_map = observation["distances"]

        for m in moves:                                      # 1. thắng ngay
            if m[0] == my_goal:
                return {"type": "move", "to": list(m)}

        hw = [tuple(w) for w in observation["walls"]["horizontal"]]
        vw = [tuple(w) for w in observation["walls"]["vertical"]]
        blocked = blocked_edges(hw, vw)
        d_opp = distance(op, opp_goal, blocked)

        if TRIGGER is not None:
            return timed(observation, me, op, my_goal, opp_goal, moves, dist_map, hw, vw,
                         blocked, d_opp)

        # Cột hành lang: mép gần mình hơn (hòa thì mép phải).
        lane_col = SIZE - 1 if me[1] >= SIZE // 2 else 0

        # 2. đi sang cột hành lang khi đối thủ còn xa, hoặc khi hàng rào đã xong
        #    (vào hành lang trước rồi mới đóng hành lang, kẻo tự nhốt mình).
        fence = [w for w in plan(my_goal, opp_goal, lane_col) if w[0] == "H"]
        fence_done = all((w[1], w[2]) in hw for w in fence)
        if me[1] != lane_col and (d_opp >= 6 or fence_done):
            step = 1 if lane_col > me[1] else -1
            side = (me[0], me[1] + step)
            if side in moves:
                return {"type": "move", "to": list(side)}

        # 3. đặt tường kế hoạch
        if you["walls_remaining"] > 0:
            legal = {(w["orientation"], w["row"], w["col"]) for w in observation["legal_walls"]}
            best, best_key = None, None
            for i, w in enumerate(plan(my_goal, opp_goal, lane_col)):
                if w not in legal:
                    continue
                nb = blocked_edges(hw + [w[1:]] if w[0] == "H" else hw,
                                   vw + [w[1:]] if w[0] == "V" else vw)
                gain = distance(op, opp_goal, nb) - d_opp
                # Không tự làm mình chậm; riêng tường rào cho phép chậm tối đa 2 bước
                # (mình đứng trên hàng rào thì đằng nào cũng phải đi vào hành lang).
                slow = distance(me, my_goal, nb) - distance(me, my_goal, blocked)
                if slow > (2 if w[0] == "H" else 0) or distance(me, my_goal, nb) < 0:
                    continue
                key = (gain, -i)
                if best_key is None or key > best_key:
                    best, best_key = w, key
            if best is not None:
                return {"type": "wall", "orientation": best[0], "row": best[1], "col": best[2]}

        # 4. đi theo đường ngắn nhất
        if moves:
            m = min(moves, key=lambda x: dist_map[x[0]][x[1]] if dist_map[x[0]][x[1]] >= 0 else 99)
            return {"type": "move", "to": list(m)}
    except Exception:
        pass
    return {"type": "pass"}
