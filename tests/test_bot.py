"""Unit test cho bot: python -m unittest discover tests

- Luật mô phỏng trong bot phải khớp Arena (nước đi, tường hợp lệ, khoảng cách).
- Bot luôn trả hành động hợp lệ, đủ nhanh, ở cả hai bên.
Đổi file bot bằng biến môi trường BOT=<đường dẫn>.
"""

import importlib.util
import os
import random
import sys
import time
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "tools"))
import _arena_path  # noqa: E402,F401
from environment import Environment  # noqa: E402

BOT_FILE = os.environ.get("BOT", str(_arena_path.SUBMISSIONS / "MSSV.py"))


def load_bot(path=BOT_FILE):
    spec = importlib.util.spec_from_file_location("bot_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_states(n_games=30, seed=1):
    """Sinh trạng thái đa dạng bằng ván ngẫu nhiên (nhiều tường)."""
    rng = random.Random(seed)
    for _ in range(n_games):
        env = Environment(max_turns=120)
        while not env.done:
            yield env
            walls = env.legal_wall_actions()
            if walls and rng.random() < 0.35:
                w = rng.choice(walls)
                env.step({"type": "wall", **w})
            else:
                env.step({"type": "move", "to": rng.choice(env.legal_moves())})


class RulesMatchArena(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = load_bot()

    def test_rules(self):
        b = self.bot
        checked = 0
        for env in random_states():
            hw, vw = frozenset(env.horizontal_walls), frozenset(env.vertical_walls)
            blocked = b.blocked_from(hw, vw)
            p = env.current_player
            me, opp = b.cell(*env.positions[p]), b.cell(*env.positions[1 - p])
            got = [[b.ROW[c], b.COL[c]] for c in b.legal_moves(me, opp, blocked)]
            self.assertEqual(got, env.legal_moves(p))
            arena_map = env.distance_map(p)
            dmap = b.distance_map(env.goal_rows[p], blocked)
            self.assertEqual(dmap, [arena_map[r][c] for r in range(9) for c in range(9)])
            self.assertEqual(b.distance(me, env.goal_rows[p], blocked), dmap[me])
            if checked % 5 == 0:   # kiểm tra tường chậm hơn, lấy mẫu
                arena = {(w["orientation"], w["row"], w["col"])
                         for w in env.legal_wall_actions(p)}
                mine = set()
                if env.walls_remaining[p] > 0:
                    for o in "HV":
                        for r in range(8):
                            for c in range(8):
                                if not b.wall_fits(o, r, c, hw, vw):
                                    continue
                                nb = b.with_wall(blocked, o, r, c)
                                if all(b.distance(b.cell(*env.positions[i]), env.goal_rows[i], nb) >= 0
                                       for i in (0, 1)):
                                    mine.add((o, r, c))
                self.assertEqual(mine, arena)
            checked += 1
        self.assertGreater(checked, 500)


class BotBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = load_bot()

    def assert_legal(self, env, action):
        obs = env.get_observation()
        kind = action.get("type")
        if kind == "move":
            self.assertIn(list(action["to"]), obs["legal_moves"])
        elif kind == "wall":
            self.assertIn({k: action[k] for k in ("orientation", "row", "col")}, obs["legal_walls"])
        else:
            self.assertEqual(kind, "pass")

    def test_self_play_legal_and_fast(self):
        env = Environment(max_turns=200)
        slowest = 0.0
        while not env.done:
            obs = env.get_observation()
            t0 = time.perf_counter()
            action = self.bot.choose_action(obs)
            slowest = max(slowest, time.perf_counter() - t0)
            self.assert_legal(env, action)
            env.step(action)
        self.assertLess(slowest, 0.9)

    def test_takes_immediate_win(self):
        env = Environment()
        env.positions = [(1, 4), (7, 0)]
        action = self.bot.choose_action(env.get_observation(0))
        self.assertEqual(action, {"type": "move", "to": [0, 4]})


if __name__ == "__main__":
    unittest.main()
