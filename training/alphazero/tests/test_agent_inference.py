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
