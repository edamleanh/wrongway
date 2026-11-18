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
