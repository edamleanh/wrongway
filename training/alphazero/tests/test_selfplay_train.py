import random

from training.alphazero.adapters import TorchPolicy
from training.alphazero.engine import SUBMISSIONS, load_agent
from training.alphazero.games import _init_worker, agent_selfplay_game, random_opening, side_eval_game
from training.alphazero.net import AZNet, save_checkpoint
from training.alphazero.selfplay_train import net_from_agent, side_scores

STD1 = SUBMISSIONS / "az_agent_std1.py"


def test_net_from_agent_matches_embedded_policy():
    net = net_from_agent(STD1)
    agent = load_agent(STD1, name="std1_for_test")
    torch_policy, rng = TorchPolicy(net), random.Random(3)
    for _ in range(5):
        state = random_opening(rng, 6)
        assert max(abs(a - b) for a, b in zip(agent._POLICY.logits(state), torch_policy.logits(state))) < 1e-3


def test_agent_selfplay_and_side_eval(tmp_path):
    ckpt = tmp_path / "m.pt"
    save_checkpoint(ckpt, AZNet(), 0)
    _init_worker({"model": str(ckpt), "new": str(ckpt), "old": str(ckpt)})
    params = {"opponent": "base", "learner_side": 1, "budget": 0.01, "wall_limit": 8, "opening_plies": 2,
              "ply_cap": 12, "explore_plies": 4, "explore_eps": 0.5}
    records, stats = agent_selfplay_game((5, params))
    assert records and all(len(f) == 651 and abs(sum(pi) - 1) < 1e-6 for f, pi, _ in records)
    assert stats["plies"] <= 12 and stats["opponent"] == "base"
    tag, side, score = side_eval_game((1, 1, 0.01, 8, 2, 12, "base"))
    assert (tag, side) == ("base", 1) and score in (0.0, 0.5, 1.0)


def test_side_scores():
    s = side_scores([(0, 1.0), (1, 0.0), (0, 1.0), (1, 0.5)])
    assert s == {"all": 0.625, "p0": 1.0, "p1": 0.25}
