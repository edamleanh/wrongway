import shutil

import torch

from training.alphazero.adapters import TorchPolicy, TorchValue
from training.alphazero.blob import pack_layers
from training.alphazero.engine import AGENT_FILE, E, load_agent
from training.alphazero.export import blobs_from_pth, embed_pth, export_pth, layers_from_linears
from training.alphazero.net import AZNet, TinyValue, load_checkpoint, policy_layers, save_checkpoint, tiny_layers
from training.alphazero.tests.helpers import random_states

STATES = None


def states():
    global STATES
    if STATES is None:
        STATES = list(random_states(n_games=3, seed=11))[:25]
    return STATES


def test_forward_shapes():
    logits, value = AZNet()(torch.zeros(4, 651))
    assert logits.shape == (4, 210) and value.shape == (4,)
    assert TinyValue()(torch.zeros(3, 651)).shape == (3,)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = AZNet()
    save_checkpoint(tmp_path / "a.pt", net, 3)
    back = load_checkpoint(tmp_path / "a.pt")
    x = torch.rand(2, 651)
    assert torch.allclose(net(x)[0], back(x)[0]) and torch.allclose(net(x)[1], back(x)[1])
    assert torch.load(tmp_path / "a.pt", weights_only=True)["iter"] == 3


def test_pure_python_policy_matches_torch():
    torch.manual_seed(1)
    net = AZNet()
    py = E.PolicyNet(pack_layers(layers_from_linears(policy_layers(net)), "f"))
    tp = TorchPolicy(net)
    for state in states():
        assert max(abs(a - b) for a, b in zip(py.logits(state), tp.logits(state))) < 1e-4


def test_pure_python_tiny_value_matches_torch():
    torch.manual_seed(2)
    tiny = TinyValue()
    py = E.TinyValueNet(pack_layers(layers_from_linears(tiny_layers(tiny)), "f"))
    tv = TorchValue(tiny)
    for state in states():
        assert abs(py.value(state) - tv.value(state)) < 1e-5


def test_pth_export_and_embed(tmp_path):
    torch.manual_seed(3)
    net, tiny = AZNet(), TinyValue()
    pth = tmp_path / "w.pth"
    export_pth(pth, net, tiny, {"note": "test"})
    model_blob, value_blob = blobs_from_pth(pth, "e")
    assert E.PolicyNet(model_blob) and E.TinyValueNet(value_blob)
    agent = tmp_path / "agent.py"
    shutil.copy(AGENT_FILE, agent)
    embed_pth(pth, agent, "e")
    module = load_agent(agent, name="agent_pth_embedded")
    tp, tv = TorchPolicy(net), TorchValue(tiny)
    for state in states()[:10]:
        assert max(abs(a - b) for a, b in zip(module._POLICY.logits(state), tp.logits(state))) < 5e-2
        assert abs(module._VALUE.value(state) - tv.value(state)) < 2e-2
