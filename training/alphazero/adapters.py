"""Torch nets behind the agent's Search interface (.logits(state) / .value(state)),
so gating games run the exact deployment search code with full-precision nets."""
import torch

from .engine import E
from .symmetry import N_ACTIONS, rotate_for


def _features(state):
    return torch.tensor([E.encode(state)], dtype=torch.float32)


class TorchPolicy:
    def __init__(self, net):
        self.net = net.eval()

    @torch.no_grad()
    def logits(self, state):
        out = self.net(_features(state))[0][0].tolist()
        rotate = rotate_for(state)
        return [out[E.transform_action(a, rotate)] for a in range(N_ACTIONS)]


class TorchValue:
    """Works with AZNet (uses its value head) or TinyValue."""
    def __init__(self, module):
        self.module = module.eval()

    @torch.no_grad()
    def value(self, state):
        out = self.module(_features(state))
        if isinstance(out, tuple):
            out = out[1]
        return float(out[0])
