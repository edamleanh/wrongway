"""Monte-Carlo tree search guided by the network (AlphaZero PUCT), used for self-play data.

tree[state] = {action: [prior, visits, total_value]}; total_value is seen from the player
who picks `action` at `state`. A fresh tree is built for every move (bounded memory,
and Dirichlet noise can be written straight into the root priors)."""
import math

import torch

from .engine import E
from .symmetry import rotate_for


class NetEvaluator:
    """(state, legal actions) -> ({action: prior}, value for the side to move)."""
    def __init__(self, net):
        self.net = net.eval()

    @torch.no_grad()
    def __call__(self, state, actions):
        logits, value = self.net(torch.tensor([E.encode(state)], dtype=torch.float32))
        out = logits[0].tolist()
        rotate = rotate_for(state)
        raw = {a: out[E.transform_action(a, rotate)] for a in actions}
        top = max(raw.values())
        weights = {a: math.exp(l - top) for a, l in raw.items()}
        total = sum(weights.values())
        return {a: w / total for a, w in weights.items()}, float(value[0])


def terminal_value(state):
    winner = E.outcome(state)
    if winner == 2:
        return 0.0
    return 1.0 if winner == state.turn % 2 else -1.0


def pick_action(visits, temperature, rng):
    """temperature 0 -> most visited; otherwise sample proportional to visits^(1/T)."""
    if temperature <= 0:
        return max(visits, key=lambda a: (visits[a], -a))
    actions = list(visits)
    weights = [visits[a] ** (1.0 / temperature) for a in actions]
    if sum(weights) <= 0:
        return rng.choice(actions)
    return rng.choices(actions, weights)[0]


class MCTS:
    def __init__(self, evaluator, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25, rng=None):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = rng

    def _score(self, edge, sqrt_total):
        prior, visits, total = edge
        q = total / visits if visits else 0.0
        return q + self.c_puct * prior * sqrt_total / (1 + visits)

    def search(self, root, simulations, noise=False):
        """Run `simulations` playouts from a non-terminal root; return {action: visits}."""
        tree = {}

        def expand(state):
            actions = E.legal_actions(state)
            priors, value = self.evaluator(state, actions)
            tree[state] = {a: [priors[a], 0, 0.0] for a in actions}
            return value

        expand(root)
        if noise and self.dirichlet_eps > 0:
            edges = list(tree[root].values())
            sample = [self.rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in edges]
            total = sum(sample) or 1.0
            for edge, g in zip(edges, sample):
                edge[0] = (1 - self.dirichlet_eps) * edge[0] + self.dirichlet_eps * g / total
        for _ in range(simulations):
            state, path = root, []
            while state in tree and E.outcome(state) is None:          # SELECT
                node = tree[state]
                sqrt_total = math.sqrt(max(1, sum(edge[1] for edge in node.values())))
                action = max(node, key=lambda a: self._score(node[a], sqrt_total))
                path.append(node[action])
                state = E.play(state, action, False)
            if E.outcome(state) is not None:
                value = terminal_value(state)
            else:
                value = expand(state)                                    # EXPAND + EVALUATE
            for edge in reversed(path):                                  # BACKPROPAGATE
                value = -value                                           # parent's point of view
                edge[1] += 1
                edge[2] += value
        return {a: edge[1] for a, edge in tree[root].items()}
