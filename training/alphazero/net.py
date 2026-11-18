"""PyTorch networks (training only; the agent runs the same math in plain Python).

AZNet:     651 -> 256 -> 128 backbone, policy head 128 -> 210 logits,
           value head 128 -> 32 -> 1 tanh (+1 = side to move wins)
TinyValue: 651 -> 32 -> 1 tanh, distilled from AZNet's value head for fast leaf eval."""
import torch
from torch import nn

N_FEATURES, N_ACTIONS = 651, 210


class AZNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(N_FEATURES, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.policy = nn.Linear(128, N_ACTIONS)
        self.value = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x):
        h = self.backbone(x)
        return self.policy(h), self.value(h).squeeze(-1)


class TinyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEATURES, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x):
        return self.net(x).squeeze(-1)


def policy_layers(net):
    """Linear layers the agent's PolicyNet runs, in order (matches POLICY_SHAPES)."""
    return [net.backbone[0], net.backbone[2], net.policy]


def tiny_layers(tiny):
    """Linear layers the agent's TinyValueNet runs, in order (matches VALUE_SHAPES)."""
    return [tiny.net[0], tiny.net[2]]


def save_checkpoint(path, net, iteration):
    torch.save({"backbone": net.backbone.state_dict(), "policy": net.policy.state_dict(),
                "value": net.value.state_dict(), "iter": iteration}, str(path))


def load_checkpoint(path, device="cpu"):
    data = torch.load(str(path), map_location=device, weights_only=True)
    net = AZNet()
    net.backbone.load_state_dict(data["backbone"])
    net.policy.load_state_dict(data["policy"])
    net.value.load_state_dict(data["value"])
    return net.to(device).eval()
