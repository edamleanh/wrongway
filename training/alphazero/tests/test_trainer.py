import torch
import torch.nn.functional as F

from training.alphazero.config import make_config
from training.alphazero.net import AZNet, TinyValue
from training.alphazero.trainer import distill_value, pearson, teacher_values, train_policy_value


class Recorder:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars.setdefault(tag, []).append((step, value))


def toy_data(n=256):
    rows = torch.arange(n)
    x = torch.zeros(n, 651)
    x[rows, rows % 10] = 1.0
    pi = torch.zeros(n, 210)
    pi[rows, rows % 5] = 1.0
    z = torch.where(rows % 2 == 1, 1.0, -1.0)
    return x.half(), pi.half(), z.float()


def loss_of(net, data):
    x, pi, z = data
    with torch.no_grad():
        logits, value = net(x.float())
        return float(-(pi.float() * F.log_softmax(logits, 1)).sum(1).mean() + F.mse_loss(value, z))


def test_training_reduces_loss_and_logs():
    torch.manual_seed(0)
    cfg = make_config("smoke", overrides=["mirror_augment=false", "batch_size=32", "log_every=1"])
    net, data, rec = AZNet(), toy_data(), Recorder()
    before = loss_of(net, data)
    step = train_policy_value(net, data, cfg, torch.device("cpu"), rec, 0, 30, 1e-2, "t")
    assert loss_of(net, data) < 0.5 * before
    assert step == 30 * 8
    assert {"t/policy_loss", "t/value_loss", "t/lr", "t/policy_top1"} <= set(rec.scalars)


def test_mirror_augmentation_runs():
    cfg = make_config("smoke", overrides=["batch_size=32"])
    train_policy_value(AZNet(), toy_data(64), cfg, torch.device("cpu"), Recorder(), 0, 1, 1e-3, "t")


def test_pearson():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert abs(pearson(a, a) - 1.0) < 1e-6 and abs(pearson(a, -a) + 1.0) < 1e-6


def test_distill_recovers_a_tiny_teacher():
    torch.manual_seed(1)
    teacher = TinyValue()
    x = (torch.rand(8192, 651) < 0.3).half()      # few samples -> 21k-param student just overfits

    class Wrap(torch.nn.Module):          # teacher_values expects (logits, value) like AZNet
        def forward(self, inp):
            return None, teacher(inp)

    targets = teacher_values(Wrap(), x, torch.device("cpu"))
    cfg = make_config("smoke", overrides=["distill_epochs=20", "batch_size=128", "lr=0.001",
                                          "distill_val_frac=0.1", "mirror_augment=false"])
    metrics = distill_value(TinyValue(), x, targets, cfg, torch.device("cpu"), Recorder())
    assert metrics["val_pearson"] > 0.8           # random teacher is a noisy target; real value head is smoother
