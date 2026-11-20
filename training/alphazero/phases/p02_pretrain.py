"""Phase 2 - pretrain the AlphaZero net on the bootstrap data (imitate the alpha-beta bot).

Needs: p01.  Writes: checkpoints/iter000.pt (becomes "best"), resets the iteration loop.
  uv run python -m training.alphazero.phases.p02_pretrain --run main"""
import sys

import torch

from ..config import pick_device
from ..data import load_buffer
from ..net import AZNet, save_checkpoint
from ..trainer import train_policy_value

PHASE, REQUIRES, TITLE = "p02", "p01", "pretrain on bootstrap data -> checkpoints/iter000.pt"


def run(ctx):
    cfg = ctx.cfg
    # Redoing pretraining restarts the loop: drop data/checkpoints of the old iterations.
    for stale in list(ctx.path("data").glob("selfplay_iter*.npz")) + list(ctx.path("checkpoints").glob("iter*.pt")):
        print("  removing stale", stale.relative_to(ctx.dir))
        stale.unlink()
    device = pick_device(cfg.device)
    print("  device:", device)
    torch.manual_seed(cfg.seed)
    data = load_buffer(ctx.path("data"), cfg.buffer_size, "bootstrap*.npz")
    net = AZNet()
    writer = ctx.writer()
    step = train_policy_value(net, data, cfg, device, writer, 0, cfg.pretrain_epochs, cfg.lr, "train")
    writer.close()
    save_checkpoint(ctx.path("checkpoints", "iter000.pt"), net.cpu(), 0)
    ctx.manifest.update({"best": "checkpoints/iter000.pt", "iteration": 0, "history": [],
                         "steps": {"train": step}})
    ctx.save()


if __name__ == "__main__":
    from ._cli import main
    main(sys.modules[__name__])
