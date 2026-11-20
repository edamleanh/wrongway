"""Phase 4 - distill the best net's value head into TinyValue (651 -> 32 -> 1).

Needs: p03.  Writes: checkpoints/tiny_value.pt, manifest["distill"] (val MSE, Pearson).
Spec target: Pearson >= 0.90 on held-out positions (warning only; p07 is the real gate)."""
import sys

import torch

from ..config import pick_device
from ..data import load_buffer
from ..net import TinyValue, load_checkpoint
from ..trainer import distill_value, teacher_values

PHASE, REQUIRES, TITLE = "p04", "p03", "distill value head -> checkpoints/tiny_value.pt"


def run(ctx):
    cfg = ctx.cfg
    device = pick_device(cfg.device)
    x, _, _ = load_buffer(ctx.path("data"), cfg.distill_positions)
    teacher = load_checkpoint(ctx.path(ctx.manifest["best"]))
    targets = teacher_values(teacher, x, device, cfg=cfg)
    torch.manual_seed(cfg.seed)
    tiny = TinyValue()
    writer = ctx.writer()
    metrics = distill_value(tiny, x, targets, cfg, device, writer)
    writer.close()
    torch.save({"tiny": tiny.cpu().state_dict(), "teacher": ctx.manifest["best"], **metrics},
               str(ctx.path("checkpoints", "tiny_value.pt")))
    ctx.manifest["distill"] = metrics
    ctx.save()
    print("  distill: val_mse %.4f, val_pearson %.3f" % (metrics["val_mse"], metrics["val_pearson"]))
    if metrics["val_pearson"] < cfg.distill_min_pearson:
        print("  WARNING: Pearson below %.2f - try --set distill_epochs=100 --force" % cfg.distill_min_pearson)


if __name__ == "__main__":
    from ._cli import main
    main(sys.modules[__name__])
