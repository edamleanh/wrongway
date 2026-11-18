"""Phase 5 - export inference weights (best policy net + tiny value net) to one .pth.

Needs: p04.  Writes: export/<agent name>.pth  (plain tensors in agent layer order + meta).
This file is the hand-off artifact: embed it anywhere with training.alphazero.embed_pth."""
import sys

import torch

from ..export import export_pth
from ..net import TinyValue, load_checkpoint

PHASE, REQUIRES, TITLE = "p05", "p04", "export weights -> export/<agent>.pth"


def run(ctx):
    net = load_checkpoint(ctx.path(ctx.manifest["best"]))
    tiny = TinyValue()
    tiny.load_state_dict(torch.load(str(ctx.path("checkpoints", "tiny_value.pt")), weights_only=True)["tiny"])
    out = ctx.path("export", ctx.agent_path.stem + ".pth")
    export_pth(out, net, tiny, {"run": ctx.name, "best": ctx.manifest["best"],
                                "iteration": ctx.manifest["iteration"], "distill": ctx.manifest.get("distill")})
    ctx.manifest["export_pth"] = str(out.relative_to(ctx.dir))
    ctx.save()
    print("  wrote", out)


if __name__ == "__main__":
    from ._cli import main
    main(sys.modules[__name__])
