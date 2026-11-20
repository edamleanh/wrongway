"""Phase 6 - embed the .pth into the single-file agent (pure-Python inference, stdlib only).

Needs: p05.  Rewrites MODEL_BLOB / VALUE_BLOB / DEPLOY_MODE between the MODEL BLOBS markers
of the agent (default submissions/az_agent.py; change with --set agent_path=...), re-imports
it and checks the pure-Python outputs against the torch nets."""
import random
import sys
import time

import torch

from ..adapters import TorchPolicy, TorchValue
from ..engine import load_agent
from ..export import embed_pth
from ..games import random_opening
from ..net import TinyValue, load_checkpoint

PHASE, REQUIRES, TITLE = "p06", "p05", "embed weights into the agent .py"


def _probe_states(n=20, seed=0):
    rng = random.Random(seed)
    return [random_opening(rng, 6) for _ in range(n)]


def run(ctx):
    agent_path = ctx.agent_path
    size = embed_pth(ctx.path(ctx.manifest["export_pth"]), agent_path, ctx.cfg.blob_dtype)
    agent = load_agent(agent_path, name="embedded_%d" % time.time_ns())
    if agent._POLICY is None or agent._VALUE is None:
        raise SystemExit("agent could not decode the embedded blobs: " + str(agent_path))
    policy = TorchPolicy(load_checkpoint(ctx.path(ctx.manifest["best"])))
    tiny = TinyValue()
    tiny.load_state_dict(torch.load(str(ctx.path("checkpoints", "tiny_value.pt")), weights_only=True)["tiny"])
    value = TorchValue(tiny)
    states = _probe_states()
    max_policy = max(max(abs(a - b) for a, b in zip(agent._POLICY.logits(s), policy.logits(s))) for s in states)
    max_value = max(abs(agent._VALUE.value(s) - value.value(s)) for s in states)
    ctx.manifest["embed"] = {"agent": str(agent_path), "blob_chars": size, "file_bytes": agent_path.stat().st_size,
                             "max_policy_diff": max_policy, "max_value_diff": max_value}
    ctx.save()
    print("  embedded into %s (%.0f KB), max |diff| policy %.4f value %.4f"
          % (agent_path, agent_path.stat().st_size / 1024, max_policy, max_value))
    if max_policy > 5e-2 or max_value > 2e-2:
        raise SystemExit("pure-Python inference differs from torch - try --set blob_dtype=f --force")


if __name__ == "__main__":
    from ._cli import main
    main(sys.modules[__name__])
