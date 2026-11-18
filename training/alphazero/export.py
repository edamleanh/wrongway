"""Inference weights: torch nets -> .pth -> text blobs embedded in the agent file.

The .pth is the hand-off artifact: plain tensors in agent order, no torch module code needed."""
import torch

from .blob import embed, pack_layers
from .engine import E
from .net import policy_layers, tiny_layers

PTH_FORMAT = "wrongway-az-v1"


def layers_from_linears(linears):
    return [(l.weight.detach().cpu().float().tolist(), l.bias.detach().cpu().float().tolist()) for l in linears]


def _tensors(linears):
    out = []
    for layer in linears:
        out.append(layer.weight.detach().cpu().float().clone())
        out.append(layer.bias.detach().cpu().float().clone())
    return out


def export_pth(path, net, tiny, meta):
    torch.save({"format": PTH_FORMAT,
                "policy_shapes": [list(s) for s in E.POLICY_SHAPES],
                "value_shapes": [list(s) for s in E.VALUE_SHAPES],
                "policy": _tensors(policy_layers(net)),
                "value": _tensors(tiny_layers(tiny)),
                "meta": meta}, str(path))


def blobs_from_pth(path, dtype="e"):
    data = torch.load(str(path), map_location="cpu", weights_only=True)
    if data.get("format") != PTH_FORMAT:
        raise ValueError("%s is not a %s file" % (path, PTH_FORMAT))
    if [tuple(s) for s in data["policy_shapes"]] != list(E.POLICY_SHAPES) or \
            [tuple(s) for s in data["value_shapes"]] != list(E.VALUE_SHAPES):
        raise ValueError("layer shapes in %s do not match the agent" % path)

    def layers(tensors):
        return [(tensors[i].tolist(), tensors[i + 1].tolist()) for i in range(0, len(tensors), 2)]

    return pack_layers(layers(data["policy"]), dtype), pack_layers(layers(data["value"]), dtype)


def embed_pth(pth_path, agent_path, dtype="e"):
    model_blob, value_blob = blobs_from_pth(pth_path, dtype)
    embed(agent_path, model_blob, value_blob)
    return len(model_blob) + len(value_blob)
