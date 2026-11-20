"""Network weights <-> a text blob that lives inside the single-file agent.

blob = base85( zlib( dtype byte + little-endian floats ) ),  dtype 'e' = float16, 'f' = float32.
Float order, layer by layer: weight rows in PyTorch layout [out][in] (row-major), then bias.
The agent's unpack_blob()/MLP read exactly this order."""
import base64
import re
import struct
import zlib
from pathlib import Path

BLOCK = re.compile(r"(# =+ MODEL BLOBS BEGIN[^\n]*\n)(.*?)(# =+ MODEL BLOBS END)", re.S)


def pack_layers(layers, dtype="e"):
    """layers: [(weight [out][in] nested lists, bias list), ...] -> blob string."""
    flat = []
    for weight, bias in layers:
        for row in weight:
            flat.extend(row)
        flat.extend(bias)
    if dtype == "e" and max(abs(x) for x in flat) > 60000.0:
        dtype = "f"                                    # float16 would overflow
    raw = dtype.encode("ascii") + struct.pack("<%d%s" % (len(flat), dtype), *flat)
    return base64.b85encode(zlib.compress(raw, 9)).decode("ascii")


def _literal(name, blob, width=4096):
    """Source for `name = "..."`, split in 4 KB chunks so editors cope (base85 has no quotes/backslashes)."""
    if not blob:
        return '%s = ""\n' % name
    chunks = "".join('    "%s"\n' % blob[i:i + width] for i in range(0, len(blob), width))
    return "%s = (\n%s)\n" % (name, chunks)


def embed(agent_path, model_blob, value_blob, mode="learned_alphabeta"):
    """Rewrite MODEL_BLOB / VALUE_BLOB / DEPLOY_MODE between the MODEL BLOBS markers."""
    path = Path(agent_path)
    text = path.read_text(encoding="utf-8")
    body = _literal("MODEL_BLOB", model_blob) + _literal("VALUE_BLOB", value_blob) + 'DEPLOY_MODE = "%s"\n' % mode
    new, count = BLOCK.subn(lambda m: m.group(1) + body + m.group(3), text)
    if count != 1:
        raise ValueError("MODEL BLOBS markers must appear exactly once in %s" % path)
    path.write_text(new, encoding="utf-8")
