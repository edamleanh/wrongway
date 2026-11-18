"""Training samples on disk: one compressed .npz shard per data-producing step.

x  float16 (N, 651)  encode(state) of the position (canonical frame)
pi float16 (N, 210)  target move distribution, canonical frame (one-hot from the alpha-beta
                     teacher, visit counts from MCTS)
z  float32 (N,)      final result for the side to move: +1 win, -1 loss, 0 draw"""
from pathlib import Path

import numpy as np
import torch

from .symmetry import N_ACTIONS, N_FEATURES


def save_shard(path, samples):
    path = Path(path)
    x = np.asarray([s[0] for s in samples], dtype=np.float16).reshape(-1, N_FEATURES)
    pi = np.asarray([s[1] for s in samples], dtype=np.float16).reshape(-1, N_ACTIONS)
    z = np.asarray([s[2] for s in samples], dtype=np.float32)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, x=x, pi=pi, z=z)
    tmp.replace(path)


def load_shard(path):
    with np.load(path) as data:
        return data["x"], data["pi"], data["z"]


def shard_paths(data_dir, pattern="*.npz"):
    paths = [p for p in Path(data_dir).glob(pattern) if not p.name.endswith(".tmp.npz")]
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def load_buffer(data_dir, max_positions, pattern="*.npz"):
    """Replay buffer: newest positions first, up to max_positions, as CPU tensors (x, pi, z)."""
    xs, pis, zs, total = [], [], [], 0
    for path in reversed(shard_paths(data_dir, pattern)):
        take = max_positions - total
        if take <= 0:
            break
        x, pi, z = load_shard(path)
        take = min(take, len(z))
        xs.append(x[len(z) - take:])
        pis.append(pi[len(z) - take:])
        zs.append(z[len(z) - take:])
        total += take
    if not xs:
        raise FileNotFoundError("no .npz shards matching %s in %s" % (pattern, data_dir))
    return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(pis)),
            torch.from_numpy(np.concatenate(zs)))
