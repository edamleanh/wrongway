import os

import pytest

from training.alphazero.data import load_buffer, load_shard, save_shard


def sample(z):
    return ([0.0] * 650 + [1.0], [1.0] + [0.0] * 209, z)


def test_shard_roundtrip(tmp_path):
    save_shard(tmp_path / "a.npz", [sample(1.0), sample(-1.0)])
    x, pi, z = load_shard(tmp_path / "a.npz")
    assert x.shape == (2, 651) and pi.shape == (2, 210) and list(z) == [1.0, -1.0]


def test_buffer_keeps_newest_positions(tmp_path):
    save_shard(tmp_path / "old.npz", [sample(1.0), sample(1.0)])
    save_shard(tmp_path / "new.npz", [sample(-1.0), sample(-1.0)])
    os.utime(tmp_path / "old.npz", (1, 1))
    x, pi, z = load_buffer(tmp_path, 3)
    assert len(z) == 3 and sorted(z.tolist()) == [-1.0, -1.0, 1.0]


def test_buffer_pattern_and_empty(tmp_path):
    save_shard(tmp_path / "bootstrap.npz", [sample(0.0)])
    save_shard(tmp_path / "selfplay_iter001.npz", [sample(1.0)])
    assert load_buffer(tmp_path, 10, "bootstrap*.npz")[2].tolist() == [0.0]
    with pytest.raises(FileNotFoundError):
        load_buffer(tmp_path / "missing", 10)
