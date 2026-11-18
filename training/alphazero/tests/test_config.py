from pathlib import Path

import pytest

from training.alphazero.config import make_config, pick_device
from training.alphazero.run_state import RunContext


def test_presets_and_overrides():
    cfg = make_config("smoke", overrides=["simulations=7", "mirror_augment=false", "boot_budget=0.5"])
    assert cfg.simulations == 7
    assert cfg.mirror_augment is False
    assert cfg.boot_budget == 0.5
    assert cfg.iterations == 1                     # from the smoke preset


def test_unknown_override_key_is_an_error():
    with pytest.raises(ValueError, match="nope"):
        make_config(overrides=["nope=1"])


def test_pick_device():
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in ("cpu", "cuda", "mps")


def test_run_context_persists_and_enforces_order(tmp_path):
    ctx = RunContext("t", preset="smoke", runs_dir=tmp_path)
    for sub in ("data", "checkpoints", "tb", "export"):
        assert (tmp_path / "t" / sub).is_dir()
    with pytest.raises(SystemExit, match="p01_bootstrap"):
        ctx.require("p01")
    ctx.mark_done("p01")
    ctx.mark_done("p02")
    again = RunContext("t", runs_dir=tmp_path, overrides=["simulations=9"])
    assert again.is_done("p02") and again.cfg.simulations == 9 and again.cfg.iterations == 1
    again.mark_done("p01")                         # redoing an early phase makes later ones stale
    assert not again.is_done("p02")


def test_existing_run_rejects_new_preset(tmp_path):
    RunContext("t", preset="smoke", runs_dir=tmp_path)
    with pytest.raises(SystemExit, match="already exists"):
        RunContext("t", preset="quick", runs_dir=tmp_path)


def test_agent_path_default_and_override(tmp_path):
    ctx = RunContext("t", preset="smoke", runs_dir=tmp_path)
    assert ctx.agent_path.name == "az_agent.py"
    other = RunContext("u", preset="smoke", runs_dir=tmp_path, overrides=["agent_path=%s" % (tmp_path / "x.py")])
    assert other.agent_path == Path(tmp_path / "x.py")
