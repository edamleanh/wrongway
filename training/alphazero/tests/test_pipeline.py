import json
import os
import shutil
import subprocess
import sys

import pytest

from training.alphazero.engine import AGENT_FILE, PROJECT, load_agent


def run_cli(args, runs_dir, timeout=900):
    env = dict(os.environ, AZ_RUNS_DIR=str(runs_dir))
    return subprocess.run([sys.executable, "-m"] + args, cwd=str(PROJECT), env=env,
                          capture_output=True, text=True, timeout=timeout)


def test_phase_refuses_to_run_before_its_dependency(tmp_path):
    r = run_cli(["training.alphazero.phases.p02_pretrain", "--run", "fresh", "--preset", "smoke"], tmp_path)
    assert r.returncode != 0
    assert "p01_bootstrap" in (r.stderr + r.stdout)


def test_bootstrap_then_pretrain(tmp_path):
    r = run_cli(["training.alphazero.phases.p01_bootstrap", "--run", "s", "--preset", "smoke"], tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    r = run_cli(["training.alphazero.phases.p02_pretrain", "--run", "s"], tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    run = tmp_path / "s"
    manifest = json.loads((run / "manifest.json").read_text())
    assert set(manifest["done"]) == {"p01", "p02"}
    assert manifest["best"] == "checkpoints/iter000.pt"
    assert (run / "data" / "bootstrap.npz").exists() and (run / "checkpoints" / "iter000.pt").exists()
    assert list((run / "tb").glob("events.out.tfevents.*"))
