import json
import os
import shutil
import subprocess
import sys

import pytest

from training.alphazero.engine import AGENT_FILE, PROJECT, load_agent
from training.alphazero.phases.p07_benchmark import parse_gauntlet


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


def test_iterate_after_pretrain(tmp_path):
    for args in (["training.alphazero.phases.p01_bootstrap", "--run", "s", "--preset", "smoke"],
                 ["training.alphazero.phases.p02_pretrain", "--run", "s"],
                 ["training.alphazero.phases.p03_iterate", "--run", "s"]):
        r = run_cli(args, tmp_path)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    run = tmp_path / "s"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["iteration"] == 1 and manifest["history"]
    assert (run / "data" / "selfplay_iter001.npz").exists()
    assert (run / "checkpoints" / "iter001.pt").exists()
    assert manifest["best"] in ("checkpoints/iter000.pt", "checkpoints/iter001.pt")


def test_distill_export_embed(tmp_path):
    agent = tmp_path / "az_smoke.py"
    shutil.copy(AGENT_FILE, agent)
    steps = [["training.alphazero.phases.p01_bootstrap", "--run", "s", "--preset", "smoke", "--set", "agent_path=%s" % agent]]
    steps += [["training.alphazero.phases.%s" % m, "--run", "s"]
              for m in ("p02_pretrain", "p03_iterate", "p04_distill", "p05_export_pth", "p06_embed")]
    for args in steps:
        r = run_cli(args, tmp_path)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    run = tmp_path / "s"
    manifest = json.loads((run / "manifest.json").read_text())
    assert {"p04", "p05", "p06"} <= set(manifest["done"])
    assert (run / "checkpoints" / "tiny_value.pt").exists()
    assert (run / "export" / "az_smoke.pth").exists()
    module = load_agent(agent, name="az_smoke_embedded")
    assert module.DEPLOY_MODE == "learned_alphabeta" and module._POLICY and module._VALUE
    assert manifest["embed"]["max_policy_diff"] < 5e-2


def test_parse_gauntlet():
    out = ("az (40 cặp ván/đối thủ, mở đầu 2 nước ngẫu nhiên)\n"
           "  vs MSSV           W  45  D   2  L  33  điểm 0.58\n"
           "  vs training/opponents/v2_trained.py W  60  D   0  L  20  điểm 0.75\n")
    assert parse_gauntlet(out) == {"MSSV": 0.58, "training/opponents/v2_trained.py": 0.75}


@pytest.mark.slow
def test_pipeline_smoke_end_to_end(tmp_path):
    agent = tmp_path / "az_e2e.py"
    shutil.copy(AGENT_FILE, agent)
    r = run_cli(["training.alphazero.pipeline", "--run", "e2e", "--preset", "smoke",
                 "--set", "agent_path=%s" % agent, "--to", "6"], tmp_path)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    manifest = json.loads((tmp_path / "e2e" / "manifest.json").read_text())
    assert set(manifest["done"]) == {"p01", "p02", "p03", "p04", "p05", "p06"}
    r = run_cli(["training.alphazero.pipeline", "--run", "e2e", "--status"], tmp_path)
    assert "p06" in r.stdout and "pending" in r.stdout          # p07 still pending
