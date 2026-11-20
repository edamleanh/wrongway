"""One training run = one folder under training/alphazero/runs/<name>/.

manifest.json  config + progress (phases done, current iteration, best checkpoint, history)
data/          .npz sample shards        checkpoints/  .pt files
tb/            TensorBoard events        export/       .pth + benchmark report"""
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from .config import make_config
from .engine import AGENT_FILE

PROCESS_STARTED = time.time()   # p03 stop_after_hours counts from here
RUNS_DIR = Path(os.environ.get("AZ_RUNS_DIR", Path(__file__).resolve().parent / "runs"))
PHASE_MODULES = {
    "p01": "p01_bootstrap", "p02": "p02_pretrain", "p03": "p03_iterate", "p04": "p04_distill",
    "p05": "p05_export_pth", "p06": "p06_embed", "p07": "p07_benchmark",
}
PHASE_ORDER = list(PHASE_MODULES)


class RunContext:
    def __init__(self, name, preset=None, overrides=(), runs_dir=None):
        self.name = name
        self.dir = (Path(runs_dir) if runs_dir else RUNS_DIR) / name
        self.manifest_path = self.dir / "manifest.json"
        if self.manifest_path.exists():
            if preset:
                raise SystemExit("run '%s' already exists; drop --preset (use --set) or pick a new --run" % name)
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.cfg = make_config(base=self.manifest["config"], overrides=overrides)
        else:
            self.cfg = make_config(preset or "standard", overrides=overrides)
            self.manifest = {"preset": preset or "standard", "done": {}, "iteration": 0,
                             "best": None, "history": [], "steps": {}}
        self.manifest["config"] = asdict(self.cfg)
        for sub in ("data", "checkpoints", "tb", "export"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.save()

    def path(self, *parts):
        return self.dir.joinpath(*parts)

    @property
    def agent_path(self):
        return Path(self.cfg.agent_path) if self.cfg.agent_path else AGENT_FILE

    def save(self):
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    def is_done(self, phase):
        return phase in self.manifest["done"]

    def require(self, phase):
        if phase and not self.is_done(phase):
            raise SystemExit("[%s] phase %s must finish first:\n  uv run python -m training.alphazero.phases.%s --run %s"
                             % (self.name, phase, PHASE_MODULES[phase], self.name))

    def mark_done(self, phase):
        for later in PHASE_ORDER[PHASE_ORDER.index(phase) + 1:]:
            self.manifest["done"].pop(later, None)       # built on old outputs -> stale
        self.manifest["done"][phase] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def writer(self):
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(self.path("tb")))
