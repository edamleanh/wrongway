"""Every knob of the pipeline in one dataclass, plus named presets.

A run stores its resolved config in manifest.json; change it later with --set key=value."""
import os
from dataclasses import dataclass, field, fields


def _default_workers():
    return max(1, (os.cpu_count() or 2) - 1)


@dataclass
class RunConfig:
    seed: int = 0
    device: str = "auto"              # auto | cpu | cuda | mps  (training + distill only)
    workers: int = field(default_factory=_default_workers)   # CPU processes for games
    agent_path: str = ""              # "" -> wrongway-student/wrongway/submissions/az_agent.py
    # p01 bootstrap: the alpha-beta bot plays itself
    boot_games: int = 2000
    boot_budget: float = 0.10
    opening_plies: int = 4            # 0..N random pawn moves at the start of every game
    ply_cap: int = 200                # longer games stop and count as draws
    wall_limit: int = 16
    # p02 / p03 training
    batch_size: int = 512
    lr: float = 1e-3
    lr_min: float = 1e-4
    weight_decay: float = 1e-4
    pretrain_epochs: int = 5
    epochs_per_iter: int = 10
    buffer_size: int = 500_000
    mirror_augment: bool = True
    log_every: int = 50
    # p03 self-play
    iterations: int = 20
    sp_games: int = 200
    simulations: int = 100
    c_puct: float = 1.5
    temp_moves: int = 10
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    # p03 gate
    eval_games: int = 40
    eval_budget: float = 0.10
    accept_score: float = 0.55
    retry_lr_factor: float = 0.5
    anchor_games: int = 20            # candidate also plays the no-network engine (heuristic() leaves)
    anchor_min: float = 0.5           # ...and must score at least this against it to be accepted
    # p04 distill
    distill_positions: int = 200_000
    distill_epochs: int = 50
    distill_val_frac: float = 0.05
    distill_min_pearson: float = 0.90
    # p06 embed
    blob_dtype: str = "e"             # e = float16 (half size), f = float32
    # p07 benchmark
    bench_pairs: int = 40
    bench_opponents: str = "MSSV,training/opponents/v2_trained.py"
    bench_workers: int = 4


PRESETS = {
    "standard": {},
    "spec": {"boot_games": 5000, "boot_budget": 0.55, "eval_budget": 0.55, "distill_positions": 500_000},
    "quick": {"boot_games": 400, "iterations": 5, "sp_games": 60, "simulations": 48,
              "epochs_per_iter": 4, "eval_games": 20, "distill_epochs": 15, "bench_pairs": 10},
    "smoke": {"workers": 2, "boot_games": 2, "boot_budget": 0.02, "ply_cap": 30, "batch_size": 32,
              "pretrain_epochs": 1, "epochs_per_iter": 1, "iterations": 1, "sp_games": 2,
              "simulations": 4, "eval_games": 2, "anchor_games": 2, "eval_budget": 0.02, "distill_positions": 2000,
              "distill_epochs": 1, "distill_min_pearson": 0.0, "bench_pairs": 1, "bench_workers": 2,
              "log_every": 1},
}


def _parse(current, raw):
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return type(current)(raw)


def make_config(preset="standard", base=None, overrides=()):
    """base: dict saved in a manifest (wins over preset). overrides: ["key=value", ...]."""
    if preset not in PRESETS:
        raise ValueError("unknown preset %r (choose: %s)" % (preset, ", ".join(PRESETS)))
    values = dict(base) if base is not None else dict(PRESETS[preset])
    cfg = RunConfig(**values)
    names = {f.name for f in fields(RunConfig)}
    for item in overrides:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or key not in names:
            raise ValueError("bad override %r (unknown key %r)" % (item, key))
        setattr(cfg, key, _parse(getattr(cfg, key), raw))
    return cfg


def pick_device(name="auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
