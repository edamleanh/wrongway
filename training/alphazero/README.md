# AlphaZero pipeline for the Wrongway agent

Trains a policy + value net by self-play and ships it inside one pure-Python file:
`wrongway-student/wrongway/submissions/az_agent.py` (fork of the submission `submissions/MSSV.py`:
same bitboard alpha-beta + TD-learned table eval; the nets act only at the root).
Spec: `docs/alphazero_pipeline.md`. Plan + deviations: `docs/superpowers/plans/2026-11-18-alphazero-pipeline.md`.

## Setup

```bash
uv sync                    # torch, tensorboard, numpy, pytest into .venv
uv run pytest -m "not slow"
```

Devices (training/distill only; games always run on CPU worker processes):
`--set device=auto` (default) picks `cuda` > `mps` > `cpu`. Linux + NVIDIA: the default PyPI
wheel already has CUDA. Windows + NVIDIA: install a CUDA wheel, e.g.
`uv pip install torch --index-url https://download.pytorch.org/whl/cu124`.

## Phases (each needs the previous one)

| # | Script (`uv run python -m training.alphazero.phases.…`) | Produces | TensorBoard |
|---|---|---|---|
| 1 | `p01_bootstrap` | `data/bootstrap.npz` | `bootstrap/*` |
| 2 | `p02_pretrain` | `checkpoints/iter000.pt` | `train/*` |
| 3 | `p03_iterate` | `data/selfplay_iterNNN.npz`, `checkpoints/iterNNN.pt` | `selfplay/*`, `train/*`, `gate/*` |
| 4 | `p04_distill` | `checkpoints/tiny_value.pt` | `distill/*` |
| 5 | `p05_export_pth` | `export/az_agent.pth` | – |
| 6 | `p06_embed` | blobs inside `submissions/az_agent.py` | – |
| 7 | `p07_benchmark` | `export/benchmark.txt` | `bench/*` |

Gate (phase 3): a candidate must score ≥ `accept_score` vs the best net **and** ≥ `anchor_min`
vs the same agent with no network (`gate/anchor_score`) - which plays move-for-move like
`submissions/MSSV.py` - so an accepted net is never weaker than the current submission.

How the agent uses the nets (constants at the top of section 2 of the agent file):
- policy: root score += `POLICY_WEIGHT`·log p(move) (floored at `POLICY_FLOOR`), root moves sorted by p,
  plus up to `POLICY_EXTRA` legal walls the policy likes that alpha-beta's generator skipped;
- value: root score += `VALUE_WEIGHT`·value(after move) (default 0 = unused).
Inside the tree the TD table eval stays: a pure-Python net is too slow per leaf.

All outputs live in `training/alphazero/runs/<run>/` (gitignored), progress in its `manifest.json`.

```bash
uv run python -m training.alphazero.pipeline --run try1 --preset smoke     # ~minutes, checks wiring
uv run python -m training.alphazero.pipeline --run main --preset quick     # first real run
uv run python -m training.alphazero.pipeline --run main --status
uv run tensorboard --logdir training/alphazero/runs
```

Presets: `smoke` (wiring test), `quick`, `standard` (default), `spec` (exact spec numbers, slow).
Change anything with `--set key=value` (fields in `config.py`). Redoing a phase (`--force`)
marks every later phase as pending. Phase 3 resumes where it stopped.

Phase 6 overwrites the agent file. To try weights without touching the real one, give the
run its own copy: `--set agent_path=/path/to/copy.py` when the run is created.

## Using the result

- Embed any exported weights into any agent file:
  `uv run python -m training.alphazero.embed_pth --pth runs/main/export/az_agent.pth --agent path/to/agent.py`
- Submit: copy `az_agent.py` to `<MSSV>.py`. It imports only the standard library.
- Benchmark rule (spec): accept only if ≥ 55% vs current `MSSV` over 80 games, no move > 0.55 s,
  tests pass. Record the result in `training/EXPERIMENTS.md`.

## Known risks

- Distillation: on ~190-hot inputs a high learning rate kills ReLUs (lr 3e-3 killed all 32 in a
  test). Keep `lr` at 1e-3 or lower for p04; if Pearson < 0.9 raise `distill_epochs` or
  `distill_positions` first.
- Pure-Python cost measured on an M-series laptop: policy ~2.5 ms per turn, tiny value
  ~0.15 ms per root move (encode included). Colab is ~2x slower.
