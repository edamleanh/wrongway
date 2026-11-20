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

## Self-play of the deployed agent (`selfplay_train.py`)

Continues from an already trained net (default: `MODEL_BLOB` of `submissions/az_agent_std1.py`,
or any `.pt`) without phases 1-2. Unlike p03 (net-only MCTS, weaker than the agent), the teacher
is the agent's own search: alpha-beta + net at the root plays itself (30% of games vs the
no-network engine = MSSV), target = the move it played, value = result. Each generation:
self-play -> train -> gate (vs best, vs no-network; scores split into moving first / second)
-> accepted nets embedded into `runs/<run>/export/az_agent_sp.py`.

```bash
uv run python -m training.alphazero.selfplay_train --run sp1                    # 10 generations
uv run python -m training.alphazero.selfplay_train --run sp1 --set generations=20 # continue
uv run python -m training.alphazero.selfplay_train --run sp1 --status
uv run python -m training.alphazero.selfplay_train --run try --preset smoke     # wiring test
python training/gauntlet.py training/alphazero/runs/sp1/export/az_agent_sp.py MSSV --pairs 40 --workers 6
```

Knobs (`--set`): `sp_budget` (teacher seconds/move, default 0.25 > `eval_budget` 0.10 so the
teacher outplays the student), `league_frac`, `explore_eps`/`explore_plies`, `sp_games`,
`epochs_per_gen`, `lr` (5e-4, fine-tuning), `accept_score`, `anchor_min`.

## Using the result

- Embed any exported weights into any agent file:
  `uv run python -m training.alphazero.embed_pth --pth runs/main/export/az_agent.pth --agent path/to/agent.py`
- Submit: copy `az_agent.py` to `<MSSV>.py`. It imports only the standard library.
- Benchmark rule (spec): accept only if ≥ 55% vs current `MSSV` over 80 games, no move > 0.55 s,
  tests pass. Record the result in `training/EXPERIMENTS.md`.

## Kaggle (2 GPUs) + Hugging Face

`training/alphazero/kaggle_train.ipynb`: clones the repo, runs the `standard` preset, mirrors the
run to a private HF model repo (`AZ_HF_REPO`, pushed after every phase and p03 iteration; best
checkpoint also at `best/<run>/best.pt`, embedded agent at `best/<run>/az_agent.py`).
`stop_after_hours` makes p03 pause (exit code 3) before an iteration that would cross Kaggle's 12 h
limit; re-running the notebook pulls the run back and continues. `data_parallel=true` wraps
training in `torch.nn.DataParallel` when 2+ CUDA GPUs are visible. Manual sync:

```bash
uv run python -m training.alphazero.hf_sync pull --run std1 --repo henrytang785/az_agent
uv run python -m training.alphazero.hf_sync push --run std1 --repo henrytang785/az_agent
```

## Known risks

- Distillation: on ~190-hot inputs a high learning rate kills ReLUs (lr 3e-3 killed all 32 in a
  test). Keep `lr` at 1e-3 or lower for p04; if Pearson < 0.9 raise `distill_epochs` or
  `distill_positions` first.
- Pure-Python cost measured on an M-series laptop: policy ~2.5 ms per turn, tiny value
  ~0.15 ms per root move (encode included). Colab is ~2x slower.
