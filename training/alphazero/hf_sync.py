"""Mirror a run folder to a Hugging Face model repo, so training can stop and resume elsewhere
(Kaggle sessions end after 12 h; the next session pulls the run and continues).

Repo layout:
  runs/<run>/...          the whole run folder (manifest, data shards, checkpoints, tb, export)
  best/<run>/best.pt      copy of the current best checkpoint (manifest "best")
  best/<run>/manifest.json
  best/<run>/az_agent.py  the agent with embedded weights (after p06)

Automatic push: set AZ_HF_REPO=<user/repo>; the pipeline then pushes after every phase and
after every p03 iteration. Needs `huggingface_hub` and a login (HF_TOKEN or huggingface_hub.login).

  python -m training.alphazero.hf_sync pull --run std1 --repo user/az_agent
  python -m training.alphazero.hf_sync push --run std1 --repo user/az_agent
  python -m training.alphazero.hf_sync pull --run sp1 --repo user/az_agent --marker selfplay.json

selfplay_train.py runs (selfplay.json instead of manifest.json) push after start and after every
generation the same way; best/<run>/ then holds best.pt, selfplay.json and az_agent_sp.py."""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

ENV_REPO = "AZ_HF_REPO"


def _api(repo):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    return api


def push(ctx, repo=None, message=None):
    """Upload the run folder + best checkpoint. Unchanged large files are not re-uploaded."""
    repo = repo or os.environ.get(ENV_REPO)
    if not repo:
        return False
    api = _api(repo)
    message = message or "%s: iteration %d, best %s" % (ctx.name, ctx.manifest["iteration"], ctx.manifest["best"])
    api.upload_folder(repo_id=repo, folder_path=str(ctx.dir), path_in_repo="runs/" + ctx.name,
                      commit_message=message, ignore_patterns=["*.tmp"])
    best = "best/" + ctx.name + "/"
    if ctx.manifest.get("best"):
        api.upload_file(repo_id=repo, path_or_fileobj=str(ctx.path(ctx.manifest["best"])),
                        path_in_repo=best + "best.pt", commit_message=message)
    api.upload_file(repo_id=repo, path_or_fileobj=str(ctx.manifest_path),
                    path_in_repo=best + "manifest.json", commit_message=message)
    if ctx.is_done("p06"):
        api.upload_file(repo_id=repo, path_or_fileobj=str(ctx.agent_path),
                        path_in_repo=best + "az_agent.py", commit_message=message)
    print("  pushed %s to https://huggingface.co/%s" % (ctx.name, repo), flush=True)
    return True


def push_selfplay(run, repo=None, message=None):
    """Upload a selfplay_train.py run: the folder + best/<run>/{best.pt, selfplay.json, az_agent_sp.py}."""
    repo = repo or os.environ.get(ENV_REPO)
    if not repo:
        return False
    api = _api(repo)
    message = message or "%s: generation %d, best %s" % (run.name, run.state["gen"], run.state["best"])
    api.upload_folder(repo_id=repo, folder_path=str(run.dir), path_in_repo="runs/" + run.name,
                      commit_message=message, ignore_patterns=["*.tmp"])
    best = "best/" + run.name + "/"
    files = [(run.state_path, "selfplay.json"), (run.path("export", "az_agent_sp.py"), "az_agent_sp.py")]
    if run.state.get("best"):
        files.append((run.path(run.state["best"]), "best.pt"))
    for local, name in files:
        if Path(local).exists():
            api.upload_file(repo_id=repo, path_or_fileobj=str(local), path_in_repo=best + name,
                            commit_message=message)
    print("  pushed %s to https://huggingface.co/%s" % (run.name, repo), flush=True)
    return True


def safe_push_selfplay(run, message=None):
    """push_selfplay() that never stops training on a network error."""
    try:
        return push_selfplay(run, message=message)
    except Exception as exc:   # noqa: BLE001 - report and keep training
        print("  WARNING: Hugging Face push failed (%s: %s)" % (type(exc).__name__, exc), flush=True)
        return False


def safe_push(ctx, message=None):
    """push() for the automatic hooks: a network error must never stop a long training run."""
    try:
        return push(ctx, message=message)
    except Exception as exc:   # noqa: BLE001 - report and keep training
        print("  WARNING: Hugging Face push failed (%s: %s)" % (type(exc).__name__, exc), flush=True)
        return False


def pull(name, repo, runs_dir, marker="manifest.json"):
    """Download runs/<name>/ from the repo into runs_dir/<name>. Returns False if the repo has none.

    marker: file that proves the run exists (selfplay_train.py runs use selfplay.json)."""
    from huggingface_hub import snapshot_download
    _api(repo)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(repo_id=repo, repo_type="model", local_dir=tmp, allow_patterns=["runs/%s/*" % name])
        source = Path(tmp) / "runs" / name
        if not (source / marker).exists():
            print("  %s has no runs/%s yet - starting fresh" % (repo, name))
            return False
        target = Path(runs_dir) / name
        shutil.copytree(source, target, dirs_exist_ok=True)
    print("  pulled runs/%s from %s into %s" % (name, repo, target))
    return True


def main():
    from .run_state import RUNS_DIR, RunContext
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("pull", "push"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", default=os.environ.get(ENV_REPO))
    ap.add_argument("--marker", default="manifest.json",
                    help="pull: file proving the run exists (selfplay.json for selfplay_train.py runs)")
    args = ap.parse_args()
    if not args.repo:
        raise SystemExit("give --repo user/name or set " + ENV_REPO)
    if args.action == "pull":
        pull(args.run, args.repo, RUNS_DIR, args.marker)
    elif (RUNS_DIR / args.run / "selfplay.json").exists():
        from .selfplay_train import SPRun
        push_selfplay(SPRun(args.run), args.repo)
    else:
        if not (RUNS_DIR / args.run / "manifest.json").exists():
            raise SystemExit("no run %s in %s" % (args.run, RUNS_DIR))
        push(RunContext(args.run), args.repo)


if __name__ == "__main__":
    main()
