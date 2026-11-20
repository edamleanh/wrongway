"""Rules + feature encoding for training = the agent file itself (single source of truth).

Training never re-implements the game: it imports submissions/az_agent.py, so encode(),
legal_actions() and play() are byte-for-byte what the submitted agent runs."""
import importlib.util
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ARENA_DIR = PROJECT / "wrongway-student" / "wrongway"
SUBMISSIONS = ARENA_DIR / "submissions"
AGENT_NAME = "az_agent"
AGENT_FILE = SUBMISSIONS / (AGENT_NAME + ".py")


def load_agent(path=AGENT_FILE, name="az_engine"):
    """Import an agent .py file as a module (cached in sys.modules under `name`)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = load_agent()
