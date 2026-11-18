"""Embed any exported .pth into any agent file (outside the phase pipeline).

  uv run python -m training.alphazero.embed_pth --pth runs/main/export/az_agent.pth \
      --agent wrongway-student/wrongway/submissions/az_agent.py"""
import argparse

from .export import embed_pth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--dtype", default="e", choices=("e", "f"))
    a = ap.parse_args()
    size = embed_pth(a.pth, a.agent, a.dtype)
    print("embedded %d blob chars into %s" % (size, a.agent))


if __name__ == "__main__":
    main()
