#!/usr/bin/env python3
"""Reproduce the robust-TPL benchmark by executing the experiment notebooks in place.

Each notebook trains all models, injects the outlier grid, and (re)writes its
``*_results.xlsx`` workbook and ``plots/`` directory. Runs are seeded inside the
notebooks (SEEDS = [42, 58, 123, 256, 789], SPLIT_SEED = 58); this runner additionally
pins ``PYTHONHASHSEED`` for determinism.

Examples
--------
    python reproduce.py --all
    python reproduce.py --uci
    python reproduce.py --synthetic
    python reproduce.py --only 07_Naval 08_Yacht
    python reproduce.py --list
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UCI_DIR = ROOT / "experiments" / "uci"
SYN_DIR = ROOT / "experiments" / "synthetic"

# The dataset notebooks, in run order. generate_paper_plots is a post-processing
# notebook and is run last (UCI only) when present.
UCI_NOTEBOOKS = [
    "03_Boston.ipynb",
    "04_Concrete.ipynb",
    "05_Energy.ipynb",
    "06_Kin8nm.ipynb",
    "07_Naval.ipynb",
    "08_Yacht.ipynb",
]
UCI_POST = ["generate_paper_plots.ipynb"]
SYN_NOTEBOOKS = [
    "01_Synthetic_Normal.ipynb",
    "02_Synthetic_Hetero.ipynb",
]


def discover(selection: str, post: bool) -> list[Path]:
    nbs: list[Path] = []
    if selection in ("all", "uci"):
        nbs += [UCI_DIR / n for n in UCI_NOTEBOOKS]
        if post:
            nbs += [UCI_DIR / n for n in UCI_POST if (UCI_DIR / n).exists()]
    if selection in ("all", "synthetic"):
        nbs += [SYN_DIR / n for n in SYN_NOTEBOOKS]
    return [p for p in nbs if p.exists()]


def run_notebook(path: Path, timeout: int) -> bool:
    """Execute a notebook in place with nbconvert. Returns True on success."""
    rel = path.relative_to(ROOT)
    print(f"\n=== Running {rel} ===", flush=True)
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"!!! FAILED {rel} (exit {exc.returncode})", file=sys.stderr, flush=True)
        return False
    except FileNotFoundError:
        print(
            "!!! jupyter nbconvert not found. Install deps: pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"--- Done {rel} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_const", dest="sel", const="all", help="run every notebook (default)")
    g.add_argument("--uci", action="store_const", dest="sel", const="uci", help="only the 6 UCI dataset notebooks")
    g.add_argument("--synthetic", action="store_const", dest="sel", const="synthetic", help="only the 2 synthetic notebooks")
    ap.add_argument("--only", nargs="+", metavar="NB", help="run only notebooks whose name contains these tokens")
    ap.add_argument("--no-post", action="store_true", help="skip generate_paper_plots.ipynb")
    ap.add_argument("--list", action="store_true", help="list the notebooks that would run, then exit")
    ap.add_argument("--timeout", type=int, default=7200, help="per-cell execution timeout in seconds (default 7200)")
    args = ap.parse_args()

    selection = args.sel or "all"
    notebooks = discover(selection, post=not args.no_post)
    if args.only:
        toks = [t.lower() for t in args.only]
        notebooks = [p for p in notebooks if any(t in p.name.lower() for t in toks)]

    if not notebooks:
        print("No matching notebooks found.", file=sys.stderr)
        return 1

    if args.list:
        for p in notebooks:
            print(p.relative_to(ROOT))
        return 0

    # Determinism: stable hashing for the whole pipeline.
    os.environ.setdefault("PYTHONHASHSEED", "0")

    print(f"Reproducing {len(notebooks)} notebook(s) [selection={selection}]")
    t0 = time.time()
    failures = [p.name for p in notebooks if not run_notebook(p, args.timeout)]
    dt = (time.time() - t0) / 60
    print(f"\n==== Finished {len(notebooks) - len(failures)}/{len(notebooks)} in {dt:.1f} min ====")
    if failures:
        print("Failed:", ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
