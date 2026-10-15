"""Run the final Symile trainer independently for the manuscript seeds 0--4.

This is intentionally a thin orchestration layer: it does not reimplement the
model or training loop. Each seed gets its own output directory.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

DEFAULT_SEEDS = [0, 1, 2, 3, 4]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    for seed in args.seeds:
        out = args.output_root.expanduser().resolve() / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["SYMILE_DATA_ROOT"] = str(args.data_root.expanduser().resolve())
        env["SYMILE_OUTPUT_ROOT"] = str(out)
        env["SYMILE_SEED"] = str(seed)
        cmd = [sys.executable, "-m", "experiments.symile_mimic.train"]
        print(f"\n=== Symile seed {seed} ===\n{' '.join(cmd)}")
        subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__":
    main()
