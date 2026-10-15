"""Run final HAIM F training independently for manuscript seeds 0--4."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    root = args.output_root.expanduser().resolve()
    for seed in args.seeds:
        out = root / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "experiments.haim_mimic_mm.train",
            "--seed", str(seed), "--output-dir", str(out),
        ]
        if args.max_epochs is not None:
            cmd += ["--max-epochs", str(args.max_epochs)]
        print(f"\n=== HAIM seed {seed} ===\n{' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
