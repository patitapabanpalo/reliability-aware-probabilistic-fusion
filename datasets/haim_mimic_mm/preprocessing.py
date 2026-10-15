"""Run the HAIM-MIMIC-MM preprocessing stages used by the experiments.

The pipeline expects the official HAIM-MIMIC-MM release beneath ``--haim-root``.
No source clinical data are distributed with this repository.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

STAGES = (
    "build_mortality_cohort.py",
    "build_master_split.py",
    "build_model_features.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--haim-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    env = os.environ.copy()
    env["HAIM_ROOT"] = str(args.haim_root.expanduser().resolve())
    env["HAIM_OUTPUT_DIR"] = str(args.output_dir.expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    for stage in STAGES:
        print(f"\n[HAIM preprocessing] {stage}")
        subprocess.run([sys.executable, str(here / stage)], env=env, check=True)


if __name__ == "__main__":
    main()
