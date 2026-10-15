"""Validation helper for the processed Symile-MIMIC arrays.

The supplied research archive contains the processed-array loader used by the
experiments, but not the original raw-to-NumPy export pipeline. This module
therefore validates an existing export rather than claiming to recreate one.
"""
import argparse
from pathlib import Path

from .dataset import SymileMIMICDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        dataset = SymileMIMICDataset(args.data_dir, split)
        print(f"{split:5s}: {len(dataset):,} examples")


if __name__ == "__main__":
    main()
