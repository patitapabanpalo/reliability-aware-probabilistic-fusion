"""Evaluate the trained HAIM full (F) model on the frozen test split.

This script intentionally performs no training and does not modify the frozen split.
It reuses the exact normalization and clean-evaluation functions from ``train.py``.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from reliability_fusion.models.haim import HAIMProbabilisticModel, MODALITY_DIMS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--haim-root", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.haim_root.expanduser().resolve()
    processed = (args.processed_dir or (root / "processed")).expanduser().resolve()
    results = (args.results_root or (root / "results")).expanduser().resolve()
    checkpoint_path = (args.checkpoint or (results / "full_f" / "F_full_best.pt")).expanduser().resolve()
    output_path = (args.output or (results / "full_f" / "test_metrics.json")).expanduser().resolve()

    # train.py reads these paths at import time. Set them before importing it.
    os.environ["HAIM_ROOT"] = str(root)
    os.environ["HAIM_PROCESSED_DIR"] = str(processed)
    os.environ["HAIM_RESULTS_ROOT"] = str(results)
    from experiments.haim_mimic_mm import train as train_impl

    data_dir = processed / "model_data"
    X = {m: np.load(data_dir / f"X_{m}.npy", mmap_mode="r") for m in MODALITY_DIMS}
    y = np.load(data_dir / "y.npy")
    split = np.load(data_dir / "split.npy", allow_pickle=True)
    test_mask = split == "test"

    stats = train_impl.load_shared_normalization()
    X_norm = train_impl.apply_standardizer(X, stats)
    test_indices = np.where(test_mask)[0]
    dataset = train_impl.HAIMDataset(X_norm, y, test_indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HAIMProbabilisticModel(fusion_type="reliability").to(train_impl.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=train_impl.DEVICE)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=True)

    metrics, variance, reliability = train_impl.evaluate_clean(model, loader)
    result = {
        "checkpoint": str(checkpoint_path),
        "n_test": int(test_mask.sum()),
        "metrics": metrics,
        "fused_variance": variance,
        "reliability": reliability,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
