"""Evaluate a trained Symile-MIMIC model on the held-out test split."""
import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing data_npy/ and split CSV files.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    data_root = args.data_root.expanduser().resolve()
    output_dir = (args.output_dir or (repo_root / "experiment_results")).expanduser().resolve()
    checkpoint_path = (args.checkpoint or (output_dir / "missing_modality_aware_model.pt")).expanduser().resolve()

    os.environ["SYMILE_DATA_ROOT"] = str(data_root)
    os.environ["SYMILE_OUTPUT_ROOT"] = str(output_dir)
    from experiments.symile_mimic import train as train_impl
    from datasets.symile_mimic.dataset import SymileMIMICDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_dataset = SymileMIMICDataset(data_root / "data_npy", "test")
    dataset = train_impl.LabeledCardiomegalyDataset(tensor_dataset, data_root / "test.csv")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    modules = [
        train_impl.CXREncoder(train_impl.EMBED_DIM).to(device),
        train_impl.ECGEncoder(train_impl.EMBED_DIM).to(device),
        train_impl.LabEncoder(input_dim=100, embedding_dim=train_impl.EMBED_DIM).to(device),
        train_impl.GaussianEvidence(train_impl.EMBED_DIM, train_impl.LATENT_DIM).to(device),
        train_impl.GaussianEvidence(train_impl.EMBED_DIM, train_impl.LATENT_DIM).to(device),
        train_impl.GaussianEvidence(train_impl.EMBED_DIM, train_impl.LATENT_DIM).to(device),
        train_impl.ReliabilityHead(train_impl.LATENT_DIM).to(device),
        train_impl.ReliabilityHead(train_impl.LATENT_DIM).to(device),
        train_impl.ReliabilityHead(train_impl.LATENT_DIM).to(device),
        train_impl.PredictionHead(train_impl.LATENT_DIM).to(device),
    ]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    states = checkpoint["model_state_dicts"]
    for i, module in enumerate(modules):
        module.load_state_dict(states[f"module_{i}"], strict=True)

    result = train_impl.evaluate_condition(modules, loader, device, missing_modalities=set())
    result["checkpoint"] = str(checkpoint_path)
    result["n_test"] = len(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "test_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
