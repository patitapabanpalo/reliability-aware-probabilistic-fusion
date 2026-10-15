"""Dataset utilities for the processed HAIM-MIMIC-MM mortality cohort."""
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITY_DIMS = {
    "structured": 1542,
    "timeseries": 451,
    "cxr": 2084,
    "text": 2304,
}


def load_processed_arrays(data_dir: Path, mmap_mode: str | None = "r"):
    """Load feature matrices, labels, and the frozen patient-level split."""
    data_dir = Path(data_dir)
    features = {
        name: np.load(data_dir / f"X_{name}.npy", mmap_mode=mmap_mode)
        for name in MODALITY_DIMS
    }
    labels = np.load(data_dir / "y.npy", mmap_mode=mmap_mode)
    split = np.load(data_dir / "split.npy", allow_pickle=True)

    n = len(labels)
    for name, dim in MODALITY_DIMS.items():
        if features[name].shape != (n, dim):
            raise ValueError(
                f"Unexpected {name} shape {features[name].shape}; expected {(n, dim)}."
            )
    if len(split) != n:
        raise ValueError("split.npy and y.npy have different lengths.")
    return features, labels, split


class HAIMDataset(Dataset):
    """Thin PyTorch dataset over precomputed HAIM modality representations."""

    def __init__(
        self,
        features: Mapping[str, np.ndarray],
        labels: np.ndarray,
        indices: Sequence[int],
    ):
        self.features = features
        self.labels = labels
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        modalities = {
            name: torch.from_numpy(np.asarray(values[row]).copy()).float()
            for name, values in self.features.items()
        }
        label = torch.tensor(float(self.labels[row]), dtype=torch.float32)
        return modalities, label
