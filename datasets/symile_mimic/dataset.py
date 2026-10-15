"""PyTorch dataset for the processed Symile-MIMIC arrays used by the experiments."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SymileMIMICDataset(Dataset):
    """Load CXR, ECG, laboratory, and admission-ID arrays for one split."""

    def __init__(self, data_dir: Path, split: str):
        self.data_dir = Path(data_dir)
        self.split = split
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)

        self.cxr = np.load(split_dir / f"cxr_{split}.npy", mmap_mode="r")
        self.ecg = np.load(split_dir / f"ecg_{split}.npy", mmap_mode="r")
        self.labs_percentiles = np.load(
            split_dir / f"labs_percentiles_{split}.npy", mmap_mode="r"
        )
        self.labs_missingness = np.load(
            split_dir / f"labs_missingness_{split}.npy", mmap_mode="r"
        )
        self.hadm_id = np.load(split_dir / f"hadm_id_{split}.npy", mmap_mode="r")

        n = len(self.hadm_id)
        expected = {
            "cxr": (n, 3, 320, 320),
            "ecg": (n, 1, 5000, 12),
            "labs_percentiles": (n, 50),
            "labs_missingness": (n, 50),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"Unexpected {name} shape {actual}; expected {shape}.")

    def __len__(self):
        return len(self.hadm_id)

    @staticmethod
    def _tensor(value):
        return torch.from_numpy(np.asarray(value).copy())

    def __getitem__(self, index):
        return {
            "cxr": self._tensor(self.cxr[index]).float(),
            "ecg": self._tensor(self.ecg[index]).float(),
            "labs_percentiles": self._tensor(self.labs_percentiles[index]).float(),
            "labs_missingness": self._tensor(self.labs_missingness[index]).float(),
            "hadm_id": torch.tensor(int(self.hadm_id[index]), dtype=torch.long),
        }
