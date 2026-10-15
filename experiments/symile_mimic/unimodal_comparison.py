from pathlib import Path
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# =====================================================================
# Paths
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_BASE = Path(os.environ.get("SYMILE_DATA_ROOT", ROOT)).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("SYMILE_OUTPUT_ROOT", ROOT / "experiment_results")).expanduser().resolve()

DATA_ROOT = DATA_BASE / "data_npy"

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

TRAIN_CSV = DATA_BASE / "train.csv"
VAL_CSV = DATA_BASE / "val.csv"

RESULTS_DIR = OUTPUT_ROOT
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = (
    RESULTS_DIR
    / "unimodal_comparison_results.json"
)

from datasets.symile_mimic.dataset import SymileMIMICDataset
from reliability_fusion.encoders.symile import LabEncoder


# =====================================================================
# Configuration
# =====================================================================

LABEL_NAME = "Cardiomegaly"

SEEDS = [0, 1, 2, 3, 4]

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3

EMBED_DIM = 128

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Dataset
# =====================================================================

class UnimodalDataset(Dataset):

    def __init__(
        self,
        tensor_dataset,
        csv_path,
        modality,
    ):

        self.tensor_dataset = tensor_dataset
        self.modality = modality

        df = pd.read_csv(csv_path)

        csv_hadm = df[
            "hadm_id"
        ].to_numpy(
            dtype=np.int64
        )

        tensor_hadm = np.asarray(
            tensor_dataset.hadm_id
        )

        if not np.array_equal(
            csv_hadm,
            tensor_hadm,
        ):

            raise ValueError(
                f"{modality}: CSV/tensor hadm_id "
                "alignment failed."
            )

        labels = df[
            LABEL_NAME
        ].to_numpy(
            dtype=np.float32
        )

        usable = np.isin(
            labels,
            [0.0, 1.0],
        )

        self.indices = np.flatnonzero(
            usable
        )

        self.labels = torch.tensor(
            labels[self.indices],
            dtype=torch.float32,
        )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        index,
    ):

        source_index = int(
            self.indices[index]
        )

        sample = self.tensor_dataset[
            source_index
        ]

        if self.modality == "CXR":

            x = sample["cxr"]

        elif self.modality == "ECG":

            x = sample["ecg"]

        elif self.modality == "Labs":

            labs = torch.cat(
                [
                    sample[
                        "labs_percentiles"
                    ].float(),

                    sample[
                        "labs_missingness"
                    ].float(),
                ],
                dim=0,
            )

            x = labs

        else:

            raise ValueError(
                f"Unknown modality: {self.modality}"
            )

        return {
            "x": x.float(),
            "label": self.labels[index],
        }


# =====================================================================
# CXR encoder
# =====================================================================

class CXREncoder(nn.Module):

    def __init__(
        self,
        embedding_dim=128,
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=7,
                stride=2,
                padding=3,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        self.projection = nn.Linear(
            128,
            embedding_dim,
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        return self.projection(x)


# =====================================================================
# ECG encoder
# =====================================================================

class ECGEncoder(nn.Module):

    def __init__(
        self,
        embedding_dim=128,
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                12,
                32,
                kernel_size=15,
                stride=2,
                padding=7,
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(4),

            nn.Conv1d(
                32,
                64,
                kernel_size=9,
                stride=2,
                padding=4,
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(4),

            nn.Conv1d(
                64,
                128,
                kernel_size=7,
                stride=2,
                padding=3,
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.projection = nn.Linear(
            128,
            embedding_dim,
        )

    def forward(self, x):

        # Input:
        # (B, 1, 5000, 12)

        x = x.squeeze(1)

        # (B, 5000, 12)

        x = x.transpose(
            1,
            2,
        )

        # (B, 12, 5000)

        x = self.features(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        return self.projection(x)


# =====================================================================
# Lab encoder
# =====================================================================

class LabEncoder(nn.Module):

    def __init__(
        self,
        input_dim=100,
        embedding_dim=128,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                embedding_dim,
            ),
        )

    def forward(self, x):

        return self.network(x)


# =====================================================================
# Prediction head
# =====================================================================

class PredictionHead(nn.Module):

    def __init__(
        self,
        input_dim=128,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, x):

        return self.network(
            x
        ).squeeze(-1)


# =====================================================================
# Build modality model
# =====================================================================

def build_model(
    modality,
):

    if modality == "CXR":

        encoder = CXREncoder(
            EMBED_DIM
        )

    elif modality == "ECG":

        encoder = ECGEncoder(
            EMBED_DIM
        )

    elif modality == "Labs":

        encoder = LabEncoder(
            input_dim=100,
            embedding_dim=EMBED_DIM,
        )

    else:

        raise ValueError(
            modality
        )

    prediction_head = PredictionHead(
        EMBED_DIM
    )

    model = nn.ModuleDict(
        {
            "encoder": encoder,
            "prediction": prediction_head,
        }
    )

    return model.to(
        DEVICE
    )


# =====================================================================
# Training
# =====================================================================

def train_one_seed(
    model,
    loader,
    pos_weight,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    model.train()

    for epoch in range(
        EPOCHS
    ):

        epoch_loss = 0.0

        total_examples = 0

        for batch in loader:

            x = batch[
                "x"
            ].to(DEVICE)

            y = batch[
                "label"
            ].to(DEVICE)

            optimizer.zero_grad()

            embedding = model[
                "encoder"
            ](x)

            logits = model[
                "prediction"
            ](
                embedding
            )

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            batch_size = y.shape[0]

            epoch_loss += (
                loss.item()
                * batch_size
            )

            total_examples += (
                batch_size
            )

        mean_loss = (
            epoch_loss
            / total_examples
        )

        print(
            f"    Epoch {epoch + 1}: "
            f"loss={mean_loss:.6f}"
        )


# =====================================================================
# Evaluation
# =====================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
):

    model.eval()

    labels = []

    probabilities = []

    for batch in loader:

        x = batch[
            "x"
        ].to(DEVICE)

        y = batch[
            "label"
        ].cpu().numpy()

        embedding = model[
            "encoder"
        ](x)

        logits = model[
            "prediction"
        ](
            embedding
        )

        probs = torch.sigmoid(
            logits
        )

        labels.extend(
            y.tolist()
        )

        probabilities.extend(
            probs.cpu()
            .numpy()
            .tolist()
        )

    y = np.asarray(
        labels,
        dtype=np.float64,
    )

    p = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    return {
        "AUROC": float(
            roc_auc_score(
                y,
                p,
            )
        ),

        "AUPRC": float(
            average_precision_score(
                y,
                p,
            )
        ),

        "Brier": float(
            np.mean(
                (p - y) ** 2
            )
        ),

        "n": int(
            len(y)
        ),

        "positive": int(
            np.sum(y == 1)
        ),

        "negative": int(
            np.sum(y == 0)
        ),
    }


# =====================================================================
# Aggregate statistics
# =====================================================================

def aggregate(
    seed_results,
):

    output = {}

    for metric in [
        "AUROC",
        "AUPRC",
        "Brier",
    ]:

        values = np.asarray(
            [
                result[metric]
                for result in seed_results.values()
            ],
            dtype=np.float64,
        )

        output[
            metric
        ] = {
            "mean": float(
                values.mean()
            ),

            "std": float(
                values.std(
                    ddof=1
                )
            ),

            "values": [
                float(x)
                for x in values
            ],
        }

    # Same dataset across seeds.
    first_result = next(
        iter(
            seed_results.values()
        )
    )

    output[
        "n"
    ] = first_result["n"]

    output[
        "positive"
    ] = first_result["positive"]

    output[
        "negative"
    ] = first_result["negative"]

    return output


# =====================================================================
# Main
# =====================================================================

def main():

    print(
        "=" * 90
    )

    print(
        "UNIMODAL COMPARISON EXPERIMENT"
    )

    print(
        "=" * 90
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            "GPU: "
            + torch.cuda.get_device_name(0)
        )

    print(
        "\nSeeds:"
    )

    print(
        f"  {SEEDS}"
    )

    modalities = [
        "CXR",
        "ECG",
        "Labs",
    ]

    # ---------------------------------------------------------------
    # Load tensor datasets.
    # ---------------------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    from datasets.symile_mimic.dataset import SymileMIMICDataset

    train_tensor = (
        SymileMIMICDataset(
            DATA_ROOT,
            "train",
        )
    )

    val_tensor = (
        SymileMIMICDataset(
            DATA_ROOT,
            "val",
        )
    )

    # ---------------------------------------------------------------
    # Results.
    # ---------------------------------------------------------------

    all_results = {}

    for modality in modalities:

        print(
            "\n" + "=" * 90
        )

        print(
            f"MODALITY: {modality}"
        )

        print(
            "=" * 90
        )

        modality_results = {}

        # -----------------------------------------------------------
        # Dataset.
        # -----------------------------------------------------------

        train_dataset = (
            UnimodalDataset(
                train_tensor,
                TRAIN_CSV,
                modality,
            )
        )

        val_dataset = (
            UnimodalDataset(
                val_tensor,
                VAL_CSV,
                modality,
            )
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        positive_count = int(
            train_dataset.labels.sum().item()
        )

        negative_count = (
            len(train_dataset)
            - positive_count
        )

        pos_weight = torch.tensor(
            negative_count
            / positive_count,
            dtype=torch.float32,
            device=DEVICE,
        )

        print(
            "\nTraining examples: "
            f"{len(train_dataset):,}"
        )

        print(
            "Validation examples: "
            f"{len(val_dataset):,}"
        )

        print(
            "Positive: "
            f"{positive_count:,}"
        )

        print(
            "Negative: "
            f"{negative_count:,}"
        )

        print(
            "Positive weight: "
            f"{pos_weight.item():.6f}"
        )

        # -----------------------------------------------------------
        # Seeds.
        # -----------------------------------------------------------

        for seed in SEEDS:

            print(
                "\n" + "-" * 70
            )

            print(
                f"{modality} | SEED {seed}"
            )

            print(
                "-" * 70
            )

            set_seed(
                seed
            )

            model = build_model(
                modality
            )

            parameter_count = sum(
                parameter.numel()
                for parameter
                in model.parameters()
            )

            print(
                f"Parameters: "
                f"{parameter_count:,}"
            )

            train_one_seed(
                model,
                train_loader,
                pos_weight,
            )

            metrics = evaluate(
                model,
                val_loader,
            )

            print(
                "\nValidation:"
            )

            print(
                f"  AUROC: "
                f"{metrics['AUROC']:.4f}"
            )

            print(
                f"  AUPRC: "
                f"{metrics['AUPRC']:.4f}"
            )

            print(
                f"  Brier: "
                f"{metrics['Brier']:.4f}"
            )

            modality_results[
                str(seed)
            ] = metrics

    # ---------------------------------------------------------------
        all_results[
            modality
        ] = modality_results

    # ---------------------------------------------------------------
    # Aggregate.
    # ---------------------------------------------------------------

    summaries = {}

    for modality in modalities:

        summaries[
            modality
        ] = aggregate(
            all_results[
                modality
            ]
        )

    # ---------------------------------------------------------------
    # Main table.
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "UNIMODAL MAIN-TABLE RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Method':<18}"
        f"{'AUROC':>20}"
        f"{'AUPRC':>20}"
        f"{'Brier':>20}"
    )

    print(
        "-" * 80
    )

    for modality in modalities:

        summary = summaries[
            modality
        ]

        print(
            f"{modality + '-only':<18}"
            f"{summary['AUROC']['mean']:.4f} "
            f"+/- {summary['AUROC']['std']:.4f}"
            f"{summary['AUPRC']['mean']:>10.4f} "
            f"+/- {summary['AUPRC']['std']:.4f}"
            f"{summary['Brier']['mean']:>10.4f} "
            f"+/- {summary['Brier']['std']:.4f}"
        )

    # ---------------------------------------------------------------
    # Save results.
    # ---------------------------------------------------------------

    output = {
        "experiment": (
            "Five-seed unimodal comparison"
        ),

        "target": LABEL_NAME,

        "seeds": SEEDS,

        "epochs": EPOCHS,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "device": str(
            DEVICE
        ),

        "modalities": modalities,

        "per_seed": all_results,

        "summary": summaries,

        "table_b_rows": [
            {
                "method": f"{modality}-only",
                "type": "Unimodal",
                "AUROC_mean":
                    summaries[
                        modality
                    ]["AUROC"]["mean"],
                "AUROC_std":
                    summaries[
                        modality
                    ]["AUROC"]["std"],
                "AUPRC_mean":
                    summaries[
                        modality
                    ]["AUPRC"]["mean"],
                "AUPRC_std":
                    summaries[
                        modality
                    ]["AUPRC"]["std"],
                "Brier_mean":
                    summaries[
                        modality
                    ]["Brier"]["mean"],
                "Brier_std":
                    summaries[
                        modality
                    ]["Brier"]["std"],
            }
            for modality in modalities
        ],
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        "\nSaved results:"
        f"\n  {OUTPUT_FILE}"
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "UNIMODAL COMPARISON COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()