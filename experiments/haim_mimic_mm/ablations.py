from pathlib import Path
import os
import sys
import json
import random
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# ============================================================
# Local imports
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from reliability_fusion.models.haim import (
    build_model_a,
    build_model_b,
    build_model_c,
    MODALITY_DIMS,
)


# ============================================================
# Configuration
# ============================================================

ROOT = Path(os.environ.get("HAIM_ROOT", ".")).expanduser().resolve()
PROCESSED_DIR = Path(os.environ.get("HAIM_PROCESSED_DIR", ROOT / "processed")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("HAIM_RESULTS_ROOT", ROOT / "results")).expanduser().resolve()

DATA_DIR = PROCESSED_DIR / "model_data"

DEFAULT_OUT_DIR = (
    RESULTS_ROOT
    / "ablation_abc"
)

OUT_DIR = DEFAULT_OUT_DIR

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEFAULT_SEED = 42
SEED = DEFAULT_SEED

BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 8

LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset
# ============================================================

class HAIMDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        indices,
    ):

        self.X = X
        self.y = y
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        batch = {
            modality:
                torch.from_numpy(
                    self.X[modality][real_idx]
                ).float()

            for modality
            in MODALITY_DIMS
        }

        label = torch.tensor(
            self.y[real_idx],
            dtype=torch.float32,
        )

        return batch, label


# ============================================================
# Normalization
# ============================================================

def fit_standardizer(
    X,
    train_mask,
):

    stats = {}

    for modality, array in X.items():

        train_array = array[
            train_mask
        ]

        mean = train_array.mean(
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

        std = train_array.std(
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

        std[
            std < 1e-6
        ] = 1.0

        stats[modality] = {
            "mean": mean,
            "std": std,
        }

    return stats


def apply_standardizer(
    X,
    stats,
):

    normalized = {}

    for modality, array in X.items():

        normalized[
            modality
        ] = (
            (
                array
                - stats[modality]["mean"]
            )
            / stats[modality]["std"]
        ).astype(
            np.float32
        )

    return normalized


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    y_true,
    probabilities,
):

    return {
        "auroc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "auprc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "brier": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
):

    model.eval()

    all_y = []
    all_prob = []

    for batch, y in loader:

        batch = {
            modality:
                value.to(
                    DEVICE,
                    non_blocking=True,
                )

            for modality, value
            in batch.items()
        }

        logits = model(batch)

        prob = torch.sigmoid(
            logits
        )

        all_y.append(
            y.numpy()
        )

        all_prob.append(
            prob
            .cpu()
            .numpy()
        )

    y_true = np.concatenate(
        all_y
    )

    probability = np.concatenate(
        all_prob
    )

    return compute_metrics(
        y_true,
        probability,
    )


# ============================================================
# Reliability diagnostics
# ============================================================

@torch.no_grad()
def reliability_summary(
    model,
    loader,
):

    model.eval()

    values = {
        modality: []
        for modality
        in MODALITY_DIMS
    }

    for batch, _ in loader:

        batch = {
            modality:
                value.to(
                    DEVICE,
                    non_blocking=True,
                )

            for modality, value
            in batch.items()
        }

        output = model(
            batch,
            return_details=True,
        )

        evidence = output[
            "evidence"
        ]

        for modality in values:

            reliability = (
                evidence[
                    modality
                ][
                    "reliability"
                ]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            values[
                modality
            ].append(
                reliability
            )

    summary = {}

    for modality, parts in values.items():

        x = np.concatenate(
            parts
        )

        summary[
            modality
        ] = {
            "mean": float(
                np.mean(x)
            ),
            "std": float(
                np.std(x)
            ),
            "min": float(
                np.min(x)
            ),
            "max": float(
                np.max(x)
            ),
        }

    return summary


# ============================================================
# Variance diagnostics
# ============================================================

@torch.no_grad()
def fused_variance_summary(
    model,
    loader,
):

    model.eval()

    values = []

    for batch, _ in loader:

        batch = {
            modality:
                value.to(
                    DEVICE,
                    non_blocking=True,
                )

            for modality, value
            in batch.items()
        }

        output = model(
            batch,
            return_details=True,
        )

        variance = (
            output[
                "fused_variance"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # One scalar uncertainty summary per sample.
        sample_variance = (
            variance.mean(
                axis=1
            )
        )

        values.append(
            sample_variance
        )

    values = np.concatenate(
        values
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "std": float(
            np.std(values)
        ),
        "min": float(
            np.min(values)
        ),
        "max": float(
            np.max(values)
        ),
    }


# ============================================================
# Training
# ============================================================

def train_one_model(
    name,
    builder,
    train_loader,
    val_loader,
    pos_weight,
):

    print("\n" + "=" * 80)
    print(name.upper())
    print("=" * 80)

    seed_everything(
        SEED
    )

    model = builder().to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight,
                dtype=torch.float32,
                device=DEVICE,
            )
        )
    )

    best_auc = -np.inf
    best_epoch = None

    epochs_without_improvement = 0

    checkpoint_path = (
        OUT_DIR
        / f"{name}_best.pt"
    )

    history = []

    start_time = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0
        n_seen = 0

        for batch, y in train_loader:

            batch = {
                modality:
                    value.to(
                        DEVICE,
                        non_blocking=True,
                    )

                for modality, value
                in batch.items()
            }

            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch
            )

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_size = (
                y.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            n_seen += (
                batch_size
            )

        train_loss = (
            total_loss
            / n_seen
        )

        val_metrics = evaluate(
            model,
            val_loader,
        )

        history.append({
            "epoch":
                epoch,

            "train_loss":
                float(
                    train_loss
                ),

            **val_metrics,
        })

        print(
            f"epoch={epoch:02d} "
            f"loss={train_loss:.5f} "
            f"val_auc="
            f"{val_metrics['auroc']:.5f} "
            f"val_pr="
            f"{val_metrics['auprc']:.5f} "
            f"val_brier="
            f"{val_metrics['brier']:.5f}"
        )

        if (
            val_metrics["auroc"]
            > best_auc
        ):

            best_auc = (
                val_metrics[
                    "auroc"
                ]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "name":
                        name,

                    "seed":
                        SEED,

                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "val_metrics":
                        val_metrics,
                },
                checkpoint_path,
            )

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "Early stopping."
            )

            break

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Restore best checkpoint
    # --------------------------------------------------------

    saved = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        saved[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Final validation metrics
    # --------------------------------------------------------

    best_val_metrics = evaluate(
        model,
        val_loader,
    )

    variance_summary = (
        fused_variance_summary(
            model,
            val_loader,
        )
    )

    # Reliability is meaningful for C only.
    # We still compute it for A/B as a diagnostic, but those
    # heads were not trained through the task objective.
    reliability = (
        reliability_summary(
            model,
            val_loader,
        )
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        OUT_DIR
        / f"{name}_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    return {
        "name":
            name,

        "seed":
            SEED,

        "best_epoch":
            int(
                best_epoch
            ),

        "val_metrics":
            best_val_metrics,

        "fused_variance":
            variance_summary,

        "reliability":
            reliability,

        "training_seconds":
            float(
                elapsed
            ),

        "checkpoint":
            str(
                checkpoint_path
            ),
    }

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train HAIM A/B/C ablation models."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Random seed. "
            f"Default: {DEFAULT_SEED}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=(
            "Directory for checkpoints, "
            "normalization statistics and summaries."
        ),
    )

    return parser.parse_args()

# ============================================================
# Main
# ============================================================

def main():

    global SEED
    global OUT_DIR

    args = parse_args()

    SEED = int(
        args.seed
    )

    OUT_DIR = Path(
        args.output_dir
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(
        SEED
    )

    print(
        "=" * 80
    )

    print(
        "HAIM A/B/C ABLATION"
    )

    print(
        "=" * 80
    )

    print(
        "Seed:",
        SEED,
    )

    print(
        "Output directory:",
        OUT_DIR,
    )

    seed_everything(
        SEED
    )

    print("=" * 80)
    print("HAIM A-B-C CONTROLLED ABLATION")
    print("=" * 80)

    print(
        "Device:",
        DEVICE,
    )

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    X = {

        modality:
            np.load(
                DATA_DIR
                / f"X_{modality}.npy",
                mmap_mode="r",
            )

        for modality
        in MODALITY_DIMS
    }

    y = np.load(
        DATA_DIR
        / "y.npy"
    )

    split = np.load(
        DATA_DIR
        / "split.npy",
        allow_pickle=True,
    )

    print(
        "\nDataset:",
        len(y),
    )

    for modality, array in X.items():

        print(
            modality,
            array.shape,
        )

    # --------------------------------------------------------
    # Frozen split
    # --------------------------------------------------------

    train_mask = (
        split == "train"
    )

    val_mask = (
        split == "val"
    )

    test_mask = (
        split == "test"
    )

    print(
        "\nTrain:",
        int(
            train_mask.sum()
        ),
    )

    print(
        "Val:",
        int(
            val_mask.sum()
        ),
    )

    print(
        "Test frozen:",
        int(
            test_mask.sum()
        ),
    )

    # --------------------------------------------------------
    # Normalize using training data only
    # --------------------------------------------------------

    print(
        "\nFitting normalization "
        "on TRAIN only..."
    )

    stats = fit_standardizer(
        X,
        train_mask,
    )

    X_norm = apply_standardizer(
        X,
        stats,
    )

    print(
        "Normalization complete."
    )

    np.savez(
        OUT_DIR
        / "normalization_stats.npz",

        **{
            f"{modality}_mean":
                stats[
                    modality
                ]["mean"]

            for modality
            in stats
        },

        **{
            f"{modality}_std":
                stats[
                    modality
                ]["std"]

            for modality
            in stats
        },
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_indices = np.where(
        train_mask
    )[0]

    val_indices = np.where(
        val_mask
    )[0]

    train_dataset = HAIMDataset(
        X_norm,
        y,
        train_indices,
    )

    val_dataset = HAIMDataset(
        X_norm,
        y,
        val_indices,
    )

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    y_train = y[
        train_mask
    ]

    n_positive = int(
        y_train.sum()
    )

    n_negative = int(
        len(y_train)
        - n_positive
    )

    pos_weight = (
        n_negative
        / n_positive
    )

    print(
        "\nTrain positives:",
        n_positive,
    )

    print(
        "Train negatives:",
        n_negative,
    )

    print(
        "pos_weight:",
        pos_weight,
    )

    # --------------------------------------------------------
    # Controlled experiments
    # --------------------------------------------------------

    experiments = [

        (
            "A_mean",
            build_model_a,
        ),

        (
            "B_gaussian",
            build_model_b,
        ),

        (
            "C_reliability",
            build_model_c,
        ),
    ]

    results = []

    for name, builder in experiments:

        # Re-create loaders with the same random seed before
        # each experiment so all three models see the same
        # deterministic shuffle sequence.

        generator = (
            torch.Generator()
        )

        generator.manual_seed(
            SEED
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        result = train_one_model(
            name=name,
            builder=builder,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_weight=pos_weight,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "seed":
            SEED,

        "task":
            "48-hour mortality",

        "selection_metric":
            "validation AUROC",

        "test_evaluated":
            False,

        "training": {
            "batch_size":
                BATCH_SIZE,

            "max_epochs":
                EPOCHS,

            "patience":
                PATIENCE,

            "learning_rate":
                LR,

            "weight_decay":
                WEIGHT_DECAY,

            "loss":
                "BCEWithLogitsLoss",

            "pos_weight":
                float(
                    pos_weight
                ),
        },

        "experiments":
            results,
    }

    summary_path = (
        OUT_DIR
        / "ablation_abc_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "A-B-C VALIDATION SUMMARY"
    )

    print(
        "=" * 80
    )

    for result in results:

        metrics = (
            result[
                "val_metrics"
            ]
        )

        variance = (
            result[
                "fused_variance"
            ]
        )

        print(
            f"{result['name']:20s} "
            f"epoch="
            f"{result['best_epoch']:3d} "
            f"AUROC="
            f"{metrics['auroc']:.5f} "
            f"AUPRC="
            f"{metrics['auprc']:.5f} "
            f"Brier="
            f"{metrics['brier']:.5f} "
            f"Var="
            f"{variance['mean']:.5f}"
        )

    print(
        "\nTest set was NOT evaluated."
    )

    print(
        "Saved:"
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()