"""
Train HAIM ablation D:

    D = Gaussian PoE + missing-modality-aware training

This script deliberately does NOT use reliability-weighted fusion.

It is intended to be compared directly against:

    B = Gaussian PoE trained with all modalities

Missingness schedule per TRAINING batch:

    50% : all 4 modalities
    20% : 3 available / 1 missing
    20% : 2 available / 2 missing
    10% : 1 available / 3 missing

Within each missingness level, the missing modalities are sampled
uniformly at random.

Validation checkpoint selection is performed using CLEAN validation
data with all four modalities available, matching the A/B/C protocol.

The held-out TEST set is never evaluated here.
"""

from pathlib import Path
import os
import sys
import json
import random
import time
from collections import Counter
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# ============================================================
# Local imports
# ============================================================

THIS_DIR = Path(
    __file__
).resolve().parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(THIS_DIR),
    )

from reliability_fusion.models.haim import (
    HAIMProbabilisticModel,
    MODALITY_DIMS,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(os.environ.get("HAIM_ROOT", ".")).expanduser().resolve()
PROCESSED_DIR = Path(os.environ.get("HAIM_PROCESSED_DIR", ROOT / "processed")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("HAIM_RESULTS_ROOT", ROOT / "results")).expanduser().resolve()

DATA_DIR = PROCESSED_DIR / "model_data"

ABC_DIR = (
    RESULTS_ROOT
    / "ablation_abc"
)

DEFAULT_OUT_DIR = (
    RESULTS_ROOT
    / "missing_aware_d"
)

OUT_DIR = DEFAULT_OUT_DIR

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_SEED = 42
SEED = DEFAULT_SEED

BATCH_SIZE = 256

EPOCHS = 50

PATIENCE = 8

LR = 1e-3

WEIGHT_DECAY = 1e-4

GRAD_CLIP = 5.0


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODALITIES = tuple(
    MODALITY_DIMS.keys()
)


# ------------------------------------------------------------
# Missingness probabilities
# ------------------------------------------------------------

P_ALL = 0.50

P_ONE_MISSING = 0.20

P_TWO_MISSING = 0.20

P_THREE_MISSING = 0.10


assert abs(
    (
        P_ALL
        + P_ONE_MISSING
        + P_TWO_MISSING
        + P_THREE_MISSING
    )
    - 1.0
) < 1e-8


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

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

        return len(
            self.indices
        )

    def __getitem__(
        self,
        idx,
    ):

        real_idx = (
            self.indices[idx]
        )

        batch = {

            modality:
                torch.from_numpy(
                    self.X[
                        modality
                    ][
                        real_idx
                    ]
                ).float()

            for modality
            in MODALITIES
        }

        label = torch.tensor(
            self.y[
                real_idx
            ],
            dtype=torch.float32,
        )

        return (
            batch,
            label,
        )


# ============================================================
# Normalization
# ============================================================

def load_abc_normalization():
    """
    Load the SAME train-derived normalization statistics
    used in the A/B/C experiment.

    This prevents normalization from becoming another
    experimental difference between B and D.
    """

    path = (
        ABC_DIR
        / "normalization_stats.npz"
    )

    if not path.exists():

        raise FileNotFoundError(
            "Could not find A/B/C normalization "
            f"statistics:\n{path}"
        )

    data = np.load(
        path
    )

    stats = {}

    for modality in MODALITIES:

        stats[
            modality
        ] = {

            "mean":
                data[
                    f"{modality}_mean"
                ].astype(
                    np.float32
                ),

            "std":
                data[
                    f"{modality}_std"
                ].astype(
                    np.float32
                ),
        }

    return stats


def apply_standardizer(
    X,
    stats,
):

    normalized = {}

    for modality in MODALITIES:

        normalized[
            modality
        ] = (
            (
                X[
                    modality
                ]
                - stats[
                    modality
                ][
                    "mean"
                ]
            )
            / stats[
                modality
            ][
                "std"
            ]
        ).astype(
            np.float32
        )

    return normalized


# ============================================================
# Missingness sampling
# ============================================================

def sample_missing_pattern():
    """
    Sample ONE missingness pattern for an entire training batch.

    Distribution:

        50% -> no modality missing
        20% -> one missing
        20% -> two missing
        10% -> three missing

    At least one modality always remains available.
    """

    p = random.random()

    if p < P_ALL:

        n_missing = 0

    elif p < (
        P_ALL
        + P_ONE_MISSING
    ):

        n_missing = 1

    elif p < (
        P_ALL
        + P_ONE_MISSING
        + P_TWO_MISSING
    ):

        n_missing = 2

    else:

        n_missing = 3

    if n_missing == 0:

        return set()

    missing = random.sample(
        list(
            MODALITIES
        ),
        n_missing,
    )

    return set(
        missing
    )


def canonical_pattern_name(
    missing,
):

    missing = set(
        missing
    )

    available = [

        modality

        for modality
        in MODALITIES

        if modality
        not in missing
    ]

    if len(missing) == 0:

        return "all"

    return (
        "available:"
        + "+".join(
            available
        )
    )


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    y_true,
    probabilities,
):

    return {

        "auroc":
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),

        "auprc":
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            ),

        "brier":
            float(
                brier_score_loss(
                    y_true,
                    probabilities,
                )
            ),
    }


# ============================================================
# Clean validation evaluation
# ============================================================

@torch.no_grad()
def evaluate_clean(
    model,
    loader,
):

    """
    Evaluate using ALL four modalities.

    This is used for checkpoint selection so D remains
    directly comparable to A/B/C.
    """

    model.eval()

    labels = []

    probabilities = []

    variances = []

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

        output = model(
            batch,
            missing=set(),
            return_details=True,
        )

        logits = output[
            "logits"
        ]

        probability = torch.sigmoid(
            logits
        )

        labels.append(
            y.numpy()
        )

        probabilities.append(
            probability
            .detach()
            .cpu()
            .numpy()
        )

        variance = (
            output[
                "fused_variance"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        sample_variance = (
            variance.mean(
                axis=1
            )
        )

        variances.append(
            sample_variance
        )

    y_true = np.concatenate(
        labels
    )

    prob = np.concatenate(
        probabilities
    )

    variance = np.concatenate(
        variances
    )

    metrics = compute_metrics(
        y_true,
        prob,
    )

    variance_summary = {

        "mean":
            float(
                np.mean(
                    variance
                )
            ),

        "std":
            float(
                np.std(
                    variance
                )
            ),

        "min":
            float(
                np.min(
                    variance
                )
            ),

        "max":
            float(
                np.max(
                    variance
                )
            ),
    }

    return (
        metrics,
        variance_summary,
    )


# ============================================================
# Optional validation missingness diagnostic
# ============================================================

@torch.no_grad()
def evaluate_missing_pattern(
    model,
    loader,
    missing,
):

    """
    Diagnostic only.

    This does NOT influence checkpoint selection.
    """

    model.eval()

    labels = []

    probabilities = []

    variances = []

    missing = set(
        missing
    )

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

        output = model(
            batch,
            missing=missing,
            return_details=True,
        )

        logits = output[
            "logits"
        ]

        probability = torch.sigmoid(
            logits
        )

        labels.append(
            y.numpy()
        )

        probabilities.append(
            probability
            .cpu()
            .numpy()
        )

        variance = (
            output[
                "fused_variance"
            ]
            .cpu()
            .numpy()
        )

        variances.append(
            variance.mean(
                axis=1
            )
        )

    y_true = np.concatenate(
        labels
    )

    probability = np.concatenate(
        probabilities
    )

    variance = np.concatenate(
        variances
    )

    metrics = compute_metrics(
        y_true,
        probability,
    )

    return {

        **metrics,

        "fused_variance_mean":
            float(
                np.mean(
                    variance
                )
            ),
    }


# ============================================================
# Model
# ============================================================

def build_model_d():
    """
    D uses STANDARD Gaussian PoE.

    Reliability heads may exist inside the shared architecture,
    but reliability is NOT used in fusion.
    """

    return HAIMProbabilisticModel(
        fusion_type="gaussian"
    )


# ============================================================
# Training
# ============================================================

def train_model(
    train_loader,
    val_loader,
    pos_weight,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "D: GAUSSIAN PoE + "
        "MISSING-AWARE TRAINING"
    )

    print(
        "=" * 80
    )

    seed_everything(
        SEED
    )

    model = (
        build_model_d()
        .to(
            DEVICE
        )
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
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

    checkpoint_path = (
        OUT_DIR
        / "D_missing_aware_best.pt"
    )

    history_path = (
        OUT_DIR
        / "D_missing_aware_history.json"
    )

    best_auc = -np.inf

    best_epoch = None

    epochs_without_improvement = 0

    history = []

    global_pattern_counter = Counter()

    start_time = time.time()

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        n_seen = 0

        epoch_pattern_counter = (
            Counter()
        )

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

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

            # -----------------------------------------------
            # Sample missingness for THIS batch
            # -----------------------------------------------

            missing = (
                sample_missing_pattern()
            )

            pattern = (
                canonical_pattern_name(
                    missing
                )
            )

            epoch_pattern_counter[
                pattern
            ] += 1

            global_pattern_counter[
                pattern
            ] += 1

            # -----------------------------------------------
            # Forward
            # -----------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch,
                missing=missing,
            )

            loss = criterion(
                logits,
                y,
            )

            # -----------------------------------------------
            # Backprop
            # -----------------------------------------------

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP,
            )

            optimizer.step()

            batch_size = (
                y.shape[0]
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            n_seen += (
                batch_size
            )

        train_loss = (
            running_loss
            / n_seen
        )

        # ----------------------------------------------------
        # CLEAN validation
        #
        # All four modalities.
        #
        # This is the ONLY metric used for checkpoint selection.
        # ----------------------------------------------------

        (
            val_metrics,
            variance_summary,
        ) = evaluate_clean(
            model,
            val_loader,
        )

        # ----------------------------------------------------
        # Store missingness distribution
        # ----------------------------------------------------

        n_batches = sum(
            epoch_pattern_counter.values()
        )

        pattern_distribution = {

            key:
                float(
                    value
                    / n_batches
                )

            for key, value
            in sorted(
                epoch_pattern_counter.items()
            )
        }

        epoch_result = {

            "epoch":
                int(
                    epoch
                ),

            "train_loss":
                float(
                    train_loss
                ),

            "clean_val":
                val_metrics,

            "clean_fused_variance":
                variance_summary,

            "training_missingness":
                pattern_distribution,
        }

        history.append(
            epoch_result
        )

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        n0 = sum(
            count
            for pattern, count
            in epoch_pattern_counter.items()
            if pattern == "all"
        )

        n1 = sum(
            count
            for pattern, count
            in epoch_pattern_counter.items()
            if pattern != "all"
            and pattern.count("+") == 2
        )

        n2 = sum(
            count
            for pattern, count
            in epoch_pattern_counter.items()
            if pattern != "all"
            and pattern.count("+") == 1
        )

        n3 = sum(
            count
            for pattern, count
            in epoch_pattern_counter.items()
            if pattern != "all"
            and pattern.count("+") == 0
        )

        print(
            f"epoch={epoch:02d} "
            f"loss={train_loss:.5f} "
            f"val_auc="
            f"{val_metrics['auroc']:.5f} "
            f"val_pr="
            f"{val_metrics['auprc']:.5f} "
            f"val_brier="
            f"{val_metrics['brier']:.5f} "
            f"var="
            f"{variance_summary['mean']:.5f} "
            f"| batches "
            f"4m={n0} "
            f"3m={n1} "
            f"2m={n2} "
            f"1m={n3}"
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            val_metrics[
                "auroc"
            ]
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
                        "D_missing_aware",

                    "seed":
                        SEED,

                    "epoch":
                        int(
                            epoch
                        ),

                    "model_state_dict":
                        model.state_dict(),

                    "val_metrics":
                        val_metrics,

                    "variance_summary":
                        variance_summary,

                    "missingness_schedule": {

                        "all":
                            P_ALL,

                        "one_missing":
                            P_ONE_MISSING,

                        "two_missing":
                            P_TWO_MISSING,

                        "three_missing":
                            P_THREE_MISSING,
                    },
                },
                checkpoint_path,
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "Early stopping."
            )

            break

    # ========================================================
    # Training finished
    # ========================================================

    training_seconds = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
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

    model.eval()

    # --------------------------------------------------------
    # Best clean validation
    # --------------------------------------------------------

    (
        best_clean_metrics,
        best_clean_variance,
    ) = evaluate_clean(
        model,
        val_loader,
    )

    # --------------------------------------------------------
    # A few immediate missingness diagnostics
    #
    # Full 15-pattern evaluation comes later using the same
    # evaluation framework as B.
    # --------------------------------------------------------

    diagnostic_patterns = {

        "all":
            set(),

        "structured_timeseries":
            {
                "cxr",
                "text",
            },

        "structured_only":
            {
                "timeseries",
                "cxr",
                "text",
            },

        "timeseries_only":
            {
                "structured",
                "cxr",
                "text",
            },

        "cxr_only":
            {
                "structured",
                "timeseries",
                "text",
            },

        "text_only":
            {
                "structured",
                "timeseries",
                "cxr",
            },
    }

    diagnostics = {}

    for name, missing in (
        diagnostic_patterns.items()
    ):

        diagnostics[
            name
        ] = (
            evaluate_missing_pattern(
                model,
                val_loader,
                missing,
            )
        )

    # --------------------------------------------------------
    # Global training pattern frequencies
    # --------------------------------------------------------

    total_batches = sum(
        global_pattern_counter.values()
    )

    global_pattern_distribution = {

        pattern:
            float(
                count
                / total_batches
            )

        for pattern, count
        in sorted(
            global_pattern_counter.items()
        )
    }

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "name":
            "D_missing_aware",

        "task":
            "48-hour mortality",

        "fusion":
            "Gaussian PoE",

        "reliability_weighting":
            False,

        "missing_aware_training":
            True,

        "test_evaluated":
            False,

        "seed":
            SEED,

        "best_epoch":
            int(
                best_epoch
            ),

        "selection_metric":
            "clean validation AUROC",

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

            "gradient_clip":
                GRAD_CLIP,

            "loss":
                "BCEWithLogitsLoss",

            "pos_weight":
                float(
                    pos_weight
                ),

            "training_seconds":
                float(
                    training_seconds
                ),
        },

        "missingness_schedule": {

            "all_modalities":
                P_ALL,

            "one_missing":
                P_ONE_MISSING,

            "two_missing":
                P_TWO_MISSING,

            "three_missing":
                P_THREE_MISSING,
        },

        "actual_training_pattern_distribution":
            global_pattern_distribution,

        "best_clean_validation": {

            **best_clean_metrics,

            "fused_variance":
                best_clean_variance,
        },

        "diagnostic_missingness":
            diagnostics,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "history":
            str(
                history_path
            ),
    }

    summary_path = (
        OUT_DIR
        / "D_missing_aware_summary.json"
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

    return (
        model,
        summary,
        summary_path,
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
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

    print(
        "=" * 80
    )

    print(
        "HAIM ABLATION D"
    )

    print(
        "GAUSSIAN PoE + "
        "MISSING-MODALITY-AWARE TRAINING"
    )

    print(
        "=" * 80
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "\nModalities:"
    )

    for modality, dim in (
        MODALITY_DIMS.items()
    ):

        print(
            f"  {modality:15s} "
            f"{dim}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X = {

        modality:
            np.load(
                DATA_DIR
                / f"X_{modality}.npy",
                mmap_mode="r",
            )

        for modality
        in MODALITIES
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
        len(
            y
        ),
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
        "Train:",
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

    print(
        "\nTrain positives:",
        int(
            y[
                train_mask
            ].sum()
        ),
    )

    print(
        "Validation positives:",
        int(
            y[
                val_mask
            ].sum()
        ),
    )

    # --------------------------------------------------------
    # Load EXACT same normalization as A/B/C
    # --------------------------------------------------------

    print(
        "\nLoading A/B/C "
        "train-derived normalization..."
    )

    stats = (
        load_abc_normalization()
    )

    X_norm = (
        apply_standardizer(
            X,
            stats,
        )
    )

    print(
        "Normalization applied."
    )

    # --------------------------------------------------------
    # Indices
    # --------------------------------------------------------

    train_indices = np.where(
        train_mask
    )[0]

    val_indices = np.where(
        val_mask
    )[0]

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = (
        HAIMDataset(
            X_norm,
            y,
            train_indices,
        )
    )

    val_dataset = (
        HAIMDataset(
            X_norm,
            y,
            val_indices,
        )
    )

    # --------------------------------------------------------
    # Loader
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------

    y_train = (
        y[
            train_mask
        ]
    )

    n_positive = int(
        y_train.sum()
    )

    n_negative = int(
        len(
            y_train
        )
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
    # Missingness schedule
    # --------------------------------------------------------

    print(
        "\nTraining missingness schedule:"
    )

    print(
        f"  all modalities : "
        f"{P_ALL:.2f}"
    )

    print(
        f"  one missing    : "
        f"{P_ONE_MISSING:.2f}"
    )

    print(
        f"  two missing    : "
        f"{P_TWO_MISSING:.2f}"
    )

    print(
        f"  three missing  : "
        f"{P_THREE_MISSING:.2f}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    (
        model,
        summary,
        summary_path,
    ) = train_model(

        train_loader=
            train_loader,

        val_loader=
            val_loader,

        pos_weight=
            pos_weight,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "D VALIDATION SUMMARY"
    )

    print(
        "=" * 80
    )

    clean = summary[
        "best_clean_validation"
    ]

    print(
        f"Best epoch: "
        f"{summary['best_epoch']}"
    )

    print(
        f"Clean AUROC: "
        f"{clean['auroc']:.5f}"
    )

    print(
        f"Clean AUPRC: "
        f"{clean['auprc']:.5f}"
    )

    print(
        f"Clean Brier: "
        f"{clean['brier']:.5f}"
    )

    print(
        f"Clean fused variance: "
        f"{clean['fused_variance']['mean']:.5f}"
    )

    print(
        "\nSelected missingness diagnostics:"
    )

    for name, result in (
        summary[
            "diagnostic_missingness"
        ].items()
    ):

        print(
            f"{name:25s} "
            f"AUROC="
            f"{result['auroc']:.5f} "
            f"AUPRC="
            f"{result['auprc']:.5f} "
            f"Brier="
            f"{result['brier']:.5f} "
            f"Var="
            f"{result['fused_variance_mean']:.5f}"
        )

    print(
        "\nTest set was NOT evaluated."
    )

    print(
        "\nSaved:"
    )

    print(
        summary_path
    )

    print(
        OUT_DIR
        / "D_missing_aware_best.pt"
    )


if __name__ == "__main__":

    main()