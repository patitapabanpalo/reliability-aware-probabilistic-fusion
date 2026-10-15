"""
Train HAIM ablation F:

    F = Reliability-weighted Gaussian PoE
        + missing-modality-aware training
        + complete-to-incomplete consistency

Controlled ablation:

    E:
        Reliability PoE
        + missing-aware mortality training

    F:
        E
        + logit consistency

Mortality-branch missingness schedule:

    50% : all 4 modalities
    20% : 3 available / 1 missing
    20% : 2 available / 2 missing
    10% : 1 available / 3 missing

Consistency branch:

    - Always has at least one modality missing.
    - Uses the same patient batch.
    - Complete-modality prediction acts as teacher.
    - Teacher logits are detached.

Loss:

    L_total =
        L_mortality
        + lambda_consistency * L_consistency

    L_consistency =
        MSE(
            incomplete_logits,
            stopgrad(complete_logits)
        )

IMPORTANT:
    - Same frozen patient split.
    - Same normalization as A-E.
    - Same weighted BCE.
    - Same optimizer / LR / batch size.
    - Same E missingness schedule.
    - lambda_consistency = 0.1.
    - Clean validation AUROC selects checkpoint.
    - Test set is NOT evaluated.
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
    / "full_f"
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

EPOCHS = 40
MAX_EPOCHS = 50
DEFAULT_MAX_EPOCHS = MAX_EPOCHS

PATIENCE = 8

LR = 1e-3

WEIGHT_DECAY = 1e-4

GRAD_CLIP = 5.0


LAMBDA_CONSISTENCY = 0.10


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODALITIES = tuple(
    MODALITY_DIMS.keys()
)


# ============================================================
# Missing-aware mortality schedule
#
# MUST remain identical to D/E.
# ============================================================

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

def seed_everything(
    seed,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

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


    def __len__(
        self,
    ):

        return len(
            self.indices
        )


    def __getitem__(
        self,
        idx,
    ):

        real_idx = (
            self.indices[
                idx
            ]
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

def load_shared_normalization():
    """
    Load exactly the same train-derived normalization used
    for A/B/C/D/E.
    """

    path = (
        ABC_DIR
        / "normalization_stats.npz"
    )

    if not path.exists():

        raise FileNotFoundError(
            "Could not find shared normalization "
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
# Missingness helpers
# ============================================================

def sample_task_missing_pattern():
    """
    Same mortality-training missingness distribution used
    for D and E.

    Returns:
        set of missing modality names.
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

    return set(
        random.sample(
            list(
                MODALITIES
            ),
            n_missing,
        )
    )


def sample_consistency_missing_pattern():
    """
    Consistency branch must ALWAYS be incomplete.

    We choose the missingness severity uniformly across:

        1 missing
        2 missing
        3 missing

    This avoids wasting the consistency objective on an
    all-modalities branch where teacher and student would
    be identical.

    This distribution affects only the auxiliary consistency
    branch. It does NOT alter E's mortality-task schedule.
    """

    n_missing = random.choice(
        [
            1,
            2,
            3,
        ]
    )

    return set(
        random.sample(
            list(
                MODALITIES
            ),
            n_missing,
        )
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

    if len(
        missing
    ) == 0:

        return "all"

    return (
        "available:"
        + "+".join(
            available
        )
    )


def number_available(
    missing,
):

    return (
        len(
            MODALITIES
        )
        - len(
            missing
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
# Clean validation
# ============================================================

@torch.no_grad()
def evaluate_clean(
    model,
    loader,
):

    model.eval()

    labels = []

    probabilities = []

    fused_variances = []

    reliability_values = {

        modality: []

        for modality
        in MODALITIES
    }

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

        # ----------------------------------------------------
        # Fused variance
        # ----------------------------------------------------

        variance = (
            output[
                "fused_variance"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        fused_variances.append(
            variance.mean(
                axis=1
            )
        )

        # ----------------------------------------------------
        # Reliability
        # ----------------------------------------------------

        evidence = output[
            "evidence"
        ]

        for modality in MODALITIES:

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

            reliability_values[
                modality
            ].append(
                reliability
            )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    y_true = np.concatenate(
        labels
    )

    probability = np.concatenate(
        probabilities
    )

    fused_variance = np.concatenate(
        fused_variances
    )

    metrics = compute_metrics(
        y_true,
        probability,
    )

    variance_summary = {

        "mean":
            float(
                np.mean(
                    fused_variance
                )
            ),

        "std":
            float(
                np.std(
                    fused_variance
                )
            ),

        "min":
            float(
                np.min(
                    fused_variance
                )
            ),

        "max":
            float(
                np.max(
                    fused_variance
                )
            ),
    }

    reliability_summary = {}

    for modality, parts in (
        reliability_values.items()
    ):

        values = np.concatenate(
            parts
        )

        reliability_summary[
            modality
        ] = {

            "mean":
                float(
                    np.mean(
                        values
                    )
                ),

            "std":
                float(
                    np.std(
                        values
                    )
                ),

            "min":
                float(
                    np.min(
                        values
                    )
                ),

            "max":
                float(
                    np.max(
                        values
                    )
                ),
        }

    return (
        metrics,
        variance_summary,
        reliability_summary,
    )


# ============================================================
# Missing-pattern validation diagnostic
# ============================================================

@torch.no_grad()
def evaluate_missing_pattern(
    model,
    loader,
    missing,
):

    model.eval()

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

    labels = []

    probabilities = []

    fused_variances = []

    reliability_values = {

        modality: []

        for modality
        in available
    }

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

        fused_variances.append(
            variance.mean(
                axis=1
            )
        )

        evidence = output[
            "evidence"
        ]

        for modality in available:

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

            reliability_values[
                modality
            ].append(
                reliability
            )

    y_true = np.concatenate(
        labels
    )

    probability = np.concatenate(
        probabilities
    )

    fused_variance = np.concatenate(
        fused_variances
    )

    metrics = compute_metrics(
        y_true,
        probability,
    )

    reliability_summary = {}

    for modality, parts in (
        reliability_values.items()
    ):

        values = np.concatenate(
            parts
        )

        reliability_summary[
            modality
        ] = {

            "mean":
                float(
                    np.mean(
                        values
                    )
                ),

            "std":
                float(
                    np.std(
                        values
                    )
                ),

            "min":
                float(
                    np.min(
                        values
                    )
                ),

            "max":
                float(
                    np.max(
                        values
                    )
                ),
        }

    return {

        **metrics,

        "fused_variance_mean":
            float(
                np.mean(
                    fused_variance
                )
            ),

        "fused_variance_std":
            float(
                np.std(
                    fused_variance
                )
            ),

        "reliability":
            reliability_summary,
    }


# ============================================================
# Model F
# ============================================================

def build_model_f():
    """
    Same predictive architecture as E.

    F differs only through the consistency training objective.
    """

    return HAIMProbabilisticModel(
        fusion_type="reliability"
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
        + "=" * 90
    )

    print(
        "F: RELIABILITY PoE + "
        "MISSING-AWARE + CONSISTENCY"
    )

    print(
        "=" * 90
    )

    seed_everything(
        SEED
    )

    model = (
        build_model_f()
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

    mortality_criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight,
                dtype=torch.float32,
                device=DEVICE,
            )
        )
    )

    consistency_criterion = (
        nn.MSELoss()
    )

    checkpoint_path = (
        OUT_DIR
        / "F_full_best.pt"
    )

    history_path = (
        OUT_DIR
        / "F_full_history.json"
    )

    best_auc = -np.inf

    best_epoch = None

    epochs_without_improvement = 0

    history = []

    task_pattern_counter_global = (
        Counter()
    )

    consistency_pattern_counter_global = (
        Counter()
    )

    start_time = time.time()

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        running_total_loss = 0.0

        running_mortality_loss = 0.0

        running_consistency_loss = 0.0

        n_seen = 0

        task_pattern_counter = (
            Counter()
        )

        consistency_pattern_counter = (
            Counter()
        )

        # ====================================================
        # Training batches
        # ====================================================

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

            # ------------------------------------------------
            # 1. Mortality task branch
            #
            # Same missingness distribution as E.
            # ------------------------------------------------

            task_missing = (
                sample_task_missing_pattern()
            )

            task_pattern_name = (
                canonical_pattern_name(
                    task_missing
                )
            )

            task_pattern_counter[
                task_pattern_name
            ] += 1

            task_pattern_counter_global[
                task_pattern_name
            ] += 1

            # ------------------------------------------------
            # 2. Consistency student branch
            #
            # Always incomplete.
            # ------------------------------------------------

            consistency_missing = (
                sample_consistency_missing_pattern()
            )

            consistency_pattern_name = (
                canonical_pattern_name(
                    consistency_missing
                )
            )

            consistency_pattern_counter[
                consistency_pattern_name
            ] += 1

            consistency_pattern_counter_global[
                consistency_pattern_name
            ] += 1

            # ------------------------------------------------
            # Forward / loss
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Mortality branch
            # ------------------------------------------------

            task_logits = model(
                batch,
                missing=task_missing,
            )

            mortality_loss = (
                mortality_criterion(
                    task_logits,
                    y,
                )
            )

            # ------------------------------------------------
            # Complete teacher
            #
            # IMPORTANT:
            # We need the complete prediction, but gradients
            # must NOT flow through the teacher target.
            #
            # We still run it normally and detach the logits.
            # ------------------------------------------------

            complete_logits = model(
                batch,
                missing=set(),
            )

            teacher_logits = (
                complete_logits.detach()
            )

            # ------------------------------------------------
            # Incomplete student
            # ------------------------------------------------

            incomplete_logits = model(
                batch,
                missing=consistency_missing,
            )

            consistency_loss = (
                consistency_criterion(
                    incomplete_logits,
                    teacher_logits,
                )
            )

            # ------------------------------------------------
            # Full F objective
            # ------------------------------------------------

            total_loss = (

                mortality_loss

                + LAMBDA_CONSISTENCY
                * consistency_loss
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP,
            )

            optimizer.step()

            # ------------------------------------------------
            # Accounting
            # ------------------------------------------------

            batch_size = (
                y.shape[0]
            )

            running_total_loss += (
                total_loss.item()
                * batch_size
            )

            running_mortality_loss += (
                mortality_loss.item()
                * batch_size
            )

            running_consistency_loss += (
                consistency_loss.item()
                * batch_size
            )

            n_seen += (
                batch_size
            )

        # ====================================================
        # Epoch losses
        # ====================================================

        train_total_loss = (
            running_total_loss
            / n_seen
        )

        train_mortality_loss = (
            running_mortality_loss
            / n_seen
        )

        train_consistency_loss = (
            running_consistency_loss
            / n_seen
        )

        # ====================================================
        # CLEAN validation
        #
        # Checkpoint selection remains clean validation AUROC.
        # ====================================================

        (
            val_metrics,
            variance_summary,
            reliability_summary,
        ) = evaluate_clean(
            model,
            val_loader,
        )

        # ====================================================
        # Pattern distributions
        # ====================================================

        n_task_batches = sum(
            task_pattern_counter.values()
        )

        task_distribution = {

            pattern:
                float(
                    count
                    / n_task_batches
                )

            for pattern, count
            in sorted(
                task_pattern_counter.items()
            )
        }


        n_consistency_batches = sum(
            consistency_pattern_counter.values()
        )

        consistency_distribution = {

            pattern:
                float(
                    count
                    / n_consistency_batches
                )

            for pattern, count
            in sorted(
                consistency_pattern_counter.items()
            )
        }

        # ====================================================
        # History
        # ====================================================

        epoch_result = {

            "epoch":
                int(
                    epoch
                ),

            "train_total_loss":
                float(
                    train_total_loss
                ),

            "train_mortality_loss":
                float(
                    train_mortality_loss
                ),

            "train_consistency_loss":
                float(
                    train_consistency_loss
                ),

            "lambda_consistency":
                float(
                    LAMBDA_CONSISTENCY
                ),

            "clean_val":
                val_metrics,

            "clean_fused_variance":
                variance_summary,

            "clean_reliability":
                reliability_summary,

            "task_missingness":
                task_distribution,

            "consistency_missingness":
                consistency_distribution,
        }

        history.append(
            epoch_result
        )

        # ====================================================
        # Console counts by number available
        # ====================================================

        task_counts = {
            4: 0,
            3: 0,
            2: 0,
            1: 0,
        }

        for pattern_name, count in (
            task_pattern_counter.items()
        ):

            if pattern_name == "all":

                task_counts[4] += count

            else:

                available_part = (
                    pattern_name.split(
                        "available:",
                        1,
                    )[1]
                )

                n_available = (
                    available_part.count("+")
                    + 1
                )

                task_counts[
                    n_available
                ] += count


        consistency_counts = {
            3: 0,
            2: 0,
            1: 0,
        }

        for pattern_name, count in (
            consistency_pattern_counter.items()
        ):

            available_part = (
                pattern_name.split(
                    "available:",
                    1,
                )[1]
            )

            n_available = (
                available_part.count("+")
                + 1
            )

            consistency_counts[
                n_available
            ] += count

        # ====================================================
        # Reliability display
        # ====================================================

        reliability_string = " ".join(

            f"{modality[:3]}="
            f"{reliability_summary[modality]['mean']:.3f}"

            for modality
            in MODALITIES
        )

        # ====================================================
        # Console
        # ====================================================

        print(
            f"epoch={epoch:02d} "
            f"total={train_total_loss:.5f} "
            f"mort={train_mortality_loss:.5f} "
            f"cons={train_consistency_loss:.5f} "
            f"val_auc={val_metrics['auroc']:.5f} "
            f"val_pr={val_metrics['auprc']:.5f} "
            f"brier={val_metrics['brier']:.5f} "
            f"var={variance_summary['mean']:.5f}"
        )

        print(
            f"         reliability "
            f"{reliability_string}"
        )

        print(
            f"         task batches "
            f"4m={task_counts[4]} "
            f"3m={task_counts[3]} "
            f"2m={task_counts[2]} "
            f"1m={task_counts[1]} "
            f"| consistency "
            f"3m={consistency_counts[3]} "
            f"2m={consistency_counts[2]} "
            f"1m={consistency_counts[1]}"
        )

        # ====================================================
        # Checkpoint
        # ====================================================

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
                        "F_full",

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

                    "reliability_summary":
                        reliability_summary,

                    "lambda_consistency":
                        LAMBDA_CONSISTENCY,

                    "task_missingness_schedule": {

                        "all":
                            P_ALL,

                        "one_missing":
                            P_ONE_MISSING,

                        "two_missing":
                            P_TWO_MISSING,

                        "three_missing":
                            P_THREE_MISSING,
                    },

                    "consistency_missingness":
                        "uniform severity over "
                        "1/2/3 missing modalities",
                },
                checkpoint_path,
            )

        else:

            epochs_without_improvement += 1

        # ====================================================
        # Early stopping
        # ====================================================

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

    # ========================================================
    # Save history
    # ========================================================

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

    # ========================================================
    # Restore best checkpoint
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # ========================================================
    # Best clean validation
    # ========================================================

    (
        best_clean_metrics,
        best_clean_variance,
        best_clean_reliability,
    ) = evaluate_clean(
        model,
        val_loader,
    )

    # ========================================================
    # Selected missingness diagnostics
    # ========================================================

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
        ] = evaluate_missing_pattern(
            model,
            val_loader,
            missing,
        )

    # ========================================================
    # Global task pattern distribution
    # ========================================================

    total_task_batches = sum(
        task_pattern_counter_global.values()
    )

    global_task_distribution = {

        pattern:
            float(
                count
                / total_task_batches
            )

        for pattern, count
        in sorted(
            task_pattern_counter_global.items()
        )
    }

    # ========================================================
    # Global consistency pattern distribution
    # ========================================================

    total_consistency_batches = sum(
        consistency_pattern_counter_global.values()
    )

    global_consistency_distribution = {

        pattern:
            float(
                count
                / total_consistency_batches
            )

        for pattern, count
        in sorted(
            consistency_pattern_counter_global.items()
        )
    }

    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "name":
            "F_full",

        "task":
            "48-hour mortality",

        "fusion":
            "Reliability-weighted Gaussian PoE",

        "reliability_weighting":
            True,

        "explicit_reliability_supervision":
            False,

        "missing_aware_training":
            True,

        "consistency_training":
            True,

        "consistency_type":
            "complete-to-incomplete logit MSE",

        "teacher_detached":
            True,

        "lambda_consistency":
            float(
                LAMBDA_CONSISTENCY
            ),

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

            "mortality_loss":
                "BCEWithLogitsLoss",

            "pos_weight":
                float(
                    pos_weight
                ),

            "consistency_loss":
                "MSELoss on logits",

            "training_seconds":
                float(
                    training_seconds
                ),
        },

        "task_missingness_schedule": {

            "all_modalities":
                P_ALL,

            "one_missing":
                P_ONE_MISSING,

            "two_missing":
                P_TWO_MISSING,

            "three_missing":
                P_THREE_MISSING,
        },

        "consistency_missingness_schedule": {

            "description":
                "Always incomplete; "
                "uniform severity",

            "one_missing":
                1.0 / 3.0,

            "two_missing":
                1.0 / 3.0,

            "three_missing":
                1.0 / 3.0,
        },

        "actual_task_pattern_distribution":
            global_task_distribution,

        "actual_consistency_pattern_distribution":
            global_consistency_distribution,

        "best_clean_validation": {

            **best_clean_metrics,

            "fused_variance":
                best_clean_variance,

            "reliability":
                best_clean_reliability,
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
        / "F_full_summary.json"
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

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help=(
            "Maximum number of training epochs. "
            f"Default: {DEFAULT_MAX_EPOCHS}"
        ),
    )

    return parser.parse_args()

# ============================================================
# Main
# ============================================================

def main():

    global SEED
    global OUT_DIR
    global MAX_EPOCHS

    args = parse_args()

    SEED = int(
        args.seed
    )

    MAX_EPOCHS = int(
        args.max_epochs
    )

    if MAX_EPOCHS <= 0:
        raise ValueError(
            "--max-epochs must be positive."
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

    print(
        "Maximum epochs:",
        MAX_EPOCHS,
    )

    seed_everything(
        SEED
    )

    print(
        "=" * 90
    )

    print(
        "HAIM ABLATION F"
    )

    print(
        "RELIABILITY PoE "
        "+ MISSING-AWARE "
        "+ CONSISTENCY"
    )

    print(
        "=" * 90
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

    # ========================================================
    # Load data
    # ========================================================

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

    # ========================================================
    # Frozen split
    # ========================================================

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

    # ========================================================
    # Shared normalization
    # ========================================================

    print(
        "\nLoading shared "
        "train-derived normalization..."
    )

    stats = (
        load_shared_normalization()
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

    # ========================================================
    # Indices
    # ========================================================

    train_indices = np.where(
        train_mask
    )[0]

    val_indices = np.where(
        val_mask
    )[0]

    # ========================================================
    # Datasets
    # ========================================================

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

    # ========================================================
    # Loaders
    # ========================================================

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

    # ========================================================
    # Class imbalance
    # ========================================================

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

    # ========================================================
    # Protocol
    # ========================================================

    print(
        "\nMortality-task missingness schedule:"
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

    print(
        "\nConsistency:"
    )

    print(
        "  teacher       : complete modalities"
    )

    print(
        "  teacher grad  : detached"
    )

    print(
        "  student       : always incomplete"
    )

    print(
        "  missing level : uniform over 1/2/3 missing"
    )

    print(
        "  loss          : logit MSE"
    )

    print(
        f"  lambda        : "
        f"{LAMBDA_CONSISTENCY}"
    )

    print(
        "\nReliability supervision: NONE"
    )

    # ========================================================
    # Train
    # ========================================================

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

    # ========================================================
    # Final summary
    # ========================================================

    clean = (
        summary[
            "best_clean_validation"
        ]
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "F VALIDATION SUMMARY"
    )

    print(
        "=" * 90
    )

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
        "\nClean reliability:"
    )

    for modality in MODALITIES:

        reliability = (
            clean[
                "reliability"
            ][
                modality
            ]
        )

        print(
            f"  {modality:15s} "
            f"mean="
            f"{reliability['mean']:.5f} "
            f"std="
            f"{reliability['std']:.5f} "
            f"range=["
            f"{reliability['min']:.5f}, "
            f"{reliability['max']:.5f}]"
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
        / "F_full_best.pt"
    )


if __name__ == "__main__":

    main()