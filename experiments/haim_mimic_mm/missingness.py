"""
HAIM missing-modality evaluation: D vs E vs F.

D:
    Gaussian PoE
    + missing-aware training

E:
    Reliability-weighted Gaussian PoE
    + missing-aware training

F:
    Reliability-weighted Gaussian PoE
    + missing-aware training
    + complete-to-incomplete consistency

Evaluation:
    - Validation split only
    - All 15 non-empty modality combinations
    - No retraining
    - Test set remains untouched

Primary contrasts:

    F - E
        Isolated effect of consistency.

    F - D
        Full method versus missing-aware Gaussian PoE.

    E - D
        Effect of reliability under missing-aware training.

Outputs:
    missingness_def_results.json
    missingness_def_results.csv
    missingness_def_comparison.csv
    missingness_def_grouped.csv
    missingness_def_summary.json
"""

from pathlib import Path
import os
import sys
import json
import itertools

import numpy as np
import pandas as pd
import torch

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


D_DIR = (
    RESULTS_ROOT
    / "missing_aware_d"
)


E_DIR = (
    RESULTS_ROOT
    / "missing_reliability_e"
)


F_DIR = (
    RESULTS_ROOT
    / "full_f"
)


OUT_DIR = (
    RESULTS_ROOT
    / "missingness_def"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 256


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODALITIES = tuple(
    MODALITY_DIMS.keys()
)


# ============================================================
# Model constructors
# ============================================================

def build_model_d():
    """
    D:
        Gaussian PoE + missing-aware training.
    """

    return HAIMProbabilisticModel(
        fusion_type="gaussian"
    )


def build_model_e():
    """
    E:
        Reliability PoE + missing-aware training.
    """

    return HAIMProbabilisticModel(
        fusion_type="reliability"
    )


def build_model_f():
    """
    F:
        Same predictive architecture as E.

    F differs from E through its consistency-training
    objective, not through inference architecture.
    """

    return HAIMProbabilisticModel(
        fusion_type="reliability"
    )


# ============================================================
# Checkpoints
# ============================================================

MODEL_CONFIGS = {

    "D_missing_aware": {

        "builder":
            build_model_d,

        "checkpoint":
            D_DIR
            / "D_missing_aware_best.pt",

        "description":
            "Gaussian PoE + missing-aware training",
    },


    "E_missing_reliability": {

        "builder":
            build_model_e,

        "checkpoint":
            E_DIR
            / "E_missing_reliability_best.pt",

        "description":
            "Reliability PoE + missing-aware training",
    },


    "F_full": {

        "builder":
            build_model_f,

        "checkpoint":
            F_DIR
            / "F_full_best.pt",

        "description":
            "Reliability PoE + missing-aware + consistency",
    },
}


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
# Shared normalization
# ============================================================

def load_normalization():
    """
    D/E/F all use the same training-derived normalization
    originally fitted during the A/B/C experiment.
    """

    path = (
        ABC_DIR
        / "normalization_stats.npz"
    )

    if not path.exists():

        raise FileNotFoundError(
            "Normalization statistics "
            f"not found:\n{path}"
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
# Missingness patterns
# ============================================================

def generate_patterns():
    """
    Generate all 15 non-empty availability combinations.

    Ordering:
        4 modalities
        3 modalities
        2 modalities
        1 modality
    """

    patterns = []

    for n_available in range(
        len(
            MODALITIES
        ),
        0,
        -1,
    ):

        for available in itertools.combinations(
            MODALITIES,
            n_available,
        ):

            available = tuple(
                available
            )

            missing = tuple(

                modality

                for modality
                in MODALITIES

                if modality
                not in available
            )

            patterns.append({

                "available":
                    available,

                "missing":
                    missing,

                "n_available":
                    len(
                        available
                    ),

                "n_missing":
                    len(
                        missing
                    ),
            })

    return patterns


def get_available_name(
    available,
):

    if len(
        available
    ) == len(
        MODALITIES
    ):

        return "all"

    return "+".join(
        available
    )


def get_missing_name(
    missing,
):

    if len(
        missing
    ) == 0:

        return "none"

    return "+".join(
        missing
    )


# ============================================================
# Model loading
# ============================================================

def load_model(
    model_name,
    config,
):

    checkpoint_path = (
        config[
            "checkpoint"
        ]
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint for "
            f"{model_name} "
            f"not found:\n"
            f"{checkpoint_path}"
        )

    model = (
        config[
            "builder"
        ]()
    ).to(
        DEVICE
    )

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

    print(
        f"Loaded {model_name}"
    )

    print(
        f"  checkpoint: "
        f"{checkpoint_path}"
    )

    if "epoch" in checkpoint:

        print(
            f"  epoch: "
            f"{checkpoint['epoch']}"
        )

    return model


# ============================================================
# Evaluate one availability pattern
# ============================================================

@torch.no_grad()
def evaluate_pattern(
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

    # ========================================================
    # Batches
    # ========================================================

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

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

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
        # Posterior variance
        # ----------------------------------------------------

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

        fused_variances.append(
            sample_variance
        )

        # ----------------------------------------------------
        # Reliability
        #
        # D's reliability heads are unused by its fusion and
        # should not be interpreted scientifically.
        #
        # E/F reliability values ARE used by fusion.
        # ----------------------------------------------------

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

    # ========================================================
    # Aggregate
    # ========================================================

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

    return {

        "metrics":
            metrics,

        "fused_variance":
            variance_summary,

        "reliability":
            reliability_summary,
    }


# ============================================================
# Contrast helper
# ============================================================

def add_contrast(
    comparison,
    metric,
    left,
    right,
    name,
):

    comparison[
        f"{metric}_{name}"
    ] = (

        comparison[
            f"{metric}_{left}"
        ]

        - comparison[
            f"{metric}_{right}"
        ]
    )


def summarize_contrast(
    comparison,
    contrast_name,
    left_model,
    right_model,
):

    auc_col = (
        f"auroc_{contrast_name}"
    )

    pr_col = (
        f"auprc_{contrast_name}"
    )

    missing_only = (
        comparison[
            comparison[
                "n_available"
            ]
            < len(
                MODALITIES
            )
        ]
        .copy()
    )

    return {

        "left_model":
            left_model,

        "right_model":
            right_model,

        "interpretation":
            f"{left_model} minus "
            f"{right_model}",

        "all_conditions": {

            "n_conditions":
                int(
                    len(
                        comparison
                    )
                ),

            "auroc_wins":
                int(
                    (
                        comparison[
                            auc_col
                        ]
                        > 0
                    ).sum()
                ),

            "auroc_losses":
                int(
                    (
                        comparison[
                            auc_col
                        ]
                        < 0
                    ).sum()
                ),

            "mean_delta_auroc":
                float(
                    comparison[
                        auc_col
                    ].mean()
                ),

            "auprc_wins":
                int(
                    (
                        comparison[
                            pr_col
                        ]
                        > 0
                    ).sum()
                ),

            "auprc_losses":
                int(
                    (
                        comparison[
                            pr_col
                        ]
                        < 0
                    ).sum()
                ),

            "mean_delta_auprc":
                float(
                    comparison[
                        pr_col
                    ].mean()
                ),
        },

        "missing_conditions_only": {

            "n_conditions":
                int(
                    len(
                        missing_only
                    )
                ),

            "auroc_wins":
                int(
                    (
                        missing_only[
                            auc_col
                        ]
                        > 0
                    ).sum()
                ),

            "auroc_losses":
                int(
                    (
                        missing_only[
                            auc_col
                        ]
                        < 0
                    ).sum()
                ),

            "mean_delta_auroc":
                float(
                    missing_only[
                        auc_col
                    ].mean()
                ),

            "auprc_wins":
                int(
                    (
                        missing_only[
                            pr_col
                        ]
                        > 0
                    ).sum()
                ),

            "auprc_losses":
                int(
                    (
                        missing_only[
                            pr_col
                        ]
                        < 0
                    ).sum()
                ),

            "mean_delta_auprc":
                float(
                    missing_only[
                        pr_col
                    ].mean()
                ),
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "HAIM D / E / F MISSING-MODALITY EVALUATION"
    )

    print(
        "=" * 120
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "\nModels:"
    )

    for model_name, config in (
        MODEL_CONFIGS.items()
    ):

        print(
            f"  {model_name:25s} "
            f"{config['description']}"
        )

    # ========================================================
    # Load model-ready arrays
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

    for modality in MODALITIES:

        print(
            f"{modality:15s}",
            X[
                modality
            ].shape,
        )

    # ========================================================
    # Validation only
    # ========================================================

    val_mask = (
        split == "val"
    )

    test_mask = (
        split == "test"
    )

    val_indices = np.where(
        val_mask
    )[0]

    print(
        "\nValidation rows:",
        int(
            val_mask.sum()
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

    print(
        "Validation prevalence:",
        float(
            y[
                val_mask
            ].mean()
        ),
    )

    print(
        "Test rows frozen:",
        int(
            test_mask.sum()
        ),
    )

    # ========================================================
    # Shared normalization
    # ========================================================

    print(
        "\nLoading shared train-derived normalization..."
    )

    stats = (
        load_normalization()
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
    # Validation loader
    # ========================================================

    val_dataset = (
        HAIMDataset(
            X_norm,
            y,
            val_indices,
        )
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available(),
    )

    # ========================================================
    # Generate all 15 patterns
    # ========================================================

    patterns = (
        generate_patterns()
    )

    assert len(
        patterns
    ) == 15

    print(
        "\nNumber of availability patterns:",
        len(
            patterns
        ),
    )

    # ========================================================
    # Evaluate D / E / F
    # ========================================================

    all_results = []

    for model_name, config in (
        MODEL_CONFIGS.items()
    ):

        print(
            "\n"
            + "=" * 120
        )

        print(
            model_name.upper()
        )

        print(
            "=" * 120
        )

        model = load_model(
            model_name,
            config,
        )

        for pattern in patterns:

            available = (
                pattern[
                    "available"
                ]
            )

            missing = (
                pattern[
                    "missing"
                ]
            )

            result = evaluate_pattern(
                model=model,
                loader=val_loader,
                missing=missing,
            )

            metrics = (
                result[
                    "metrics"
                ]
            )

            variance = (
                result[
                    "fused_variance"
                ]
            )

            row = {

                "model":
                    model_name,

                "available":
                    list(
                        available
                    ),

                "missing":
                    list(
                        missing
                    ),

                "available_name":
                    get_available_name(
                        available
                    ),

                "missing_name":
                    get_missing_name(
                        missing
                    ),

                "n_available":
                    int(
                        pattern[
                            "n_available"
                        ]
                    ),

                "n_missing":
                    int(
                        pattern[
                            "n_missing"
                        ]
                    ),

                "auroc":
                    metrics[
                        "auroc"
                    ],

                "auprc":
                    metrics[
                        "auprc"
                    ],

                "brier":
                    metrics[
                        "brier"
                    ],

                "fused_variance_mean":
                    variance[
                        "mean"
                    ],

                "fused_variance_std":
                    variance[
                        "std"
                    ],

                "fused_variance_min":
                    variance[
                        "min"
                    ],

                "fused_variance_max":
                    variance[
                        "max"
                    ],

                "reliability":
                    result[
                        "reliability"
                    ],
            }

            all_results.append(
                row
            )

            print(
                f"available="
                f"{row['available_name']:<43s} "
                f"missing="
                f"{row['missing_name']:<36s} "
                f"AUROC="
                f"{row['auroc']:.5f} "
                f"AUPRC="
                f"{row['auprc']:.5f} "
                f"Brier="
                f"{row['brier']:.5f} "
                f"Var="
                f"{row['fused_variance_mean']:.5f}"
            )

    # ========================================================
    # Save complete JSON
    # ========================================================

    full_output = {

        "task":
            "48-hour mortality",

        "evaluation_split":
            "validation",

        "test_evaluated":
            False,

        "n_validation":
            int(
                val_mask.sum()
            ),

        "n_validation_positive":
            int(
                y[
                    val_mask
                ].sum()
            ),

        "validation_prevalence":
            float(
                y[
                    val_mask
                ].mean()
            ),

        "modalities":
            list(
                MODALITIES
            ),

        "model_descriptions": {

            name:
                config[
                    "description"
                ]

            for name, config
            in MODEL_CONFIGS.items()
        },

        "results":
            all_results,
    }

    json_path = (
        OUT_DIR
        / "missingness_def_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            full_output,
            f,
            indent=2,
        )

    # ========================================================
    # Flat CSV
    # ========================================================

    flat_rows = []

    for row in all_results:

        flat = {

            "model":
                row[
                    "model"
                ],

            "available":
                row[
                    "available_name"
                ],

            "missing":
                row[
                    "missing_name"
                ],

            "n_available":
                row[
                    "n_available"
                ],

            "n_missing":
                row[
                    "n_missing"
                ],

            "auroc":
                row[
                    "auroc"
                ],

            "auprc":
                row[
                    "auprc"
                ],

            "brier":
                row[
                    "brier"
                ],

            "fused_variance_mean":
                row[
                    "fused_variance_mean"
                ],

            "fused_variance_std":
                row[
                    "fused_variance_std"
                ],
        }

        # Reliability means.
        for modality in MODALITIES:

            reliability = (
                row[
                    "reliability"
                ].get(
                    modality
                )
            )

            if reliability is None:

                flat[
                    f"reliability_{modality}"
                ] = np.nan

            else:

                flat[
                    f"reliability_{modality}"
                ] = (
                    reliability[
                        "mean"
                    ]
                )

        flat_rows.append(
            flat
        )

    flat_df = pd.DataFrame(
        flat_rows
    )

    flat_path = (
        OUT_DIR
        / "missingness_def_results.csv"
    )

    flat_df.to_csv(
        flat_path,
        index=False,
    )

    # ========================================================
    # Pivot comparison
    # ========================================================

    comparison = flat_df.pivot(

        index=[
            "n_available",
            "available",
            "missing",
        ],

        columns="model",

        values=[
            "auroc",
            "auprc",
            "brier",
            "fused_variance_mean",
        ],
    )

    comparison.columns = [

        f"{metric}_{model}"

        for metric, model
        in comparison.columns
    ]

    comparison = (
        comparison
        .reset_index()
    )

    # ========================================================
    # F - E
    #
    # Isolated consistency effect.
    # ========================================================

    add_contrast(
        comparison,
        metric="auroc",
        left="F_full",
        right="E_missing_reliability",
        name="F_minus_E",
    )

    add_contrast(
        comparison,
        metric="auprc",
        left="F_full",
        right="E_missing_reliability",
        name="F_minus_E",
    )

    # ========================================================
    # F - D
    #
    # Full method versus missing-aware Gaussian.
    # ========================================================

    add_contrast(
        comparison,
        metric="auroc",
        left="F_full",
        right="D_missing_aware",
        name="F_minus_D",
    )

    add_contrast(
        comparison,
        metric="auprc",
        left="F_full",
        right="D_missing_aware",
        name="F_minus_D",
    )

    # ========================================================
    # E - D
    #
    # Reliability effect under missing-aware training.
    # ========================================================

    add_contrast(
        comparison,
        metric="auroc",
        left="E_missing_reliability",
        right="D_missing_aware",
        name="E_minus_D",
    )

    add_contrast(
        comparison,
        metric="auprc",
        left="E_missing_reliability",
        right="D_missing_aware",
        name="E_minus_D",
    )

    # ========================================================
    # Brier improvement
    #
    # Positive = left model has LOWER Brier.
    # ========================================================

    comparison[
        "brier_improvement_F_over_E"
    ] = (

        comparison[
            "brier_E_missing_reliability"
        ]

        - comparison[
            "brier_F_full"
        ]
    )


    comparison[
        "brier_improvement_F_over_D"
    ] = (

        comparison[
            "brier_D_missing_aware"
        ]

        - comparison[
            "brier_F_full"
        ]
    )


    comparison[
        "brier_improvement_E_over_D"
    ] = (

        comparison[
            "brier_D_missing_aware"
        ]

        - comparison[
            "brier_E_missing_reliability"
        ]
    )

    # ========================================================
    # Sort
    # ========================================================

    comparison = (
        comparison
        .sort_values(
            by=[
                "n_available",
                "available",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    comparison_path = (
        OUT_DIR
        / "missingness_def_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # ========================================================
    # Grouped means
    # ========================================================

    grouped = (

        comparison
        .groupby(
            "n_available"
        )
        .agg(

            n_conditions=(
                "available",
                "size",
            ),

            mean_auc_D=(
                "auroc_D_missing_aware",
                "mean",
            ),

            mean_auc_E=(
                "auroc_E_missing_reliability",
                "mean",
            ),

            mean_auc_F=(
                "auroc_F_full",
                "mean",
            ),

            mean_pr_D=(
                "auprc_D_missing_aware",
                "mean",
            ),

            mean_pr_E=(
                "auprc_E_missing_reliability",
                "mean",
            ),

            mean_pr_F=(
                "auprc_F_full",
                "mean",
            ),

            mean_delta_auc_F_minus_E=(
                "auroc_F_minus_E",
                "mean",
            ),

            mean_delta_auc_F_minus_D=(
                "auroc_F_minus_D",
                "mean",
            ),

            mean_delta_auc_E_minus_D=(
                "auroc_E_minus_D",
                "mean",
            ),

            mean_delta_pr_F_minus_E=(
                "auprc_F_minus_E",
                "mean",
            ),

            mean_delta_pr_F_minus_D=(
                "auprc_F_minus_D",
                "mean",
            ),

            mean_delta_pr_E_minus_D=(
                "auprc_E_minus_D",
                "mean",
            ),

            mean_var_D=(
                "fused_variance_mean_D_missing_aware",
                "mean",
            ),

            mean_var_E=(
                "fused_variance_mean_E_missing_reliability",
                "mean",
            ),

            mean_var_F=(
                "fused_variance_mean_F_full",
                "mean",
            ),
        )

        .reset_index()

        .sort_values(
            "n_available",
            ascending=False,
        )
    )

    grouped_path = (
        OUT_DIR
        / "missingness_def_grouped.csv"
    )

    grouped.to_csv(
        grouped_path,
        index=False,
    )

    # ========================================================
    # Pairwise summaries
    # ========================================================

    summary_FE = summarize_contrast(
        comparison,
        contrast_name="F_minus_E",
        left_model="F_full",
        right_model="E_missing_reliability",
    )

    summary_FD = summarize_contrast(
        comparison,
        contrast_name="F_minus_D",
        left_model="F_full",
        right_model="D_missing_aware",
    )

    summary_ED = summarize_contrast(
        comparison,
        contrast_name="E_minus_D",
        left_model="E_missing_reliability",
        right_model="D_missing_aware",
    )

    # ========================================================
    # Best model per condition
    # ========================================================

    auc_columns = {

        "D_missing_aware":
            "auroc_D_missing_aware",

        "E_missing_reliability":
            "auroc_E_missing_reliability",

        "F_full":
            "auroc_F_full",
    }

    best_model_counts = {

        model: 0

        for model
        in auc_columns
    }

    best_rows = []

    for _, row in (
        comparison.iterrows()
    ):

        scores = {

            model:
                float(
                    row[
                        column
                    ]
                )

            for model, column
            in auc_columns.items()
        }

        best_model = max(
            scores,
            key=scores.get,
        )

        best_model_counts[
            best_model
        ] += 1

        best_rows.append({

            "n_available":
                int(
                    row[
                        "n_available"
                    ]
                ),

            "available":
                row[
                    "available"
                ],

            "missing":
                row[
                    "missing"
                ],

            "best_model":
                best_model,

            "best_auroc":
                scores[
                    best_model
                ],
        })

    # ========================================================
    # Consistency-specific wins
    # ========================================================

    missing_only = (
        comparison[
            comparison[
                "n_available"
            ]
            < len(
                MODALITIES
            )
        ]
        .copy()
    )

    f_improves_over_e_auc = int(
        (
            missing_only[
                "auroc_F_minus_E"
            ]
            > 0
        ).sum()
    )

    f_improves_over_d_auc = int(
        (
            missing_only[
                "auroc_F_minus_D"
            ]
            > 0
        ).sum()
    )

    # ========================================================
    # Summary JSON
    # ========================================================

    summary = {

        "task":
            "48-hour mortality",

        "evaluation_split":
            "validation",

        "test_evaluated":
            False,

        "n_conditions":
            15,

        "n_missing_conditions":
            14,

        "models": {

            name:
                config[
                    "description"
                ]

            for name, config
            in MODEL_CONFIGS.items()
        },

        "contrasts": {

            "F_minus_E":
                summary_FE,

            "F_minus_D":
                summary_FD,

            "E_minus_D":
                summary_ED,
        },

        "consistency_summary": {

            "interpretation":
                "F minus E isolates "
                "the consistency objective",

            "F_AUROC_wins_over_E_under_missingness":
                f_improves_over_e_auc,

            "F_AUROC_wins_over_D_under_missingness":
                f_improves_over_d_auc,

            "mean_F_minus_E_AUROC_missing":
                float(
                    missing_only[
                        "auroc_F_minus_E"
                    ].mean()
                ),

            "mean_F_minus_E_AUPRC_missing":
                float(
                    missing_only[
                        "auprc_F_minus_E"
                    ].mean()
                ),
        },

        "best_model_by_auroc_count":
            best_model_counts,

        "best_model_by_condition":
            best_rows,

        "grouped_by_n_available":
            grouped.to_dict(
                orient="records"
            ),
    }

    summary_path = (
        OUT_DIR
        / "missingness_def_summary.json"
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

    # ========================================================
    # Console: AUROC table
    # ========================================================

    print(
        "\n"
        + "=" * 145
    )

    print(
        "D / E / F AUROC COMPARISON"
    )

    print(
        "=" * 145
    )

    display_columns = [

        "n_available",

        "available",

        "auroc_D_missing_aware",

        "auroc_E_missing_reliability",

        "auroc_F_full",

        "auroc_F_minus_E",

        "auroc_F_minus_D",

        "auroc_E_minus_D",
    ]

    print(

        comparison[
            display_columns
        ].to_string(

            index=False,

            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # ========================================================
    # Console: grouped table
    # ========================================================

    print(
        "\n"
        + "=" * 145
    )

    print(
        "MEAN PERFORMANCE BY NUMBER "
        "OF AVAILABLE MODALITIES"
    )

    print(
        "=" * 145
    )

    print(

        grouped.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # ========================================================
    # Console: pairwise robustness
    # ========================================================

    print(
        "\n"
        + "=" * 145
    )

    print(
        "PAIRWISE ROBUSTNESS SUMMARY"
    )

    print(
        "=" * 145
    )

    for label, result in [

        (
            "F - E "
            "(isolated consistency effect)",
            summary_FE,
        ),

        (
            "F - D "
            "(full method vs missing-aware Gaussian)",
            summary_FD,
        ),

        (
            "E - D "
            "(reliability effect under missing-aware training)",
            summary_ED,
        ),
    ]:

        missing_result = (
            result[
                "missing_conditions_only"
            ]
        )

        print(
            f"\n{label}"
        )

        print(
            f"  AUROC wins   : "
            f"{missing_result['auroc_wins']}"
            f"/14"
        )

        print(
            f"  AUROC losses : "
            f"{missing_result['auroc_losses']}"
            f"/14"
        )

        print(
            f"  Mean ΔAUROC  : "
            f"{missing_result['mean_delta_auroc']:+.5f}"
        )

        print(
            f"  AUPRC wins   : "
            f"{missing_result['auprc_wins']}"
            f"/14"
        )

        print(
            f"  AUPRC losses : "
            f"{missing_result['auprc_losses']}"
            f"/14"
        )

        print(
            f"  Mean ΔAUPRC  : "
            f"{missing_result['mean_delta_auprc']:+.5f}"
        )

    # ========================================================
    # Console: best model counts
    # ========================================================

    print(
        "\n"
        + "=" * 145
    )

    print(
        "BEST MODEL BY AUROC"
    )

    print(
        "=" * 145
    )

    for model, count in (
        best_model_counts.items()
    ):

        print(
            f"{model:28s}: "
            f"{count}/15"
        )

    # ========================================================
    # Console: posterior variance
    # ========================================================

    print(
        "\n"
        + "=" * 145
    )

    print(
        "MEAN POSTERIOR VARIANCE"
    )

    print(
        "=" * 145
    )

    print(

        grouped[
            [
                "n_available",
                "mean_var_D",
                "mean_var_E",
                "mean_var_F",
            ]
        ].to_string(

            index=False,

            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\nTest set was NOT evaluated."
    )

    print(
        "\nSaved:"
    )

    print(
        json_path
    )

    print(
        flat_path
    )

    print(
        comparison_path
    )

    print(
        grouped_path
    )

    print(
        summary_path
    )


if __name__ == "__main__":

    main()