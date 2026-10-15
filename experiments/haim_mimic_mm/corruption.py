"""
HAIM corruption / reliability evaluation.

Models:
    C = Reliability PoE
        complete-modality training

    E = Reliability PoE
        + missing-aware training

    F = Reliability PoE
        + missing-aware training
        + consistency

Purpose:
    Evaluate whether learned modality reliability responds
    appropriately when ONE modality is progressively corrupted
    while the other three remain clean.

IMPORTANT:
    HAIM inputs in this project are precomputed feature /
    embedding representations.

    Therefore corruption is performed in STANDARDIZED
    REPRESENTATION SPACE.

    We are NOT claiming:
        - raw image blur
        - raw text token deletion
        - raw physiological signal corruption

Corruption:
    For target modality x:

        x_corrupt = mask * x + Gaussian noise

    where masking probability and noise standard deviation
    increase with severity.

Severity levels:

    clean:
        mask_prob = 0.00
        noise_std = 0.00

    mild:
        mask_prob = 0.10
        noise_std = 0.10

    moderate:
        mask_prob = 0.30
        noise_std = 0.30

    severe:
        mask_prob = 0.50
        noise_std = 0.50

Because features have already been standardized using TRAIN-only
statistics, Gaussian noise magnitude is comparable across feature
dimensions.

Evaluation:
    - Validation split only
    - One corrupted modality at a time
    - Other modalities remain clean
    - Deterministic corruption seeds
    - Test set untouched

Outputs:
    corruption_reliability_results.json
    corruption_reliability_results.csv
    corruption_reliability_summary.csv
    corruption_reliability_summary.json
"""

from pathlib import Path
import os
import sys
import json
import hashlib

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
    build_model_c,
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
    / "corruption_reliability"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 256

BASE_SEED = 42


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODALITIES = tuple(
    MODALITY_DIMS.keys()
)


# ============================================================
# Corruption definitions
# ============================================================

CORRUPTION_LEVELS = {

    "clean": {
        "severity":
            0,

        "mask_prob":
            0.00,

        "noise_std":
            0.00,
    },


    "mild": {
        "severity":
            1,

        "mask_prob":
            0.10,

        "noise_std":
            0.10,
    },


    "moderate": {
        "severity":
            2,

        "mask_prob":
            0.30,

        "noise_std":
            0.30,
    },


    "severe": {
        "severity":
            3,

        "mask_prob":
            0.50,

        "noise_std":
            0.50,
    },
}


# ============================================================
# Model constructors
# ============================================================

def build_model_e():

    return HAIMProbabilisticModel(
        fusion_type="reliability"
    )


def build_model_f():

    return HAIMProbabilisticModel(
        fusion_type="reliability"
    )


# ============================================================
# Model checkpoints
# ============================================================

MODEL_CONFIGS = {

    "C_reliability": {

        "builder":
            build_model_c,

        "checkpoint":
            ABC_DIR
            / "C_reliability_best.pt",

        "description":
            "Reliability PoE; complete-modality training",
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
# Deterministic corruption seed
# ============================================================

def stable_seed(
    model_name,
    modality,
    severity_name,
):
    """
    Generate deterministic seed independent of Python's
    randomized hash() behavior.
    """

    text = (
        f"{BASE_SEED}|"
        f"{model_name}|"
        f"{modality}|"
        f"{severity_name}"
    )

    digest = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()

    seed = int(
        digest[:8],
        16,
    )

    return seed


# ============================================================
# Representation-space corruption
# ============================================================

def corrupt_tensor(
    x,
    mask_prob,
    noise_std,
    generator,
):
    """
    Corrupt a standardized representation tensor.

    x:
        [batch, features]

    Step 1:
        Randomly mask features to zero.

    Step 2:
        Add zero-mean Gaussian noise.

    Clean condition returns x unchanged.
    """

    if (
        mask_prob == 0.0
        and noise_std == 0.0
    ):

        return x

    # --------------------------------------------------------
    # Feature mask
    # --------------------------------------------------------

    if mask_prob > 0:

        random_values = torch.rand(
            x.shape,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )

        keep_mask = (
            random_values
            >= mask_prob
        ).to(
            x.dtype
        )

        corrupted = (
            x
            * keep_mask
        )

    else:

        corrupted = x.clone()

    # --------------------------------------------------------
    # Gaussian perturbation
    # --------------------------------------------------------

    if noise_std > 0:

        noise = torch.randn(
            x.shape,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )

        corrupted = (
            corrupted
            + noise_std
            * noise
        )

    return corrupted


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
        f"  {checkpoint_path}"
    )

    if "epoch" in checkpoint:

        print(
            f"  epoch="
            f"{checkpoint['epoch']}"
        )

    return model


# ============================================================
# Evaluate one corruption condition
# ============================================================

@torch.no_grad()
def evaluate_corruption(
    model,
    model_name,
    loader,
    target_modality,
    severity_name,
    corruption_config,
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

    # --------------------------------------------------------
    # One deterministic generator for this full condition.
    #
    # The generator state advances batch-by-batch, giving
    # deterministic but non-identical corruption to samples.
    # --------------------------------------------------------

    seed = stable_seed(
        model_name,
        target_modality,
        severity_name,
    )

    generator = torch.Generator(
        device=DEVICE
    )

    generator.manual_seed(
        seed
    )

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

        # ----------------------------------------------------
        # Corrupt ONE modality.
        #
        # Other three remain clean.
        # ----------------------------------------------------

        corrupted_batch = {

            modality:
                value

            for modality, value
            in batch.items()
        }

        corrupted_batch[
            target_modality
        ] = corrupt_tensor(

            batch[
                target_modality
            ],

            mask_prob=
                corruption_config[
                    "mask_prob"
                ],

            noise_std=
                corruption_config[
                    "noise_std"
                ],

            generator=
                generator,
        )

        # ----------------------------------------------------
        # Forward
        #
        # Nothing is "missing" here.
        # All four modalities remain present.
        # One is corrupted.
        # ----------------------------------------------------

        output = model(
            corrupted_batch,
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

    return {

        "metrics":
            metrics,

        "variance":
            variance_summary,

        "reliability":
            reliability_summary,

        "seed":
            int(
                seed
            ),
    }


# ============================================================
# Monotonicity helpers
# ============================================================

def monotonic_decreasing(
    values,
    tolerance=1e-8,
):
    """
    True when:

        clean >= mild >= moderate >= severe
    """

    return all(

        values[i]
        >= values[i + 1]
        - tolerance

        for i
        in range(
            len(values) - 1
        )
    )


def monotonic_increasing(
    values,
    tolerance=1e-8,
):
    """
    True when:

        clean <= mild <= moderate <= severe
    """

    return all(

        values[i]
        <= values[i + 1]
        + tolerance

        for i
        in range(
            len(values) - 1
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 110
    )

    print(
        "HAIM REPRESENTATION-SPACE "
        "CORRUPTION / RELIABILITY EVALUATION"
    )

    print(
        "=" * 110
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Corruption is applied to standardized "
        "HAIM representations, not raw modalities."
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
        "Test rows frozen:",
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
    # Dataset
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
    # Print corruption definition
    # ========================================================

    print(
        "\nCorruption levels:"
    )

    for name, config in (
        CORRUPTION_LEVELS.items()
    ):

        print(
            f"  {name:10s} "
            f"mask={config['mask_prob']:.2f} "
            f"noise={config['noise_std']:.2f}"
        )

    # ========================================================
    # Evaluation
    # ========================================================

    all_results = []

    for model_name, config in (
        MODEL_CONFIGS.items()
    ):

        print(
            "\n"
            + "=" * 110
        )

        print(
            model_name.upper()
        )

        print(
            "=" * 110
        )

        model = load_model(
            model_name,
            config,
        )

        # ----------------------------------------------------
        # Corrupt one modality at a time
        # ----------------------------------------------------

        for target_modality in MODALITIES:

            print(
                f"\nTarget modality: "
                f"{target_modality}"
            )

            for severity_name, corruption in (
                CORRUPTION_LEVELS.items()
            ):

                result = evaluate_corruption(

                    model=
                        model,

                    model_name=
                        model_name,

                    loader=
                        val_loader,

                    target_modality=
                        target_modality,

                    severity_name=
                        severity_name,

                    corruption_config=
                        corruption,
                )

                metrics = (
                    result[
                        "metrics"
                    ]
                )

                variance = (
                    result[
                        "variance"
                    ]
                )

                target_reliability = (
                    result[
                        "reliability"
                    ][
                        target_modality
                    ][
                        "mean"
                    ]
                )

                row = {

                    "model":
                        model_name,

                    "target_modality":
                        target_modality,

                    "severity_name":
                        severity_name,

                    "severity":
                        int(
                            corruption[
                                "severity"
                            ]
                        ),

                    "mask_prob":
                        float(
                            corruption[
                                "mask_prob"
                            ]
                        ),

                    "noise_std":
                        float(
                            corruption[
                                "noise_std"
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

                    "target_reliability_mean":
                        target_reliability,

                    "reliability":
                        result[
                            "reliability"
                        ],

                    "corruption_seed":
                        result[
                            "seed"
                        ],
                }

                all_results.append(
                    row
                )

                print(
                    f"  {severity_name:10s} "
                    f"AUROC="
                    f"{row['auroc']:.5f} "
                    f"AUPRC="
                    f"{row['auprc']:.5f} "
                    f"Brier="
                    f"{row['brier']:.5f} "
                    f"r_target="
                    f"{row['target_reliability_mean']:.5f} "
                    f"Var="
                    f"{row['fused_variance_mean']:.5f}"
                )

    # ========================================================
    # Save detailed JSON
    # ========================================================

    full_output = {

        "task":
            "48-hour mortality",

        "evaluation_split":
            "validation",

        "test_evaluated":
            False,

        "corruption_space":
            "standardized HAIM representation space",

        "corruption_definition": {

            name:
                config

            for name, config
            in CORRUPTION_LEVELS.items()
        },

        "models": {

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
        / "corruption_reliability_results.json"
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

            "target_modality":
                row[
                    "target_modality"
                ],

            "severity_name":
                row[
                    "severity_name"
                ],

            "severity":
                row[
                    "severity"
                ],

            "mask_prob":
                row[
                    "mask_prob"
                ],

            "noise_std":
                row[
                    "noise_std"
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

            "target_reliability_mean":
                row[
                    "target_reliability_mean"
                ],

            "corruption_seed":
                row[
                    "corruption_seed"
                ],
        }

        # ----------------------------------------------------
        # Reliability of every modality
        # ----------------------------------------------------

        for modality in MODALITIES:

            flat[
                f"reliability_{modality}"
            ] = (
                row[
                    "reliability"
                ][
                    modality
                ][
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
        / "corruption_reliability_results.csv"
    )

    flat_df.to_csv(
        flat_path,
        index=False,
    )

    # ========================================================
    # Build corruption summary
    # ========================================================

    summary_rows = []

    severity_order = [
        "clean",
        "mild",
        "moderate",
        "severe",
    ]

    for model_name in MODEL_CONFIGS:

        for modality in MODALITIES:

            part = flat_df[
                (
                    flat_df[
                        "model"
                    ]
                    == model_name
                )
                &
                (
                    flat_df[
                        "target_modality"
                    ]
                    == modality
                )
            ].copy()

            part = (
                part
                .set_index(
                    "severity_name"
                )
                .loc[
                    severity_order
                ]
                .reset_index()
            )

            reliability_values = (
                part[
                    "target_reliability_mean"
                ]
                .tolist()
            )

            auc_values = (
                part[
                    "auroc"
                ]
                .tolist()
            )

            auprc_values = (
                part[
                    "auprc"
                ]
                .tolist()
            )

            variance_values = (
                part[
                    "fused_variance_mean"
                ]
                .tolist()
            )

            clean_r = (
                reliability_values[0]
            )

            severe_r = (
                reliability_values[-1]
            )

            clean_auc = (
                auc_values[0]
            )

            severe_auc = (
                auc_values[-1]
            )

            clean_pr = (
                auprc_values[0]
            )

            severe_pr = (
                auprc_values[-1]
            )

            summary_rows.append({

                "model":
                    model_name,

                "modality":
                    modality,

                "r_clean":
                    reliability_values[0],

                "r_mild":
                    reliability_values[1],

                "r_moderate":
                    reliability_values[2],

                "r_severe":
                    reliability_values[3],

                "reliability_clean_to_severe":
                    severe_r
                    - clean_r,

                "reliability_monotonic_decrease":
                    monotonic_decreasing(
                        reliability_values
                    ),

                "auc_clean":
                    auc_values[0],

                "auc_mild":
                    auc_values[1],

                "auc_moderate":
                    auc_values[2],

                "auc_severe":
                    auc_values[3],

                "auc_clean_to_severe":
                    severe_auc
                    - clean_auc,

                "auc_monotonic_decrease":
                    monotonic_decreasing(
                        auc_values
                    ),

                "auprc_clean":
                    auprc_values[0],

                "auprc_severe":
                    auprc_values[-1],

                "auprc_clean_to_severe":
                    severe_pr
                    - clean_pr,

                "variance_clean":
                    variance_values[0],

                "variance_mild":
                    variance_values[1],

                "variance_moderate":
                    variance_values[2],

                "variance_severe":
                    variance_values[3],

                "variance_clean_to_severe":
                    (
                        variance_values[-1]
                        - variance_values[0]
                    ),

                "variance_monotonic_increase":
                    monotonic_increasing(
                        variance_values
                    ),
            })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_csv_path = (
        OUT_DIR
        / "corruption_reliability_summary.csv"
    )

    summary_df.to_csv(
        summary_csv_path,
        index=False,
    )

    # ========================================================
    # Aggregate model-level summary
    # ========================================================

    model_summary = {}

    for model_name in MODEL_CONFIGS:

        part = summary_df[
            summary_df[
                "model"
            ]
            == model_name
        ]

        model_summary[
            model_name
        ] = {

            "modalities_with_monotonic_reliability_decrease":
                int(
                    part[
                        "reliability_monotonic_decrease"
                    ].sum()
                ),

            "modalities_with_monotonic_auc_decrease":
                int(
                    part[
                        "auc_monotonic_decrease"
                    ].sum()
                ),

            "modalities_with_monotonic_variance_increase":
                int(
                    part[
                        "variance_monotonic_increase"
                    ].sum()
                ),

            "mean_reliability_change_clean_to_severe":
                float(
                    part[
                        "reliability_clean_to_severe"
                    ].mean()
                ),

            "mean_auc_change_clean_to_severe":
                float(
                    part[
                        "auc_clean_to_severe"
                    ].mean()
                ),

            "mean_auprc_change_clean_to_severe":
                float(
                    part[
                        "auprc_clean_to_severe"
                    ].mean()
                ),

            "mean_variance_change_clean_to_severe":
                float(
                    part[
                        "variance_clean_to_severe"
                    ].mean()
                ),
        }

    summary_json = {

        "task":
            "48-hour mortality",

        "evaluation_split":
            "validation",

        "test_evaluated":
            False,

        "corruption_space":
            "standardized HAIM representation space",

        "corruption_levels":
            CORRUPTION_LEVELS,

        "model_summary":
            model_summary,

        "modality_summary":
            summary_rows,
    }

    summary_json_path = (
        OUT_DIR
        / "corruption_reliability_summary.json"
    )

    with open(
        summary_json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary_json,
            f,
            indent=2,
        )

    # ========================================================
    # Console: reliability table
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "TARGET-MODALITY RELIABILITY "
        "UNDER CORRUPTION"
    )

    print(
        "=" * 120
    )

    reliability_display = (
        summary_df[
            [
                "model",
                "modality",
                "r_clean",
                "r_mild",
                "r_moderate",
                "r_severe",
                "reliability_clean_to_severe",
                "reliability_monotonic_decrease",
            ]
        ]
    )

    print(

        reliability_display.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # ========================================================
    # Console: AUROC table
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "AUROC UNDER TARGET-MODALITY CORRUPTION"
    )

    print(
        "=" * 120
    )

    auc_display = (
        summary_df[
            [
                "model",
                "modality",
                "auc_clean",
                "auc_mild",
                "auc_moderate",
                "auc_severe",
                "auc_clean_to_severe",
                "auc_monotonic_decrease",
            ]
        ]
    )

    print(

        auc_display.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # ========================================================
    # Console: variance table
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FUSED VARIANCE UNDER TARGET-MODALITY CORRUPTION"
    )

    print(
        "=" * 120
    )

    variance_display = (
        summary_df[
            [
                "model",
                "modality",
                "variance_clean",
                "variance_mild",
                "variance_moderate",
                "variance_severe",
                "variance_clean_to_severe",
                "variance_monotonic_increase",
            ]
        ]
    )

    print(

        variance_display.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # ========================================================
    # Console: model summary
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "MODEL-LEVEL CORRUPTION SUMMARY"
    )

    print(
        "=" * 120
    )

    for model_name, result in (
        model_summary.items()
    ):

        print(
            f"\n{model_name}"
        )

        print(
            "  monotonic reliability decrease: "
            f"{result['modalities_with_monotonic_reliability_decrease']}"
            "/4"
        )

        print(
            "  monotonic AUROC decrease      : "
            f"{result['modalities_with_monotonic_auc_decrease']}"
            "/4"
        )

        print(
            "  monotonic variance increase   : "
            f"{result['modalities_with_monotonic_variance_increase']}"
            "/4"
        )

        print(
            "  mean clean->severe reliability: "
            f"{result['mean_reliability_change_clean_to_severe']:+.5f}"
        )

        print(
            "  mean clean->severe AUROC      : "
            f"{result['mean_auc_change_clean_to_severe']:+.5f}"
        )

        print(
            "  mean clean->severe AUPRC      : "
            f"{result['mean_auprc_change_clean_to_severe']:+.5f}"
        )

        print(
            "  mean clean->severe variance   : "
            f"{result['mean_variance_change_clean_to_severe']:+.5f}"
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
        summary_csv_path
    )

    print(
        summary_json_path
    )


if __name__ == "__main__":

    main()