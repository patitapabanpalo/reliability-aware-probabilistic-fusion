"""
HAIM fusion-method comparison.

Adapted directly from the Symile-MIMIC fusion comparison protocol.

Methods
-------
1. early_concat
2. late_average
3. mean_latent
4. gaussian_poe
5. reliability_poe
6. modality_dropout

HAIM modalities
---------------
structured
timeseries
cxr
text

Task
----
48-hour mortality prediction.

IMPORTANT
---------
This is a ONE-SEED DEVELOPMENT / IMPLEMENTATION-CHECK experiment.

    seed = 0
    validation split only
    test set NEVER evaluated

The purpose is to validate the HAIM implementations before the
final five-seed test-set Table B/C/D experiments.

Representation-space corruption
-------------------------------
HAIM inputs are precomputed representations.

Therefore corruption is performed after train-derived
standardization:

    mild:
        mask_prob = 0.10
        noise_std = 0.10

    moderate:
        mask_prob = 0.30
        noise_std = 0.30

    severe:
        mask_prob = 0.50
        noise_std = 0.50

For reliability_poe, the Symile corruption-ranking objective is
preserved conceptually:

    r_clean > r_mild > r_moderate > r_severe

using a margin-ranking loss.

Outputs
-------
results/fusion_comparison/
    haim_fusion_comparison_seed0.json
    haim_fusion_comparison_clean.csv
    haim_fusion_comparison_missingness.csv
    haim_fusion_comparison_corruption.csv
"""

from pathlib import Path
import os
import json
import random
import itertools
import hashlib

import numpy as np
import pandas as pd

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
# Paths
# ============================================================

ROOT = Path(os.environ.get("HAIM_ROOT", ".")).expanduser().resolve()
PROCESSED_DIR = Path(os.environ.get("HAIM_PROCESSED_DIR", ROOT / "processed")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("HAIM_RESULTS_ROOT", ROOT / "results")).expanduser().resolve()


DATA_DIR = PROCESSED_DIR / "model_data"


NORMALIZATION_DIR = (
    RESULTS_ROOT
    / "ablation_abc"
)


OUT_DIR = (
    RESULTS_ROOT
    / "fusion_comparison"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

SEED = 0


BATCH_SIZE = 256


# Preserve the original Symile fusion-comparison training length.
EPOCHS = 5


LEARNING_RATE = 1e-3


WEIGHT_DECAY = 0.0


GRAD_CLIP = 5.0


EMBED_DIM = 128


LATENT_DIM = 32


RELIABILITY_RANKING_WEIGHT = 0.5


RELIABILITY_MARGIN = 0.05


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODALITY_DIMS = {

    "structured":
        1542,

    "timeseries":
        451,

    "cxr":
        2084,

    "text":
        2304,
}


MODALITIES = tuple(
    MODALITY_DIMS.keys()
)


METHODS = [

    "early_concat",

    "late_average",

    "mean_latent",

    "gaussian_poe",

    "reliability_poe",

    "modality_dropout",
]


# ============================================================
# Corruption
# ============================================================

CORRUPTION_LEVELS = {

    0: {
        "name":
            "clean",

        "mask_prob":
            0.00,

        "noise_std":
            0.00,
    },

    1: {
        "name":
            "mild",

        "mask_prob":
            0.10,

        "noise_std":
            0.10,
    },

    2: {
        "name":
            "moderate",

        "mask_prob":
            0.30,

        "noise_std":
            0.30,
    },

    3: {
        "name":
            "severe",

        "mask_prob":
            0.50,

        "noise_std":
            0.50,
    },
}


CORRUPTION_SEVERITIES = [
    1,
    2,
    3,
]


BASE_CORRUPTION_SEED = 20261005


# ============================================================
# Reproducibility
# ============================================================

def set_seed(
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


        sample = {

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
            sample,
            label,
        )


# ============================================================
# Normalization
# ============================================================

def load_normalization():

    path = (
        NORMALIZATION_DIR
        / "normalization_stats.npz"
    )


    if not path.exists():

        raise FileNotFoundError(
            "Shared HAIM normalization statistics "
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
# Modality encoder
# ============================================================

class ModalityEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
    ):

        super().__init__()


        hidden_dim = 256


        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),

            nn.ReLU(),
        )


    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ============================================================
# Gaussian evidence
# ============================================================

class GaussianEvidence(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim,
    ):

        super().__init__()


        self.mu = nn.Linear(
            input_dim,
            latent_dim,
        )


        self.logvar = nn.Linear(
            input_dim,
            latent_dim,
        )


    def forward(
        self,
        x,
    ):

        mu = self.mu(
            x
        )


        logvar = (
            self.logvar(
                x
            )
            .clamp(
                -6.0,
                4.0,
            )
        )


        return {

            "mu":
                mu,

            "logvar":
                logvar,
        }


# ============================================================
# Reliability head
# ============================================================

class ReliabilityHead(nn.Module):

    def __init__(
        self,
        latent_dim,
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                2 * latent_dim,
                32,
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1,
            ),
        )


    def forward(
        self,
        mu,
        logvar,
    ):

        x = torch.cat(
            [
                mu,
                logvar,
            ],
            dim=1,
        )


        return torch.sigmoid(
            self.network(
                x
            )
        )


# ============================================================
# Prediction head
# ============================================================

class PredictionHead(nn.Module):

    def __init__(
        self,
        input_dim,
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


    def forward(
        self,
        x,
    ):

        return (
            self.network(
                x
            )
            .squeeze(
                -1
            )
        )


# ============================================================
# Fusion model
# ============================================================

class HAIMFusionModel(nn.Module):

    def __init__(
        self,
        method,
    ):

        super().__init__()


        if method not in METHODS:

            raise ValueError(
                f"Unknown method: {method}"
            )


        self.method = method


        # ----------------------------------------------------
        # Four HAIM modality encoders
        # ----------------------------------------------------

        self.encoders = nn.ModuleDict({

            modality:
                ModalityEncoder(
                    input_dim=
                        MODALITY_DIMS[
                            modality
                        ],

                    embedding_dim=
                        EMBED_DIM,
                )

            for modality
            in MODALITIES
        })


        # ----------------------------------------------------
        # Gaussian heads
        # ----------------------------------------------------

        self.gaussians = nn.ModuleDict({

            modality:
                GaussianEvidence(
                    input_dim=
                        EMBED_DIM,

                    latent_dim=
                        LATENT_DIM,
                )

            for modality
            in MODALITIES
        })


        # ----------------------------------------------------
        # Reliability heads
        # ----------------------------------------------------

        self.reliability_heads = nn.ModuleDict({

            modality:
                ReliabilityHead(
                    LATENT_DIM
                )

            for modality
            in MODALITIES
        })


        # ----------------------------------------------------
        # Prediction head
        # ----------------------------------------------------

        if method == "early_concat":

            prediction_dim = (
                EMBED_DIM
                * len(
                    MODALITIES
                )
            )

        else:

            prediction_dim = (
                LATENT_DIM
            )


        self.prediction = (
            PredictionHead(
                prediction_dim
            )
        )


        # ----------------------------------------------------
        # Late-fusion unimodal heads
        # ----------------------------------------------------

        self.late_heads = nn.ModuleDict({

            modality:
                PredictionHead(
                    EMBED_DIM
                )

            for modality
            in MODALITIES
        })


# ============================================================
# Encoding
# ============================================================

def encode_modalities(
    model,
    batch,
):

    embeddings = {}


    for modality in MODALITIES:

        embeddings[
            modality
        ] = (
            model.encoders[
                modality
            ](
                batch[
                    modality
                ]
            )
        )


    return embeddings


def encode_gaussian(
    model,
    embeddings,
):

    evidence = {}


    for modality in MODALITIES:

        gaussian = (
            model.gaussians[
                modality
            ](
                embeddings[
                    modality
                ]
            )
        )


        reliability = (
            model.reliability_heads[
                modality
            ](
                gaussian[
                    "mu"
                ],

                gaussian[
                    "logvar"
                ],
            )
        )


        evidence[
            modality
        ] = {

            "mu":
                gaussian[
                    "mu"
                ],

            "logvar":
                gaussian[
                    "logvar"
                ],

            "reliability":
                reliability,
        }


    return evidence


# ============================================================
# Fusion operators
# ============================================================

def fuse_mean(
    evidence,
    missing,
):

    available = [

        modality

        for modality
        in MODALITIES

        if modality
        not in missing
    ]


    if not available:

        raise ValueError(
            "No available modalities."
        )


    mus = torch.stack(

        [
            evidence[
                modality
            ][
                "mu"
            ]

            for modality
            in available
        ],

        dim=0,
    )


    mu = mus.mean(
        dim=0
    )


    variance = torch.ones_like(
        mu
    )


    return {

        "mu":
            mu,

        "variance":
            variance,
    }


def fuse_gaussian_poe(
    evidence,
    missing,
):
    """
    Gaussian PoE with unit Gaussian prior.
    """

    available = [

        modality

        for modality
        in MODALITIES

        if modality
        not in missing
    ]


    if not available:

        raise ValueError(
            "No available modalities."
        )


    reference = (
        evidence[
            available[0]
        ][
            "mu"
        ]
    )


    precision = torch.ones_like(
        reference
    )


    weighted_mu = torch.zeros_like(
        reference
    )


    for modality in available:

        item = (
            evidence[
                modality
            ]
        )


        variance = torch.exp(
            item[
                "logvar"
            ]
        )


        expert_precision = (
            1.0
            / variance.clamp_min(
                1e-6
            )
        )


        precision = (
            precision
            + expert_precision
        )


        weighted_mu = (
            weighted_mu
            + expert_precision
            * item[
                "mu"
            ]
        )


    fused_variance = (
        1.0
        / precision.clamp_min(
            1e-6
        )
    )


    fused_mu = (
        fused_variance
        * weighted_mu
    )


    return {

        "mu":
            fused_mu,

        "variance":
            fused_variance,
    }


def fuse_reliability_poe(
    evidence,
    missing,
):
    """
    Reliability-weighted Gaussian PoE:

        effective precision
            =
        reliability / variance
    """

    available = [

        modality

        for modality
        in MODALITIES

        if modality
        not in missing
    ]


    if not available:

        raise ValueError(
            "No available modalities."
        )


    reference = (
        evidence[
            available[0]
        ][
            "mu"
        ]
    )


    precision = torch.ones_like(
        reference
    )


    weighted_mu = torch.zeros_like(
        reference
    )


    for modality in available:

        item = (
            evidence[
                modality
            ]
        )


        variance = torch.exp(
            item[
                "logvar"
            ]
        )


        expert_precision = (
            1.0
            / variance.clamp_min(
                1e-6
            )
        )


        reliability = (
            item[
                "reliability"
            ]
        )


        effective_precision = (
            reliability
            * expert_precision
        )


        precision = (
            precision
            + effective_precision
        )


        weighted_mu = (
            weighted_mu
            + effective_precision
            * item[
                "mu"
            ]
        )


    fused_variance = (
        1.0
        / precision.clamp_min(
            1e-6
        )
    )


    fused_mu = (
        fused_variance
        * weighted_mu
    )


    return {

        "mu":
            fused_mu,

        "variance":
            fused_variance,
    }


# ============================================================
# Forward
# ============================================================

def forward_model(
    model,
    batch,
    missing=None,
):

    if missing is None:

        missing = set()


    missing = set(
        missing
    )


    embeddings = encode_modalities(
        model,
        batch,
    )


    # --------------------------------------------------------
    # Early concatenation
    # --------------------------------------------------------

    if model.method == "early_concat":

        ordered = []


        for modality in MODALITIES:

            if modality in missing:

                ordered.append(
                    torch.zeros_like(
                        embeddings[
                            modality
                        ]
                    )
                )

            else:

                ordered.append(
                    embeddings[
                        modality
                    ]
                )


        fused = torch.cat(
            ordered,
            dim=1,
        )


        logits = model.prediction(
            fused
        )


        return {

            "logits":
                logits,

            "variance":
                torch.ones_like(
                    logits
                ),

            "reliability":
                {},
        }


    # --------------------------------------------------------
    # Late probability averaging
    # --------------------------------------------------------

    if model.method == "late_average":

        modality_logits = []


        for modality in MODALITIES:

            if modality in missing:

                continue


            modality_logits.append(

                model.late_heads[
                    modality
                ](
                    embeddings[
                        modality
                    ]
                )
            )


        if not modality_logits:

            raise ValueError(
                "No available modalities."
            )


        probability = (

            torch.stack(
                [
                    torch.sigmoid(
                        logits
                    )

                    for logits
                    in modality_logits
                ],
                dim=0,
            )

            .mean(
                dim=0
            )
        )


        logits = torch.logit(
            probability.clamp(
                1e-6,
                1.0 - 1e-6,
            )
        )


        return {

            "logits":
                logits,

            "variance":
                torch.ones_like(
                    logits
                ),

            "reliability":
                {},
        }


    # --------------------------------------------------------
    # Latent methods
    # --------------------------------------------------------

    evidence = encode_gaussian(
        model,
        embeddings,
    )


    if model.method == "mean_latent":

        fused = fuse_mean(
            evidence,
            missing,
        )


    elif model.method == "gaussian_poe":

        fused = fuse_gaussian_poe(
            evidence,
            missing,
        )


    elif model.method == "reliability_poe":

        fused = fuse_reliability_poe(
            evidence,
            missing,
        )


    elif model.method == "modality_dropout":

        # Symile definition:
        # mean latent fusion, trained with random missingness.
        fused = fuse_mean(
            evidence,
            missing,
        )


    else:

        raise ValueError(
            f"Unknown method: "
            f"{model.method}"
        )


    logits = model.prediction(
        fused[
            "mu"
        ]
    )


    reliability = {}


    for modality in MODALITIES:

        if modality not in missing:

            reliability[
                modality
            ] = (
                evidence[
                    modality
                ][
                    "reliability"
                ]
                .squeeze(
                    -1
                )
            )


    return {

        "logits":
            logits,

        "variance":
            fused[
                "variance"
            ].mean(
                dim=1
            ),

        "reliability":
            reliability,
    }


# ============================================================
# Corruption helpers
# ============================================================

def stable_corruption_seed(
    modality,
    severity,
    context,
):

    text = (
        f"{BASE_CORRUPTION_SEED}|"
        f"{SEED}|"
        f"{modality}|"
        f"{severity}|"
        f"{context}"
    )


    digest = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


    return int(
        digest[:8],
        16,
    )


def corrupt_tensor(
    x,
    severity,
    generator=None,
):

    config = (
        CORRUPTION_LEVELS[
            severity
        ]
    )


    mask_prob = (
        config[
            "mask_prob"
        ]
    )


    noise_std = (
        config[
            "noise_std"
        ]
    )


    if severity == 0:

        return x


    if generator is None:

        random_values = (
            torch.rand_like(
                x
            )
        )

        noise = (
            torch.randn_like(
                x
            )
        )

    else:

        random_values = torch.rand(

            x.shape,

            dtype=
                x.dtype,

            device=
                x.device,

            generator=
                generator,
        )


        noise = torch.randn(

            x.shape,

            dtype=
                x.dtype,

            device=
                x.device,

            generator=
                generator,
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


    corrupted = (
        corrupted
        + noise_std
        * noise
    )


    return corrupted


def build_corrupted_batch(
    batch,
    target_modality,
    severity,
    generator=None,
):

    local = {

        modality:
            value

        for modality, value
        in batch.items()
    }


    local[
        target_modality
    ] = corrupt_tensor(

        batch[
            target_modality
        ],

        severity=
            severity,

        generator=
            generator,
    )


    return local


# ============================================================
# Reliability ranking
# ============================================================

def reliability_ranking_loss(
    values,
    margin=RELIABILITY_MARGIN,
):
    """
    Preserve the Symile ranking objective:

        clean > mild > moderate > severe
    """

    total = torch.tensor(
        0.0,
        device=DEVICE,
    )


    for index in range(
        len(
            values
        )
        - 1,
    ):

        total = (

            total

            + torch.relu(

                margin

                - values[
                    index
                ]

                + values[
                    index + 1
                ]
            ).mean()
        )


    return total


def reliability_supervision_loss(
    model,
    batch,
):
    """
    Symile-style corruption-ranking supervision adapted to
    HAIM standardized representations.

    Every modality receives:

        clean
        mild
        moderate
        severe

    reliability evaluations.
    """

    modality_losses = []


    for modality in MODALITIES:

        level_values = []


        for severity in [
            0,
            1,
            2,
            3,
        ]:

            if severity == 0:

                local_batch = batch

            else:

                local_batch = (
                    build_corrupted_batch(

                        batch,

                        target_modality=
                            modality,

                        severity=
                            severity,
                    )
                )


            embeddings = encode_modalities(
                model,
                local_batch,
            )


            evidence = encode_gaussian(
                model,
                embeddings,
            )


            level_values.append(

                evidence[
                    modality
                ][
                    "reliability"
                ]
                .squeeze(
                    -1
                )
            )


        modality_losses.append(

            reliability_ranking_loss(
                level_values
            )
        )


    return sum(
        modality_losses
    )


# ============================================================
# Modality-dropout training pattern
# ============================================================

def sample_modality_dropout_pattern():
    """
    Preserve the original Symile definition:

    choose between:
        complete
        exactly one missing modality

    Extended naturally from 3 to 4 HAIM modalities.
    """

    patterns = [
        set(),
    ]


    for modality in MODALITIES:

        patterns.append(
            {
                modality
            }
        )


    return random.choice(
        patterns
    )


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    loader,
    method,
    pos_weight,
):

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=
            LEARNING_RATE,

        weight_decay=
            WEIGHT_DECAY,
    )


    criterion = nn.BCEWithLogitsLoss(

        pos_weight=torch.tensor(

            pos_weight,

            dtype=
                torch.float32,

            device=
                DEVICE,
        )
    )


    history = []


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()


        epoch_total = 0.0

        epoch_prediction = 0.0

        epoch_reliability = 0.0

        n_seen = 0


        for batch, labels in loader:

            batch = {

                modality:
                    value.to(
                        DEVICE,
                        non_blocking=True,
                    )

                for modality, value
                in batch.items()
            }


            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            # ------------------------------------------------
            # Modality dropout only affects this baseline.
            # ------------------------------------------------

            if method == "modality_dropout":

                missing = (
                    sample_modality_dropout_pattern()
                )

            else:

                missing = set()


            output = forward_model(
                model,
                batch,
                missing=missing,
            )


            prediction_loss = criterion(
                output[
                    "logits"
                ],
                labels,
            )


            total_loss = (
                prediction_loss
            )


            reliability_loss_value = (
                torch.tensor(
                    0.0,
                    device=DEVICE,
                )
            )


            # ------------------------------------------------
            # Preserve Symile reliability-PoE supervision.
            # ------------------------------------------------

            if method == "reliability_poe":

                reliability_loss_value = (
                    reliability_supervision_loss(
                        model,
                        batch,
                    )
                )


                total_loss = (

                    total_loss

                    + RELIABILITY_RANKING_WEIGHT

                    * reliability_loss_value
                )


            total_loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=
                    GRAD_CLIP,
            )


            optimizer.step()


            batch_n = (
                labels.shape[
                    0
                ]
            )


            epoch_total += (
                total_loss.item()
                * batch_n
            )


            epoch_prediction += (
                prediction_loss.item()
                * batch_n
            )


            epoch_reliability += (
                reliability_loss_value
                .detach()
                .item()
                * batch_n
            )


            n_seen += (
                batch_n
            )


        epoch_result = {

            "epoch":
                int(
                    epoch
                ),

            "total_loss":
                float(
                    epoch_total
                    / n_seen
                ),

            "prediction_loss":
                float(
                    epoch_prediction
                    / n_seen
                ),

            "reliability_loss":
                float(
                    epoch_reliability
                    / n_seen
                ),
        }


        history.append(
            epoch_result
        )


        print(
            f"    epoch={epoch:02d} "
            f"loss="
            f"{epoch_result['total_loss']:.6f} "
            f"pred="
            f"{epoch_result['prediction_loss']:.6f} "
            f"rel="
            f"{epoch_result['reliability_loss']:.6f}"
        )


    return history


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_condition(
    model,
    loader,
    missing=None,
    corruption_modality=None,
    corruption_severity=0,
):

    if missing is None:

        missing = set()


    model.eval()


    labels_all = []

    probabilities_all = []

    variances_all = []


    reliability_values = {

        modality: []

        for modality
        in MODALITIES
    }


    generator = None


    if (
        corruption_modality is not None
        and corruption_severity > 0
    ):

        corruption_seed = (
            stable_corruption_seed(

                corruption_modality,

                corruption_severity,

                context=
                    "validation",
            )
        )


        generator = torch.Generator(
            device=DEVICE
        )


        generator.manual_seed(
            corruption_seed
        )


    for batch, labels in loader:

        batch = {

            modality:
                value.to(
                    DEVICE,
                    non_blocking=True,
                )

            for modality, value
            in batch.items()
        }


        if (
            corruption_modality is not None
            and corruption_severity > 0
        ):

            batch = build_corrupted_batch(

                batch,

                target_modality=
                    corruption_modality,

                severity=
                    corruption_severity,

                generator=
                    generator,
            )


        output = forward_model(
            model,
            batch,
            missing=missing,
        )


        probabilities = torch.sigmoid(
            output[
                "logits"
            ]
        )


        labels_all.append(
            labels.numpy()
        )


        probabilities_all.append(
            probabilities
            .cpu()
            .numpy()
        )


        variances_all.append(
            output[
                "variance"
            ]
            .cpu()
            .numpy()
        )


        for modality, values in (
            output[
                "reliability"
            ].items()
        ):

            reliability_values[
                modality
            ].append(
                values
                .cpu()
                .numpy()
            )


    y = np.concatenate(
        labels_all
    )


    p = np.concatenate(
        probabilities_all
    )


    variance = np.concatenate(
        variances_all
    )


    result = {

        "n":
            int(
                len(
                    y
                )
            ),

        "positive":
            int(
                y.sum()
            ),

        "AUROC":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "AUPRC":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),

        "Brier":
            float(
                brier_score_loss(
                    y,
                    p,
                )
            ),

        "variance":
            float(
                variance.mean()
            ),

        "reliability":
            {},
    }


    for modality in MODALITIES:

        if reliability_values[
            modality
        ]:

            values = np.concatenate(
                reliability_values[
                    modality
                ]
            )


            result[
                "reliability"
            ][
                modality
            ] = float(
                values.mean()
            )


    return result


# ============================================================
# Missingness patterns
# ============================================================

def build_missingness_conditions():
    """
    All 15 non-empty availability configurations.
    """

    conditions = {}


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


            if len(
                missing
            ) == 0:

                name = (
                    "all modalities"
                )

            else:

                name = (
                    "missing:"
                    + "+".join(
                        missing
                    )
                )


            conditions[
                name
            ] = set(
                missing
            )


    return conditions


# ============================================================
# Run one method
# ============================================================

def run_method(
    method,
    train_loader,
    val_loader,
    pos_weight,
):

    print(
        "\n"
        + "=" * 100
    )


    print(
        f"METHOD: {method}"
    )


    print(
        "=" * 100
    )


    set_seed(
        SEED
    )


    model = (
        HAIMFusionModel(
            method
        )
        .to(
            DEVICE
        )
    )


    parameter_count = sum(

        parameter.numel()

        for parameter
        in model.parameters()
    )


    print(
        f"  Parameters: "
        f"{parameter_count:,}"
    )


    history = train_model(

        model=
            model,

        loader=
            train_loader,

        method=
            method,

        pos_weight=
            pos_weight,
    )


    # --------------------------------------------------------
    # Clean validation
    # --------------------------------------------------------

    clean = evaluate_condition(

        model,

        val_loader,

        missing=set(),
    )


    print(
        "\n  Clean validation:"
    )


    print(
        f"    AUROC="
        f"{clean['AUROC']:.5f} "
        f"AUPRC="
        f"{clean['AUPRC']:.5f} "
        f"Brier="
        f"{clean['Brier']:.5f}"
    )


    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------

    missingness = {}


    conditions = (
        build_missingness_conditions()
    )


    for condition_name, missing in (
        conditions.items()
    ):

        result = evaluate_condition(

            model,

            val_loader,

            missing=
                missing,
        )


        missingness[
            condition_name
        ] = result


    # --------------------------------------------------------
    # Corruption
    # --------------------------------------------------------

    corruption = {

        modality: {}

        for modality
        in MODALITIES
    }


    for modality in MODALITIES:

        for severity in (
            CORRUPTION_SEVERITIES
        ):

            result = evaluate_condition(

                model,

                val_loader,

                missing=set(),

                corruption_modality=
                    modality,

                corruption_severity=
                    severity,
            )


            corruption[
                modality
            ][
                str(
                    severity
                )
            ] = result


    return {

        "method":
            method,

        "seed":
            SEED,

        "parameter_count":
            int(
                parameter_count
            ),

        "history":
            history,

        "clean":
            clean,

        "missingness":
            missingness,

        "corruption":
            corruption,
    }


# ============================================================
# CSV builders
# ============================================================

def build_clean_rows(
    results,
):

    rows = []


    for method, item in (
        results.items()
    ):

        clean = (
            item[
                "clean"
            ]
        )


        rows.append({

            "method":
                method,

            "seed":
                SEED,

            "AUROC":
                clean[
                    "AUROC"
                ],

            "AUPRC":
                clean[
                    "AUPRC"
                ],

            "Brier":
                clean[
                    "Brier"
                ],

            "variance":
                clean[
                    "variance"
                ],

            "parameters":
                item[
                    "parameter_count"
                ],
        })


    return rows


def build_missingness_rows(
    results,
):

    rows = []


    for method, item in (
        results.items()
    ):

        for condition, result in (
            item[
                "missingness"
            ].items()
        ):

            if condition == "all modalities":

                n_missing = 0

            else:

                missing_text = (
                    condition
                    .replace(
                        "missing:",
                        "",
                    )
                )


                n_missing = len(
                    missing_text.split(
                        "+"
                    )
                )


            rows.append({

                "method":
                    method,

                "seed":
                    SEED,

                "condition":
                    condition,

                "n_missing":
                    n_missing,

                "n_available":
                    (
                        len(
                            MODALITIES
                        )
                        - n_missing
                    ),

                "AUROC":
                    result[
                        "AUROC"
                    ],

                "AUPRC":
                    result[
                        "AUPRC"
                    ],

                "Brier":
                    result[
                        "Brier"
                    ],

                "variance":
                    result[
                        "variance"
                    ],
            })


    return rows


def build_corruption_rows(
    results,
):

    rows = []


    for method, item in (
        results.items()
    ):

        clean_auc = (
            item[
                "clean"
            ][
                "AUROC"
            ]
        )


        for modality in MODALITIES:

            for severity in (
                CORRUPTION_SEVERITIES
            ):

                result = (
                    item[
                        "corruption"
                    ][
                        modality
                    ][
                        str(
                            severity
                        )
                    ]
                )


                rows.append({

                    "method":
                        method,

                    "seed":
                        SEED,

                    "modality":
                        modality,

                    "severity":
                        severity,

                    "severity_name":
                        CORRUPTION_LEVELS[
                            severity
                        ][
                            "name"
                        ],

                    "mask_prob":
                        CORRUPTION_LEVELS[
                            severity
                        ][
                            "mask_prob"
                        ],

                    "noise_std":
                        CORRUPTION_LEVELS[
                            severity
                        ][
                            "noise_std"
                        ],

                    "AUROC":
                        result[
                            "AUROC"
                        ],

                    "AUPRC":
                        result[
                            "AUPRC"
                        ],

                    "Brier":
                        result[
                            "Brier"
                        ],

                    "variance":
                        result[
                            "variance"
                        ],

                    "AUROC_delta_from_clean":
                        (
                            result[
                                "AUROC"
                            ]
                            - clean_auc
                        ),
                })


    return rows


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        SEED
    )


    print(
        "=" * 100
    )


    print(
        "HAIM FUSION COMPARISON"
    )


    print(
        "=" * 100
    )


    print(
        "Task: 48-hour mortality"
    )


    print(
        "Seed:",
        SEED,
    )


    print(
        "Device:",
        DEVICE,
    )


    print(
        "Epochs:",
        EPOCHS,
    )


    print(
        "Test evaluation: DISABLED"
    )


    print(
        "\nMethods:"
    )


    for method in METHODS:

        print(
            " ",
            method,
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
        "\nDataset:"
    )


    print(
        "  total:",
        len(
            y
        ),
    )


    print(
        "  train:",
        int(
            train_mask.sum()
        ),
    )


    print(
        "  validation:",
        int(
            val_mask.sum()
        ),
    )


    print(
        "  test frozen:",
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
        "Normalization complete."
    )


    # ========================================================
    # Datasets
    # ========================================================

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


    # ========================================================
    # Class weighting
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
    # Run methods
    # ========================================================

    all_results = {}


    for method_index, method in enumerate(
        METHODS
    ):

        # ----------------------------------------------------
        # Reset seed before creating each method.
        #
        # This follows the Symile comparison idea of a common
        # experiment seed rather than allowing method order to
        # determine initialization randomness.
        # ----------------------------------------------------

        set_seed(
            SEED
        )


        train_generator = (
            torch.Generator()
        )


        train_generator.manual_seed(
            SEED
        )


        train_loader = DataLoader(

            train_dataset,

            batch_size=
                BATCH_SIZE,

            shuffle=
                True,

            num_workers=
                0,

            pin_memory=
                torch.cuda.is_available(),

            generator=
                train_generator,
        )


        val_loader = DataLoader(

            val_dataset,

            batch_size=
                BATCH_SIZE,

            shuffle=
                False,

            num_workers=
                0,

            pin_memory=
                torch.cuda.is_available(),
        )


        result = run_method(

            method=
                method,

            train_loader=
                train_loader,

            val_loader=
                val_loader,

            pos_weight=
                pos_weight,
        )


        all_results[
            method
        ] = result


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # ========================================================
    # Build CSV tables
    # ========================================================

    clean_rows = (
        build_clean_rows(
            all_results
        )
    )


    missingness_rows = (
        build_missingness_rows(
            all_results
        )
    )


    corruption_rows = (
        build_corruption_rows(
            all_results
        )
    )


    clean_df = pd.DataFrame(
        clean_rows
    )


    missingness_df = pd.DataFrame(
        missingness_rows
    )


    corruption_df = pd.DataFrame(
        corruption_rows
    )


    # ========================================================
    # Sort
    # ========================================================

    clean_df = (
        clean_df
        .sort_values(
            "AUROC",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    missingness_df = (
        missingness_df
        .sort_values(
            [
                "method",
                "n_available",
                "condition",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    corruption_df = (
        corruption_df
        .sort_values(
            [
                "method",
                "modality",
                "severity",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Save CSV
    # ========================================================

    clean_path = (
        OUT_DIR
        / "haim_fusion_comparison_clean.csv"
    )


    missingness_path = (
        OUT_DIR
        / "haim_fusion_comparison_missingness.csv"
    )


    corruption_path = (
        OUT_DIR
        / "haim_fusion_comparison_corruption.csv"
    )


    clean_df.to_csv(
        clean_path,
        index=False,
    )


    missingness_df.to_csv(
        missingness_path,
        index=False,
    )


    corruption_df.to_csv(
        corruption_path,
        index=False,
    )


    # ========================================================
    # JSON
    # ========================================================

    output = {

        "experiment":
            "HAIM one-seed fusion comparison",

        "source_protocol":
            (
                "Adapted from Symile-MIMIC "
                "run_fusion_comparisons.py"
            ),

        "task":
            "48-hour mortality",

        "seed":
            SEED,

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "device":
            str(
                DEVICE
            ),

        "test_evaluated":
            False,

        "modalities":
            list(
                MODALITIES
            ),

        "methods":
            METHODS,

        "method_definitions": {

            "early_concat":
                (
                    "Concatenate four HAIM modality embeddings "
                    "and predict with an MLP. Missing modality "
                    "embeddings are zero-filled."
                ),

            "late_average":
                (
                    "One classifier per modality; average "
                    "available-modality predicted probabilities."
                ),

            "mean_latent":
                (
                    "Map each modality to a Gaussian latent "
                    "representation and average available "
                    "modality means."
                ),

            "gaussian_poe":
                (
                    "Precision-weighted Gaussian "
                    "Product-of-Experts with unit prior."
                ),

            "reliability_poe":
                (
                    "Gaussian PoE with effective precision "
                    "reliability / variance and Symile-style "
                    "corruption-ranking supervision."
                ),

            "modality_dropout":
                (
                    "Mean latent fusion trained with either "
                    "all modalities or one randomly missing "
                    "modality."
                ),
        },

        "corruption_definition": {

            str(
                severity
            ):
                config

            for severity, config
            in CORRUPTION_LEVELS.items()
        },

        "results":
            all_results,
    }


    json_path = (
        OUT_DIR
        / "haim_fusion_comparison_seed0.json"
    )


    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )


    # ========================================================
    # Console clean table
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )


    print(
        "CLEAN VALIDATION COMPARISON"
    )


    print(
        "=" * 100
    )


    print(

        clean_df[
            [
                "method",
                "AUROC",
                "AUPRC",
                "Brier",
                "variance",
                "parameters",
            ]
        ].to_string(

            index=False,

            float_format=lambda x:
                f"{x:.5f}",
        )
    )


    # ========================================================
    # Missingness aggregate
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )


    print(
        "MEAN OVER 14 MISSING-MODALITY CONDITIONS"
    )


    print(
        "=" * 100
    )


    incomplete = (
        missingness_df[
            missingness_df[
                "n_missing"
            ]
            > 0
        ]
    )


    missing_summary = (

        incomplete
        .groupby(
            "method"
        )
        .agg(

            AUROC=(
                "AUROC",
                "mean",
            ),

            AUPRC=(
                "AUPRC",
                "mean",
            ),

            Brier=(
                "Brier",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "AUROC",
            ascending=False,
        )
    )


    print(

        missing_summary.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.5f}",
        )
    )


    # ========================================================
    # Corruption aggregate
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )


    print(
        "MEAN OVER 12 CORRUPTED CONDITIONS"
    )


    print(
        "=" * 100
    )


    corruption_summary = (

        corruption_df
        .groupby(
            "method"
        )
        .agg(

            AUROC=(
                "AUROC",
                "mean",
            ),

            AUPRC=(
                "AUPRC",
                "mean",
            ),

            Brier=(
                "Brier",
                "mean",
            ),

            mean_AUROC_delta=(
                "AUROC_delta_from_clean",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "AUROC",
            ascending=False,
        )
    )


    print(

        corruption_summary.to_string(

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
        clean_path
    )


    print(
        missingness_path
    )


    print(
        corruption_path
    )


    print(
        "\n"
        + "=" * 100
    )


    print(
        "HAIM FUSION COMPARISON COMPLETE"
    )


    print(
        "=" * 100
    )


if __name__ == "__main__":

    main()