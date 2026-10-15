from pathlib import Path
import os
import sys
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_BASE = Path(os.environ.get("SYMILE_DATA_ROOT", ROOT)).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("SYMILE_OUTPUT_ROOT", ROOT / "experiment_results")).expanduser().resolve()


from datasets.symile_mimic.dataset import SymileMIMICDataset
from reliability_fusion.encoders.symile import LabEncoder


DATA_ROOT = DATA_BASE / "data_npy"

TRAIN_CSV = DATA_BASE / "train.csv"
VAL_CSV = DATA_BASE / "val.csv"

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

RESULTS_DIR = OUTPUT_ROOT
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = (
    RESULTS_DIR
    / "fusion_comparison_results.json"
)


# =====================================================================
# CONFIGURATION
# =====================================================================

LABEL_NAME = "Cardiomegaly"

SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

BATCH_SIZE = 16

EPOCHS = 5

LEARNING_RATE = 1e-3

EMBED_DIM = 128
LATENT_DIM = 32

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


METHODS = [
    "early_concat",
    "late_average",
    "mean_latent",
    "gaussian_poe",
    "reliability_poe",
    "modality_dropout",
]


MISSING_CONDITIONS = {
    "all modalities": set(),
    "CXR missing": {"CXR"},
    "ECG missing": {"ECG"},
    "Labs missing": {"Labs"},
    "CXR + ECG missing": {"CXR", "ECG"},
    "CXR + Labs missing": {"CXR", "Labs"},
    "ECG + Labs missing": {"ECG", "Labs"},
}


CORRUPTION_MODALITIES = [
    "CXR",
    "ECG",
    "Labs",
]

CORRUPTION_SEVERITIES = [
    1,
    2,
    3,
]


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# DATASET
# =====================================================================

class LabeledDataset(Dataset):

    def __init__(
        self,
        tensor_dataset,
        csv_path,
    ):

        self.tensor_dataset = tensor_dataset

        df = pd.read_csv(
            csv_path
        )

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
                "CSV/tensor hadm_id alignment failed."
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
            labels[
                self.indices
            ],
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
        ].copy()

        return {
            "cxr": sample["cxr"].float(),

            "ecg": sample["ecg"].float(),

            "labs_percentiles":
                sample[
                    "labs_percentiles"
                ].float(),

            "labs_missingness":
                sample[
                    "labs_missingness"
                ].float(),

            "hadm_id":
                sample["hadm_id"],

            "label":
                self.labels[index],
        }


# =====================================================================
# ENCODERS
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
                7,
                stride=2,
                padding=3,
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(
                3,
                stride=2,
                padding=1,
            ),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1,
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1,
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                128,
                3,
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
                15,
                stride=2,
                padding=7,
            ),

            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.MaxPool1d(4),

            nn.Conv1d(
                32,
                64,
                9,
                stride=2,
                padding=4,
            ),

            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(4),

            nn.Conv1d(
                64,
                128,
                7,
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

        x = x.squeeze(1)

        x = x.transpose(
            1,
            2,
        )

        x = self.features(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        return self.projection(x)


# =====================================================================
# GAUSSIAN + RELIABILITY
# =====================================================================

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

    def forward(self, x):

        mu = self.mu(x)

        logvar = self.logvar(
            x
        ).clamp(
            -6.0,
            4.0,
        )

        return {
            "mu": mu,
            "logvar": logvar,
        }


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
            self.network(x)
        )


# =====================================================================
# PREDICTION HEADS
# =====================================================================

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

    def forward(self, x):

        return self.network(
            x
        ).squeeze(-1)


# =====================================================================
# MODEL CONTAINER
# =====================================================================

class FusionModel(nn.Module):

    def __init__(
        self,
        method,
    ):

        super().__init__()

        self.method = method

        self.cxr_encoder = CXREncoder(
            EMBED_DIM
        )

        self.ecg_encoder = ECGEncoder(
            EMBED_DIM
        )

        self.lab_encoder = LabEncoder(
            input_dim=100,
            embedding_dim=EMBED_DIM,
        )

        # Gaussian modules are used for Gaussian fusion methods.
        self.cxr_gaussian = GaussianEvidence(
            EMBED_DIM,
            LATENT_DIM,
        )

        self.ecg_gaussian = GaussianEvidence(
            EMBED_DIM,
            LATENT_DIM,
        )

        self.lab_gaussian = GaussianEvidence(
            EMBED_DIM,
            LATENT_DIM,
        )

        self.cxr_reliability = ReliabilityHead(
            LATENT_DIM
        )

        self.ecg_reliability = ReliabilityHead(
            LATENT_DIM
        )

        self.lab_reliability = ReliabilityHead(
            LATENT_DIM
        )

        if method == "early_concat":

            self.prediction = PredictionHead(
                EMBED_DIM * 3
            )

        else:

            self.prediction = PredictionHead(
                LATENT_DIM
            )

        self.cxr_classifier = PredictionHead(
            EMBED_DIM
        )

        self.ecg_classifier = PredictionHead(
            EMBED_DIM
        )

        self.lab_classifier = PredictionHead(
            EMBED_DIM
        )

        self.late_heads = nn.ModuleDict(
            {
                "CXR":
                    PredictionHead(
                        EMBED_DIM
                    ),

                "ECG":
                    PredictionHead(
                        EMBED_DIM
                    ),

                "Labs":
                    PredictionHead(
                        EMBED_DIM
                    ),
            }
        )


# =====================================================================
# CORRUPTIONS
# =====================================================================

def corrupt_cxr(
    x,
    severity,
):

    if severity == 0:
        return x

    kernel = {
        1: 3,
        2: 7,
        3: 13,
    }[severity]

    blurred = F.avg_pool2d(
        x,
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )

    strength = {
        1: 0.25,
        2: 0.55,
        3: 0.85,
    }[severity]

    noise = {
        1: 0.02,
        2: 0.05,
        3: 0.10,
    }[severity]

    return (
        (1.0 - strength) * x
        + strength * blurred
        + noise * torch.randn_like(x)
    )


def corrupt_ecg(
    x,
    severity,
):

    if severity == 0:
        return x

    strength = {
        1: 0.05,
        2: 0.15,
        3: 0.30,
    }[severity]

    return (
        x
        + strength
        * torch.randn_like(x)
    ).clamp(
        -1.0,
        1.0,
    )


def corrupt_labs(
    x,
    severity,
):

    if severity == 0:
        return x

    alpha = {
        1: 0.25,
        2: 0.60,
        3: 1.00,
    }[severity]

    random_values = torch.rand_like(x)

    return (
        (1.0 - alpha) * x
        + alpha * random_values
    )


# =====================================================================
# ENCODING
# =====================================================================

def encode_modalities(
    model,
    cxr,
    ecg,
    labs,
):

    cxr_embedding = (
        model.cxr_encoder(
            cxr
        )
    )

    ecg_embedding = (
        model.ecg_encoder(
            ecg
        )
    )

    lab_embedding = (
        model.lab_encoder(
            labs
        )
    )

    return {
        "CXR": cxr_embedding,
        "ECG": ecg_embedding,
        "Labs": lab_embedding,
    }


def encode_gaussian(
    model,
    embeddings,
):

    cxr_ev = (
        model.cxr_gaussian(
            embeddings["CXR"]
        )
    )

    ecg_ev = (
        model.ecg_gaussian(
            embeddings["ECG"]
        )
    )

    lab_ev = (
        model.lab_gaussian(
            embeddings["Labs"]
        )
    )

    cxr_ev["reliability"] = (
        model.cxr_reliability(
            cxr_ev["mu"],
            cxr_ev["logvar"],
        )
    )

    ecg_ev["reliability"] = (
        model.ecg_reliability(
            ecg_ev["mu"],
            ecg_ev["logvar"],
        )
    )

    lab_ev["reliability"] = (
        model.lab_reliability(
            lab_ev["mu"],
            lab_ev["logvar"],
        )
    )

    return {
        "CXR": cxr_ev,
        "ECG": ecg_ev,
        "Labs": lab_ev,
    }


# =====================================================================
# FUSION
# =====================================================================

def available_embeddings(
    embeddings,
    missing,
):

    return [
        embeddings[name]
        for name in [
            "CXR",
            "ECG",
            "Labs",
        ]
        if name not in missing
    ]


def available_evidence(
    evidence,
    missing,
):

    return [
        evidence[name]
        for name in [
            "CXR",
            "ECG",
            "Labs",
        ]
        if name not in missing
    ]


def fuse_mean(
    evidence,
    missing,
):

    selected = available_evidence(
        evidence,
        missing,
    )

    if not selected:

        raise ValueError(
            "No modalities available."
        )

    mu = torch.stack(
        [
            item["mu"]
            for item in selected
        ],
        dim=0,
    ).mean(
        dim=0
    )

    return {
        "mu": mu,
        "variance": torch.ones_like(mu),
    }


def fuse_gaussian_poe(
    evidence,
    missing,
):

    selected = available_evidence(
        evidence,
        missing,
    )

    if not selected:

        raise ValueError(
            "No modalities available."
        )

    batch_size, latent_dim = (
        selected[0]["mu"].shape
    )

    device = selected[0]["mu"].device

    dtype = selected[0]["mu"].dtype

    precision = torch.ones(
        batch_size,
        latent_dim,
        device=device,
        dtype=dtype,
    )

    weighted_mu = torch.zeros(
        batch_size,
        latent_dim,
        device=device,
        dtype=dtype,
    )

    for expert in selected:

        variance = torch.exp(
            expert["logvar"]
        ).clamp_min(
            1e-6
        )

        expert_precision = (
            1.0 / variance
        )

        precision += (
            expert_precision
        )

        weighted_mu += (
            expert_precision
            * expert["mu"]
        )

    variance = (
        1.0
        / precision.clamp_min(
            1e-6
        )
    )

    mu = (
        variance
        * weighted_mu
    )

    return {
        "mu": mu,
        "variance": variance,
    }


def fuse_reliability_poe(
    evidence,
    missing,
):

    selected = available_evidence(
        evidence,
        missing,
    )

    if not selected:

        raise ValueError(
            "No modalities available."
        )

    batch_size, latent_dim = (
        selected[0]["mu"].shape
    )

    device = selected[0]["mu"].device

    dtype = selected[0]["mu"].dtype

    precision = torch.ones(
        batch_size,
        latent_dim,
        device=device,
        dtype=dtype,
    )

    weighted_mu = torch.zeros(
        batch_size,
        latent_dim,
        device=device,
        dtype=dtype,
    )

    for expert in selected:

        variance = torch.exp(
            expert["logvar"]
        ).clamp_min(
            1e-6
        )

        expert_precision = (
            1.0 / variance
        )

        reliability = expert[
            "reliability"
        ].clamp(
            0.0,
            1.0,
        )

        effective_precision = (
            reliability
            * expert_precision
        )

        precision += (
            effective_precision
        )

        weighted_mu += (
            effective_precision
            * expert["mu"]
        )

    variance = (
        1.0
        / precision.clamp_min(
            1e-6
        )
    )

    mu = (
        variance
        * weighted_mu
    )

    return {
        "mu": mu,
        "variance": variance,
    }


# =====================================================================
# FORWARD PASS
# =====================================================================

def forward_model(
    model,
    batch,
    missing=set(),
):

    cxr = batch["cxr"].to(
        DEVICE
    )

    ecg = batch["ecg"].to(
        DEVICE
    )

    labs = torch.cat(
        [
            batch[
                "labs_percentiles"
            ].to(DEVICE),

            batch[
                "labs_missingness"
            ].to(DEVICE),
        ],
        dim=1,
    )

    embeddings = encode_modalities(
        model,
        cxr,
        ecg,
        labs,
    )

    # ---------------------------------------------------------------
    # Early concatenation
    # ---------------------------------------------------------------

    if model.method == "early_concat":

        selected = available_embeddings(
            embeddings,
            missing,
        )

        # Keep the dimensionality fixed by zero filling missing
        # modality embeddings.
        zero = torch.zeros_like(
            embeddings["CXR"]
        )

        ordered = []

        for name in [
            "CXR",
            "ECG",
            "Labs",
        ]:

            if name in missing:

                ordered.append(
                    zero
                )

            else:

                ordered.append(
                    embeddings[name]
                )

        fused = torch.cat(
            ordered,
            dim=1,
        )

        logits = model.prediction(
            fused
        )

        return {
            "logits": logits,
            "variance": torch.ones_like(
                logits
            ),
            "embeddings": embeddings,
            "evidence": None,
            "reliability": {},
        }

    # ---------------------------------------------------------------
    # Late average
    # ---------------------------------------------------------------

    if model.method == "late_average":

        logits = []

        for name in [
            "CXR",
            "ECG",
            "Labs",
        ]:

            if name in missing:
                continue

            logits.append(
                model.late_heads[name](
                    embeddings[name]
                )
            )

        if not logits:

            raise ValueError(
                "No available modalities."
            )

        fused_probability = torch.stack(
            [
                torch.sigmoid(x)
                for x in logits
            ],
            dim=0,
        ).mean(
            dim=0
        )

        return {
            "logits":
                torch.logit(
                    fused_probability.clamp(
                        1e-6,
                        1 - 1e-6,
                    )
                ),

            "variance":
                torch.ones_like(
                    fused_probability
                ),

            "embeddings":
                embeddings,

            "evidence":
                None,

            "reliability":
                {},
        }

    # ---------------------------------------------------------------
    # Gaussian methods
    # ---------------------------------------------------------------

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

        # Modality-dropout is a deterministic mean latent baseline
        # at evaluation time.
        fused = fuse_mean(
            evidence,
            missing,
        )

    else:

        raise ValueError(
            f"Unknown method: {model.method}"
        )

    logits = model.prediction(
        fused["mu"]
    )

    reliability = {}

    for name in [
        "CXR",
        "ECG",
        "Labs",
    ]:

        if name not in missing:

            reliability[name] = (
                evidence[name][
                    "reliability"
                ]
                .squeeze(-1)
            )

    return {
        "logits": logits,
        "variance":
            fused["variance"]
            .mean(dim=1),

        "embeddings":
            embeddings,

        "evidence":
            evidence,

        "reliability":
            reliability,
    }


# =====================================================================
# TRAINING HELPERS
# =====================================================================

def get_trainable_parameters(
    model,
):

    return [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]


def reliability_ranking_loss(
    values,
    margin=0.05,
):

    total = 0.0

    for index in range(
        len(values) - 1
    ):

        total = (
            total
            + torch.relu(
                margin
                - values[index]
                + values[index + 1]
            ).mean()
        )

    return total


def build_corrupted_inputs(
    batch,
    severity,
    modality,
):

    cxr = batch[
        "cxr"
    ].to(DEVICE)

    ecg = batch[
        "ecg"
    ].to(DEVICE)

    labs_values = batch[
        "labs_percentiles"
    ].to(DEVICE)

    labs_missing = batch[
        "labs_missingness"
    ].to(DEVICE)

    if modality == "CXR":

        cxr = corrupt_cxr(
            cxr,
            severity,
        )

    elif modality == "ECG":

        ecg = corrupt_ecg(
            ecg,
            severity,
        )

    elif modality == "Labs":

        labs_values = corrupt_labs(
            labs_values,
            severity,
        )

    labs = torch.cat(
        [
            labs_values,
            labs_missing,
        ],
        dim=1,
    )

    return {
        "cxr": cxr,
        "ecg": ecg,
        "labs": labs,
    }


def encode_for_reliability(
    model,
    inputs,
):

    embeddings = encode_modalities(
        model,
        inputs["cxr"],
        inputs["ecg"],
        inputs["labs"],
    )

    evidence = encode_gaussian(
        model,
        embeddings,
    )

    return evidence


# =====================================================================
# TRAIN ONE METHOD
# =====================================================================

def train_model(
    model,
    loader,
    method,
    pos_weight,
):

    optimizer = torch.optim.Adam(
        get_trainable_parameters(
            model
        ),
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

        total_n = 0

        for batch in loader:

            labels = batch[
                "label"
            ].to(DEVICE)

            optimizer.zero_grad()

            # -------------------------------------------------------
            # Standard forward condition.
            # -------------------------------------------------------

            if (
                method == "modality_dropout"
            ):

                patterns = [
                    set(),
                    {"CXR"},
                    {"ECG"},
                    {"Labs"},
                ]

                missing = random.choice(
                    patterns
                )

            else:

                missing = set()

            output = forward_model(
                model,
                batch,
                missing=missing,
            )

            prediction_loss = criterion(
                output["logits"],
                labels,
            )

            total_loss = (
                prediction_loss
            )

            # -------------------------------------------------------
            # Reliability supervision.
            #
            # Only the reliability fusion model is explicitly
            # supervised as a reliability-aware model.
            # -------------------------------------------------------

            if method == "reliability_poe":

                reliability_losses = []

                for modality in [
                    "CXR",
                    "ECG",
                    "Labs",
                ]:

                    level_values = []

                    for severity in [
                        0,
                        1,
                        2,
                        3,
                    ]:

                        if severity == 0:

                            inputs = {
                                "cxr":
                                    batch[
                                        "cxr"
                                    ].to(DEVICE),

                                "ecg":
                                    batch[
                                        "ecg"
                                    ].to(DEVICE),

                                "labs":
                                    torch.cat(
                                        [
                                            batch[
                                                "labs_percentiles"
                                            ].to(DEVICE),

                                            batch[
                                                "labs_missingness"
                                            ].to(DEVICE),
                                        ],
                                        dim=1,
                                    ),
                            }

                        else:

                            inputs = (
                                build_corrupted_inputs(
                                    batch,
                                    severity,
                                    modality,
                                )
                            )

                        evidence = (
                            encode_for_reliability(
                                model,
                                inputs,
                            )
                        )

                        level_values.append(
                            evidence[
                                modality
                            ][
                                "reliability"
                            ].squeeze(-1)
                        )

                    reliability_losses.append(
                        reliability_ranking_loss(
                            level_values
                        )
                    )

                reliability_loss = sum(
                    reliability_losses
                )

                total_loss = (
                    total_loss
                    + 0.5
                    * reliability_loss
                )

            # -------------------------------------------------------
            # Backprop.
            # -------------------------------------------------------

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                get_trainable_parameters(
                    model
                ),
                max_norm=5.0,
            )

            optimizer.step()

            batch_n = labels.shape[0]

            epoch_loss += (
                total_loss.item()
                * batch_n
            )

            total_n += batch_n

        print(
            f"      Epoch {epoch + 1}: "
            f"loss={epoch_loss / total_n:.6f}"
        )


# =====================================================================
# EVALUATION
# =====================================================================

@torch.no_grad()
def evaluate_condition(
    model,
    loader,
    missing,
    corruption_modality=None,
    corruption_severity=0,
):

    model.eval()

    labels_all = []

    probabilities_all = []

    variances_all = []

    reliability_values = {
        "CXR": [],
        "ECG": [],
        "Labs": [],
    }

    for batch in loader:

        local_batch = {
            key: value.clone()
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }

        # -----------------------------------------------------------
        # Apply corruption before forwarding.
        # -----------------------------------------------------------

        if corruption_modality == "CXR":

            local_batch["cxr"] = corrupt_cxr(
                local_batch["cxr"].to(DEVICE),
                corruption_severity,
            ).cpu()

        elif corruption_modality == "ECG":

            local_batch["ecg"] = corrupt_ecg(
                local_batch["ecg"].to(DEVICE),
                corruption_severity,
            ).cpu()

        elif corruption_modality == "Labs":

            local_batch[
                "labs_percentiles"
            ] = corrupt_labs(
                local_batch[
                    "labs_percentiles"
                ].to(DEVICE),
                corruption_severity,
            ).cpu()

        output = forward_model(
            model,
            local_batch,
            missing=missing,
        )

        probabilities = torch.sigmoid(
            output["logits"]
        )

        labels_all.extend(
            batch[
                "label"
            ].numpy().tolist()
        )

        probabilities_all.extend(
            probabilities.cpu()
            .numpy()
            .tolist()
        )

        variances_all.extend(
            output["variance"]
            .cpu()
            .numpy()
            .tolist()
        )

        for modality, values in (
            output["reliability"].items()
        ):

            reliability_values[
                modality
            ].extend(
                values.cpu()
                .numpy()
                .tolist()
            )

    y = np.asarray(
        labels_all,
        dtype=float,
    )

    p = np.asarray(
        probabilities_all,
        dtype=float,
    )

    variance = np.asarray(
        variances_all,
        dtype=float,
    )

    result = {
        "n":
            int(len(y)),

        "positive":
            int(np.sum(y == 1)),

        "negative":
            int(np.sum(y == 0)),

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
                np.mean(
                    (
                        p - y
                    ) ** 2
                )
            ),

        "variance":
            float(
                variance.mean()
            ),

        "reliability": {},
    }

    for modality in [
        "CXR",
        "ECG",
        "Labs",
    ]:

        if reliability_values[
            modality
        ]:

            result[
                "reliability"
            ][modality] = float(
                np.mean(
                    reliability_values[
                        modality
                    ]
                )
            )

    return result


# =====================================================================
# ONE SEED
# =====================================================================

def run_one_seed(
    method,
    seed,
    train_loader,
    val_loader,
    pos_weight,
):

    print(
        "\n" + "-" * 80
    )

    print(
        f"{method} | seed {seed}"
    )

    print(
        "-" * 80
    )

    set_seed(seed)

    model = FusionModel(
        method
    ).to(
        DEVICE
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    train_model(
        model,
        train_loader,
        method,
        pos_weight,
    )

    seed_results = {
        "clean": None,
        "missing_modality": {},
        "corruption": {
            "CXR": {},
            "ECG": {},
            "Labs": {},
        },
    }

    # ---------------------------------------------------------------
    # Clean
    # ---------------------------------------------------------------

    seed_results[
        "clean"
    ] = evaluate_condition(
        model,
        val_loader,
        missing=set(),
    )

    # ---------------------------------------------------------------
    # Missing modalities
    # ---------------------------------------------------------------

    for condition, missing in (
        MISSING_CONDITIONS.items()
    ):

        if condition == "all modalities":
            continue

        seed_results[
            "missing_modality"
        ][condition] = (
            evaluate_condition(
                model,
                val_loader,
                missing=missing,
            )
        )

    # ---------------------------------------------------------------
    # Corruption
    # ---------------------------------------------------------------

    for modality in CORRUPTION_MODALITIES:

        for severity in (
            CORRUPTION_SEVERITIES
        ):

            seed_results[
                "corruption"
            ][modality][
                str(severity)
            ] = evaluate_condition(
                model,
                val_loader,
                missing=set(),
                corruption_modality=modality,
                corruption_severity=severity,
            )

    return seed_results


# =====================================================================
# AGGREGATE
# =====================================================================

def aggregate_metric(
    seed_results,
    path,
):

    values = []

    for seed in SEEDS:

        current = seed_results[
            seed
        ]

        for key in path:

            current = current[
                key
            ]

        values.append(
            float(current)
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean":
            float(values.mean()),

        "std":
            float(
                values.std(
                    ddof=1
                )
            ),

        "values":
            [
                float(x)
                for x in values
            ],
    }


def aggregate_results(
    per_seed,
):

    summary = {}

    # ---------------------------------------------------------------
    # Clean
    # ---------------------------------------------------------------

    summary[
        "clean"
    ] = {}

    for metric in [
        "AUROC",
        "AUPRC",
        "Brier",
        "variance",
    ]:

        summary[
            "clean"
        ][metric] = (
            aggregate_metric(
                per_seed,
                [
                    "clean",
                    metric,
                ],
            )
        )

    # ---------------------------------------------------------------
    # Missing
    # ---------------------------------------------------------------

    summary[
        "missing_modality"
    ] = {}

    for condition in (
        MISSING_CONDITIONS
    ):

        if condition == "all modalities":
            continue

        summary[
            "missing_modality"
        ][condition] = {}

        for metric in [
            "AUROC",
            "AUPRC",
            "Brier",
            "variance",
        ]:

            summary[
                "missing_modality"
            ][condition][metric] = (
                aggregate_metric(
                    per_seed,
                    [
                        "missing_modality",
                        condition,
                        metric,
                    ],
                )
            )

    # ---------------------------------------------------------------
    # Corruption
    # ---------------------------------------------------------------

    summary[
        "corruption"
    ] = {}

    for modality in (
        CORRUPTION_MODALITIES
    ):

        summary[
            "corruption"
        ][modality] = {}

        for severity in (
            CORRUPTION_SEVERITIES
        ):

            summary[
                "corruption"
            ][modality][
                str(severity)
            ] = {}

            for metric in [
                "AUROC",
                "AUPRC",
                "Brier",
                "variance",
            ]:

                summary[
                    "corruption"
                ][modality][
                    str(severity)
                ][metric] = (
                    aggregate_metric(
                        per_seed,
                        [
                            "corruption",
                            modality,
                            str(severity),
                            metric,
                        ],
                    )
                )

    return summary


# =====================================================================
# BUILD TABLE-B / C / D ROWS
# =====================================================================

def build_table_b_row(
    method,
    summary,
):

    return {
        "method":
            method,

        "type":
            "Fusion",

        "AUROC_mean":
            summary[
                "clean"
            ]["AUROC"]["mean"],

        "AUROC_std":
            summary[
                "clean"
            ]["AUROC"]["std"],

        "AUPRC_mean":
            summary[
                "clean"
            ]["AUPRC"]["mean"],

        "AUPRC_std":
            summary[
                "clean"
            ]["AUPRC"]["std"],

        "Brier_mean":
            summary[
                "clean"
            ]["Brier"]["mean"],

        "Brier_std":
            summary[
                "clean"
            ]["Brier"]["std"],
    }


def build_table_c_rows(
    method,
    summary,
):

    rows = []

    for condition in [
        "all modalities",
        "CXR missing",
        "ECG missing",
        "Labs missing",
        "CXR + ECG missing",
        "CXR + Labs missing",
        "ECG + Labs missing",
    ]:

        if condition == "all modalities":

            clean = summary[
                "clean"
            ]

            row = {
                "method":
                    method,

                "condition":
                    condition,

                "AUROC_mean":
                    clean[
                        "AUROC"
                    ]["mean"],

                "AUROC_std":
                    clean[
                        "AUROC"
                    ]["std"],

                "AUPRC_mean":
                    clean[
                        "AUPRC"
                    ]["mean"],

                "AUPRC_std":
                    clean[
                        "AUPRC"
                    ]["std"],

                "Brier_mean":
                    clean[
                        "Brier"
                    ]["mean"],

                "Brier_std":
                    clean[
                        "Brier"
                    ]["std"],

                "variance_mean":
                    clean[
                        "variance"
                    ]["mean"],

                "variance_std":
                    clean[
                        "variance"
                    ]["std"],
            }

        else:

            item = summary[
                "missing_modality"
            ][condition]

            row = {
                "method":
                    method,

                "condition":
                    condition,

                "AUROC_mean":
                    item[
                        "AUROC"
                    ]["mean"],

                "AUROC_std":
                    item[
                        "AUROC"
                    ]["std"],

                "AUPRC_mean":
                    item[
                        "AUPRC"
                    ]["mean"],

                "AUPRC_std":
                    item[
                        "AUPRC"
                    ]["std"],

                "Brier_mean":
                    item[
                        "Brier"
                    ]["mean"],

                "Brier_std":
                    item[
                        "Brier"
                    ]["std"],

                "variance_mean":
                    item[
                        "variance"
                    ]["mean"],

                "variance_std":
                    item[
                        "variance"
                    ]["std"],
            }

        rows.append(row)

    return rows


def build_table_d_rows(
    method,
    summary,
):

    rows = []

    for modality in (
        CORRUPTION_MODALITIES
    ):

        clean_auc = summary[
            "clean"
        ]["AUROC"]["mean"]

        for severity in (
            CORRUPTION_SEVERITIES
        ):

            item = summary[
                "corruption"
            ][modality][
                str(severity)
            ]

            auc_mean = item[
                "AUROC"
            ]["mean"]

            rows.append({
                "method":
                    method,

                "modality":
                    modality,

                "severity":
                    severity,

                "AUROC_mean":
                    auc_mean,

                "AUROC_std":
                    item[
                        "AUROC"
                    ]["std"],

                "AUPRC_mean":
                    item[
                        "AUPRC"
                    ]["mean"],

                "AUPRC_std":
                    item[
                        "AUPRC"
                    ]["std"],

                "Brier_mean":
                    item[
                        "Brier"
                    ]["mean"],

                "Brier_std":
                    item[
                        "Brier"
                    ]["std"],

                "AUROC_delta_from_clean":
                    auc_mean
                    - clean_auc,
            })

    return rows


# =====================================================================
# MAIN
# =====================================================================

def main():

    print(
        "=" * 100
    )

    print(
        "FUSION COMPARISON EXPERIMENT"
    )

    print(
        "=" * 100
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "\nMethods:"
    )

    for method in METHODS:

        print(
            f"  {method}"
        )

    print(
        "\nSeeds:"
        f" {SEEDS}"
    )

    # ---------------------------------------------------------------
    # Load datasets.
    # ---------------------------------------------------------------

    train_tensor = SymileMIMICDataset(
        DATA_ROOT,
        "train",
    )

    val_tensor = SymileMIMICDataset(
        DATA_ROOT,
        "val",
    )

    train_dataset = LabeledDataset(
        train_tensor,
        TRAIN_CSV,
    )

    val_dataset = LabeledDataset(
        val_tensor,
        VAL_CSV,
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

    positive = int(
        train_dataset.labels.sum().item()
    )

    negative = (
        len(train_dataset)
        - positive
    )

    pos_weight = torch.tensor(
        negative / positive,
        dtype=torch.float32,
        device=DEVICE,
    )

    print(
        "\nDataset:"
    )

    print(
        f"  Train examples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"  Validation examples: "
        f"{len(val_dataset):,}"
    )

    print(
        f"  Positive: "
        f"{positive:,}"
    )

    print(
        f"  Negative: "
        f"{negative:,}"
    )

    print(
        f"  Positive weight: "
        f"{pos_weight.item():.6f}"
    )

    # ---------------------------------------------------------------
    # Run all methods.
    # ---------------------------------------------------------------

    per_method = {}

    summaries = {}

    table_b_rows = []

    table_c_rows = []

    table_d_rows = []

    for method in METHODS:

        print(
            "\n" + "=" * 100
        )

        print(
            f"METHOD: {method}"
        )

        print(
            "=" * 100
        )

        per_seed = {}

        for seed in SEEDS:

            per_seed[
                seed
            ] = run_one_seed(
                method,
                seed,
                train_loader,
                val_loader,
                pos_weight,
            )

        per_method[
            method
        ] = per_seed

        summary = aggregate_results(
            per_seed
        )

        summaries[
            method
        ] = summary

        table_b_rows.append(
            build_table_b_row(
                method,
                summary,
            )
        )

        table_c_rows.extend(
            build_table_c_rows(
                method,
                summary,
            )
        )

        table_d_rows.extend(
            build_table_d_rows(
                method,
                summary,
            )
        )

    # ---------------------------------------------------------------
    # Print Table B.
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "TABLE B INPUT: MAIN MULTIMODAL PREDICTION"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Method':<25}"
        f"{'AUROC':>20}"
        f"{'AUPRC':>20}"
        f"{'Brier':>20}"
    )

    print(
        "-" * 90
    )

    for row in table_b_rows:

        print(
            f"{row['method']:<25}"
            f"{row['AUROC_mean']:.4f} +/- "
            f"{row['AUROC_std']:.4f}"
            f"{row['AUPRC_mean']:>10.4f} +/- "
            f"{row['AUPRC_std']:.4f}"
            f"{row['Brier_mean']:>10.4f} +/- "
            f"{row['Brier_std']:.4f}"
        )

    # ---------------------------------------------------------------
    # Print Table C.
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "TABLE C INPUT: MISSING-MODALITY ROBUSTNESS"
    )

    print(
        "=" * 100
    )

    for method in METHODS:

        print(
            f"\n{method}"
        )

        for condition in [
            "CXR missing",
            "ECG missing",
            "Labs missing",
            "CXR + ECG missing",
            "CXR + Labs missing",
            "ECG + Labs missing",
        ]:

            row = next(
                item
                for item in table_c_rows
                if (
                    item["method"] == method
                    and item["condition"] == condition
                )
            )

            print(
                f"  {condition:<25}"
                f"{row['AUROC_mean']:.4f} +/- "
                f"{row['AUROC_std']:.4f}"
            )

    # ---------------------------------------------------------------
    # Print Table D.
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "TABLE D INPUT: CORRUPTION ROBUSTNESS"
    )

    print(
        "=" * 100
    )

    for method in METHODS:

        print(
            f"\n{method}"
        )

        for modality in (
            CORRUPTION_MODALITIES
        ):

            values = []

            for severity in (
                CORRUPTION_SEVERITIES
            ):

                row = next(
                    item
                    for item in table_d_rows
                    if (
                        item["method"] == method
                        and item["modality"] == modality
                        and item["severity"] == severity
                    )
                )

                values.append(
                    row["AUROC_mean"]
                )

            print(
                f"  {modality}: "
                + " -> ".join(
                    f"{value:.4f}"
                    for value in values
                )
            )

    # ---------------------------------------------------------------
    # Save.
    # ---------------------------------------------------------------

    output = {
        "experiment":
            "Five-seed fusion comparison",

        "target":
            LABEL_NAME,

        "seeds":
            SEEDS,

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "device":
            str(DEVICE),

        "methods":
            METHODS,

        "method_definitions": {

            "early_concat":
                "Concatenate the three modality embeddings "
                "and predict with an MLP.",

            "late_average":
                "Train one modality classifier per modality "
                "and average predicted probabilities.",

            "mean_latent":
                "Average modality Gaussian means.",

            "gaussian_poe":
                "Precision-weighted Gaussian Product-of-Experts "
                "without learned reliability.",

            "reliability_poe":
                "Gaussian PoE with effective precision "
                "r_m / sigma_m^2 and corruption-ranking supervision.",

            "modality_dropout":
                "Mean latent fusion trained with randomly "
                "dropped modality subsets.",
        },

        "per_seed":
            per_method,

        "summary":
            summaries,

        "table_b_rows":
            table_b_rows,

        "table_c_rows":
            table_c_rows,

        "table_d_rows":
            table_d_rows,
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
        "\n" + "=" * 100
    )

    print(
        "FUSION COMPARISON COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()