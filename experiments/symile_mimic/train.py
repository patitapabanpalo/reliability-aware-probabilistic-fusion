from pathlib import Path
import os
import sys
import random
import json

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
# Paths
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SYMILE_DATA_ROOT", ROOT)).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("SYMILE_OUTPUT_ROOT", ROOT / "experiment_results")).expanduser().resolve()

from datasets.symile_mimic.dataset import SymileMIMICDataset
from reliability_fusion.encoders.symile import LabEncoder


# =====================================================================
# Configuration
# =====================================================================

SEED = int(os.environ.get("SYMILE_SEED", "42"))

BATCH_SIZE = 16
VAL_BATCH_SIZE = 32

EMBED_DIM = 128
LATENT_DIM = 32

EPOCHS = 10
LEARNING_RATE = 1e-3

LAMBDA_UNI = 0.5
LAMBDA_REL = 0.5
LAMBDA_CONS = 0.25

REL_MARGIN = 0.05

LABEL_NAME = "Cardiomegaly"

RESULTS_DIR = OUTPUT_ROOT

RESULTS_DIR.mkdir(
    exist_ok=True
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Encoders
# =====================================================================

class CXREncoder(nn.Module):

    def __init__(self, embedding_dim=128):
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

    def __init__(self, embedding_dim=128):
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
        x = x.transpose(1, 2)

        x = self.features(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        return self.projection(x)


# =====================================================================
# Gaussian evidence
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

        logvar = self.logvar(x).clamp(
            -6.0,
            4.0,
        )

        return {
            "mu": mu,
            "logvar": logvar,
        }


# =====================================================================
# Reliability head
# =====================================================================

class ReliabilityHead(nn.Module):

    def __init__(self, latent_dim):

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
# Prediction head
# =====================================================================

class PredictionHead(nn.Module):

    def __init__(self, latent_dim):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                latent_dim,
                32,
            ),
            nn.ReLU(),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(self, z):

        return self.network(z).squeeze(-1)


# =====================================================================
# Dataset
# =====================================================================

class LabeledCardiomegalyDataset(Dataset):

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
                "CSV and tensor hadm_id ordering do not match."
            )

        labels = df[
            LABEL_NAME
        ].to_numpy(
            dtype=np.float32
        )

        self.indices = np.flatnonzero(
            np.isin(
                labels,
                [0.0, 1.0],
            )
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

    def __getitem__(self, index):

        source_index = int(
            self.indices[index]
        )

        sample = self.tensor_dataset[
            source_index
        ].copy()

        sample["label"] = self.labels[
            index
        ]

        return sample

# =====================================================================
# Corruption functions
#
# Not used to simulate missing modalities.
# These are only used during training to teach reliability semantics.
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

    blur_strength = {
        1: 0.25,
        2: 0.55,
        3: 0.85,
    }[severity]

    x = (
        (1.0 - blur_strength) * x
        + blur_strength * blurred
    )

    noise_strength = {
        1: 0.02,
        2: 0.05,
        3: 0.10,
    }[severity]

    return (
        x
        + noise_strength
        * torch.randn_like(x)
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
# Reliability ranking
# =====================================================================

def ranking_loss(
    higher,
    lower,
    margin,
):

    return torch.relu(
        margin
        - higher
        + lower
    ).mean()


def monotonic_loss(
    levels,
    margin,
):

    if len(levels) != 4:

        raise ValueError(
            "Expected four reliability levels."
        )

    return (
        ranking_loss(
            levels[0],
            levels[1],
            margin,
        )
        + ranking_loss(
            levels[1],
            levels[2],
            margin,
        )
        + ranking_loss(
            levels[2],
            levels[3],
            margin,
        )
    )


# =====================================================================
# Fusion
# =====================================================================

def reliability_poe(
    experts,
):

    if len(experts) == 0:

        raise ValueError(
            "No experts available."
        )

    batch_size, latent_dim = (
        experts[0]["mu"].shape
    )

    device = experts[0]["mu"].device
    dtype = experts[0]["mu"].dtype

    # Unit Gaussian prior.
    precision_sum = torch.ones(
        batch_size,
        latent_dim,
        device=device,
        dtype=dtype,
    )

    precision_mu_sum = torch.zeros(
        batch_size,
        latent_dim,
        device=device,
        dtype=dtype,
    )

    for expert in experts:

        variance = torch.exp(
            expert["logvar"]
        ).clamp_min(
            1e-6
        )

        precision = (
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
            * precision
        )

        precision_sum += (
            effective_precision
        )

        precision_mu_sum += (
            effective_precision
            * expert["mu"]
        )

    variance = (
        1.0
        / precision_sum.clamp_min(
            1e-6
        )
    )

    mu = (
        variance
        * precision_mu_sum
    )

    return {
        "mu": mu,
        "variance": variance,
    }


# =====================================================================
# Encode available modalities
# =====================================================================

def encode_available(
    cxr,
    ecg,
    labs,
    missing,
    modules,
):

    (
        cxr_encoder,
        ecg_encoder,
        lab_encoder,
        cxr_gaussian,
        ecg_gaussian,
        lab_gaussian,
        cxr_reliability,
        ecg_reliability,
        lab_reliability,
    ) = modules

    experts = []

    if "CXR" not in missing:

        cxr_ev = cxr_gaussian(
            cxr_encoder(cxr)
        )

        cxr_ev[
            "reliability"
        ] = cxr_reliability(
            cxr_ev["mu"],
            cxr_ev["logvar"],
        )

        experts.append(
            (
                "CXR",
                cxr_ev,
            )
        )

    if "ECG" not in missing:

        ecg_ev = ecg_gaussian(
            ecg_encoder(ecg)
        )

        ecg_ev[
            "reliability"
        ] = ecg_reliability(
            ecg_ev["mu"],
            ecg_ev["logvar"],
        )

        experts.append(
            (
                "ECG",
                ecg_ev,
            )
        )

    if "Labs" not in missing:

        lab_ev = lab_gaussian(
            lab_encoder(labs)
        )

        lab_ev[
            "reliability"
        ] = lab_reliability(
            lab_ev["mu"],
            lab_ev["logvar"],
        )

        experts.append(
            (
                "Labs",
                lab_ev,
            )
        )

    return experts


# =====================================================================
# Sample missing-modality pattern
# =====================================================================

def sample_missing_pattern():

    patterns = [
        (),
        ("CXR",),
        ("ECG",),
        ("Labs",),
        ("CXR", "ECG"),
        ("CXR", "Labs"),
        ("ECG", "Labs"),
    ]

    # Heavier weight on single-missing cases.
    weights = [
        0.25,
        0.15,
        0.15,
        0.15,
        0.10,
        0.10,
        0.10,
    ]

    selected = random.choices(
        patterns,
        weights=weights,
        k=1,
    )[0]

    return set(selected)


# =====================================================================
# Training
# =====================================================================

def train_model(
    modules,
    loader,
    device,
    pos_weight,
):

    (
        cxr_encoder,
        ecg_encoder,
        lab_encoder,
        cxr_gaussian,
        ecg_gaussian,
        lab_gaussian,
        cxr_reliability,
        ecg_reliability,
        lab_reliability,
        prediction_head,
    ) = modules

    for module in modules:

        module.train()

    optimizer = torch.optim.Adam(
        [
            parameter
            for module in modules
            for parameter in module.parameters()
        ],
        lr=LEARNING_RATE,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    for epoch in range(EPOCHS):

        total_sum = 0.0
        prediction_sum = 0.0
        unimodal_sum = 0.0
        reliability_sum = 0.0
        consistency_sum = 0.0

        total_examples = 0

        for batch_index, batch in enumerate(
            loader
        ):

            cxr = batch[
                "cxr"
            ].to(device)

            ecg = batch[
                "ecg"
            ].to(device)

            lab_values = batch[
                "labs_percentiles"
            ].to(device)

            lab_missing = batch[
                "labs_missingness"
            ].to(device)

            labels = batch[
                "label"
            ].to(device)

            labs = torch.cat(
                [
                    lab_values,
                    lab_missing,
                ],
                dim=1,
            )

            optimizer.zero_grad()

            # -------------------------------------------------------
            # Full-modality reference
            # -------------------------------------------------------

            full_experts = encode_available(
                cxr,
                ecg,
                labs,
                set(),
                modules[
                    :9
                ],
            )

            full_evidence = [
                item[1]
                for item in full_experts
            ]

            full_fused = reliability_poe(
                full_evidence
            )

            full_logits = prediction_head(
                full_fused["mu"]
            )

            full_prediction_loss = (
                criterion(
                    full_logits,
                    labels,
                )
            )

            # -------------------------------------------------------
            # Randomly missing training condition
            # -------------------------------------------------------

            missing = sample_missing_pattern()

            observed_experts = encode_available(
                cxr,
                ecg,
                labs,
                missing,
                modules[
                    :9
                ],
            )

            observed_evidence = [
                item[1]
                for item in observed_experts
            ]

            observed_fused = reliability_poe(
                observed_evidence
            )

            observed_logits = prediction_head(
                observed_fused["mu"]
            )

            missing_prediction_loss = (
                criterion(
                    observed_logits,
                    labels,
                )
            )

            # -------------------------------------------------------
            # Prediction objective
            #
            # Both full and missing-modality predictions should
            # learn the actual target.
            # -------------------------------------------------------

            prediction_loss = (
                0.5 * full_prediction_loss
                + 0.5 * missing_prediction_loss
            )

            # -------------------------------------------------------
            # Unimodal consistency
            # -------------------------------------------------------

            cxr_logits = prediction_head(
                full_evidence[0]["mu"]
            )

            ecg_logits = prediction_head(
                full_evidence[1]["mu"]
            )

            lab_logits = prediction_head(
                full_evidence[2]["mu"]
            )

            unimodal_loss = (
                criterion(
                    cxr_logits,
                    labels,
                )
                + criterion(
                    ecg_logits,
                    labels,
                )
                + criterion(
                    lab_logits,
                    labels,
                )
            )

            # -------------------------------------------------------
            # Severity reliability objective
            # -------------------------------------------------------

            cxr_levels = []
            ecg_levels = []
            lab_levels = []

            for severity in range(4):

                # ---------------------------------------------------
                # Importantly, reliability training uses one
                # corruption at a time, not missingness.
                # ---------------------------------------------------

                cxr_input = corrupt_cxr(
                    cxr,
                    severity,
                )

                ecg_input = corrupt_ecg(
                    ecg,
                    severity,
                )

                lab_input_values = corrupt_labs(
                    lab_values,
                    severity,
                )

                lab_input = torch.cat(
                    [
                        lab_input_values,
                        lab_missing,
                    ],
                    dim=1,
                )

                cxr_s = cxr_gaussian(
                    cxr_encoder(
                        cxr_input
                    )
                )

                ecg_s = ecg_gaussian(
                    ecg_encoder(
                        ecg_input
                    )
                )

                lab_s = lab_gaussian(
                    lab_encoder(
                        lab_input
                    )
                )

                cxr_levels.append(
                    cxr_reliability(
                        cxr_s["mu"],
                        cxr_s["logvar"],
                    ).squeeze(-1)
                )

                ecg_levels.append(
                    ecg_reliability(
                        ecg_s["mu"],
                        ecg_s["logvar"],
                    ).squeeze(-1)
                )

                lab_levels.append(
                    lab_reliability(
                        lab_s["mu"],
                        lab_s["logvar"],
                    ).squeeze(-1)
                )

            reliability_loss = (
                monotonic_loss(
                    cxr_levels,
                    REL_MARGIN,
                )
                + monotonic_loss(
                    ecg_levels,
                    REL_MARGIN,
                )
                + monotonic_loss(
                    lab_levels,
                    REL_MARGIN,
                )
            )

            # -------------------------------------------------------
            # Consistency loss
            #
            # Full-modality probabilities are treated as the
            # teacher target, but detached so they do not receive
            # the consistency gradient.
            # -------------------------------------------------------

            full_probability = torch.sigmoid(
                full_logits
            ).detach()

            observed_probability = torch.sigmoid(
                observed_logits
            )

            consistency_loss = F.mse_loss(
                observed_probability,
                full_probability,
            )

            # -------------------------------------------------------
            # Total loss
            # -------------------------------------------------------

            loss = (
                prediction_loss
                + LAMBDA_UNI * unimodal_loss
                + LAMBDA_REL * reliability_loss
                + LAMBDA_CONS * consistency_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for module in modules
                    for parameter in module.parameters()
                ],
                max_norm=5.0,
            )

            optimizer.step()

            n = labels.size(0)

            total_sum += (
                loss.item() * n
            )

            prediction_sum += (
                prediction_loss.item() * n
            )

            unimodal_sum += (
                unimodal_loss.item() * n
            )

            reliability_sum += (
                reliability_loss.item() * n
            )

            consistency_sum += (
                consistency_loss.item() * n
            )

            total_examples += n

            if batch_index == 0:

                print(
                    f"\nEpoch {epoch + 1}"
                )

                print(
                    f"  missing pattern: "
                    f"{sorted(missing) if missing else 'none'}"
                )

                print(
                    f"  prediction loss: "
                    f"{prediction_loss.item():.6f}"
                )

                print(
                    f"  unimodal loss: "
                    f"{unimodal_loss.item():.6f}"
                )

                print(
                    f"  reliability loss: "
                    f"{reliability_loss.item():.6f}"
                )

                print(
                    f"  consistency loss: "
                    f"{consistency_loss.item():.6f}"
                )

                print(
                    f"  total loss: "
                    f"{loss.item():.6f}"
                )

        print(
            f"  epoch total loss: "
            f"{total_sum / total_examples:.6f}"
        )

        print(
            f"  epoch prediction loss: "
            f"{prediction_sum / total_examples:.6f}"
        )

        print(
            f"  epoch unimodal loss: "
            f"{unimodal_sum / total_examples:.6f}"
        )

        print(
            f"  epoch reliability loss: "
            f"{reliability_sum / total_examples:.6f}"
        )

        print(
            f"  epoch consistency loss: "
            f"{consistency_sum / total_examples:.6f}"
        )


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_condition(
    modules,
    loader,
    device,
    missing_modalities,
):

    (
        cxr_encoder,
        ecg_encoder,
        lab_encoder,
        cxr_gaussian,
        ecg_gaussian,
        lab_gaussian,
        cxr_reliability,
        ecg_reliability,
        lab_reliability,
        prediction_head,
    ) = modules

    for module in modules:
        module.eval()

    labels_all = []

    probabilities = []

    variances = []

    reliability_records = {
        "CXR": [],
        "ECG": [],
        "Labs": [],
    }

    with torch.no_grad():

        for batch in loader:

            cxr = batch[
                "cxr"
            ].to(device)

            ecg = batch[
                "ecg"
            ].to(device)

            lab_values = batch[
                "labs_percentiles"
            ].to(device)

            lab_missing = batch[
                "labs_missingness"
            ].to(device)

            labels = batch[
                "label"
            ].cpu().numpy()

            labs = torch.cat(
                [
                    lab_values,
                    lab_missing,
                ],
                dim=1,
            )

            experts = encode_available(
                cxr,
                ecg,
                labs,
                missing_modalities,
                modules[
                    :9
                ],
            )

            evidence = [
                item[1]
                for item in experts
            ]

            result = reliability_poe(
                evidence
            )

            logits = prediction_head(
                result["mu"]
            )

            probs = torch.sigmoid(
                logits
            )

            labels_all.extend(
                labels.tolist()
            )

            probabilities.extend(
                probs.cpu()
                .numpy()
                .tolist()
            )

            variances.extend(
                result["variance"]
                .mean(dim=1)
                .cpu()
                .numpy()
                .tolist()
            )

            for name, expert in experts:

                reliability_records[
                    name
                ].extend(
                    expert[
                        "reliability"
                    ]
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                    .tolist()
                )

    y = np.asarray(
        labels_all
    )

    p = np.asarray(
        probabilities
    )

    result = {
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
                (
                    p - y
                ) ** 2
            )
        ),

        "variance": float(
            np.mean(
                variances
            )
        ),

        "reliability": {},
    }

    for modality in [
        "CXR",
        "ECG",
        "Labs",
    ]:

        if len(
            reliability_records[
                modality
            ]
        ) > 0:

            result[
                "reliability"
            ][modality] = float(
                np.mean(
                    reliability_records[
                        modality
                    ]
                )
            )

    return result


# =====================================================================
# Main
# =====================================================================

def main():

    print("=" * 80)
    print(
        "Missing-modality-aware training"
    )
    print("=" * 80)

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    train_tensor = SymileMIMICDataset(
        DATA_ROOT / "data_npy",
        "train",
    )

    val_tensor = SymileMIMICDataset(
        DATA_ROOT / "data_npy",
        "val",
    )

    train_dataset = LabeledCardiomegalyDataset(
        train_tensor,
        DATA_ROOT / "train.csv",
    )

    val_dataset = LabeledCardiomegalyDataset(
        val_tensor,
        DATA_ROOT / "val.csv",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"\nTrain examples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation examples: "
        f"{len(val_dataset):,}"
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
        device=device,
    )

    print(
        f"Positive: {positive:,}"
    )

    print(
        f"Negative: {negative:,}"
    )

    print(
        f"Positive weight: "
        f"{pos_weight.item():.4f}"
    )

    # ---------------------------------------------------------------
    # Modules
    # ---------------------------------------------------------------

    modules = [

        CXREncoder(
            EMBED_DIM
        ).to(device),

        ECGEncoder(
            EMBED_DIM
        ).to(device),

        LabEncoder(
            input_dim=100,
            embedding_dim=EMBED_DIM,
        ).to(device),

        GaussianEvidence(
            EMBED_DIM,
            LATENT_DIM,
        ).to(device),

        GaussianEvidence(
            EMBED_DIM,
            LATENT_DIM,
        ).to(device),

        GaussianEvidence(
            EMBED_DIM,
            LATENT_DIM,
        ).to(device),

        ReliabilityHead(
            LATENT_DIM
        ).to(device),

        ReliabilityHead(
            LATENT_DIM
        ).to(device),

        ReliabilityHead(
            LATENT_DIM
        ).to(device),

        PredictionHead(
            LATENT_DIM
        ).to(device),
    ]

    total_parameters = sum(
        p.numel()
        for module in modules
        for p in module.parameters()
    )

    print(
        f"\nModel parameters: "
        f"{total_parameters:,}"
    )

    print(
        "\nTraining objective:"
    )

    print(
        "  full-modality prediction"
    )

    print(
        "  missing-modality prediction"
    )

    print(
        "  unimodal consistency"
    )

    print(
        "  corruption-based reliability ranking"
    )

    print(
        "  full -> missing prediction consistency"
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    train_model(
        modules,
        train_loader,
        device,
        pos_weight,
    )

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    conditions = [

        (
            "all modalities",
            set(),
        ),

        (
            "CXR missing",
            {"CXR"},
        ),

        (
            "ECG missing",
            {"ECG"},
        ),

        (
            "Labs missing",
            {"Labs"},
        ),

        (
            "CXR + ECG missing",
            {"CXR", "ECG"},
        ),

        (
            "CXR + Labs missing",
            {"CXR", "Labs"},
        ),

        (
            "ECG + Labs missing",
            {"ECG", "Labs"},
        ),
    ]

    results = {}

    print(
        "\n" + "=" * 80
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=" * 80
    )

    for condition_name, missing in conditions:

        result = evaluate_condition(
            modules,
            val_loader,
            device,
            missing,
        )

        results[
            condition_name
        ] = result

        print(
            f"\n{condition_name}:"
        )

        print(
            f"  AUROC={result['AUROC']:.4f}"
            f" AUPRC={result['AUPRC']:.4f}"
            f" Brier={result['Brier']:.4f}"
            f" Var={result['variance']:.4f}"
        )

        if result[
            "reliability"
        ]:

            print(
                "  reliability:"
            )

            for modality in [
                "CXR",
                "ECG",
                "Labs",
            ]:

                if modality in result[
                    "reliability"
                ]:

                    print(
                        f"    {modality}: "
                        f"{result['reliability'][modality]:.4f}"
                    )

    # ---------------------------------------------------------------
    # Compact comparison
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "COMPACT COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        f"{'Condition':<24}"
        f"{'AUROC':>10}"
        f"{'AUPRC':>10}"
        f"{'Brier':>10}"
    )

    print(
        "-" * 58
    )

    for condition_name, _ in conditions:

        r = results[
            condition_name
        ]

        print(
            f"{condition_name:<24}"
            f"{r['AUROC']:>10.4f}"
            f"{r['AUPRC']:>10.4f}"
            f"{r['Brier']:>10.4f}"
        )

    # ---------------------------------------------------------------
    # AUROC degradation
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "AUROC CHANGE FROM FULL MODALITY"
    )

    print(
        "=" * 80
    )

    clean_auc = results[
        "all modalities"
    ]["AUROC"]

    for condition_name in [
        "CXR missing",
        "ECG missing",
        "Labs missing",
        "CXR + ECG missing",
        "CXR + Labs missing",
        "ECG + Labs missing",
    ]:

        auc = results[
            condition_name
        ]["AUROC"]

        delta = (
            auc
            - clean_auc
        )

        print(
            f"{condition_name:<24}"
            f"{delta:+.4f}"
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    checkpoint_path = (
        RESULTS_DIR
        / "missing_modality_aware_model.pt"
    )

    state_dicts = {}

    for index, module in enumerate(
        modules
    ):

        state_dicts[
            f"module_{index}"
        ] = module.state_dict()

    torch.save(
        {
            "model_state_dicts": state_dicts,
            "label": LABEL_NAME,
            "seed": SEED,
            "lambda_uni": LAMBDA_UNI,
            "lambda_rel": LAMBDA_REL,
            "lambda_cons": LAMBDA_CONS,
            "rel_margin": REL_MARGIN,
        },
        checkpoint_path,
    )

    results_path = (
        RESULTS_DIR
        / "missing_modality_aware_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        "\nSaved checkpoint:"
        f"\n  {checkpoint_path}"
    )

    print(
        "\nSaved results:"
        f"\n  {results_path}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "MISSING-MODALITY-AWARE EXPERIMENT COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()