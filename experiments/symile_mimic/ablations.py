from pathlib import Path
import os
import sys
import json
import hashlib

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


TEST_CSV = DATA_ROOT / "test.csv"
TEST_DIR = DATA_ROOT / "data_npy" / "test"

RESULTS_DIR = OUTPUT_ROOT

CHECKPOINT = (
    RESULTS_DIR
    / "missing_modality_aware_model.pt"
)

PROTOCOL_FILE = (
    RESULTS_DIR
    / "final_test_protocol.json"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "final_test_ablation_results.json"
)


# =====================================================================
# Configuration
# =====================================================================

BATCH_SIZE = 32

EMBED_DIM = 128
LATENT_DIM = 32

LABEL_NAME = "Cardiomegaly"

EXPECTED_TEST_ROWS = 4640
EXPECTED_USABLE_TEST_ROWS = 1918

SEED = 42


# =====================================================================
# Seed
# =====================================================================

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# File hash
# =====================================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# =====================================================================
# Encoders
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
# Reliability
# =====================================================================

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
# Prediction head
# =====================================================================

class PredictionHead(nn.Module):

    def __init__(
        self,
        latent_dim,
    ):

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

        return self.network(
            z
        ).squeeze(-1)


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
                "CSV and tensor hadm_id arrays "
                "are not exactly aligned."
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

        sample["label"] = self.labels[
            index
        ]

        return sample


# =====================================================================
# Corruption definitions
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

    corrupted = (
        (1.0 - blur_strength) * x
        + blur_strength * blurred
    )

    noise_strength = {
        1: 0.02,
        2: 0.05,
        3: 0.10,
    }[severity]

    return (
        corrupted
        + noise_strength
        * torch.randn_like(corrupted)
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
# Fusion methods
# =====================================================================

def mean_fusion(
    experts,
):

    if not experts:
        raise ValueError(
            "No experts available."
        )

    mu = torch.stack(
        [
            expert["mu"]
            for expert in experts
        ],
        dim=0,
    ).mean(
        dim=0
    )

    return {
        "mu": mu,
        "variance": torch.ones_like(mu),
    }


def gaussian_poe(
    experts,
):

    if not experts:
        raise ValueError(
            "No experts available."
        )

    batch_size, latent_dim = (
        experts[0]["mu"].shape
    )

    device = experts[0]["mu"].device
    dtype = experts[0]["mu"].dtype

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

        precision_sum += precision

        precision_mu_sum += (
            precision
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


def reliability_poe(
    experts,
):

    if not experts:
        raise ValueError(
            "No experts available."
        )

    batch_size, latent_dim = (
        experts[0]["mu"].shape
    )

    device = experts[0]["mu"].device
    dtype = experts[0]["mu"].dtype

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
# Load checkpoint
# =====================================================================

def load_frozen_model(
    device,
):

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

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    state_dicts = checkpoint.get(
        "model_state_dicts"
    )

    if state_dicts is None:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dicts'."
        )

    if len(state_dicts) != len(
        modules
    ):

        raise ValueError(
            "Checkpoint contains "
            f"{len(state_dicts)} modules but "
            f"{len(modules)} are expected."
        )

    for index, module in enumerate(
        modules
    ):

        key = f"module_{index}"

        if key not in state_dicts:

            raise KeyError(
                f"Missing checkpoint key: {key}"
            )

        module.load_state_dict(
            state_dicts[key]
        )

        module.eval()

        for parameter in module.parameters():

            parameter.requires_grad = False

    return modules, checkpoint


# =====================================================================
# Encode
# =====================================================================

def encode_all(
    modules,
    cxr,
    ecg,
    lab_values,
    lab_missing,
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

    labs = torch.cat(
        [
            lab_values,
            lab_missing,
        ],
        dim=1,
    )

    cxr_ev = cxr_gaussian(
        cxr_encoder(cxr)
    )

    ecg_ev = ecg_gaussian(
        ecg_encoder(ecg)
    )

    lab_ev = lab_gaussian(
        lab_encoder(labs)
    )

    cxr_ev[
        "reliability"
    ] = cxr_reliability(
        cxr_ev["mu"],
        cxr_ev["logvar"],
    )

    ecg_ev[
        "reliability"
    ] = ecg_reliability(
        ecg_ev["mu"],
        ecg_ev["logvar"],
    )

    lab_ev[
        "reliability"
    ] = lab_reliability(
        lab_ev["mu"],
        lab_ev["logvar"],
    )

    return {
        "CXR": cxr_ev,
        "ECG": ecg_ev,
        "Labs": lab_ev,
    }


# =====================================================================
# Evaluate one condition
# =====================================================================

def evaluate_condition(
    modules,
    loader,
    device,
    missing_modalities=None,
    corruption_modality=None,
    corruption_severity=0,
):

    if missing_modalities is None:
        missing_modalities = set()

    labels_all = []

    outputs = {
        "mean": [],
        "poe": [],
        "reliability": [],
    }

    variances = {
        "mean": [],
        "poe": [],
        "reliability": [],
    }

    reliability_records = {
        "CXR": [],
        "ECG": [],
        "Labs": [],
    }

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

        # -----------------------------------------------------------
        # Corruption
        # -----------------------------------------------------------

        if (
            corruption_modality == "CXR"
            and "CXR" not in missing_modalities
        ):

            cxr = corrupt_cxr(
                cxr,
                corruption_severity,
            )

        elif (
            corruption_modality == "ECG"
            and "ECG" not in missing_modalities
        ):

            ecg = corrupt_ecg(
                ecg,
                corruption_severity,
            )

        elif (
            corruption_modality == "Labs"
            and "Labs" not in missing_modalities
        ):

            lab_values = corrupt_labs(
                lab_values,
                corruption_severity,
            )

        evidence = encode_all(
            modules,
            cxr,
            ecg,
            lab_values,
            lab_missing,
        )

        experts = []

        for modality in [
            "CXR",
            "ECG",
            "Labs",
        ]:

            if modality in missing_modalities:
                continue

            experts.append(
                evidence[modality]
            )

            reliability_records[
                modality
            ].extend(
                evidence[
                    modality
                ][
                    "reliability"
                ]
                .squeeze(-1)
                .cpu()
                .numpy()
                .tolist()
            )

        if not experts:

            raise RuntimeError(
                "No modalities remain."
            )

        # -----------------------------------------------------------
        # Mean
        # -----------------------------------------------------------

        mean_result = mean_fusion(
            experts
        )

        # -----------------------------------------------------------
        # Gaussian PoE
        # -----------------------------------------------------------

        poe_result = gaussian_poe(
            experts
        )

        # -----------------------------------------------------------
        # Reliability PoE
        # -----------------------------------------------------------

        reliability_result = (
            reliability_poe(
                experts
            )
        )

        fused_results = {
            "mean": mean_result,
            "poe": poe_result,
            "reliability": reliability_result,
        }

        for method, result in fused_results.items():

            logits = modules[-1](
                result["mu"]
            )

            probs = torch.sigmoid(
                logits
            )

            outputs[
                method
            ].extend(
                probs.cpu()
                .numpy()
                .tolist()
            )

            variances[
                method
            ].extend(
                result[
                    "variance"
                ]
                .mean(dim=1)
                .cpu()
                .numpy()
                .tolist()
            )

        labels_all.extend(
            labels.tolist()
        )

    y = np.asarray(
        labels_all,
        dtype=float,
    )

    result = {
        "n": int(
            len(y)
        ),

        "positive": int(
            np.sum(y == 1.0)
        ),

        "negative": int(
            np.sum(y == 0.0)
        ),

        "methods": {},
        "reliability": {},
    }

    for method in [
        "mean",
        "poe",
        "reliability",
    ]:

        probability = np.asarray(
            outputs[method],
            dtype=float,
        )

        result[
            "methods"
        ][method] = {
            "AUROC": float(
                roc_auc_score(
                    y,
                    probability,
                )
            ),

            "AUPRC": float(
                average_precision_score(
                    y,
                    probability,
                )
            ),

            "Brier": float(
                np.mean(
                    (
                        probability
                        - y
                    ) ** 2
                )
            ),

            "variance": float(
                np.mean(
                    variances[method]
                )
            ),
        }

    for modality in [
        "CXR",
        "ECG",
        "Labs",
    ]:

        if reliability_records[
            modality
        ]:

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
# Print comparison row
# =====================================================================

def print_comparison_row(
    label,
    result,
):

    print(
        f"\n{label}:"
    )

    for method in [
        "mean",
        "poe",
        "reliability",
    ]:

        metrics = result[
            "methods"
        ][method]

        print(
            f"  {method:<12}"
            f"AUROC={metrics['AUROC']:.4f} "
            f"AUPRC={metrics['AUPRC']:.4f} "
            f"Brier={metrics['Brier']:.4f}"
        )


# =====================================================================
# Main
# =====================================================================

def main():

    print("=" * 90)

    print(
        "FINAL TEST FUSION ABLATION"
    )

    print("=" * 90)

    print(
        "\nIMPORTANT:"
    )

    print(
        "  Frozen checkpoint."
    )

    print(
        "  No training."
    )

    print(
        "  No test-based model selection."
    )

    print(
        "  Test split evaluated once."
    )

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
    # Verify protocol and checkpoint
    # ---------------------------------------------------------------

    if not PROTOCOL_FILE.exists():

        raise FileNotFoundError(
            f"Missing protocol file:\n"
            f"{PROTOCOL_FILE}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Missing checkpoint:\n"
            f"{CHECKPOINT}"
        )

    protocol_hash = sha256_file(
        PROTOCOL_FILE
    )

    checkpoint_hash = sha256_file(
        CHECKPOINT
    )

    print(
        "\nProtocol SHA256:"
        f"\n  {protocol_hash}"
    )

    print(
        "\nCheckpoint SHA256:"
        f"\n  {checkpoint_hash}"
    )

    # ---------------------------------------------------------------
    # Test data
    # ---------------------------------------------------------------

    test_tensor = SymileMIMICDataset(
        TEST_DIR.parent,
        "test",
    )

    if len(
        test_tensor.hadm_id
    ) != EXPECTED_TEST_ROWS:

        raise RuntimeError(
            "Unexpected test tensor size: "
            f"{len(test_tensor.hadm_id)}"
        )

    test_dataset = (
        LabeledCardiomegalyDataset(
            test_tensor,
            TEST_CSV,
        )
    )

    if len(
        test_dataset
    ) != EXPECTED_USABLE_TEST_ROWS:

        raise RuntimeError(
            "Unexpected usable test size: "
            f"{len(test_dataset)}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        "\nTest examples:"
        f" {len(test_dataset):,}"
    )

    print(
        "Test positives:"
        f" {int(test_dataset.labels.sum().item()):,}"
    )

    print(
        "Test negatives:"
        f" {len(test_dataset) - int(test_dataset.labels.sum().item()):,}"
    )

    # ---------------------------------------------------------------
    # Load frozen checkpoint.
    # ---------------------------------------------------------------

    modules, checkpoint = (
        load_frozen_model(
            device
        )
    )

    print(
        "\nLoaded checkpoint:"
        f"\n  {CHECKPOINT}"
    )

    print(
        f"  Saved seed: "
        f"{checkpoint.get('seed', 'unknown')}"
    )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    results = {
        "metadata": {
            "label": LABEL_NAME,
            "test_total_rows": EXPECTED_TEST_ROWS,
            "test_usable_rows": EXPECTED_USABLE_TEST_ROWS,
            "checkpoint": str(
                CHECKPOINT
            ),
            "checkpoint_sha256": checkpoint_hash,
            "protocol": str(
                PROTOCOL_FILE
            ),
            "protocol_sha256": protocol_hash,
            "saved_training_seed": checkpoint.get(
                "seed"
            ),
            "evaluation_seed": SEED,
            "training_performed": False,
            "test_based_selection": False,
        },

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

    print(
        "\n" + "=" * 90
    )

    print(
        "CLEAN TEST"
    )

    print(
        "=" * 90
    )

    clean = evaluate_condition(
        modules,
        test_loader,
        device,
    )

    results[
        "clean"
    ] = clean

    print_comparison_row(
        "All modalities",
        clean,
    )

    # ---------------------------------------------------------------
    # Missing modalities
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "MISSING-MODALITY TEST"
    )

    print(
        "=" * 90
    )

    missing_conditions = [
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

    for name, missing in missing_conditions:

        result = evaluate_condition(
            modules,
            test_loader,
            device,
            missing_modalities=missing,
        )

        results[
            "missing_modality"
        ][name] = result

        print_comparison_row(
            name,
            result,
        )

    # ---------------------------------------------------------------
    # Corruption
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "CORRUPTION TEST"
    )

    print(
        "=" * 90
    )

    for modality in [
        "CXR",
        "ECG",
        "Labs",
    ]:

        for severity in [
            1,
            2,
            3,
        ]:

            result = evaluate_condition(
                modules,
                test_loader,
                device,
                corruption_modality=modality,
                corruption_severity=severity,
            )

            results[
                "corruption"
            ][modality][
                str(severity)
            ] = result

            compact_name = (
                f"{modality} severity {severity}"
            )

            print_comparison_row(
                compact_name,
                result,
            )

    # ---------------------------------------------------------------
    # Final compact table: clean + missing
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "FINAL TEST AUROC - FUSION ABLATION"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Condition':<28}"
        f"{'Mean':>10}"
        f"{'PoE':>10}"
        f"{'Reliability':>14}"
    )

    print(
        "-" * 65
    )

    compact_conditions = [
        (
            "All modalities",
            clean,
        )
    ]

    for name, _ in missing_conditions:

        compact_conditions.append(
            (
                name,
                results[
                    "missing_modality"
                ][name],
            )
        )

    for name, result in compact_conditions:

        print(
            f"{name:<28}"
            f"{result['methods']['mean']['AUROC']:>10.4f}"
            f"{result['methods']['poe']['AUROC']:>10.4f}"
            f"{result['methods']['reliability']['AUROC']:>14.4f}"
        )

    # ---------------------------------------------------------------
    # Corruption compact table
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "FINAL TEST CORRUPTION AUROC"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Condition':<28}"
        f"{'Mean':>10}"
        f"{'PoE':>10}"
        f"{'Reliability':>14}"
    )

    print(
        "-" * 65
    )

    for modality in [
        "CXR",
        "ECG",
        "Labs",
    ]:

        for severity in [
            1,
            2,
            3,
        ]:

            result = results[
                "corruption"
            ][modality][
                str(severity)
            ]

            print(
                f"{modality + ' s' + str(severity):<28}"
                f"{result['methods']['mean']['AUROC']:>10.4f}"
                f"{result['methods']['poe']['AUROC']:>10.4f}"
                f"{result['methods']['reliability']['AUROC']:>14.4f}"
            )

    # ---------------------------------------------------------------
    # Reliability behavior
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "RELIABILITY UNDER CORRUPTION"
    )

    print(
        "=" * 90
    )

    for modality in [
        "CXR",
        "ECG",
        "Labs",
    ]:

        values = []

        for severity in [
            0,
            1,
            2,
            3,
        ]:

            if severity == 0:

                condition_result = clean

            else:

                condition_result = results[
                    "corruption"
                ][modality][
                    str(severity)
                ]

            values.append(
                condition_result[
                    "reliability"
                ].get(
                    modality
                )
            )

        print(
            f"{modality}: "
            + " -> ".join(
                "NA"
                if value is None
                else f"{value:.4f}"
                for value in values
            )
        )

    # ---------------------------------------------------------------
    # AUROC deltas relative to clean
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "AUROC DELTAS FROM CLEAN"
    )

    print(
        "=" * 90
    )

    for modality in [
        "CXR",
        "ECG",
        "Labs",
    ]:

        print(
            f"\n{modality} corruption:"
        )

        for method in [
            "mean",
            "poe",
            "reliability",
        ]:

            clean_auc = clean[
                "methods"
            ][method]["AUROC"]

            severe_auc = results[
                "corruption"
            ][modality]["3"][
                "methods"
            ][method]["AUROC"]

            delta = (
                severe_auc
                - clean_auc
            )

            print(
                f"  {method:<12}"
                f"{delta:+.4f}"
            )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    output_hash = sha256_file(
        OUTPUT_FILE
    )

    print(
        "\nSaved final ablation results:"
        f"\n  {OUTPUT_FILE}"
    )

    print(
        "\nResults SHA256:"
        f"\n  {output_hash}"
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "FINAL TEST ABLATION COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        f"\nEvaluated {len(test_dataset):,} "
        "usable held-out test examples."
    )

    print(
        "No training or test-based model selection "
        "was performed."
    )


if __name__ == "__main__":
    main()