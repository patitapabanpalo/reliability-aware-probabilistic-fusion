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
TEST_DIR = DATA_ROOT / "data_npy" / "test"
TEST_CSV = DATA_ROOT / "test.csv"
RESULTS_DIR = Path(os.environ.get("SYMILE_OUTPUT_ROOT", ROOT / "experiment_results")).expanduser().resolve()
CHECKPOINT = RESULTS_DIR / "missing_modality_aware_model.pt"
PROTOCOL_FILE = RESULTS_DIR / "final_test_protocol.json"

from datasets.symile_mimic.dataset import SymileMIMICDataset
from reliability_fusion.encoders.symile import LabEncoder


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
# Reproducibility
# =====================================================================

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# SHA256
# =====================================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

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
# Test dataset
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
                "Test CSV and tensor hadm_id "
                "arrays are not exactly aligned."
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
# Corruptions
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
# Reliability-weighted PoE
# =====================================================================

def reliability_poe(
    experts,
):

    if len(experts) == 0:

        raise ValueError(
            "Cannot fuse zero experts."
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
# Load checkpoint
# =====================================================================

def load_checkpoint(
    modules,
    checkpoint_path,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
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
            "Checkpoint module count does not "
            "match current architecture."
        )

    for index, module in enumerate(
        modules
    ):

        module.load_state_dict(
            state_dicts[
                f"module_{index}"
            ]
        )

        module.eval()

        for parameter in module.parameters():

            parameter.requires_grad = False

    return checkpoint


# =====================================================================
# Encode modalities
# =====================================================================

def encode_modalities(
    cxr,
    ecg,
    lab_values,
    lab_missing,
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
# Evaluate condition
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

    for module in modules:

        module.eval()

    labels_all = []
    probabilities_all = []
    variances_all = []

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

            # -------------------------------------------------------
            # Apply corruption only when this is a corruption
            # evaluation. It is never used to represent missing data.
            # -------------------------------------------------------

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

            evidence = encode_modalities(
                cxr,
                ecg,
                lab_values,
                lab_missing,
                modules,
            )

            experts = []

            for modality in [
                "CXR",
                "ECG",
                "Labs",
            ]:

                if modality not in missing_modalities:

                    experts.append(
                        evidence[
                            modality
                        ]
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

            if len(experts) == 0:

                raise RuntimeError(
                    "All modalities are missing."
                )

            fused = reliability_poe(
                experts
            )

            logits = modules[-1](
                fused["mu"]
            )

            probabilities = torch.sigmoid(
                logits
            )

            labels_all.extend(
                labels.tolist()
            )

            probabilities_all.extend(
                probabilities
                .cpu()
                .numpy()
                .tolist()
            )

            variances_all.extend(
                fused["variance"]
                .mean(dim=1)
                .cpu()
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

    v = np.asarray(
        variances_all,
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

        "variance": float(
            v.mean()
        ),

        "reliability": {},
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
# Print result
# =====================================================================

def print_result(
    name,
    result,
):

    print(
        f"\n{name}:"
    )

    print(
        f"  N       = {result['n']:,}"
    )

    print(
        f"  Positive= {result['positive']:,}"
    )

    print(
        f"  Negative= {result['negative']:,}"
    )

    print(
        f"  AUROC   = {result['AUROC']:.4f}"
    )

    print(
        f"  AUPRC   = {result['AUPRC']:.4f}"
    )

    print(
        f"  Brier   = {result['Brier']:.4f}"
    )

    print(
        f"  Variance= {result['variance']:.4f}"
    )

    if result[
        "reliability"
    ]:

        print(
            "  Reliability:"
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


# =====================================================================
# Main
# =====================================================================

def main():

    print("=" * 90)

    print(
        "FINAL HELD-OUT TEST EVALUATION"
    )

    print("=" * 90)

    print(
        "\nIMPORTANT:"
    )

    print(
        "  No training is performed."
    )

    print(
        "  No validation metrics are used."
    )

    print(
        "  No test-based model selection is performed."
    )

    print(
        "  The test split is evaluated once using "
        "the frozen checkpoint."
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
    # Verify protocol record.
    # ---------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not PROTOCOL_FILE.exists():
        protocol = {
            "purpose": "Held-out test robustness evaluation protocol",
            "test_based_model_selection": False,
            "training_performed_during_evaluation": False,
            "checkpoint": str(CHECKPOINT),
            "evaluation_seed": SEED,
            "note": "Portable release record generated before first held-out robustness evaluation.",
        }
        PROTOCOL_FILE.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            "Frozen checkpoint is missing:\n"
            f"{CHECKPOINT}"
        )

    protocol_sha256 = sha256_file(
        PROTOCOL_FILE
    )

    checkpoint_sha256 = sha256_file(
        CHECKPOINT
    )

    print(
        "\nProtocol SHA256:"
        f"\n  {protocol_sha256}"
    )

    print(
        "\nCheckpoint SHA256:"
        f"\n  {checkpoint_sha256}"
    )

    # ---------------------------------------------------------------
    # Load test data.
    # ---------------------------------------------------------------

    test_tensor = SymileMIMICDataset(
        TEST_DIR.parent,
        "test",
    )

    if len(test_tensor.hadm_id) != (
        EXPECTED_TEST_ROWS
    ):

        raise RuntimeError(
            "Unexpected test tensor row count: "
            f"{len(test_tensor.hadm_id)} "
            f"(expected {EXPECTED_TEST_ROWS})."
        )

    test_dataset = (
        LabeledCardiomegalyDataset(
            test_tensor,
            TEST_CSV,
        )
    )

    if len(test_dataset) != (
        EXPECTED_USABLE_TEST_ROWS
    ):

        raise RuntimeError(
            "Unexpected usable test row count: "
            f"{len(test_dataset)} "
            f"(expected {EXPECTED_USABLE_TEST_ROWS})."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        "\nTest dataset:"
    )

    print(
        f"  Total rows: "
        f"{EXPECTED_TEST_ROWS:,}"
    )

    print(
        f"  Usable labels: "
        f"{len(test_dataset):,}"
    )

    print(
        f"  Positives: "
        f"{int(test_dataset.labels.sum().item()):,}"
    )

    print(
        f"  Negatives: "
        f"{len(test_dataset) - int(test_dataset.labels.sum().item()):,}"
    )

    # ---------------------------------------------------------------
    # Build frozen model.
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

    checkpoint = load_checkpoint(
        modules,
        CHECKPOINT,
        device,
    )

    print(
        "\nCheckpoint loaded:"
        f"\n  {CHECKPOINT}"
    )

    if "seed" in checkpoint:

        print(
            f"  Saved seed: "
            f"{checkpoint['seed']}"
        )

    # ---------------------------------------------------------------
    # Final results container.
    # ---------------------------------------------------------------

    results = {
        "metadata": {
            "label": LABEL_NAME,
            "test_total_rows": EXPECTED_TEST_ROWS,
            "test_usable_rows": EXPECTED_USABLE_TEST_ROWS,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": checkpoint_sha256,
            "protocol": str(PROTOCOL_FILE),
            "protocol_sha256": protocol_sha256,
            "evaluation_seed": SEED,
            "training_performed_during_evaluation": False,
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
    # Clean.
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "CLEAN TEST EVALUATION"
    )

    print(
        "=" * 90
    )

    clean = evaluate_condition(
        modules,
        test_loader,
        device,
        missing_modalities=set(),
    )

    results[
        "clean"
    ] = clean

    print_result(
        "All modalities",
        clean,
    )

    # ---------------------------------------------------------------
    # Missing-modality evaluation.
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "MISSING-MODALITY TEST EVALUATION"
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

        print_result(
            name,
            result,
        )

    # ---------------------------------------------------------------
    # Corruption evaluation.
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "CORRUPTION ROBUSTNESS TEST EVALUATION"
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
            f"\n{modality}"
        )

        print(
            "-" * 60
        )

        for severity in [
            1,
            2,
            3,
        ]:

            result = evaluate_condition(
                modules,
                test_loader,
                device,
                missing_modalities=set(),
                corruption_modality=modality,
                corruption_severity=severity,
            )

            results[
                "corruption"
            ][modality][
                str(severity)
            ] = result

            print_result(
                f"{modality} severity {severity}",
                result,
            )

    # ---------------------------------------------------------------
    # Compact summary.
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "FINAL TEST SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Condition':<28}"
        f"{'AUROC':>10}"
        f"{'AUPRC':>10}"
        f"{'Brier':>10}"
    )

    print(
        "-" * 60
    )

    print(
        f"{'All modalities':<28}"
        f"{clean['AUROC']:>10.4f}"
        f"{clean['AUPRC']:>10.4f}"
        f"{clean['Brier']:>10.4f}"
    )

    for name, _ in missing_conditions:

        result = results[
            "missing_modality"
        ][name]

        print(
            f"{name:<28}"
            f"{result['AUROC']:>10.4f}"
            f"{result['AUPRC']:>10.4f}"
            f"{result['Brier']:>10.4f}"
        )

    print(
        "\n" + "=" * 90
    )

    print(
        "FINAL TEST AUROC CHANGE"
    )

    print(
        "=" * 90
    )

    for name, _ in missing_conditions:

        result = results[
            "missing_modality"
        ][name]

        delta = (
            result["AUROC"]
            - clean["AUROC"]
        )

        print(
            f"{name:<28}"
            f"{delta:+.4f}"
        )

    # ---------------------------------------------------------------
    # Corruption summary.
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
        f"{'s1':>10}"
        f"{'s2':>10}"
        f"{'s3':>10}"
    )

    print(
        "-" * 60
    )

    for modality in [
        "CXR",
        "ECG",
        "Labs",
    ]:

        values = []

        for severity in [
            1,
            2,
            3,
        ]:

            values.append(
                results[
                    "corruption"
                ][modality][
                    str(severity)
                ]["AUROC"]
            )

        print(
            f"{modality:<28}"
            f"{values[0]:>10.4f}"
            f"{values[1]:>10.4f}"
            f"{values[2]:>10.4f}"
        )

    # ---------------------------------------------------------------
    # Save results.
    # ---------------------------------------------------------------

    output_path = (
        RESULTS_DIR
        / "final_test_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    results_sha256 = sha256_file(
        output_path
    )

    print(
        "\nSaved final test results:"
        f"\n  {output_path}"
    )

    print(
        "\nResults SHA256:"
        f"\n  {results_sha256}"
    )

    # ---------------------------------------------------------------
    # Completion marker.
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "FINAL HELD-OUT TEST EVALUATION COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        "\nTest split evaluated:"
        f" {EXPECTED_USABLE_TEST_ROWS:,} usable examples."
    )

    print(
        "No training or test-based model selection was performed."
    )


if __name__ == "__main__":
    main()