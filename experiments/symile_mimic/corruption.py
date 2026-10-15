from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =====================================================================
# PATHS
# =====================================================================

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SYMILE_DATA_ROOT", ROOT)).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("SYMILE_OUTPUT_ROOT", ROOT / "experiment_results")).expanduser().resolve()

PROB_EVIDENCE_ROOT = ROOT

RESULTS_DIR = OUTPUT_ROOT

RESULTS_DIR.mkdir(
    exist_ok=True
)

CHECKPOINT_DIR = (
    RESULTS_DIR
    / "final_test_table_d_checkpoints"
)

CHECKPOINT_DIR.mkdir(
    exist_ok=True
)

PROTOCOL_FILE = (
    RESULTS_DIR
    / "final_test_protocol.json"
)

FINAL_TEST_FILE = (
    RESULTS_DIR
    / "final_test_results.json"
)

OUTPUT_JSON = (
    RESULTS_DIR
    / "final_test_table_d.json"
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "final_test_table_d.csv"
)


# =====================================================================
# CONFIGURATION
# =====================================================================

DATASET_NAME = "Symile-MIMIC"

TASK_NAME = "Cardiomegaly"

SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

BATCH_SIZE = 16

EPOCHS = 5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# METHODS
# =====================================================================

UNIMODAL_METHODS = [
    (
        "CXR-only",
        "CXR",
    ),
    (
        "ECG-only",
        "ECG",
    ),
    (
        "Labs-only",
        "Labs",
    ),
]


FUSION_METHODS = [
    (
        "Early Fusion",
        "early_concat",
    ),
    (
        "Late Average",
        "late_average",
    ),
    (
        "Mean Latent",
        "mean_latent",
    ),
    (
        "Gaussian PoE",
        "gaussian_poe",
    ),
    (
        "Reliability PoE",
        "reliability_poe",
    ),
    (
        "Modality Dropout",
        "modality_dropout",
    ),
]


RECENT_METHODS = [
    (
        "DrFuse-adapted",
        "recent_method_adapters.drfuse_adapter",
    ),
    (
        "M3Care-adapted",
        "recent_method_adapters.m3care_adapter",
    ),
    (
        "MUSE-adapted",
        "recent_method_adapters.muse_adapter",
    ),
    (
        "Flex-MoE-adapted",
        "recent_method_adapters.flex_moe_adapter",
    ),
]


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


# =====================================================================
# HASH
# =====================================================================

def sha256(
    path,
):

    if not path.exists():

        return None

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


# =====================================================================
# DATASET
# =====================================================================

def import_symile_dataset():

    pkg = str(
        PROB_EVIDENCE_ROOT
    )

    if pkg not in sys.path:

        sys.path.insert(
            0,
            pkg,
        )

    from datasets.symile_mimic.dataset import SymileMIMICDataset

    return SymileMIMICDataset


def load_fusion_module():

    return importlib.import_module(
        "experiments.symile_mimic.fusion_comparison"
    )


def build_labeled_dataset(
    split,
):

    fusion_module = (
        load_fusion_module()
    )

    SymileMIMICDataset = (
        import_symile_dataset()
    )

    tensor_dataset = (
        SymileMIMICDataset(
            data_dir=(
                DATA_ROOT / "data_npy"
            ),
            split=split,
        )
    )

    csv_path = (
        ROOT / f"{split}.csv"
    )

    return fusion_module.LabeledDataset(
        tensor_dataset,
        csv_path,
    )


def build_loaders():

    train_dataset = (
        build_labeled_dataset(
            "train"
        )
    )

    test_dataset = (
        build_labeled_dataset(
            "test"
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    train_positive = int(
        train_dataset.labels.sum().item()
    )

    train_negative = (
        len(train_dataset)
        - train_positive
    )

    test_positive = int(
        test_dataset.labels.sum().item()
    )

    test_negative = (
        len(test_dataset)
        - test_positive
    )

    if len(train_dataset) != 4232:

        raise RuntimeError(
            "Expected 4,232 usable train examples; "
            f"got {len(train_dataset)}."
        )

    if (
        train_positive != 3346
        or train_negative != 886
    ):

        raise RuntimeError(
            "Expected train classes 3,346 / 886; "
            f"got {train_positive} / {train_negative}."
        )

    if len(test_dataset) != 1918:

        raise RuntimeError(
            "Expected 1,918 usable test examples; "
            f"got {len(test_dataset)}."
        )

    if (
        test_positive != 1444
        or test_negative != 474
    ):

        raise RuntimeError(
            "Expected test classes 1,444 / 474; "
            f"got {test_positive} / {test_negative}."
        )

    print()
    print(
        "DATASET CHECK"
    )

    print(
        f"  Train examples : "
        f"{len(train_dataset):,}"
    )

    print(
        f"  Train positive : "
        f"{train_positive:,}"
    )

    print(
        f"  Train negative : "
        f"{train_negative:,}"
    )

    print(
        f"  Test examples  : "
        f"{len(test_dataset):,}"
    )

    print(
        f"  Test positive  : "
        f"{test_positive:,}"
    )

    print(
        f"  Test negative  : "
        f"{test_negative:,}"
    )

    return (
        train_loader,
        test_loader,
        train_positive,
        train_negative,
    )


# =====================================================================
# EXACT CORRUPTION OPERATORS
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
    }[
        severity
    ]

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
    }[
        severity
    ]

    noise = {
        1: 0.02,
        2: 0.05,
        3: 0.10,
    }[
        severity
    ]

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
    }[
        severity
    ]

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
    }[
        severity
    ]

    random_values = (
        torch.rand_like(x)
    )

    return (
        (1.0 - alpha) * x
        + alpha * random_values
    )


# =====================================================================
# GENERIC CORRUPTION
# =====================================================================

def corrupt_input(
    x,
    modality,
    severity,
):

    if modality == "CXR":

        return corrupt_cxr(
            x,
            severity,
        )

    if modality == "ECG":

        return corrupt_ecg(
            x,
            severity,
        )

    if modality == "Labs":

        return corrupt_labs(
            x,
            severity,
        )

    raise ValueError(
        f"Unknown modality: {modality}"
    )


# =====================================================================
# AGGREGATION
# =====================================================================

def aggregate_values(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    return {

        "mean":
            float(
                values.mean()
            ),

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


def aggregate_corruption(
    per_seed,
):

    output = {}

    for modality in (
        CORRUPTION_MODALITIES
    ):

        output[
            modality
        ] = {}

        for severity in (
            CORRUPTION_SEVERITIES
        ):

            key = str(
                severity
            )

            output[
                modality
            ][key] = {}

            for metric in (
                "AUROC",
                "AUPRC",
                "Brier",
            ):

                values = []

                for seed in SEEDS:

                    item = (
                        per_seed[
                            str(seed)
                        ][
                            modality
                        ][
                            key
                        ]
                    )

                    if item.get(
                        "status"
                    ) != "OK":

                        continue

                    values.append(
                        float(
                            item[
                                metric
                            ]
                        )
                    )

                output[
                    modality
                ][key][
                    metric
                ] = aggregate_values(
                    values
                )

    return output


# =====================================================================
# UNIMODAL CORRUPTION EVALUATION
# =====================================================================

def run_unimodal_corruption(
    display_name,
    modality,
    train_tensor,
    test_tensor,
):

    module = importlib.import_module(
        "experiments.symile_mimic.unimodal_comparison"
    )

    train_dataset = (
        module.UnimodalDataset(
            train_tensor,
            DATA_ROOT / "train.csv",
            modality,
        )
    )

    test_dataset = (
        module.UnimodalDataset(
            test_tensor,
            DATA_ROOT / "test.csv",
            modality,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=module.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=module.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    positive_count = int(
        train_dataset.labels.sum().item()
    )

    negative_count = (
        len(train_dataset)
        - positive_count
    )

    pos_weight = torch.tensor(
        negative_count
        / positive_count,
        dtype=torch.float32,
        device=module.DEVICE,
    )

    per_seed = {}

    checkpoint_dir = (
        CHECKPOINT_DIR
        / display_name.replace(
            " ",
            "_",
        )
    )

    checkpoint_dir.mkdir(
        exist_ok=True
    )

    for seed in SEEDS:

        print()
        print(
            "=" * 90
        )

        print(
            f"{display_name} | TEST SEED {seed}"
        )

        print(
            "=" * 90
        )

        module.set_seed(
            seed
        )

        model = module.build_model(
            modality
        )

        module.train_one_seed(
            model,
            train_loader,
            pos_weight,
        )

        model.eval()

        seed_result = {}

        for corruption_modality in (
            CORRUPTION_MODALITIES
        ):

            seed_result[
                corruption_modality
            ] = {}

            for severity in (
                CORRUPTION_SEVERITIES
            ):

                labels_all = []
                probabilities_all = []

                with torch.no_grad():

                    for batch in test_loader:

                        x = batch[
                            "x"
                        ].to(
                            module.DEVICE
                        )

                        y = batch[
                            "label"
                        ].cpu().numpy()

                        # A unimodal model only observes its own
                        # modality. Therefore:
                        #
                        #   - corrupting its own modality changes its
                        #     prediction;
                        #   - corrupting another modality does not
                        #     change its prediction.
                        #
                        if (
                            corruption_modality
                            == modality
                        ):

                            x = (
                                corrupt_input(
                                    x,
                                    modality,
                                    severity,
                                )
                            )

                        embedding = model[
                            "encoder"
                        ](
                            x
                        )

                        logits = model[
                            "prediction"
                        ](
                            embedding
                        )

                        probabilities = (
                            torch.sigmoid(
                                logits
                            )
                        )

                        labels_all.extend(
                            y.tolist()
                        )

                        probabilities_all.extend(
                            probabilities.cpu()
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

                # Import metrics here so that this controller
                # remains dependent on the exact established model
                # implementation only.
                from sklearn.metrics import (
                    roc_auc_score,
                    average_precision_score,
                )

                seed_result[
                    corruption_modality
                ][
                    str(severity)
                ] = {

                    "status":
                        "OK",

                    "N":
                        int(
                            len(y)
                        ),

                    "positive":
                        int(
                            np.sum(y == 1)
                        ),

                    "negative":
                        int(
                            np.sum(y == 0)
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
                            np.mean(
                                (
                                    p - y
                                ) ** 2
                            )
                        ),
                }

                print(
                    f"  {corruption_modality} "
                    f"s{severity}: "
                    f"AUROC="
                    f"{seed_result[corruption_modality][str(severity)]['AUROC']:.6f}"
                )

        per_seed[
            str(seed)
        ] = seed_result

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "seed":
                    seed,

                "method":
                    display_name,

                "modality":
                    modality,

                "task":
                    TASK_NAME,

                "evaluation_split":
                    "test",
            },

            checkpoint_dir
            / f"seed_{seed}.pt",
        )

    return {

        "per_seed":
            per_seed,

        "aggregate":
            aggregate_corruption(
                per_seed
            ),
    }


# =====================================================================
# CLASSICAL FUSION CORRUPTION
# =====================================================================

def run_fusion_corruption(
    display_name,
    method,
    train_loader,
    test_loader,
    train_positive,
    train_negative,
):

    module = load_fusion_module()

    pos_weight = torch.tensor(
        train_negative
        / train_positive,
        dtype=torch.float32,
        device=module.DEVICE,
    )

    per_seed = {}

    checkpoint_dir = (
        CHECKPOINT_DIR
        / display_name.replace(
            " ",
            "_",
        )
    )

    checkpoint_dir.mkdir(
        exist_ok=True
    )

    for seed in SEEDS:

        print()
        print(
            "=" * 90
        )

        print(
            f"{display_name} | TEST SEED {seed}"
        )

        print(
            "=" * 90
        )

        module.set_seed(
            seed
        )

        model = (
            module.FusionModel(
                method
            )
            .to(
                module.DEVICE
            )
        )

        module.train_model(
            model,
            train_loader,
            method,
            pos_weight,
        )

        seed_result = {}

        for modality in (
            CORRUPTION_MODALITIES
        ):

            seed_result[
                modality
            ] = {}

            for severity in (
                CORRUPTION_SEVERITIES
            ):

                result = (
                    module.evaluate_condition(
                        model,
                        test_loader,
                        missing=set(),
                        corruption_modality=modality,
                        corruption_severity=severity,
                    )
                )

                seed_result[
                    modality
                ][
                    str(severity)
                ] = {

                    "status":
                        "OK",

                    "N":
                        int(
                            result["n"]
                        ),

                    "positive":
                        int(
                            result["positive"]
                        ),

                    "negative":
                        int(
                            result["negative"]
                        ),

                    "AUROC":
                        float(
                            result["AUROC"]
                        ),

                    "AUPRC":
                        float(
                            result["AUPRC"]
                        ),

                    "Brier":
                        float(
                            result["Brier"]
                        ),

                    "variance":
                        float(
                            result.get(
                                "variance",
                                0.0,
                            )
                        ),

                    "reliability":
                        result.get(
                            "reliability",
                            {},
                        ),
                }

                print(
                    f"  {modality} "
                    f"s{severity}: "
                    f"AUROC="
                    f"{result['AUROC']:.6f}"
                )

        per_seed[
            str(seed)
        ] = seed_result

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "seed":
                    seed,

                "method":
                    method,

                "display_name":
                    display_name,

                "task":
                    TASK_NAME,

                "evaluation_split":
                    "test",
            },

            checkpoint_dir
            / f"seed_{seed}.pt",
        )

    return {

        "per_seed":
            per_seed,

        "aggregate":
            aggregate_corruption(
                per_seed
            ),
    }


# =====================================================================
# RECENT-METHOD CORRUPTION
# =====================================================================

def run_recent_corruption(
    display_name,
    module_name,
    train_loader,
    test_loader,
):

    module = importlib.import_module(
        module_name
    )

    missing_conditions = {

        "CXR missing":
            {"CXR"},

        "ECG missing":
            {"ECG"},

        "Labs missing":
            {"Labs"},

        "CXR + ECG missing":
            {"CXR", "ECG"},

        "CXR + Labs missing":
            {"CXR", "Labs"},

        "ECG + Labs missing":
            {"ECG", "Labs"},
    }

    corruption_modalities = [
        "CXR",
        "ECG",
        "Labs",
    ]

    corruption_severities = [
        1,
        2,
        3,
    ]

    per_seed = {}

    checkpoint_dir = (
        CHECKPOINT_DIR
        / display_name.replace(
            " ",
            "_",
        )
    )

    checkpoint_dir.mkdir(
        exist_ok=True
    )

    for seed in SEEDS:

        print()
        print(
            "=" * 90
        )

        print(
            f"{display_name} | TEST SEED {seed}"
        )

        print(
            "=" * 90
        )

        set_seed(
            seed
        )

        adapter = module.Adapter()

        context = {

            "device":
                DEVICE,

            "train_loader":
                train_loader,

            # Adapter interface calls this val_loader.
            # For this final test benchmark it is the held-out TEST
            # loader and is used only during post-training evaluation.
            "val_loader":
                test_loader,

            "modalities":
                [
                    "CXR",
                    "ECG",
                    "Labs",
                ],

            "missing_conditions":
                missing_conditions,

            "corruption_modalities":
                corruption_modalities,

            "corruption_severities":
                corruption_severities,

            "epochs":
                EPOCHS,
        }

        result = adapter.run(
            seed=seed,
            context=context,
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                f"{display_name} returned "
                "an invalid result."
            )

        if "corruption" not in result:

            raise RuntimeError(
                f"{display_name} did not return "
                "'corruption' results."
            )

        corruption_result = (
            result[
                "corruption"
            ]
        )

        seed_result = {}

        for modality in (
            CORRUPTION_MODALITIES
        ):

            if modality not in (
                corruption_result
            ):

                raise RuntimeError(
                    f"{display_name} missing "
                    f"corruption modality {modality}."
                )

            seed_result[
                modality
            ] = {}

            for severity in (
                CORRUPTION_SEVERITIES
            ):

                key = str(
                    severity
                )

                if key not in (
                    corruption_result[
                        modality
                    ]
                ):

                    raise RuntimeError(
                        f"{display_name} missing "
                        f"{modality} severity {severity}."
                    )

                item = (
                    corruption_result[
                        modality
                    ][
                        key
                    ]
                )

                seed_result[
                    modality
                ][
                    key
                ] = {

                    "status":
                        "OK",

                    "N":
                        int(
                            item.get(
                                "N",
                                1918,
                            )
                        ),

                    "positive":
                        int(
                            item.get(
                                "positive",
                                1444,
                            )
                        ),

                    "negative":
                        int(
                            item.get(
                                "negative",
                                474,
                            )
                        ),

                    "AUROC":
                        float(
                            item[
                                "AUROC"
                            ]
                        ),

                    "AUPRC":
                        float(
                            item[
                                "AUPRC"
                            ]
                        ),

                    "Brier":
                        float(
                            item[
                                "Brier"
                            ]
                        ),
                }

                print(
                    f"  {modality} "
                    f"s{severity}: "
                    f"AUROC="
                    f"{item['AUROC']:.6f}"
                )

        per_seed[
            str(seed)
        ] = seed_result

        with open(
            checkpoint_dir
            / f"seed_{seed}_completed.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "seed":
                        seed,

                    "method":
                        display_name,

                    "task":
                        TASK_NAME,

                    "evaluation_split":
                        "test",

                    "corruption_modalities":
                        corruption_modalities,

                    "corruption_severities":
                        corruption_severities,
                },
                f,
                indent=2,
            )

    return {

        "per_seed":
            per_seed,

        "aggregate":
            aggregate_corruption(
                per_seed
            ),
    }


# =====================================================================
# LOAD PROPOSED FROZEN CORRUPTION RESULTS
# =====================================================================

def load_proposed_corruption():

    if not FINAL_TEST_FILE.exists():

        raise FileNotFoundError(
            f"Missing finalized test file:\n"
            f"{FINAL_TEST_FILE}"
        )

    with open(
        FINAL_TEST_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    corruption = (
        data.get(
            "corruption"
        )
    )

    if corruption is None:

        corruption = (
            data.get(
                "corruption_test"
            )
        )

    if corruption is None:

        corruption = (
            data.get(
                "corruption_results"
            )
        )

    if not isinstance(
        corruption,
        dict,
    ):

        raise RuntimeError(
            "Could not locate corruption results "
            "in final_test_results.json."
        )

    output = {}

    for modality in (
        CORRUPTION_MODALITIES
    ):

        if modality not in corruption:

            raise RuntimeError(
                f"Proposed result missing "
                f"corruption modality {modality}."
            )

        output[
            modality
        ] = {}

        for severity in (
            CORRUPTION_SEVERITIES
        ):

            key = str(
                severity
            )

            item = corruption[
                modality
            ].get(
                key
            )

            if item is None:

                raise RuntimeError(
                    f"Proposed result missing "
                    f"{modality} severity {severity}."
                )

            output[
                modality
            ][key] = {

                "status":
                    "OK",

                "N":
                    int(
                        item.get(
                            "N",
                            1918,
                        )
                    ),

                "AUROC":
                    float(
                        item[
                            "AUROC"
                        ]
                    ),

                "AUPRC":
                    float(
                        item[
                            "AUPRC"
                        ]
                    ),

                "Brier":
                    float(
                        item[
                            "Brier"
                        ]
                    ),
            }

    return output


# =====================================================================
# BUILD TABLE D ROWS
# =====================================================================

def build_table_rows(
    unimodal_results,
    fusion_results,
    recent_results,
    proposed_results,
):

    rows = []

    method_order = (
        [
            x[0]
            for x in UNIMODAL_METHODS
        ]
        +
        [
            x[0]
            for x in FUSION_METHODS
        ]
        +
        [
            x[0]
            for x in RECENT_METHODS
        ]
        +
        [
            "Proposed: Reliability + Missingness-Aware"
        ]
    )

    result_lookup = {}

    for name, value in (
        unimodal_results.items()
    ):

        result_lookup[
            name
        ] = value

    for name, value in (
        fusion_results.items()
    ):

        result_lookup[
            name
        ] = value

    for name, value in (
        recent_results.items()
    ):

        result_lookup[
            name
        ] = value

    # ---------------------------------------------------------------
    # Trainable comparator rows
    # ---------------------------------------------------------------

    for display_name in method_order:

        if display_name.startswith(
            "Proposed:"
        ):

            continue

        aggregate = (
            result_lookup[
                display_name
            ][
                "aggregate"
            ]
        )

        for modality in (
            CORRUPTION_MODALITIES
        ):

            for severity in (
                CORRUPTION_SEVERITIES
            ):

                key = str(
                    severity
                )

                item = aggregate[
                    modality
                ][
                    key
                ]

                rows.append({

                    "method":
                        display_name,

                    "condition":
                        f"{modality} s{severity}",

                    "modality":
                        modality,

                    "severity":
                        severity,

                    "status":
                        "OK",

                    "AUROC_mean":
                        item[
                            "AUROC"
                        ][
                            "mean"
                        ],

                    "AUROC_std":
                        item[
                            "AUROC"
                        ][
                            "std"
                        ],

                    "AUPRC_mean":
                        item[
                            "AUPRC"
                        ][
                            "mean"
                        ],

                    "AUPRC_std":
                        item[
                            "AUPRC"
                        ][
                            "std"
                        ],

                    "Brier_mean":
                        item[
                            "Brier"
                        ][
                            "mean"
                        ],

                    "Brier_std":
                        item[
                            "Brier"
                        ][
                            "std"
                        ],

                    "seeds":
                        SEEDS,

                    "evaluation_type":
                        (
                            "five-seed "
                            "train-on-train/evaluate-on-test"
                        ),
                })

    # ---------------------------------------------------------------
    # Proposed
    # ---------------------------------------------------------------

    for modality in (
        CORRUPTION_MODALITIES
    ):

        for severity in (
            CORRUPTION_SEVERITIES
        ):

            key = str(
                severity
            )

            item = proposed_results[
                modality
            ][
                key
            ]

            rows.append({

                "method":
                    (
                        "Proposed: Reliability + "
                        "Missingness-Aware"
                    ),

                "condition":
                    f"{modality} s{severity}",

                "modality":
                    modality,

                "severity":
                    severity,

                "status":
                    "OK",

                "AUROC":
                    item[
                        "AUROC"
                    ],

                "AUPRC":
                    item[
                        "AUPRC"
                    ],

                "Brier":
                    item[
                        "Brier"
                    ],

                "N":
                    item[
                        "N"
                    ],

                "evaluation_type":
                    (
                        "single frozen "
                        "final checkpoint"
                    ),
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
        "COMPLETE HELD-OUT TEST TABLE D"
    )

    print(
        "=" * 100
    )

    print(
        "\nCorruption robustness experiment."
    )

    print(
        "All 13 comparator methods are retrained."
    )

    print(
        "Each seed is trained using TRAIN only."
    )

    print(
        "Each frozen seed model is evaluated on TEST."
    )

    print(
        "Corruption conditions: "
        "CXR/ECG/Labs x severity 1/2/3."
    )

    print(
        "Recent methods are retrained in Table D."
    )

    print(
        "The proposed model remains frozen."
    )

    print(
        "\nNo validation metrics are reused as test metrics."
    )

    print(
        "No test-based model selection is performed."
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        "Protocol SHA256:"
        f" {sha256(PROTOCOL_FILE)}"
    )

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------

    (
        train_loader,
        test_loader,
        train_positive,
        train_negative,
    ) = build_loaders()

    SymileMIMICDataset = (
        import_symile_dataset()
    )

    train_tensor = (
        SymileMIMICDataset(
            data_dir=(
                DATA_ROOT / "data_npy"
            ),
            split="train",
        )
    )

    test_tensor = (
        SymileMIMICDataset(
            data_dir=(
                DATA_ROOT / "data_npy"
            ),
            split="test",
        )
    )

    # ---------------------------------------------------------------
    # UNIMODAL
    # ---------------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "UNIMODAL HELD-OUT TEST - CORRUPTION"
    )

    print(
        "=" * 100
    )

    unimodal_results = {}

    for display_name, modality in (
        UNIMODAL_METHODS
    ):

        unimodal_results[
            display_name
        ] = run_unimodal_corruption(
            display_name,
            modality,
            train_tensor,
            test_tensor,
        )

    # ---------------------------------------------------------------
    # CLASSICAL FUSION
    # ---------------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "CLASSICAL FUSION HELD-OUT TEST - CORRUPTION"
    )

    print(
        "=" * 100
    )

    fusion_results = {}

    for display_name, method in (
        FUSION_METHODS
    ):

        fusion_results[
            display_name
        ] = run_fusion_corruption(
            display_name,
            method,
            train_loader,
            test_loader,
            train_positive,
            train_negative,
        )

    # ---------------------------------------------------------------
    # RECENT METHODS
    # ---------------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "RECENT METHODS HELD-OUT TEST - CORRUPTION"
    )

    print(
        "=" * 100
    )

    print(
        "IMPORTANT: these methods are retrained here "
        "for Table D."
    )

    recent_results = {}

    for display_name, module_name in (
        RECENT_METHODS
    ):

        recent_results[
            display_name
        ] = run_recent_corruption(
            display_name,
            module_name,
            train_loader,
            test_loader,
        )

    # ---------------------------------------------------------------
    # PROPOSED
    # ---------------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "PROPOSED FROZEN MODEL - CORRUPTION"
    )

    print(
        "=" * 100
    )

    proposed_results = (
        load_proposed_corruption()
    )

    for modality in (
        CORRUPTION_MODALITIES
    ):

        for severity in (
            CORRUPTION_SEVERITIES
        ):

            item = proposed_results[
                modality
            ][
                str(severity)
            ]

            print(
                f"  {modality} s{severity}: "
                f"AUROC="
                f"{item['AUROC']:.6f}"
            )

    # ---------------------------------------------------------------
    # Rows
    # ---------------------------------------------------------------

    table_rows = build_table_rows(
        unimodal_results,
        fusion_results,
        recent_results,
        proposed_results,
    )

    # ---------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------

    payload = {

        "experiment":
            "Complete held-out-test Table D",

        "dataset":
            DATASET_NAME,

        "task":
            TASK_NAME,

        "train_examples":
            4232,

        "train_positive":
            3346,

        "train_negative":
            886,

        "test_examples":
            1918,

        "test_positive":
            1444,

        "test_negative":
            474,

        "seeds":
            SEEDS,

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "device":
            str(
                DEVICE
            ),

        "protocol_sha256":
            sha256(
                PROTOCOL_FILE
            ),

        "validation_results_reused_as_test":
            False,

        "test_selection_used":
            False,

        "recent_methods_retrained":
            True,

        "corruption_modalities":
            CORRUPTION_MODALITIES,

        "corruption_severities":
            CORRUPTION_SEVERITIES,

        "unimodal_results":
            unimodal_results,

        "classical_fusion_results":
            fusion_results,

        "recent_method_results":
            recent_results,

        "proposed_frozen_results":
            proposed_results,

        "table_d_rows":
            table_rows,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # CSV
    # ---------------------------------------------------------------

    fields = [

        "method",
        "condition",
        "modality",
        "severity",
        "status",

        "AUROC_mean",
        "AUROC_std",

        "AUPRC_mean",
        "AUPRC_std",

        "Brier_mean",
        "Brier_std",

        "AUROC",
        "AUPRC",
        "Brier",

        "N",
        "evaluation_type",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            table_rows
        )

    # ---------------------------------------------------------------
    # Final console tables
    # ---------------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL HELD-OUT TEST TABLE D"
    )

    print(
        "=" * 100
    )

    method_order = (
        [
            x[0]
            for x in UNIMODAL_METHODS
        ]
        +
        [
            x[0]
            for x in FUSION_METHODS
        ]
        +
        [
            x[0]
            for x in RECENT_METHODS
        ]
        +
        [
            "Proposed: Reliability + Missingness-Aware"
        ]
    )

    for modality in (
        CORRUPTION_MODALITIES
    ):

        print()
        print(
            modality
        )

        print(
            "-" * 100
        )

        print(
            f"{'Method':42s}"
            f"{'s1':19s}"
            f"{'s2':19s}"
            f"{'s3':19s}"
        )

        print(
            "-" * 100
        )

        for method_name in (
            method_order
        ):

            values = []

            for severity in (
                CORRUPTION_SEVERITIES
            ):

                rows = [
                    row
                    for row in table_rows
                    if (
                        row[
                            "method"
                        ]
                        == method_name
                        and
                        row[
                            "modality"
                        ]
                        == modality
                        and
                        row[
                            "severity"
                        ]
                        == severity
                    )
                ]

                if not rows:

                    values.append(
                        "N/A"
                    )

                    continue

                row = rows[0]

                if method_name.startswith(
                    "Proposed:"
                ):
                    value = (
                        f"{row['AUROC']:.4f}"
                    )
                else:
                    value = (
                        f"{row['AUROC_mean']:.4f}"
                        f" +/- "
                        f"{row['AUROC_std']:.4f}"
                    )

                values.append(
                    value
                )

            print(
                f"{method_name:42s}"
                f"{values[0]:19s}"
                f"{values[1]:19s}"
                f"{values[2]:19s}"
            )

    print()
    print(
        "Saved JSON:"
    )

    print(
        f"  {OUTPUT_JSON}"
    )

    print(
        "Saved CSV:"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    print(
        "Per-seed checkpoints/records:"
    )

    print(
        f"  {CHECKPOINT_DIR}"
    )

    print()
    print(
        "=" * 100
    )

    print(
        "COMPLETE HELD-OUT TEST TABLE D COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()