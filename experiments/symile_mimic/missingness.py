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
import pandas as pd
import torch
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

RESULTS_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIR = (
    RESULTS_DIR / "final_test_table_c_checkpoints"
)

CHECKPOINT_DIR.mkdir(exist_ok=True)

PROTOCOL_FILE = (
    RESULTS_DIR / "final_test_protocol.json"
)

FINAL_TEST_FILE = (
    RESULTS_DIR / "final_test_results.json"
)

OUTPUT_JSON = (
    RESULTS_DIR / "final_test_table_c.json"
)

OUTPUT_CSV = (
    RESULTS_DIR / "final_test_table_c.csv"
)


# =====================================================================
# CONFIGURATION
# =====================================================================

TASK_NAME = "Cardiomegaly"
DATASET_NAME = "Symile-MIMIC"

SEEDS = [0, 1, 2, 3, 4]

EPOCHS = 5
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# METHODS
# =====================================================================

UNIMODAL_METHODS = [
    ("CXR-only", "CXR"),
    ("ECG-only", "ECG"),
    ("Labs-only", "Labs"),
]


FUSION_METHODS = [
    ("Early Fusion", "early_concat"),
    ("Late Average", "late_average"),
    ("Mean Latent", "mean_latent"),
    ("Gaussian PoE", "gaussian_poe"),
    ("Reliability PoE", "reliability_poe"),
    ("Modality Dropout", "modality_dropout"),
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


# =====================================================================
# MISSINGNESS CONDITIONS
# =====================================================================

MISSING_CONDITIONS = [
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
        {
            "CXR",
            "ECG",
        },
    ),
    (
        "CXR + Labs missing",
        {
            "CXR",
            "Labs",
        },
    ),
    (
        "ECG + Labs missing",
        {
            "ECG",
            "Labs",
        },
    ),
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

    h = hashlib.sha256()

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

            h.update(
                chunk
            )

    return h.hexdigest()


# =====================================================================
# DATASET IMPORT
# =====================================================================

def import_symile_dataset():

    package_path = str(
        PROB_EVIDENCE_ROOT
    )

    if package_path not in sys.path:

        sys.path.insert(
            0,
            package_path,
        )

    from datasets.symile_mimic.dataset import SymileMIMICDataset

    return SymileMIMICDataset


# =====================================================================
# REUSE THE EXACT LABELED DATASET FROM THE FUSION IMPLEMENTATION
# =====================================================================

def load_fusion_module():

    return importlib.import_module(
        "experiments.symile_mimic.fusion_comparison"
    )


def build_labeled_dataset(
    split,
):

    fusion_module = load_fusion_module()

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


# =====================================================================
# COMMON LOADERS
# =====================================================================

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

    # ---------------------------------------------------------------
    # Finalized protocol checks.
    # ---------------------------------------------------------------

    if len(train_dataset) != 4232:

        raise RuntimeError(
            "Expected 4,232 usable train examples, "
            f"got {len(train_dataset)}."
        )

    if (
        train_positive != 3346
        or train_negative != 886
    ):

        raise RuntimeError(
            "Expected train class counts 3,346 / 886, "
            f"got {train_positive} / {train_negative}."
        )

    if len(test_dataset) != 1918:

        raise RuntimeError(
            "Expected 1,918 usable test examples, "
            f"got {len(test_dataset)}."
        )

    if (
        test_positive != 1444
        or test_negative != 474
    ):

        raise RuntimeError(
            "Expected test class counts 1,444 / 474, "
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


def aggregate_nested(
    per_seed,
):

    first_seed = per_seed[
        str(
            SEEDS[0]
        )
    ]

    result = {}

    for condition in first_seed:

        result[
            condition
        ] = {}

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
                        condition
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

            if values:

                result[
                    condition
                ][
                    metric
                ] = aggregate_values(
                    values
                )

    return result


# =====================================================================
# UNIMODAL TABLE C
# =====================================================================

def run_unimodal_missingness(
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
        negative_count / positive_count,
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
            "-" * 90
        )

        print(
            f"{display_name} | TEST SEED {seed}"
        )

        print(
            "-" * 90
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

        seed_result = {}

        for condition, missing in (
            MISSING_CONDITIONS
        ):

            if modality in missing:

                seed_result[
                    condition
                ] = {
                    "status":
                        "NA",

                    "reason":
                        (
                            f"{modality} is the sole input "
                            "and is missing."
                        ),
                }

                print(
                    f"  {condition}: N/A"
                )

                continue

            metrics = module.evaluate(
                model,
                test_loader,
            )

            seed_result[
                condition
            ] = {

                "status":
                    "OK",

                "AUROC":
                    float(
                        metrics[
                            "AUROC"
                        ]
                    ),

                "AUPRC":
                    float(
                        metrics[
                            "AUPRC"
                        ]
                    ),

                "Brier":
                    float(
                        metrics[
                            "Brier"
                        ]
                    ),
            }

            print(
                f"  {condition}: "
                f"AUROC="
                f"{metrics['AUROC']:.6f}"
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
            aggregate_nested(
                per_seed
            ),
    }


# =====================================================================
# CLASSICAL FUSION TABLE C
# =====================================================================

def run_fusion_missingness(
    display_name,
    method,
    train_loader,
    test_loader,
    train_positive,
    train_negative,
):

    module = load_fusion_module()

    pos_weight = torch.tensor(
        train_negative / train_positive,
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
            "-" * 90
        )

        print(
            f"{display_name} | TEST SEED {seed}"
        )

        print(
            "-" * 90
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

        for condition, missing in (
            MISSING_CONDITIONS
        ):

            result = (
                module.evaluate_condition(
                    model,
                    test_loader,
                    missing=missing,
                )
            )

            seed_result[
                condition
            ] = {

                "status":
                    "OK",

                "N":
                    int(
                        result[
                            "n"
                        ]
                    ),

                "positive":
                    int(
                        result[
                            "positive"
                        ]
                    ),

                "negative":
                    int(
                        result[
                            "negative"
                        ]
                    ),

                "AUROC":
                    float(
                        result[
                            "AUROC"
                        ]
                    ),

                "AUPRC":
                    float(
                        result[
                            "AUPRC"
                        ]
                    ),

                "Brier":
                    float(
                        result[
                            "Brier"
                        ]
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
                f"  {condition}: "
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
            aggregate_nested(
                per_seed
            ),
    }


# =====================================================================
# RECENT METHOD TABLE C
# =====================================================================

def run_recent_missingness(
    display_name,
    module_name,
    train_loader,
    test_loader,
):

    module = importlib.import_module(
        module_name
    )

    # IMPORTANT:
    # Adapter expects a dictionary and uses .items().
    missing_conditions = {

        "CXR missing":
            {"CXR"},

        "ECG missing":
            {"ECG"},

        "Labs missing":
            {"Labs"},

        "CXR + ECG missing":
            {
                "CXR",
                "ECG",
            },

        "CXR + Labs missing":
            {
                "CXR",
                "Labs",
            },

        "ECG + Labs missing":
            {
                "ECG",
                "Labs",
            },
    }

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

            # Adapter interface names this field val_loader.
            # In this final Table C controller it is the held-out TEST
            # loader and is used only after the adapter's training loop.
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

            # No corruption evaluation in Table C.
            "corruption_modalities":
                [],

            "corruption_severities":
                [],

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

        if (
            "missing_modality"
            not in result
        ):

            raise RuntimeError(
                f"{display_name} did not return "
                "'missing_modality'."
            )

        adapter_missing = result[
            "missing_modality"
        ]

        seed_result = {}

        for condition, missing in (
            MISSING_CONDITIONS
        ):

            if condition not in adapter_missing:

                raise RuntimeError(
                    f"{display_name} did not return "
                    f"{condition}."
                )

            item = adapter_missing[
                condition
            ]

            seed_result[
                condition
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
                f"  {condition}: "
                f"AUROC="
                f"{item['AUROC']:.6f}"
            )

        per_seed[
            str(seed)
        ] = seed_result

        # Adapters encapsulate their model state. Save an audit record
        # rather than pretending to have a state_dict when one was not
        # returned.
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

                    "missing_conditions":
                        [
                            condition
                            for condition, _
                            in MISSING_CONDITIONS
                        ],
                },

                f,

                indent=2,
            )

    return {

        "per_seed":
            per_seed,

        "aggregate":
            aggregate_nested(
                per_seed
            ),
    }


# =====================================================================
# PROPOSED FINAL MODEL
# =====================================================================

def load_proposed_missingness():

    if not FINAL_TEST_FILE.exists():

        raise FileNotFoundError(
            f"Missing finalized result file:\n"
            f"{FINAL_TEST_FILE}"
        )

    with open(
        FINAL_TEST_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    # ---------------------------------------------------------------
    # Search likely containers in the saved final-test JSON.
    # ---------------------------------------------------------------

    containers = [
        data.get("missing_modality"),
        data.get("missing_modality_test"),
        data.get("missing_modality_results"),
        data.get("missing"),
    ]

    # Some versions store all evaluation sections beneath "results".
    if isinstance(
        data.get("results"),
        dict,
    ):

        results_block = data["results"]

        containers.extend([
            results_block.get(
                "missing_modality"
            ),
            results_block.get(
                "missing_modality_test"
            ),
            results_block.get(
                "missing_modality_results"
            ),
            results_block.get(
                "missing"
            ),
        ])

    missing_results = None

    for candidate in containers:

        if isinstance(
            candidate,
            dict,
        ):

            # Make sure this is actually the block containing
            # condition names.
            if any(
                condition in candidate
                for condition, _
                in MISSING_CONDITIONS
            ):

                missing_results = candidate
                break

    if missing_results is None:

        raise RuntimeError(
            "Could not locate the missing-modality "
            "condition block in final_test_results.json.\n"
            f"Top-level keys: {list(data.keys())}"
        )

    output = {}

    # ---------------------------------------------------------------
    # Helper for accommodating upper/lowercase metric names and
    # optional nested metric dictionaries.
    # ---------------------------------------------------------------

    def extract_metric(
        item,
        metric,
    ):

        # Direct representation:
        # {"AUROC": 0.69, ...}
        for key in [
            metric,
            metric.lower(),
        ]:

            if key in item:

                value = item[key]

                if isinstance(
                    value,
                    (int, float),
                ):

                    return float(value)

        # Nested representation:
        # {"metrics": {"AUROC": ...}}
        metrics = item.get(
            "metrics"
        )

        if isinstance(
            metrics,
            dict,
        ):

            for key in [
                metric,
                metric.lower(),
            ]:

                if key in metrics:

                    return float(
                        metrics[key]
                    )

        # Another possible representation:
        # {"reliability": {"AUROC": ...}}
        reliability = item.get(
            "reliability"
        )

        if isinstance(
            reliability,
            dict,
        ):

            for key in [
                metric,
                metric.lower(),
            ]:

                if key in reliability:

                    return float(
                        reliability[key]
                    )

        raise KeyError(
            f"Could not locate {metric}. "
            f"Available keys: {list(item.keys())}"
        )

    # ---------------------------------------------------------------
    # Extract six conditions.
    # ---------------------------------------------------------------

    for condition, missing in (
        MISSING_CONDITIONS
    ):

        item = missing_results.get(
            condition
        )

        if item is None:

            raise RuntimeError(
                f"Missing proposed-model result "
                f"for '{condition}'.\n"
                f"Available conditions: "
                f"{list(missing_results.keys())}"
            )

        if not isinstance(
            item,
            dict,
        ):

            raise RuntimeError(
                f"Unexpected result structure "
                f"for '{condition}': "
                f"{type(item)}"
            )

        auroc = extract_metric(
            item,
            "AUROC",
        )

        auprc = extract_metric(
            item,
            "AUPRC",
        )

        brier = extract_metric(
            item,
            "Brier",
        )

        n = item.get(
            "N",
            item.get(
                "n",
                1918,
            ),
        )

        output[
            condition
        ] = {

            "status":
                "OK",

            "N":
                int(n),

            "AUROC":
                float(auroc),

            "AUPRC":
                float(auprc),

            "Brier":
                float(brier),
        }

        print(
            f"  {condition}: "
            f"AUROC={auroc:.4f} "
            f"AUPRC={auprc:.4f} "
            f"Brier={brier:.4f}"
        )

    return output

# =====================================================================
# PUBLICATION ROWS
# =====================================================================

def build_table_rows(
    unimodal_results,
    fusion_results,
    recent_results,
    proposed_results,
):

    rows = []

    # ---------------------------------------------------------------
    # Unimodal
    # ---------------------------------------------------------------

    for display_name, modality in (
        UNIMODAL_METHODS
    ):

        aggregate_result = (
            unimodal_results[
                display_name
            ][
                "aggregate"
            ]
        )

        for condition, missing in (
            MISSING_CONDITIONS
        ):

            item = aggregate_result.get(
                condition
            )

            if item is None:

                rows.append({

                    "method":
                        display_name,

                    "category":
                        "Unimodal",

                    "condition":
                        condition,

                    "status":
                        "N/A",
                })

                continue

            rows.append({

                "method":
                    display_name,

                "category":
                    "Unimodal",

                "condition":
                    condition,

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
    # Classical fusion
    # ---------------------------------------------------------------

    for display_name, method in (
        FUSION_METHODS
    ):

        aggregate_result = (
            fusion_results[
                display_name
            ][
                "aggregate"
            ]
        )

        for condition, missing in (
            MISSING_CONDITIONS
        ):

            item = aggregate_result[
                condition
            ]

            rows.append({

                "method":
                    display_name,

                "category":
                    "Classical Fusion",

                "condition":
                    condition,

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
    # Recent methods
    # ---------------------------------------------------------------

    for display_name, module_name in (
        RECENT_METHODS
    ):

        aggregate_result = (
            recent_results[
                display_name
            ][
                "aggregate"
            ]
        )

        for condition, missing in (
            MISSING_CONDITIONS
        ):

            item = aggregate_result[
                condition
            ]

            rows.append({

                "method":
                    display_name,

                "category":
                    "Recent Method",

                "condition":
                    condition,

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

    for condition, missing in (
        MISSING_CONDITIONS
    ):

        item = proposed_results[
            condition
        ]

        rows.append({

            "method":
                (
                    "Proposed: Reliability + "
                    "Missingness-Aware"
                ),

            "category":
                "Proposed",

            "condition":
                condition,

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
                "single frozen final checkpoint",
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
        "COMPLETE HELD-OUT TEST TABLE C"
    )

    print(
        "=" * 100
    )

    print(
        "\nMissing-modality robustness comparison."
    )

    print(
        "All trainable comparator models are retrained for Table C."
    )

    print(
        "Each seed is trained on TRAIN only."
    )

    print(
        "Each frozen seed model is evaluated on TEST under all "
        "six missingness patterns."
    )

    print(
        "Recent methods are retrained here; previous Table B "
        "results are NOT reused."
    )

    print(
        "Proposed model uses the finalized frozen checkpoint."
    )

    print(
        "\nNo validation result is relabeled as test performance."
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
    # Load common data
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
        "UNIMODAL HELD-OUT TEST"
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
        ] = run_unimodal_missingness(
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
        "CLASSICAL FUSION HELD-OUT TEST"
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
        ] = run_fusion_missingness(
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
        "RECENT METHOD HELD-OUT TEST"
    )

    print(
        "=" * 100
    )

    print(
        "IMPORTANT: recent methods are TRAINED AGAIN "
        "for Table C."
    )

    recent_results = {}

    for display_name, module_name in (
        RECENT_METHODS
    ):

        recent_results[
            display_name
        ] = run_recent_missingness(
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
        "PROPOSED FROZEN MODEL"
    )

    print(
        "=" * 100
    )

    proposed_results = (
        load_proposed_missingness()
    )

    # ---------------------------------------------------------------
    # TABLE ROWS
    # ---------------------------------------------------------------

    table_rows = build_table_rows(
        unimodal_results,
        fusion_results,
        recent_results,
        proposed_results,
    )

    # ---------------------------------------------------------------
    # SAVE JSON
    # ---------------------------------------------------------------

    payload = {

        "experiment":
            "Complete held-out-test Table C",

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

        "missing_conditions":
            [
                condition
                for condition, _
                in MISSING_CONDITIONS
            ],

        "unimodal_results":
            unimodal_results,

        "classical_fusion_results":
            fusion_results,

        "recent_method_results":
            recent_results,

        "proposed_frozen_results":
            proposed_results,

        "table_c_rows":
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
            default=str,
        )

    # ---------------------------------------------------------------
    # SAVE CSV
    # ---------------------------------------------------------------

    fields = [

        "method",
        "category",
        "condition",
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
        "reason",
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
    # PRINT FINAL TABLE
    # ---------------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL HELD-OUT TEST TABLE C"
    )

    print(
        "=" * 100
    )

    for condition, missing in (
        MISSING_CONDITIONS
    ):

        print()
        print(
            condition
        )

        print(
            "-" * 100
        )

        print(
            f"{'Method':42s}"
            f"{'AUROC':21s}"
            f"{'AUPRC':21s}"
            f"{'Brier':21s}"
        )

        print(
            "-" * 105
        )

        for row in table_rows:

            if (
                row[
                    "condition"
                ]
                != condition
            ):

                continue

            if row[
                "status"
            ] == "N/A":

                print(
                    f"{row['method']:42s}"
                    f"{'N/A':21s}"
                    f"{'N/A':21s}"
                    f"{'N/A':21s}"
                )

                continue

            if row[
                "category"
            ] == "Proposed":

                auroc = (
                    f"{row['AUROC']:.4f}"
                )

                auprc = (
                    f"{row['AUPRC']:.4f}"
                )

                brier = (
                    f"{row['Brier']:.4f}"
                )

            else:

                auroc = (
                    f"{row['AUROC_mean']:.4f}"
                    f" +/- "
                    f"{row['AUROC_std']:.4f}"
                )

                auprc = (
                    f"{row['AUPRC_mean']:.4f}"
                    f" +/- "
                    f"{row['AUPRC_std']:.4f}"
                )

                brier = (
                    f"{row['Brier_mean']:.4f}"
                    f" +/- "
                    f"{row['Brier_std']:.4f}"
                )

            print(
                f"{row['method']:42s}"
                f"{auroc:21s}"
                f"{auprc:21s}"
                f"{brier:21s}"
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

    print()
    print(
        "Recent-method training status:"
    )

    print(
        "  DrFuse, M3Care, MUSE and Flex-MoE "
        "were retrained for Table C."
    )

    print()
    print(
        "=" * 100
    )

    print(
        "COMPLETE HELD-OUT TEST TABLE C COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()