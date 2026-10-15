# Reliability-Aware Probabilistic Fusion

Research code accompanying **Reliability-Aware Probabilistic Fusion for Multimodal Learning with Missing and Corrupted Modalities**.

This repository contains the implementation and experimental code accompanying Reliability-Aware Probabilistic Fusion for Multimodal Learning with Missing and Corrupted Modalities. Paths and documentation have been made portable, while provenance-sensitive model and training behavior is intentionally preserved. The repository may receive small updates later.

## Overview

The method represents each available modality as a diagonal Gaussian expert and learns a separate reliability score controlling how strongly that expert contributes to Product-of-Experts (PoE) fusion. For modality `m`, the effective precision is `reliability / variance`. Missing modalities are omitted from fusion rather than imputed into the PoE.

Experiments cover two clinical multimodal settings:

- **Symile-MIMIC:** cardiomegaly prediction from CXR, ECG, and laboratory measurements.
- **HAIM-MIMIC-MM:** 48-hour mortality prediction from structured data, physiological time series, CXR, and clinical text representations.

This release is intentionally **stage-by-stage**, not a one-command reproduction pipeline. The underlying datasets must be obtained separately, and preprocessing is part of the experimental procedure.

## Repository layout

```text
.
├── src/reliability_fusion/       # Shared probabilistic-fusion implementation
├── datasets/
│   ├── symile_mimic/             # Processed-array loader and validation
│   └── haim_mimic_mm/            # HAIM cohort/split/feature preparation
├── experiments/
│   ├── symile_mimic/             # Training and evaluation scripts
│   └── haim_mimic_mm/            # Training and evaluation scripts
├── configs/                       # Reference experiment settings
├── tests/                         # Provenance/equivalence tests
├── docs/PROVENANCE.md
├── docs/REPRODUCIBILITY.md
└── pyproject.toml
```

The YAML files under `configs/` are reference snapshots of important experimental settings. The final research scripts remain the source of truth.

## 1. Installation

Python 3.10 or later is recommended. Create an isolated environment and install the repository in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

Install the appropriate CUDA-enabled PyTorch build first if your system requires a platform-specific PyTorch installation.

To run the provenance tests:

```bash
pip install pytest
pytest -q
```

The tests check the Gaussian PoE equations, reliability-weighted PoE, evidence constraints, dataset-specific consistency semantics, and HAIM checkpoint-key compatibility.

---

# 2. Symile-MIMIC

## 2.1 Obtain and prepare the data

The clinical data are **not distributed with this repository**. Obtain the required Symile-MIMIC data through its official access procedure and comply with the applicable data-use terms.

The research archive available for this reconstruction contained the final processed arrays and loader, but **did not contain the original raw-to-NumPy export pipeline**. For that reason, this repository does not claim to recreate that unavailable stage. Training begins from the processed representation used by the experiments.

Create a data directory with the following layout:

```text
/path/to/symile_data/
├── train.csv
├── val.csv
├── test.csv
└── data_npy/
    ├── train/
    │   ├── cxr_train.npy
    │   ├── ecg_train.npy
    │   ├── labs_percentiles_train.npy
    │   ├── labs_missingness_train.npy
    │   └── hadm_id_train.npy
    ├── val/
    │   ├── cxr_val.npy
    │   ├── ecg_val.npy
    │   ├── labs_percentiles_val.npy
    │   ├── labs_missingness_val.npy
    │   └── hadm_id_val.npy
    └── test/
        ├── cxr_test.npy
        ├── ecg_test.npy
        ├── labs_percentiles_test.npy
        ├── labs_missingness_test.npy
        └── hadm_id_test.npy
```

Expected modality shapes for each split are:

```text
CXR                 (N, 3, 320, 320)
ECG                 (N, 1, 5000, 12)
lab percentiles     (N, 50)
lab missingness     (N, 50)
```

Each split CSV must contain `hadm_id` in the same order as the corresponding `hadm_id_<split>.npy`, together with the `Cardiomegaly` target used by the experiment.

Validate an existing processed export with:

```bash
python -m datasets.symile_mimic.preprocessing \
    --data-dir /path/to/symile_data/data_npy
```

## 2.2 Train the proposed model

Set the data and output locations and run the final missing-modality-aware training implementation:

```bash
export SYMILE_DATA_ROOT=/path/to/symile_data
export SYMILE_OUTPUT_ROOT=/path/to/symile_results

python -m experiments.symile_mimic.train
```

On Windows PowerShell:

```powershell
$env:SYMILE_DATA_ROOT="C:\path\to\symile_data"
$env:SYMILE_OUTPUT_ROOT="C:\path\to\symile_results"
python -m experiments.symile_mimic.train
```

The principal checkpoint is written as:

```text
/path/to/symile_results/missing_modality_aware_model.pt
```

The final Symile implementation preserves the original training behavior, including explicit corruption-severity reliability ranking and probability-space complete-to-incomplete consistency.

## 2.3 Evaluate the held-out test set

After training:

```bash
python -m experiments.symile_mimic.evaluate_test \
    --data-root /path/to/symile_data \
    --checkpoint /path/to/symile_results/missing_modality_aware_model.pt \
    --output-dir /path/to/symile_results
```

This evaluates the complete-modality test condition and writes `test_metrics.json`.

### Robustness evaluation of the proposed checkpoint

To evaluate the same frozen checkpoint under missing-modality and controlled-corruption conditions, run:

```bash
python -m experiments.symile_mimic.evaluate_robustness
```

This uses `SYMILE_DATA_ROOT` and `SYMILE_OUTPUT_ROOT`, writes `final_test_results.json`, and creates a portable `final_test_protocol.json` record before the first held-out robustness evaluation if one is not already present. The downstream missingness and corruption comparison scripts read the proposed-model results from `final_test_results.json`.

### Multi-seed training

The manuscript Symile confirmation uses seeds `0, 1, 2, 3, 4`. The thin driver below invokes the unchanged final trainer independently for each seed:

```bash
python -m experiments.symile_mimic.multiseed \
    --data-root /path/to/symile_data \
    --output-root /path/to/symile_results/multiseed
```

## 2.4 Additional analyses

The experiment directory also contains the research scripts used for missing-modality, corruption, ablation, fusion, and unimodal analyses:

```text
experiments/symile_mimic/missingness.py
experiments/symile_mimic/corruption.py
experiments/symile_mimic/ablations.py
experiments/symile_mimic/fusion_comparison.py
experiments/symile_mimic/unimodal_comparison.py
```

These scripts preserve the original final-test protocols and may train or expect additional comparison checkpoints. They are included for provenance and detailed reproduction rather than presented as a single automated benchmark command.

---

# 3. HAIM-MIMIC-MM

## 3.1 Obtain the data

The HAIM/MIMIC source data are **not distributed with this repository**. Obtain the official HAIM-MIMIC-MM release and any required MIMIC access through the corresponding official access procedures.

The preprocessing code expects the official HAIM release beneath the supplied HAIM root, including:

```text
/path/to/haim_root/
└── files/
    └── haim-multimodal/
        └── 1.0.1/
            ├── haim_mimiciv_key_ids.csv
            └── Extracted_HAIM_Embeddings/
                └── cxr_ic_fusion_1103.csv
```

## 3.2 Build the mortality cohort, split, and model features

Run the three preprocessing stages in order:

```bash
python -m datasets.haim_mimic_mm.preprocessing \
    --haim-root /path/to/haim_root \
    --output-dir /path/to/haim_processed
```

This runs:

```text
1. build_mortality_cohort.py
2. build_master_split.py
3. build_model_features.py
```

The resulting model directory contains the frozen split, labels, and modality features:

```text
/path/to/haim_processed/model_data/
├── X_structured.npy
├── X_timeseries.npy
├── X_cxr.npy
├── X_text.npy
├── y.npy
└── split.npy
```

The modality dimensions used by the final implementation are:

```text
structured     1542
timeseries      451
CXR            2084
text           2304
```

## 3.3 Prepare the shared train-derived normalization

The final HAIM experiments use the same training-set-derived normalization as the controlled A/B/C ablations. The original experiment sequence therefore creates these statistics with the A/B/C script:

```bash
export HAIM_ROOT=/path/to/haim_root
export HAIM_PROCESSED_DIR=/path/to/haim_processed
export HAIM_RESULTS_ROOT=/path/to/haim_results

python -m experiments.haim_mimic_mm.ablations \
    --output-dir /path/to/haim_results/ablation_abc
```

This creates `normalization_stats.npz` together with the A/B/C checkpoints. Keeping this step explicit preserves the normalization provenance of the original experiments.

## 3.4 Train the proposed HAIM model

Train the final F model:

```bash
python -m experiments.haim_mimic_mm.train \
    --seed 42 \
    --output-dir /path/to/haim_results/full_f
```

The selected checkpoint is:

```text
/path/to/haim_results/full_f/F_full_best.pt
```

Checkpoint selection is based on clean validation AUROC. The training script intentionally does **not** evaluate the held-out test split.

The final HAIM F implementation uses reliability-weighted PoE, missing-modality-aware mortality training, and logit-space complete-to-incomplete consistency. It does not use the explicit corruption-ranking reliability loss used by the Symile training implementation.

## 3.5 Evaluate the held-out test set

Only after training/model selection, run:

```bash
python -m experiments.haim_mimic_mm.evaluate_test \
    --haim-root /path/to/haim_root \
    --processed-dir /path/to/haim_processed \
    --results-root /path/to/haim_results \
    --checkpoint /path/to/haim_results/full_f/F_full_best.pt
```

The default output is:

```text
/path/to/haim_results/full_f/test_metrics.json
```

## 3.6 Missing-modality and corruption analyses

The corresponding research evaluation scripts are:

```text
experiments/haim_mimic_mm/missingness.py
experiments/haim_mimic_mm/corruption.py
experiments/haim_mimic_mm/fusion_comparison.py
```

The missingness and corruption scripts compare several controlled ablation checkpoints (for example D/E/F or C/E/F), so run the required ablation training scripts before those comparisons:

```text
experiments/haim_mimic_mm/ablations.py
experiments/haim_mimic_mm/train_missing_aware.py
experiments/haim_mimic_mm/train_missing_reliability.py
experiments/haim_mimic_mm/train.py
```

The manuscript results use five independent seeds, `0, 1, 2, 3, 4`, for HAIM as well as Symile. The supplied development archive predates the final five-seed reruns; use the release driver below for the manuscript protocol:

```bash
python -m experiments.haim_mimic_mm.multiseed \
    --output-root /path/to/haim_results/multiseed
```

---

### Implementation provenance note

HAIM experiment drivers import the canonical shared model implementation directly. The final Symile training/analysis scripts intentionally retain their original in-script architecture and fusion definitions to preserve checkpoint and execution provenance. Equivalent extracted components live under `src/reliability_fusion` and are covered by provenance/equivalence tests.

# 4. Reproducibility and provenance

This repository was reconstructed from the final research implementations rather than independently reimplementing the method from the manuscript. Important invariants include:

```text
embedding dimension       128
latent dimension           32
Gaussian logvar clamp     [-6, 4]
reliability input          [mu, logvar]
PoE prior                  N(0, I)
missing modality           expert omitted from fusion
```

Two dataset-specific choices are deliberately preserved rather than homogenized:

- **Symile-MIMIC:** explicit corruption-severity reliability ranking and probability-space consistency.
- **HAIM-MIMIC-MM:** final F training uses missing-aware mortality training and logit-space consistency without explicit corruption-ranking reliability supervision.

See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) for the source mapping and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the invariants checked during reconstruction.

## External baselines

The repository focuses on the proposed method. Third-party implementations of DrFuse, M3Care, MUSE, and Flex-MoE are not redistributed here. The manuscript comparisons were conducted separately using the corresponding experimental adapters/development code. This release does not claim that those external baselines can be reproduced from this repository alone.

## Data and checkpoints

Clinical source data, processed patient data, generated NumPy arrays, experiment outputs, and model checkpoints must not be committed to this repository. The `.gitignore` excludes the standard local data/output locations and common checkpoint formats.

## License

Original code in this repository is released under the [MIT License](LICENSE). Datasets and third-party dependencies remain subject to their own licenses, access requirements, and terms of use.

## Citation

Citation information will be added when the manuscript is released on arXiv.
