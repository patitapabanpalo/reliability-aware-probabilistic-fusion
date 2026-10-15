from pathlib import Path
import os
import pandas as pd
import numpy as np
import json
import gc


ROOT = Path(os.environ.get("HAIM_ROOT", ".")).expanduser().resolve()

HAIM = (
    ROOT
    / "files"
    / "haim-multimodal"
    / "1.0.1"
)

EMBED = (
    HAIM
    / "Extracted_HAIM_Embeddings"
    / "cxr_ic_fusion_1103.csv"
)

PROC = Path(os.environ.get("HAIM_OUTPUT_DIR", ROOT / "processed")).expanduser().resolve()

MASTER = PROC / "mortality48_master.csv"

OUT_DIR = PROC / "model_data"
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHUNK_SIZE = 2000


# ============================================================
# Feature definitions
# ============================================================

GROUP_PREFIXES = {

    "structured": [
        "de_",
        "ce_",
        "le_",
        "pe_",
    ],

    "timeseries": [
        "ts_ce_",
        "ts_le_",
        "ts_pe_",
    ],

    "cxr": [
        "vd_",
        "vmd_",
        "vp_",
        "vmp_",
    ],

    "text": [
        "n_ecg_",
        "n_ech_",
        "n_rad_",
    ],
}


def get_group_columns(columns):

    groups = {}

    for group, prefixes in GROUP_PREFIXES.items():

        selected = []

        for col in columns:

            if any(
                col.startswith(prefix)
                for prefix in prefixes
            ):
                selected.append(col)

        groups[group] = selected

    return groups


def main():

    print("=" * 80)
    print("BUILD HAIM MODEL FEATURES")
    print("=" * 80)

    # --------------------------------------------------------
    # Read header only
    # --------------------------------------------------------

    header = pd.read_csv(
        EMBED,
        nrows=0,
    )

    all_columns = header.columns.tolist()

    groups = get_group_columns(
        all_columns
    )

    print("\nFeature groups:")

    for group, cols in groups.items():

        print(
            f"{group:15s}: "
            f"{len(cols):5d}"
        )

    expected = {
        "structured": 1542,
        "timeseries": 451,
        "cxr": 2084,
        "text": 2304,
    }

    for group, expected_dim in expected.items():

        actual = len(groups[group])

        if actual != expected_dim:

            raise RuntimeError(
                f"{group}: expected "
                f"{expected_dim}, found {actual}"
            )

    # --------------------------------------------------------
    # Load master cohort
    # --------------------------------------------------------

    master = pd.read_csv(
        MASTER
    )

    print(
        "\nMaster rows:",
        len(master),
    )

    # Preserve exact row identity.
    master["master_row"] = np.arange(
        len(master),
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Important:
    #
    # The embedding CSV may contain duplicate combinations.
    # We merge using HAIM ID + image ID.
    # --------------------------------------------------------

    key_cols = [
        "haim_id",
        "img_id",
    ]

    duplicate_master = master.duplicated(
        key_cols,
        keep=False,
    )

    print(
        "Master duplicate (haim_id,img_id) rows:",
        int(duplicate_master.sum()),
    )

    # We will retain duplicates exactly as present.
    # To avoid ambiguous many-to-many merges, add an
    # occurrence counter within each key.

    master["_occurrence"] = (
        master
        .groupby(key_cols)
        .cumcount()
    )

    # --------------------------------------------------------
    # Columns to read from 4 GB CSV
    # --------------------------------------------------------

    feature_columns = []

    for cols in groups.values():
        feature_columns.extend(cols)

    read_columns = (
        key_cols
        + feature_columns
    )

    print(
        "\nTotal feature columns:",
        len(feature_columns),
    )

    print(
        "Columns being read:",
        len(read_columns),
    )

    # --------------------------------------------------------
    # Read embedding CSV in chunks
    # --------------------------------------------------------

    extracted_chunks = []

    total_seen = 0

    print("\nReading embedding CSV...")

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            EMBED,
            usecols=read_columns,
            chunksize=CHUNK_SIZE,
        ),
        1,
    ):

        total_seen += len(chunk)

        chunk["_occurrence"] = (
            chunk
            .groupby(key_cols)
            .cumcount()
        )

        # NOTE:
        # cumcount resets per chunk, so this isn't safe
        # for duplicated keys crossing chunk boundaries.
        #
        # Therefore we don't use occurrence for matching
        # at this stage. We first filter to keys present
        # in master.

        relevant_haim = set(
            master["haim_id"].unique()
        )

        chunk = chunk[
            chunk["haim_id"].isin(
                relevant_haim
            )
        ].copy()

        if len(chunk):
            extracted_chunks.append(
                chunk.drop(
                    columns=["_occurrence"]
                )
            )

        if chunk_no % 5 == 0:

            print(
                f"  chunks={chunk_no:4d} "
                f"rows_seen={total_seen:7d}"
            )

    print(
        "\nEmbedding rows scanned:",
        total_seen,
    )

    embeddings = pd.concat(
        extracted_chunks,
        ignore_index=True,
    )

    del extracted_chunks
    gc.collect()

    print(
        "Relevant embedding rows:",
        len(embeddings),
    )

    # --------------------------------------------------------
    # Build occurrence counters globally
    # --------------------------------------------------------

    embeddings["_occurrence"] = (
        embeddings
        .groupby(key_cols)
        .cumcount()
    )

    # Master occurrence already computed globally.

    # --------------------------------------------------------
    # Merge master metadata onto embeddings
    # --------------------------------------------------------

    merge_keys = [
        "haim_id",
        "img_id",
        "_occurrence",
    ]

    metadata_cols = [
        "haim_id",
        "img_id",
        "_occurrence",
        "master_row",
        "subject_id",
        "hadm_id",
        "stay_id",
        "mortality_48h",
        "our_split",
        "img_charttime",
    ]

    combined = embeddings.merge(
        master[metadata_cols],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )

    print(
        "\nMatched model rows:",
        len(combined),
    )

    print(
        "Expected master rows:",
        len(master),
    )

    if len(combined) != len(master):

        missing = (
            len(master)
            - len(combined)
        )

        raise RuntimeError(
            f"Feature matching failed. "
            f"{missing} master rows unmatched."
        )

    # Restore master ordering.
    combined = combined.sort_values(
        "master_row"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Extract arrays
    # --------------------------------------------------------

    arrays = {}

    for group, cols in groups.items():

        print(
            f"\nProcessing {group}..."
        )

        X = combined[
            cols
        ].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        finite_before = np.isfinite(X)

        n_nonfinite = int(
            (~finite_before).sum()
        )

        print(
            "  shape:",
            X.shape,
        )

        print(
            "  non-finite values:",
            n_nonfinite,
        )

        # HAIM embeddings can contain missing values.
        # For now replace NaN/inf with zero.
        #
        # Later normalization will be fit on TRAIN only.

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        arrays[group] = X

    # --------------------------------------------------------
    # Metadata arrays
    # --------------------------------------------------------

    y = combined[
        "mortality_48h"
    ].to_numpy(
        dtype=np.int64
    )

    subject_id = combined[
        "subject_id"
    ].to_numpy(
        dtype=np.int64
    )

    haim_id = combined[
        "haim_id"
    ].to_numpy(
        dtype=np.int64
    )

    img_id = combined[
        "img_id"
    ].astype(str).to_numpy()

    split = combined[
        "our_split"
    ].astype(str).to_numpy()

    # --------------------------------------------------------
    # Final consistency checks
    # --------------------------------------------------------

    N = len(y)

    assert arrays["structured"].shape == (
        N,
        1542,
    )

    assert arrays["timeseries"].shape == (
        N,
        451,
    )

    assert arrays["cxr"].shape == (
        N,
        2084,
    )

    assert arrays["text"].shape == (
        N,
        2304,
    )

    assert len(subject_id) == N
    assert len(haim_id) == N
    assert len(img_id) == N
    assert len(split) == N

    print("\n" + "=" * 80)
    print("FINAL DATASET")
    print("=" * 80)

    print("N:", N)
    print("Positive:", int(y.sum()))
    print("Prevalence:", float(y.mean()))

    for name in [
        "train",
        "val",
        "test",
    ]:

        mask = split == name

        print(
            f"{name:8s}: "
            f"N={mask.sum():6d} "
            f"positive={y[mask].sum():4d} "
            f"prev={y[mask].mean():.5f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    for group, X in arrays.items():

        np.save(
            OUT_DIR / f"X_{group}.npy",
            X,
        )

    np.save(
        OUT_DIR / "y.npy",
        y,
    )

    np.save(
        OUT_DIR / "subject_id.npy",
        subject_id,
    )

    np.save(
        OUT_DIR / "haim_id.npy",
        haim_id,
    )

    np.save(
        OUT_DIR / "img_id.npy",
        img_id,
    )

    np.save(
        OUT_DIR / "split.npy",
        split,
    )

    manifest = {

        "n_samples": int(N),

        "positive": int(
            y.sum()
        ),

        "prevalence": float(
            y.mean()
        ),

        "dimensions": {
            group: int(
                X.shape[1]
            )
            for group, X in arrays.items()
        },

        "feature_columns": groups,

        "files": {
            group: f"X_{group}.npy"
            for group in arrays
        },

        "label_file": "y.npy",
        "subject_file": "subject_id.npy",
        "haim_file": "haim_id.npy",
        "image_file": "img_id.npy",
        "split_file": "split.npy",
    }

    with open(
        OUT_DIR / "manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    print("\nSaved to:")
    print(OUT_DIR)

    print("\nMODEL DATA READY")


if __name__ == "__main__":
    main()