from pathlib import Path
import os
import pandas as pd
import numpy as np
import json

ROOT = Path(os.environ.get("HAIM_ROOT", ".")).expanduser().resolve()
HAIM = ROOT / "files" / "haim-multimodal" / "1.0.1"

EMBED = (
    HAIM
    / "Extracted_HAIM_Embeddings"
    / "cxr_ic_fusion_1103.csv"
)

OUT_DIR = Path(os.environ.get("HAIM_OUTPUT_DIR", ROOT / "processed")).expanduser().resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORT_CSV = OUT_DIR / "mortality48_cohort.csv"
REPORT_JSON = OUT_DIR / "mortality48_cohort_report.json"


META_COLS = [
    "haim_id",
    "img_id",
    "img_charttime",
    "img_deltacharttime",
    "img_length_of_stay",
    "death_status",
    "split",
]


def main():

    print("=" * 80)
    print("HAIM 48-HOUR MORTALITY COHORT")
    print("=" * 80)

    # Only load metadata -- not 6405 features.
    header = pd.read_csv(EMBED, nrows=0)

    available = [
        c for c in META_COLS
        if c in header.columns
    ]

    print("Metadata columns:")
    print(available)

    df = pd.read_csv(
        EMBED,
        usecols=available,
    )

    print("\nRows loaded:", len(df))

    # --------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------

    df["img_charttime"] = pd.to_datetime(
        df["img_charttime"],
        errors="coerce",
    )

    required = [
        "haim_id",
        "img_id",
        "img_charttime",
        "img_length_of_stay",
        "death_status",
    ]

    missing_required = df[required].isna().any(axis=1)

    print(
        "Rows with missing required metadata:",
        int(missing_required.sum()),
    )

    if missing_required.any():
        df = df.loc[~missing_required].copy()

    # --------------------------------------------------
    # Mortality label
    # --------------------------------------------------

    # Reproduce candidate HAIM definition:
    #
    # positive =
    #   death_status == 1
    #   AND remaining LOS from image < 48 h
    #
    # negative =
    #   alive OR death >= 48 h after image

    df["mortality_48h"] = (
        (df["death_status"] == 1)
        & (df["img_length_of_stay"] < 48.0)
    ).astype(np.int8)

    # --------------------------------------------------
    # Duplicate diagnostics
    # --------------------------------------------------

    df["duplicate_img_id"] = df.duplicated(
        subset=["img_id"],
        keep=False,
    )

    df["duplicate_haim_img"] = df.duplicated(
        subset=["haim_id", "img_id"],
        keep=False,
    )

    # An image can occur more than once in the extracted
    # representation. Do NOT silently remove anything yet.
    # We first quantify it.

    # --------------------------------------------------
    # HAIM-level diagnostics
    # --------------------------------------------------

    per_haim = (
        df.groupby("haim_id")
        .agg(
            n_rows=("img_id", "size"),
            n_images=("img_id", "nunique"),
            n_positive=("mortality_48h", "sum"),
            death_status=("death_status", "max"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    report = {
        "rows": int(len(df)),
        "unique_haim_ids": int(df["haim_id"].nunique()),
        "unique_img_ids": int(df["img_id"].nunique()),

        "mortality_48h_positive": int(
            df["mortality_48h"].sum()
        ),

        "mortality_48h_negative": int(
            (df["mortality_48h"] == 0).sum()
        ),

        "mortality_48h_prevalence": float(
            df["mortality_48h"].mean()
        ),

        "death_status_positive": int(
            (df["death_status"] == 1).sum()
        ),

        "duplicate_img_rows": int(
            df["duplicate_img_id"].sum()
        ),

        "duplicate_haim_img_rows": int(
            df["duplicate_haim_img"].sum()
        ),

        "haim_ids_with_positive_row": int(
            (per_haim["n_positive"] > 0).sum()
        ),

        "max_rows_per_haim": int(
            per_haim["n_rows"].max()
        ),

        "median_rows_per_haim": float(
            per_haim["n_rows"].median()
        ),
    }

    print("\n" + "=" * 80)
    print("COHORT SUMMARY")
    print("=" * 80)

    for key, value in report.items():
        print(f"{key:35s}: {value}")

    # --------------------------------------------------
    # Check label consistency for duplicate image IDs
    # --------------------------------------------------

    img_label_counts = (
        df.groupby("img_id")["mortality_48h"]
        .nunique()
    )

    inconsistent_images = int(
        (img_label_counts > 1).sum()
    )

    report["images_with_inconsistent_labels"] = (
        inconsistent_images
    )

    print(
        "\nImages appearing with >1 mortality label:",
        inconsistent_images,
    )

    # --------------------------------------------------
    # Existing HAIM split diagnostic
    # --------------------------------------------------

    if "split" in df.columns:

        print("\nExisting HAIM split distribution:")
        print(
            df["split"]
            .value_counts(dropna=False)
            .to_string()
        )

        split_per_haim = (
            df.groupby("haim_id")["split"]
            .nunique(dropna=True)
        )

        crossing = int(
            (split_per_haim > 1).sum()
        )

        report[
            "haim_ids_crossing_existing_splits"
        ] = crossing

        print(
            "\nHAIM IDs appearing in >1 existing split:",
            crossing,
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    save_cols = [
        c for c in [
            "haim_id",
            "img_id",
            "img_charttime",
            "img_deltacharttime",
            "img_length_of_stay",
            "death_status",
            "mortality_48h",
            "split",
            "duplicate_img_id",
            "duplicate_haim_img",
        ]
        if c in df.columns
    ]

    df[save_cols].to_csv(
        COHORT_CSV,
        index=False,
    )

    with open(
        REPORT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print("\nSaved:")
    print(COHORT_CSV)
    print(REPORT_JSON)


if __name__ == "__main__":
    main()