from pathlib import Path
import os
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split


ROOT = Path(os.environ.get("HAIM_ROOT", ".")).expanduser().resolve()
HAIM = ROOT / "files" / "haim-multimodal" / "1.0.1"
PROC = Path(os.environ.get("HAIM_OUTPUT_DIR", ROOT / "processed")).expanduser().resolve()

KEY_IDS = HAIM / "haim_mimiciv_key_ids.csv"
COHORT = PROC / "mortality48_cohort.csv"

MASTER_OUT = PROC / "mortality48_master.csv"
SPLIT_OUT = PROC / "mortality48_split_report.json"

SEED = 42


def main():

    print("=" * 80)
    print("HAIM MASTER COHORT + PATIENT-LEVEL SPLIT")
    print("=" * 80)

    # --------------------------------------------------
    # Load cohort
    # --------------------------------------------------

    cohort = pd.read_csv(COHORT)

    print("\nCohort rows:", len(cohort))
    print("Unique HAIM IDs:", cohort["haim_id"].nunique())

    # --------------------------------------------------
    # Load HAIM -> MIMIC identity mapping
    # --------------------------------------------------

    keys = pd.read_csv(KEY_IDS)

    print("\nKey table rows:", len(keys))

    # IMPORTANT:
    # official HAIM code uses:
    #
    # df_haim_ids.iloc[haim_patient_idx]
    #
    # Therefore row position is the HAIM identifier.

    keys = keys.reset_index(drop=True)

    keys["haim_id"] = np.arange(
        len(keys),
        dtype=np.int64,
    )

    # Convert identifiers safely to nullable integer first.
    for col in ["subject_id", "hadm_id", "stay_id"]:

        keys[col] = pd.to_numeric(
            keys[col],
            errors="coerce",
        ).astype("Int64")

    print("\nIdentity mapping summary:")
    print(f"  rows: {len(keys):,}")
    print(f"  mapped subject_id: {int(keys['subject_id'].notna().sum()):,}")
    print(f"  mapped hadm_id: {int(keys['hadm_id'].notna().sum()):,}")
    print(f"  mapped stay_id: {int(keys['stay_id'].notna().sum()):,}")

    # --------------------------------------------------
    # Merge
    # --------------------------------------------------

    master = cohort.merge(
        keys[
            [
                "haim_id",
                "subject_id",
                "hadm_id",
                "stay_id",
            ]
        ],
        on="haim_id",
        how="left",
        validate="many_to_one",
    )

    print("\nMaster rows:", len(master))

    missing_identity = master[
        ["subject_id", "hadm_id", "stay_id"]
    ].isna().any(axis=1)

    print(
        "Rows missing identity:",
        int(missing_identity.sum()),
    )

    if missing_identity.any():

        raise RuntimeError(
            "Some cohort rows could not be mapped to subject/hadm/stay IDs. "
            f"Unmapped rows: {int(missing_identity.sum()):,}."
        )

    # Convert after successful mapping.
    for col in ["subject_id", "hadm_id", "stay_id"]:
        master[col] = master[col].astype(np.int64)

    # --------------------------------------------------
    # Patient-level outcome for stratification
    # --------------------------------------------------

    subjects = (
        master.groupby("subject_id")
        .agg(
            patient_positive=(
                "mortality_48h",
                "max",
            ),
            n_samples=(
                "mortality_48h",
                "size",
            ),
            n_positive_samples=(
                "mortality_48h",
                "sum",
            ),
            n_haim_ids=(
                "haim_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    print("\nUnique subjects:", len(subjects))

    print("\nPatient-level outcome:")
    print(
        subjects["patient_positive"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nPatient-level positive prevalence:",
        subjects["patient_positive"].mean(),
    )

    # --------------------------------------------------
    # Split subjects
    #
    # 70% train
    # 10% validation
    # 20% test
    # --------------------------------------------------

    train_subjects, temp_subjects = train_test_split(
        subjects,
        test_size=0.30,
        random_state=SEED,
        stratify=subjects["patient_positive"],
    )

    # temp is 30%.
    # validation should be 10% total
    # test should be 20% total.
    #
    # Therefore validation is 1/3 of temp.

    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=2.0 / 3.0,
        random_state=SEED,
        stratify=temp_subjects["patient_positive"],
    )

    train_ids = set(
        train_subjects["subject_id"].tolist()
    )

    val_ids = set(
        val_subjects["subject_id"].tolist()
    )

    test_ids = set(
        test_subjects["subject_id"].tolist()
    )

    # --------------------------------------------------
    # Leakage checks
    # --------------------------------------------------

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    print("\n" + "=" * 80)
    print("PATIENT LEAKAGE CHECK")
    print("=" * 80)

    print(
        "Train / Val overlap :",
        len(train_val_overlap),
    )

    print(
        "Train / Test overlap:",
        len(train_test_overlap),
    )

    print(
        "Val / Test overlap  :",
        len(val_test_overlap),
    )

    assert len(train_val_overlap) == 0
    assert len(train_test_overlap) == 0
    assert len(val_test_overlap) == 0

    # --------------------------------------------------
    # Assign split to every CXR-timepoint sample
    # --------------------------------------------------

    split_map = {}

    split_map.update({
        sid: "train"
        for sid in train_ids
    })

    split_map.update({
        sid: "val"
        for sid in val_ids
    })

    split_map.update({
        sid: "test"
        for sid in test_ids
    })

    master["our_split"] = (
        master["subject_id"]
        .map(split_map)
    )

    assert master["our_split"].notna().all()

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    report = {
        "seed": SEED,

        "total_rows": int(len(master)),
        "total_subjects": int(
            master["subject_id"].nunique()
        ),

        "overall_positive_rows": int(
            master["mortality_48h"].sum()
        ),

        "overall_prevalence": float(
            master["mortality_48h"].mean()
        ),

        "splits": {},

        "patient_overlap": {
            "train_val": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "val_test": len(val_test_overlap),
        },
    }

    print("\n" + "=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)

    for split in ["train", "val", "test"]:

        part = master[
            master["our_split"] == split
        ]

        patient_part = subjects[
            subjects["subject_id"].isin(
                part["subject_id"].unique()
            )
        ]

        info = {
            "rows": int(len(part)),

            "subjects": int(
                part["subject_id"].nunique()
            ),

            "haim_ids": int(
                part["haim_id"].nunique()
            ),

            "images": int(
                part["img_id"].nunique()
            ),

            "positive_rows": int(
                part["mortality_48h"].sum()
            ),

            "negative_rows": int(
                (part["mortality_48h"] == 0).sum()
            ),

            "row_prevalence": float(
                part["mortality_48h"].mean()
            ),

            "positive_patients": int(
                patient_part["patient_positive"].sum()
            ),

            "patient_prevalence": float(
                patient_part["patient_positive"].mean()
            ),
        }

        report["splits"][split] = info

        print(f"\n[{split.upper()}]")

        for key, value in info.items():
            print(
                f"{key:25s}: {value}"
            )

    # --------------------------------------------------
    # Image overlap checks
    # --------------------------------------------------

    split_images = {}

    for split in ["train", "val", "test"]:

        split_images[split] = set(
            master.loc[
                master["our_split"] == split,
                "img_id",
            ]
        )

    image_overlap = {
        "train_val": len(
            split_images["train"]
            & split_images["val"]
        ),
        "train_test": len(
            split_images["train"]
            & split_images["test"]
        ),
        "val_test": len(
            split_images["val"]
            & split_images["test"]
        ),
    }

    report["image_overlap"] = image_overlap

    print("\n" + "=" * 80)
    print("IMAGE OVERLAP CHECK")
    print("=" * 80)

    for key, value in image_overlap.items():
        print(f"{key:20s}: {value}")

    # Patient split should normally imply image separation.
    # Don't silently accept overlap.

    if any(v != 0 for v in image_overlap.values()):

        print(
            "\nWARNING: identical img_id occurs across "
            "patient-level splits."
        )

    # --------------------------------------------------
    # HAIM-ID overlap (should trivially be zero because
    # each HAIM ID belongs to one subject)
    # --------------------------------------------------

    split_haim = {}

    for split in ["train", "val", "test"]:

        split_haim[split] = set(
            master.loc[
                master["our_split"] == split,
                "haim_id",
            ]
        )

    haim_overlap = {
        "train_val": len(
            split_haim["train"]
            & split_haim["val"]
        ),
        "train_test": len(
            split_haim["train"]
            & split_haim["test"]
        ),
        "val_test": len(
            split_haim["val"]
            & split_haim["test"]
        ),
    }

    report["haim_overlap"] = haim_overlap

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    master.to_csv(
        MASTER_OUT,
        index=False,
    )

    with open(
        SPLIT_OUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("SAVED")
    print("=" * 80)

    print(MASTER_OUT)
    print(SPLIT_OUT)

    print("\nMASTER COHORT READY")


if __name__ == "__main__":
    main()