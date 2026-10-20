from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.xgboost_models import MultiTargetXGBoost


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"

X_PATH = PROCESSED_DIR / "X_hybrid.npy"
LABELS_PATH = PROCESSED_DIR / "labels.csv"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"

MODEL_DIR = ROOT / "results" / "models"
METRICS_PATH = ROOT / "results" / "xgboost_metrics.json"
SPLIT_PATH = ROOT / "results" / "modeling_split.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


# ============================================================
# GROUP SPLIT
# ============================================================

def make_group_splits(
    X: np.ndarray,
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
):
    """
    Split by report_id so every entity mention from one report
    remains in exactly one partition.

    Final approximate proportions:
    64% train / 16% validation / 20% test.
    """
    groups = metadata["report_id"]

    train_val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_val_idx, test_idx = next(
        train_val_splitter.split(
            X,
            labels,
            groups=groups,
        )
    )

    X_train_val = X[train_val_idx]
    labels_train_val = labels.iloc[
        train_val_idx
    ].reset_index(drop=True)

    metadata_train_val = metadata.iloc[
        train_val_idx
    ].reset_index(drop=True)

    relative_validation_size = (
        VALIDATION_SIZE
        / (1.0 - TEST_SIZE)
    )

    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_validation_size,
        random_state=RANDOM_STATE,
    )

    train_rel_idx, val_rel_idx = next(
        validation_splitter.split(
            X_train_val,
            labels_train_val,
            groups=metadata_train_val["report_id"],
        )
    )

    train_idx = train_val_idx[
        train_rel_idx
    ]

    val_idx = train_val_idx[
        val_rel_idx
    ]

    return train_idx, val_idx, test_idx


# ============================================================
# CLASS COVERAGE CHECK
# ============================================================

def validate_class_coverage(
    labels: pd.DataFrame,
    train_idx: np.ndarray,
) -> None:
    """
    Ensure every class present in the full dataset also appears
    in the training partition.
    """
    for target in MultiTargetXGBoost.TARGETS:
        all_classes = set(
            labels[target].unique()
        )

        train_classes = set(
            labels.iloc[train_idx][target].unique()
        )

        missing = all_classes - train_classes

        if missing:
            raise ValueError(
                f"Training split is missing classes for "
                f"{target}: {sorted(missing)}"
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    X = np.load(
        X_PATH
    )

    labels = pd.read_csv(
        LABELS_PATH
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    train_idx, val_idx, test_idx = make_group_splits(
        X,
        labels,
        metadata,
    )

    validate_class_coverage(
        labels,
        train_idx,
    )

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    y_train = labels.iloc[
        train_idx
    ].reset_index(drop=True)

    y_val = labels.iloc[
        val_idx
    ].reset_index(drop=True)

    y_test = labels.iloc[
        test_idx
    ].reset_index(drop=True)

    model = MultiTargetXGBoost(
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_metrics = model.evaluate(
        X_val,
        y_val,
    )

    test_metrics = model.evaluate(
        X_test,
        y_test,
    )

    model.save(
        MODEL_DIR
    )

    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    split_labels = np.full(
        len(metadata),
        "",
        dtype=object,
    )

    split_labels[train_idx] = "train"
    split_labels[val_idx] = "validation"
    split_labels[test_idx] = "test"

    split_df = metadata[
        [
            "report_id",
            "entity",
            "normalized_entity",
            "template_family",
        ]
    ].copy()

    split_df["split"] = split_labels

    split_df.to_csv(
        SPLIT_PATH,
        index=False,
    )

    print(
        "XGBoost training completed."
    )

    print(
        "\nSplit sizes:"
    )

    print(
        "Train:",
        len(train_idx),
    )

    print(
        "Validation:",
        len(val_idx),
    )

    print(
        "Test:",
        len(test_idx),
    )

    print(
        "\nValidation metrics:"
    )

    for target, values in validation_metrics.items():
        print(
            f"{target}: "
            f"accuracy={values['accuracy']:.3f}, "
            f"macro_f1={values['macro_f1']:.3f}"
        )

    print(
        "\nTest metrics:"
    )

    for target, values in test_metrics.items():
        print(
            f"{target}: "
            f"accuracy={values['accuracy']:.3f}, "
            f"macro_f1={values['macro_f1']:.3f}"
        )

    print(
        "\nSaved models:",
        MODEL_DIR,
    )

    print(
        "Saved metrics:",
        METRICS_PATH,
    )

    print(
        "Saved split:",
        SPLIT_PATH,
    )


if __name__ == "__main__":
    main()
