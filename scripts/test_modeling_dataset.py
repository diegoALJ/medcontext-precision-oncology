from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

X_PATH = PROCESSED_DIR / "X_hybrid.npy"
LABELS_PATH = PROCESSED_DIR / "labels.csv"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"
FEATURE_NAMES_PATH = PROCESSED_DIR / "feature_names.txt"


def main() -> None:
    X = np.load(X_PATH)
    labels = pd.read_csv(LABELS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    feature_names = (
        FEATURE_NAMES_PATH
        .read_text(encoding="utf-8")
        .splitlines()
    )

    print("X SHAPE:", X.shape)
    print("LABELS SHAPE:", labels.shape)
    print("METADATA SHAPE:", metadata.shape)
    print("FEATURE NAMES:", len(feature_names))

    print("\nFIRST SAMPLE")
    print("ENTITY:", metadata.loc[0, "entity"])
    print("CONTEXT:", metadata.loc[0, "context_text"])
    print("LABELS:", labels.loc[0].to_dict())
    print("FIRST 9 SYMBOLIC FEATURES:", X[0, :9])
    print("FIRST 10 EMBEDDING FEATURES:", X[0, 9:19])


if __name__ == "__main__":
    main()
