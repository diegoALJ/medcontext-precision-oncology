from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.context_window import ContextWindowExtractor
from src.rules import ClinicalRuleExtractor
from src.embeddings import FrozenBioEmbedder
from src.hybrid_features import HybridFeatureBuilder


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

REPORTS_PATH = ROOT / "data" / "synthetic" / "reports.csv"
ANNOTATIONS_PATH = ROOT / "data" / "synthetic" / "annotations.csv"

OUTPUT_DIR = ROOT / "data" / "processed"

X_PATH = OUTPUT_DIR / "X_hybrid.npy"
LABELS_PATH = OUTPUT_DIR / "labels.csv"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
FEATURE_NAMES_PATH = OUTPUT_DIR / "feature_names.txt"


# ============================================================
# INPUT PREPARATION
# ============================================================

def build_candidate_from_annotation(row: pd.Series) -> dict:
    """
    Recreate the minimal candidate structure required by the
    ContextWindowExtractor from one ground-truth annotation.

    We do not re-run CandidateFinder here because annotations
    already contain the exact entity span.
    """
    return {
        "entity_text": row["entity"],
        "normalized_entity": row["normalized_entity"],
        "entity_type": row["entity_type"],
        "disease": row["disease"],
        "start_char": int(row["start_char"]),
        "end_char": int(row["end_char"]),
    }


# ============================================================
# MAIN DATASET BUILDING
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reports_df = pd.read_csv(REPORTS_PATH)
    annotations_df = pd.read_csv(ANNOTATIONS_PATH)

    report_lookup = (
        reports_df
        .set_index("report_id")["report_text"]
        .to_dict()
    )

    window_extractor = ContextWindowExtractor()
    rule_extractor = ClinicalRuleExtractor()
    embedder = FrozenBioEmbedder()
    feature_builder = HybridFeatureBuilder()

    contexts = []
    rule_features_list = []
    metadata_rows = []
    label_rows = []

    # --------------------------------------------------------
    # Context windows + symbolic features
    # --------------------------------------------------------

    for _, row in annotations_df.iterrows():
        report_id = row["report_id"]
        report_text = report_lookup[report_id]

        candidate = build_candidate_from_annotation(row)

        context = window_extractor.extract(
            report_text,
            candidate,
        )

        ruled = rule_extractor.extract(context)

        contexts.append(
            ruled["context_text"]
        )

        rule_features_list.append(
            ruled["rule_features"]
        )

        metadata_rows.append(
            {
                "report_id": report_id,
                "entity": row["entity"],
                "normalized_entity": row["normalized_entity"],
                "entity_type": row["entity_type"],
                "disease": row["disease"],
                "template_family": row["template_family"],
                "start_char": int(row["start_char"]),
                "end_char": int(row["end_char"]),
                "context_text": ruled["context_text"],
                "context_start": ruled["context_start"],
                "context_end": ruled["context_end"],
                "entity_start_in_context": ruled[
                    "entity_start_in_context"
                ],
                "entity_end_in_context": ruled[
                    "entity_end_in_context"
                ],
                "num_sentences": ruled["num_sentences"],
            }
        )

        label_rows.append(
            {
                "assertion": row["assertion"],
                "experiencer": row["experiencer"],
                "temporality": row["temporality"],
                "certainty": row["certainty"],
            }
        )

    # --------------------------------------------------------
    # BioLORD embeddings
    # --------------------------------------------------------

    embeddings = embedder.encode(contexts)

    # --------------------------------------------------------
    # Hybrid feature matrix
    # --------------------------------------------------------

    X = feature_builder.build_batch(
        rule_features_list,
        embeddings,
    )

    metadata_df = pd.DataFrame(metadata_rows)
    labels_df = pd.DataFrame(label_rows)

    feature_names = feature_builder.feature_names(
        embedder.embedding_dimension
    )

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    n_samples = len(annotations_df)

    if X.shape[0] != n_samples:
        raise ValueError(
            "Feature matrix row count does not match annotations."
        )

    if len(labels_df) != n_samples:
        raise ValueError(
            "Labels row count does not match annotations."
        )

    if len(metadata_df) != n_samples:
        raise ValueError(
            "Metadata row count does not match annotations."
        )

    if X.shape[1] != len(feature_names):
        raise ValueError(
            "Feature matrix column count does not match feature names."
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    np.save(X_PATH, X)

    labels_df.to_csv(
        LABELS_PATH,
        index=False,
    )

    metadata_df.to_csv(
        METADATA_PATH,
        index=False,
    )

    FEATURE_NAMES_PATH.write_text(
        "\n".join(feature_names),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("Modeling dataset built successfully.")
    print("Samples:", X.shape[0])
    print("Features:", X.shape[1])
    print("X shape:", X.shape)

    print("\nLabel distributions:")

    for column in labels_df.columns:
        print(f"\n{column.upper()}")
        print(labels_df[column].value_counts())

    print("\nSaved:")
    print(X_PATH)
    print(LABELS_PATH)
    print(METADATA_PATH)
    print(FEATURE_NAMES_PATH)


if __name__ == "__main__":
    main()
