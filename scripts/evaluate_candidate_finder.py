from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.candidate_finder import CandidateFinder


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

VOCAB_PATH = (
    ROOT
    / "data"
    / "vocabulary"
    / "oncology_vocabulary.yaml"
)

REPORTS_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "reports.csv"
)

ANNOTATIONS_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "annotations.csv"
)

RESULTS_DIR = (
    ROOT
    / "results"
)

MISSED_PATH = (
    RESULTS_DIR
    / "candidate_finder_missed.csv"
)


# ============================================================
# MATCH KEY
# ============================================================

def annotation_key(
    normalized_entity: str,
    entity_type: str,
    start_char: int,
    end_char: int,
) -> tuple:
    """
    Build the unique identity of one biomedical mention.

    Offsets are now part of the key because the same entity can
    legitimately appear multiple times in one report.
    """
    return (
        normalized_entity,
        entity_type,
        int(start_char),
        int(end_char),
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_df = pd.read_csv(
        REPORTS_PATH
    )

    annotations_df = pd.read_csv(
        ANNOTATIONS_PATH
    )

    finder = CandidateFinder(
        VOCAB_PATH
    )

    total_expected = 0
    total_detected = 0
    total_correct = 0

    missed_rows = []

    annotations_by_report = {
        report_id: group
        for report_id, group
        in annotations_df.groupby(
            "report_id"
        )
    }

    for report in reports_df.itertuples():

        report_id = report.report_id
        report_text = report.report_text

        report_annotations = (
            annotations_by_report.get(
                report_id
            )
        )

        if report_annotations is None:
            continue

        # ----------------------------------------------------
        # Ground truth mentions
        # ----------------------------------------------------

        expected = set()

        for row in report_annotations.itertuples():

            key = annotation_key(
                normalized_entity=(
                    row.normalized_entity
                ),
                entity_type=(
                    row.entity_type
                ),
                start_char=(
                    row.start_char
                ),
                end_char=(
                    row.end_char
                ),
            )

            expected.add(
                key
            )

        # ----------------------------------------------------
        # Candidate Finder predictions
        # ----------------------------------------------------

        candidates = finder.find(
            report_text
        )

        detected = set()

        for candidate in candidates:

            key = annotation_key(
                normalized_entity=(
                    candidate[
                        "normalized_entity"
                    ]
                ),
                entity_type=(
                    candidate[
                        "entity_type"
                    ]
                ),
                start_char=(
                    candidate[
                        "start_char"
                    ]
                ),
                end_char=(
                    candidate[
                        "end_char"
                    ]
                ),
            )

            detected.add(
                key
            )

        # ----------------------------------------------------
        # Compare
        # ----------------------------------------------------

        correct = (
            expected
            & detected
        )

        missed = (
            expected
            - detected
        )

        total_expected += len(
            expected
        )

        total_detected += len(
            detected
        )

        total_correct += len(
            correct
        )

        # ----------------------------------------------------
        # Save missed mentions
        # ----------------------------------------------------

        for missed_key in missed:

            (
                normalized_entity,
                entity_type,
                start_char,
                end_char,
            ) = missed_key

            missed_rows.append(
                {
                    "report_id": report_id,
                    "normalized_entity": (
                        normalized_entity
                    ),
                    "entity_type": (
                        entity_type
                    ),
                    "start_char": (
                        start_char
                    ),
                    "end_char": (
                        end_char
                    ),
                    "source_text": (
                        report_text[
                            start_char:end_char
                        ]
                    ),
                    "report_text": (
                        report_text
                    ),
                }
            )

    # ========================================================
    # METRICS
    # ========================================================

    if total_expected > 0:
        recall = (
            total_correct
            / total_expected
        )
    else:
        recall = 0.0

    print(
        "\nCandidate Finder Evaluation"
    )

    print(
        "-" * 30
    )

    print(
        f"Total expected entities: "
        f"{total_expected}"
    )

    print(
        f"Total detected entities: "
        f"{total_detected}"
    )

    print(
        f"Correct matches:         "
        f"{total_correct}"
    )

    print(
        f"Missed entities:         "
        f"{total_expected - total_correct}"
    )

    print(
        f"Entity recall:           "
        f"{recall:.3f}"
    )

    # ========================================================
    # SAVE MISSED
    # ========================================================

    missed_df = pd.DataFrame(
        missed_rows
    )

    missed_df.to_csv(
        MISSED_PATH,
        index=False,
    )

    print(
        "\nMissed entities saved to:"
    )

    print(
        MISSED_PATH
    )


if __name__ == "__main__":
    main()