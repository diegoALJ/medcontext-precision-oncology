from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.clinical_fact_compiler import ClinicalFactCompiler
from src.evaluation import PipelineEvaluator


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

SPLIT_PATH = (
    ROOT
    / "results"
    / "modeling_split.csv"
)

MODEL_DIR = (
    ROOT
    / "results"
    / "models"
)

METRICS_PATH = (
    ROOT
    / "results"
    / "pipeline_evaluation.json"
)

ERRORS_PATH = (
    ROOT
    / "results"
    / "pipeline_errors.csv"
)


# ============================================================
# TEST DATA SELECTION
# ============================================================

def get_test_report_ids(
    split_df: pd.DataFrame,
) -> list[str]:
    """
    Return the unique report IDs assigned to the test partition
    during Step 8.
    """
    return (
        split_df.loc[
            split_df["split"] == "test",
            "report_id",
        ]
        .drop_duplicates()
        .tolist()
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    reports_df = pd.read_csv(
        REPORTS_PATH
    )

    annotations_df = pd.read_csv(
        ANNOTATIONS_PATH
    )

    split_df = pd.read_csv(
        SPLIT_PATH
    )

    test_report_ids = get_test_report_ids(
        split_df
    )

    test_reports = reports_df[
        reports_df["report_id"].isin(
            test_report_ids
        )
    ].copy()

    test_annotations = annotations_df[
        annotations_df["report_id"].isin(
            test_report_ids
        )
    ].copy()

    compiler = ClinicalFactCompiler(
        vocabulary_path=VOCAB_PATH,
        model_dir=MODEL_DIR,
    )

    evaluator = PipelineEvaluator()

    result = evaluator.evaluate_reports(
        compiler=compiler,
        reports_df=test_reports,
        annotations_df=test_annotations,
    )

    metrics = result["metrics"]
    errors_df = result["errors"]

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    errors_df.to_csv(
        ERRORS_PATH,
        index=False,
    )

    extraction = metrics[
        "fact_extraction"
    ]

    print(
        "END-TO-END PIPELINE EVALUATION"
    )

    print(
        "=" * 50
    )

    print(
        "\nTest reports:",
        len(test_reports),
    )

    print(
        "\nFact extraction:"
    )

    print(
        "Expected:",
        extraction["expected"],
    )

    print(
        "Predicted:",
        extraction["predicted"],
    )

    print(
        "Matched:",
        extraction["matched"],
    )

    print(
        f"Precision: "
        f"{extraction['precision']:.3f}"
    )

    print(
        f"Recall:    "
        f"{extraction['recall']:.3f}"
    )

    print(
        f"F1:        "
        f"{extraction['f1']:.3f}"
    )

    print(
        "\nContext classification:"
    )

    for target, values in metrics[
        "context_classification"
    ].items():

        print(
            f"{target}: "
            f"accuracy={values['accuracy']:.3f}, "
            f"macro_f1={values['macro_f1']:.3f}"
        )

    print(
        "\nExact 4-target context match:",
        f"{metrics['exact_context_match_accuracy']:.3f}",
    )

    latency = metrics["latency"]

    print(
        "\nLatency:"
    )

    print(
        f"Mean:   "
        f"{latency['mean_seconds_per_report']:.4f} s/report"
    )

    print(
        f"Median: "
        f"{latency['median_seconds_per_report']:.4f} s/report"
    )

    print(
        f"P95:    "
        f"{latency['p95_seconds_per_report']:.4f} s/report"
    )

    print(
        "\nErrors:",
        len(errors_df),
    )

    print(
        "\nSaved metrics:",
        METRICS_PATH,
    )

    print(
        "Saved errors:",
        ERRORS_PATH,
    )


if __name__ == "__main__":
    main()
