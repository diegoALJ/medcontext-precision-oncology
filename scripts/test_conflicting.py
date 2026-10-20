from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

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


# ============================================================
# MAIN TEST
# ============================================================

def main() -> None:

    reports_df = pd.read_csv(
        REPORTS_PATH
    )

    annotations_df = pd.read_csv(
        ANNOTATIONS_PATH
    )

    conflicting_df = annotations_df[
        annotations_df[
            "template_family"
        ]
        == "CONFLICTING"
    ].copy()

    if conflicting_df.empty:

        print(
            "No CONFLICTING annotations found."
        )

        return

    report_lookup = (
        reports_df
        .set_index(
            "report_id"
        )["report_text"]
        .to_dict()
    )

    print(
        "\nCONFLICTING CASE TEST"
    )

    print(
        "=" * 80
    )

    # Show only a few reports to keep output readable.
    report_ids = (
        conflicting_df[
            "report_id"
        ]
        .drop_duplicates()
        .head(5)
        .tolist()
    )

    for report_id in report_ids:

        report_text = (
            report_lookup[
                report_id
            ]
        )

        group = conflicting_df[
            conflicting_df[
                "report_id"
            ]
            == report_id
        ]

        print(
            "\n" + "=" * 80
        )

        print(
            "REPORT ID:",
            report_id,
        )

        print(
            "REPORT:",
            report_text,
        )

        for row in group.itertuples():

            extracted_entity = (
                report_text[
                    row.start_char:
                    row.end_char
                ]
            )

            print()

            print(
                "ENTITY:",
                extracted_entity,
            )

            print(
                "NORMALIZED ENTITY:",
                row.normalized_entity,
            )

            print(
                "ENTITY TYPE:",
                row.entity_type,
            )

            print(
                "SPAN:",
                row.start_char,
                row.end_char,
            )

            print(
                "ASSERTION:",
                row.assertion,
            )

            print(
                "TEMPORALITY:",
                row.temporality,
            )

            print(
                "EXPERIENCER:",
                row.experiencer,
            )

            print(
                "CERTAINTY:",
                row.certainty,
            )

    # ========================================================
    # STRUCTURAL VALIDATION
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "STRUCTURAL VALIDATION"
    )

    errors = 0

    for report_id, group in (
        conflicting_df.groupby(
            "report_id"
        )
    ):

        report_text = (
            report_lookup[
                report_id
            ]
        )

        for row in group.itertuples():

            extracted = report_text[
                row.start_char:
                row.end_char
            ]

            if extracted != row.entity:

                errors += 1

                print(
                    "\nSPAN ERROR"
                )

                print(
                    "Report:",
                    report_id,
                )

                print(
                    "Expected:",
                    row.entity,
                )

                print(
                    "Extracted:",
                    extracted,
                )

    if errors == 0:

        print(
            "All CONFLICTING spans are valid."
        )

    else:

        print(
            f"Found {errors} invalid spans."
        )


if __name__ == "__main__":
    main()