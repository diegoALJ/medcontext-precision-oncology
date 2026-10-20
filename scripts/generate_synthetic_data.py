from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import yaml


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

OUTPUT_DIR = (
    ROOT
    / "data"
    / "synthetic"
)

REPORTS_PATH = OUTPUT_DIR / "reports.csv"
ANNOTATIONS_PATH = OUTPUT_DIR / "annotations.csv"

N_REPORTS = 120
MIN_FINDINGS_PER_REPORT = 3
MAX_FINDINGS_PER_REPORT = 5

RANDOM_SEED = 42

TARGET_ENTITY_TYPES = {
    "GENE",
    "VARIANT",
    "FUSION",
    "BIOMARKER",
}


# ============================================================
# TEMPLATE DEFINITIONS
# ============================================================

TEMPLATES = {
    "POSITIVE": [
        "{entity} was detected.",
        "{entity} was identified.",
        "Molecular testing confirmed {entity}.",
    ],

    "NEGATIVE": [
        "No {entity} was detected.",
        "Testing was negative for {entity}.",
        "Molecular analysis did not identify {entity}.",
    ],

    "UNCERTAIN": [
        "{entity} cannot be excluded.",
        "A possible {entity} was identified.",
        "Findings are suggestive of {entity}.",
    ],

    "HISTORICAL": [
        "{entity} was detected at diagnosis.",
        "{entity} was previously identified.",
        "{entity} was present on initial molecular testing.",
    ],

    "FAMILY": [
        "The patient's mother carried {entity}.",
        "The patient's father carried {entity}.",
        "A family member was previously found to carry {entity}.",
    ],

    # Important:
    # CONFLICTING must contain TWO explicit entity mentions.
    "CONFLICTING": [
        (
            "{entity} was previously detected, "
            "but current testing was negative for {entity}."
        ),
        (
            "{entity} was identified at diagnosis, "
            "but repeat sequencing did not detect {entity}."
        ),
    ],
}


# ============================================================
# LABEL DEFINITIONS
# ============================================================

SINGLE_LABELS = {
    "POSITIVE": {
        "assertion": "PRESENT",
        "experiencer": "PATIENT",
        "temporality": "CURRENT",
        "certainty": "CONFIRMED",
    },

    "NEGATIVE": {
        "assertion": "ABSENT",
        "experiencer": "PATIENT",
        "temporality": "CURRENT",
        "certainty": "CONFIRMED",
    },

    "UNCERTAIN": {
        "assertion": "POSSIBLE",
        "experiencer": "PATIENT",
        "temporality": "CURRENT",
        "certainty": "SUSPECTED",
    },

    "HISTORICAL": {
        "assertion": "PRESENT",
        "experiencer": "PATIENT",
        "temporality": "HISTORICAL",
        "certainty": "CONFIRMED",
    },

    "FAMILY": {
        "assertion": "PRESENT",
        "experiencer": "FAMILY",
        "temporality": "HISTORICAL",
        "certainty": "CONFIRMED",
    },
}


# ============================================================
# VOCABULARY LOADING
# ============================================================

def load_vocabulary() -> dict:
    """
    Load the controlled oncology vocabulary.
    """
    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def collect_entities(vocabulary: dict) -> list[dict]:
    """
    Convert the YAML vocabulary into a flat list of entities
    that can be sampled by the synthetic generator.

    Expected entity types:
    GENE, VARIANT, FUSION, BIOMARKER.

    DRUG and DISEASE are intentionally excluded from the
    contextual classification benchmark.
    """
    entities = []

    diseases = vocabulary.get("diseases", {})

    for disease_name, disease_data in diseases.items():

        for key, entity_type in (
            ("genes", "GENE"),
            ("variants", "VARIANT"),
            ("fusions", "FUSION"),
            ("biomarkers", "BIOMARKER"),
        ):

            items = disease_data.get(key, [])

            for item in items:

                if isinstance(item, str):
                    canonical = item
                    aliases = [item]

                elif isinstance(item, dict):
                    canonical = (
                        item.get("canonical")
                        or item.get("name")
                        or item.get("entity")
                    )

                    if canonical is None:
                        continue

                    aliases = item.get(
                        "aliases",
                        [canonical],
                    )

                else:
                    continue

                if canonical not in aliases:
                    aliases = [canonical] + list(aliases)

                entities.append(
                    {
                        "normalized_entity": canonical,
                        "entity_type": entity_type,
                        "disease": disease_name,
                        "aliases": aliases,
                    }
                )

    return entities


# ============================================================
# SPAN UTILITIES
# ============================================================

def find_all_entity_spans(
    text: str,
    entity_text: str,
) -> list[tuple[int, int]]:
    """
    Find every exact occurrence of entity_text inside text.

    Returns:
        [(start_char, end_char), ...]
    """
    spans = []

    search_start = 0

    while True:

        index = text.find(
            entity_text,
            search_start,
        )

        if index == -1:
            break

        end = index + len(entity_text)

        spans.append(
            (index, end)
        )

        search_start = end

    return spans


# ============================================================
# FINDING GENERATION
# ============================================================

def generate_single_finding(
    entity: dict,
    template_family: str,
) -> tuple[str, list[dict]]:
    """
    Generate one synthetic clinical sentence plus its local
    annotations.

    Character offsets returned here are relative to the sentence,
    not yet to the complete report.
    """

    alias = random.choice(
        entity["aliases"]
    )

    template = random.choice(
        TEMPLATES[template_family]
    )

    sentence = template.format(
        entity=alias
    )

    spans = find_all_entity_spans(
        sentence,
        alias,
    )

    annotations = []

    # --------------------------------------------------------
    # CONFLICTING
    # --------------------------------------------------------

    if template_family == "CONFLICTING":

        if len(spans) != 2:
            raise ValueError(
                "CONFLICTING templates must contain exactly "
                f"two mentions of {alias!r}. "
                f"Found {len(spans)} in: {sentence!r}"
            )

        historical_span = spans[0]
        current_span = spans[1]

        annotations.append(
            {
                "entity": alias,
                "normalized_entity": entity[
                    "normalized_entity"
                ],
                "entity_type": entity[
                    "entity_type"
                ],
                "disease": entity[
                    "disease"
                ],
                "start_char": historical_span[0],
                "end_char": historical_span[1],
                "assertion": "PRESENT",
                "experiencer": "PATIENT",
                "temporality": "HISTORICAL",
                "certainty": "CONFIRMED",
                "template_family": "CONFLICTING",
            }
        )

        annotations.append(
            {
                "entity": alias,
                "normalized_entity": entity[
                    "normalized_entity"
                ],
                "entity_type": entity[
                    "entity_type"
                ],
                "disease": entity[
                    "disease"
                ],
                "start_char": current_span[0],
                "end_char": current_span[1],
                "assertion": "ABSENT",
                "experiencer": "PATIENT",
                "temporality": "CURRENT",
                "certainty": "CONFIRMED",
                "template_family": "CONFLICTING",
            }
        )

        return sentence, annotations

    # --------------------------------------------------------
    # NORMAL SINGLE-MENTION FAMILIES
    # --------------------------------------------------------

    if len(spans) != 1:
        raise ValueError(
            f"Expected one mention of {alias!r}, "
            f"found {len(spans)} in: {sentence!r}"
        )

    labels = SINGLE_LABELS[
        template_family
    ]

    annotations.append(
        {
            "entity": alias,
            "normalized_entity": entity[
                "normalized_entity"
            ],
            "entity_type": entity[
                "entity_type"
            ],
            "disease": entity[
                "disease"
            ],
            "start_char": spans[0][0],
            "end_char": spans[0][1],
            "assertion": labels[
                "assertion"
            ],
            "experiencer": labels[
                "experiencer"
            ],
            "temporality": labels[
                "temporality"
            ],
            "certainty": labels[
                "certainty"
            ],
            "template_family": template_family,
        }
    )

    return sentence, annotations


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(
    report_id: str,
    entities: list[dict],
) -> tuple[dict, list[dict]]:
    """
    Generate one report containing multiple molecular findings.

    Local sentence offsets are converted here into absolute
    offsets relative to the complete report.
    """

    n_findings = random.randint(
        MIN_FINDINGS_PER_REPORT,
        MAX_FINDINGS_PER_REPORT,
    )

    selected_entities = random.sample(
        entities,
        k=n_findings,
    )

    report_parts = []
    report_annotations = []

    current_offset = 0

    template_families = list(
        TEMPLATES.keys()
    )

    for entity in selected_entities:

        template_family = random.choice(
            template_families
        )

        sentence, local_annotations = (
            generate_single_finding(
                entity=entity,
                template_family=template_family,
            )
        )

        # Add one whitespace between sentences.
        if report_parts:
            current_offset += 1

        report_parts.append(
            sentence
        )

        for annotation in local_annotations:

            absolute_annotation = (
                annotation.copy()
            )

            absolute_annotation[
                "report_id"
            ] = report_id

            absolute_annotation[
                "start_char"
            ] += current_offset

            absolute_annotation[
                "end_char"
            ] += current_offset

            report_annotations.append(
                absolute_annotation
            )

        current_offset += len(
            sentence
        )

    report_text = " ".join(
        report_parts
    )

    report_record = {
        "report_id": report_id,
        "report_text": report_text,
    }

    return (
        report_record,
        report_annotations,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_annotations(
    reports_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
) -> None:
    """
    Confirm that every annotation span retrieves exactly the
    expected entity mention from its report.
    """

    report_lookup = (
        reports_df
        .set_index("report_id")[
            "report_text"
        ]
        .to_dict()
    )

    for row in annotations_df.itertuples():

        report_text = report_lookup[
            row.report_id
        ]

        extracted_text = report_text[
            row.start_char:
            row.end_char
        ]

        if extracted_text != row.entity:

            raise ValueError(
                "\nAnnotation span validation failed.\n"
                f"Report: {row.report_id}\n"
                f"Expected: {row.entity!r}\n"
                f"Extracted: {extracted_text!r}\n"
                f"Span: "
                f"[{row.start_char}:{row.end_char}]"
            )


# ============================================================
# MAIN DATASET GENERATION
# ============================================================

def main() -> None:

    random.seed(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vocabulary = load_vocabulary()

    entities = collect_entities(
        vocabulary
    )

    if len(entities) < MAX_FINDINGS_PER_REPORT:
        raise ValueError(
            "Not enough vocabulary entities "
            "to generate reports."
        )

    reports = []
    annotations = []

    for index in range(
        1,
        N_REPORTS + 1,
    ):

        report_id = (
            f"R{index:04d}"
        )

        report_record, report_annotations = (
            generate_report(
                report_id=report_id,
                entities=entities,
            )
        )

        reports.append(
            report_record
        )

        annotations.extend(
            report_annotations
        )

    reports_df = pd.DataFrame(
        reports
    )

    annotations_df = pd.DataFrame(
        annotations
    )

    validate_annotations(
        reports_df,
        annotations_df,
    )

    reports_df.to_csv(
        REPORTS_PATH,
        index=False,
    )

    annotations_df.to_csv(
        ANNOTATIONS_PATH,
        index=False,
    )

    print(
        "Synthetic dataset generated successfully."
    )

    print(
        "Reports:",
        len(reports_df),
    )

    print(
        "Annotations:",
        len(annotations_df),
    )

    print(
        "\nTemplate family distribution:"
    )

    print(
        annotations_df[
            "template_family"
        ].value_counts()
    )

    conflicting = annotations_df[
        annotations_df[
            "template_family"
        ]
        == "CONFLICTING"
    ]

    print(
        "\nCONFLICTING annotations:",
        len(conflicting),
    )

    print(
        "\nSaved:"
    )

    print(
        REPORTS_PATH
    )

    print(
        ANNOTATIONS_PATH
    )


if __name__ == "__main__":
    main()