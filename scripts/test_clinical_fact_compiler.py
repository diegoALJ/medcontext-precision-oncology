from __future__ import annotations

import json
from pathlib import Path

from src.clinical_fact_compiler import ClinicalFactCompiler


ROOT = Path(__file__).resolve().parents[1]

VOCAB_PATH = (
    ROOT
    / "data"
    / "vocabulary"
    / "oncology_vocabulary.yaml"
)

MODEL_DIR = (
    ROOT
    / "results"
    / "models"
)


def main() -> None:
    compiler = ClinicalFactCompiler(
        vocabulary_path=VOCAB_PATH,
        model_dir=MODEL_DIR,
    )

    report = (
        "IDH1 p.R132H cannot be excluded. "
        "The patient's father carried TP53 p.R175H. "
        "PML::RARA was previously detected, "
        "but current testing was negative for PML::RARA."
    )

    facts = compiler.compile(
        report
    )

    print(
        "NUMBER OF FACTS:",
        len(facts),
    )

    for fact in facts:
        print(
            "\n" + "=" * 80
        )

        print(
            json.dumps(
                fact,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
