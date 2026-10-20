from __future__ import annotations

import json
from pathlib import Path

from src.clinical_fact_compiler import ClinicalFactCompiler
from src.medcontext_packet import MedContextPacketBuilder


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

OUTPUT_PATH = (
    ROOT
    / "results"
    / "example_medcontext_packet.json"
)


def main() -> None:

    compiler = ClinicalFactCompiler(
        vocabulary_path=VOCAB_PATH,
        model_dir=MODEL_DIR,
    )

    packet_builder = MedContextPacketBuilder(
        compiler
    )

    report = (
        "IDH1 p.R132H cannot be excluded. "
        "The patient's father carried TP53 p.R175H. "
        "PML::RARA was previously detected, "
        "but current testing was negative for PML::RARA."
    )

    packet = packet_builder.compile_report(
        report_text=report,
        report_id="DEMO_001",
    )

    packet_builder.save_json(
        packet,
        OUTPUT_PATH,
    )

    print(
        "REPORT ID:",
        packet["report_id"],
    )

    print(
        "FACT COUNT:",
        packet["summary"]["fact_count"],
    )

    print(
        "UNIQUE ENTITIES:",
        packet["summary"]["unique_entities"],
    )

    print(
        "\nSUMMARY:"
    )

    print(
        json.dumps(
            packet["summary"],
            indent=2,
        )
    )

    print(
        "\nFIRST FACT:"
    )

    print(
        json.dumps(
            packet["facts"][0],
            indent=2,
        )
    )

    print(
        "\nSAVED TO:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
