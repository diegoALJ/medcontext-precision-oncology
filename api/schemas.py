from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

class CompileRequest(BaseModel):
    report_text: str = Field(
        ...,
        min_length=1,
        description="Precision-oncology report text.",
    )
    report_id: str | None = Field(
        default=None,
        description="Optional report identifier.",
    )

class MolecularRepresentation(BaseModel):
    gene: str | None = None
    protein_change: str | None = None
    alteration_class: str | None = None
    gene_1: str | None = None
    gene_2: str | None = None
    value: str | float | int | None = None
    unit: str | None = None
    drug: str | None = None
    associated_target: str | None = None


class CompactFact(BaseModel):
    entity: str
    type: str

    assertion: Literal[
        "PRESENT",
        "ABSENT",
        "POSSIBLE",
    ]

    experiencer: Literal[
        "PATIENT",
        "FAMILY",
        "OTHER",
    ]

    temporality: Literal[
        "CURRENT",
        "HISTORICAL",
    ]

    certainty: Literal[
        "CONFIRMED",
        "SUSPECTED",
    ]

    molecular: MolecularRepresentation | None = None

class CompileResponse(BaseModel):
    schema_version: str
    packet_type: Literal["compact_ai"]
    facts: list[CompactFact]
    report_id: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]