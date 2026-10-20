from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.clinical_fact_compiler import ClinicalFactCompiler


class MedContextPacketBuilder:
    """
    Build report-level MedContext outputs from compiled clinical facts.

    Two representations are supported:

    1. Full Packet:
       Designed for auditability, debugging, UI display, and traceability.

    2. Compact AI Packet:
       Designed for downstream LLMs, RAG systems, and agents where
       unnecessary text and metadata should be removed.
    """

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        compiler: ClinicalFactCompiler,
    ):
        self.compiler = compiler

    # ========================================================
    # FULL PACKET
    # ========================================================

    def compile_report(
        self,
        report_text: str,
        report_id: str | None = None,
    ) -> dict:
        """
        Run the Clinical Fact Compiler and build the full
        audit-oriented MedContext Packet.
        """
        facts = self.compiler.compile(
            report_text
        )

        return self.build_packet(
            report_text=report_text,
            facts=facts,
            report_id=report_id,
        )

    def build_packet(
        self,
        report_text: str,
        facts: list[dict],
        report_id: str | None = None,
    ) -> dict:
        """
        Build the complete audit-oriented packet.

        This representation preserves:
        - original report
        - summaries
        - molecular parsing
        - confidence scores
        - offsets
        - source context
        """
        summary = self._build_summary(
            facts
        )

        return {
            "schema_version": self.SCHEMA_VERSION,
            "packet_type": "full",
            "report_id": report_id,
            "source_report": report_text,
            "summary": summary,
            "facts": facts,
        }

    # ========================================================
    # COMPACT AI PACKET
    # ========================================================

    def compile_compact_report(
        self,
        report_text: str,
        report_id: str | None = None,
    ) -> dict:
        """
        Run the Clinical Fact Compiler and return a compact packet
        intended for downstream AI systems.

        The original report and audit metadata are intentionally
        omitted.
        """
        facts = self.compiler.compile(
            report_text
        )

        return self.build_compact_packet(
            facts=facts,
            report_id=report_id,
        )

    def build_compact_packet(
        self,
        facts: list[dict],
        report_id: str | None = None,
    ) -> dict:
        """
        Convert full Clinical Facts into a token-efficient
        representation for LLMs, RAG systems, or agents.
        """
        compact_facts = [
            self._compact_fact(fact)
            for fact in facts
        ]

        packet = {
            "schema_version": self.SCHEMA_VERSION,
            "packet_type": "compact_ai",
            "facts": compact_facts,
        }

        if report_id:
            packet["report_id"] = report_id

        return packet

    @staticmethod
    def _compact_fact(
        fact: dict,
    ) -> dict:
        """
        Keep only information that is useful for downstream
        contextual reasoning.

        Audit information such as offsets, source text,
        confidence breakdowns, and repeated parser metadata
        is intentionally removed.
        """
        compact = {
            "entity": fact[
                "normalized_entity"
            ],
            "type": fact[
                "entity_type"
            ],
            "assertion": fact[
                "assertion"
            ],
            "experiencer": fact[
                "experiencer"
            ],
            "temporality": fact[
                "temporality"
            ],
            "certainty": fact[
                "certainty"
            ],
        }

        parsed = (
            fact.get(
                "molecular",
                {}
            )
            .get(
                "parsed",
                {}
            )
        )

        molecular = (
            MedContextPacketBuilder
            ._compact_molecular_representation(
                parsed
            )
        )

        if molecular:
            compact[
                "molecular"
            ] = molecular

        return compact

    @staticmethod
    def _compact_molecular_representation(
        parsed: dict,
    ) -> dict:
        """
        Preserve useful molecular structure without copying all
        Precision Parser metadata into the compact packet.
        """
        useful_fields = (
            "gene",
            "protein_change",
            "alteration_class",
            "gene_1",
            "gene_2",
            "value",
            "unit",
            "drug",
            "associated_target",
        )

        return {
            key: parsed[key]
            for key in useful_fields
            if (
                key in parsed
                and parsed[key] is not None
            )
        }

    # ========================================================
    # SUMMARY
    # ========================================================

    @staticmethod
    def _build_summary(
        facts: list[dict],
    ) -> dict:
        """
        Create lightweight report-level counts for the full packet.

        The summary is descriptive only and does not modify
        the underlying Clinical Facts.
        """
        entity_types = Counter(
            fact["entity_type"]
            for fact in facts
        )

        assertions = Counter(
            fact["assertion"]
            for fact in facts
        )

        experiencers = Counter(
            fact["experiencer"]
            for fact in facts
        )

        temporalities = Counter(
            fact["temporality"]
            for fact in facts
        )

        certainties = Counter(
            fact["certainty"]
            for fact in facts
        )

        unique_entities = sorted(
            {
                fact[
                    "normalized_entity"
                ]
                for fact in facts
            }
        )

        return {
            "fact_count": len(facts),
            "unique_entity_count": len(
                unique_entities
            ),
            "unique_entities": (
                unique_entities
            ),
            "entity_type_counts": dict(
                entity_types
            ),
            "assertion_counts": dict(
                assertions
            ),
            "experiencer_counts": dict(
                experiencers
            ),
            "temporality_counts": dict(
                temporalities
            ),
            "certainty_counts": dict(
                certainties
            ),
        }

    # ========================================================
    # EXPORT
    # ========================================================

    @staticmethod
    def save_json(
        packet: dict,
        output_path: Path,
    ) -> None:
        """
        Save any MedContext packet as readable UTF-8 JSON.
        """
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                packet,
                file,
                indent=2,
                ensure_ascii=False,
            )