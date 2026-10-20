from __future__ import annotations

import re
from pathlib import Path

import yaml


class PrecisionParser:
    """
    Convert Candidate Finder entities into structured molecular objects.

    This stage does NOT infer clinical context such as PRESENT/ABSENT,
    temporality, experiencer, or certainty.
    """

    PROTEIN_VARIANT_RE = re.compile(
        r"^p\.([A-Z])(\d+)([A-Z])$",
        flags=re.IGNORECASE,
    )

    EXON_SKIPPING_RE = re.compile(
        r"exon\s+(\d+)\s+skipping",
        flags=re.IGNORECASE,
    )

    def __init__(self, vocabulary_path: str | Path):
        self.vocabulary_path = Path(vocabulary_path)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """
        Build a lookup table keyed by:
            (canonical_entity, entity_type)

        This gives the parser access to metadata already stored in the YAML,
        such as parent genes, biomarker value type, units, or drug targets.
        """
        with self.vocabulary_path.open("r", encoding="utf-8") as f:
            vocab = yaml.safe_load(f)

        lookup = {}

        section_to_type = {
            "genes": "GENE",
            "variants": "VARIANT",
            "fusions": "FUSION",
            "biomarkers": "BIOMARKER",
            "drugs": "DRUG",
        }

        for disease_key, disease in vocab["diseases"].items():
            lookup[(disease["canonical"], "DISEASE")] = {
                "disease": disease_key,
                **disease,
            }

            for section, entity_type in section_to_type.items():
                for item in disease.get(section, []):
                    key = (item["canonical"], entity_type)

                    if key not in lookup:
                        lookup[key] = {
                            "disease": disease_key,
                            **item,
                        }

        return lookup

    def parse(self, candidate: dict, source_text: str | None = None) -> dict:
        """
        Parse one Candidate Finder result into a structured molecular object.
        """
        entity_type = candidate["entity_type"]
        canonical = candidate["normalized_entity"]

        result = {
            **candidate,
            "parsed": {},
        }

        if entity_type == "GENE":
            result["parsed"] = self._parse_gene(canonical)

        elif entity_type == "VARIANT":
            result["parsed"] = self._parse_variant(canonical)

        elif entity_type == "FUSION":
            result["parsed"] = self._parse_fusion(canonical)

        elif entity_type == "BIOMARKER":
            result["parsed"] = self._parse_biomarker(
                canonical=canonical,
                candidate=candidate,
                source_text=source_text,
            )

        elif entity_type == "DRUG":
            result["parsed"] = self._parse_drug(canonical)

        elif entity_type == "DISEASE":
            result["parsed"] = {
                "disease": canonical,
            }

        return result

    def _get_metadata(self, canonical: str, entity_type: str) -> dict:
        return self.metadata.get((canonical, entity_type), {})

    def _parse_gene(self, canonical: str) -> dict:
        return {
            "gene": canonical,
        }

    def _parse_variant(self, canonical: str) -> dict:
        metadata = self._get_metadata(canonical, "VARIANT")
        parent_gene = metadata.get("parent_gene")

        parsed = {
            "gene": parent_gene,
            "variant": canonical,
        }

        if " p." in canonical:
            protein_change = canonical.split(" ", 1)[1]
            match = self.PROTEIN_VARIANT_RE.fullmatch(protein_change)

            if match:
                ref_aa, position, alt_aa = match.groups()

                parsed.update({
                    "alteration_class": "PROTEIN_SUBSTITUTION",
                    "protein_change": protein_change,
                    "reference_aa": ref_aa.upper(),
                    "position": int(position),
                    "alternate_aa": alt_aa.upper(),
                })
                return parsed

        if canonical.endswith("-ITD"):
            parsed["alteration_class"] = "ITD"
            return parsed

        if canonical.endswith("-TKD"):
            parsed["alteration_class"] = "TKD"
            return parsed

        exon_match = self.EXON_SKIPPING_RE.search(canonical)
        if exon_match:
            parsed.update({
                "alteration_class": "EXON_SKIPPING",
                "exon": int(exon_match.group(1)),
            })
            return parsed

        parsed["alteration_class"] = "GENERIC_MUTATION"
        return parsed

    def _parse_fusion(self, canonical: str) -> dict:
        metadata = self._get_metadata(canonical, "FUSION")
        parent_genes = metadata.get("parent_genes", [])

        parsed = {
            "fusion": canonical,
            "genes": parent_genes,
        }

        if len(parent_genes) == 2:
            parsed["gene_1"] = parent_genes[0]
            parsed["gene_2"] = parent_genes[1]
            parsed["alteration_class"] = "GENE_FUSION"
        else:
            parsed["alteration_class"] = "REARRANGEMENT"

        return parsed

    def _parse_biomarker(
        self,
        canonical: str,
        candidate: dict,
        source_text: str | None,
    ) -> dict:
        metadata = self._get_metadata(canonical, "BIOMARKER")

        parsed = {
            "biomarker": canonical,
            "value_type": metadata.get("value_type"),
        }

        if "unit" in metadata:
            parsed["unit"] = metadata["unit"]

        if source_text:
            value = self._extract_nearby_value(candidate, source_text)
            if value is not None:
                parsed["value"] = value

        return parsed

    @staticmethod
    def _extract_nearby_value(candidate: dict, source_text: str):
        """
        Search a short window after the biomarker mention for a percentage.

        Example:
            "PD-L1 TPS was 65%."
                       -> 65.0
        """
        end = candidate["end_char"]
        window = source_text[end:end + 40]

        match = re.search(r"(\d+(?:\.\d+)?)\s*%", window)
        if match:
            return float(match.group(1))

        return None

    def _parse_drug(self, canonical: str) -> dict:
        metadata = self._get_metadata(canonical, "DRUG")

        return {
            "drug": canonical,
            "associated_target": metadata.get("associated_target"),
        }
