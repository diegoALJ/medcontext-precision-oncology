from __future__ import annotations

import re
from pathlib import Path

import yaml


class CandidateFinder:
    def __init__(self, vocabulary_path: str | Path):
        self.vocabulary_path = Path(vocabulary_path)
        self.entities = self._load_entities()

    def _load_entities(self) -> list[dict]:
        with self.vocabulary_path.open("r", encoding="utf-8") as f:
            vocab = yaml.safe_load(f)

        entities = []

        for disease_key, disease in vocab["diseases"].items():
            entities.append({
                "canonical": disease["canonical"],
                "aliases": disease.get("aliases", []),
                "entity_type": "DISEASE",
                "disease": disease_key,
            })

            mapping = {
                "genes": "GENE",
                "variants": "VARIANT",
                "fusions": "FUSION",
                "biomarkers": "BIOMARKER",
                "drugs": "DRUG",
            }

            for section, entity_type in mapping.items():
                for item in disease.get(section, []):
                    entities.append({
                        "canonical": item["canonical"],
                        "aliases": item.get("aliases", [item["canonical"]]),
                        "entity_type": entity_type,
                        "disease": disease_key,
                    })

        return entities

    @staticmethod
    def _find_alias(text: str, alias: str) -> list[tuple[int, int]]:
        pattern = re.compile(
            rf"(?<!\w){re.escape(alias)}(?!\w)",
            flags=re.IGNORECASE,
        )
        return [(m.start(), m.end()) for m in pattern.finditer(text)]

    def find(self, text: str) -> list[dict]:
        candidates = []

        for entity in self.entities:
            for alias in entity["aliases"]:
                for start_char, end_char in self._find_alias(text, alias):
                    candidates.append({
                        "entity_text": text[start_char:end_char],
                        "normalized_entity": entity["canonical"],
                        "entity_type": entity["entity_type"],
                        "disease": entity["disease"],
                        "start_char": start_char,
                        "end_char": end_char,
                    })

        return self._remove_overlaps(candidates)

    @staticmethod
    def _remove_overlaps(candidates: list[dict]) -> list[dict]:
        candidates = sorted(
            candidates,
            key=lambda x: (
                x["start_char"],
                -(x["end_char"] - x["start_char"]),
            ),
        )

        selected = []

        for candidate in candidates:
            overlaps = any(
                candidate["start_char"] < existing["end_char"]
                and candidate["end_char"] > existing["start_char"]
                for existing in selected
            )

            if not overlaps:
                selected.append(candidate)

        return sorted(selected, key=lambda x: x["start_char"])
