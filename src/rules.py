from __future__ import annotations

import re


class ClinicalRuleExtractor:
    """
    Extract deterministic clinical context features from an entity context window.

    This class does NOT make the final clinical classification.
    It only converts explicit linguistic cues into structured features.
    """

    CUE_GROUPS = {
        "negation": (
            "no",
            "not",
            "negative for",
            "not detected",
            "did not identify",
            "failed to detect",
            "no evidence of",
        ),
        "uncertainty": (
            "suspected",
            "possible",
            "cannot be excluded",
            "suggestive of",
            "uncertain",
            "may represent",
        ),
        "historical": (
            "previously",
            "at diagnosis",
            "initially",
            "earlier",
            "prior",
            "previous",
        ),
        "current": (
            "current",
            "currently",
            "repeat testing",
            "repeat sequencing",
            "most recent",
            "now",
        ),
        "family": (
            "mother",
            "father",
            "sibling",
            "family member",
            "family history",
            "relative",
        ),
        "confirmation": (
            "confirmed",
            "detected",
            "identified",
            "positive for",
            "demonstrated",
        ),
    }

    def extract(self, context: dict) -> dict:
        """
        Convert one ContextWindowExtractor result into symbolic features.
        """
        text = context["context_text"]
        entity_start = context["entity_start_in_context"]
        entity_end = context["entity_end_in_context"]

        cue_matches = {
            cue_type: self._find_cues(text, cues)
            for cue_type, cues in self.CUE_GROUPS.items()
        }

        features = {
            "has_negation": int(bool(cue_matches["negation"])),
            "has_uncertainty": int(bool(cue_matches["uncertainty"])),
            "has_historical_cue": int(bool(cue_matches["historical"])),
            "has_current_cue": int(bool(cue_matches["current"])),
            "has_family_context": int(bool(cue_matches["family"])),
            "has_confirmation": int(bool(cue_matches["confirmation"])),
            "negation_distance": self._nearest_distance(
                cue_matches["negation"], entity_start, entity_end
            ),
            "uncertainty_distance": self._nearest_distance(
                cue_matches["uncertainty"], entity_start, entity_end
            ),
            "family_distance": self._nearest_distance(
                cue_matches["family"], entity_start, entity_end
            ),
        }

        return {
            **context,
            "rule_features": features,
            "rule_matches": cue_matches,
        }

    @staticmethod
    def _find_cues(text: str, cues: tuple[str, ...]) -> list[dict]:
        """
        Find every cue occurrence while preserving character offsets.

        Longer cues are searched first so phrases such as "not detected"
        are preferred over shorter overlapping cues such as "not".
        """
        matches = []
        occupied_spans = []

        for cue in sorted(cues, key=len, reverse=True):
            pattern = re.compile(
                rf"(?<!\w){re.escape(cue)}(?!\w)",
                flags=re.IGNORECASE,
            )

            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()

                overlaps = any(
                    start < occupied_end and end > occupied_start
                    for occupied_start, occupied_end in occupied_spans
                )

                if overlaps:
                    continue

                matches.append({
                    "cue": text[start:end],
                    "start": start,
                    "end": end,
                })
                occupied_spans.append((start, end))

        return sorted(matches, key=lambda item: item["start"])

    @staticmethod
    def _nearest_distance(
        matches: list[dict],
        entity_start: int,
        entity_end: int,
    ) -> int:
        """
        Return the shortest character distance between a cue and the entity.

        If no cue exists, return -1.
        """
        if not matches:
            return -1

        distances = []

        for match in matches:
            cue_start = match["start"]
            cue_end = match["end"]

            if cue_end <= entity_start:
                distance = entity_start - cue_end
            elif cue_start >= entity_end:
                distance = cue_start - entity_end
            else:
                distance = 0

            distances.append(distance)

        return min(distances)
