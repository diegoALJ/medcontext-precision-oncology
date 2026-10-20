from __future__ import annotations

import re


class ContextWindowExtractor:
    """
    Extract the smallest clinically relevant text window around a biomedical entity.

    Default behavior:
    - keep the sentence containing the entity;
    - optionally include the previous or next sentence when clinical continuation
      cues suggest that the neighboring sentence modifies or updates the finding.
    """

    CONTINUATION_CUES = (
    "however",
    "but",
    "later",
    "subsequently",
    "repeat testing",
    "repeat sequencing",
    "follow-up",
    "current testing",
    "current sequencing",
    "no longer",
    "became",
    "remained",
    "previously",
    "initially",
)

    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

    def extract(self, report_text: str, candidate: dict) -> dict:
        """
        Extract an adaptive context window for one Candidate Finder entity.
        """
        sentences = self._segment_sentences(report_text)

        entity_start = candidate["start_char"]
        entity_end = candidate["end_char"]

        sentence_index = self._find_entity_sentence(
            sentences=sentences,
            entity_start=entity_start,
            entity_end=entity_end,
        )

        if sentence_index is None:
            raise ValueError(
                f"Could not locate entity '{candidate['entity_text']}' "
                "inside segmented report text."
            )

        selected_indices = [sentence_index]

        if self._should_include_previous(sentences, sentence_index):
            selected_indices.insert(0, sentence_index - 1)

        if self._should_include_next(sentences, sentence_index):
            selected_indices.append(sentence_index + 1)

        context_start = sentences[selected_indices[0]]["start"]
        context_end = sentences[selected_indices[-1]]["end"]

        raw_context = report_text[context_start:context_end]
        left_trim = len(raw_context) - len(raw_context.lstrip())
        context_start += left_trim
        context_text = report_text[context_start:context_end].strip()

        entity_start_in_context = entity_start - context_start
        entity_end_in_context = entity_end - context_start

        return {
            **candidate,
            "context_text": context_text,
            "context_start": context_start,
            "context_end": context_end,
            "entity_start_in_context": entity_start_in_context,
            "entity_end_in_context": entity_end_in_context,
            "num_sentences": len(selected_indices),
        }

    def _segment_sentences(self, text: str) -> list[dict]:
        """
        Split the report into sentence-like spans while preserving character offsets.
        """
        sentences = []
        last_end = 0

        for match in self.SENTENCE_SPLIT_RE.finditer(text):
            start = last_end
            end = match.start()
            sentence_text = text[start:end]

            if sentence_text.strip():
                left_trim = len(sentence_text) - len(sentence_text.lstrip())
                right_trim = len(sentence_text) - len(sentence_text.rstrip())

                sentence_start = start + left_trim
                sentence_end = end - right_trim

                sentences.append({
                    "text": text[sentence_start:sentence_end],
                    "start": sentence_start,
                    "end": sentence_end,
                })

            last_end = match.end()

        if last_end < len(text):
            sentence_text = text[last_end:]

            if sentence_text.strip():
                left_trim = len(sentence_text) - len(sentence_text.lstrip())
                right_trim = len(sentence_text) - len(sentence_text.rstrip())

                sentence_start = last_end + left_trim
                sentence_end = len(text) - right_trim

                sentences.append({
                    "text": text[sentence_start:sentence_end],
                    "start": sentence_start,
                    "end": sentence_end,
                })

        return sentences

    @staticmethod
    def _find_entity_sentence(
        sentences: list[dict],
        entity_start: int,
        entity_end: int,
    ) -> int | None:
        """
        Return the index of the sentence span containing the entity.
        """
        for i, sentence in enumerate(sentences):
            if (
                entity_start >= sentence["start"]
                and entity_end <= sentence["end"]
            ):
                return i

        return None

    def _should_include_previous(
        self,
        sentences: list[dict],
        sentence_index: int,
        ) -> bool:
        """
        Include the previous sentence only when the current sentence itself
        signals that it continues or updates previous clinical information.
        """
        if sentence_index == 0:
            return False

        current = sentences[sentence_index]["text"].lower().strip()

        previous_link_cues = (
            "however",
        "but",
        "later",
        "subsequently",
        "repeat testing",
        "repeat sequencing",
        "follow-up",
        "current testing",
        "current sequencing",
        "confirmation",
        "no longer",
    )

        return any(
            current.startswith(cue)
            for cue in previous_link_cues
    )
    def _should_include_next(
    self,
    sentences: list[dict],
    sentence_index: int,
    ) -> bool:
        """
        Include the next sentence only when it explicitly signals
        continuation or an update of the current clinical finding.
        """
        if sentence_index >= len(sentences) - 1:
            return False

        next_sentence = sentences[sentence_index + 1]["text"].lower().strip()

        next_link_cues = (
            "however",
            "but",
            "later",
            "subsequently",
            "repeat testing",
            "repeat sequencing",
            "follow-up",
            "current testing",
            "current sequencing",
            "no longer",
        )

        return any(
            next_sentence.startswith(cue)
            for cue in next_link_cues
        )

    def _contains_continuation_cue(self, text: str) -> bool:
        """
        Return True when a sentence contains a predefined continuation cue.
        """
        return any(cue in text for cue in self.CONTINUATION_CUES)
