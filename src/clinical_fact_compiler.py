from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.candidate_finder import CandidateFinder
from src.precision_parser import PrecisionParser
from src.context_window import ContextWindowExtractor
from src.rules import ClinicalRuleExtractor
from src.embeddings import FrozenBioEmbedder
from src.hybrid_features import HybridFeatureBuilder


class ClinicalFactCompiler:
    """
    Compile unstructured precision-oncology text into structured
    contextualized clinical facts.
    """

    TARGETS = (
        "assertion",
        "experiencer",
        "temporality",
        "certainty",
    )

    def __init__(
        self,
        vocabulary_path: Path,
        model_dir: Path,
        device: str | None = None,
    ):
        self.finder = CandidateFinder(vocabulary_path)
        self.parser = PrecisionParser(vocabulary_path)
        self.window_extractor = ContextWindowExtractor()
        self.rule_extractor = ClinicalRuleExtractor()
        self.embedder = FrozenBioEmbedder(device=device)
        self.feature_builder = HybridFeatureBuilder()

        self.models = {}
        self.encoders = {}

        self._load_models(model_dir)

    def _load_models(
        self,
        model_dir: Path,
    ) -> None:
        """
        Load the four trained XGBoost classifiers and their
        corresponding LabelEncoders.
        """
        for target in self.TARGETS:
            model_path = (
                model_dir
                / f"{target}_xgboost.joblib"
            )

            encoder_path = (
                model_dir
                / f"{target}_label_encoder.joblib"
            )

            self.models[target] = joblib.load(
                model_path
            )

            self.encoders[target] = joblib.load(
                encoder_path
            )

    def compile(
        self,
        report_text: str,
    ) -> list[dict]:
        """
        Run the full MedContext inference pipeline over one report.

        Returns one structured clinical fact per detected entity mention.
        """
        candidates = self.finder.find(
            report_text
        )

        if not candidates:
            return []

        prepared = []

        for candidate in candidates:
            parsed_entity = self.parser.parse(
                candidate,
                source_text=report_text,
            )

            context = self.window_extractor.extract(
                report_text,
                candidate,
            )

            ruled = self.rule_extractor.extract(
                context
            )

            prepared.append(
                {
                    "candidate": candidate,
                    "parsed_entity": parsed_entity,
                    "context": ruled,
                }
            )

        contexts = [
            item["context"]["context_text"]
            for item in prepared
        ]

        rule_features = [
            item["context"]["rule_features"]
            for item in prepared
        ]

        embeddings = self.embedder.encode(
            contexts
        )

        X = self.feature_builder.build_batch(
            rule_features,
            embeddings,
        )

        predictions, confidences = (
            self._predict_context(X)
        )

        facts = []

        for index, item in enumerate(prepared):
            fact = self._build_fact(
                candidate=item["candidate"],
                parsed_entity=item["parsed_entity"],
                context=item["context"],
                predictions={
                    target: predictions[target][index]
                    for target in self.TARGETS
                },
                confidences={
                    target: confidences[target][index]
                    for target in self.TARGETS
                },
            )

            facts.append(fact)

        return facts

    def _predict_context(
        self,
        X: np.ndarray,
    ) -> tuple[dict, dict]:
        """
        Predict the four contextual labels and their confidence scores.

        Confidence is the maximum class probability returned by the
        corresponding XGBoost classifier.
        """
        predictions = {}
        confidences = {}

        for target in self.TARGETS:
            model = self.models[target]
            encoder = self.encoders[target]

            encoded_predictions = (
                model.predict(X)
                .astype(int)
            )

            probabilities = model.predict_proba(
                X
            )

            predictions[target] = (
                encoder.inverse_transform(
                    encoded_predictions
                )
            )

            confidences[target] = (
                probabilities.max(
                    axis=1
                )
            )

        return predictions, confidences

    def _build_fact(
        self,
        candidate: dict,
        parsed_entity: dict,
        context: dict,
        predictions: dict,
        confidences: dict,
    ) -> dict:
        """
        Merge molecular parsing and contextual predictions into one
        machine-readable clinical fact.
        """
        overall_confidence = float(
            np.mean(
                list(
                    confidences.values()
                )
            )
        )

        return {
            "entity": candidate[
                "entity_text"
            ],
            "normalized_entity": candidate[
                "normalized_entity"
            ],
            "entity_type": candidate[
                "entity_type"
            ],
            "molecular": parsed_entity,
            "assertion": predictions[
                "assertion"
            ],
            "experiencer": predictions[
                "experiencer"
            ],
            "temporality": predictions[
                "temporality"
            ],
            "certainty": predictions[
                "certainty"
            ],
            "confidence": round(
                overall_confidence,
                4,
            ),
            "target_confidence": {
                target: round(
                    float(score),
                    4,
                )
                for target, score
                in confidences.items()
            },
            "source_text": context[
                "context_text"
            ],
            "start_char": candidate[
                "start_char"
            ],
            "end_char": candidate[
                "end_char"
            ],
        }
