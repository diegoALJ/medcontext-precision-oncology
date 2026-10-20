from __future__ import annotations

import numpy as np


class HybridFeatureBuilder:
    """
    Combine symbolic clinical features with Transformer embeddings.

    This class does not train a model.
    It only builds the final numerical feature vector used by XGBoost.
    """

    SYMBOLIC_FEATURE_ORDER = (
        "has_negation",
        "has_uncertainty",
        "has_historical_cue",
        "has_current_cue",
        "has_family_context",
        "has_confirmation",
        "negation_distance",
        "uncertainty_distance",
        "family_distance",
    )

    def build_one(
        self,
        rule_features: dict,
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Build one hybrid feature vector.

        Output:
        [symbolic features | Transformer embedding]
        """
        symbolic_vector = self._symbolic_to_array(rule_features)

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.ndim != 1:
            raise ValueError(
                "Expected a 1D embedding for build_one()."
            )

        return np.concatenate(
            [symbolic_vector, embedding]
        ).astype(np.float32)

    def build_batch(
        self,
        rule_features_list: list[dict],
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Build hybrid feature vectors for multiple entity-context pairs.

        Returns a matrix with shape:
        (n_samples, n_symbolic_features + embedding_dim)
        """
        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Expected a 2D embedding matrix for build_batch()."
            )

        if len(rule_features_list) != len(embeddings):
            raise ValueError(
                "Number of symbolic feature rows must match "
                "the number of embeddings."
            )

        symbolic_matrix = np.vstack([
            self._symbolic_to_array(features)
            for features in rule_features_list
        ])

        return np.hstack(
            [symbolic_matrix, embeddings]
        ).astype(np.float32)

    def _symbolic_to_array(
        self,
        rule_features: dict,
    ) -> np.ndarray:
        """
        Convert the symbolic feature dictionary into a fixed-order
        numeric vector.

        Missing cue distances (-1) are converted to 0 because their
        corresponding binary feature already indicates cue absence.
        """
        values = []

        for feature_name in self.SYMBOLIC_FEATURE_ORDER:
            value = rule_features[feature_name]

            if feature_name.endswith("_distance") and value == -1:
                value = 0

            values.append(value)

        return np.asarray(
            values,
            dtype=np.float32,
        )

    @property
    def symbolic_dimension(self) -> int:
        """
        Return the number of symbolic features.
        """
        return len(self.SYMBOLIC_FEATURE_ORDER)

    def feature_names(
        self,
        embedding_dimension: int,
    ) -> list[str]:
        """
        Return names for every feature in the hybrid vector.

        Useful later for debugging and feature importance.
        """
        embedding_names = [
            f"embedding_{i}"
            for i in range(embedding_dimension)
        ]

        return list(self.SYMBOLIC_FEATURE_ORDER) + embedding_names
