from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class MultiTargetXGBoost:
    """
    Train one independent XGBoost classifier for each MedContext target:
    assertion, experiencer, temporality, and certainty.
    """

    TARGETS = (
        "assertion",
        "experiencer",
        "temporality",
        "certainty",
    )

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: dict[str, XGBClassifier] = {}
        self.encoders: dict[str, LabelEncoder] = {}

    def _build_model(self, n_classes: int) -> XGBClassifier:
        """
        Create a compact XGBoost classifier.

        Binary targets use binary logistic loss.
        Multiclass targets use softprob.
        """
        common_params = {
            "n_estimators": 250,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": self.random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        if n_classes == 2:
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **common_params,
            )

        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            **common_params,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: pd.DataFrame,
    ) -> None:
        """
        Train one model per target.

        Each text label is converted to an integer using LabelEncoder.
        """
        for target in self.TARGETS:
            encoder = LabelEncoder()

            y_encoded = encoder.fit_transform(
                y_train[target]
            )

            model = self._build_model(
                n_classes=len(encoder.classes_)
            )

            model.fit(
                X_train,
                y_encoded,
            )

            self.encoders[target] = encoder
            self.models[target] = model

    def predict(
        self,
        X: np.ndarray,
    ) -> pd.DataFrame:
        """
        Predict all four clinical context targets.
        """
        predictions = {}

        for target in self.TARGETS:
            model = self.models[target]
            encoder = self.encoders[target]

            encoded_pred = model.predict(X)

            predictions[target] = encoder.inverse_transform(
                encoded_pred.astype(int)
            )

        return pd.DataFrame(predictions)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: pd.DataFrame,
    ) -> dict:
        """
        Evaluate every target using accuracy and macro F1.

        Macro F1 gives the same importance to every class.
        """
        y_pred = self.predict(X)

        metrics = {}

        for target in self.TARGETS:
            metrics[target] = {
                "accuracy": accuracy_score(
                    y_true[target],
                    y_pred[target],
                ),
                "macro_f1": f1_score(
                    y_true[target],
                    y_pred[target],
                    average="macro",
                ),
                "classification_report": classification_report(
                    y_true[target],
                    y_pred[target],
                    output_dict=True,
                    zero_division=0,
                ),
            }

        return metrics

    def save(
        self,
        output_dir: Path,
    ) -> None:
        """
        Persist all trained models and label encoders.
        """
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for target in self.TARGETS:
            joblib.dump(
                self.models[target],
                output_dir / f"{target}_xgboost.joblib",
            )

            joblib.dump(
                self.encoders[target],
                output_dir / f"{target}_label_encoder.joblib",
            )
