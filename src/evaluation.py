from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


class PipelineEvaluator:
    """
    Evaluate MedContext end-to-end at the Clinical Fact level.

    Ground-truth facts and predicted facts are matched using:
    normalized_entity + start_char + end_char.
    """

    TARGETS = (
        "assertion",
        "experiencer",
        "temporality",
        "certainty",
    )

    def __init__(self):
        self.latencies = []

    @staticmethod
    def fact_key(
        normalized_entity: str,
        start_char: int,
        end_char: int,
    ) -> tuple:
        """
        Build a unique identity for one entity mention.

        Offsets are required because the same molecular entity may
        appear more than once in the same report.
        """
        return (
            normalized_entity,
            int(start_char),
            int(end_char),
        )

    def evaluate_reports(
        self,
        compiler,
        reports_df: pd.DataFrame,
        annotations_df: pd.DataFrame,
    ) -> dict:
        """
        Run the complete ClinicalFactCompiler on every test report
        and compare predicted facts against ground truth.
        """
        annotation_groups = {
            report_id: group
            for report_id, group
            in annotations_df.groupby("report_id")
        }

        matched_truth = defaultdict(list)
        matched_pred = defaultdict(list)

        total_expected = 0
        total_predicted = 0
        total_matched = 0
        exact_context_matches = 0

        error_rows = []

        self.latencies = []

        for report in reports_df.itertuples():
            report_id = report.report_id
            report_text = report.report_text

            expected_group = annotation_groups.get(
                report_id,
                pd.DataFrame(),
            )

            expected_map = {}

            for row in expected_group.itertuples():
                key = self.fact_key(
                    row.normalized_entity,
                    row.start_char,
                    row.end_char,
                )

                expected_map[key] = {
                    "assertion": row.assertion,
                    "experiencer": row.experiencer,
                    "temporality": row.temporality,
                    "certainty": row.certainty,
                }

            start_time = perf_counter()

            predicted_facts = compiler.compile(
                report_text
            )

            elapsed = perf_counter() - start_time
            self.latencies.append(elapsed)

            predicted_map = {}

            for fact in predicted_facts:
                key = self.fact_key(
                    fact["normalized_entity"],
                    fact["start_char"],
                    fact["end_char"],
                )

                predicted_map[key] = fact

            expected_keys = set(expected_map)
            predicted_keys = set(predicted_map)

            matched_keys = expected_keys & predicted_keys
            missed_keys = expected_keys - predicted_keys
            extra_keys = predicted_keys - expected_keys

            total_expected += len(expected_keys)
            total_predicted += len(predicted_keys)
            total_matched += len(matched_keys)

            for key in matched_keys:
                truth = expected_map[key]
                pred = predicted_map[key]

                all_targets_correct = True

                for target in self.TARGETS:
                    matched_truth[target].append(
                        truth[target]
                    )

                    matched_pred[target].append(
                        pred[target]
                    )

                    if truth[target] != pred[target]:
                        all_targets_correct = False

                if all_targets_correct:
                    exact_context_matches += 1

                if not all_targets_correct:
                    error_rows.append(
                        {
                            "report_id": report_id,
                            "normalized_entity": key[0],
                            "start_char": key[1],
                            "end_char": key[2],
                            "error_type": "label_mismatch",
                            "source_text": pred["source_text"],
                            "true_assertion": truth["assertion"],
                            "pred_assertion": pred["assertion"],
                            "true_experiencer": truth["experiencer"],
                            "pred_experiencer": pred["experiencer"],
                            "true_temporality": truth["temporality"],
                            "pred_temporality": pred["temporality"],
                            "true_certainty": truth["certainty"],
                            "pred_certainty": pred["certainty"],
                            "confidence": pred["confidence"],
                        }
                    )

            for key in missed_keys:
                error_rows.append(
                    {
                        "report_id": report_id,
                        "normalized_entity": key[0],
                        "start_char": key[1],
                        "end_char": key[2],
                        "error_type": "missed_fact",
                    }
                )

            for key in extra_keys:
                pred = predicted_map[key]

                error_rows.append(
                    {
                        "report_id": report_id,
                        "normalized_entity": key[0],
                        "start_char": key[1],
                        "end_char": key[2],
                        "error_type": "extra_fact",
                        "source_text": pred["source_text"],
                        "confidence": pred["confidence"],
                    }
                )

        metrics = self._build_metrics(
            matched_truth=matched_truth,
            matched_pred=matched_pred,
            total_expected=total_expected,
            total_predicted=total_predicted,
            total_matched=total_matched,
            exact_context_matches=exact_context_matches,
        )

        errors_df = pd.DataFrame(
            error_rows
        )

        return {
            "metrics": metrics,
            "errors": errors_df,
        }

    def _build_metrics(
        self,
        matched_truth: dict,
        matched_pred: dict,
        total_expected: int,
        total_predicted: int,
        total_matched: int,
        exact_context_matches: int,
    ) -> dict:
        """
        Compute extraction, classification, exact-match, and latency metrics.
        """
        extraction_precision = (
            total_matched / total_predicted
            if total_predicted
            else 0.0
        )

        extraction_recall = (
            total_matched / total_expected
            if total_expected
            else 0.0
        )

        extraction_f1 = (
            2
            * extraction_precision
            * extraction_recall
            / (
                extraction_precision
                + extraction_recall
            )
            if (
                extraction_precision
                + extraction_recall
            )
            else 0.0
        )

        target_metrics = {}

        for target in self.TARGETS:
            y_true = matched_truth[target]
            y_pred = matched_pred[target]

            target_metrics[target] = {
                "accuracy": accuracy_score(
                    y_true,
                    y_pred,
                ),
                "macro_f1": f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "classification_report": classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    zero_division=0,
                ),
            }

        exact_match_accuracy = (
            exact_context_matches
            / total_matched
            if total_matched
            else 0.0
        )

        latency_array = np.asarray(
            self.latencies,
            dtype=float,
        )

        latency_metrics = {
            "mean_seconds_per_report": float(
                latency_array.mean()
            ),
            "median_seconds_per_report": float(
                np.median(latency_array)
            ),
            "p95_seconds_per_report": float(
                np.percentile(
                    latency_array,
                    95,
                )
            ),
        }

        return {
            "fact_extraction": {
                "expected": total_expected,
                "predicted": total_predicted,
                "matched": total_matched,
                "precision": extraction_precision,
                "recall": extraction_recall,
                "f1": extraction_f1,
            },
            "context_classification": target_metrics,
            "exact_context_match_accuracy": exact_match_accuracy,
            "latency": latency_metrics,
        }
