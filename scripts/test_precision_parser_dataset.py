from pathlib import Path

import pandas as pd

from src.candidate_finder import CandidateFinder
from src.precision_parser import PrecisionParser


ROOT = Path(__file__).resolve().parents[1]

REPORTS_PATH = ROOT / "data" / "synthetic" / "reports.csv"
VOCAB_PATH = ROOT / "data" / "vocabulary" / "oncology_vocabulary.yaml"


def main():
    reports = pd.read_csv(REPORTS_PATH)

    finder = CandidateFinder(VOCAB_PATH)
    parser = PrecisionParser(VOCAB_PATH)

    # Probamos sólo unos pocos reports.
    sample_reports = reports.head(5)

    for row in sample_reports.itertuples():
        print("\n" + "=" * 80)
        print(f"Report: {row.report_id}")
        print(f"Disease: {row.disease}")
        print(row.report_text)

        candidates = finder.find(row.report_text)

        print("\nParsed entities:")

        for candidate in candidates:
            parsed = parser.parse(
                candidate,
                source_text=row.report_text,
            )
            print(parsed)


if __name__ == "__main__":
    main()