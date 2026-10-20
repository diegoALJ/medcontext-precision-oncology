from pathlib import Path

from src.candidate_finder import CandidateFinder
from src.context_window import ContextWindowExtractor
from src.rules import ClinicalRuleExtractor


ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = ROOT / "data" / "vocabulary" / "oncology_vocabulary.yaml"

finder = CandidateFinder(VOCAB_PATH)
window_extractor = ContextWindowExtractor()
rule_extractor = ClinicalRuleExtractor()

reports = [
    "No ALK rearrangement was detected.",
    "IDH1 p.R132H cannot be excluded.",
    "The patient's father carried TP53 p.R175H.",
    (
        "FLT3 was previously detected. "
        "However, current testing is negative."
    ),
]

for report in reports:
    print("\n" + "=" * 80)
    print("REPORT:", report)

    candidates = finder.find(report)

    for candidate in candidates:
        context = window_extractor.extract(report, candidate)
        result = rule_extractor.extract(context)

        print("\nENTITY:", candidate["entity_text"])
        print("CONTEXT:", result["context_text"])
        print("FEATURES:", result["rule_features"])
        print("MATCHES:", result["rule_matches"])
