from pathlib import Path

from src.candidate_finder import CandidateFinder
from src.context_window import ContextWindowExtractor


ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = ROOT / "data" / "vocabulary" / "oncology_vocabulary.yaml"

finder = CandidateFinder(VOCAB_PATH)
extractor = ContextWindowExtractor()

report = (
    "TP53 p.R175H was identified at diagnosis. "
    "However, repeat sequencing later failed to confirm the alteration. "
    "PD-L1 TPS was 60%."
)

candidates = finder.find(report)

for candidate in candidates:
    context = extractor.extract(report, candidate)

    print("\nENTITY:", candidate["entity_text"])
    print("CONTEXT:", context["context_text"])
    print(
        "ENTITY OFFSETS IN CONTEXT:",
        context["entity_start_in_context"],
        context["entity_end_in_context"],
    )
    print("NUMBER OF SENTENCES:", context["num_sentences"])
