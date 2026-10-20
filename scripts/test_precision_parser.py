from pathlib import Path

from src.candidate_finder import CandidateFinder
from src.precision_parser import PrecisionParser


ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = ROOT / "data" / "vocabulary" / "oncology_vocabulary.yaml"

finder = CandidateFinder(VOCAB_PATH)
parser = PrecisionParser(VOCAB_PATH)

report = (
    "Molecular testing confirmed EGFR L858R. "
    "No PML-RARA fusion was detected. "
    "PD-L1 TPS was 65%. "
    "The patient started osimertinib."
)

candidates = finder.find(report)

for candidate in candidates:
    parsed = parser.parse(candidate, source_text=report)
    print(parsed)
