from pathlib import Path

from src.candidate_finder import CandidateFinder


ROOT = Path(__file__).resolve().parents[1]

finder = CandidateFinder(
    ROOT / "data" / "vocabulary" / "oncology_vocabulary.yaml"
)

report = (
    "Molecular testing confirmed EGFR L858R. "
    "No ALK rearrangement was detected. "
    "PD-L1 TPS was 65%. "
    "The patient started osimertinib."
)

candidates = finder.find(report)

for candidate in candidates:
    print(candidate)
