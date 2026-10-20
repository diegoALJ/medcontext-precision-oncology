from pathlib import Path

from src.candidate_finder import CandidateFinder
from src.context_window import ContextWindowExtractor
from src.embeddings import FrozenBioEmbedder


ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = ROOT / "data" / "vocabulary" / "oncology_vocabulary.yaml"

finder = CandidateFinder(VOCAB_PATH)
window_extractor = ContextWindowExtractor()
embedder = FrozenBioEmbedder()

report = (
    "IDH1 p.R132H cannot be excluded. "
    "PML::RARA was confirmed by molecular testing."
)

candidates = finder.find(report)

contexts = []

for candidate in candidates:
    context = window_extractor.extract(report, candidate)
    contexts.append(context["context_text"])

embeddings = embedder.encode(contexts)

print("MODEL:", embedder.model_name)
print("EMBEDDING DIMENSION:", embedder.embedding_dimension)
print("NUMBER OF CONTEXTS:", len(contexts))
print("EMBEDDING MATRIX SHAPE:", embeddings.shape)

for context_text, embedding in zip(contexts, embeddings):
    print("\nCONTEXT:", context_text)
    print("FIRST 10 VALUES:", embedding[:10])
