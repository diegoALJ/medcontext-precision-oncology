from pathlib import Path

from src.candidate_finder import CandidateFinder
from src.context_window import ContextWindowExtractor
from src.rules import ClinicalRuleExtractor
from src.embeddings import FrozenBioEmbedder
from src.hybrid_features import HybridFeatureBuilder


ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = ROOT / "data" / "vocabulary" / "oncology_vocabulary.yaml"

finder = CandidateFinder(VOCAB_PATH)
window_extractor = ContextWindowExtractor()
rule_extractor = ClinicalRuleExtractor()
embedder = FrozenBioEmbedder()
feature_builder = HybridFeatureBuilder()

report = (
    "IDH1 p.R132H cannot be excluded. "
    "PML::RARA was confirmed by molecular testing."
)

candidates = finder.find(report)

contexts = []
rule_features_list = []

for candidate in candidates:
    context = window_extractor.extract(
        report,
        candidate,
    )

    ruled = rule_extractor.extract(context)

    contexts.append(
        ruled["context_text"]
    )

    rule_features_list.append(
        ruled["rule_features"]
    )

embeddings = embedder.encode(contexts)

X = feature_builder.build_batch(
    rule_features_list,
    embeddings,
)

feature_names = feature_builder.feature_names(
    embedder.embedding_dimension
)

print("NUMBER OF SAMPLES:", X.shape[0])
print("SYMBOLIC DIMENSION:", feature_builder.symbolic_dimension)
print("EMBEDDING DIMENSION:", embedder.embedding_dimension)
print("HYBRID MATRIX SHAPE:", X.shape)
print("TOTAL FEATURE NAMES:", len(feature_names))

for i, candidate in enumerate(candidates):
    print("\nENTITY:", candidate["entity_text"])
    print("CONTEXT:", contexts[i])
    print(
        "SYMBOLIC VECTOR:",
        X[i, :feature_builder.symbolic_dimension],
    )
    print(
        "FIRST 10 EMBEDDING VALUES:",
        X[
            i,
            feature_builder.symbolic_dimension:
            feature_builder.symbolic_dimension + 10
        ],
    )
