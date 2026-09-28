"""
src/matching — Pairwise feature engineering, dataset construction, baseline matching,
supervised modeling, probability calibration, and entity-level inference.
"""

from src.matching.similarity import (
    levenshtein_ratio,
    jaro_winkler_similarity,
    token_jaccard,
    token_overlap_coefficient,
    token_containment,
    char_ngram_jaccard,
    numeric_overlap,
    house_number_agreement,
    postal_code_agreement,
)

__all__ = [
    "levenshtein_ratio",
    "jaro_winkler_similarity",
    "token_jaccard",
    "token_overlap_coefficient",
    "token_containment",
    "char_ngram_jaccard",
    "numeric_overlap",
    "house_number_agreement",
    "postal_code_agreement",
]
