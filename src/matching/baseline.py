"""
baseline.py — Deterministic Rule-Based Pairwise Baseline Matcher.

Establishes the transparent, non-ML baseline performance before introducing
supervised models.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from src.matching.features import ALL_FEATURE_NAMES, PairwiseFeatureExtractor
from src.normalization.record import BusinessRecord


class DeterministicBaselineMatcher:
    """
    Rule-based deterministic matcher combining strongest string and address similarity signals.
    """

    def __init__(
        self,
        name_exact_threshold: float = 1.0,
        high_name_sim_threshold: float = 0.92,
        combined_name_threshold: float = 0.82,
        combined_addr_threshold: float = 0.25,
    ) -> None:
        self.name_exact_threshold = name_exact_threshold
        self.high_name_sim_threshold = high_name_sim_threshold
        self.combined_name_threshold = combined_name_threshold
        self.combined_addr_threshold = combined_addr_threshold

    def score_pair(
        self,
        s1: BusinessRecord,
        target: BusinessRecord,
    ) -> float:
        """
        Compute deterministic rule-based score in [0.0, 1.0].
        """
        from src.matching.similarity import (
            house_number_agreement,
            jaro_winkler_similarity,
            postal_code_agreement,
            token_jaccard,
        )

        # 1. Exact normalized name match
        if s1.name.normalized and s1.name.normalized == target.name.normalized:
            # If both have addresses, check if they don't strongly contradict
            if s1.address.tokens and target.address.tokens:
                addr_sim = token_jaccard(s1.address.tokens, target.address.tokens)
                if addr_sim >= 0.15 or house_number_agreement(s1.address.house_number_candidates, target.address.house_number_candidates) > 0:
                    return 0.98
                return 0.88
            return 0.92

        # 2. Compact name match (handles spacing / punctuation differences)
        if s1.name.compact and s1.name.compact == target.name.compact:
            return 0.85

        name_jw = jaro_winkler_similarity(s1.name.normalized, target.name.normalized)

        # 3. Very high name similarity
        if name_jw >= self.high_name_sim_threshold:
            # Check address compatibility
            if s1.address.tokens and target.address.tokens:
                addr_sim = token_jaccard(s1.address.tokens, target.address.tokens)
                if addr_sim >= 0.20:
                    return 0.90
                return 0.75
            return 0.80

        # 4. Moderate name similarity + strong address evidence
        if name_jw >= self.combined_name_threshold:
            addr_sim = token_jaccard(s1.address.tokens, target.address.tokens)
            house_match = house_number_agreement(s1.address.house_number_candidates, target.address.house_number_candidates)
            postal_match = postal_code_agreement(s1.address.postal_candidates, target.address.postal_candidates)

            if house_match > 0 and postal_match > 0:
                return 0.85
            if addr_sim >= self.combined_addr_threshold or house_match > 0 or postal_match > 0:
                return 0.70
            return 0.40

        # 5. Low name similarity: allow only if exact house number + exact postal + moderate name
        if name_jw >= 0.70:
            house_match = house_number_agreement(s1.address.house_number_candidates, target.address.house_number_candidates)
            postal_match = postal_code_agreement(s1.address.postal_candidates, target.address.postal_candidates)
            if house_match > 0 and postal_match > 0:
                return 0.65

        return float(name_jw * 0.4)

    def score_feature_matrix(self, X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
        """
        Compute baseline scores directly from feature matrix.
        """
        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        name_exact = X[:, name_to_idx["name_exact_normalized"]]
        name_compact = X[:, name_to_idx["name_compact_equal"]]
        name_jw = X[:, name_to_idx["name_jaro_winkler"]]
        addr_jaccard = X[:, name_to_idx["addr_token_jaccard"]]
        house_agree = X[:, name_to_idx["addr_house_number_agreement"]]
        postal_agree = X[:, name_to_idx["addr_postal_code_agreement"]]

        scores = np.zeros(X.shape[0], dtype=np.float32)

        for i in range(X.shape[0]):
            if name_exact[i] > 0.5:
                if addr_jaccard[i] >= 0.15 or house_agree[i] > 0.5:
                    scores[i] = 0.98
                else:
                    scores[i] = 0.88
            elif name_compact[i] > 0.5:
                scores[i] = 0.85
            elif name_jw[i] >= self.high_name_sim_threshold:
                if addr_jaccard[i] >= 0.20:
                    scores[i] = 0.90
                else:
                    scores[i] = 0.75
            elif name_jw[i] >= self.combined_name_threshold:
                if house_agree[i] > 0.5 and postal_agree[i] > 0.5:
                    scores[i] = 0.85
                elif addr_jaccard[i] >= self.combined_addr_threshold or house_agree[i] > 0.5 or postal_agree[i] > 0.5:
                    scores[i] = 0.70
                else:
                    scores[i] = 0.40
            elif name_jw[i] >= 0.70 and house_agree[i] > 0.5 and postal_agree[i] > 0.5:
                scores[i] = 0.65
            else:
                scores[i] = float(name_jw[i] * 0.4)

        return scores
