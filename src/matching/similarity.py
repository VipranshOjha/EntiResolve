"""
similarity.py — Fast string, token, character, and numeric similarity functions.

Leverages rapidfuzz (C++ accelerated) when available for high-throughput string metrics
with clean pure-Python fallbacks.
"""

from __future__ import annotations

from typing import Sequence

try:
    from rapidfuzz import distance
    from rapidfuzz.distance import Levenshtein, JaroWinkler
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


def levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Normalized Levenshtein similarity in [0.0, 1.0].
    ratio = 1.0 - (distance / max(len(s1), len(s2)))
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    if _HAS_RAPIDFUZZ:
        return Levenshtein.normalized_similarity(s1, s2)

    # Pure Python fallback
    len1, len2 = len(s1), len(s2)
    dp = list(range(len2 + 1))
    for i, c1 in enumerate(s1):
        new_dp = [i + 1] * (len2 + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            new_dp[j + 1] = min(dp[j + 1] + 1, new_dp[j] + 1, dp[j] + cost)
        dp = new_dp
    dist = dp[len2]
    return 1.0 - (dist / max(len1, len2))


def jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """
    Jaro-Winkler string similarity in [0.0, 1.0].
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    if _HAS_RAPIDFUZZ:
        return JaroWinkler.similarity(s1, s2, prefix_weight=prefix_weight)

    # Basic Jaro fallback
    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
    transpositions //= 2

    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0
    l = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            l += 1
        else:
            break
    return jaro + l * prefix_weight * (1.0 - jaro)


def token_jaccard(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """
    Jaccard similarity between two token sequences:
    |A ∩ B| / |A ∪ B|
    """
    set1 = set(tokens1)
    set2 = set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def token_overlap_coefficient(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """
    Overlap coefficient between two token sequences:
    |A ∩ B| / min(|A|, |B|)
    """
    set1 = set(tokens1)
    set2 = set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    min_size = min(len(set1), len(set2))
    return intersection / min_size if min_size > 0 else 0.0


def token_containment(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """
    1.0 if one token set is a complete subset of the other, else 0.0.
    """
    set1 = set(tokens1)
    set2 = set(tokens2)
    if not set1 or not set2:
        return 0.0
    if set1.issubset(set2) or set2.issubset(set1):
        return 1.0
    return 0.0


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """
    Character n-gram Jaccard similarity on strings.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0

    grams1 = {s1[i : i + n] for i in range(len(s1) - n + 1)}
    grams2 = {s2[i : i + n] for i in range(len(s2) - n + 1)}

    intersection = len(grams1.intersection(grams2))
    union = len(grams1.union(grams2))
    return intersection / union if union > 0 else 0.0


def longest_common_prefix_length(s1: str, s2: str) -> int:
    """
    Length of identical leading character prefix.
    """
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len


def numeric_overlap(nums1: Sequence[str], nums2: Sequence[str]) -> tuple[float, int]:
    """
    Returns (jaccard_similarity, overlap_count) for two numeric token sequences.
    """
    set1 = set(nums1)
    set2 = set(nums2)
    if not set1 and not set2:
        return (0.0, 0)
    overlap = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard = overlap / union if union > 0 else 0.0
    return (jaccard, overlap)


def house_number_agreement(h1: Sequence[str], h2: Sequence[str]) -> float:
    """
    1.0 if any house number candidate matches, else 0.0.
    """
    if not h1 or not h2:
        return 0.0
    set1 = set(h1)
    set2 = set(h2)
    if set1.intersection(set2):
        return 1.0
    return 0.0


def postal_code_agreement(p1: Sequence[str], p2: Sequence[str]) -> float:
    """
    1.0 if any postal code candidate matches, else 0.0.
    """
    if not p1 or not p2:
        return 0.0
    set1 = set(p1)
    set2 = set(p2)
    if set1.intersection(set2):
        return 1.0
    return 0.0
