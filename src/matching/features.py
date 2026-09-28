"""
features.py — Pairwise Feature Engineering Engine for Entity Resolution.

Extracts multi-view name, address, cross-field interaction, candidate provenance,
and cross-script features for (S1, Candidate) pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from src.matching.cross_script import CrossScriptAligner
from src.matching.similarity import (
    char_ngram_jaccard,
    house_number_agreement,
    jaro_winkler_similarity,
    levenshtein_ratio,
    numeric_overlap,
    postal_code_agreement,
    token_containment,
    token_jaccard,
    token_overlap_coefficient,
)
from src.normalization.record import BusinessRecord


@dataclass(slots=True)
class CandidateProvenance:
    """Provenance tracking how a candidate was retrieved."""
    candidate_rank: int = 1
    composite_priority: float = 1.0
    blocker_count: int = 1
    exact_name_hit: int = 0
    compact_name_hit: int = 0
    name_token_hit: int = 0
    ngram_hit: int = 0
    address_token_hit: int = 0
    structural_hit: int = 0
    source_is_s2: int = 1  # 1 for S2, 0 for S3


# Feature group definitions for ablations and interpretability
NAME_FEATURE_NAMES: list[str] = [
    "name_exact_normalized",
    "name_exact_suffix_normalized",
    "name_compact_equal",
    "name_compact_no_tld_equal",
    "name_levenshtein",
    "name_jaro_winkler",
    "name_char_ngram_jaccard",
    "name_token_jaccard",
    "name_token_overlap_coef",
    "name_token_containment",
    "name_token_count_diff",
    "name_char_len_ratio",
    "name_compact_len_ratio",
    "name_quality_both_good",
    "name_quality_agreement",
    "name_script_agreement",
    "name_s1_has_latin",
    "name_target_has_latin",
    "name_s1_has_non_latin",
    "name_target_has_non_latin",
    "name_both_latin",
    "name_cross_script",
    "name_s1_mixed_script",
    "name_target_mixed_script",
    "name_s1_is_website",
    "name_target_is_website",
    "name_both_websites",
]

ADDRESS_FEATURE_NAMES: list[str] = [
    "addr_levenshtein",
    "addr_token_jaccard",
    "addr_token_overlap_coef",
    "addr_char_ngram_jaccard",
    "addr_numeric_jaccard",
    "addr_numeric_overlap_count",
    "addr_house_number_agreement",
    "addr_postal_code_agreement",
    "addr_locality_token_overlap",
    "addr_component_count_diff",
    "addr_char_len_ratio",
    "addr_s1_empty",
    "addr_target_empty",
    "addr_both_empty",
    "addr_s1_po_box",
    "addr_target_po_box",
    "addr_both_po_box",
]

INTERACTION_FEATURE_NAMES: list[str] = [
    "inter_name_addr_sim",
    "inter_name_quality_addr_quality",
    "inter_script_addr_overlap",
    "inter_house_number_name_sim",
    "inter_postal_name_sim",
    "inter_exact_name_numeric_agree",
]

PROVENANCE_FEATURE_NAMES: list[str] = [
    "prov_exact_name_hit",
    "prov_compact_name_hit",
    "prov_name_token_hit",
    "prov_ngram_hit",
    "prov_address_token_hit",
    "prov_structural_hit",
    "prov_blocker_count",
    "prov_composite_priority",
    "prov_candidate_rank",
    "prov_source_is_s2",
]

CROSS_SCRIPT_FEATURE_NAMES: list[str] = [
    "cross_script_learned_sim",
]

ALL_FEATURE_NAMES: list[str] = (
    NAME_FEATURE_NAMES
    + ADDRESS_FEATURE_NAMES
    + INTERACTION_FEATURE_NAMES
    + PROVENANCE_FEATURE_NAMES
    + CROSS_SCRIPT_FEATURE_NAMES
)


class PairwiseFeatureExtractor:
    """
    Extracts high-dimensional pairwise feature vectors between an S1 entity
    and a candidate target entity using existing Phase 2 representations.
    """

    def __init__(
        self,
        cross_script_aligner: CrossScriptAligner | None = None,
        include_cross_script: bool = True,
    ) -> None:
        self.cross_script_aligner = cross_script_aligner
        self.include_cross_script = include_cross_script

    @property
    def feature_names(self) -> list[str]:
        if self.include_cross_script:
            return ALL_FEATURE_NAMES
        return (
            NAME_FEATURE_NAMES
            + ADDRESS_FEATURE_NAMES
            + INTERACTION_FEATURE_NAMES
            + PROVENANCE_FEATURE_NAMES
        )

    def extract_features(
        self,
        s1: BusinessRecord,
        target: BusinessRecord,
        provenance: CandidateProvenance | None = None,
    ) -> np.ndarray:
        """
        Extract numerical feature vector for a pair.

        Returns
        -------
        np.ndarray of shape (num_features,), dtype=np.float32
        """
        if provenance is None:
            source_is_s2 = 1 if target.source == "s2" or target.entity_id.startswith("S2") else 0
            provenance = CandidateProvenance(source_is_s2=source_is_s2)

        s1_n = s1.name
        tgt_n = target.name
        s1_a = s1.address
        tgt_a = target.address

        # -------------------------------------------------------------
        # 1. NAME FEATURES
        # -------------------------------------------------------------
        name_exact_norm = 1.0 if s1_n.normalized and s1_n.normalized == tgt_n.normalized else 0.0
        s1_suffix_str = " ".join(s1_n.tokens_suffix_norm)
        tgt_suffix_str = " ".join(tgt_n.tokens_suffix_norm)
        name_exact_suffix = 1.0 if s1_suffix_str and s1_suffix_str == tgt_suffix_str else 0.0
        name_compact_eq = 1.0 if s1_n.compact and s1_n.compact == tgt_n.compact else 0.0
        name_compact_no_tld_eq = 1.0 if s1_n.compact_no_tld and s1_n.compact_no_tld == tgt_n.compact_no_tld else 0.0

        name_lev = levenshtein_ratio(s1_n.normalized, tgt_n.normalized)
        name_jw = jaro_winkler_similarity(s1_n.normalized, tgt_n.normalized)
        name_ngram = char_ngram_jaccard(s1_n.normalized, tgt_n.normalized, n=3)

        name_tok_jaccard = token_jaccard(s1_n.tokens, tgt_n.tokens)
        name_tok_overlap = token_overlap_coefficient(s1_n.tokens, tgt_n.tokens)
        name_tok_containment = token_containment(s1_n.tokens, tgt_n.tokens)
        name_tok_count_diff = float(abs(len(s1_n.tokens) - len(tgt_n.tokens)))

        len1, len2 = len(s1_n.normalized), len(tgt_n.normalized)
        name_char_len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 1.0

        clen1, clen2 = len(s1_n.compact), len(tgt_n.compact)
        name_compact_len_ratio = (min(clen1, clen2) / max(clen1, clen2)) if max(clen1, clen2) > 0 else 1.0

        name_qual_both_good = 1.0 if s1_n.quality == "good" and tgt_n.quality == "good" else 0.0
        name_qual_agree = 1.0 if s1_n.quality == tgt_n.quality else 0.0
        name_script_agree = 1.0 if s1_n.script.dominant_script == tgt_n.script.dominant_script else 0.0

        s1_lat = 1.0 if s1_n.script.latin_ratio > 0.0 or "LATIN" in s1_n.script.scripts_present else 0.0
        tgt_lat = 1.0 if tgt_n.script.latin_ratio > 0.0 or "LATIN" in tgt_n.script.scripts_present else 0.0
        s1_non_lat = 1.0 if s1_n.script.is_non_latin or s1_n.script.is_mixed_script else 0.0
        tgt_non_lat = 1.0 if tgt_n.script.is_non_latin or tgt_n.script.is_mixed_script else 0.0
        both_latin = 1.0 if (s1_lat and tgt_lat and not (s1_non_lat or tgt_non_lat)) else 0.0
        cross_script = 1.0 if ((s1_lat and tgt_non_lat) or (s1_non_lat and tgt_lat)) else 0.0

        s1_mixed = 1.0 if s1_n.script.is_mixed_script else 0.0
        tgt_mixed = 1.0 if tgt_n.script.is_mixed_script else 0.0
        s1_web = 1.0 if s1_n.quality == "website" else 0.0
        tgt_web = 1.0 if tgt_n.quality == "website" else 0.0
        both_web = 1.0 if (s1_web and tgt_web) else 0.0

        # -------------------------------------------------------------
        # 2. ADDRESS FEATURES
        # -------------------------------------------------------------
        addr_lev = levenshtein_ratio(s1_a.normalized, tgt_a.normalized)
        addr_tok_jaccard = token_jaccard(s1_a.tokens, tgt_a.tokens)
        addr_tok_overlap = token_overlap_coefficient(s1_a.tokens, tgt_a.tokens)
        addr_ngram = char_ngram_jaccard(s1_a.normalized, tgt_a.normalized, n=3)

        num_jaccard, num_overlap_cnt = numeric_overlap(s1_a.numeric_tokens, tgt_a.numeric_tokens)
        house_agree = house_number_agreement(s1_a.house_number_candidates, tgt_a.house_number_candidates)
        postal_agree = postal_code_agreement(s1_a.postal_candidates, tgt_a.postal_candidates)

        # Locality tokens (tokens excluding pure numeric tokens)
        s1_locality = [t for t in s1_a.tokens if not t.isdigit()]
        tgt_locality = [t for t in tgt_a.tokens if not t.isdigit()]
        addr_locality_overlap = token_overlap_coefficient(s1_locality, tgt_locality)

        addr_comp_diff = float(abs(len(s1_a.tokens) - len(tgt_a.tokens)))
        alen1, alen2 = len(s1_a.normalized), len(tgt_a.normalized)
        addr_len_ratio = (min(alen1, alen2) / max(alen1, alen2)) if max(alen1, alen2) > 0 else 1.0

        addr_s1_empty = 1.0 if len(s1_a.tokens) == 0 or s1_a.quality == "missing" else 0.0
        addr_tgt_empty = 1.0 if len(tgt_a.tokens) == 0 or tgt_a.quality == "missing" else 0.0
        addr_both_empty = 1.0 if (addr_s1_empty and addr_tgt_empty) else 0.0

        addr_s1_po = 1.0 if s1_a.has_po_box else 0.0
        addr_tgt_po = 1.0 if tgt_a.has_po_box else 0.0
        addr_both_po = 1.0 if (addr_s1_po and addr_tgt_po) else 0.0

        # -------------------------------------------------------------
        # 3. CROSS-FIELD INTERACTIONS
        # -------------------------------------------------------------
        inter_name_addr = name_jw * addr_tok_jaccard
        inter_qual = name_qual_both_good * (1.0 if not (addr_s1_empty or addr_tgt_empty) else 0.0)
        inter_script_addr = name_script_agree * addr_tok_jaccard
        inter_house_name = house_agree * name_jw
        inter_postal_name = postal_agree * name_jw
        inter_exact_name_numeric = name_exact_norm * (1.0 if num_overlap_cnt > 0 else 0.0)

        # -------------------------------------------------------------
        # 4. BLOCKER PROVENANCE FEATURES
        # -------------------------------------------------------------
        prov_exact = float(provenance.exact_name_hit)
        prov_compact = float(provenance.compact_name_hit)
        prov_name_tok = float(provenance.name_token_hit)
        prov_ngram = float(provenance.ngram_hit)
        prov_addr_tok = float(provenance.address_token_hit)
        prov_struct = float(provenance.structural_hit)
        prov_count = float(provenance.blocker_count)
        prov_priority = float(provenance.composite_priority)
        prov_rank = float(provenance.candidate_rank)
        prov_source = float(provenance.source_is_s2)

        # -------------------------------------------------------------
        # 5. CROSS-SCRIPT AUXILIARY FEATURE
        # -------------------------------------------------------------
        if self.include_cross_script:
            if self.cross_script_aligner is not None:
                cs_sim = self.cross_script_aligner.compute_similarity(s1, target)
            else:
                cs_sim = 0.0

            feats = [
                name_exact_norm,
                name_exact_suffix,
                name_compact_eq,
                name_compact_no_tld_eq,
                name_lev,
                name_jw,
                name_ngram,
                name_tok_jaccard,
                name_tok_overlap,
                name_tok_containment,
                name_tok_count_diff,
                name_char_len_ratio,
                name_compact_len_ratio,
                name_qual_both_good,
                name_qual_agree,
                name_script_agree,
                s1_lat,
                tgt_lat,
                s1_non_lat,
                tgt_non_lat,
                both_latin,
                cross_script,
                s1_mixed,
                tgt_mixed,
                s1_web,
                tgt_web,
                both_web,
                # Address
                addr_lev,
                addr_tok_jaccard,
                addr_tok_overlap,
                addr_ngram,
                num_jaccard,
                float(num_overlap_cnt),
                house_agree,
                postal_agree,
                addr_locality_overlap,
                addr_comp_diff,
                addr_len_ratio,
                addr_s1_empty,
                addr_tgt_empty,
                addr_both_empty,
                addr_s1_po,
                addr_tgt_po,
                addr_both_po,
                # Interaction
                inter_name_addr,
                inter_qual,
                inter_script_addr,
                inter_house_name,
                inter_postal_name,
                inter_exact_name_numeric,
                # Provenance
                prov_exact,
                prov_compact,
                prov_name_tok,
                prov_ngram,
                prov_addr_tok,
                prov_struct,
                prov_count,
                prov_priority,
                prov_rank,
                prov_source,
                # Cross-script
                cs_sim,
            ]
        else:
            feats = [
                name_exact_norm,
                name_exact_suffix,
                name_compact_eq,
                name_compact_no_tld_eq,
                name_lev,
                name_jw,
                name_ngram,
                name_tok_jaccard,
                name_tok_overlap,
                name_tok_containment,
                name_tok_count_diff,
                name_char_len_ratio,
                name_compact_len_ratio,
                name_qual_both_good,
                name_qual_agree,
                name_script_agree,
                s1_lat,
                tgt_lat,
                s1_non_lat,
                tgt_non_lat,
                both_latin,
                cross_script,
                s1_mixed,
                tgt_mixed,
                s1_web,
                tgt_web,
                both_web,
                # Address
                addr_lev,
                addr_tok_jaccard,
                addr_tok_overlap,
                addr_ngram,
                num_jaccard,
                float(num_overlap_cnt),
                house_agree,
                postal_agree,
                addr_locality_overlap,
                addr_comp_diff,
                addr_len_ratio,
                addr_s1_empty,
                addr_tgt_empty,
                addr_both_empty,
                addr_s1_po,
                addr_tgt_po,
                addr_both_po,
                # Interaction
                inter_name_addr,
                inter_qual,
                inter_script_addr,
                inter_house_name,
                inter_postal_name,
                inter_exact_name_numeric,
                # Provenance
                prov_exact,
                prov_compact,
                prov_name_tok,
                prov_ngram,
                prov_addr_tok,
                prov_struct,
                prov_count,
                prov_priority,
                prov_rank,
                prov_source,
            ]

        return np.array(feats, dtype=np.float32)

    def extract_batch(
        self,
        pairs: Sequence[tuple[BusinessRecord, BusinessRecord, CandidateProvenance | None]],
    ) -> np.ndarray:
        """
        Vectorized feature extraction for a batch of pairs.

        Returns
        -------
        np.ndarray of shape (N, num_features), dtype=np.float32
        """
        N = len(pairs)
        num_features = len(self.feature_names)
        matrix = np.empty((N, num_features), dtype=np.float32)
        for i, (s1, target, prov) in enumerate(pairs):
            matrix[i] = self.extract_features(s1, target, prov)
        return matrix
