"""
evaluator.py — Evaluation metrics for candidate generation and blocking.

Computes:
- Overall candidate recall = (true matches retrieved) / (total true matches)
- S1 -> S2 candidate recall
- S1 -> S3 candidate recall
- Country-specific recall (US, India, etc.)
- Match-count bucket recall (1, 2-3, 4-5, 6+)
- Non-Latin candidate recall
- Corrupted-name candidate recall
- Candidate volume statistics: mean, median, p95, p99, max candidates/S1
- Reduction ratio: 1.0 - (total candidates / Cartesian product)
- Blocker-level contribution and incremental union recall
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.normalization.record import BusinessRecord


@dataclass
class BlockingMetrics:
    """Detailed candidate recall and volume metrics."""

    total_queries: int = 0
    total_true_matches: int = 0
    true_matches_retrieved: int = 0
    overall_recall: float = 0.0

    # Source breakdown
    s2_true_matches: int = 0
    s2_retrieved: int = 0
    s2_recall: float = 0.0

    s3_true_matches: int = 0
    s3_retrieved: int = 0
    s3_recall: float = 0.0

    # Category breakdowns (recall percentages)
    country_recall: dict[str, float] = field(default_factory=dict)
    bucket_recall: dict[str, float] = field(default_factory=dict)
    non_latin_recall: float = 0.0
    corrupted_recall: float = 0.0

    # Volume statistics
    total_candidates_generated: int = 0
    mean_candidates_per_query: float = 0.0
    median_candidates_per_query: float = 0.0
    p95_candidates_per_query: float = 0.0
    p99_candidates_per_query: float = 0.0
    max_candidates_per_query: int = 0
    reduction_ratio: float = 0.0

    # Blocker attribution recall (individual blocker recalls)
    individual_blocker_recalls: dict[str, float] = field(default_factory=dict)


class BlockingEvaluator:
    """
    Evaluates candidate generation performance against ground truth.
    """

    def evaluate(
        self,
        query_records: list[BusinessRecord],
        retrieved_candidates: list[list[str]],
        ground_truth: dict[str, set[str]],
        total_target_pool_size: int,
        attribution_list: list[dict[str, list[str]]] | None = None,
    ) -> BlockingMetrics:
        """
        Evaluate candidate recall and volume statistics.

        Parameters
        ----------
        query_records:
            List of S1 BusinessRecord queries.
        retrieved_candidates:
            List of retrieved candidate entity_id lists for each query.
        ground_truth:
            Dict mapping s1_id -> set of true matching entity IDs.
        total_target_pool_size:
            Total number of target records in the indexed search space (for reduction ratio).
        attribution_list:
            Optional list of dicts mapping blocker_name -> candidate list per query.
        """
        n_queries = len(query_records)
        if n_queries == 0:
            return BlockingMetrics()

        total_true = 0
        total_retrieved = 0

        s2_true = 0
        s2_ret = 0
        s3_true = 0
        s3_ret = 0

        # Sub-cohort tracking
        country_true: dict[str, int] = {}
        country_ret: dict[str, int] = {}

        bucket_true: dict[str, int] = {"1": 0, "2-3": 0, "4-5": 0, "6+": 0}
        bucket_ret: dict[str, int] = {"1": 0, "2-3": 0, "4-5": 0, "6+": 0}

        non_latin_true = 0
        non_latin_ret = 0

        corrupted_true = 0
        corrupted_ret = 0

        candidate_counts: list[int] = []

        # Attribution tracking
        blocker_ret_counts: dict[str, int] = {}
        if attribution_list:
            for b_name in attribution_list[0].keys():
                blocker_ret_counts[b_name] = 0

        for i, q in enumerate(query_records):
            s1_id = q.entity_id
            candidates = set(retrieved_candidates[i])
            candidate_counts.append(len(candidates))

            true_matches = ground_truth.get(s1_id, set())
            n_true = len(true_matches)
            if n_true == 0:
                continue  # Singleton: no true matches to recall

            total_true += n_true
            retrieved_matches = candidates.intersection(true_matches)
            total_retrieved += len(retrieved_matches)

            # Attribution
            if attribution_list:
                attr = attribution_list[i]
                for b_name, b_cands in attr.items():
                    b_set = set(b_cands)
                    blocker_ret_counts[b_name] += len(b_set.intersection(true_matches))

            # Source breakdown
            for m in true_matches:
                is_ret = m in candidates
                if m.startswith("S2-") or m.startswith("s2-"):
                    s2_true += 1
                    if is_ret:
                        s2_ret += 1
                elif m.startswith("S3-") or m.startswith("s3-"):
                    s3_true += 1
                    if is_ret:
                        s3_ret += 1

            # Country
            ctry = q.country or "UNKNOWN"
            country_true[ctry] = country_true.get(ctry, 0) + n_true
            country_ret[ctry] = country_ret.get(ctry, 0) + len(retrieved_matches)

            # Bucket
            if n_true == 1:
                b_key = "1"
            elif 2 <= n_true <= 3:
                b_key = "2-3"
            elif 4 <= n_true <= 5:
                b_key = "4-5"
            else:
                b_key = "6+"
            bucket_true[b_key] += n_true
            bucket_ret[b_key] += len(retrieved_matches)

            # Non-Latin
            if q.name.script.is_non_latin or q.name.script.is_mixed_script:
                non_latin_true += n_true
                non_latin_ret += len(retrieved_matches)

            # Corrupted
            if q.name.quality in ("CORRUPTED", "NOISE_PREFIX"):
                corrupted_true += n_true
                corrupted_ret += len(retrieved_matches)

        # Candidate count percentiles
        candidate_counts.sort()
        total_cands = sum(candidate_counts)
        mean_cands = total_cands / n_queries
        median_cands = candidate_counts[n_queries // 2]
        p95_cands = candidate_counts[min(int(math.ceil(0.95 * n_queries)) - 1, n_queries - 1)]
        p99_cands = candidate_counts[min(int(math.ceil(0.99 * n_queries)) - 1, n_queries - 1)]
        max_cands = candidate_counts[-1] if candidate_counts else 0

        # Reduction ratio
        cartesian_product = n_queries * max(1, total_target_pool_size)
        reduction_ratio = 1.0 - (total_cands / cartesian_product)

        # Recalls
        overall_recall = total_retrieved / total_true if total_true > 0 else 0.0
        s2_recall = s2_ret / s2_true if s2_true > 0 else 0.0
        s3_recall = s3_ret / s3_true if s3_true > 0 else 0.0

        country_recall = {
            c: country_ret[c] / country_true[c]
            for c in country_true if country_true[c] > 0
        }
        bucket_recall = {
            b: bucket_ret[b] / bucket_true[b]
            for b in bucket_true if bucket_true[b] > 0
        }

        non_latin_recall = non_latin_ret / non_latin_true if non_latin_true > 0 else 1.0
        corrupted_recall = corrupted_ret / corrupted_true if corrupted_true > 0 else 1.0

        blocker_recalls = {}
        if attribution_list and total_true > 0:
            for b_name, count in blocker_ret_counts.items():
                blocker_recalls[b_name] = count / total_true

        return BlockingMetrics(
            total_queries=n_queries,
            total_true_matches=total_true,
            true_matches_retrieved=total_retrieved,
            overall_recall=overall_recall,
            s2_true_matches=s2_true,
            s2_retrieved=s2_ret,
            s2_recall=s2_recall,
            s3_true_matches=s3_true,
            s3_retrieved=s3_ret,
            s3_recall=s3_recall,
            country_recall=country_recall,
            bucket_recall=bucket_recall,
            non_latin_recall=non_latin_recall,
            corrupted_recall=corrupted_recall,
            total_candidates_generated=total_cands,
            mean_candidates_per_query=mean_cands,
            median_candidates_per_query=float(median_cands),
            p95_candidates_per_query=float(p95_cands),
            p99_candidates_per_query=float(p99_cands),
            max_candidates_per_query=max_cands,
            reduction_ratio=reduction_ratio,
            individual_blocker_recalls=blocker_recalls,
        )
