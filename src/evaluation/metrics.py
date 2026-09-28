"""
metrics.py — F_0.5 evaluation implementation for the Business Entity
Resolution challenge.

Evaluation specification (from README.md)
-----------------------------------------
Metric: F_β with β = 0.5 (precision-heavy).

For each Source 1 entity:

    predicted = set of predicted matched IDs (S2- and/or S3- IDs)
    truth     = set of ground-truth matched IDs

    tp        = |predicted ∩ truth|
    precision = tp / |predicted|    (1.0 when predicted is empty — vacuous)
    recall    = tp / |truth|        (1.0 when truth is empty — vacuous)

    F_0.5 = (1.25 × precision × recall) / (0.25 × precision + recall)

Special cases (from README.md, verified):
    - truth=∅, predicted=∅  →  F = 1.0  (correctly identified singleton)
    - truth=∅, predicted≠∅  →  F = 0.0  (false merge on singleton)
    - truth≠∅, predicted=∅  →  F = 0.0  (missed all matches; vacuous precision=1.0
                                          but recall=0.0 makes F=0.0)

Macro-average: F_0.5 is computed per entity, then averaged across **all**
Source 1 entities in the evaluation set (including singletons).

Design notes
------------
* The evaluator works entirely on Python sets/frozensets — no pandas dependency.
* It never materialises a pairwise comparison table; it operates entity-by-entity.
* Candidate-set metrics (recall ceiling, reduction ratio) are optional and only
  populated when a candidate mapping is provided. These will be fully exercised
  once the blocking stage is implemented (Phase 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional

# ---------------------------------------------------------------------------
# Per-entity computation
# ---------------------------------------------------------------------------

BETA: float = 0.5
_BETA2: float = BETA ** 2   # 0.25 — used in the F_beta denominator weight


def _entity_scores(
    predicted: frozenset[str],
    truth: frozenset[str],
) -> tuple[float, float, float]:
    """
    Compute (precision, recall, F_0.5) for a single Source 1 entity.

    Parameters
    ----------
    predicted:
        The set of S2-/S3- IDs returned by the matcher for this entity.
        Must already be de-duplicated (frozenset ensures this).
    truth:
        The ground-truth set of matching S2-/S3- IDs. Empty for singletons.

    Returns
    -------
    (precision, recall, f05) — all floats in [0.0, 1.0].
    """
    # Case 1: correct singleton detection
    if not predicted and not truth:
        return 1.0, 1.0, 1.0

    # Case 2: false merge on singleton (truth=∅ but we predicted something)
    if not truth:
        return 0.0, 0.0, 0.0

    # Case 3: all matches missed (predicted=∅ but truth is non-empty)
    if not predicted:
        # Vacuous precision = 1.0 (no false positives), recall = 0.0 → F = 0.0
        # Formula: (1.25 * 1.0 * 0.0) / (0.25 * 1.0 + 0.0) = 0 / 0.25 = 0.0
        return 1.0, 0.0, 0.0

    # Case 4: both non-empty — standard formula
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)

    denom = _BETA2 * precision + recall
    if denom == 0.0:
        f05 = 0.0
    else:
        f05 = (1.0 + _BETA2) * precision * recall / denom

    return precision, recall, f05


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Aggregated evaluation metrics for a set of Source 1 predictions.

    Primary metric
    --------------
    f05_macro:
        Macro-averaged F_0.5 across ALL evaluated S1 entities (the challenge
        leaderboard metric).

    Breakdown metrics
    -----------------
    f05_singleton / f05_non_singleton:
        Macro-averaged F_0.5 restricted to true-singleton / true-non-singleton
        entities respectively.
    f05_s2 / f05_s3:
        Macro-averaged F_0.5 computed using only S2- / S3- IDs in both
        predicted and truth sets. Useful for diagnosing source-specific issues.

    Count metrics
    -------------
    n_entities:     Total number of evaluated S1 entities.
    n_singletons:   Number with truth = ∅.
    n_non_singletons: Number with truth ≠ ∅.
    tp / fp / fn:   Aggregate true/false positives/negatives across all entities.

    Candidate-set metrics (Phase 3+)
    ---------------------------------
    candidate_recall:
        Fraction of true match IDs that appeared in the candidate set.
        None until the blocking stage is active.
    candidates_per_s1:
        Mean number of candidates generated per S1 entity.
        None until the blocking stage is active.
    candidate_reduction_ratio:
        1 − (total_candidates / (n_s1 × n_s2s3)).
        None until the blocking stage is active.
    """

    # Core macro metrics
    f05_macro: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0

    # Subset macro metrics
    f05_singleton: float = 0.0
    f05_non_singleton: float = 0.0
    precision_singleton: float = 0.0
    recall_singleton: float = 0.0
    precision_non_singleton: float = 0.0
    recall_non_singleton: float = 0.0

    # Source-specific macro metrics
    f05_s2: float = 0.0          # only S2- IDs
    f05_s3: float = 0.0          # only S3- IDs
    precision_s2: float = 0.0
    recall_s2: float = 0.0
    precision_s3: float = 0.0
    recall_s3: float = 0.0

    # Aggregate counts
    n_entities: int = 0
    n_singletons: int = 0
    n_non_singletons: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    # Candidate-set metrics — None until Phase 3
    candidate_recall: Optional[float] = None
    candidates_per_s1: Optional[float] = None
    candidate_reduction_ratio: Optional[float] = None

    # Optional: per-entity breakdown for error analysis
    per_entity: dict[str, dict] = field(default_factory=dict, repr=False)

    def summary(self, verbose: bool = False) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 60,
            "  EVALUATION RESULTS",
            "=" * 60,
            f"  Entities evaluated : {self.n_entities:>10,}",
            f"  Singletons         : {self.n_singletons:>10,}",
            f"  Non-singletons     : {self.n_non_singletons:>10,}",
            "-" * 60,
            f"  Macro F_0.5        : {self.f05_macro:>10.4f}   ← primary metric",
            f"  Macro Precision    : {self.precision_macro:>10.4f}",
            f"  Macro Recall       : {self.recall_macro:>10.4f}",
            "-" * 60,
            f"  F_0.5 (singleton)  : {self.f05_singleton:>10.4f}",
            f"  F_0.5 (non-single) : {self.f05_non_singleton:>10.4f}",
            "-" * 60,
            f"  F_0.5 (S2 only)    : {self.f05_s2:>10.4f}",
            f"  F_0.5 (S3 only)    : {self.f05_s3:>10.4f}",
            "-" * 60,
            f"  TP  : {self.tp:,}",
            f"  FP  : {self.fp:,}",
            f"  FN  : {self.fn:,}",
        ]

        # Candidate metrics (Phase 3+)
        if self.candidate_recall is not None:
            lines += [
                "-" * 60,
                f"  Candidate recall        : {self.candidate_recall:>8.4f}",
                f"  Candidates per S1       : {self.candidates_per_s1:>8.1f}",
                f"  Candidate reduction ratio: {self.candidate_reduction_ratio:>7.4f}",
            ]
        else:
            lines.append("  [Candidate metrics: N/A — blocking stage not yet active]")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EvaluationResult(f05_macro={self.f05_macro:.4f}, "
            f"precision={self.precision_macro:.4f}, "
            f"recall={self.recall_macro:.4f}, "
            f"n={self.n_entities})"
        )


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate(
    predictions: Mapping[str, frozenset[str]],
    ground_truth: Mapping[str, frozenset[str]],
    candidates: Optional[Mapping[str, frozenset[str]]] = None,
    store_per_entity: bool = False,
) -> EvaluationResult:
    """
    Evaluate a set of entity-resolution predictions against ground truth.

    This function is the central evaluation entry point used by every
    experiment. It reproduces the challenge's F_0.5 macro-average exactly.

    Parameters
    ----------
    predictions:
        ``{s1_id: frozenset_of_predicted_ids}`` — output of the matcher.
        Every S1 entity in ``ground_truth`` should have an entry; missing
        entries are treated as empty predictions (= missed all matches).
    ground_truth:
        ``{s1_id: frozenset_of_matched_ids}`` — the held-out labels.
        Singleton entities have an empty frozenset as value.
    candidates:
        Optional ``{s1_id: frozenset_of_candidate_ids}`` from the blocking
        stage. When provided, candidate-set metrics are computed. Every
        final prediction *should* be a subset of the candidate set for that
        entity; violations are counted (as in the challenge validator) but
        do not raise errors.
    store_per_entity:
        When True, the ``per_entity`` dict is populated with per-entity
        precision/recall/F_0.5 for downstream error analysis. Use only
        on validation-sized data (not on the full test set).

    Returns
    -------
    EvaluationResult
        Fully populated result object.

    Notes
    -----
    * The function iterates over ``ground_truth`` as the authoritative list of
      entities to evaluate. Extra entries in ``predictions`` that have no
      ground-truth counterpart are ignored.
    * We use frozensets throughout so callers cannot accidentally mutate the
      sets mid-evaluation.
    """
    result = EvaluationResult()

    # Running sums for each metric group
    # Overall
    sum_f05 = sum_prec = sum_rec = 0.0
    # Singleton subset
    sum_f05_s = sum_prec_s = sum_rec_s = 0.0
    # Non-singleton subset
    sum_f05_ns = sum_prec_ns = sum_rec_ns = 0.0
    # S2-only subset
    sum_f05_s2 = sum_prec_s2 = sum_rec_s2 = 0.0
    # S3-only subset
    sum_f05_s3 = sum_prec_s3 = sum_rec_s3 = 0.0

    n_s2_entities = n_s3_entities = 0  # entities with any S2 / S3 truth or prediction

    # Aggregate TP/FP/FN
    total_tp = total_fp = total_fn = 0

    # Candidate-set tracking
    total_true_ids = total_covered_ids = total_candidates = 0
    n_singleton_candidates = 0  # candidates generated for true singletons

    n_entities = n_singletons = n_non_singletons = 0

    for s1_id, truth in ground_truth.items():
        # Fetch prediction — default empty if missing
        pred = predictions.get(s1_id, frozenset())
        if not isinstance(pred, frozenset):
            pred = frozenset(pred)

        is_singleton = len(truth) == 0

        # ---- Overall entity scores ----------------------------------------
        p, r, f = _entity_scores(pred, truth)
        sum_f05 += f
        sum_prec += p
        sum_rec += r

        # ---- Subset: singleton / non-singleton ----------------------------
        if is_singleton:
            n_singletons += 1
            sum_f05_s += f
            sum_prec_s += p
            sum_rec_s += r
        else:
            n_non_singletons += 1
            sum_f05_ns += f
            sum_prec_ns += p
            sum_rec_ns += r

        # ---- Aggregate TP / FP / FN ---------------------------------------
        tp = len(pred & truth)
        total_tp += tp
        total_fp += len(pred - truth)
        total_fn += len(truth - pred)

        # ---- Source-specific metrics (S2 and S3 IDs separately) ----------
        truth_s2 = frozenset(x for x in truth if x.startswith("S2-"))
        pred_s2 = frozenset(x for x in pred if x.startswith("S2-"))
        if truth_s2 or pred_s2:
            p2, r2, f2 = _entity_scores(pred_s2, truth_s2)
            sum_f05_s2 += f2
            sum_prec_s2 += p2
            sum_rec_s2 += r2
            n_s2_entities += 1

        truth_s3 = frozenset(x for x in truth if x.startswith("S3-"))
        pred_s3 = frozenset(x for x in pred if x.startswith("S3-"))
        if truth_s3 or pred_s3:
            p3, r3, f3 = _entity_scores(pred_s3, truth_s3)
            sum_f05_s3 += f3
            sum_prec_s3 += p3
            sum_rec_s3 += r3
            n_s3_entities += 1

        # ---- Candidate-set metrics ----------------------------------------
        if candidates is not None:
            cands = candidates.get(s1_id, frozenset())
            if not isinstance(cands, frozenset):
                cands = frozenset(cands)
            total_candidates += len(cands)
            total_true_ids += len(truth)
            total_covered_ids += len(truth & cands)

        # ---- Per-entity storage (error analysis) --------------------------
        if store_per_entity:
            entry: dict = {
                "precision": p, "recall": r, "f05": f,
                "tp": tp,
                "fp": len(pred - truth),
                "fn": len(truth - pred),
                "predicted": pred,
                "truth": truth,
                "is_singleton": is_singleton,
            }
            if candidates is not None:
                cands = candidates.get(s1_id, frozenset())
                entry["candidates"] = frozenset(cands)
                entry["candidate_recall"] = (
                    len(truth & cands) / len(truth) if truth else 1.0
                )
            result.per_entity[s1_id] = entry

        n_entities += 1

    # ---- Macro averages ---------------------------------------------------
    if n_entities == 0:
        return result   # nothing evaluated

    result.n_entities = n_entities
    result.n_singletons = n_singletons
    result.n_non_singletons = n_non_singletons

    result.f05_macro = sum_f05 / n_entities
    result.precision_macro = sum_prec / n_entities
    result.recall_macro = sum_rec / n_entities

    if n_singletons > 0:
        result.f05_singleton = sum_f05_s / n_singletons
        result.precision_singleton = sum_prec_s / n_singletons
        result.recall_singleton = sum_rec_s / n_singletons

    if n_non_singletons > 0:
        result.f05_non_singleton = sum_f05_ns / n_non_singletons
        result.precision_non_singleton = sum_prec_ns / n_non_singletons
        result.recall_non_singleton = sum_rec_ns / n_non_singletons

    if n_s2_entities > 0:
        result.f05_s2 = sum_f05_s2 / n_s2_entities
        result.precision_s2 = sum_prec_s2 / n_s2_entities
        result.recall_s2 = sum_rec_s2 / n_s2_entities

    if n_s3_entities > 0:
        result.f05_s3 = sum_f05_s3 / n_s3_entities
        result.precision_s3 = sum_prec_s3 / n_s3_entities
        result.recall_s3 = sum_rec_s3 / n_s3_entities

    result.tp = total_tp
    result.fp = total_fp
    result.fn = total_fn

    # ---- Candidate-set metrics -------------------------------------------
    if candidates is not None:
        result.candidates_per_s1 = total_candidates / n_entities
        result.candidate_recall = (
            total_covered_ids / total_true_ids
            if total_true_ids > 0
            else 1.0      # all singletons → no true IDs to cover
        )
        # Reduction ratio is computed externally (needs S2+S3 pool size).
        # Set to None here; the caller should populate it after calling
        # evaluate() if the pool size is known.
        result.candidate_reduction_ratio = None

    return result
