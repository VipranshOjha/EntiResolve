"""
inference.py — Entity-Level Scoring, Decision Thresholding, and Evaluation.

Translates pairwise scores into S1 entity-level prediction sets and computes
challenge-standard macro F0.5, singleton accuracy, and complete-set recall.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from src.blocking.composite import CompositeBlocker
from src.evaluation.metrics import EvaluationResult, evaluate
from src.matching.features import CandidateProvenance, PairwiseFeatureExtractor
from src.normalization.record import BusinessRecord


class EntityLevelPredictor:
    """
    Orchestrates end-to-end entity-level inference:
    Candidate Retrieval -> Pairwise Feature Extraction -> Model Scoring -> Decision Threshold.
    """

    def __init__(
        self,
        blocker: CompositeBlocker,
        feature_extractor: PairwiseFeatureExtractor,
        model: Any,
    ) -> None:
        self.blocker = blocker
        self.feature_extractor = feature_extractor
        self.model = model

    def score_query_candidates(
        self,
        s1: BusinessRecord,
        target_dict: dict[str, BusinessRecord],
        candidates: list[str] | None = None,
        attribution: dict[str, list[str]] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Score all retrieved candidates for a single S1 record.

        Returns
        -------
        list of (candidate_target_id, match_probability_or_score)
        """
        if candidates is None or attribution is None:
            cands, attr = self.blocker.query(s1, track_attribution=True)
        else:
            cands, attr = candidates, attribution

        if not cands:
            return []

        # Invert attribution
        candidate_blockers: dict[str, set[str]] = {}
        for b_name, targets in attr.items():
            for tid in targets:
                if tid not in candidate_blockers:
                    candidate_blockers[tid] = set()
                candidate_blockers[tid].add(b_name)

        valid_cands = [tid for tid in cands if tid in target_dict]
        if not valid_cands:
            return []

        pairs_to_extract = []
        for rank, tid in enumerate(valid_cands, start=1):
            tgt = target_dict[tid]
            hits = candidate_blockers.get(tid, set())
            prov = CandidateProvenance(
                candidate_rank=rank,
                composite_priority=max(1.0, 10.0 - rank * 0.1),
                blocker_count=len(hits),
                exact_name_hit=1 if "exact_normalized_name" in hits else 0,
                compact_name_hit=1 if "compact_name" in hits else 0,
                name_token_hit=1 if "name_token" in hits else 0,
                ngram_hit=1 if "char_ngram" in hits else 0,
                address_token_hit=1 if "address_token" in hits else 0,
                structural_hit=1 if "address_structural" in hits else 0,
                source_is_s2=1 if tgt.source == "s2" or tid.startswith("S2") else 0,
            )
            pairs_to_extract.append((s1, tgt, prov))

        X = self.feature_extractor.extract_batch(pairs_to_extract)

        # Check if model has score_feature_matrix (e.g. baseline) or predict_proba
        if hasattr(self.model, "score_feature_matrix"):
            scores = self.model.score_feature_matrix(X, self.feature_extractor.feature_names)
        elif hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(X)
        elif hasattr(self.model, "score_pair"):
            scores = np.array([self.model.score_pair(p[0], p[1]) for p in pairs_to_extract], dtype=np.float32)
        else:
            raise ValueError(f"Model {type(self.model)} has neither predict_proba nor score_pair.")

        return list(zip(valid_cands, [float(s) for s in scores]))

    def predict_entity(
        self,
        s1: BusinessRecord,
        target_dict: dict[str, BusinessRecord],
        threshold: float = 0.5,
        candidates: list[str] | None = None,
        attribution: dict[str, list[str]] | None = None,
    ) -> list[str]:
        """
        Produce predicted target ID set for S1 entity at specified threshold.
        """
        scored_candidates = self.score_query_candidates(
            s1=s1,
            target_dict=target_dict,
            candidates=candidates,
            attribution=attribution,
        )
        return [tid for tid, score in scored_candidates if score >= threshold]

    def evaluate_cohort(
        self,
        s1_records: Sequence[BusinessRecord],
        target_dict: dict[str, BusinessRecord],
        ground_truth: dict[str, set[str]],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Evaluate complete cohort at a fixed threshold using Phase 1 evaluate function.
        """
        predictions: dict[str, frozenset[str]] = {}
        for s1 in s1_records:
            pred = self.predict_entity(s1, target_dict, threshold=threshold)
            predictions[s1.entity_id] = frozenset(pred)

        gt_frozen = {sid: frozenset(targets) for sid, targets in ground_truth.items()}
        res = evaluate(predictions=predictions, ground_truth=gt_frozen)
        return self._result_to_dict(res)

    def sweep_thresholds(
        self,
        s1_records: Sequence[BusinessRecord],
        target_dict: dict[str, BusinessRecord],
        ground_truth: dict[str, set[str]],
        thresholds: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9),
    ) -> list[dict[str, Any]]:
        """
        Efficient threshold sweep: scores all candidates once, then evaluates each threshold.
        """
        # Pre-compute scores for all entities
        all_scored: dict[str, list[tuple[str, float]]] = {}
        for s1 in s1_records:
            scored = self.score_query_candidates(s1, target_dict)
            all_scored[s1.entity_id] = scored

        gt_frozen = {sid: frozenset(targets) for sid, targets in ground_truth.items()}
        results = []

        for thresh in thresholds:
            preds: dict[str, frozenset[str]] = {}
            for s1_id, scored in all_scored.items():
                preds[s1_id] = frozenset(tid for tid, score in scored if score >= thresh)

            res = evaluate(predictions=preds, ground_truth=gt_frozen)
            d = self._result_to_dict(res)
            d["threshold"] = thresh
            results.append(d)

        return results

    @staticmethod
    def _result_to_dict(res: EvaluationResult) -> dict[str, Any]:
        tp, fp, fn = res.tp, res.fp, res.fn
        pair_prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        pair_rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        return {
            "macro_f05": res.f05_macro,
            "macro_precision": res.precision_macro,
            "macro_recall": res.recall_macro,
            "f05_singleton": res.f05_singleton,
            "f05_non_singleton": res.f05_non_singleton,
            "f05_s2": res.f05_s2,
            "f05_s3": res.f05_s3,
            "n_entities": res.n_entities,
            "n_singletons": res.n_singletons,
            "n_non_singletons": res.n_non_singletons,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "pair_precision": pair_prec,
            "pair_recall": pair_rec,
        }
