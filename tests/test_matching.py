"""
tests/test_matching.py — Comprehensive unit tests for Phase 4 Pairwise Matching.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.matching.baseline import DeterministicBaselineMatcher
from src.matching.calibration import CalibratedModel, brier_score, calibration_curve_stats
from src.matching.cross_script import CrossScriptAligner
from src.matching.dataset import PairwiseDatasetBuilder
from src.matching.features import (
    ALL_FEATURE_NAMES,
    NAME_FEATURE_NAMES,
    CandidateProvenance,
    PairwiseFeatureExtractor,
)
from src.matching.inference import EntityLevelPredictor
from src.matching.model import PairwiseLightGBMMatcher, PairwiseLogisticRegressionMatcher
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
from src.normalization.record import BusinessRecord, NormalizationPipeline


@pytest.fixture
def pipeline() -> NormalizationPipeline:
    return NormalizationPipeline()


@pytest.fixture
def sample_records(pipeline: NormalizationPipeline) -> tuple[BusinessRecord, BusinessRecord, BusinessRecord]:
    s1 = pipeline.normalize(
        entity_id="S1-001",
        raw_name="Acme Corp.",
        raw_address="123 Pike St, Seattle, WA 98101",
        country="US",
    )
    # Similar target with expanded legal suffix
    s2 = pipeline.normalize(
        entity_id="S2-001",
        raw_name="Acme Corporation",
        raw_address="123 Pike Street, Seattle, WA 98101",
        country="US",
    )
    # Disjoint target
    s3 = pipeline.normalize(
        entity_id="S2-999",
        raw_name="Walmart Supercenter",
        raw_address="702 SW 8th St, Bentonville, AR 72716",
        country="US",
    )
    return s1, s2, s3


def test_similarity_functions():
    assert levenshtein_ratio("apple", "apple") == 1.0
    assert levenshtein_ratio("apple", "") == 0.0
    assert 0.7 < levenshtein_ratio("starbucks", "starbuck") < 1.0

    assert jaro_winkler_similarity("starbucks", "starbucks") == 1.0
    assert jaro_winkler_similarity("starbucks", "dunkin") < 0.6

    toks1 = ["starbucks", "coffee", "store"]
    toks2 = ["starbucks", "coffee"]
    assert token_jaccard(toks1, toks2) == 2.0 / 3.0
    assert token_overlap_coefficient(toks1, toks2) == 1.0
    assert token_containment(toks1, toks2) == 1.0

    assert char_ngram_jaccard("starbucks", "starbucks") == 1.0
    assert char_ngram_jaccard("starbucks", "walmart") == 0.0

    assert house_number_agreement(["123"], ["123", "b"]) == 1.0
    assert house_number_agreement(["123"], ["124"]) == 0.0

    assert postal_code_agreement(["98101"], ["98101"]) == 1.0
    assert postal_code_agreement(["98101"], ["10001"]) == 0.0

    jacc, cnt = numeric_overlap(["123", "98101"], ["123", "555"])
    assert cnt == 1
    assert 0.3 < jacc < 0.4


def test_cross_script_aligner(pipeline: NormalizationPipeline):
    aligner = CrossScriptAligner(min_cooccurrence=1, min_dice_threshold=0.05)

    s1_eng = pipeline.normalize("S1-1", "Apollo Pharmacy", "Main St", "IN")
    s2_tel = pipeline.normalize("S2-1", "అపోలో ఫార్మసీ", "Main St", "IN")

    # Fit strictly on this pair
    aligner.fit([(s1_eng, s2_tel)])
    assert aligner.fitted
    assert len(aligner.token_associations) > 0

    # Similarity should be positive
    sim = aligner.compute_similarity(s1_eng, s2_tel)
    assert sim > 0.0

    # Pure Latin pair should be 0.0
    s3_eng = pipeline.normalize("S2-2", "Apollo Pharmacy", "Main St", "IN")
    assert aligner.compute_similarity(s1_eng, s3_eng) == 0.0

    # Serialization test
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "cross_script.json"
        aligner.save(fpath)
        loaded = CrossScriptAligner.load(fpath)
        assert loaded.fitted
        assert loaded.compute_similarity(s1_eng, s2_tel) == sim


def test_cross_script_training_leakage_guarantee(pipeline: NormalizationPipeline):
    """Confirm validation/test entities cannot enter the learned alignment map."""
    aligner = CrossScriptAligner(min_cooccurrence=1, min_dice_threshold=0.05)

    s1_train = pipeline.normalize("S1-TRAIN-01", "Shree Ganesh", "MG Road", "IN")
    s2_train = pipeline.normalize("S2-TRAIN-01", "श्री गणेश", "MG Road", "IN")

    s1_val = pipeline.normalize("S1-VAL-999", "Tata Motors", "Station Rd", "IN")
    s2_val = pipeline.normalize("S2-VAL-999", "टाटा मोटर्स", "Station Rd", "IN")

    disallowed_validation_ids = {"S1-VAL-999", "S2-VAL-999"}

    # 1. Clean training pair succeeds
    aligner.fit([(s1_train, s2_train)], disallowed_entity_ids=disallowed_validation_ids)
    assert aligner.fitted
    assert "ganesh" in aligner.token_associations or "shree" in aligner.token_associations

    # 2. Leaked validation entity must raise ValueError
    with pytest.raises(ValueError, match="Validation leakage detected"):
        aligner.fit([(s1_val, s2_val)], disallowed_entity_ids=disallowed_validation_ids)

    # 3. Canary token from validation entity must NEVER enter token associations
    for latin_tok, non_latin_map in aligner.token_associations.items():
        assert "tata" not in latin_tok
        assert "टाटा" not in non_latin_map


def test_feature_extractor(sample_records):
    s1, s2, s3 = sample_records
    extractor = PairwiseFeatureExtractor(include_cross_script=True)

    prov = CandidateProvenance(candidate_rank=1, blocker_count=3, exact_name_hit=1)
    feat1 = extractor.extract_features(s1, s2, prov)

    assert feat1.shape == (len(ALL_FEATURE_NAMES),)
    assert np.all(np.isfinite(feat1))
    assert feat1[0] == 0.0  # Exact normalized name: "acme corp" != "acme corporation"
    assert feat1[1] == 1.0  # Exact suffix-normalized name matches ("acme corporation" == "acme corporation")!

    # Compare with disjoint record
    feat2 = extractor.extract_features(s1, s3, CandidateProvenance(candidate_rank=50))
    assert feat2[1] == 0.0  # Disjoint suffix-normalized
    assert feat1[5] > feat2[5]  # Jaro-Winkler higher for s2 than s3

    # Batch extraction
    batch = extractor.extract_batch([(s1, s2, prov), (s1, s3, prov)])
    assert batch.shape == (2, len(ALL_FEATURE_NAMES))


def test_dataset_builder_and_negative_sampling(pipeline: NormalizationPipeline):
    s1 = pipeline.normalize("S1-1", "Apple Store", "1 Infinite Loop", "US")
    s2_pos = pipeline.normalize("S2-1", "Apple Store", "1 Infinite Loop", "US")
    s2_neg1 = pipeline.normalize("S2-2", "Apple Farm", "123 Country Rd", "US")
    s2_neg2 = pipeline.normalize("S2-3", "Banana Stand", "456 Beach Rd", "US")

    target_dict = {"S2-1": s2_pos, "S2-2": s2_neg1, "S2-3": s2_neg2}
    gt_targets = {"S2-1"}
    candidates = ["S2-1", "S2-2", "S2-3"]
    attribution = {
        "exact_normalized_name": ["S2-1"],
        "name_token": ["S2-1", "S2-2"],
        "address_token": ["S2-3"],
    }

    extractor = PairwiseFeatureExtractor(include_cross_script=False)
    builder = PairwiseDatasetBuilder(feature_extractor=extractor, seed=42)

    # Strategy A: all
    pairs_all, info_all = builder.build_pairs_for_query(
        s1, target_dict, gt_targets, candidates, attribution, negative_strategy="all"
    )
    assert len(pairs_all) == 3
    assert info_all["num_positives"] == 1
    assert info_all["num_negatives_selected"] == 2

    # Strategy B: random with ratio 1
    pairs_rnd, info_rnd = builder.build_pairs_for_query(
        s1, target_dict, gt_targets, candidates, attribution, negative_strategy="random", negative_ratio=1
    )
    assert len(pairs_rnd) == 2
    assert info_rnd["num_negatives_selected"] == 1

    # Strategy C: hard_similarity
    pairs_hard, _ = builder.build_pairs_for_query(
        s1, target_dict, gt_targets, candidates, attribution, negative_strategy="hard_similarity", negative_ratio=1
    )
    # The negative should be "Apple Farm" (S2-2) which has higher name similarity than "Banana Stand"
    assert pairs_hard[1][1].entity_id == "S2-2"

    # Strategy D: blocker_overlap
    pairs_blocker, _ = builder.build_pairs_for_query(
        s1, target_dict, gt_targets, candidates, attribution, negative_strategy="blocker_overlap", negative_ratio=1
    )
    assert len(pairs_blocker) == 2


def test_deterministic_baseline_matcher(sample_records):
    s1, s2, s3 = sample_records
    baseline = DeterministicBaselineMatcher()

    score_match = baseline.score_pair(s1, s2)
    score_disjoint = baseline.score_pair(s1, s3)

    assert score_match >= 0.85
    assert score_disjoint < 0.40

    extractor = PairwiseFeatureExtractor()
    feat1 = extractor.extract_features(s1, s2)
    feat2 = extractor.extract_features(s1, s3)
    X = np.vstack([feat1, feat2])
    matrix_scores = baseline.score_feature_matrix(X, extractor.feature_names)
    assert matrix_scores[0] > 0.80
    assert matrix_scores[1] < 0.40


def test_pairwise_models(sample_records):
    s1, s2, s3 = sample_records
    extractor = PairwiseFeatureExtractor()
    prov = CandidateProvenance()

    feat_pos = extractor.extract_features(s1, s2, prov)
    feat_neg = extractor.extract_features(s1, s3, prov)

    # Synthetic dataset with 20 repeats
    X = np.vstack([feat_pos] * 15 + [feat_neg] * 15)
    y = np.array([1] * 15 + [0] * 15, dtype=np.int32)

    # 1. Logistic Regression
    lr = PairwiseLogisticRegressionMatcher(C=1.0)
    lr.fit(X, y, feature_names=extractor.feature_names)
    probs_lr = lr.predict_proba(X)
    assert probs_lr.shape == (30,)
    assert probs_lr[0] > probs_lr[-1]
    assert len(lr.feature_importances_) == len(extractor.feature_names)

    # 2. LightGBM
    lgb_matcher = PairwiseLightGBMMatcher(n_estimators=10, min_child_samples=2)
    lgb_matcher.fit(X, y, feature_names=extractor.feature_names)
    probs_lgb = lgb_matcher.predict_proba(X)
    assert probs_lgb.shape == (30,)
    assert probs_lgb[0] > probs_lgb[-1]

    # Save & Load
    with tempfile.TemporaryDirectory() as tmpdir:
        lr_path = Path(tmpdir) / "lr.pkl"
        lr.save(lr_path)
        lr_loaded = PairwiseLogisticRegressionMatcher.load(lr_path)
        assert np.allclose(lr.predict_proba(X), lr_loaded.predict_proba(X))


def test_calibration_and_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.float64)
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.85, 0.15], dtype=np.float64)

    brier = brier_score(y_true, y_prob)
    assert 0.0 <= brier <= 0.1

    stats = calibration_curve_stats(y_true, y_prob, n_bins=5)
    assert "ece" in stats
    assert len(stats["bin_centers"]) == 5


def test_entity_level_predictor(sample_records):
    s1, s2, s3 = sample_records
    target_dict = {"S2-001": s2, "S2-999": s3}
    ground_truth = {"S1-001": {"S2-001"}}

    # Mock blocker returning candidate IDs
    class MockBlocker:
        def query(self, rec, track_attribution=False):
            if track_attribution:
                return ["S2-001", "S2-999"], {"exact_normalized_name": ["S2-001"], "char_ngram": ["S2-999"]}
            return ["S2-001", "S2-999"]

    extractor = PairwiseFeatureExtractor()
    baseline = DeterministicBaselineMatcher()

    predictor = EntityLevelPredictor(
        blocker=MockBlocker(),
        feature_extractor=extractor,
        model=baseline,
    )

    scored = predictor.score_query_candidates(s1, target_dict)
    assert len(scored) == 2
    assert scored[0][0] == "S2-001"
    assert scored[0][1] > scored[1][1]

    # High threshold should only keep true match S2-001
    pred = predictor.predict_entity(s1, target_dict, threshold=0.75)
    assert pred == ["S2-001"]

    # Evaluate cohort
    metrics = predictor.evaluate_cohort([s1], target_dict, ground_truth, threshold=0.75)
    assert metrics["macro_f05"] == 1.0
    assert metrics["pair_recall"] == 1.0

    # Sweep thresholds
    sweep = predictor.sweep_thresholds([s1], target_dict, ground_truth, thresholds=[0.5, 0.75, 0.95])
    assert len(sweep) == 3
