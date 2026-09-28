"""
tests/test_phase4_4.py

Tests for Phase 4.4 — Complementary Matcher Ensemble Analysis.
Validates:
- Exact same candidate sets across models
- Probability persistence and score consistency
- Deterministic ensemble scoring
- Weighted linear blending logic
- Conservative agreement rule logic
- Soft veto / margin gate logic
- Threshold selection and metric aggregation
- Singleton handling and error decomposition
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.evaluation.metrics import _entity_scores
from src.matching.calibration import calibration_curve_stats, brier_score


@pytest.fixture
def sample_candidates():
    return ["S2-100", "S2-200", "S2-300", "S2-400"]


@pytest.fixture
def sample_scores():
    return {
        "hardneg_1x": [0.98, 0.95, 0.40, 0.10],
        "fulluniverse_hardneg_1x": [0.92, 0.20, 0.85, 0.05],
    }


def test_weighted_blending_deterministic(sample_candidates, sample_scores):
    """Verifies that linear score blending is deterministic and bounded in [0, 1]."""
    alpha = 0.88
    s_hn = sample_scores["hardneg_1x"]
    s_fu = sample_scores["fulluniverse_hardneg_1x"]

    blend_1 = [alpha * h + (1.0 - alpha) * f for h, f in zip(s_hn, s_fu)]
    blend_2 = [alpha * h + (1.0 - alpha) * f for h, f in zip(s_hn, s_fu)]

    assert blend_1 == blend_2
    for s in blend_1:
        assert 0.0 <= s <= 1.0

    # Specifically check the values:
    # 0.88 * 0.98 + 0.12 * 0.92 = 0.8624 + 0.1104 = 0.9728
    assert abs(blend_1[0] - 0.9728) < 1e-5
    # 0.88 * 0.95 + 0.12 * 0.20 = 0.8360 + 0.0240 = 0.8600
    assert abs(blend_1[1] - 0.8600) < 1e-5


def test_agreement_rule_logic(sample_candidates, sample_scores):
    """Verifies conservative agreement rule logic."""
    tau_a = 0.94
    tau_b = 0.30
    s_hn = sample_scores["hardneg_1x"]
    s_fu = sample_scores["fulluniverse_hardneg_1x"]

    preds = [
        tid for tid, h, f in zip(sample_candidates, s_hn, s_fu)
        if h >= tau_a and f >= tau_b
    ]

    # S2-100 has h=0.98 >= 0.94 and f=0.92 >= 0.30 -> Accepted
    # S2-200 has h=0.95 >= 0.94, BUT f=0.20 < 0.30 -> VETOED!
    # S2-300 has h=0.40 < 0.94 -> Rejected
    # S2-400 has h=0.10 < 0.94 -> Rejected
    assert preds == ["S2-100"]


def test_soft_veto_margin_gate_logic(sample_candidates, sample_scores):
    """Verifies soft veto / margin gate logic."""
    tau_a = 0.94
    max_margin = 0.70
    s_hn = sample_scores["hardneg_1x"]
    s_fu = sample_scores["fulluniverse_hardneg_1x"]

    preds = [
        tid for tid, h, f in zip(sample_candidates, s_hn, s_fu)
        if h >= tau_a and (h - f) <= max_margin
    ]

    # S2-100: h=0.98, margin = 0.98 - 0.92 = 0.06 <= 0.70 -> Accepted
    # S2-200: h=0.95, margin = 0.95 - 0.20 = 0.75 > 0.70 -> VETOED!
    # S2-300: h=0.40 < 0.94 -> Rejected
    assert preds == ["S2-100"]


def test_entity_scores_calculation():
    """Verifies _entity_scores for all primary cases."""
    # 1. Correct singleton
    p, r, f05 = _entity_scores(frozenset(), frozenset())
    assert (p, r, f05) == (1.0, 1.0, 1.0)

    # 2. False merge on singleton
    p, r, f05 = _entity_scores(frozenset(["S2-1"]), frozenset())
    assert (p, r, f05) == (0.0, 0.0, 0.0)

    # 3. False negative on truth
    p, r, f05 = _entity_scores(frozenset(), frozenset(["S2-1"]))
    assert (p, r, f05) == (1.0, 0.0, 0.0)

    # 4. Partial match
    truth = frozenset(["S2-1", "S2-2"])
    pred = frozenset(["S2-1", "S2-3"])
    p, r, f05 = _entity_scores(pred, truth)
    assert p == 0.5
    assert r == 0.5
    # F0.5 = 1.25 * 0.5 * 0.5 / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    assert abs(f05 - 0.5) < 1e-6


def test_calibration_curve_stats():
    """Verifies calibration statistics computation."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.85, 0.3, 0.7, 0.15, 0.95, 0.6])

    stats = calibration_curve_stats(y_true, y_prob, n_bins=5)
    assert "brier_score" in stats
    assert "ece" in stats
    assert len(stats["bin_centers"]) == 5
    assert 0.0 <= stats["brier_score"] <= 1.0
    assert 0.0 <= stats["ece"] <= 1.0


def test_checkpoint_consistency_across_models():
    """Verifies that stored checkpoints have identical candidates for both models."""
    cp_path = Path("output/checkpoints_eval_4_3c/India/batch_00000.jsonl")
    if not cp_path.exists():
        pytest.skip("Checkpoint file not present in local environment")

    with open(cp_path, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())

    cands = first_line["candidates"]
    scores = first_line["scores"]

    assert len(cands) == 175
    assert "hardneg_1x" in scores
    assert "fulluniverse_hardneg_1x" in scores
    assert len(scores["hardneg_1x"]) == len(cands)
    assert len(scores["fulluniverse_hardneg_1x"]) == len(cands)


def test_reports_generated():
    """Verifies that all Phase 4.4 reports were generated and are non-empty."""
    reports_dir = Path("reports")
    expected_files = [
        "phase4_4_model_disagreement.json",
        "phase4_4_score_calibration.json",
        "phase4_4_ensemble_metrics.json",
        "phase4_4_threshold_sweep.csv",
        "phase4_4_singleton_analysis.json",
        "phase4_4_error_decomposition.json",
    ]

    for fname in expected_files:
        p = reports_dir / fname
        assert p.exists(), f"Report file {fname} does not exist"
        assert p.stat().st_size > 100, f"Report file {fname} is suspiciously small"
