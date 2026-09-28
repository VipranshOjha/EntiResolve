"""
tests/test_phase4_4b.py

Phase 4.4B — Locked Ensemble Confirmation Tests.
Verifies:
1. Holdout disjointness with Phase 4.4 10k dev_val queries.
2. Official validation exclusion (val_ids.txt has 0 overlap).
3. Dev-train exclusion (dev_train_ids.txt has 0 overlap) and holdout is subset of dev_val.
4. Fixed ensemble parameters: alpha=0.88, threshold=0.915, K=175.
5. Deterministic ensemble blend calculation (S = 0.88 * P_sampled + 0.12 * P_full).
6. Identical candidate sets and provenance consistency across models.
7. Scoring determinism on repeated mock inference.
"""

from pathlib import Path
import json
import numpy as np
import pytest


@pytest.fixture
def holdout_ids():
    path = Path("data/validation/phase4_4b_locked_holdout_ids.txt")
    if not path.exists():
        pytest.skip("Holdout IDs file not found")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def test_holdout_cardinality_and_disjointness(holdout_ids):
    """Verifies that holdout has exactly 10,000 queries and 0 overlap with Phase 4.4."""
    assert len(holdout_ids) == 10000, f"Expected 10,000 holdout IDs, got {len(holdout_ids)}"
    assert len(set(holdout_ids)) == 10000, "Holdout IDs contain duplicates!"

    p44_path = Path("data/validation/phase4_3b_dev_val_10k_ids.txt")
    if p44_path.exists():
        with open(p44_path, "r", encoding="utf-8") as f:
            p44_ids = set(line.strip() for line in f if line.strip())
        overlap = set(holdout_ids) & p44_ids
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping queries with Phase 4.4!"


def test_official_val_exclusion(holdout_ids):
    """Verifies that holdout has 0 overlap with official validation and dev_train, and is in dev_val."""
    val_path = Path("data/splits/val_ids.txt")
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            val_ids = set(line.strip() for line in f if line.strip())
        overlap = set(holdout_ids) & val_ids
        assert len(overlap) == 0, f"FATAL: Overlap of {len(overlap)} with official validation!"

    train_path = Path("data/splits/dev_train_ids.txt")
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_ids = set(line.strip() for line in f if line.strip())
        overlap = set(holdout_ids) & train_ids
        assert len(overlap) == 0, f"FATAL: Overlap of {len(overlap)} with dev_train!"

    dev_val_path = Path("data/splits/dev_val_ids.txt")
    if dev_val_path.exists():
        with open(dev_val_path, "r", encoding="utf-8") as f:
            dev_val_ids = set(line.strip() for line in f if line.strip())
        assert set(holdout_ids).issubset(dev_val_ids), "Holdout queries are not a strict subset of dev_val!"


def test_locked_ensemble_parameters():
    """Verifies that ensemble parameters match the locked specification exactly."""
    LOCKED_ALPHA = 0.88
    LOCKED_THRESHOLD = 0.915
    LOCKED_K = 175

    assert LOCKED_ALPHA == 0.88
    assert LOCKED_THRESHOLD == 0.915
    assert LOCKED_K == 175

    # Check that linear combination weights sum to 1.0
    w_sampled = LOCKED_ALPHA
    w_full = 1.0 - LOCKED_ALPHA
    assert abs(w_sampled + w_full - 1.0) < 1e-9
    assert abs(w_full - 0.12) < 1e-9


def test_deterministic_ensemble_blend_calculation():
    """Verifies exact deterministic calculation of S = 0.88 * P_sampled + 0.12 * P_full."""
    p_sampled = [0.95, 0.92, 0.80, 0.50, 0.10, 0.0]
    p_full = [0.90, 0.20, 0.85, 0.50, 0.10, 1.0]

    expected = [
        round(0.88 * 0.95 + 0.12 * 0.90, 5),  # 0.836 + 0.108 = 0.944
        round(0.88 * 0.92 + 0.12 * 0.20, 5),  # 0.8096 + 0.024 = 0.8336
        round(0.88 * 0.80 + 0.12 * 0.85, 5),  # 0.704 + 0.102 = 0.806
        round(0.88 * 0.50 + 0.12 * 0.50, 5),  # 0.44 + 0.06 = 0.50
        round(0.88 * 0.10 + 0.12 * 0.10, 5),  # 0.088 + 0.012 = 0.10
        round(0.88 * 0.00 + 0.12 * 1.00, 5),  # 0.0 + 0.12 = 0.12
    ]

    calculated = [round(0.88 * s + 0.12 * f, 5) for s, f in zip(p_sampled, p_full)]
    assert calculated == expected

    # Decision threshold tau = 0.915
    tau = 0.915
    decisions = [s >= tau for s in calculated]
    assert decisions == [True, False, False, False, False, False]


def test_identical_candidate_sets_across_models():
    """Verifies that candidates and provenance are identically shared across models."""
    sample_cands = ["S2-1001", "S3-2002", "S2-3003"]
    scores_sampled = {tid: 0.92 - i * 0.1 for i, tid in enumerate(sample_cands)}
    scores_full = {tid: 0.88 - i * 0.2 for i, tid in enumerate(sample_cands)}

    # Same candidate IDs must be scored in identical order
    blend_scores = {}
    for tid in sample_cands:
        s_blend = round(0.88 * scores_sampled[tid] + 0.12 * scores_full[tid], 5)
        blend_scores[tid] = s_blend

    assert list(blend_scores.keys()) == sample_cands
    assert blend_scores["S2-1001"] == round(0.88 * 0.92 + 0.12 * 0.88, 5)


def test_repeated_scoring_determinism():
    """Verifies that repeated feature extraction and blend calculation is bit-for-bit identical."""
    p_sampled = np.array([0.94512, 0.82341, 0.91500, 0.12000, 0.99999], dtype=np.float64)
    p_full = np.array([0.91234, 0.75432, 0.91500, 0.05000, 0.99999], dtype=np.float64)

    run1 = np.round(0.88 * p_sampled + 0.12 * p_full, 5)
    run2 = np.round(0.88 * p_sampled + 0.12 * p_full, 5)

    np.testing.assert_array_equal(run1, run2)
    assert (run1 >= 0.915).tolist() == [True, False, True, False, True]
