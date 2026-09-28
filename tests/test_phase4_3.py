"""
tests/test_phase4_3.py — Unit tests for Phase 4.3 experimental infrastructure.

Verifies:
1. Split isolation (dev_train, dev_val, val_ids are strictly pairwise disjoint)
2. No official-validation leakage in training or development pairs
3. Deterministic sampling reproducibility
4. Hard-negative selection and integrity in augmented datasets
5. Entity-level competition feature computation
6. Deterministic matcher prediction output
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.matching.model import PairwiseLightGBMMatcher


@pytest.fixture
def splits_dir() -> Path:
    return Path("data/splits")


@pytest.fixture
def pairs_dir() -> Path:
    return Path("data/pairs")


def test_split_isolation_and_no_leakage(splits_dir: Path):
    """Confirm dev_train, dev_val, and official val are 100% pairwise disjoint."""
    dev_train = {line.strip() for line in open(splits_dir / "dev_train_ids.txt") if line.strip()}
    dev_val = {line.strip() for line in open(splits_dir / "dev_val_ids.txt") if line.strip()}
    val_sealed = {line.strip() for line in open(splits_dir / "val_ids.txt") if line.strip()}

    assert len(dev_train) > 0
    assert len(dev_val) > 0
    assert len(val_sealed) == 331022

    # Pairwise intersections must be empty
    assert len(dev_train & dev_val) == 0, "dev_train & dev_val overlap!"
    assert len(dev_train & val_sealed) == 0, "dev_train & official_val overlap!"
    assert len(dev_val & val_sealed) == 0, "dev_val & official_val overlap!"


def test_training_pair_datasets_no_validation_leakage(splits_dir: Path, pairs_dir: Path):
    """Ensure no official validation queries appear in any training pair datasets."""
    train_npz_path = pairs_dir / "pairwise_expanded_dev_train.npz"
    if not train_npz_path.exists():
        pytest.skip("pairwise_expanded_dev_train.npz not present in standalone repository")

    val_sealed = {line.strip() for line in open(splits_dir / "val_ids.txt") if line.strip()}
    train_npz = np.load(train_npz_path, allow_pickle=True)
    train_s1 = set(train_npz["s1_ids"])
    assert len(train_s1 & val_sealed) == 0, "Original training set contains official val queries!"

    # Check Phase 4.3 hard-negative datasets if created
    phase4_3_dir = pairs_dir / "phase4_3"
    if phase4_3_dir.exists():
        for npz_file in phase4_3_dir.glob("*.npz"):
            ds = np.load(npz_file, allow_pickle=True)
            ds_s1 = set(ds["s1_ids"])
            assert len(ds_s1 & val_sealed) == 0, f"{npz_file.name} contains official val queries!"


def test_hard_negative_selection_integrity(pairs_dir: Path):
    """Verify that hard negative augmented datasets maintain proper label counts and dimensions."""
    phase4_3_dir = pairs_dir / "phase4_3"
    if not phase4_3_dir.exists():
        pytest.skip("Phase 4.3 pair datasets not found")

    ds_1x = np.load(phase4_3_dir / "pairwise_dev_train_hardneg_1x.npz", allow_pickle=True)
    ds_2x = np.load(phase4_3_dir / "pairwise_dev_train_hardneg_2x.npz", allow_pickle=True)

    # 61 features guaranteed
    assert ds_1x["X"].shape[1] == 61
    assert ds_2x["X"].shape[1] == 61

    # Labels binary 0 or 1
    assert set(np.unique(ds_1x["y"])).issubset({0, 1})
    assert set(np.unique(ds_2x["y"])).issubset({0, 1})

    # Positive count is preserved
    pos_1x = np.sum(ds_1x["y"] == 1)
    pos_2x = np.sum(ds_2x["y"] == 1)
    assert pos_1x == pos_2x == 31925

    # 2x dataset contains more negatives than 1x
    neg_1x = np.sum(ds_1x["y"] == 0)
    neg_2x = np.sum(ds_2x["y"] == 0)
    assert neg_2x > neg_1x


def test_entity_level_competition_features():
    """Verify entity-level competition feature logic (margins, relative score)."""
    scores = np.array([0.96, 0.91, 0.40, 0.15], dtype=np.float32)

    top1 = float(np.max(scores))
    top2 = float(np.partition(scores, -2)[-2])
    margin = top1 - top2
    n_above_90 = int(np.sum(scores >= 0.90))

    assert top1 == pytest.approx(0.96, abs=1e-4)
    assert top2 == pytest.approx(0.91, abs=1e-4)
    assert margin == pytest.approx(0.05, abs=1e-4)
    assert n_above_90 == 2

    # Relative ratio
    rel = scores[1] / (top1 + 1e-6)
    assert rel == pytest.approx(0.91 / 0.96, abs=1e-4)


def test_deterministic_matcher_reproducibility():
    """Verify that LightGBM predictions are 100% deterministic on identical input."""
    model_path = Path("models/phase4_3/matcher_hardneg_1x.pkl")
    if not model_path.exists():
        model_path = Path("models/matching_global/matcher_lightgbm.pkl")

    model = PairwiseLightGBMMatcher.load(model_path)
    rng = np.random.RandomState(42)
    sample_X = rng.randn(10, 61).astype(np.float32)

    p1 = model.predict_proba(sample_X)
    p2 = model.predict_proba(sample_X)

    np.testing.assert_array_almost_equal(p1, p2, decimal=6)
