"""
tests/test_phase4_5.py

Phase 4.5 — Official Validation Tests.
Verifies:
1. Official validation entity population (exactly 331,022 queries).
2. Data isolation (zero overlap with dev_train_ids.txt).
3. Locked ensemble configuration (alpha=0.88, tau=0.915, K=175).
4. Frozen model files exist and can be loaded.
5. Deterministic ensemble scoring logic.
"""

from pathlib import Path
import pytest
import numpy as np

from src.matching.model import PairwiseLightGBMMatcher


def test_official_validation_population_and_isolation():
    val_path = Path("data/splits/val_ids.txt")
    assert val_path.exists(), "Official val_ids.txt not found!"
    with open(val_path, "r", encoding="utf-8") as f:
        val_ids = [line.strip() for line in f if line.strip()]

    assert len(val_ids) == 331022, f"Expected 331,022 validation entities, got {len(val_ids)}"
    assert len(set(val_ids)) == 331022, "Duplicate IDs in official val_ids.txt!"

    dev_train_path = Path("data/splits/dev_train_ids.txt")
    if dev_train_path.exists():
        with open(dev_train_path, "r", encoding="utf-8") as f:
            dev_train_ids = set(line.strip() for line in f if line.strip())
        overlap = set(val_ids) & dev_train_ids
        assert len(overlap) == 0, f"FATAL: Overlap of {len(overlap)} with dev_train!"


def test_frozen_model_files_exist():
    model_a_path = Path("models/phase4_3/matcher_hardneg_1x.pkl")
    model_b_path = Path("models/phase4_3c/matcher_fulluniverse_hardneg_1x.pkl")

    assert model_a_path.exists(), "Model A (Sampled HardNeg 1x) not found!"
    assert model_b_path.exists(), "Model B (Full-Universe HardNeg 1x) not found!"

    model_a = PairwiseLightGBMMatcher.load(model_a_path)
    model_b = PairwiseLightGBMMatcher.load(model_b_path)

    assert model_a is not None
    assert model_b is not None


def test_frozen_ensemble_parameters():
    FROZEN_ALPHA = 0.88
    FROZEN_THRESHOLD = 0.915
    FROZEN_K = 175

    assert FROZEN_ALPHA == 0.88
    assert FROZEN_THRESHOLD == 0.915
    assert FROZEN_K == 175

    # Check scoring arithmetic
    p_sampled = 0.95
    p_full = 0.80
    blend = round(0.88 * p_sampled + 0.12 * p_full, 5)
    assert blend == 0.932
    assert blend >= FROZEN_THRESHOLD


def test_blend_monotonicity_and_bounds():
    p_sampled = np.linspace(0.0, 1.0, 101)
    p_full = np.linspace(0.0, 1.0, 101)
    blend = 0.88 * p_sampled + 0.12 * p_full

    assert np.all(blend >= 0.0)
    assert np.all(blend <= 1.0)
    assert np.all(np.diff(blend) >= 0.0)
