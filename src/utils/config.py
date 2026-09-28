"""
config.py — Central configuration for the Business Entity Resolution pipeline.

All paths, seeds, and constants are defined here so they flow through the
entire codebase without hard-coding scattered literals.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
# This file lives at:  business_entity_resolution/src/utils/config.py
# Repo root is 3 levels up.
_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = _THIS_FILE.parent.parent.parent          # business_entity_resolution/
PROJECT_ROOT = REPO_ROOT.parent                       # UnstopHackathon/

# ---------------------------------------------------------------------------
# Challenge data — treat as READ-ONLY
# ---------------------------------------------------------------------------
CHALLENGE_ROOT = PROJECT_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource"
TRAIN_DIR = CHALLENGE_ROOT / "dataset" / "train"
TEST_DIR = CHALLENGE_ROOT / "dataset" / "test"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

CHALLENGE_VALIDATOR = CHALLENGE_ROOT / "utils" / "validate_submission.py"

# ---------------------------------------------------------------------------
# Our generated artefacts (NOT part of the challenge data)
# ---------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUT_DIR = REPO_ROOT / "output"
MODELS_DIR = REPO_ROOT / "models"

# ---------------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------------
SPLIT_SEED: int = 42
VAL_FRACTION: float = 0.15          # 15% of training S1 held out for validation

# Match-count strata boundaries (upper-exclusive).
# Bucket 0: 0 matches (singletons)
# Bucket 1: exactly 1 match
# Bucket 2: 2–3 matches
# Bucket 3: 4–5 matches
# Bucket 4: 6+ matches
MATCH_COUNT_BUCKETS: tuple[int, ...] = (0, 1, 2, 4, 6)

# ---------------------------------------------------------------------------
# Metric / evaluation constants
# ---------------------------------------------------------------------------
BETA: float = 0.5          # F_beta score with beta = 0.5

# ID prefix constants (derived from the challenge spec)
S1_PREFIX = "S1-"
S2_PREFIX = "S2-"
S3_PREFIX = "S3-"

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create all writable output directories if they do not exist."""
    for d in (SPLITS_DIR, OUTPUT_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
