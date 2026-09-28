"""
test_evaluator.py — Unit tests for the F_0.5 evaluation engine.

All tests use small synthetic examples so the evaluator can be verified
completely independently of the large challenge datasets.

Test coverage
-------------
Each test function corresponds to a named evaluation scenario:

01. perfect_prediction       — all matches found, nothing extra
02. empty_prediction         — predicted nothing at all
03. singleton_correct        — singleton correctly predicted as empty
04. singleton_incorrect      — singleton receives a false match
05. partial_match            — some matches found, some missed
06. false_positive_only      — predicted ID not in truth
07. false_negative_only      — truth ID not predicted
08. duplicate_prediction     — duplicate in predicted set (should be de-duped)
09. s2_s3_mixed_matches      — entity has both S2 and S3 true matches
10. macro_average            — macro average over multiple entities
11. f05_formula_from_readme  — explicit formula check from README example
12. source_breakdown_metrics — S2-only and S3-only metric decomposition
13. missing_prediction       — entity in GT but absent from predictions dict
14. all_singletons           — corpus of only singletons
15. high_fp_penalty          — verifies precision-heavy penalty of F_0.5
16. candidate_recall_basic   — candidate-set metrics computed correctly

Run with:
    python -m pytest tests/test_evaluator.py -v
or:
    python tests/test_evaluator.py
"""

from __future__ import annotations

import math
import sys

# Windows cp1252 console fix — must happen before any print()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import traceback
from pathlib import Path
from typing import Callable

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import _entity_scores, evaluate, EvaluationResult

# ---------------------------------------------------------------------------
# Mini test harness (no pytest dependency — stdlib only)
# ---------------------------------------------------------------------------

_PASSED: list[str] = []
_FAILED: list[str] = []


def _run(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        _PASSED.append(name)
        print(f"  PASS  {name}")
    except AssertionError as exc:
        _FAILED.append(name)
        print(f"  FAIL  {name}  -- {exc}")
    except Exception:
        _FAILED.append(name)
        print(f"  FAIL  {name}  -- unexpected exception:")
        traceback.print_exc()


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Helper: wrap frozensets
# ---------------------------------------------------------------------------

def fs(*ids: str) -> frozenset[str]:
    return frozenset(ids)


def preds(**kw: frozenset[str]) -> dict[str, frozenset[str]]:
    return dict(kw)


def gt(**kw: frozenset[str]) -> dict[str, frozenset[str]]:
    return dict(kw)


# ============================================================================
# Test 01 — Perfect prediction
# ============================================================================

def test_perfect_prediction():
    """All matches found, nothing extra. Expect F_0.5 = 1.0."""
    p, r, f = _entity_scores(fs("S2-1", "S3-1"), fs("S2-1", "S3-1"))
    assert _approx(p, 1.0), f"precision={p}"
    assert _approx(r, 1.0), f"recall={r}"
    assert _approx(f, 1.0), f"f05={f}"

    result = evaluate(
        preds(e1=fs("S2-1", "S3-1")),
        gt(e1=fs("S2-1", "S3-1")),
    )
    assert _approx(result.f05_macro, 1.0), f"macro f05={result.f05_macro}"
    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0


# ============================================================================
# Test 02 — Completely empty prediction (non-singleton)
# ============================================================================

def test_empty_prediction():
    """
    Truth non-empty, predicted=∅. Recall=0 so F=0.
    Precision should be 1.0 (vacuous) but F still 0.
    """
    p, r, f = _entity_scores(fs(), fs("S2-1", "S3-1"))
    assert _approx(p, 1.0), f"precision={p}; expected vacuous precision=1.0"
    assert _approx(r, 0.0), f"recall={r}"
    assert _approx(f, 0.0), f"f05={f}"

    result = evaluate(preds(e1=fs()), gt(e1=fs("S2-1", "S3-1")))
    assert _approx(result.f05_macro, 0.0), f"macro f05={result.f05_macro}"
    assert result.tp == 0
    assert result.fp == 0
    assert result.fn == 2


# ============================================================================
# Test 03 — Singleton correctly predicted
# ============================================================================

def test_singleton_correct():
    """truth=∅, predicted=∅ → F = 1.0 (correctly identified singleton)."""
    p, r, f = _entity_scores(fs(), fs())
    assert _approx(f, 1.0), f"f05={f}"

    result = evaluate(preds(s=fs()), gt(s=fs()))
    assert _approx(result.f05_macro, 1.0)
    assert result.n_singletons == 1
    assert _approx(result.f05_singleton, 1.0)
    assert result.tp == 0
    assert result.fp == 0
    assert result.fn == 0


# ============================================================================
# Test 04 — Singleton incorrectly matched
# ============================================================================

def test_singleton_incorrect():
    """truth=∅, predicted≠∅ → F = 0.0 (false merge on singleton)."""
    p, r, f = _entity_scores(fs("S2-99"), fs())
    assert _approx(f, 0.0), f"f05={f}"

    result = evaluate(preds(s=fs("S2-99")), gt(s=fs()))
    assert _approx(result.f05_macro, 0.0)
    assert _approx(result.f05_singleton, 0.0)
    assert result.n_singletons == 1
    assert result.fp == 1


# ============================================================================
# Test 05 — Partial match
# ============================================================================

def test_partial_match():
    """Found 1 of 2 true matches. Precision=1.0, Recall=0.5."""
    p, r, f = _entity_scores(fs("S2-1"), fs("S2-1", "S3-1"))
    assert _approx(p, 1.0)
    assert _approx(r, 0.5)
    expected_f = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)   # = 0.625/0.75
    assert _approx(f, expected_f), f"f05={f}, expected={expected_f}"


# ============================================================================
# Test 06 — False positive only
# ============================================================================

def test_false_positive_only():
    """Predicted a wrong ID; truth has a different ID we missed entirely."""
    p, r, f = _entity_scores(fs("S2-WRONG"), fs("S2-RIGHT"))
    assert _approx(p, 0.0)
    assert _approx(r, 0.0)
    assert _approx(f, 0.0)

    result = evaluate(
        preds(e=fs("S2-WRONG")),
        gt(e=fs("S2-RIGHT")),
    )
    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 1


# ============================================================================
# Test 07 — False negative only (missed one, predicted one correctly)
# ============================================================================

def test_false_negative_only():
    """Predicted 1 correct, missed 1. Precision=1.0, Recall=0.5."""
    result = evaluate(
        preds(e=fs("S2-A")),
        gt(e=fs("S2-A", "S3-B")),
    )
    assert _approx(result.precision_macro, 1.0), f"prec={result.precision_macro}"
    assert _approx(result.recall_macro, 0.5), f"rec={result.recall_macro}"
    assert result.tp == 1
    assert result.fp == 0
    assert result.fn == 1


# ============================================================================
# Test 08 — Duplicate IDs in prediction (must be de-duplicated)
# ============================================================================

def test_duplicate_prediction_handling():
    """
    frozenset de-duplicates automatically; evaluator should behave as if the
    duplicate never existed (same as predicting the unique set).
    """
    # Both representations produce the same frozenset
    pred_with_dupe = frozenset(["S2-1", "S2-1", "S3-1"])  # frozenset de-dupes
    pred_without = frozenset(["S2-1", "S3-1"])

    p1, r1, f1 = _entity_scores(pred_with_dupe, fs("S2-1", "S3-1"))
    p2, r2, f2 = _entity_scores(pred_without, fs("S2-1", "S3-1"))
    assert _approx(f1, f2), "Duplicate in frozenset should not affect score"
    assert _approx(f1, 1.0), "Perfect match expected"


# ============================================================================
# Test 09 — S2 and S3 mixed matches
# ============================================================================

def test_s2_s3_mixed_matches():
    """Entity has truth from both S2 and S3. Evaluate per-source metrics."""
    truth = fs("S2-A", "S2-B", "S3-X", "S3-Y")
    # Predict all S2, miss all S3
    pred = fs("S2-A", "S2-B")

    p, r, f = _entity_scores(pred, truth)
    assert _approx(p, 1.0), f"precision={p}"
    assert _approx(r, 0.5), f"recall={r}"

    result = evaluate(
        preds(e=pred),
        gt(e=truth),
        store_per_entity=True,
    )
    # S2 metrics: perfectly predicted both S2 IDs
    assert _approx(result.f05_s2, 1.0), f"f05_s2={result.f05_s2}"
    # S3 metrics: predicted nothing for S3 → F_0.5 = 0
    assert _approx(result.f05_s3, 0.0), f"f05_s3={result.f05_s3}"


# ============================================================================
# Test 10 — Macro average over multiple entities
# ============================================================================

def test_macro_average():
    """
    Three entities with different outcomes; macro F_0.5 = average of entity F_0.5.
    """
    # Entity A: perfect match
    pA = fs("S2-1")
    tA = fs("S2-1")
    _, _, fA = _entity_scores(pA, tA)   # 1.0

    # Entity B: singleton correct
    pB = fs()
    tB = fs()
    _, _, fB = _entity_scores(pB, tB)   # 1.0

    # Entity C: false positive on singleton → 0.0
    pC = fs("S3-9")
    tC = fs()
    _, _, fC = _entity_scores(pC, tC)   # 0.0

    expected_macro = (fA + fB + fC) / 3
    result = evaluate(
        preds(A=pA, B=pB, C=pC),
        gt(A=tA, B=tB, C=tC),
    )
    assert _approx(result.f05_macro, expected_macro), (
        f"macro={result.f05_macro}, expected={expected_macro}"
    )
    assert result.n_entities == 3
    assert result.n_singletons == 2


# ============================================================================
# Test 11 — Exact formula verification from README example
# ============================================================================

def test_f05_formula_from_readme():
    """
    README example:
        predicted = [S2-00047, S2-00193, S3-00812]
        truth     = [S2-00047, S3-00812]
        Precision = 2/3, Recall = 2/2 = 1.0
        F_0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) ≈ 0.714
    """
    pred = fs("S2-00047", "S2-00193", "S3-00812")
    truth = fs("S2-00047", "S3-00812")

    p, r, f = _entity_scores(pred, truth)
    assert _approx(p, 2 / 3, tol=1e-9), f"precision={p}"
    assert _approx(r, 1.0), f"recall={r}"

    expected_f = (1.25 * (2 / 3) * 1.0) / (0.25 * (2 / 3) + 1.0)
    assert _approx(f, expected_f, tol=1e-9), f"f05={f}, expected={expected_f}"
    # README says ≈ 0.714
    assert _approx(f, 5 / 7, tol=1e-9), f"f05={f} not equal to 5/7"


# ============================================================================
# Test 12 — Source-specific metric decomposition
# ============================================================================

def test_source_breakdown_metrics():
    """
    Multiple entities; verify that S2/S3 breakdowns are computed on the right
    subsets.
    """
    # Entity 1: has both S2 and S3 truth; predict all correctly
    # Entity 2: has only S2 truth; miss it entirely

    result = evaluate(
        preds(
            e1=fs("S2-A", "S3-B"),
            e2=fs(),                    # missed S2-C
        ),
        gt(
            e1=fs("S2-A", "S3-B"),
            e2=fs("S2-C"),
        ),
    )

    # S2 picture: e1 predicted S2-A ✓, e2 predicted nothing for S2-C ✗
    #   e1: p_s2=1, r_s2=1, f_s2=1 | e2: p_s2=1, r_s2=0, f_s2=0
    #   macro_s2 = 0.5
    assert _approx(result.f05_s2, 0.5), f"f05_s2={result.f05_s2}"

    # S3 picture: e1 predicted S3-B ✓ (e2 has no S3 truth or pred → not counted)
    assert _approx(result.f05_s3, 1.0), f"f05_s3={result.f05_s3}"


# ============================================================================
# Test 13 — Entity in GT missing from predictions
# ============================================================================

def test_missing_prediction():
    """
    An entity that appears in ground_truth but is absent from predictions dict
    should be treated as empty prediction (missed all matches).
    """
    result = evaluate(
        preds(),                      # no predictions at all
        gt(e=fs("S2-1", "S2-2")),
    )
    assert result.n_entities == 1
    assert _approx(result.f05_macro, 0.0)
    assert result.fn == 2


# ============================================================================
# Test 14 — All singletons
# ============================================================================

def test_all_singletons():
    """Corpus of only singletons; perfect singleton prediction → F = 1.0."""
    n = 10
    predictions = {f"S1-{i}": fs() for i in range(n)}
    ground_truth = {f"S1-{i}": fs() for i in range(n)}

    result = evaluate(predictions, ground_truth)
    assert result.n_singletons == n
    assert result.n_non_singletons == 0
    assert _approx(result.f05_macro, 1.0)
    assert _approx(result.f05_singleton, 1.0)


# ============================================================================
# Test 15 — High FP penalty (F_0.5 precision-heavy property)
# ============================================================================

def test_high_fp_penalty():
    """
    Adding extra false positives should decrease F_0.5 more than dropping a
    true positive (because β=0.5 weights precision more than recall).
    """
    truth = fs("S2-1", "S2-2", "S2-3")

    # Scenario A: predict all correctly (F=1.0)
    _, _, f_a = _entity_scores(fs("S2-1", "S2-2", "S2-3"), truth)

    # Scenario B: predict all correct + 1 FP (precision drops, recall stays 1)
    _, _, f_b = _entity_scores(fs("S2-1", "S2-2", "S2-3", "S2-WRONG"), truth)

    # Scenario C: predict 2 of 3 correct (precision stays 1, recall drops)
    _, _, f_c = _entity_scores(fs("S2-1", "S2-2"), truth)

    # Under F_0.5, adding an FP (B) should penalise more than dropping a TP (C)
    penalty_from_fp = f_a - f_b
    penalty_from_fn = f_a - f_c
    assert penalty_from_fp > penalty_from_fn, (
        f"Expected FP penalty ({penalty_from_fp:.4f}) > FN penalty ({penalty_from_fn:.4f})"
    )


# ============================================================================
# Test 16 — Candidate recall basic
# ============================================================================

def test_candidate_recall_basic():
    """Candidate-set metrics are correctly computed when candidates provided."""
    ground_truth = {
        "e1": fs("S2-A", "S3-B"),
        "e2": fs("S2-C"),
        "e3": fs(),               # singleton
    }
    predictions = {
        "e1": fs("S2-A"),         # missed S3-B
        "e2": fs("S2-C"),         # correct
        "e3": fs(),               # singleton correct
    }
    candidates = {
        "e1": fs("S2-A", "S3-B", "S2-EXTRA"),   # all truth in candidates ✓
        "e2": fs("S2-C", "S2-D"),                # truth in candidates ✓
        "e3": fs(),                               # no candidates for singleton
    }

    result = evaluate(predictions, ground_truth, candidates=candidates)

    # Candidate recall = covered / total_true = (2+1+0) / (2+1+0) = 1.0
    assert result.candidate_recall is not None
    assert _approx(result.candidate_recall, 1.0), (
        f"candidate_recall={result.candidate_recall}"
    )

    # candidates_per_s1 = (3 + 2 + 0) / 3
    assert _approx(result.candidates_per_s1, 5 / 3), (
        f"candidates_per_s1={result.candidates_per_s1}"
    )


# ============================================================================
# Test 17 — Partial candidate recall
# ============================================================================

def test_partial_candidate_recall():
    """When a true match is missing from candidates, candidate recall < 1.0."""
    ground_truth = {"e1": fs("S2-A", "S3-B", "S3-C")}
    predictions = {"e1": fs("S2-A")}
    # S3-C not in candidates
    candidates = {"e1": fs("S2-A", "S3-B")}

    result = evaluate(predictions, ground_truth, candidates=candidates)
    # Covered = 2 (S2-A + S3-B); total_true = 3
    assert result.candidate_recall is not None
    assert _approx(result.candidate_recall, 2 / 3), (
        f"candidate_recall={result.candidate_recall}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all_tests() -> int:
    """Run all tests; return exit code (0 = all passed, 1 = any failed)."""
    tests = [
        ("01 perfect_prediction", test_perfect_prediction),
        ("02 empty_prediction", test_empty_prediction),
        ("03 singleton_correct", test_singleton_correct),
        ("04 singleton_incorrect", test_singleton_incorrect),
        ("05 partial_match", test_partial_match),
        ("06 false_positive_only", test_false_positive_only),
        ("07 false_negative_only", test_false_negative_only),
        ("08 duplicate_prediction_handling", test_duplicate_prediction_handling),
        ("09 s2_s3_mixed_matches", test_s2_s3_mixed_matches),
        ("10 macro_average", test_macro_average),
        ("11 f05_formula_from_readme", test_f05_formula_from_readme),
        ("12 source_breakdown_metrics", test_source_breakdown_metrics),
        ("13 missing_prediction", test_missing_prediction),
        ("14 all_singletons", test_all_singletons),
        ("15 high_fp_penalty", test_high_fp_penalty),
        ("16 candidate_recall_basic", test_candidate_recall_basic),
        ("17 partial_candidate_recall", test_partial_candidate_recall),
    ]

    print("\n" + "=" * 60)
    print("  Running evaluator unit tests")
    print("=" * 60)

    for name, fn in tests:
        _run(name, fn)

    print("-" * 60)
    total = len(tests)
    n_passed = len(_PASSED)
    n_failed = len(_FAILED)
    print(f"  {n_passed}/{total} tests passed")
    if _FAILED:
        print(f"  FAILED: {', '.join(_FAILED)}")
        return 1
    else:
        print("  All tests passed [OK]")
        return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
