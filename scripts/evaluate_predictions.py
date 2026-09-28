"""
evaluate_predictions.py — Evaluate a matching_results.tsv against the
held-out validation ground truth.

This script is the primary tool used after every experiment to measure
how well a given set of predictions performs on the validation split.

Usage
-----
    python scripts/evaluate_predictions.py \
        --predictions path/to/matching_results.tsv \
        [--candidates path/to/candidate_pairs.tsv] \
        [--per-entity]

Or call programmatically:

    from scripts.evaluate_predictions import evaluate_from_tsv
    result = evaluate_from_tsv("output/matching_results.tsv")
    print(result.summary())

The script reads only validation entity IDs (from data/splits/val_ids.txt)
and the training ground truth — it never touches the test set.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.metrics import evaluate, EvaluationResult
from src.utils.config import SPLITS_DIR, TRAIN_GROUND_TRUTH
from src.validation.split import load_split, load_validation_ground_truth

_TSV_DELIM = "\t"
_ID_DELIM = ","


def _load_tsv_predictions(path: Path) -> dict[str, frozenset[str]]:
    """
    Load a matching_results.tsv or candidate_pairs.tsv into a dict.

    Expected format (two columns, tab-separated):
        source1_entity_id   <matched/candidate IDs, comma-separated>

    Parameters
    ----------
    path:
        Path to the TSV file.

    Returns
    -------
    dict mapping S1 entity IDs to frozensets of predicted/candidate IDs.
    """
    result: dict[str, frozenset[str]] = {}
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter=_TSV_DELIM)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            ids_raw = row[1].strip() if len(row) > 1 else ""
            if ids_raw:
                ids: frozenset[str] = frozenset(
                    x.strip() for x in ids_raw.split(_ID_DELIM) if x.strip()
                )
            else:
                ids = frozenset()
            result[s1_id] = ids
    return result


def evaluate_from_tsv(
    predictions_path: str | Path,
    candidates_path: str | Path | None = None,
    splits_dir: Path = SPLITS_DIR,
    ground_truth_path: Path = TRAIN_GROUND_TRUTH,
    per_entity: bool = False,
) -> EvaluationResult:
    """
    Evaluate a predictions TSV against the held-out validation ground truth.

    Parameters
    ----------
    predictions_path:
        Path to matching_results.tsv produced by the pipeline.
    candidates_path:
        Optional path to candidate_pairs.tsv. When given, candidate-set
        metrics (recall ceiling, candidates per S1) are computed.
    splits_dir:
        Directory containing train_ids.txt and val_ids.txt.
    ground_truth_path:
        Path to train_ground_truth.tsv.
    per_entity:
        When True, per-entity breakdown is stored in the result for error
        analysis.

    Returns
    -------
    EvaluationResult
    """
    split = load_split(splits_dir)
    val_gt = load_validation_ground_truth(ground_truth_path, split.val_ids)

    predictions = _load_tsv_predictions(Path(predictions_path))

    candidates = None
    if candidates_path is not None:
        candidates = _load_tsv_predictions(Path(candidates_path))

    return evaluate(predictions, val_gt, candidates=candidates, store_per_entity=per_entity)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a matching_results.tsv against the validation split."
    )
    parser.add_argument(
        "--predictions", "-p", required=True,
        help="Path to matching_results.tsv to evaluate.",
    )
    parser.add_argument(
        "--candidates", "-c", default=None,
        help="Optional path to candidate_pairs.tsv (enables candidate metrics).",
    )
    parser.add_argument(
        "--per-entity", action="store_true",
        help="Print per-entity breakdown (verbose; for error analysis).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = evaluate_from_tsv(
        predictions_path=args.predictions,
        candidates_path=args.candidates,
        per_entity=args.per_entity,
    )
    elapsed = time.perf_counter() - t0

    print(result.summary())
    print(f"\nEvaluation completed in {elapsed:.2f}s")

    if args.per_entity and result.per_entity:
        print("\nPer-entity breakdown (worst 10 by F_0.5):")
        sorted_entities = sorted(
            result.per_entity.items(),
            key=lambda kv: kv[1]["f05"],
        )[:10]
        for eid, info in sorted_entities:
            print(
                f"  {eid}: f05={info['f05']:.3f}, "
                f"tp={info['tp']}, fp={info['fp']}, fn={info['fn']}, "
                f"singleton={info['is_singleton']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
