"""
split.py — Reproducible stratified train/validation split for the Business
Entity Resolution challenge.

Split Design
------------
The split operates at the **Source 1 entity level**: each S1 entity is
assigned to either the TRAIN or the VALIDATION set. Source 2 and Source 3
records are never directly split — they remain in their source files and
are accessed at query time by downstream pipeline components.

Stratification
--------------
To ensure that both TRAIN and VALIDATION have representative distributions,
each S1 entity is assigned a stratum key:

    (country, match_count_bucket)

Where match_count_bucket groups the number of S2/S3 matches:
    0        — singleton (0 matches)
    1        — exactly 1 match
    2–3      — small multi-match
    4–5      — medium multi-match
    6+       — large multi-match

Within each stratum, entities are shuffled with a fixed seed and the last
``val_fraction`` are designated validation.

Separation Guarantee
--------------------
TRAIN entities and VALIDATION entities are stored in separate files:

    data/splits/train_ids.txt
    data/splits/val_ids.txt

Any component that learns from data (e.g., a future cross-script
representation learned from name pairs in the training ground truth) MUST
only access TRAIN entity IDs and their S2/S3 matches — never the validation
ground truth. The split files are the authoritative boundary.

Usage
-----
    from src.validation.split import create_split, load_split

    # Create and persist (only do once)
    split = create_split(
        source1_path=config.TRAIN_SOURCE1,
        ground_truth_path=config.TRAIN_GROUND_TRUTH,
        output_dir=config.SPLITS_DIR,
    )

    # Load persisted split
    split = load_split(config.SPLITS_DIR)
    # split.train_ids  — set[str]
    # split.val_ids    — set[str]
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.data.loader import load_s1_metadata
from src.utils.config import (
    MATCH_COUNT_BUCKETS,
    SPLIT_SEED,
    VAL_FRACTION,
)


# ---------------------------------------------------------------------------
# Match-count bucketing
# ---------------------------------------------------------------------------

def _match_count_bucket(n: int) -> str:
    """
    Map a match count to a string bucket label.

    Parameters
    ----------
    n:
        The number of matched S2/S3 IDs for an S1 entity.

    Returns
    -------
    str
        One of: '0', '1', '2-3', '4-5', '6+'
    """
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 5:
        return "4-5"
    return "6+"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DataSplit:
    """
    Holds the train/validation split as two disjoint sets of S1 entity IDs.

    Attributes
    ----------
    train_ids:
        S1 entity IDs assigned to the TRAIN portion.
    val_ids:
        S1 entity IDs assigned to the VALIDATION portion.
    seed:
        Random seed used to produce this split.
    val_fraction:
        Target validation fraction (actual may differ slightly due to rounding
        within small strata).
    meta:
        Optional dict of split statistics (country/bucket distributions etc.)
        populated by ``create_split``.
    """

    train_ids: set[str]
    val_ids: set[str]
    seed: int = SPLIT_SEED
    val_fraction: float = VAL_FRACTION
    meta: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.train_ids)

    @property
    def n_val(self) -> int:
        return len(self.val_ids)

    @property
    def actual_val_fraction(self) -> float:
        total = self.n_train + self.n_val
        return self.n_val / total if total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  DATA SPLIT SUMMARY",
            "=" * 60,
            f"  Seed                : {self.seed}",
            f"  Target val fraction : {self.val_fraction:.2%}",
            f"  Actual val fraction : {self.actual_val_fraction:.2%}",
            f"  Train entities      : {self.n_train:>10,}",
            f"  Val entities        : {self.n_val:>10,}",
            f"  Total               : {self.n_train + self.n_val:>10,}",
        ]
        if "country_train" in self.meta:
            lines.append("-" * 60)
            lines.append("  Country distribution (train | val):")
            all_countries = sorted(
                set(self.meta["country_train"]) | set(self.meta["country_val"])
            )
            for c in all_countries:
                n_tr = self.meta["country_train"].get(c, 0)
                n_va = self.meta["country_val"].get(c, 0)
                lines.append(f"    {c:<10} : {n_tr:>8,} | {n_va:>8,}")
        if "bucket_train" in self.meta:
            lines.append("-" * 60)
            lines.append("  Match-count bucket distribution (train | val):")
            all_buckets = sorted(
                set(self.meta["bucket_train"]) | set(self.meta["bucket_val"]),
                key=lambda b: (int(b[0]) if b[0].isdigit() else 999),
            )
            for b in all_buckets:
                n_tr = self.meta["bucket_train"].get(b, 0)
                n_va = self.meta["bucket_val"].get(b, 0)
                lines.append(f"    {b:<6} matches : {n_tr:>8,} | {n_va:>8,}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------

def create_split(
    source1_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
    val_fraction: float = VAL_FRACTION,
    seed: int = SPLIT_SEED,
    force: bool = False,
) -> DataSplit:
    """
    Create a reproducible stratified train/validation split and persist it.

    The function streams source1 and ground_truth to build minimal metadata
    (entity_id, country, match_count), stratifies, splits, and writes:

        output_dir/train_ids.txt  — one S1 entity ID per line
        output_dir/val_ids.txt    — one S1 entity ID per line
        output_dir/split_meta.json — statistics and configuration

    Parameters
    ----------
    source1_path:
        Path to train_source1.tsv (READ-ONLY challenge data).
    ground_truth_path:
        Path to train_ground_truth.tsv (READ-ONLY challenge data).
    output_dir:
        Directory where the split files are written (our writable data dir).
    val_fraction:
        Target fraction of entities to hold out for validation [0, 1).
    seed:
        Random seed for reproducibility.
    force:
        If True, recreate split even if output files already exist.
        If False (default), skip if files exist.

    Returns
    -------
    DataSplit
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_ids.txt"
    val_path = output_dir / "val_ids.txt"
    meta_path = output_dir / "split_meta.json"

    # ----- Skip if already created and force=False -------------------------
    if train_path.exists() and val_path.exists() and not force:
        print(f"[split] Split files found at {output_dir}. Loading existing split.")
        return load_split(output_dir)

    print(f"[split] Loading S1 metadata from training data (streaming)…")
    meta = load_s1_metadata(source1_path, ground_truth_path)
    print(f"[split] Loaded metadata for {len(meta):,} S1 entities.")

    # ----- Assign each entity a stratum key --------------------------------
    strata: dict[str, list[str]] = {}  # key → list of entity_ids
    for eid, info in meta.items():
        bucket = _match_count_bucket(info["match_count"])
        stratum_key = f"{info['country']}|{bucket}"
        strata.setdefault(stratum_key, []).append(eid)

    print(f"[split] Found {len(strata)} strata.")

    # ----- Stratified split ------------------------------------------------
    rng = random.Random(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []

    for stratum_key, ids in strata.items():
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_fraction)) if len(ids) > 1 else 0
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    # ---- Compute split statistics -----------------------------------------
    country_train: dict[str, int] = {}
    country_val: dict[str, int] = {}
    bucket_train: dict[str, int] = {}
    bucket_val: dict[str, int] = {}

    train_set = set(train_ids)
    val_set = set(val_ids)

    for eid, info in meta.items():
        bucket = _match_count_bucket(info["match_count"])
        country = info["country"]
        if eid in train_set:
            country_train[country] = country_train.get(country, 0) + 1
            bucket_train[bucket] = bucket_train.get(bucket, 0) + 1
        else:
            country_val[country] = country_val.get(country, 0) + 1
            bucket_val[bucket] = bucket_val.get(bucket, 0) + 1

    split_meta = {
        "seed": seed,
        "val_fraction_target": val_fraction,
        "val_fraction_actual": len(val_ids) / (len(train_ids) + len(val_ids)),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_total": len(train_ids) + len(val_ids),
        "country_train": country_train,
        "country_val": country_val,
        "bucket_train": bucket_train,
        "bucket_val": bucket_val,
        "source1_path": str(source1_path),
        "ground_truth_path": str(ground_truth_path),
    }

    # ----- Persist ---------------------------------------------------------
    _write_id_file(train_path, train_ids)
    _write_id_file(val_path, val_ids)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(split_meta, fh, indent=2)

    print(
        f"[split] Created split -> train={len(train_ids):,} | val={len(val_ids):,} "
        f"(actual val fraction={split_meta['val_fraction_actual']:.2%})"
    )
    print(f"[split] Saved to {output_dir}")

    result = DataSplit(
        train_ids=train_set,
        val_ids=val_set,
        seed=seed,
        val_fraction=val_fraction,
        meta=split_meta,
    )
    return result


# ---------------------------------------------------------------------------
# Split loading
# ---------------------------------------------------------------------------

def load_split(output_dir: Path) -> DataSplit:
    """
    Load a previously created split from disk.

    Parameters
    ----------
    output_dir:
        Directory containing ``train_ids.txt``, ``val_ids.txt``, and
        ``split_meta.json`` (written by ``create_split``).

    Returns
    -------
    DataSplit
    """
    output_dir = Path(output_dir)
    train_ids = _read_id_file(output_dir / "train_ids.txt")
    val_ids = _read_id_file(output_dir / "val_ids.txt")

    meta: dict = {}
    meta_path = output_dir / "split_meta.json"
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)

    return DataSplit(
        train_ids=set(train_ids),
        val_ids=set(val_ids),
        seed=meta.get("seed", SPLIT_SEED),
        val_fraction=meta.get("val_fraction_target", VAL_FRACTION),
        meta=meta,
    )


def load_validation_ground_truth(
    ground_truth_path: Path,
    val_ids: set[str],
) -> dict[str, frozenset[str]]:
    """
    Stream the ground-truth file and return only the validation subset.

    This keeps the validation ground truth clearly separated from the training
    ground truth. Downstream learning components must only receive the training
    subset (i.e., they must not receive the output of this function).

    Parameters
    ----------
    ground_truth_path:
        Path to train_ground_truth.tsv.
    val_ids:
        Set of S1 entity IDs that belong to the validation partition.

    Returns
    -------
    dict[str, frozenset[str]]
        Ground-truth mapping for validation entities only.
    """
    from src.data.loader import iter_ground_truth

    val_gt: dict[str, frozenset[str]] = {}
    for s1_id, matched in iter_ground_truth(ground_truth_path):
        if s1_id in val_ids:
            val_gt[s1_id] = matched
    return val_gt


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _write_id_file(path: Path, ids: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for eid in ids:
            fh.write(eid + "\n")


def _read_id_file(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]
