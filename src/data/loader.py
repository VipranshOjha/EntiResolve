"""
loader.py — Lightweight, streaming-friendly loaders for challenge TSV files.

Design principles
-----------------
* The challenge TSVs can be 200–500 MB each. We NEVER load more columns than
  needed for a given operation.
* All functions use generators / streaming by default. Callers that need
  random access can convert to dict/list explicitly.
* The challenge data is READ-ONLY. These loaders never write to it.
* Encoding: UTF-8 with ``errors='replace'`` to handle the mojibake observed
  in Source 2 business_name fields without crashing.

Column indices (0-based) for all source files:
    0: entity_id
    1: business_name
    2: business_address
    3: country

Column indices for train_ground_truth.tsv:
    0: source1_entity_id
    1: matched_entity_ids   (comma-separated, may be empty)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Generator, Iterator

# Challenge TSV uses tabs; ID lists inside matched_entity_ids use commas.
_TSV_DELIM = "\t"
_ID_LIST_DELIM = ","


# ---------------------------------------------------------------------------
# Low-level iterators
# ---------------------------------------------------------------------------

def _iter_tsv(path: Path) -> Generator[list[str], None, None]:
    """
    Yield each non-empty data row (as a raw list of strings) from a TSV file,
    skipping the header. Thread-safe for multiple independent iterators over
    the same file (each call opens its own file handle).

    Parameters
    ----------
    path:
        Absolute path to a TSV file. Must exist.

    Yields
    ------
    list[str]
        One list per data row; each element is a raw (untrimmed) field value.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TSV not found: {path}")

    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter=_TSV_DELIM)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                yield row


# ---------------------------------------------------------------------------
# Source entity loaders (columns 0, 3 — id and country)
# ---------------------------------------------------------------------------

def iter_source_ids_and_countries(
    path: Path,
) -> Generator[tuple[str, str], None, None]:
    """
    Stream (entity_id, country) tuples from any source TSV.

    This is the minimal footprint loader used during split creation and
    any operation that only needs entity identities + country labels.

    Parameters
    ----------
    path:
        Path to *_source1.tsv, *_source2.tsv, or *_source3.tsv.

    Yields
    ------
    (entity_id, country)
        Both strings; country may be empty (but empirically never is in this
        challenge).
    """
    for row in _iter_tsv(path):
        entity_id = row[0].strip()
        country = row[3].strip() if len(row) > 3 else ""
        yield entity_id, country


def load_source_id_to_country(path: Path) -> dict[str, str]:
    """
    Load the full (entity_id → country) mapping from a source TSV into memory.

    Use for smaller-footprint needs (e.g., S1 only, which is ~2.2M rows and
    fits comfortably in RAM). Avoid calling on S2/S3 unless necessary.

    Parameters
    ----------
    path:
        Path to a source TSV.

    Returns
    -------
    dict[str, str]
        Maps each entity_id to its country string.
    """
    return {eid: country for eid, country in iter_source_ids_and_countries(path)}


# ---------------------------------------------------------------------------
# Ground-truth loader
# ---------------------------------------------------------------------------

def iter_ground_truth(
    path: Path,
) -> Generator[tuple[str, frozenset[str]], None, None]:
    """
    Stream (source1_entity_id, matched_ids) tuples from the ground-truth TSV.

    Parameters
    ----------
    path:
        Path to train_ground_truth.tsv.

    Yields
    ------
    (source1_entity_id, matched_ids)
        ``matched_ids`` is a frozenset of ID strings (may be the empty
        frozenset for singleton entities — this is the canonical
        challenge representation of "no match").
    """
    for row in _iter_tsv(path):
        s1_id = row[0].strip()
        matched_raw = row[1].strip() if len(row) > 1 else ""
        if matched_raw:
            matched: frozenset[str] = frozenset(
                mid.strip()
                for mid in matched_raw.split(_ID_LIST_DELIM)
                if mid.strip()
            )
        else:
            matched = frozenset()
        yield s1_id, matched


def load_ground_truth(path: Path) -> dict[str, frozenset[str]]:
    """
    Load the complete ground-truth mapping into memory.

    ``ground_truth[s1_id]`` is a ``frozenset`` of matched S2/S3 IDs.
    For singleton S1 entities the frozenset is empty.

    Parameters
    ----------
    path:
        Path to train_ground_truth.tsv.

    Returns
    -------
    dict[str, frozenset[str]]
    """
    return dict(iter_ground_truth(path))


# ---------------------------------------------------------------------------
# Minimal S1 metadata loader (id + country + match count)
# ---------------------------------------------------------------------------

def load_s1_metadata(
    source1_path: Path,
    ground_truth_path: Path,
) -> dict[str, dict]:
    """
    Build a lightweight per-S1-entity metadata dict used for split creation.

    Only loads entity_id and country from source1, then joins with match count
    from ground truth. No business_name or business_address is loaded.

    Parameters
    ----------
    source1_path:
        Path to train_source1.tsv.
    ground_truth_path:
        Path to train_ground_truth.tsv.

    Returns
    -------
    dict[str, dict]
        Keys are S1 entity IDs. Values are dicts with:
            ``country``     : str
            ``match_count`` : int   (number of matched S2/S3 IDs in ground truth)
    """
    # Step 1: load entity_id → country from source1 (streaming, then store)
    meta: dict[str, dict] = {}
    for eid, country in iter_source_ids_and_countries(source1_path):
        meta[eid] = {"country": country, "match_count": 0}

    # Step 2: fill in match counts from ground truth (streaming)
    for s1_id, matched_ids in iter_ground_truth(ground_truth_path):
        if s1_id in meta:
            meta[s1_id]["match_count"] = len(matched_ids)
        # Entities in GT not in source1 are ignored (shouldn't happen in clean data)

    return meta
