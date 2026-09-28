"""
composite.py — Composite Multi-Index Blocker and Candidate Generation Engine.

Orchestrates multiple independent blocking strategies:
1. Strategy A: Exact Normalized Name
2. Strategy B: Compact Name & Website TLD Stripping
3. Strategy C: Name Token Index (with rare token prioritization)
4. Strategy D: Character N-Gram & Prefix Retrieval
5. Strategy E: Address Token Index
6. Strategies F & G: Address Structural & Compound Keys

Features:
- Country-First Partitioning: builds and queries indices per country to bound RAM.
- Multi-Index Union & Deduplication.
- Blocker Attribution: records which blocker(s) retrieved each candidate.
- Candidate Volume Control: caps maximum candidates per S1 entity with priority-based
  truncation (exact > compact > structural > token > ngram).
- Full source preservation: candidate IDs retain original 'S2-' and 'S3-' prefixes.
"""

from __future__ import annotations

from typing import Any, Iterable

from src.blocking.address_structural import AddressStructuralBlocker
from src.blocking.address_token import AddressTokenBlocker
from src.blocking.base import BaseBlocker
from src.blocking.compact_name import CompactNameBlocker
from src.blocking.exact_name import ExactNameBlocker
from src.blocking.name_token import NameTokenBlocker
from src.blocking.ngram_blocker import CharNgramBlocker
from src.normalization.record import BusinessRecord


class CompositeBlocker:
    """
    Unified multi-index candidate generator.
    """

    def __init__(
        self,
        blockers: list[BaseBlocker] | None = None,
        max_total_candidates: int = 250,
    ) -> None:
        self.max_total_candidates = max_total_candidates
        if blockers is None:
            self.blockers: list[BaseBlocker] = [
                ExactNameBlocker(),
                CompactNameBlocker(),
                NameTokenBlocker(),
                AddressStructuralBlocker(),
                AddressTokenBlocker(),
                CharNgramBlocker(),
            ]
        else:
            self.blockers = blockers

        # Target records pool for the current active index
        self._target_ids: list[str] = []
        self._is_indexed = False

    def reset(self) -> None:
        """Clear all blocker indices and target records."""
        for b in self.blockers:
            b.reset()
        self._target_ids.clear()
        self._is_indexed = False

    def index_target_records(self, records: Iterable[BusinessRecord]) -> int:
        """
        Index an iterable of target records (S2 and S3 records).

        Parameters
        ----------
        records:
            Iterable of normalized BusinessRecord objects.

        Returns
        -------
        int
            Number of records indexed.
        """
        self.reset()
        count = 0
        for rec in records:
            idx = count
            self._target_ids.append(rec.entity_id)
            for b in self.blockers:
                b.index_record(idx, rec)
            count += 1

        for b in self.blockers:
            b.finalize()

        self._is_indexed = True
        return count

    def query(
        self,
        query_record: BusinessRecord,
        track_attribution: bool = False,
    ) -> list[str] | tuple[list[str], dict[str, list[str]]]:
        """
        Retrieve candidate target entity IDs for an S1 query record.

        Parameters
        ----------
        query_record:
            Normalized S1 BusinessRecord.
        track_attribution:
            If True, also returns a dict mapping blocker_name -> list of candidate IDs.

        Returns
        -------
        list[str] or (list[str], dict[str, list[str]])
            List of candidate target entity IDs (e.g. ['S2-123', 'S3-456']).
        """
        if not self._is_indexed:
            raise RuntimeError("CompositeBlocker must be indexed before querying.")

        candidate_scores: dict[int, int] = {}
        blocker_attribution: dict[str, list[str]] = {b.name: [] for b in self.blockers}

        # Query each blocker in order of priority
        # Exact and compact have higher priority weight
        weights = {
            "exact_normalized_name": 10,
            "compact_name": 8,
            "address_structural": 5,
            "name_token": 3,
            "address_token": 2,
            "char_ngram": 1,
        }

        for b in self.blockers:
            matched_indices = b.query(query_record)
            w = weights.get(b.name, 1)
            for idx in matched_indices:
                candidate_scores[idx] = candidate_scores.get(idx, 0) + w
                if track_attribution:
                    blocker_attribution[b.name].append(self._target_ids[idx])

        if not candidate_scores:
            if track_attribution:
                return [], blocker_attribution
            return []

        # Sort candidates by combined score descending (multi-blocker matches first)
        sorted_indices = sorted(
            candidate_scores.keys(),
            key=lambda idx: candidate_scores[idx],
            reverse=True,
        )

        # Apply maximum total candidate cap
        if len(sorted_indices) > self.max_total_candidates:
            sorted_indices = sorted_indices[: self.max_total_candidates]

        final_candidates = [self._target_ids[idx] for idx in sorted_indices]

        if track_attribution:
            return final_candidates, blocker_attribution
        return final_candidates

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_target_records": len(self._target_ids),
            "max_total_candidates": self.max_total_candidates,
            "blockers": [b.get_stats() for b in self.blockers],
        }
