"""
base.py — Base interface and data structures for blocking strategies.

Every blocker implements BaseBlocker:
- Builds an in-memory inverted index over target records (S2 and S3).
- Queries candidates for a given S1 query record.
- Operates within country partitions (country-first architecture).
- Tracks index statistics and memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from src.normalization.record import BusinessRecord


class BaseBlocker(ABC):
    """
    Abstract base class for an independent candidate retrieval strategy.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def reset(self) -> None:
        """Clear all indexed data and release memory."""
        raise NotImplementedError

    @abstractmethod
    def index_record(self, idx: int, record: BusinessRecord) -> None:
        """
        Add a single target record (S2 or S3) to the index.

        Parameters
        ----------
        idx:
            Integer index corresponding to this record in the candidate pool.
        record:
            Normalized BusinessRecord.
        """
        raise NotImplementedError

    def finalize(self) -> None:
        """Optional post-processing after all records are indexed (e.g. pruning)."""
        pass

    @abstractmethod
    def query(self, query_record: BusinessRecord) -> list[int]:
        """
        Retrieve candidate target indices for an S1 query record.

        Parameters
        ----------
        query_record:
            Normalized S1 BusinessRecord.

        Returns
        -------
        list[int]
            List of matching target record integer indices.
        """
        raise NotImplementedError

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic statistics about the blocker index."""
        return {"name": self.name}
