"""
exact_name.py — Blocking Strategy A: Exact Normalized Name.

Builds an inverted index mapping exact normalized name representations
(including legal-suffix-normalized strings) to candidate entity indices.
High precision, fast hash lookup.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.blocking.base import BaseBlocker
from src.normalization.record import BusinessRecord


class ExactNameBlocker(BaseBlocker):
    """
    Retrieves candidates sharing the exact normalized business name
    or suffix-standardized name string.
    """

    def __init__(self, name: str = "exact_normalized_name") -> None:
        super().__init__(name=name)
        # key -> list of candidate integer indices
        self._index: dict[str, list[int]] = defaultdict(list)
        self._total_keys = 0

    def reset(self) -> None:
        self._index.clear()
        self._total_keys = 0

    def index_record(self, idx: int, record: BusinessRecord) -> None:
        keys_to_index: set[str] = set()

        # 1. Exact normalized name
        norm = record.name.normalized
        if norm and len(norm) >= 2:
            keys_to_index.add(norm)

        # 2. Suffix-normalized tokens joined
        suffix_norm = " ".join(record.name.tokens_suffix_norm)
        if suffix_norm and len(suffix_norm) >= 2 and suffix_norm != norm:
            keys_to_index.add(suffix_norm)

        for k in keys_to_index:
            self._index[k].append(idx)

    def finalize(self) -> None:
        self._total_keys = len(self._index)

    def query(self, query_record: BusinessRecord) -> list[int]:
        results: list[int] = []
        seen: set[int] = set()

        # Look up exact normalized
        norm = query_record.name.normalized
        if norm and norm in self._index:
            for idx in self._index[norm]:
                if idx not in seen:
                    seen.add(idx)
                    results.append(idx)

        # Look up suffix-normalized
        suffix_norm = " ".join(query_record.name.tokens_suffix_norm)
        if suffix_norm and suffix_norm != norm and suffix_norm in self._index:
            for idx in self._index[suffix_norm]:
                if idx not in seen:
                    seen.add(idx)
                    results.append(idx)

        return results

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_keys": len(self._index),
        }
