"""
compact_name.py — Blocking Strategy B: Compact Name & Website TLD Stripping.

Builds an inverted index over:
- record.name.compact (spaces and punctuation removed)
- record.name.compact_no_tld (website domains like .com, .org, .in, http:// stripped)

Recovers matches with formatting/spacing variations and website URLs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.blocking.base import BaseBlocker
from src.normalization.record import BusinessRecord


class CompactNameBlocker(BaseBlocker):
    """
    Retrieves candidates sharing the compact alphanumeric representation
    or website-stripped compact representation.
    """

    def __init__(self, name: str = "compact_name") -> None:
        super().__init__(name=name)
        self._index: dict[str, list[int]] = defaultdict(list)

    def reset(self) -> None:
        self._index.clear()

    def index_record(self, idx: int, record: BusinessRecord) -> None:
        keys_to_index: set[str] = set()

        # 1. Compact representation
        compact = record.name.compact
        if compact and len(compact) >= 3:
            keys_to_index.add(compact)

        # 2. Compact without website TLD (e.g. siiainvestments.com -> siiainvestments)
        compact_no_tld = record.name.compact_no_tld
        if compact_no_tld and len(compact_no_tld) >= 3 and compact_no_tld != compact:
            keys_to_index.add(compact_no_tld)

        for k in keys_to_index:
            self._index[k].append(idx)

    def query(self, query_record: BusinessRecord) -> list[int]:
        results: list[int] = []
        seen: set[int] = set()

        keys_to_query: list[str] = []
        compact = query_record.name.compact
        if compact and len(compact) >= 3:
            keys_to_query.append(compact)

        compact_no_tld = query_record.name.compact_no_tld
        if compact_no_tld and len(compact_no_tld) >= 3 and compact_no_tld != compact:
            keys_to_query.append(compact_no_tld)

        for k in keys_to_query:
            if k in self._index:
                for idx in self._index[k]:
                    if idx not in seen:
                        seen.add(idx)
                        results.append(idx)

        return results

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_keys": len(self._index),
        }
