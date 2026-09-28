"""
ngram_blocker.py — Blocking Strategy D: Character N-Gram & Prefix Retrieval.

Designed for memory-safety and scalability:
- Extracts compact prefix signatures (first 5 and 6 characters of compact name).
- Extracts character 4-grams for typo and spelling variation recovery.
- Uses strict max_df pruning (drops common n-grams) to prevent RAM explosion.
- Caps candidates per query to prevent pathological fan-out.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.blocking.base import BaseBlocker
from src.normalization.record import BusinessRecord


class CharNgramBlocker(BaseBlocker):
    """
    Retrieves candidates sharing character 4-grams or compact prefixes.
    Recovers typos, spelling variations, and phonetic distortions.
    """

    def __init__(
        self,
        name: str = "char_ngram",
        ngram_size: int = 4,
        prefix_len: int = 5,
        max_df: int = 1500,
        max_candidates_per_query: int = 50,
    ) -> None:
        super().__init__(name=name)
        self.ngram_size = ngram_size
        self.prefix_len = prefix_len
        self.max_df = max_df
        self.max_candidates_per_query = max_candidates_per_query
        self._index: dict[str, list[int]] = defaultdict(list)
        self._pruned_count = 0

    def reset(self) -> None:
        self._index.clear()
        self._pruned_count = 0

    def _extract_keys(self, record: BusinessRecord) -> set[str]:
        keys: set[str] = set()
        compact = record.name.compact
        if not compact or len(compact) < self.prefix_len:
            return keys

        # 1. Prefix key (tagged with 'p:')
        prefix = compact[: self.prefix_len]
        keys.add(f"p:{prefix}")

        # 2. Distinctive 4-grams from first 20 characters of compact name
        limit_str = compact[:24]
        n = self.ngram_size
        for i in range(len(limit_str) - n + 1):
            gram = limit_str[i : i + n]
            # Avoid purely repetitive grams like 'aaaa'
            if len(set(gram)) > 1:
                keys.add(f"g:{gram}")

        return keys

    def index_record(self, idx: int, record: BusinessRecord) -> None:
        keys = self._extract_keys(record)
        for k in keys:
            self._index[k].append(idx)

    def finalize(self) -> None:
        """Prune high-frequency keys exceeding max_df to bound memory and fan-out."""
        keys_to_prune = [
            k for k, postings in self._index.items()
            if len(postings) > self.max_df
        ]
        for k in keys_to_prune:
            del self._index[k]
            self._pruned_count += 1

    def query(self, query_record: BusinessRecord) -> list[int]:
        keys = [
            k for k in self._extract_keys(query_record)
            if k in self._index
        ]
        if not keys:
            return []

        # Sort keys by posting list length (rarest first)
        keys.sort(key=lambda k: len(self._index[k]))

        results: list[int] = []
        seen: set[int] = set()

        for k in keys:
            postings = self._index[k]
            for idx in postings:
                if idx not in seen:
                    seen.add(idx)
                    results.append(idx)
                    if len(results) >= self.max_candidates_per_query:
                        return results

        return results

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_keys": len(self._index),
            "pruned_keys": self._pruned_count,
            "max_df": self.max_df,
        }
