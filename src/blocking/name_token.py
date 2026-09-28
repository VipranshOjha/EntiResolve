"""
name_token.py — Blocking Strategy C: Name Token Index.

Builds an inverted index over individual informative name tokens.
Features:
- Stop-token filtering (common legal entity types and English/Indic stop words).
- Document frequency capping (prunes high-frequency tokens exceeding max_df).
- Rare-token prioritization at query time: queries rarest tokens first to maximize
  discriminative retrieval power while capping per-query candidates.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.blocking.base import BaseBlocker
from src.normalization.common import LEGAL_SUFFIX_MAP
from src.normalization.record import BusinessRecord

# Common generic stop tokens in company names across US and India
NAME_STOP_TOKENS: frozenset[str] = frozenset(
    {
        # Common English articles / prepositions / conjunctions
        "the", "and", "of", "in", "for", "at", "by", "on", "to", "with",
        "a", "an", "as",
        # Generic business words & legal entity designations
        "company", "co", "corporation", "corp", "incorporated", "inc",
        "limited", "ltd", "pvt", "private", "llc", "llp", "l.l.c.", "l.l.p.",
        "enterprises", "enterprise", "group", "holdings", "holding",
        "services", "service", "solutions", "solution", "technologies",
        "technology", "industries", "industry", "international", "intl",
        "global", "associates", "partners", "management", "consulting",
        "center", "centre", "club", "store", "shop", "mart", "agency",
        "trading", "traders", "venture", "ventures", "systems", "system",
        "india", "usa", "us", "america", "national",
    }
    | set(LEGAL_SUFFIX_MAP.keys())
    | set(LEGAL_SUFFIX_MAP.values())
)


class NameTokenBlocker(BaseBlocker):
    """
    Retrieves candidates sharing informative business name tokens.
    """

    def __init__(
        self,
        name: str = "name_token",
        min_token_len: int = 3,
        max_df: int = 3000,
        max_candidates_per_query: int = 60,
    ) -> None:
        super().__init__(name=name)
        self.min_token_len = min_token_len
        self.max_df = max_df
        self.max_candidates_per_query = max_candidates_per_query
        self._index: dict[str, list[int]] = defaultdict(list)
        self._pruned_count = 0

    def reset(self) -> None:
        self._index.clear()
        self._pruned_count = 0

    def _is_informative_token(self, tok: str) -> bool:
        if len(tok) < self.min_token_len:
            return False
        if tok in NAME_STOP_TOKENS:
            return False
        if tok.isdigit():
            return False
        return True

    def index_record(self, idx: int, record: BusinessRecord) -> None:
        seen_tokens: set[str] = set()
        for tok in record.name.tokens_suffix_norm:
            if self._is_informative_token(tok) and tok not in seen_tokens:
                seen_tokens.add(tok)
                self._index[tok].append(idx)

    def finalize(self) -> None:
        """Prune tokens that exceed max_df to prevent pathological posting lists."""
        tokens_to_prune = [
            tok for tok, postings in self._index.items()
            if len(postings) > self.max_df
        ]
        for tok in tokens_to_prune:
            del self._index[tok]
            self._pruned_count += 1

    def query(self, query_record: BusinessRecord) -> list[int]:
        # Collect informative query tokens
        query_tokens = [
            tok for tok in query_record.name.tokens_suffix_norm
            if self._is_informative_token(tok) and tok in self._index
        ]

        if not query_tokens:
            return []

        # Sort tokens by posting list length (rarest first = most discriminative)
        query_tokens.sort(key=lambda t: len(self._index[t]))

        results: list[int] = []
        seen: set[int] = set()

        for tok in query_tokens:
            postings = self._index[tok]
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
            "total_tokens": len(self._index),
            "pruned_high_freq_tokens": self._pruned_count,
            "max_df": self.max_df,
        }
