"""
address_token.py — Blocking Strategy E: Address Token Index.

Builds an inverted index over distinctive normalized address tokens.
Essential for:
- Non-Latin name pairs (where S1 is Latin but S2/S3 is Devanagari, Tamil, etc.).
- Corrupted name pairs (where business_name is corrupted but address is intact).
- Minor name variations with matching locality / street names.

Features:
- Common address stopword filtering (street, road, ave, suite, etc.).
- Document frequency capping to prune city/state names that appear too frequently.
- Rare-token prioritization at query time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.blocking.base import BaseBlocker
from src.normalization.record import BusinessRecord

ADDRESS_STOP_TOKENS: frozenset[str] = frozenset(
    {
        # Generic thoroughfare types
        "street", "st", "saint", "road", "rd", "avenue", "ave", "boulevard",
        "blvd", "lane", "ln", "drive", "dr", "way", "highway", "hwy",
        "route", "rt", "court", "ct", "circle", "cir", "place", "pl",
        "terrace", "ter", "parkway", "pkwy", "trail", "trl", "square", "sq",
        # Generic secondary units
        "suite", "ste", "apt", "apartment", "unit", "floor", "fl",
        "building", "bldg", "room", "rm", "box", "po", "pob", "pobox",
        # Common prepositions / descriptors
        "near", "opp", "opposite", "behind", "beside", "above", "below",
        "north", "south", "east", "west", "n", "s", "e", "w",
        "upper", "lower", "main", "cross", "old", "new",
        # Common Indian generic address terms (very high frequency)
        "plot", "house", "no", "hno", "kh", "khasra",
        # English structural words
        "the", "and", "of", "in", "at", "by", "for", "to", "from",
    }
)


class AddressTokenBlocker(BaseBlocker):
    """
    Retrieves candidates sharing informative street, locality, or landmark tokens.
    """

    def __init__(
        self,
        name: str = "address_token",
        min_token_len: int = 3,
        max_df: int = 2500,
        max_candidates_per_query: int = 50,
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
        if tok in ADDRESS_STOP_TOKENS:
            return False
        if tok.isdigit():
            # Pure digits handled by structural blocker
            return False
        return True

    def index_record(self, idx: int, record: BusinessRecord) -> None:
        seen: set[str] = set()
        for tok in record.address.alpha_tokens:
            tok_lower = tok.lower()
            if self._is_informative_token(tok_lower) and tok_lower not in seen:
                seen.add(tok_lower)
                self._index[tok_lower].append(idx)

    def finalize(self) -> None:
        tokens_to_prune = [
            tok for tok, postings in self._index.items()
            if len(postings) > self.max_df
        ]
        for tok in tokens_to_prune:
            del self._index[tok]
            self._pruned_count += 1

    def query(self, query_record: BusinessRecord) -> list[int]:
        query_tokens = [
            tok.lower() for tok in query_record.address.alpha_tokens
            if self._is_informative_token(tok.lower()) and tok.lower() in self._index
        ]
        if not query_tokens:
            return []

        # Sort by posting list length (rarest first)
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
            "pruned_tokens": self._pruned_count,
            "max_df": self.max_df,
        }
