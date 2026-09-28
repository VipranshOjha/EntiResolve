"""
address_structural.py — Blocking Strategies F & G: Structural Address & Compound Blocks.

Generates high-precision, cross-view compound retrieval keys:
- house_number + postal_code (e.g. "110001:570/13" or "62701:123")
- house_number + first_name_token (e.g. "3315:payne")
- postal_code + first_name_token (e.g. "27262:urology")
- house_number + rare_address_token (e.g. "85:ticonderoga" or "6(29):mylapore")
- standalone postal_code (with frequency capping)

These compound blocks are the silver bullet for:
- Non-Latin business names paired with Latin queries
- Severe spelling corruptions (e.g. "Drxkor" vs "Maure Williams")
- Typos and token transpositions
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.blocking.address_token import ADDRESS_STOP_TOKENS
from src.blocking.base import BaseBlocker
from src.blocking.name_token import NAME_STOP_TOKENS
from src.normalization.record import BusinessRecord


class AddressStructuralBlocker(BaseBlocker):
    """
    Retrieves candidates using compound keys combining structural address
    signals (house number, postal code) with informative address and name tokens.
    """

    def __init__(
        self,
        name: str = "address_structural",
        max_df: int = 1500,
        max_candidates_per_query: int = 50,
    ) -> None:
        super().__init__(name=name)
        self.max_df = max_df
        self.max_candidates_per_query = max_candidates_per_query
        self._index: dict[str, list[int]] = defaultdict(list)
        self._pruned_count = 0

    def reset(self) -> None:
        self._index.clear()
        self._pruned_count = 0

    def _extract_compound_keys(self, record: BusinessRecord) -> set[str]:
        keys: set[str] = set()

        house_nums = record.address.house_number_candidates
        postal_codes = record.address.postal_candidates
        num_tokens = record.address.numeric_tokens

        # Informative name tokens
        name_tokens = [
            t for t in record.name.tokens_suffix_norm
            if len(t) >= 3 and t not in NAME_STOP_TOKENS and not t.isdigit()
        ]
        first_name_tok = name_tokens[0] if name_tokens else ""

        # Informative address tokens
        addr_tokens = [
            t.lower() for t in record.address.alpha_tokens
            if len(t) >= 4 and t.lower() not in ADDRESS_STOP_TOKENS
        ]

        # 1. House number + Postal code
        for h in house_nums[:2]:
            for p in postal_codes[:2]:
                keys.add(f"h_p:{h}:{p}")

        # 2. House number + First name token
        if first_name_tok:
            for h in house_nums[:2]:
                keys.add(f"h_n:{h}:{first_name_tok}")

        # 3. Postal code + First name token
        if first_name_tok:
            for p in postal_codes[:2]:
                keys.add(f"p_n:{p}:{first_name_tok}")

        # 4. House number + Informative address token (vital for Non-Latin / Corrupted names)
        for h in (house_nums or num_tokens)[:2]:
            for a in addr_tokens[:3]:
                keys.add(f"h_a:{h}:{a}")

        # 5. Standalone postal code (if available)
        for p in postal_codes[:1]:
            keys.add(f"post:{p}")

        return keys

    def index_record(self, idx: int, record: BusinessRecord) -> None:
        keys = self._extract_compound_keys(record)
        for k in keys:
            self._index[k].append(idx)

    def finalize(self) -> None:
        keys_to_prune = [
            k for k, postings in self._index.items()
            if len(postings) > self.max_df
        ]
        for k in keys_to_prune:
            del self._index[k]
            self._pruned_count += 1

    def query(self, query_record: BusinessRecord) -> list[int]:
        keys = [
            k for k in self._extract_compound_keys(query_record)
            if k in self._index
        ]
        if not keys:
            return []

        # Sort keys by posting list length (rarest first = most discriminative)
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
