"""
optimized_inference.py — Ultra-High Performance Inference Engine for Entity Resolution.

Preserves EXACT SEMANTICS of:
- CompositeBlocker(max_total_candidates=175)
- PairwiseFeatureExtractor (61 features)
- Model A (models/phase4_3/matcher_hardneg_1x.pkl)
- Model B (models/phase4_3c/matcher_fulluniverse_hardneg_1x.pkl)
- Blend: S = 0.88 * P_A + 0.12 * P_B (round(..., 5))
- Threshold: S >= 0.915

Eliminates:
- Disk seeks and readline on raw text TSV files during candidate evaluation.
- Redundant target re-normalization during candidate evaluation.
- Redundant query re-normalization and repeated string metric setups.
- Memory bloat by using SQLite-backed disk store + bounded in-memory LRU cache.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import marshal
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Sequence
import numpy as np

from src.matching.cross_script import CrossScriptAligner
from src.matching.features import CandidateProvenance, ALL_FEATURE_NAMES
from src.matching.similarity import (
    char_ngram_jaccard,
    house_number_agreement,
    jaro_winkler_similarity,
    levenshtein_ratio,
    numeric_overlap,
    postal_code_agreement,
    token_containment,
    token_jaccard,
    token_overlap_coefficient,
)
from src.normalization.record import BusinessRecord


class PrecomputedQuery:
    """Precomputed query representation reused across all candidate targets."""
    __slots__ = (
        "entity_id", "norm_name", "suffix_str", "compact", "compact_no_tld",
        "tokens", "tokens_set", "quality", "dominant_script",
        "lat", "non_lat", "mixed", "web", "len_name", "len_compact",
        "norm_addr", "addr_tokens", "addr_tokens_set", "numeric_tokens",
        "house_cands", "postal_cands", "locality_tokens",
        "len_addr", "addr_empty", "has_po_box", "record"
    )

    def __init__(self, s1: BusinessRecord):
        self.record = s1
        self.entity_id = s1.entity_id
        s1_n = s1.name
        s1_a = s1.address
        self.norm_name = s1_n.normalized
        self.suffix_str = " ".join(s1_n.tokens_suffix_norm)
        self.compact = s1_n.compact
        self.compact_no_tld = s1_n.compact_no_tld
        self.tokens = s1_n.tokens
        self.tokens_set = set(s1_n.tokens)
        self.quality = s1_n.quality
        self.dominant_script = s1_n.script.dominant_script
        self.lat = 1.0 if s1_n.script.latin_ratio > 0.0 or "LATIN" in s1_n.script.scripts_present else 0.0
        self.non_lat = 1.0 if s1_n.script.is_non_latin or s1_n.script.is_mixed_script else 0.0
        self.mixed = 1.0 if s1_n.script.is_mixed_script else 0.0
        self.web = 1.0 if s1_n.quality == "website" else 0.0
        self.len_name = len(s1_n.normalized)
        self.len_compact = len(s1_n.compact)

        self.norm_addr = s1_a.normalized
        self.addr_tokens = s1_a.tokens
        self.addr_tokens_set = set(s1_a.tokens)
        self.numeric_tokens = s1_a.numeric_tokens
        self.house_cands = s1_a.house_number_candidates
        self.postal_cands = s1_a.postal_candidates
        self.locality_tokens = [t for t in s1_a.tokens if not t.isdigit()]
        self.len_addr = len(s1_a.normalized)
        self.addr_empty = 1.0 if len(s1_a.tokens) == 0 or s1_a.quality == "missing" else 0.0
        self.has_po_box = 1.0 if s1_a.has_po_box else 0.0


class PrecomputedTarget:
    """Precomputed target representation stored in memory-efficient cache."""
    __slots__ = (
        "entity_id", "source_is_s2", "norm_name", "suffix_str", "compact", "compact_no_tld",
        "tokens", "tokens_set", "quality", "dominant_script",
        "lat", "non_lat", "mixed", "web", "len_name", "len_compact",
        "norm_addr", "addr_tokens", "addr_tokens_set", "numeric_tokens",
        "house_cands", "postal_cands", "locality_tokens",
        "len_addr", "addr_empty", "has_po_box"
    )

    def __init__(
        self,
        entity_id: str,
        source_is_s2: int,
        norm_name: str,
        suffix_str: str,
        compact: str,
        compact_no_tld: str,
        tokens: Sequence[str],
        quality: str,
        dominant_script: str,
        lat: float,
        non_lat: float,
        mixed: float,
        web: float,
        len_name: int,
        len_compact: int,
        norm_addr: str,
        addr_tokens: Sequence[str],
        numeric_tokens: Sequence[str],
        house_cands: Sequence[str],
        postal_cands: Sequence[str],
        locality_tokens: Sequence[str],
        len_addr: int,
        addr_empty: float,
        has_po_box: float,
    ):
        self.entity_id = entity_id
        self.source_is_s2 = source_is_s2
        self.norm_name = norm_name
        self.suffix_str = suffix_str
        self.compact = compact
        self.compact_no_tld = compact_no_tld
        self.tokens = tokens
        self.tokens_set = set(tokens)
        self.quality = quality
        self.dominant_script = dominant_script
        self.lat = lat
        self.non_lat = non_lat
        self.mixed = mixed
        self.web = web
        self.len_name = len_name
        self.len_compact = len_compact
        self.norm_addr = norm_addr
        self.addr_tokens = addr_tokens
        self.addr_tokens_set = set(addr_tokens)
        self.numeric_tokens = numeric_tokens
        self.house_cands = house_cands
        self.postal_cands = postal_cands
        self.locality_tokens = locality_tokens
        self.len_addr = len_addr
        self.addr_empty = addr_empty
        self.has_po_box = has_po_box

    @classmethod
    def from_record(cls, target: BusinessRecord) -> PrecomputedTarget:
        """Construct from a freshly normalized BusinessRecord."""
        entity_id = target.entity_id
        source_is_s2 = 1 if target.source == "s2" or entity_id.startswith("S2") else 0
        tgt_n = target.name
        tgt_a = target.address

        lat = 1.0 if tgt_n.script.latin_ratio > 0.0 or "LATIN" in tgt_n.script.scripts_present else 0.0
        non_lat = 1.0 if tgt_n.script.is_non_latin or tgt_n.script.is_mixed_script else 0.0
        mixed = 1.0 if tgt_n.script.is_mixed_script else 0.0
        web = 1.0 if tgt_n.quality == "website" else 0.0
        locality = [t for t in tgt_a.tokens if not t.isdigit()]
        addr_empty = 1.0 if len(tgt_a.tokens) == 0 or tgt_a.quality == "missing" else 0.0
        has_po = 1.0 if tgt_a.has_po_box else 0.0

        return cls(
            entity_id=entity_id,
            source_is_s2=source_is_s2,
            norm_name=tgt_n.normalized,
            suffix_str=" ".join(tgt_n.tokens_suffix_norm),
            compact=tgt_n.compact,
            compact_no_tld=tgt_n.compact_no_tld,
            tokens=tuple(tgt_n.tokens),
            quality=tgt_n.quality,
            dominant_script=tgt_n.script.dominant_script,
            lat=lat,
            non_lat=non_lat,
            mixed=mixed,
            web=web,
            len_name=len(tgt_n.normalized),
            len_compact=len(tgt_n.compact),
            norm_addr=tgt_a.normalized,
            addr_tokens=tuple(tgt_a.tokens),
            numeric_tokens=tuple(tgt_a.numeric_tokens),
            house_cands=tuple(tgt_a.house_number_candidates),
            postal_cands=tuple(tgt_a.postal_candidates),
            locality_tokens=tuple(locality),
            len_addr=len(tgt_a.normalized),
            addr_empty=addr_empty,
            has_po_box=has_po,
        )

    def pack(self) -> bytes:
        """Serialize to compact marshal bytes."""
        payload = (
            self.entity_id,
            self.source_is_s2,
            self.norm_name,
            self.suffix_str,
            self.compact,
            self.compact_no_tld,
            tuple(self.tokens),
            self.quality,
            self.dominant_script,
            self.lat,
            self.non_lat,
            self.mixed,
            self.web,
            self.len_name,
            self.len_compact,
            self.norm_addr,
            tuple(self.addr_tokens),
            tuple(self.numeric_tokens),
            tuple(self.house_cands),
            tuple(self.postal_cands),
            tuple(self.locality_tokens),
            self.len_addr,
            self.addr_empty,
            self.has_po_box,
        )
        return marshal.dumps(payload)

    @classmethod
    def unpack(cls, b: bytes) -> PrecomputedTarget:
        """Deserialize from compact marshal bytes."""
        p = marshal.loads(b)
        return cls(*p)


class PersistentTargetStore:
    """
    Two-tier persistent target store:
    - Tier 1: Fast bounded in-memory LRU cache (e.g. 100,000 targets, ~150 MB RAM).
    - Tier 2: Disk-backed SQLite database with bulk query operations.
    """

    def __init__(self, db_path: str | Path, lru_capacity: int = 100_000):
        self.db_path = str(db_path)
        self.lru_capacity = lru_capacity
        self.lru: OrderedDict[str, PrecomputedTarget] = OrderedDict()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self._conn.cursor()
        cur.execute("PRAGMA synchronous = OFF")
        cur.execute("PRAGMA journal_mode = OFF")
        cur.execute("CREATE TABLE IF NOT EXISTS targets (tid TEXT PRIMARY KEY, data BLOB)")
        self._conn.commit()

    def put_batch(self, items: list[tuple[str, bytes]]) -> None:
        """Bulk insert precomputed targets into the database."""
        cur = self._conn.cursor()
        cur.executemany("INSERT OR REPLACE INTO targets VALUES (?, ?)", items)
        self._conn.commit()

    def get_batch(self, tids: Sequence[str]) -> dict[str, PrecomputedTarget]:
        """
        Batch retrieve precomputed targets.
        Hits LRU cache first; bulk queries SQLite for misses and populates LRU.
        """
        results: dict[str, PrecomputedTarget] = {}
        missing_tids: list[str] = []

        for tid in tids:
            if tid in self.lru:
                self.lru.move_to_end(tid)
                results[tid] = self.lru[tid]
            else:
                missing_tids.append(tid)

        if missing_tids:
            cur = self._conn.cursor()
            # SQLite limit for variables is 999 in default configurations
            chunk_size = 999
            for i in range(0, len(missing_tids), chunk_size):
                chunk = missing_tids[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                cur.execute(f"SELECT tid, data FROM targets WHERE tid IN ({placeholders})", chunk)
                for r_tid, blob in cur.fetchall():
                    target_obj = PrecomputedTarget.unpack(blob)
                    results[r_tid] = target_obj
                    self.lru[r_tid] = target_obj
                    if len(self.lru) > self.lru_capacity:
                        self.lru.popitem(last=False)

        return results

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


class ThreadSafeTargetStore:
    """
    Thread-safe persistent target store using thread-local SQLite connections.
    Eliminates all cross-thread lock contention while sharing the on-disk database.
    """

    def __init__(self, db_path: str | Path, shared_cache: dict[str, PrecomputedTarget] | None = None):
        self.db_path = str(db_path)
        self.shared_cache = shared_cache if shared_cache is not None else {}
        self._local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA journal_mode = OFF")
            conn.execute("PRAGMA query_only = ON")
            self._local.conn = conn
        return self._local.conn

    def get_batch(self, tids: Sequence[str]) -> dict[str, PrecomputedTarget]:
        results: dict[str, PrecomputedTarget] = {}
        missing: list[str] = []
        for tid in tids:
            if tid in self.shared_cache:
                results[tid] = self.shared_cache[tid]
            else:
                missing.append(tid)

        if missing:
            conn = self._get_conn()
            cur = conn.cursor()
            chunk_size = 999
            for i in range(0, len(missing), chunk_size):
                chunk = missing[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                cur.execute(f"SELECT tid, data FROM targets WHERE tid IN ({placeholders})", chunk)
                for r_tid, blob in cur.fetchall():
                    target_obj = PrecomputedTarget.unpack(blob)
                    results[r_tid] = target_obj
        return results

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None


def extract_features_single(
    q: PrecomputedQuery,
    t: PrecomputedTarget,
    prov: CandidateProvenance,
    cs_sim: float = 0.0,
) -> list[float]:
    """
    Extracts the exact 61 pairwise features bit-for-bit identical to
    PairwiseFeatureExtractor.extract_features.
    """
    # 1. NAME FEATURES
    name_exact_norm = 1.0 if q.norm_name and q.norm_name == t.norm_name else 0.0
    name_exact_suffix = 1.0 if q.suffix_str and q.suffix_str == t.suffix_str else 0.0
    name_compact_eq = 1.0 if q.compact and q.compact == t.compact else 0.0
    name_compact_no_tld_eq = 1.0 if q.compact_no_tld and q.compact_no_tld == t.compact_no_tld else 0.0

    name_lev = levenshtein_ratio(q.norm_name, t.norm_name)
    name_jw = jaro_winkler_similarity(q.norm_name, t.norm_name)
    name_ngram = char_ngram_jaccard(q.norm_name, t.norm_name, n=3)

    name_tok_jaccard = token_jaccard(q.tokens, t.tokens)
    name_tok_overlap = token_overlap_coefficient(q.tokens, t.tokens)
    name_tok_containment = token_containment(q.tokens, t.tokens)
    name_tok_count_diff = float(abs(len(q.tokens) - len(t.tokens)))

    len1, len2 = q.len_name, t.len_name
    name_char_len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 1.0

    clen1, clen2 = q.len_compact, t.len_compact
    name_compact_len_ratio = (min(clen1, clen2) / max(clen1, clen2)) if max(clen1, clen2) > 0 else 1.0

    name_qual_both_good = 1.0 if q.quality == "good" and t.quality == "good" else 0.0
    name_qual_agree = 1.0 if q.quality == t.quality else 0.0
    name_script_agree = 1.0 if q.dominant_script == t.dominant_script else 0.0

    both_latin = 1.0 if (q.lat and t.lat and not (q.non_lat or t.non_lat)) else 0.0
    cross_script = 1.0 if ((q.lat and t.non_lat) or (q.non_lat and t.lat)) else 0.0
    both_web = 1.0 if (q.web and t.web) else 0.0

    # 2. ADDRESS FEATURES
    addr_lev = levenshtein_ratio(q.norm_addr, t.norm_addr)
    addr_tok_jaccard = token_jaccard(q.addr_tokens, t.addr_tokens)
    addr_tok_overlap = token_overlap_coefficient(q.addr_tokens, t.addr_tokens)
    addr_ngram = char_ngram_jaccard(q.norm_addr, t.norm_addr, n=3)

    num_jaccard, num_overlap_cnt = numeric_overlap(q.numeric_tokens, t.numeric_tokens)
    house_agree = house_number_agreement(q.house_cands, t.house_cands)
    postal_agree = postal_code_agreement(q.postal_cands, t.postal_cands)
    addr_locality_overlap = token_overlap_coefficient(q.locality_tokens, t.locality_tokens)

    addr_comp_diff = float(abs(len(q.addr_tokens) - len(t.addr_tokens)))
    alen1, alen2 = q.len_addr, t.len_addr
    addr_len_ratio = (min(alen1, alen2) / max(alen1, alen2)) if max(alen1, alen2) > 0 else 1.0

    addr_both_empty = 1.0 if (q.addr_empty and t.addr_empty) else 0.0
    addr_both_po = 1.0 if (q.has_po_box and t.has_po_box) else 0.0

    # 3. INTERACTION FEATURES
    inter_name_addr = name_jw * addr_tok_jaccard
    inter_qual = name_qual_both_good * (1.0 if not (q.addr_empty or t.addr_empty) else 0.0)
    inter_script_addr = name_script_agree * addr_tok_jaccard
    inter_house_name = house_agree * name_jw
    inter_postal_name = postal_agree * name_jw
    inter_exact_name_numeric = name_exact_norm * (1.0 if num_overlap_cnt > 0 else 0.0)

    # 4. BLOCKER PROVENANCE FEATURES
    prov_exact = float(prov.exact_name_hit)
    prov_compact = float(prov.compact_name_hit)
    prov_name_tok = float(prov.name_token_hit)
    prov_ngram = float(prov.ngram_hit)
    prov_addr_tok = float(prov.address_token_hit)
    prov_struct = float(prov.structural_hit)
    prov_count = float(prov.blocker_count)
    prov_priority = float(prov.composite_priority)
    prov_rank = float(prov.candidate_rank)
    prov_source = float(prov.source_is_s2)

    return [
        name_exact_norm, name_exact_suffix, name_compact_eq, name_compact_no_tld_eq,
        name_lev, name_jw, name_ngram, name_tok_jaccard, name_tok_overlap,
        name_tok_containment, name_tok_count_diff, name_char_len_ratio,
        name_compact_len_ratio, name_qual_both_good, name_qual_agree,
        name_script_agree, q.lat, t.lat, q.non_lat, t.non_lat, both_latin,
        cross_script, q.mixed, t.mixed, q.web, t.web, both_web,
        addr_lev, addr_tok_jaccard, addr_tok_overlap, addr_ngram,
        num_jaccard, float(num_overlap_cnt), house_agree, postal_agree,
        addr_locality_overlap, addr_comp_diff, addr_len_ratio,
        q.addr_empty, t.addr_empty, addr_both_empty,
        q.has_po_box, t.has_po_box, addr_both_po,
        inter_name_addr, inter_qual, inter_script_addr, inter_house_name,
        inter_postal_name, inter_exact_name_numeric,
        prov_exact, prov_compact, prov_name_tok, prov_ngram, prov_addr_tok,
        prov_struct, prov_count, prov_priority, prov_rank, prov_source,
        cs_sim,
    ]


def extract_batch_optimized(
    pairs: Sequence[tuple[PrecomputedQuery, PrecomputedTarget, CandidateProvenance]],
    cross_script_aligner: CrossScriptAligner | None = None,
) -> np.ndarray:
    """
    Vectorized batch feature matrix construction.
    Returns np.ndarray of shape (N, 61), dtype=np.float32.
    """
    N = len(pairs)
    matrix = np.empty((N, 61), dtype=np.float32)
    for i, (q, t, prov) in enumerate(pairs):
        cs_sim = 0.0
        if cross_script_aligner is not None:
            cs_sim = cross_script_aligner.compute_similarity(q.record, t)  # type: ignore
        matrix[i] = extract_features_single(q, t, prov, cs_sim=cs_sim)
    return matrix
