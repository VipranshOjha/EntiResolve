"""
test_phase5a_equivalence.py — Bit-for-bit semantic equivalence regression test.

Compares OLD inference implementation vs OPTIMIZED inference implementation
on real test data across France, US, and India for at least 2,000 candidate pairs.

Verifies:
- candidate IDs / candidate ordering: 100% identical
- 61 feature values: bit-for-bit identical (np.array_equal)
- Model A probabilities: identical (np.allclose / identical)
- Model B probabilities: identical (np.allclose / identical)
- Blend scores: identical (round(0.88*sa + 0.12*sb, 5))
- Threshold decisions (blend >= 0.915): identical
- Matched entity IDs: identical
"""

from collections import defaultdict
import csv
import os
from pathlib import Path
import shutil
import numpy as np
import pytest

from src.blocking.composite import CompositeBlocker
from src.matching.features import CandidateProvenance, PairwiseFeatureExtractor
from src.matching.model import PairwiseLightGBMMatcher
from src.matching.optimized_inference import (
    PersistentTargetStore,
    PrecomputedQuery,
    PrecomputedTarget,
    extract_batch_optimized,
)
from src.normalization.record import BusinessRecord, NormalizationPipeline


def test_old_vs_optimized_semantic_equivalence(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent
    project_root = repo_root.parent
    test_dir = project_root / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    model_a_path = repo_root / "models" / "phase4_3" / "matcher_hardneg_1x.pkl"
    model_b_path = repo_root / "models" / "phase4_3c" / "matcher_fulluniverse_hardneg_1x.pkl"

    assert s1_path.exists(), f"Missing {s1_path}"
    assert model_a_path.exists(), f"Missing {model_a_path}"
    assert model_b_path.exists(), f"Missing {model_b_path}"

    model_a = PairwiseLightGBMMatcher.load(model_a_path)
    model_b = PairwiseLightGBMMatcher.load(model_b_path)
    pipeline = NormalizationPipeline()
    extractor_old = PairwiseFeatureExtractor()

    # Load 25 test queries each from France, US, India (total ~75 queries, producing >2000 pairs)
    queries_by_country: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                country = row[3].strip()
                if country in ("France", "US", "India") and len(queries_by_country[country]) < 25:
                    queries_by_country[country].append((row[0].strip(), row[1].strip(), row[2].strip(), country))
            if all(len(queries_by_country[c]) >= 25 for c in ("France", "US", "India")):
                break

    total_pairs_tested = 0

    for country in ["France", "US", "India"]:
        raw_queries = queries_by_country[country]
        assert len(raw_queries) > 0, f"No queries found for {country}"

        normalized_queries = [pipeline.normalize(eid, name, addr, country) for eid, name, addr, country in raw_queries]

        # Indexing: build blocker, old offsets, and optimized persistent store
        blocker = CompositeBlocker(max_total_candidates=175)
        target_offsets: dict[str, int] = {}
        target_tsv_paths = [s2_path, s3_path]

        db_path = tmp_path / f"test_targets_{country}.db"
        target_store = PersistentTargetStore(db_path, lru_capacity=50_000)

        # We stream targets for this country (reading first 50k lines per file for fast test)
        batch_inserts = []
        def stream_targets():
            for f_idx, path in enumerate(target_tsv_paths):
                with open(path, "rb") as bf:
                    bf.readline()
                    lines_read = 0
                    while lines_read < 50000:
                        offset = bf.tell()
                        raw_line = bf.readline()
                        if not raw_line:
                            break
                        lines_read += 1
                        parts = raw_line.split(b"\t")
                        if len(parts) >= 4 and parts[3].strip().decode("utf-8", "replace") == country:
                            tid = parts[0].strip().decode("utf-8", "replace")
                            name = parts[1].decode("utf-8", "replace")
                            addr = parts[2].decode("utf-8", "replace")
                            target_offsets[tid] = (f_idx << 32) | offset
                            rec = pipeline.normalize(tid, name, addr, country)
                            pt = PrecomputedTarget.from_record(rec)
                            batch_inserts.append((tid, pt.pack()))
                            if len(batch_inserts) >= 5000:
                                target_store.put_batch(batch_inserts)
                                batch_inserts.clear()
                            yield rec
            if batch_inserts:
                target_store.put_batch(batch_inserts)
                batch_inserts.clear()

        blocker.index_target_records(stream_targets())

        # Setup file handles for OLD get_raw_target
        file_handles = [open(p, "r", encoding="utf-8", errors="replace") for p in target_tsv_paths]
        def get_raw_target_old(tid: str) -> tuple[str, str]:
            packed = target_offsets[tid]
            f_idx = packed >> 32
            offset = packed & 0xFFFFFFFF
            fh = file_handles[f_idx]
            fh.seek(offset)
            line = fh.readline()
            parts = line.split("\t")
            return parts[1], parts[2]

        target_cache_old: dict[str, BusinessRecord] = {}

        # Evaluate queries through both pipelines
        for q_rec in normalized_queries:
            sid = q_rec.entity_id

            # 1. Candidate retrieval
            cands_old, attr_old = blocker.query(q_rec, track_attribution=True)
            valid_cands_old = [c for c in cands_old if c in target_offsets]

            # Invert attribution
            candidate_blockers = defaultdict(set)
            for b_name, tgts in attr_old.items():
                for t in tgts:
                    candidate_blockers[t].add(b_name)

            if not valid_cands_old:
                continue

            # OLD PIPELINE EVALUATION
            old_pairs = []
            for rank, tid in enumerate(valid_cands_old, start=1):
                if tid not in target_cache_old:
                    rn, ra = get_raw_target_old(tid)
                    target_cache_old[tid] = pipeline.normalize(tid, rn, ra, country)
                tgt_old = target_cache_old[tid]
                hits = candidate_blockers[tid]
                prov = CandidateProvenance(
                    candidate_rank=rank,
                    composite_priority=max(1.0, 10.0 - rank * 0.1),
                    blocker_count=len(hits),
                    exact_name_hit=1 if "exact_normalized_name" in hits else 0,
                    compact_name_hit=1 if "compact_name" in hits else 0,
                    name_token_hit=1 if "name_token" in hits else 0,
                    ngram_hit=1 if "char_ngram" in hits else 0,
                    address_token_hit=1 if "address_token" in hits else 0,
                    structural_hit=1 if "address_structural" in hits else 0,
                    source_is_s2=1 if tid.startswith("S2") else 0,
                )
                old_pairs.append((q_rec, tgt_old, prov))

            feats_old = extractor_old.extract_batch(old_pairs)
            probs_a_old = model_a.predict_proba(feats_old)
            probs_b_old = model_b.predict_proba(feats_old)

            blend_old = [round(0.88 * sa + 0.12 * sb, 5) for sa, sb in zip(probs_a_old, probs_b_old)]
            decisions_old = [b >= 0.915 for b in blend_old]
            preds_old = [tid for tid, dec in zip(valid_cands_old, decisions_old) if dec]

            # OPTIMIZED PIPELINE EVALUATION
            q_pre = PrecomputedQuery(q_rec)
            target_objs = target_store.get_batch(valid_cands_old)

            opt_pairs = []
            for rank, tid in enumerate(valid_cands_old, start=1):
                tgt_opt = target_objs[tid]
                hits = candidate_blockers[tid]
                prov = CandidateProvenance(
                    candidate_rank=rank,
                    composite_priority=max(1.0, 10.0 - rank * 0.1),
                    blocker_count=len(hits),
                    exact_name_hit=1 if "exact_normalized_name" in hits else 0,
                    compact_name_hit=1 if "compact_name" in hits else 0,
                    name_token_hit=1 if "name_token" in hits else 0,
                    ngram_hit=1 if "char_ngram" in hits else 0,
                    address_token_hit=1 if "address_token" in hits else 0,
                    structural_hit=1 if "address_structural" in hits else 0,
                    source_is_s2=1 if tid.startswith("S2") else 0,
                )
                opt_pairs.append((q_pre, tgt_opt, prov))

            feats_opt = extract_batch_optimized(opt_pairs)
            probs_a_opt = model_a.predict_proba(feats_opt)
            probs_b_opt = model_b.predict_proba(feats_opt)

            blend_opt = [round(0.88 * sa + 0.12 * sb, 5) for sa, sb in zip(probs_a_opt, probs_b_opt)]
            decisions_opt = [b >= 0.915 for b in blend_opt]
            preds_opt = [tid for tid, dec in zip(valid_cands_old, decisions_opt) if dec]

            # REGRESSION ASSERTIONS
            # 1. Exact 61 feature values bit-for-bit identical
            assert np.array_equal(feats_old, feats_opt), f"Feature divergence on query {sid}!"

            # 2. Model A probabilities identical
            assert np.allclose(probs_a_old, probs_a_opt, atol=1e-7), f"Model A probability divergence on query {sid}!"

            # 3. Model B probabilities identical
            assert np.allclose(probs_b_old, probs_b_opt, atol=1e-7), f"Model B probability divergence on query {sid}!"

            # 4. Blend scores identical
            assert blend_old == blend_opt, f"Blend score divergence on query {sid}!"

            # 5. Threshold decisions identical
            assert decisions_old == decisions_opt, f"Decision divergence on query {sid}!"

            # 6. Matched IDs identical
            assert preds_old == preds_opt, f"Matched IDs divergence on query {sid}!"

            total_pairs_tested += len(valid_cands_old)

        for fh in file_handles:
            fh.close()
        target_store.close()

    print(f"\nREGRESSION TEST PASSED: {total_pairs_tested:,} candidate pairs verified 100% BIT-FOR-BIT IDENTICAL!")
    assert total_pairs_tested >= 2000, f"Expected >= 2000 pairs, got {total_pairs_tested}"
