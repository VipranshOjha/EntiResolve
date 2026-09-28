#!/usr/bin/env python3
"""
predict.py — End-to-End Test Inference & Reproduction CLI
Project: EntiResolve — Large-Scale Business Entity Resolution Engine
Amazon ML Challenge 2026

Orchestrates the complete certified frozen pipeline:
1. Multi-view Normalization (legal suffix stripping, script transliteration, address parsing)
2. Multi-channel Composite Blocking (K=175 candidate budget)
3. 61-dimensional pairwise feature extraction ( RapidFuzz similarities, structural address matching, cross-script alignment)
4. Frozen Complementary LightGBM Matcher Ensemble:
   - Model A: Sampled-Universe Hard-Negative Matcher (weight = 0.88)
   - Model B: Full-Universe Hard-Negative Matcher (weight = 0.12)
5. Decision Thresholding at tau = 0.915
6. Output Generation:
   - matching_results.tsv (leaderboard submission)
   - candidate_pairs.tsv (blocking candidates fed to matching model)
"""

from __future__ import annotations

import os
os.environ["PYTHONHASHSEED"] = "2026"

import argparse
import concurrent.futures
import csv
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent
if _ROOT.name == "scripts":
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

from src.blocking.composite import CompositeBlocker
from src.matching.features import CandidateProvenance, PairwiseFeatureExtractor
from src.matching.model import PairwiseLightGBMMatcher
from src.matching.optimized_inference import (
    PersistentTargetStore,
    PrecomputedQuery,
    PrecomputedTarget,
    ThreadSafeTargetStore,
    extract_batch_optimized,
)
from src.normalization.record import BusinessRecord, NormalizationPipeline


def resolve_model_path(path_str: str, default_name: str, fallback_subdir: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    # Check under _ROOT / models
    candidate1 = _ROOT / "models" / default_name
    if candidate1.exists():
        return candidate1
    candidate2 = _ROOT / "models" / fallback_subdir / default_name
    if candidate2.exists():
        return candidate2
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EntiResolve end-to-end entity resolution inference on test data."
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="dataset/test",
        help="Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory where matching_results.tsv and candidate_pairs.tsv will be saved",
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default="models/matcher_hardneg_1x.pkl",
        help="Path to Model A (Sampled-Universe Hard-Negative LightGBM Matcher)",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="models/matcher_fulluniverse_hardneg_1x.pkl",
        help="Path to Model B (Full-Universe Hard-Negative LightGBM Matcher)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.88,
        help="Ensemble blend weight for Model A (S = alpha * P_A + (1 - alpha) * P_B)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.915,
        help="Decision threshold for match declaration (default: 0.915)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=175,
        help="Maximum candidate budget K per query entity (default: 175)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250,
        help="Query batch size for feature extraction & inference (default: 250)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Number of concurrent query stream workers (default: 6)",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Optional: restrict inference to a single country (default: process all countries dynamically)",
    )
    return parser.parse_args()


def read_s1_entities(tsv_path: Path) -> list[tuple[str, str, str, str]]:
    """Read all S1 queries preserving original file row order."""
    records = []
    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 4:
                continue
            eid, name, addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            records.append((eid, name, addr, country))
    return records


def process_batch(
    batch_q: list[BusinessRecord],
    batch_q_pre: list[PrecomputedQuery],
    blocker: CompositeBlocker,
    target_ids_indexed: set[str],
    target_store: ThreadSafeTargetStore | PersistentTargetStore,
    model_a: PairwiseLightGBMMatcher,
    model_b: PairwiseLightGBMMatcher,
    alpha: float,
    threshold: float,
) -> list[tuple[str, list[str], list[str]]]:
    """
    Evaluates one query batch.
    Returns list of (eid, candidate_ids, matched_ids).
    """
    query_candidates_info = []
    all_needed_tids = set()

    for q_rec, q_pre in zip(batch_q, batch_q_pre):
        cands, attr = blocker.query(q_rec, track_attribution=True)
        valid_cands = [c for c in cands if c in target_ids_indexed]
        candidate_blockers = defaultdict(set)
        for b_name, tgts in attr.items():
            for t in tgts:
                candidate_blockers[t].add(b_name)
        query_candidates_info.append((q_pre, valid_cands, candidate_blockers))
        all_needed_tids.update(valid_cands)

    target_objects = target_store.get_batch(list(all_needed_tids))

    batch_pairs = []
    batch_meta = []
    for q_pre, valid_cands, candidate_blockers in query_candidates_info:
        sid = q_pre.entity_id
        for rank, tid in enumerate(valid_cands, start=1):
            tgt_obj = target_objects[tid]
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
            batch_pairs.append((q_pre, tgt_obj, prov))
            batch_meta.append((sid, tid))

    cand_scores_a = defaultdict(dict)
    cand_scores_b = defaultdict(dict)

    if batch_pairs:
        feats = extract_batch_optimized(batch_pairs)
        probs_a = model_a.predict_proba(feats)
        probs_b = model_b.predict_proba(feats)
        for (sid, tid), sa, sb in zip(batch_meta, probs_a, probs_b):
            cand_scores_a[sid][tid] = float(sa)
            cand_scores_b[sid][tid] = float(sb)

    results = []
    for q_pre, valid_cands, _ in query_candidates_info:
        sid = q_pre.entity_id
        preds = []
        for tid in valid_cands:
            sa = cand_scores_a[sid].get(tid, 0.0)
            sb = cand_scores_b[sid].get(tid, 0.0)
            blend_score = round(alpha * sa + (1.0 - alpha) * sb, 5)
            if blend_score >= threshold:
                preds.append(tid)
        results.append((sid, valid_cands, preds))

    return results


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    for req_f in [s1_path, s2_path, s3_path]:
        if not req_f.exists():
            print(f"Error: Required test data file not found: {req_f}", file=sys.stderr)
            print(f"Please ensure test files exist in {test_dir}", file=sys.stderr)
            sys.exit(1)

    model_a_path = resolve_model_path(args.model_a, "matcher_hardneg_1x.pkl", "phase4_3")
    model_b_path = resolve_model_path(args.model_b, "matcher_fulluniverse_hardneg_1x.pkl", "phase4_3c")

    print("=" * 80)
    print("EntiResolve — End-to-End Business Entity Resolution Inference")
    print("=" * 80)
    print(f"Test Directory:      {test_dir.resolve()}")
    print(f"Output Directory:    {out_dir.resolve()}")
    print(f"Model A (Sampled):   {model_a_path}")
    print(f"Model B (FullUniv):  {model_b_path}")
    print(f"Ensemble Blend:      S = {args.alpha:.2f} * P_A + {1.0 - args.alpha:.2f} * P_B")
    print(f"Decision Threshold:  S >= {args.threshold}")
    print(f"Parallel Streams:    {args.num_workers}")
    print("=" * 80, flush=True)

    t0_start = time.perf_counter()

    print(f"\n[1/5] Loading frozen matcher models...")
    model_a = PairwiseLightGBMMatcher.load(model_a_path)
    model_b = PairwiseLightGBMMatcher.load(model_b_path)
    pipeline = NormalizationPipeline()

    print(f"\n[2/5] Reading Source-1 query entities...")
    all_s1_records = read_s1_entities(s1_path)
    total_queries = len(all_s1_records)
    print(f"  Loaded {total_queries:,} queries from {s1_path.name}")

    country_queries = defaultdict(list)
    for eid, name, addr, country in all_s1_records:
        country_queries[country].append((eid, name, addr, country))

    countries_to_process = [args.country] if args.country else sorted(country_queries.keys())

    all_query_candidates: dict[str, list[str]] = {}
    all_query_matches: dict[str, list[str]] = {}
    total_candidates_count = 0
    total_matches_count = 0

    target_paths = [s2_path, s3_path]

    for c_idx, country in enumerate(countries_to_process, start=1):
        c_queries_raw = country_queries[country]
        if not c_queries_raw:
            continue

        print(f"\n[{country}] Partition ({c_idx}/{len(countries_to_process)}): {len(c_queries_raw):,} queries")
        t0_country = time.perf_counter()

        queries_norm = [pipeline.normalize(eid, n, a, c) for eid, n, a, c in c_queries_raw]
        queries_pre = [PrecomputedQuery(q) for q in queries_norm]

        blocker = CompositeBlocker(max_total_candidates=args.max_candidates)
        target_ids_indexed = set()

        temp_db_path = out_dir / f".targets_{country}.db"
        if temp_db_path.exists():
            try:
                temp_db_path.unlink()
            except Exception:
                pass

        temp_store = PersistentTargetStore(temp_db_path, lru_capacity=100_000)
        batch_inserts = []

        print(f"  Indexing target records for {country} ...", flush=True)
        t0_idx = time.perf_counter()

        def stream_targets():
            for path in target_paths:
                with open(path, "rb") as bf:
                    bf.readline()
                    while True:
                        raw_line = bf.readline()
                        if not raw_line:
                            break
                        parts = raw_line.split(b"\t")
                        if len(parts) >= 4 and parts[3].strip().decode("utf-8", "replace") == country:
                            tid = parts[0].strip().decode("utf-8", "replace")
                            name = parts[1].decode("utf-8", "replace")
                            addr = parts[2].decode("utf-8", "replace")
                            target_ids_indexed.add(tid)
                            rec = pipeline.normalize(tid, name, addr, country)
                            pt = PrecomputedTarget.from_record(rec)
                            batch_inserts.append((tid, pt.pack()))
                            if len(batch_inserts) >= 20000:
                                temp_store.put_batch(batch_inserts)
                                batch_inserts.clear()
                            yield rec
            if batch_inserts:
                temp_store.put_batch(batch_inserts)
                batch_inserts.clear()

        indexed_targets = blocker.index_target_records(stream_targets())
        temp_store.close()
        idx_sec = time.perf_counter() - t0_idx
        print(f"  Indexed {indexed_targets:,} target records in {idx_sec:.1f} s.", flush=True)

        target_store = ThreadSafeTargetStore(temp_db_path)

        num_batches = (len(queries_norm) + args.batch_size - 1) // args.batch_size
        batches = []
        for b_idx in range(num_batches):
            b_norm = queries_norm[b_idx * args.batch_size : (b_idx + 1) * args.batch_size]
            b_pre = queries_pre[b_idx * args.batch_size : (b_idx + 1) * args.batch_size]
            batches.append((b_norm, b_pre))

        done_c_queries = 0
        t0_eval = time.perf_counter()

        if args.num_workers <= 1:
            for b_norm, b_pre in batches:
                res = process_batch(
                    b_norm, b_pre, blocker, target_ids_indexed, target_store,
                    model_a, model_b, args.alpha, args.threshold
                )
                for eid, cands, preds in res:
                    all_query_candidates[eid] = cands
                    all_query_matches[eid] = preds
                    total_candidates_count += len(cands)
                    total_matches_count += len(preds)
                done_c_queries += len(res)
                qps = done_c_queries / (time.perf_counter() - t0_eval)
                print(f"  [{country}] Evaluated {done_c_queries:,}/{len(c_queries_raw):,} ({qps:.1f} q/s)", end="\r", flush=True)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_batch = {
                    executor.submit(
                        process_batch,
                        b_norm,
                        b_pre,
                        blocker,
                        target_ids_indexed,
                        target_store,
                        model_a,
                        model_b,
                        args.alpha,
                        args.threshold,
                    ): b_idx
                    for b_idx, (b_norm, b_pre) in enumerate(batches)
                }
                for future in concurrent.futures.as_completed(future_to_batch):
                    res = future.result()
                    for eid, cands, preds in res:
                        all_query_candidates[eid] = cands
                        all_query_matches[eid] = preds
                        total_candidates_count += len(cands)
                        total_matches_count += len(preds)
                    done_c_queries += len(res)
                    qps = done_c_queries / (time.perf_counter() - t0_eval)
                    print(f"  [{country}] Evaluated {done_c_queries:,}/{len(c_queries_raw):,} ({qps:.1f} q/s)", end="\r", flush=True)

        print()
        c_sec = time.perf_counter() - t0_country
        print(f"  [{country}] Finished in {c_sec:.1f} s ({len(c_queries_raw)/c_sec:.1f} q/s).", flush=True)

        target_store.close()
        try:
            temp_db_path.unlink()
        except Exception:
            pass
        del blocker
        del target_ids_indexed
        del queries_norm
        del queries_pre
        gc.collect()

    print("\n[5/5] Writing output files in exact original order...")
    matching_tsv = out_dir / "matching_results.tsv"
    candidate_tsv = out_dir / "candidate_pairs.tsv"

    singletons = 0
    with open(matching_tsv, "w", encoding="utf-8", newline="") as mf, \
         open(candidate_tsv, "w", encoding="utf-8", newline="") as cf:

        m_writer = csv.writer(mf, delimiter="\t", lineterminator="\n")
        c_writer = csv.writer(cf, delimiter="\t", lineterminator="\n")

        m_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        c_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for eid, _, _, _ in all_s1_records:
            preds = all_query_matches.get(eid, [])
            cands = all_query_candidates.get(eid, [])

            m_writer.writerow([eid, ",".join(preds)])
            c_writer.writerow([eid, ",".join(cands)])

            if not preds:
                singletons += 1

    total_time = time.perf_counter() - t0_start
    m_sz = matching_tsv.stat().st_size / (1024 * 1024)
    c_sz = candidate_tsv.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE — OUTPUT FILES READY")
    print(f"  matching_results.tsv: {matching_tsv.resolve()} ({m_sz:.2f} MB)")
    print(f"  candidate_pairs.tsv:  {candidate_tsv.resolve()} ({c_sz:.2f} MB)")
    print(f"  Total Queries:        {len(all_s1_records):,}")
    print(f"  Total Matches:        {total_matches_count:,}")
    print(f"  Total Candidates:     {total_candidates_count:,}")
    print(f"  Singletons (Empty):   {singletons:,} ({singletons/len(all_s1_records)*100:.1f}%)")
    print(f"  Total Wall Time:      {total_time:.1f} s ({total_time/3600:.2f} hours)")
    print("=" * 80)


if __name__ == "__main__":
    main()
