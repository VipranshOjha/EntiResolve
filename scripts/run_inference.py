#!/usr/bin/env python3
"""
scripts/run_phase5_test_inference.py — Full Test Inference & Submission Generation
Amazon ML Challenge 2026: Business Entity Resolution

Orchestrates full-scale test inference on all 1,732,544 test S1 entities against
the complete ~9.97M target universe (test_source2.tsv + test_source3.tsv).

FROZEN CONFIGURATION:
- Model A: models/phase4_3/matcher_hardneg_1x.pkl (Sampled HardNeg 1x)
- Model B: models/phase4_3c/matcher_fulluniverse_hardneg_1x.pkl (Full-Universe HardNeg 1x)
- Blocker: CompositeBlocker(max_total_candidates=175)
- Features: PairwiseFeatureExtractor (61 features, cross-script enabled)
- Score Blend: S = 0.88 * P_A + 0.12 * P_B
- Threshold: S >= 0.915
- Partitioning: Country-aware / Open-country handling (France -> US -> India)
- Batch size: 250
- Parallel query streams: 6 (single process with thread-safe shared read-only target index)
- PYTHONHASHSEED: 2026

OUTPUTS:
- output/matching_results.tsv (leaderboard submission)
- output/candidate_pairs.tsv (blocking candidates)
"""

from __future__ import annotations

import os
os.environ["PYTHONHASHSEED"] = "2026"

import argparse
import concurrent.futures
import csv
import gc
import json
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

# Add project root to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import psutil

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


def get_process_ram_mb() -> float:
    """Returns current process peak working set or RSS in megabytes."""
    proc = psutil.Process(os.getpid())
    info = proc.memory_info()
    peak = getattr(info, "peak_wset", info.rss)
    return max(info.rss, peak) / (1024 * 1024)


def get_system_ram_mb() -> float:
    """Returns total system used memory in megabytes."""
    return psutil.virtual_memory().used / (1024 * 1024)


class SystemTelemetryMonitor:
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_system_ram_mb = 0.0
        self.peak_process_ram_mb = 0.0
        self.cpu_readings: list[float] = []

    def _monitor(self):
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            try:
                sys_ram = get_system_ram_mb()
                proc_ram = get_process_ram_mb()
                cpu = psutil.cpu_percent(interval=None)
                if sys_ram > self.peak_system_ram_mb:
                    self.peak_system_ram_mb = sys_ram
                if proc_ram > self.peak_process_ram_mb:
                    self.peak_process_ram_mb = proc_ram
                self.cpu_readings.append(cpu)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self._stop_event.clear()
        self.peak_process_ram_mb = get_process_ram_mb()
        self.peak_system_ram_mb = get_system_ram_mb()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
        return {
            "peak_system_ram_mb": round(self.peak_system_ram_mb, 1),
            "peak_process_ram_mb": round(self.peak_process_ram_mb, 1),
            "mean_cpu_percent": round(float(np.mean(self.cpu_readings)), 1) if self.cpu_readings else 0.0,
            "max_cpu_percent": round(float(np.max(self.cpu_readings)), 1) if self.cpu_readings else 0.0,
        }


def process_batch_worker(
    b_idx: int,
    batch_q: list[BusinessRecord],
    batch_q_pre: list[PrecomputedQuery],
    blocker: CompositeBlocker,
    target_ids_indexed: set[str],
    target_store: ThreadSafeTargetStore,
    model_a: PairwiseLightGBMMatcher,
    model_b: PairwiseLightGBMMatcher,
    country: str,
    cp_dir: Path,
) -> tuple[int, list[dict], int]:
    """
    Evaluates one batch of queries in a worker thread.
    Writes atomic checkpoint file and returns (b_idx, batch_records, total_pairs).
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

    batch_records = []
    for q_pre, valid_cands, _ in query_candidates_info:
        sid = q_pre.entity_id
        preds = []
        for tid in valid_cands:
            sa = cand_scores_a[sid].get(tid, 0.0)
            sb = cand_scores_b[sid].get(tid, 0.0)
            blend_score = round(0.88 * sa + 0.12 * sb, 5)
            if blend_score >= 0.915:
                preds.append(tid)

        rec_dict = {
            "entity_id": sid,
            "country": country,
            "candidates": valid_cands,
            "predictions": preds,
        }
        batch_records.append(rec_dict)

    # Atomic write to temporary file then rename
    tmp_path = cp_dir / f"batch_{b_idx:05d}.jsonl.tmp"
    final_path = cp_dir / f"batch_{b_idx:05d}.jsonl"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in batch_records:
            f.write(json.dumps(r) + "\n")
    if final_path.exists():
        final_path.unlink()
    tmp_path.rename(final_path)

    return b_idx, batch_records, len(batch_pairs)


def run_country_partition(
    country: str,
    queries_raw: list[tuple[str, str, str, str]],
    target_tsv_paths: list[Path],
    model_a: PairwiseLightGBMMatcher,
    model_b: PairwiseLightGBMMatcher,
    extractor: PairwiseFeatureExtractor,
    pipeline: NormalizationPipeline,
    cp_base_dir: Path,
    db_base_dir: Path,
    batch_size: int = 250,
    num_workers: int = 6,
) -> dict:
    """
    Evaluates test queries for a single country partition with atomic batch checkpointing
    and 6-stream CPU parallel inference.
    """
    print(f"\n{'=' * 80}")
    print(f"TEST INFERENCE PARTITION — {country.upper()}")
    print(f"Queries: {len(queries_raw):,} | Batch Size: {batch_size} | Parallel Streams: {num_workers}")
    print(f"{'=' * 80}", flush=True)

    cp_dir = cp_base_dir / country
    cp_dir.mkdir(parents=True, exist_ok=True)

    # Scan existing batch checkpoints
    num_batches = (len(queries_raw) + batch_size - 1) // batch_size
    existing_batch_indices = set()
    completed_queries_count = 0

    for bf in cp_dir.glob("batch_*.jsonl"):
        try:
            b_idx_str = bf.stem.split("_")[1]
            b_idx_val = int(b_idx_str)
            existing_batch_indices.add(b_idx_val)
        except Exception:
            pass

    # Count how many queries are already completed
    if existing_batch_indices:
        for b_idx in existing_batch_indices:
            bf = cp_dir / f"batch_{b_idx:05d}.jsonl"
            with open(bf, "r", encoding="utf-8") as f:
                completed_queries_count += sum(1 for line in f if line.strip())

    if len(existing_batch_indices) == num_batches and completed_queries_count == len(queries_raw):
        print(f"[{country}] All {completed_queries_count:,} queries ({num_batches:,} batches) completed in checkpoints! Skipping partition evaluation...", flush=True)
        return {
            "country": country,
            "queries_count": len(queries_raw),
            "resumed_count": completed_queries_count,
            "newly_processed_count": 0,
            "indexed_count": 0,
            "index_build_sec": 0.0,
            "evaluation_sec": 0.0,
            "total_candidate_pairs": 0,
            "qps": 0.0,
            "cands_per_sec": 0.0,
            "peak_worker_ram_mb": get_process_ram_mb(),
            "failures": 0,
        }

    if completed_queries_count > 0:
        print(f"[{country}] Resuming from checkpoints: {completed_queries_count:,} queries ({len(existing_batch_indices):,}/{num_batches:,} batches) already completed.", flush=True)

    # Precompute queries
    t0_prep = time.perf_counter()
    queries = [pipeline.normalize(eid, n, a, c) for eid, n, a, c in queries_raw]
    precomputed_queries = [PrecomputedQuery(q) for q in queries]
    prep_time = time.perf_counter() - t0_prep
    print(f"[{country}] Normalized and precomputed {len(queries):,} queries in {prep_time:.2f} s.", flush=True)

    # Check if target database already exists in db_base_dir or cp_dir
    possible_dbs = [
        db_base_dir / f"targets_{country}.db",
        cp_dir / f"targets_{country}.db",
    ]
    db_path = None
    for p in possible_dbs:
        if p.exists() and p.stat().st_size > 0:
            db_path = p
            break

    build_db = False
    if db_path is None:
        db_path = cp_dir / f"targets_{country}.db"
        build_db = True
        print(f"[{country}] No precomputed target store found. Will build SQLite store at {db_path} ...", flush=True)
        temp_store = PersistentTargetStore(db_path, lru_capacity=100_000)
    else:
        print(f"[{country}] Reusing verified target store at {db_path} ({db_path.stat().st_size / (1024*1024):.1f} MB)", flush=True)

    # Stream targets into CompositeBlocker
    print(f"[{country}] Indexing complete target universe for blocker...", flush=True)
    t0_idx = time.perf_counter()
    blocker = CompositeBlocker(max_total_candidates=175)
    target_ids_indexed: set[str] = set()
    batch_inserts: list[tuple[str, bytes]] = []

    def stream_targets():
        for path in target_tsv_paths:
            with open(path, "rb") as bf:
                bf.readline()  # skip header
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
                        if build_db:
                            pt = PrecomputedTarget.from_record(rec)
                            batch_inserts.append((tid, pt.pack()))
                            if len(batch_inserts) >= 20000:
                                temp_store.put_batch(batch_inserts)
                                batch_inserts.clear()
                        yield rec
        if build_db and batch_inserts:
            temp_store.put_batch(batch_inserts)
            batch_inserts.clear()

    indexed_count = blocker.index_target_records(stream_targets())
    idx_time = time.perf_counter() - t0_idx
    peak_ram_post_idx = get_process_ram_mb()
    print(f"[{country}] Indexed {indexed_count:,} target records in {idx_time:.2f} s. Peak Worker RAM: {peak_ram_post_idx:.1f} MB.", flush=True)

    if build_db:
        temp_store.close()

    # Open ThreadSafeTargetStore
    target_store = ThreadSafeTargetStore(db_path)

    # Identify uncompleted batches
    uncompleted_batches = []
    for b_idx in range(num_batches):
        if b_idx in existing_batch_indices:
            continue
        batch_q = queries[b_idx * batch_size : (b_idx + 1) * batch_size]
        batch_q_pre = precomputed_queries[b_idx * batch_size : (b_idx + 1) * batch_size]
        uncompleted_batches.append((b_idx, batch_q, batch_q_pre))

    print(f"[{country}] Dispatching {len(uncompleted_batches):,} batches ({len(uncompleted_batches)*batch_size:,} queries) across {num_workers} parallel streams...", flush=True)

    t0_eval = time.perf_counter()
    done_queries = completed_queries_count
    total_candidate_pairs = 0
    total_failures = 0
    total_to_process = len(queries)

    if num_workers == 1:
        # Sequential execution
        for b_idx, batch_q, batch_q_pre in uncompleted_batches:
            _, batch_records, num_pairs = process_batch_worker(
                b_idx, batch_q, batch_q_pre, blocker, target_ids_indexed,
                target_store, model_a, model_b, country, cp_dir
            )
            done_queries += len(batch_records)
            total_candidate_pairs += num_pairs
            elapsed = time.perf_counter() - t0_eval
            qps = (done_queries - completed_queries_count) / elapsed if elapsed > 0 else 0.0
            cand_rate = total_candidate_pairs / elapsed if elapsed > 0 else 0.0
            pct = done_queries / total_to_process * 100
            mem = get_process_ram_mb()
            rem_q = total_to_process - done_queries
            eta_sec = rem_q / qps if qps > 0 else 0.0
            print(f"[{country}] {done_queries:,}/{total_to_process:,} ({pct:.1f}%) | {qps:.1f} q/s | {cand_rate:.0f} cands/s | ETA: {eta_sec/60:.1f}m | RAM: {mem:.1f} MB", end="\r", flush=True)
    else:
        # Concurrent execution via ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch = {
                executor.submit(
                    process_batch_worker,
                    b_idx,
                    batch_q,
                    batch_q_pre,
                    blocker,
                    target_ids_indexed,
                    target_store,
                    model_a,
                    model_b,
                    country,
                    cp_dir,
                ): b_idx
                for b_idx, batch_q, batch_q_pre in uncompleted_batches
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    b_idx, batch_records, num_pairs = future.result()
                    done_queries += len(batch_records)
                    total_candidate_pairs += num_pairs
                except Exception as e:
                    total_failures += 1
                    print(f"\n[{country} ERROR] Batch failed with exception: {e}", flush=True)
                    raise e

                elapsed = time.perf_counter() - t0_eval
                qps = (done_queries - completed_queries_count) / elapsed if elapsed > 0 else 0.0
                cand_rate = total_candidate_pairs / elapsed if elapsed > 0 else 0.0
                pct = done_queries / total_to_process * 100
                mem = get_process_ram_mb()
                rem_q = total_to_process - done_queries
                eta_sec = rem_q / qps if qps > 0 else 0.0
                print(f"[{country}] {done_queries:,}/{total_to_process:,} ({pct:.1f}%) | {qps:.1f} q/s | {cand_rate:.0f} cands/s | ETA: {eta_sec/60:.1f}m | RAM: {mem:.1f} MB", end="\r", flush=True)

    print()
    total_eval_time = time.perf_counter() - t0_eval
    newly_processed = done_queries - completed_queries_count
    qps_overall = newly_processed / total_eval_time if total_eval_time > 0 else 0.0
    cand_rate_overall = total_candidate_pairs / total_eval_time if total_eval_time > 0 else 0.0

    print(f"[{country}] Completed {newly_processed:,} new queries in {total_eval_time:.2f} s ({qps_overall:.1f} q/s, {cand_rate_overall:.0f} cands/s).", flush=True)
    print(f"[{country}] Total partition candidates generated: {total_candidate_pairs:,}.", flush=True)

    target_store.close()
    del blocker
    del target_ids_indexed
    del queries
    del precomputed_queries
    gc.collect()

    return {
        "country": country,
        "queries_count": len(queries_raw),
        "resumed_count": completed_queries_count,
        "newly_processed_count": newly_processed,
        "indexed_count": indexed_count,
        "index_build_sec": round(idx_time, 2),
        "evaluation_sec": round(total_eval_time, 2),
        "total_candidate_pairs": total_candidate_pairs,
        "qps": round(qps_overall, 1),
        "cands_per_sec": round(cand_rate_overall, 1),
        "peak_worker_ram_mb": round(peak_ram_post_idx, 1),
        "failures": total_failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5B: Full Test Inference on 1,732,544 queries with 6 parallel streams."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Number of parallel query streams (default: 6)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250,
        help="Batch size per query stream (default: 250)",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=str(_REPO_ROOT / "output" / "bench_phase5a"),
        help="Directory containing precomputed SQLite target stores (default: output/bench_phase5a)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(_REPO_ROOT / "output" / "checkpoints_phase5"),
        help="Directory for atomic batch checkpoints (default: output/checkpoints_phase5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026: PHASE 5B FULL TEST INFERENCE")
    print("Parallel Query Streams: 6 | Batch Size: 250")
    print("Deterministic Order | Frozen Blend (0.88/0.12) | Threshold: 0.915")
    print("=" * 80, flush=True)

    t_start = time.perf_counter()

    project_root = _REPO_ROOT.parent
    test_dir = project_root / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"
    output_dirs = [project_root / "output", _REPO_ROOT / "output"]
    for od in output_dirs:
        od.mkdir(parents=True, exist_ok=True)

    cp_base_dir = Path(args.checkpoint_dir)
    cp_base_dir.mkdir(parents=True, exist_ok=True)
    db_base_dir = Path(args.db_dir)

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    model_a_path = _REPO_ROOT / "models" / "phase4_3" / "matcher_hardneg_1x.pkl"
    model_b_path = _REPO_ROOT / "models" / "phase4_3c" / "matcher_fulluniverse_hardneg_1x.pkl"

    # Start system telemetry
    monitor = SystemTelemetryMonitor(interval=5.0)
    monitor.start()

    # 1. Load frozen matcher models
    print(f"Loading Model A: {model_a_path.name} ...")
    model_a = PairwiseLightGBMMatcher.load(model_a_path)
    print(f"Loading Model B: {model_b_path.name} ...")
    model_b = PairwiseLightGBMMatcher.load(model_b_path)

    # 2. Initialize normalization & feature extractor
    pipeline = NormalizationPipeline()
    extractor = PairwiseFeatureExtractor()

    # 3. Read all test S1 entities preserving exact original file order
    print(f"\nReading test_source1.tsv from {s1_path} ...")
    all_s1_records: list[tuple[str, str, str, str]] = []
    country_queries: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)

    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 4:
                continue
            eid, name, addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            all_s1_records.append((eid, name, addr, country))
            country_queries[country].append((eid, name, addr, country))

    total_s1 = len(all_s1_records)
    print(f"Loaded {total_s1:,} test S1 entities across {len(country_queries)} countries:")
    for c, q_list in sorted(country_queries.items()):
        print(f"  - {c}: {len(q_list):,} entities ({len(q_list)/total_s1*100:.1f}%)")

    # Order partitions: France first (smallest), then US, then India
    partition_order = ["France", "US", "India"]
    for c in sorted(country_queries.keys()):
        if c not in partition_order:
            partition_order.append(c)

    target_tsv_paths = [s2_path, s3_path]
    partition_profiles: dict[str, dict] = {}

    for c in partition_order:
        if c not in country_queries:
            continue
        c_queries = country_queries[c]
        prof = run_country_partition(
            country=c,
            queries_raw=c_queries,
            target_tsv_paths=target_tsv_paths,
            model_a=model_a,
            model_b=model_b,
            extractor=extractor,
            pipeline=pipeline,
            cp_base_dir=cp_base_dir,
            db_base_dir=db_base_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        partition_profiles[c] = prof

    total_inference_sec = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print("ALL PARTITIONS FINISHED — ASSEMBLING FINAL SUBMISSION TSV FILES")
    print("=" * 80, flush=True)

    # 4. Assemble final results from checkpoints in exact original test_source1.tsv order
    t0_assemble = time.perf_counter()
    print("Loading entity results from atomic checkpoints...", flush=True)
    all_entity_results: dict[str, tuple[str, str]] = {}
    total_checkpoint_queries = 0

    for c in partition_order:
        cp_dir = cp_base_dir / c
        batch_files = sorted(cp_dir.glob("batch_*.jsonl"))
        print(f"[{c}] Reading {len(batch_files):,} batch checkpoint files...", flush=True)
        for bf in batch_files:
            with open(bf, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        eid = item["entity_id"]
                        preds_str = ",".join(item.get("predictions", []))
                        cands_str = ",".join(item.get("candidates", []))
                        all_entity_results[eid] = (preds_str, cands_str)
                        total_checkpoint_queries += 1

    assemble_load_sec = time.perf_counter() - t0_assemble
    print(f"Loaded {len(all_entity_results):,} entities from checkpoints in {assemble_load_sec:.2f} s.", flush=True)
    assert len(all_entity_results) == total_s1, f"Mismatch: expected {total_s1} entities, got {len(all_entity_results)}"

    # 5. Write final matching_results.tsv and candidate_pairs.tsv
    total_matched_pairs = 0
    total_candidate_pairs = 0
    total_singletons = 0
    out_file_sizes = {}

    for out_base in output_dirs:
        matching_tsv_path = out_base / "matching_results.tsv"
        candidate_tsv_path = out_base / "candidate_pairs.tsv"

        tmp_m = out_base / "matching_results.tsv.tmp"
        tmp_c = out_base / "candidate_pairs.tsv.tmp"

        print(f"\nWriting final TSVs to {out_base} ...", flush=True)
        cur_matched = 0
        cur_cands = 0
        cur_singletons = 0

        with open(tmp_m, "w", encoding="utf-8", newline="") as mf, \
             open(tmp_c, "w", encoding="utf-8", newline="") as cf:

            m_writer = csv.writer(mf, delimiter="\t", lineterminator="\n")
            c_writer = csv.writer(cf, delimiter="\t", lineterminator="\n")

            m_writer.writerow(["source1_entity_id", "matched_entity_ids"])
            c_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

            for eid, _, _, _ in all_s1_records:
                preds_str, cands_str = all_entity_results[eid]
                m_writer.writerow([eid, preds_str])
                c_writer.writerow([eid, cands_str])

                if preds_str:
                    cur_matched += preds_str.count(",") + 1
                else:
                    cur_singletons += 1

                if cands_str:
                    cur_cands += cands_str.count(",") + 1

        if matching_tsv_path.exists():
            matching_tsv_path.unlink()
        tmp_m.rename(matching_tsv_path)

        if candidate_tsv_path.exists():
            candidate_tsv_path.unlink()
        tmp_c.rename(candidate_tsv_path)

        m_sz_mb = matching_tsv_path.stat().st_size / (1024 * 1024)
        c_sz_mb = candidate_tsv_path.stat().st_size / (1024 * 1024)
        out_file_sizes[str(matching_tsv_path)] = round(m_sz_mb, 2)
        out_file_sizes[str(candidate_tsv_path)] = round(c_sz_mb, 2)
        print(f"  Saved: {matching_tsv_path} ({m_sz_mb:.2f} MB, {total_s1:,} rows)")
        print(f"  Saved: {candidate_tsv_path} ({c_sz_mb:.2f} MB, {total_s1:,} rows)")

        total_matched_pairs = cur_matched
        total_candidate_pairs = cur_cands
        total_singletons = cur_singletons

    del all_entity_results
    gc.collect()

    total_wall_sec = time.perf_counter() - t_start
    sys_telemetry = monitor.stop()

    print("\n" + "=" * 80)
    print("INFERENCE AND TSV GENERATION SUMMARY")
    print(f"Total Evaluated Entities: {total_s1:,}")
    print(f"Total Matches Emitted:    {total_matched_pairs:,}")
    print(f"Total Candidates Emitted: {total_candidate_pairs:,}")
    print(f"Total Singletons:         {total_singletons:,} ({total_singletons/total_s1*100:.2f}%)")
    print(f"Total Wall Clock Time:    {total_wall_sec:.1f} s ({total_wall_sec/3600:.2f} hours)")
    print(f"Peak System RAM:          {sys_telemetry['peak_system_ram_mb']:.1f} MB")
    print(f"Peak Process RAM:         {sys_telemetry['peak_process_ram_mb']:.1f} MB")
    print("=" * 80, flush=True)

    # Save summary metadata
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_queries": total_s1,
        "total_matched_pairs": total_matched_pairs,
        "total_candidate_pairs": total_candidate_pairs,
        "total_singletons": total_singletons,
        "runtime_sec": round(total_wall_sec, 2),
        "telemetry": sys_telemetry,
        "partitions": partition_profiles,
        "output_file_sizes_mb": out_file_sizes,
    }
    meta_path = _REPO_ROOT / "reports" / "phase5b_test_inference_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved telemetry metadata to {meta_path}")

    # 6. Run Official Validator
    print("\n" + "=" * 80)
    print("RUNNING OFFICIAL VALIDATOR ON GENERATED SUBMISSION TSVS")
    print("=" * 80, flush=True)

    validator_script = project_root / "6ab10eb3b23ba_student_resource" / "student_resource" / "utils" / "validate_submission.py"
    matching_tsv = project_root / "output" / "matching_results.tsv"
    candidate_tsv = project_root / "output" / "candidate_pairs.tsv"

    val_cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_tsv),
        "--candidate", str(candidate_tsv),
        "--test-dir", str(test_dir),
    ]
    print(f"Command: {' '.join(val_cmd)}", flush=True)
    val_proc = subprocess.run(val_cmd, capture_output=True, text=True)
    print(val_proc.stdout)
    if val_proc.stderr:
        print(val_proc.stderr, file=sys.stderr)

    if val_proc.returncode != 0:
        print(f"\n[FATAL] Official validator failed with exit code {val_proc.returncode}!", flush=True)
        sys.exit(val_proc.returncode)
    print("[PASS] Official submission validator PASSED with exit code 0!", flush=True)

    # 7. Run Submission Packaging
    print("\n" + "=" * 80)
    print("BUILDING SUBMISSION ZIP ARCHIVE")
    print("=" * 80, flush=True)

    build_cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "build_submission.py"),
        "--team-name", "antigravity_er",
        "--output-dir", str(project_root / "output"),
        "--code-dir", str(project_root / "code" / "business_entity_resolution"),
        "--doc-file", str(project_root / "Documentation_template.md"),
        "--test-dir", str(test_dir),
        "--validator-script", str(validator_script),
        "--zip-output-dir", str(project_root),
    ]
    print(f"Command: {' '.join(build_cmd)}", flush=True)
    build_proc = subprocess.run(build_cmd, capture_output=True, text=True)
    print(build_proc.stdout)
    if build_proc.stderr:
        print(build_proc.stderr, file=sys.stderr)

    if build_proc.returncode != 0:
        print(f"\n[FATAL] Submission build failed with exit code {build_proc.returncode}!", flush=True)
        sys.exit(build_proc.returncode)

    zip_path = project_root / "antigravity_er_submission.zip"
    print(f"\n[SUCCESS] Final submission zip archive built: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")
    print("\n" + "=" * 80)
    print("PHASE 5B FULL TEST INFERENCE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
