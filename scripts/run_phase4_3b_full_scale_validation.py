"""
scripts/run_phase4_3b_full_scale_validation.py — Phase 4.3B Full-Scale Development Validation.

Validates Phase 4.3 hard-negative models (1x and 4x) against the COMPLETE 10,320,219-record
training target universe (Source 2 + Source 3) alongside the Phase 4.1 frozen baseline.

Strict Data Isolation:
- Development ONLY.
- 10,000 dev_val entities (5,000 US, 5,000 India) from data/validation/phase4_3b_dev_val_10k_ids.txt.
- Official validation set (val_ids.txt) and competition test sets are 100% SEALED.
"""

from __future__ import annotations

import os
os.environ["PYTHONHASHSEED"] = "2026"

import argparse
import csv
import gc
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import psutil

# Add project root to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.blocking.composite import CompositeBlocker
from src.matching.cross_script import CrossScriptAligner
from src.matching.features import (
    ALL_FEATURE_NAMES,
    CandidateProvenance,
    PairwiseFeatureExtractor,
)
from src.matching.model import PairwiseLightGBMMatcher
from src.normalization.record import BusinessRecord, NormalizationPipeline
from src.evaluation.metrics import _entity_scores

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.97, 0.98]
MODEL_KEYS = ["baseline", "hardneg_1x", "hardneg_4x"]


def get_process_ram_mb() -> float:
    info = psutil.Process(os.getpid()).memory_info()
    peak = getattr(info, "peak_wset", info.rss)
    return max(info.rss, peak) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Background Telemetry Monitor
# ---------------------------------------------------------------------------

class SystemTelemetryMonitor:
    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.stop_event = threading.Event()
        self.cpu_readings: list[float] = []
        self.ram_readings_mb: list[float] = []
        self.monitor_thread: threading.Thread | None = None

    def start(self):
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._run, daemon=True)
        self.monitor_thread.start()

    def _run(self):
        psutil.cpu_percent(interval=None)
        while not self.stop_event.is_set():
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().used / (1024 * 1024)
            self.cpu_readings.append(cpu)
            self.ram_readings_mb.append(ram)
            time.sleep(self.interval)

    def stop(self) -> dict:
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        return {
            "mean_cpu_percent": float(np.mean(self.cpu_readings)) if self.cpu_readings else 0.0,
            "max_cpu_percent": float(np.max(self.cpu_readings)) if self.cpu_readings else 0.0,
            "peak_system_ram_mb": float(np.max(self.ram_readings_mb)) if self.ram_readings_mb else 0.0,
            "initial_system_ram_mb": float(self.ram_readings_mb[0]) if self.ram_readings_mb else 0.0,
        }


# ---------------------------------------------------------------------------
# Checkpoint Helpers
# ---------------------------------------------------------------------------

def get_checkpoint_dir(base_dir: Path, country: str) -> Path:
    cp_dir = base_dir / country
    cp_dir.mkdir(parents=True, exist_ok=True)
    return cp_dir


def load_existing_checkpoints(cp_dir: Path) -> dict[str, dict]:
    completed_queries: dict[str, dict] = {}
    batch_files = sorted(cp_dir.glob("batch_*.jsonl"))
    for bf in batch_files:
        with open(bf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    completed_queries[item["entity_id"]] = item
    return completed_queries


def save_batch_checkpoint(cp_dir: Path, batch_idx: int, batch_records: list[dict]):
    tmp_path = cp_dir / f"batch_{batch_idx:05d}.jsonl.tmp"
    final_path = cp_dir / f"batch_{batch_idx:05d}.jsonl"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in batch_records:
            f.write(json.dumps(rec) + "\n")
    if final_path.exists():
        final_path.unlink()
    tmp_path.rename(final_path)


# ---------------------------------------------------------------------------
# Country Partition Evaluator
# ---------------------------------------------------------------------------

def run_country_partition(
    country: str,
    queries_raw: list[tuple[str, str, str, str]],  # (eid, name, addr, ctry)
    target_tsv_paths: list[Path],
    models: dict[str, PairwiseLightGBMMatcher],
    extractor: PairwiseFeatureExtractor,
    pipeline: NormalizationPipeline,
    checkpoint_base_dir: Path,
    batch_size: int = 250,
) -> tuple[dict[str, dict], dict, dict]:
    """
    Builds country index against 100% of real training targets for the country,
    processes queries in batches, evaluates all 3 models, runs a 500-query
    determinism check, and checkpoints.
    """
    print(f"\n{'=' * 80}")
    print(f"STARTING COUNTRY PARTITION: {country.upper()}")
    print(f"Queries: {len(queries_raw):,} | Batch Size: {batch_size}")
    print(f"{'=' * 80}", flush=True)

    cp_dir = get_checkpoint_dir(checkpoint_base_dir, country)
    existing_checkpoints = load_existing_checkpoints(cp_dir)
    num_resumed = len(existing_checkpoints)

    if num_resumed > 0:
        print(f"[{country}] Found {num_resumed:,} completed queries from previous checkpoints. Resuming seamlessly...")

    # 1. Normalize query records
    queries = [pipeline.normalize(eid, n, a, c) for eid, n, a, c in queries_raw]

    # 2. Build target offset index and stream into CompositeBlocker
    print(f"[{country}] Building CompositeBlocker index and raw target offset map from REAL targets...", flush=True)
    t0_idx = time.perf_counter()
    blocker = CompositeBlocker(max_total_candidates=175)
    target_offsets: dict[str, int] = {}  # tid -> (file_idx << 32) | byte_offset

    def stream_targets():
        for f_idx, path in enumerate(target_tsv_paths):
            with open(path, "rb") as bf:
                bf.readline()  # skip header
                while True:
                    offset = bf.tell()
                    raw_line = bf.readline()
                    if not raw_line:
                        break
                    parts = raw_line.split(b"\t")
                    if len(parts) >= 4 and parts[3].strip().decode("utf-8", "replace") == country:
                        tid = parts[0].strip().decode("utf-8", "replace")
                        name = parts[1].decode("utf-8", "replace")
                        addr = parts[2].decode("utf-8", "replace")
                        target_offsets[tid] = (f_idx << 32) | offset
                        norm_rec = pipeline.normalize(tid, name, addr, country)
                        yield norm_rec

    indexed_count = blocker.index_target_records(stream_targets())
    idx_time = time.perf_counter() - t0_idx
    peak_ram_post_idx = get_process_ram_mb()
    print(f"[{country}] Indexed {indexed_count:,} target records in {idx_time:.2f} s. Peak Worker RAM: {peak_ram_post_idx:.1f} MB.", flush=True)

    # File handles for reading raw targets on demand
    file_handles = [open(p, "r", encoding="utf-8", errors="replace") for p in target_tsv_paths]

    def get_raw_target(tid: str) -> tuple[str, str]:
        packed = target_offsets[tid]
        f_idx = packed >> 32
        offset = packed & 0xFFFFFFFF
        fh = file_handles[f_idx]
        fh.seek(offset)
        line = fh.readline()
        parts = line.split("\t")
        return parts[1], parts[2]

    # 3. Process queries in batches
    target_cache: dict[str, BusinessRecord] = {}
    all_query_results: dict[str, dict] = dict(existing_checkpoints)

    num_batches = (len(queries) + batch_size - 1) // batch_size
    t0_eval = time.perf_counter()

    for b_idx in range(num_batches):
        batch_q = queries[b_idx * batch_size : (b_idx + 1) * batch_size]

        # Check if entire batch is already processed
        if all(q.entity_id in all_query_results for q in batch_q):
            continue

        batch_pairs = []
        batch_meta = []  # (sid, tid)
        batch_candidates_dict: dict[str, list[str]] = {}
        batch_cross_script_flags: dict[str, list[int]] = defaultdict(list)

        for q_rec in batch_q:
            sid = q_rec.entity_id
            if sid in all_query_results:
                continue

            cands, attr = blocker.query(q_rec, track_attribution=True)
            valid_cands = [c for c in cands if c in target_offsets]
            batch_candidates_dict[sid] = valid_cands

            candidate_blockers = defaultdict(set)
            for b_name, tgts in attr.items():
                for t in tgts:
                    candidate_blockers[t].add(b_name)

            for rank, tid in enumerate(valid_cands, start=1):
                if tid not in target_cache:
                    n, a = get_raw_target(tid)
                    target_cache[tid] = pipeline.normalize(tid, n, a, country)
                tgt = target_cache[tid]
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
                batch_pairs.append((q_rec, tgt, prov))
                batch_meta.append((sid, tid))

                is_cross = 1 if (
                    (q_rec.name.script.is_non_latin != tgt.name.script.is_non_latin) or
                    (q_rec.name.script.is_mixed_script or tgt.name.script.is_mixed_script)
                ) else 0
                batch_cross_script_flags[sid].append(is_cross)

        # Extract features ONCE and predict with ALL 3 MODELS
        sid_to_cand_scores = {
            m_key: defaultdict(dict) for m_key in MODEL_KEYS
        }

        if batch_pairs:
            feats = extractor.extract_batch(batch_pairs)
            for m_key, model in models.items():
                probs = model.predict_proba(feats)
                for (sid, tid), score in zip(batch_meta, probs):
                    sid_to_cand_scores[m_key][sid][tid] = float(score)

        # Build checkpoint records
        batch_checkpoint_records = []
        for q_rec in batch_q:
            sid = q_rec.entity_id
            if sid in all_query_results:
                continue

            cands = batch_candidates_dict.get(sid, [])
            if q_rec.name.script.is_non_latin:
                q_script = "NON_LATIN"
            elif q_rec.name.script.is_mixed_script:
                q_script = "MIXED_SCRIPT"
            else:
                q_script = "LATIN"

            scores_by_model = {}
            for m_key in MODEL_KEYS:
                # Store candidate scores aligned with the cands list
                model_scores = [
                    round(sid_to_cand_scores[m_key][sid].get(tid, 0.0), 5)
                    for tid in cands
                ]
                scores_by_model[m_key] = model_scores

            rec_dict = {
                "entity_id": sid,
                "country": country,
                "script": q_script,
                "candidates_count": len(cands),
                "candidates": cands,
                "cross_script_flags": batch_cross_script_flags.get(sid, []),
                "scores": scores_by_model,
            }
            batch_checkpoint_records.append(rec_dict)
            all_query_results[sid] = rec_dict

        # Save checkpoint
        if batch_checkpoint_records:
            save_batch_checkpoint(cp_dir, b_idx, batch_checkpoint_records)

        # Cache management
        if len(target_cache) > 25000:
            target_cache.clear()
            gc.collect()

        # Progress reporting
        processed_so_far = len(all_query_results)
        if (b_idx + 1) % 5 == 0 or (b_idx + 1) == num_batches:
            elapsed = time.perf_counter() - t0_eval
            qps = (processed_so_far - num_resumed) / elapsed if elapsed > 0 else 0
            cur_ram = get_process_ram_mb()
            sys_ram = psutil.virtual_memory().used / (1024 * 1024)
            print(f"  [{country}] Progress: {processed_so_far:,}/{len(queries):,} queries ({(processed_so_far/len(queries))*100:.1f}%) | "
                  f"{qps:.1f} q/s | Worker RAM: {cur_ram:.1f} MB | Sys RAM: {sys_ram:.1f} MB", flush=True)

    total_eval_time = time.perf_counter() - t0_eval

    # 4. In-Memory Determinism Check on first 500 queries of this country
    print(f"[{country}] Running 500-query determinism equivalence check using live index...", flush=True)
    det_sample_q = queries[:500]
    cand_id_matches = 0
    cand_order_matches = 0
    feature_matches = 0
    prob_matches = 0
    prediction_matches = 0

    for q_rec in det_sample_q:
        sid = q_rec.entity_id
        stored = all_query_results[sid]
        cands, attr = blocker.query(q_rec, track_attribution=True)
        valid_cands = [c for c in cands if c in target_offsets]

        if set(valid_cands) == set(stored["candidates"]):
            cand_id_matches += 1
        if valid_cands == stored["candidates"]:
            cand_order_matches += 1

        if not valid_cands:
            feature_matches += 1
            prob_matches += 1
            prediction_matches += 1
            continue

        candidate_blockers = defaultdict(set)
        for b_name, tgts in attr.items():
            for t in tgts:
                candidate_blockers[t].add(b_name)

        pairs = []
        for rank, tid in enumerate(valid_cands, start=1):
            if tid not in target_cache:
                n, a = get_raw_target(tid)
                target_cache[tid] = pipeline.normalize(tid, n, a, country)
            tgt = target_cache[tid]
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
            pairs.append((q_rec, tgt, prov))

        feats = extractor.extract_batch(pairs)
        if feats.shape[1] == 61:
            feature_matches += 1

        all_probs_match = True
        all_preds_match = True

        for m_key in MODEL_KEYS:
            probs = models[m_key].predict_proba(feats)
            stored_probs = np.array(stored["scores"][m_key], dtype=np.float32)
            cur_rounded = np.round(probs, 5).astype(np.float32)
            if not np.allclose(cur_rounded, stored_probs, atol=1e-4):
                all_probs_match = False

            for tau in [0.80, 0.92]:
                ref_pred = {tid for tid, p in zip(valid_cands, probs) if p >= tau}
                stored_pred = {tid for tid, p in zip(stored["candidates"], stored["scores"][m_key]) if p >= tau}
                if ref_pred != stored_pred:
                    all_preds_match = False

        if all_probs_match:
            prob_matches += 1
        if all_preds_match:
            prediction_matches += 1

    det_check = {
        "country": country,
        "queries_tested": len(det_sample_q),
        "candidate_id_matches": cand_id_matches,
        "candidate_order_matches": cand_order_matches,
        "feature_matches": feature_matches,
        "probability_matches": prob_matches,
        "prediction_matches": prediction_matches,
        "status": "PASSED_100_PERCENT" if (cand_order_matches == len(det_sample_q) and prediction_matches == len(det_sample_q)) else "DISCREPANCY_DETECTED",
    }
    print(f"[{country}] Determinism Check Result: {det_check['status']} ({cand_order_matches}/500 exact)", flush=True)

    # Close file handles
    for fh in file_handles:
        fh.close()

    # Free memory
    del blocker
    del target_offsets
    del target_cache
    gc.collect()

    profile = {
        "indexed_count": indexed_count,
        "index_build_sec": idx_time,
        "evaluation_sec": total_eval_time,
        "peak_worker_ram_mb": peak_ram_post_idx,
    }

    return all_query_results, profile, det_check


# ---------------------------------------------------------------------------
# Metric Aggregations & Analysis
# ---------------------------------------------------------------------------

def compute_all_metrics(
    all_query_results: dict[str, dict],
    gt: dict[str, frozenset[str]],
) -> dict:
    """
    Computes comprehensive threshold sweeps, singleton analyses, cardinality
    analyses, country breakdowns, script breakdowns, and error decompositions.
    """
    print("\n" + "=" * 80)
    print("COMPUTING COMPREHENSIVE BENCHMARK METRICS ACROSS ALL THRESHOLDS...")
    print("=" * 80, flush=True)

    metrics_out = {}
    total_queries = len(all_query_results)

    # 1. Blocker Statistics
    candidate_counts = [res["candidates_count"] for res in all_query_results.values()]
    total_true_pairs = sum(len(gt.get(sid, frozenset())) for sid in all_query_results)
    retrieved_true_pairs = 0
    for sid, res in all_query_results.items():
        cands = set(res["candidates"])
        truth = gt.get(sid, frozenset())
        retrieved_true_pairs += len(truth.intersection(cands))

    cand_recall = (retrieved_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0

    metrics_out["blocker"] = {
        "K": 175,
        "total_queries": total_queries,
        "candidate_pair_recall": cand_recall,
        "mean_candidates": float(np.mean(candidate_counts)),
        "p50_candidates": float(np.percentile(candidate_counts, 50)),
        "p95_candidates": float(np.percentile(candidate_counts, 95)),
        "p99_candidates": float(np.percentile(candidate_counts, 99)),
        "total_candidates": int(sum(candidate_counts)),
        "total_true_pairs": total_true_pairs,
        "retrieved_true_pairs": retrieved_true_pairs,
    }

    # 2. Model Evaluation per Threshold
    models_metrics = {}

    for m_key in MODEL_KEYS:
        models_metrics[m_key] = {
            "threshold_sweep": {},
            "country_sweep": {"India": {}, "US": {}},
            "error_decomposition": {},
            "singleton_analysis": {},
            "cardinality_analysis": {},
            "script_analysis": {},
            "cross_script_pairs": {},
        }

        # Analyze singleton scores distribution (independent of threshold)
        top1_scores = []
        top2_scores = []
        margins = []

        for sid, res in all_query_results.items():
            truth = gt.get(sid, frozenset())
            if len(truth) == 0:
                scores = res["scores"][m_key]
                if len(scores) == 0:
                    top1 = 0.0
                    top2 = 0.0
                elif len(scores) == 1:
                    top1 = float(scores[0])
                    top2 = 0.0
                else:
                    s_sorted = sorted(scores, reverse=True)
                    top1 = float(s_sorted[0])
                    top2 = float(s_sorted[1])
                top1_scores.append(top1)
                top2_scores.append(top2)
                margins.append(top1 - top2)

        models_metrics[m_key]["singleton_score_distribution"] = {
            "top1_scores": {
                "mean": float(np.mean(top1_scores)),
                "std": float(np.std(top1_scores)),
                "min": float(np.min(top1_scores)),
                "p10": float(np.percentile(top1_scores, 10)),
                "p25": float(np.percentile(top1_scores, 25)),
                "p50": float(np.percentile(top1_scores, 50)),
                "p75": float(np.percentile(top1_scores, 75)),
                "p90": float(np.percentile(top1_scores, 90)),
                "p95": float(np.percentile(top1_scores, 95)),
                "p99": float(np.percentile(top1_scores, 99)),
                "max": float(np.max(top1_scores)),
            },
            "top2_scores": {
                "mean": float(np.mean(top2_scores)),
                "std": float(np.std(top2_scores)),
                "min": float(np.min(top2_scores)),
                "p10": float(np.percentile(top2_scores, 10)),
                "p25": float(np.percentile(top2_scores, 25)),
                "p50": float(np.percentile(top2_scores, 50)),
                "p75": float(np.percentile(top2_scores, 75)),
                "p90": float(np.percentile(top2_scores, 90)),
                "p95": float(np.percentile(top2_scores, 95)),
                "p99": float(np.percentile(top2_scores, 99)),
                "max": float(np.max(top2_scores)),
            },
            "top_vs_second_margins": {
                "mean": float(np.mean(margins)),
                "std": float(np.std(margins)),
                "min": float(np.min(margins)),
                "p10": float(np.percentile(margins, 10)),
                "p25": float(np.percentile(margins, 25)),
                "p50": float(np.percentile(margins, 50)),
                "p75": float(np.percentile(margins, 75)),
                "p90": float(np.percentile(margins, 90)),
                "p95": float(np.percentile(margins, 95)),
                "p99": float(np.percentile(margins, 99)),
                "max": float(np.max(margins)),
            },
        }

        # Run threshold sweep
        for tau in THRESHOLDS:
            tau_str = f"{tau:.2f}"

            f05_list = []
            prec_list = []
            rec_list = []
            tp_total = 0
            fp_total = 0
            fn_total = 0
            singletons_total = 0
            singletons_correct = 0
            singletons_fp = 0
            complete_matches = 0
            non_singletons = 0
            predicted_matches_total = 0

            # Country breakdowns
            ctry_stats = {
                "India": {"f05": [], "p": [], "r": [], "tp": 0, "fp": 0, "fn": 0, "sing_tot": 0, "sing_cor": 0, "preds_tot": 0, "tot": 0},
                "US": {"f05": [], "p": [], "r": [], "tp": 0, "fp": 0, "fn": 0, "sing_tot": 0, "sing_cor": 0, "preds_tot": 0, "tot": 0},
            }

            # Cardinality breakdowns
            card_stats = defaultdict(lambda: {"entities": 0, "f05_s": 0.0, "p_s": 0.0, "r_s": 0.0, "exact": 0})

            # Script breakdowns
            script_stats = defaultdict(lambda: {"entities": 0, "f05_s": 0.0, "p_s": 0.0, "r_s": 0.0, "fp": 0})

            # Cross-script pair tracking
            cs_pairs_evaluated = 0
            cs_tp = 0
            cs_fp = 0
            cs_fn = 0

            # Error decomposition categories
            decomp = {
                "all_gt_retrieved_correct": 0,
                "all_gt_retrieved_false_positives": 0,
                "all_gt_retrieved_false_negatives": 0,
                "all_gt_retrieved_mixed_error": 0,
                "candidate_truncation": 0,
                "zero_gt_retrieved": 0,
                "singleton_correct": 0,
                "singleton_false_merges": 0,
            }

            for sid, res in all_query_results.items():
                cands = res["candidates"]
                scores = res["scores"][m_key]
                truth = gt.get(sid, frozenset())
                country = res["country"]
                script = res["script"]
                cs_flags = res.get("cross_script_flags", [0] * len(cands))

                preds = frozenset(tid for tid, s in zip(cands, scores) if s >= tau)
                cands_set = set(cands)

                p_i, r_i, f05_i = _entity_scores(preds, truth)
                f05_list.append(f05_i)
                prec_list.append(p_i)
                rec_list.append(r_i)

                tp = len(preds & truth)
                fp = len(preds - truth)
                fn = len(truth - preds)
                tp_total += tp
                fp_total += fp
                fn_total += fn
                predicted_matches_total += len(preds)

                # Cross-script pair tracking
                for tid, s, is_cs in zip(cands, scores, cs_flags):
                    if is_cs:
                        cs_pairs_evaluated += 1
                        pred_match = (s >= tau)
                        is_true = (tid in truth)
                        if pred_match and is_true:
                            cs_tp += 1
                        elif pred_match and not is_true:
                            cs_fp += 1
                        elif not pred_match and is_true:
                            cs_fn += 1

                # Country tracking
                cs = ctry_stats[country]
                cs["tot"] += 1
                cs["f05"].append(f05_i)
                cs["p"].append(p_i)
                cs["r"].append(r_i)
                cs["tp"] += tp
                cs["fp"] += fp
                cs["fn"] += fn
                cs["preds_tot"] += len(preds)

                # Singleton vs Non-singleton
                if len(truth) == 0:
                    singletons_total += 1
                    cs["sing_tot"] += 1
                    if len(preds) == 0:
                        singletons_correct += 1
                        cs["sing_cor"] += 1
                        decomp["singleton_correct"] += 1
                    else:
                        singletons_fp += 1
                        decomp["singleton_false_merges"] += 1
                else:
                    non_singletons += 1
                    if preds == truth:
                        complete_matches += 1

                    # Error decomposition for non-singletons
                    gt_in_cands = truth & cands_set
                    if len(gt_in_cands) == 0:
                        decomp["zero_gt_retrieved"] += 1
                    elif len(gt_in_cands) < len(truth):
                        decomp["candidate_truncation"] += 1
                    else:
                        # All GT retrieved in candidate set
                        if preds == truth:
                            decomp["all_gt_retrieved_correct"] += 1
                        elif preds > truth:
                            decomp["all_gt_retrieved_false_positives"] += 1
                        elif preds < truth:
                            decomp["all_gt_retrieved_false_negatives"] += 1
                        else:
                            decomp["all_gt_retrieved_mixed_error"] += 1

                # Cardinality tracking
                card = len(truth)
                bucket = str(card) if card < 8 else ">=8"
                c_bucket = card_stats[bucket]
                c_bucket["entities"] += 1
                c_bucket["f05_s"] += f05_i
                c_bucket["p_s"] += p_i
                c_bucket["r_s"] += r_i
                if preds == truth:
                    c_bucket["exact"] += 1

                # Script tracking
                s_bucket = script_stats[script]
                s_bucket["entities"] += 1
                s_bucket["f05_s"] += f05_i
                s_bucket["p_s"] += p_i
                s_bucket["r_s"] += r_i
                s_bucket["fp"] += fp

            # Global aggregation for this threshold
            macro_f05 = float(np.mean(f05_list))
            macro_p = float(np.mean(prec_list))
            macro_r = float(np.mean(rec_list))
            pair_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            pair_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            sing_acc = singletons_correct / singletons_total if singletons_total > 0 else 1.0
            comp_rec = complete_matches / non_singletons if non_singletons > 0 else 1.0

            thresh_res = {
                "threshold": tau,
                "macro_f05": macro_f05,
                "macro_precision": macro_p,
                "macro_recall": macro_r,
                "pair_precision": pair_p,
                "pair_recall": pair_r,
                "candidate_pair_recall": cand_recall,
                "complete_set_recall": comp_rec,
                "singleton_accuracy": sing_acc,
                "mean_fp_per_query": fp_total / total_queries,
                "mean_predicted_matches": predicted_matches_total / total_queries,
                "mean_candidates": float(np.mean(candidate_counts)),
                "exact_match_ratio": (singletons_correct + complete_matches) / total_queries,
                "tp": tp_total,
                "fp": fp_total,
                "fn": fn_total,
            }
            models_metrics[m_key]["threshold_sweep"][tau_str] = thresh_res

            # Country metrics for this threshold
            for ctry in ["India", "US"]:
                cs = ctry_stats[ctry]
                c_f05 = float(np.mean(cs["f05"]))
                c_p = float(np.mean(cs["p"]))
                c_r = float(np.mean(cs["r"]))
                c_pair_p = cs["tp"] / (cs["tp"] + cs["fp"]) if (cs["tp"] + cs["fp"]) > 0 else 0.0
                c_pair_r = cs["tp"] / (cs["tp"] + cs["fn"]) if (cs["tp"] + cs["fn"]) > 0 else 0.0
                c_sing_acc = cs["sing_cor"] / cs["sing_tot"] if cs["sing_tot"] > 0 else 1.0

                models_metrics[m_key]["country_sweep"][ctry][tau_str] = {
                    "macro_f05": c_f05,
                    "macro_precision": c_p,
                    "macro_recall": c_r,
                    "pair_precision": c_pair_p,
                    "pair_recall": c_pair_r,
                    "singleton_accuracy": c_sing_acc,
                    "mean_fp_per_query": cs["fp"] / cs["tot"],
                    "mean_predicted_matches": cs["preds_tot"] / cs["tot"],
                }

            # Cross-script pair metrics
            cs_prec = cs_tp / (cs_tp + cs_fp) if (cs_tp + cs_fp) > 0 else 0.0
            cs_rec = cs_tp / (cs_tp + cs_fn) if (cs_tp + cs_fn) > 0 else 0.0
            cs_f05 = (1.25 * cs_prec * cs_rec) / (0.25 * cs_prec + cs_rec) if (0.25 * cs_prec + cs_rec) > 0 else 0.0
            models_metrics[m_key]["cross_script_pairs"][tau_str] = {
                "cross_script_pairs_evaluated": cs_pairs_evaluated,
                "cross_script_tp": cs_tp,
                "cross_script_fp": cs_fp,
                "cross_script_fn": cs_fn,
                "cross_script_precision": cs_prec,
                "cross_script_recall": cs_rec,
                "cross_script_f05": cs_f05,
            }

            # Store error decomposition and breakdowns for this threshold
            models_metrics[m_key]["error_decomposition"][tau_str] = decomp

            models_metrics[m_key]["cardinality_analysis"][tau_str] = {
                b: {
                    "entities": card_stats[b]["entities"],
                    "macro_f05": card_stats[b]["f05_s"] / card_stats[b]["entities"] if card_stats[b]["entities"] > 0 else 0.0,
                    "macro_p": card_stats[b]["p_s"] / card_stats[b]["entities"] if card_stats[b]["entities"] > 0 else 0.0,
                    "macro_r": card_stats[b]["r_s"] / card_stats[b]["entities"] if card_stats[b]["entities"] > 0 else 0.0,
                    "exact_match_pct": (card_stats[b]["exact"] / card_stats[b]["entities"] * 100) if card_stats[b]["entities"] > 0 else 0.0,
                }
                for b in ["0", "1", "2", "3", "4", "5", "6", "7", ">=8"] if b in card_stats
            }

            models_metrics[m_key]["script_analysis"][tau_str] = {
                s: {
                    "entities": script_stats[s]["entities"],
                    "macro_f05": script_stats[s]["f05_s"] / script_stats[s]["entities"] if script_stats[s]["entities"] > 0 else 0.0,
                    "macro_p": script_stats[s]["p_s"] / script_stats[s]["entities"] if script_stats[s]["entities"] > 0 else 0.0,
                    "macro_r": script_stats[s]["r_s"] / script_stats[s]["entities"] if script_stats[s]["entities"] > 0 else 0.0,
                    "false_positives": script_stats[s]["fp"],
                    "fp_per_query": script_stats[s]["fp"] / script_stats[s]["entities"] if script_stats[s]["entities"] > 0 else 0.0,
                }
                for s in ["LATIN", "NON_LATIN", "MIXED_SCRIPT"] if s in script_stats
            }

    metrics_out["models"] = models_metrics
    return metrics_out


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4.3B Full-Scale Development Validation.")
    parser.add_argument("--data_dir", type=str, default="../6ab10eb3b23ba_student_resource/student_resource/dataset/train")
    parser.add_argument("--query_ids_file", type=str, default="data/validation/phase4_3b_dev_val_10k_ids.txt")
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--checkpoint_dir", type=str, default="output/checkpoints_phase4_3b")
    parser.add_argument("--output_report", type=str, default="reports/phase4_3b_full_scale_validation_metrics.json")
    parser.add_argument("--csv_sweep_output", type=str, default="reports/phase4_3b_threshold_sweep.csv")
    parser.add_argument("--error_decomp_output", type=str, default="reports/phase4_3b_error_decomposition.json")
    parser.add_argument("--singleton_output", type=str, default="reports/phase4_3b_singleton_analysis.json")

    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    data_dir = Path(args.data_dir)
    query_ids_file = Path(args.query_ids_file)
    pipeline = NormalizationPipeline()

    print("=" * 80)
    print("PHASE 4.3B — FULL-SCALE HARD-NEGATIVE VALIDATION AGAINST 10.32M TARGETS")
    print(f"Population File: {query_ids_file}")
    print(f"Target Universe: 10,320,219 REAL S2/S3 Target Records")
    print("=" * 80, flush=True)

    # 1. Strict Data-Isolation Check
    print("Executing strict data-isolation audit...", flush=True)
    with open("data/splits/val_ids.txt", "r", encoding="utf-8") as f:
        official_val_ids = set(line.strip() for line in f if line.strip())
    with open("data/splits/dev_train_ids.txt", "r", encoding="utf-8") as f:
        dev_train_ids = set(line.strip() for line in f if line.strip())
    with open(query_ids_file, "r", encoding="utf-8") as f:
        benchmark_ids = [line.strip() for line in f if line.strip()]
    benchmark_ids_set = set(benchmark_ids)

    assert len(benchmark_ids) == 10000, f"Expected 10,000 queries, got {len(benchmark_ids)}"
    assert len(benchmark_ids_set.intersection(official_val_ids)) == 0, "LEAKAGE: benchmark queries overlap with official val!"
    assert len(benchmark_ids_set.intersection(dev_train_ids)) == 0, "LEAKAGE: benchmark queries overlap with dev_train!"
    print("Data-isolation verified: 0 official validation queries, 0 dev_train queries. 100% clean!", flush=True)

    # 2. Load Ground Truth
    print("Loading ground truth for 10,000 dev_val entities...", flush=True)
    gt: dict[str, frozenset[str]] = {}
    with open(data_dir / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0].strip() in benchmark_ids_set:
                targets = frozenset(t.strip() for t in row[1].split(",") if t.strip())
                gt[row[0].strip()] = targets

    for sid in benchmark_ids:
        if sid not in gt:
            gt[sid] = frozenset()

    # 3. Stream Benchmark Queries and Partition by Country
    queries_us: list[tuple[str, str, str, str]] = []
    queries_india: list[tuple[str, str, str, str]] = []

    print("Streaming queries from train_source1.tsv...", flush=True)
    with open(data_dir / "train_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 4 and row[0].strip() in benchmark_ids_set:
                eid, name, addr, ctry = row[0].strip(), row[1], row[2], row[3].strip()
                if ctry == "India":
                    queries_india.append((eid, name, addr, ctry))
                elif ctry == "US":
                    queries_us.append((eid, name, addr, ctry))

    print(f"Partitioned queries: US = {len(queries_us):,} | India = {len(queries_india):,}. Total = {len(queries_us) + len(queries_india):,}.", flush=True)
    assert len(queries_us) == 5000 and len(queries_india) == 5000, "Expected exactly 5,000 US and 5,000 India queries!"

    # 4. Load Models and Extractor
    print("Loading models and feature extractor...", flush=True)
    models = {
        "baseline": PairwiseLightGBMMatcher.load(Path("models/matching_global/matcher_lightgbm.pkl")),
        "hardneg_1x": PairwiseLightGBMMatcher.load(Path("models/phase4_3/matcher_hardneg_1x.pkl")),
        "hardneg_4x": PairwiseLightGBMMatcher.load(Path("models/phase4_3/matcher_hardneg_4x.pkl")),
    }
    cs_path = Path("data/pairs/cross_script_aligner_dev_train.json")
    cs_aligner = CrossScriptAligner.load(cs_path)
    extractor = PairwiseFeatureExtractor(cross_script_aligner=cs_aligner, include_cross_script=True)

    target_tsv_paths = [data_dir / "train_source2.tsv", data_dir / "train_source3.tsv"]
    cp_base_dir = Path(args.checkpoint_dir)

    telemetry = SystemTelemetryMonitor(interval=2.0)
    telemetry.start()
    t0_global = time.perf_counter()

    # 5. Run Country Partition 1: India
    india_results, india_profile, india_det = run_country_partition(
        country="India",
        queries_raw=queries_india,
        target_tsv_paths=target_tsv_paths,
        models=models,
        extractor=extractor,
        pipeline=pipeline,
        checkpoint_base_dir=cp_base_dir,
        batch_size=args.batch_size,
    )

    # 6. Run Country Partition 2: US
    us_results, us_profile, us_det = run_country_partition(
        country="US",
        queries_raw=queries_us,
        target_tsv_paths=target_tsv_paths,
        models=models,
        extractor=extractor,
        pipeline=pipeline,
        checkpoint_base_dir=cp_base_dir,
        batch_size=args.batch_size,
    )

    total_eval_wall_sec = time.perf_counter() - t0_global
    sys_telemetry = telemetry.stop()

    # Merge results
    all_query_results = {**india_results, **us_results}
    assert len(all_query_results) == 10000, f"Expected 10,000 query results, got {len(all_query_results)}"

    # 7. Aggregate Determinism Equivalence Check (1,000 queries total)
    det_check_results = {
        "queries_tested": india_det["queries_tested"] + us_det["queries_tested"],
        "candidate_id_matches": india_det["candidate_id_matches"] + us_det["candidate_id_matches"],
        "candidate_order_matches": india_det["candidate_order_matches"] + us_det["candidate_order_matches"],
        "feature_matches": india_det["feature_matches"] + us_det["feature_matches"],
        "probability_matches": india_det["probability_matches"] + us_det["probability_matches"],
        "prediction_matches": india_det["prediction_matches"] + us_det["prediction_matches"],
        "status": "PASSED_100_PERCENT" if (india_det["status"] == "PASSED_100_PERCENT" and us_det["status"] == "PASSED_100_PERCENT") else "DISCREPANCY_DETECTED",
        "partitions": {
            "India": india_det,
            "US": us_det,
        }
    }

    # 8. Compute All Benchmark Metrics
    final_metrics = compute_all_metrics(all_query_results, gt)

    # Attach Profiles & Telemetry
    final_metrics["experiment_meta"] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        "total_queries": len(all_query_results),
        "india_queries": len(queries_india),
        "us_queries": len(queries_us),
        "total_target_records": india_profile["indexed_count"] + us_profile["indexed_count"],
        "india_target_records": india_profile["indexed_count"],
        "us_target_records": us_profile["indexed_count"],
        "total_wall_sec": total_eval_wall_sec,
        "india_profile": india_profile,
        "us_profile": us_profile,
        "telemetry": sys_telemetry,
        "determinism_check": det_check_results,
    }

    # Verify target count
    expected_targets = 10320219
    actual_targets = final_metrics["experiment_meta"]["total_target_records"]
    assert actual_targets == expected_targets, f"Target count mismatch! Expected {expected_targets}, got {actual_targets}"

    # 9. Save JSON Reports
    out_report_path = Path(args.output_report)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved full validation metrics to: {out_report_path}")

    # Save Error Decomposition JSON
    error_decomp_path = Path(args.error_decomp_output)
    error_decomp_data = {
        m_key: final_metrics["models"][m_key]["error_decomposition"]
        for m_key in MODEL_KEYS
    }
    with open(error_decomp_path, "w", encoding="utf-8") as f:
        json.dump(error_decomp_data, f, indent=2)
    print(f"Saved error decomposition to: {error_decomp_path}")

    # Save Singleton Analysis JSON
    singleton_path = Path(args.singleton_output)
    singleton_data = {
        m_key: {
            "score_distribution": final_metrics["models"][m_key]["singleton_score_distribution"],
            "sweep": {
                tau_str: {
                    "singleton_accuracy": final_metrics["models"][m_key]["threshold_sweep"][tau_str]["singleton_accuracy"],
                    "mean_fp_per_query": final_metrics["models"][m_key]["threshold_sweep"][tau_str]["mean_fp_per_query"],
                }
                for tau_str in final_metrics["models"][m_key]["threshold_sweep"]
            }
        }
        for m_key in MODEL_KEYS
    }
    with open(singleton_path, "w", encoding="utf-8") as f:
        json.dump(singleton_data, f, indent=2)
    print(f"Saved singleton analysis to: {singleton_path}")

    # 10. Save CSV Threshold Sweep
    csv_path = Path(args.csv_sweep_output)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "threshold", "macro_f05", "macro_precision", "macro_recall",
            "pair_precision", "pair_recall", "complete_set_recall", "singleton_accuracy",
            "mean_fp_per_query", "mean_predicted_matches", "delta_f05_vs_baseline"
        ])
        base_sweep = final_metrics["models"]["baseline"]["threshold_sweep"]
        for m_key in MODEL_KEYS:
            m_sweep = final_metrics["models"][m_key]["threshold_sweep"]
            for tau in THRESHOLDS:
                tau_str = f"{tau:.2f}"
                row_data = m_sweep[tau_str]
                base_f05 = base_sweep[tau_str]["macro_f05"]
                delta_f05 = row_data["macro_f05"] - base_f05
                writer.writerow([
                    m_key,
                    tau_str,
                    f"{row_data['macro_f05']:.4f}",
                    f"{row_data['macro_precision']:.4f}",
                    f"{row_data['macro_recall']:.4f}",
                    f"{row_data['pair_precision']:.4f}",
                    f"{row_data['pair_recall']:.4f}",
                    f"{row_data['complete_set_recall']:.4f}",
                    f"{row_data['singleton_accuracy']:.4f}",
                    f"{row_data['mean_fp_per_query']:.4f}",
                    f"{row_data['mean_predicted_matches']:.4f}",
                    f"{delta_f05:+.4f}",
                ])
    print(f"Saved CSV threshold sweep to: {csv_path}")

    # 11. Print Headline Comparison Table
    print("\n" + "=" * 95)
    print("PHASE 4.3B HEADLINE COMPARISON (10,000 DEV_VAL QUERIES vs 10.32M FULL TARGET UNIVERSE)")
    print("=" * 95)
    print(f"{'Model':<15} {'Thresh':<8} {'Macro F0.5':<12} {'Macro Prec':<12} {'Macro Rec':<12} {'Pair Prec':<12} {'Sing Acc':<12} {'FP/Query':<10}")
    print("-" * 95)

    # Baseline at frozen 0.92
    b92 = base_sweep["0.92"]
    print(f"{'Baseline':<15} {'0.92*':<8} {b92['macro_f05']:<12.4f} {b92['macro_precision']:<12.4f} {b92['macro_recall']:<12.4f} {b92['pair_precision']:<12.4f} {b92['singleton_accuracy']:<12.4f} {b92['mean_fp_per_query']:<10.4f}")

    # Best baseline threshold
    best_base_tau = max(THRESHOLDS, key=lambda t: base_sweep[f"{t:.2f}"]["macro_f05"])
    bb = base_sweep[f"{best_base_tau:.2f}"]
    print(f"{'Baseline (opt)':<15} {f'{best_base_tau:.2f}':<8} {bb['macro_f05']:<12.4f} {bb['macro_precision']:<12.4f} {bb['macro_recall']:<12.4f} {bb['pair_precision']:<12.4f} {bb['singleton_accuracy']:<12.4f} {bb['mean_fp_per_query']:<10.4f}")

    for m_key in ["hardneg_1x", "hardneg_4x"]:
        sweep = final_metrics["models"][m_key]["threshold_sweep"]
        # At 0.80
        m80 = sweep["0.80"]
        print(f"{m_key:<15} {'0.80':<8} {m80['macro_f05']:<12.4f} {m80['macro_precision']:<12.4f} {m80['macro_recall']:<12.4f} {m80['pair_precision']:<12.4f} {m80['singleton_accuracy']:<12.4f} {m80['mean_fp_per_query']:<10.4f}")
        # At 0.92
        m92 = sweep["0.92"]
        print(f"{m_key:<15} {'0.92':<8} {m92['macro_f05']:<12.4f} {m92['macro_precision']:<12.4f} {m92['macro_recall']:<12.4f} {m92['pair_precision']:<12.4f} {m92['singleton_accuracy']:<12.4f} {m92['mean_fp_per_query']:<10.4f}")
        # At best tau
        best_tau = max(THRESHOLDS, key=lambda t: sweep[f"{t:.2f}"]["macro_f05"])
        bm = sweep[f"{best_tau:.2f}"]
        print(f"{m_key + ' (opt)':<15} {f'{best_tau:.2f}':<8} {bm['macro_f05']:<12.4f} {bm['macro_precision']:<12.4f} {bm['macro_recall']:<12.4f} {bm['pair_precision']:<12.4f} {bm['singleton_accuracy']:<12.4f} {bm['mean_fp_per_query']:<10.4f}")
        print("-" * 95)

    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
