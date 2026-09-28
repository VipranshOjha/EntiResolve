"""
scripts/run_phase4_3c_full_universe_mining.py — Phase 4.3C Full-Universe Hard-Negative Mining.

Mines hard negatives directly from the COMPLETE 10,320,219-record target universe (Source 2 + Source 3)
using 10,000 dev_train queries, trains new full-universe hard-negative matchers, and evaluates them
against the sealed 10,000 dev_val benchmark population across thresholds [0.90 ... 0.99].

Strict Data Isolation:
- Development ONLY.
- Mining queries from data/pairs/phase4_3c/mining_query_ids.txt (dev_train ONLY).
- Evaluation queries from data/validation/phase4_3b_dev_val_10k_ids.txt (dev_val ONLY).
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

THRESHOLDS_EVAL = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99]


def get_process_ram_mb() -> float:
    info = psutil.Process(os.getpid()).memory_info()
    peak = getattr(info, "peak_wset", info.rss)
    return max(info.rss, peak) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Telemetry Monitor
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


def load_completed_query_ids(cp_dir: Path) -> set[str]:
    completed = set()
    for f in cp_dir.glob("batch_*.jsonl"):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    item = json.loads(line)
                    completed.add(item["entity_id"])
    return completed


# ---------------------------------------------------------------------------
# Category Classifier for False Positives
# ---------------------------------------------------------------------------

def classify_negative(feats: np.ndarray, q_rec: BusinessRecord, tgt_rec: BusinessRecord) -> list[str]:
    """
    Classifies a false positive pair into Categories A through I using 61 features.
    A negative may belong to multiple categories.
    """
    categories = []

    # A. Cross-script near-match
    if (
        feats[21] > 0.5 or  # name_cross_script
        feats[60] > 0.5 or  # cross_script_learned_sim
        (q_rec.name.script.is_non_latin != tgt_rec.name.script.is_non_latin) or
        q_rec.name.script.is_mixed_script or tgt_rec.name.script.is_mixed_script
    ):
        categories.append("A_cross_script")

    # B. High normalized-name similarity
    if feats[4] >= 0.75 or feats[5] >= 0.85:  # name_levenshtein >= 0.75 or name_jaro_winkler >= 0.85
        categories.append("B_high_name_sim")

    # C. High address similarity
    if feats[27] >= 0.70 or feats[28] >= 0.60:  # addr_levenshtein >= 0.70 or addr_token_jaccard >= 0.60
        categories.append("C_high_addr_sim")

    # D. Same postal code, wrong entity
    if feats[34] > 0.5:  # addr_postal_code_agreement == 1.0
        categories.append("D_same_postal")

    # E. Same house number, wrong entity
    if feats[33] > 0.5:  # addr_house_number_agreement == 1.0
        categories.append("E_same_house_number")

    # F. Generic commercial-token overlap
    if feats[7] >= 0.35 or feats[8] >= 0.50:  # name_token_jaccard >= 0.35 or name_token_overlap_coef >= 0.50
        categories.append("F_generic_token_overlap")

    # G. Website-like near-match
    if feats[24] > 0.5 or feats[25] > 0.5 or feats[26] > 0.5:
        categories.append("G_website_match")

    # H. Multiple-blocker false positive
    if feats[56] >= 2.0:  # prov_blocker_count >= 2
        categories.append("H_multiple_blockers")

    # I. Other high-scoring distractor
    if not categories:
        categories.append("I_other_distractor")

    return categories


# ---------------------------------------------------------------------------
# STAGE 1: Full-Universe Hard-Negative Mining on dev_train
# ---------------------------------------------------------------------------

def run_mining_partition(
    country: str,
    mining_queries_raw: list[tuple[str, str, str, str]],
    target_tsv_paths: list[Path],
    teacher_model: PairwiseLightGBMMatcher,
    extractor: PairwiseFeatureExtractor,
    pipeline: NormalizationPipeline,
    gt: dict[str, frozenset[str]],
    cp_base_dir: Path,
    batch_size: int = 250,
) -> tuple[list[dict], dict]:
    """
    Indexes the complete target universe for the country, runs teacher model scoring
    on dev_train queries, classifies false positives into categories A-I, collects
    score distributions, and applies per-query capping.
    """
    print(f"\n{'=' * 80}")
    print(f"STAGE 1: MINING PARTITION — {country.upper()}")
    print(f"Mining Queries: {len(mining_queries_raw):,} | Batch Size: {batch_size}")
    print(f"{'=' * 80}", flush=True)

    cp_dir = get_checkpoint_dir(cp_base_dir, country)
    completed_sids = load_completed_query_ids(cp_dir)
    num_resumed = len(completed_sids)
    if num_resumed > 0:
        print(f"[{country}] Found {num_resumed:,} completed mining queries from previous checkpoints. Resuming...")

    queries = [pipeline.normalize(eid, n, a, c) for eid, n, a, c in mining_queries_raw]

    # Index all target records
    print(f"[{country}] Indexing 100% of real targets into CompositeBlocker...", flush=True)
    t0_idx = time.perf_counter()
    blocker = CompositeBlocker(max_total_candidates=175)
    target_offsets: dict[str, int] = {}

    def stream_targets():
        for f_idx, path in enumerate(target_tsv_paths):
            with open(path, "rb") as bf:
                bf.readline()
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
                        yield pipeline.normalize(tid, name, addr, country)

    indexed_count = blocker.index_target_records(stream_targets())
    idx_time = time.perf_counter() - t0_idx
    peak_ram_post_idx = get_process_ram_mb()
    print(f"[{country}] Indexed {indexed_count:,} target records in {idx_time:.2f} s. Peak Worker RAM: {peak_ram_post_idx:.1f} MB.", flush=True)

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

    target_cache: dict[str, BusinessRecord] = {}
    mined_records_all: list[dict] = []

    # Load existing mined records if resuming
    for f in sorted(cp_dir.glob("batch_*.jsonl")):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    mined_records_all.append(json.loads(line))

    num_batches = (len(queries) + batch_size - 1) // batch_size
    t0_eval = time.perf_counter()

    for b_idx in range(num_batches):
        batch_q = queries[b_idx * batch_size : (b_idx + 1) * batch_size]

        if all(q.entity_id in completed_sids for q in batch_q):
            continue

        batch_pairs = []
        batch_meta = []  # (q_idx, sid, tid, is_true)
        batch_cands_dict = {}

        for q_idx, q_rec in enumerate(batch_q):
            sid = q_rec.entity_id
            if sid in completed_sids:
                continue

            cands, attr = blocker.query(q_rec, track_attribution=True)
            valid_cands = [c for c in cands if c in target_offsets]
            batch_cands_dict[sid] = valid_cands
            truth = gt.get(sid, frozenset())

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
                is_true = (tid in truth)
                batch_meta.append((q_idx, sid, tid, is_true, tgt))

        # Extract features and score with teacher model
        batch_checkpoint_records = []
        if batch_pairs:
            feats = extractor.extract_batch(batch_pairs)
            scores = teacher_model.predict_proba(feats)

            # Group candidates by query
            query_candidates = defaultdict(list)
            for (q_idx, sid, tid, is_true, tgt_rec), score, f_vec in zip(batch_meta, scores, feats):
                query_candidates[sid].append({
                    "tid": tid,
                    "is_true": is_true,
                    "score": float(score),
                    "features": f_vec,
                    "tgt_rec": tgt_rec,
                })

            for q_rec in batch_q:
                sid = q_rec.entity_id
                if sid in completed_sids:
                    continue

                c_list = query_candidates.get(sid, [])
                tp_scores = [c["score"] for c in c_list if c["is_true"]]
                fp_cands = [c for c in c_list if not c["is_true"]]

                # Rank false positives by score descending
                fp_cands.sort(key=lambda x: x["score"], reverse=True)

                # Classify all false positives into categories
                for c in fp_cands:
                    c["categories"] = classify_negative(c["features"], q_rec, c["tgt_rec"])

                # Negative selection with capping:
                # 1. Hard negatives: score >= 0.50 (cap at max 10 per query)
                hard_pool = [c for c in fp_cands if c["score"] >= 0.50]
                selected_hard = hard_pool[:10]

                # 2. Medium negatives: 0.20 <= score < 0.50 (cap at max 5 per query)
                med_pool = [c for c in fp_cands if 0.20 <= c["score"] < 0.50]
                selected_med = med_pool[:5]

                selected_mined = selected_hard + selected_med

                # Prepare checkpoint record
                rec = {
                    "entity_id": sid,
                    "country": country,
                    "candidates_count": len(c_list),
                    "tp_count": len(tp_scores),
                    "fp_count": len(fp_cands),
                    "tp_scores": [round(s, 5) for s in tp_scores],
                    "fp_score_summary": {
                        "max": round(max([c["score"] for c in fp_cands]), 5) if fp_cands else 0.0,
                        "mean": round(float(np.mean([c["score"] for c in fp_cands])), 5) if fp_cands else 0.0,
                        "band_ge_095": sum(1 for c in fp_cands if c["score"] >= 0.95),
                        "band_085_095": sum(1 for c in fp_cands if 0.85 <= c["score"] < 0.95),
                        "band_070_085": sum(1 for c in fp_cands if 0.70 <= c["score"] < 0.85),
                        "band_050_070": sum(1 for c in fp_cands if 0.50 <= c["score"] < 0.70),
                        "band_020_050": sum(1 for c in fp_cands if 0.20 <= c["score"] < 0.50),
                        "band_lt_020": sum(1 for c in fp_cands if c["score"] < 0.20),
                    },
                    "mined_negatives": [
                        {
                            "tid": c["tid"],
                            "score": round(c["score"], 5),
                            "categories": c["categories"],
                            "features": [round(float(v), 5) for v in c["features"]],
                        }
                        for c in selected_mined
                    ],
                }
                batch_checkpoint_records.append(rec)
                mined_records_all.append(rec)
                completed_sids.add(sid)

            # Save batch checkpoint
            if batch_checkpoint_records:
                tmp_path = cp_dir / f"batch_{b_idx:05d}.jsonl.tmp"
                final_path = cp_dir / f"batch_{b_idx:05d}.jsonl"
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    for r in batch_checkpoint_records:
                        fh.write(json.dumps(r) + "\n")
                if final_path.exists():
                    final_path.unlink()
                tmp_path.rename(final_path)

        if len(target_cache) > 25000:
            target_cache.clear()
            gc.collect()

        processed_so_far = len(completed_sids)
        if (b_idx + 1) % 5 == 0 or (b_idx + 1) == num_batches:
            elapsed = time.perf_counter() - t0_eval
            qps = (processed_so_far - num_resumed) / elapsed if elapsed > 0 else 0
            cur_ram = get_process_ram_mb()
            sys_ram = psutil.virtual_memory().used / (1024 * 1024)
            print(f"  [{country}] Mining Progress: {processed_so_far:,}/{len(queries):,} queries ({(processed_so_far/len(queries))*100:.1f}%) | "
                  f"{qps:.1f} q/s | Worker RAM: {cur_ram:.1f} MB | Sys RAM: {sys_ram:.1f} MB", flush=True)

    total_mining_time = time.perf_counter() - t0_eval

    for fh in file_handles:
        fh.close()

    del blocker
    del target_offsets
    del target_cache
    gc.collect()

    profile = {
        "indexed_count": indexed_count,
        "index_build_sec": idx_time,
        "mining_sec": total_mining_time,
        "peak_worker_ram_mb": peak_ram_post_idx,
    }

    return mined_records_all, profile


# ---------------------------------------------------------------------------
# STAGE 2: Assemble Training Corpus and Train Full-Universe Models
# ---------------------------------------------------------------------------

def assemble_and_train_models(
    mined_records_all: list[dict],
    base_pairs_path: Path,
    out_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    seed: int = 2026,
) -> dict:
    """
    Builds pairwise_dev_train_fulluniverse_hardneg.npz and trains Full-Universe HardNeg 1x and 2x models.
    """
    print("\n" + "=" * 80)
    print("STAGE 2: ASSEMBLING FULL-UNIVERSE HARD-NEGATIVE TRAINING CORPUS...")
    print("=" * 80, flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load original positive pairs and background negatives
    print("Loading baseline positive pairs and background negatives from pairwise_expanded_dev_train.npz...", flush=True)
    orig_npz = np.load(base_pairs_path, allow_pickle=True)
    orig_X = orig_npz["X"]
    orig_y = orig_npz["y"]
    orig_s1 = orig_npz["s1_ids"]
    orig_tgt = orig_npz["target_ids"]

    pos_mask = (orig_y == 1)
    neg_mask = (orig_y == 0)

    X_pos = orig_X[pos_mask]
    y_pos = orig_y[pos_mask]
    s1_pos = orig_s1[pos_mask]
    tgt_pos = orig_tgt[pos_mask]
    n_pos = len(y_pos)
    print(f"  Loaded {n_pos:,} true positive pairs.")

    # Sample controlled easy background negatives (~100,000)
    rng = np.random.default_type() if hasattr(np.random, "default_type") else np.random.RandomState(seed)
    neg_indices = np.where(neg_mask)[0]
    n_bg_sample = min(100000, len(neg_indices))
    selected_bg_idx = rng.choice(neg_indices, size=n_bg_sample, replace=False)

    X_bg = orig_X[selected_bg_idx]
    y_bg = orig_y[selected_bg_idx]
    s1_bg = orig_s1[selected_bg_idx]
    tgt_bg = orig_tgt[selected_bg_idx]
    print(f"  Sampled {n_bg_sample:,} controlled background easy negatives.")

    # 2. Extract mined hard and medium negatives
    mined_X = []
    mined_s1 = []
    mined_tgt = []
    mined_scores = []
    mined_categories = Counter()
    category_counts_per_neg = defaultdict(list)
    query_neg_counts = []
    queries_with_zero = 0
    queries_with_max = 0

    all_tp_scores = []
    all_fp_scores_by_band = Counter()
    total_fp_evaluated = 0

    for r in mined_records_all:
        sid = r["entity_id"]
        all_tp_scores.extend(r["tp_scores"])
        f_sum = r["fp_score_summary"]
        total_fp_evaluated += r["fp_count"]
        for b_name in ["band_ge_095", "band_085_095", "band_070_085", "band_050_070", "band_020_050", "band_lt_020"]:
            all_fp_scores_by_band[b_name] += f_sum[b_name]

        negs = r["mined_negatives"]
        q_count = len(negs)
        query_neg_counts.append(q_count)
        if q_count == 0:
            queries_with_zero += 1
        if q_count >= 15:
            queries_with_max += 1

        for item in negs:
            mined_X.append(item["features"])
            mined_s1.append(sid)
            mined_tgt.append(item["tid"])
            mined_scores.append(item["score"])
            for cat in item["categories"]:
                mined_categories[cat] += 1

    mined_X = np.array(mined_X, dtype=np.float32)
    mined_s1 = np.array(mined_s1)
    mined_tgt = np.array(mined_tgt)
    mined_y = np.zeros(len(mined_s1), dtype=np.int32)
    n_mined = len(mined_y)

    print(f"  Mined {n_mined:,} full-universe false positives from 10,000 dev_train queries.")
    print(f"  Query negative count distribution: median={np.median(query_neg_counts):.1f}, "
          f"P95={np.percentile(query_neg_counts, 95):.1f}, P99={np.percentile(query_neg_counts, 99):.1f}, max={max(query_neg_counts)}")
    print(f"  Queries contributing 0 negatives: {queries_with_zero:,} | Queries contributing max (15): {queries_with_max:,}")

    # 3. Save Score Distribution & Category Analysis Reports
    score_dist_report = {
        "total_queries": len(mined_records_all),
        "total_tp_evaluated": len(all_tp_scores),
        "total_fp_evaluated": total_fp_evaluated,
        "tp_scores": {
            "mean": float(np.mean(all_tp_scores)) if all_tp_scores else 0.0,
            "std": float(np.std(all_tp_scores)) if all_tp_scores else 0.0,
            "min": float(np.min(all_tp_scores)) if all_tp_scores else 0.0,
            "p10": float(np.percentile(all_tp_scores, 10)) if all_tp_scores else 0.0,
            "p25": float(np.percentile(all_tp_scores, 25)) if all_tp_scores else 0.0,
            "p50": float(np.percentile(all_tp_scores, 50)) if all_tp_scores else 0.0,
            "p75": float(np.percentile(all_tp_scores, 75)) if all_tp_scores else 0.0,
            "p90": float(np.percentile(all_tp_scores, 90)) if all_tp_scores else 0.0,
            "p95": float(np.percentile(all_tp_scores, 95)) if all_tp_scores else 0.0,
            "max": float(np.max(all_tp_scores)) if all_tp_scores else 0.0,
        },
        "fp_scores_by_band": dict(all_fp_scores_by_band),
        "fp_scores_by_band_pct": {
            k: (v / total_fp_evaluated * 100) if total_fp_evaluated > 0 else 0.0
            for k, v in all_fp_scores_by_band.items()
        },
        "mining_score_bands_used": {
            "hard_negatives": "score >= 0.50 (capped at 10/query)",
            "medium_negatives": "0.20 <= score < 0.50 (capped at 5/query)",
            "background_negatives": "sampled from original background population",
        }
    }
    with open(reports_dir / "phase4_3c_score_distribution.json", "w", encoding="utf-8") as f:
        json.dump(score_dist_report, f, indent=2)
    print(f"Saved score distribution report to: {reports_dir / 'phase4_3c_score_distribution.json'}")

    category_report = {
        "total_mined_negatives": n_mined,
        "category_counts": dict(mined_categories),
        "category_percentages": {
            k: (v / n_mined * 100) if n_mined > 0 else 0.0
            for k, v in mined_categories.items()
        },
        "query_distribution": {
            "mean": float(np.mean(query_neg_counts)),
            "median": float(np.median(query_neg_counts)),
            "p95": float(np.percentile(query_neg_counts, 95)),
            "p99": float(np.percentile(query_neg_counts, 99)),
            "max": int(max(query_neg_counts)),
            "queries_with_zero": queries_with_zero,
            "queries_with_max": queries_with_max,
        }
    }
    with open(reports_dir / "phase4_3c_category_analysis.json", "w", encoding="utf-8") as f:
        json.dump(category_report, f, indent=2)
    print(f"Saved category analysis report to: {reports_dir / 'phase4_3c_category_analysis.json'}")

    # 4. Construct Dataset 1: Full-Universe HardNeg 1x
    # Combines positives + background negatives + 1x mined negatives
    X_1x = np.vstack([X_pos, X_bg, mined_X])
    y_1x = np.concatenate([y_pos, y_bg, mined_y])
    s1_1x = np.concatenate([s1_pos, s1_bg, mined_s1])
    tgt_1x = np.concatenate([tgt_pos, tgt_bg, mined_tgt])

    ds_1x_path = out_dir / "pairwise_dev_train_fulluniverse_hardneg.npz"
    np.savez_compressed(
        ds_1x_path,
        X=X_1x,
        y=y_1x,
        s1_ids=s1_1x,
        target_ids=tgt_1x,
        feature_names=ALL_FEATURE_NAMES,
    )
    print(f"Saved Full-Universe HardNeg 1x dataset to: {ds_1x_path} ({X_1x.shape[0]:,} pairs).")

    # 5. Train Full-Universe HardNeg 1x Model
    print(f"\nTraining LightGBM on Full-Universe HardNeg 1x ({X_1x.shape[0]:,} pairs)...", flush=True)
    t0_tr = time.perf_counter()
    model_1x = PairwiseLightGBMMatcher(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
    )
    model_1x.fit(X_1x, y_1x, feature_names=ALL_FEATURE_NAMES)
    t_tr_1x = time.perf_counter() - t0_tr
    model_1x_path = models_dir / "matcher_fulluniverse_hardneg_1x.pkl"
    model_1x.save(model_1x_path)
    print(f"Trained Full-Universe HardNeg 1x in {t_tr_1x:.2f} s. Saved to: {model_1x_path}")

    trained_models = {
        "fulluniverse_hardneg_1x": model_1x,
    }

    # 6. Train Full-Universe HardNeg 2x (double weight on mined negatives by replication)
    print(f"\nTraining LightGBM on Full-Universe HardNeg 2x (replicating mined hard negatives)...", flush=True)
    X_2x = np.vstack([X_pos, X_bg, mined_X, mined_X])
    y_2x = np.concatenate([y_pos, y_bg, mined_y, mined_y])
    s1_2x = np.concatenate([s1_pos, s1_bg, mined_s1, mined_s1])
    tgt_2x = np.concatenate([tgt_pos, tgt_bg, mined_tgt, mined_tgt])

    t0_tr_2x = time.perf_counter()
    model_2x = PairwiseLightGBMMatcher(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
    )
    model_2x.fit(X_2x, y_2x, feature_names=ALL_FEATURE_NAMES)
    t_tr_2x = time.perf_counter() - t0_tr_2x
    model_2x_path = models_dir / "matcher_fulluniverse_hardneg_2x.pkl"
    model_2x.save(model_2x_path)
    print(f"Trained Full-Universe HardNeg 2x in {t_tr_2x:.2f} s. Saved to: {model_2x_path}")
    trained_models["fulluniverse_hardneg_2x"] = model_2x

    meta = {
        "dataset_1x_pairs": int(X_1x.shape[0]),
        "dataset_2x_pairs": int(X_2x.shape[0]),
        "positives_count": int(n_pos),
        "background_negatives_count": int(n_bg_sample),
        "mined_negatives_count": int(n_mined),
        "train_time_1x_sec": t_tr_1x,
        "train_time_2x_sec": t_tr_2x,
    }
    return trained_models, meta


# ---------------------------------------------------------------------------
# STAGE 3: Full-Scale Development Evaluation on dev_val
# ---------------------------------------------------------------------------

def run_eval_partition(
    country: str,
    val_queries_raw: list[tuple[str, str, str, str]],
    target_tsv_paths: list[Path],
    eval_models: dict[str, PairwiseLightGBMMatcher],
    extractor: PairwiseFeatureExtractor,
    pipeline: NormalizationPipeline,
    cp_base_dir: Path,
    batch_size: int = 250,
) -> tuple[dict[str, dict], dict]:
    """
    Evaluates newly trained full-universe models on dev_val against the complete 10.32M target universe.
    """
    print(f"\n{'=' * 80}")
    print(f"STAGE 3: EVALUATION PARTITION — {country.upper()}")
    print(f"Validation Queries: {len(val_queries_raw):,} | Batch Size: {batch_size}")
    print(f"{'=' * 80}", flush=True)

    cp_dir = get_checkpoint_dir(cp_base_dir, country)
    existing_checkpoints = {}
    for bf in sorted(cp_dir.glob("batch_*.jsonl")):
        with open(bf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_checkpoints[item["entity_id"]] = item

    num_resumed = len(existing_checkpoints)
    if num_resumed > 0:
        print(f"[{country}] Found {num_resumed:,} completed eval queries from previous checkpoints. Resuming...")

    queries = [pipeline.normalize(eid, n, a, c) for eid, n, a, c in val_queries_raw]

    print(f"[{country}] Building CompositeBlocker index against 100% real targets...", flush=True)
    t0_idx = time.perf_counter()
    blocker = CompositeBlocker(max_total_candidates=175)
    target_offsets: dict[str, int] = {}

    def stream_targets():
        for f_idx, path in enumerate(target_tsv_paths):
            with open(path, "rb") as bf:
                bf.readline()
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
                        yield pipeline.normalize(tid, name, addr, country)

    indexed_count = blocker.index_target_records(stream_targets())
    idx_time = time.perf_counter() - t0_idx
    peak_ram_post_idx = get_process_ram_mb()
    print(f"[{country}] Indexed {indexed_count:,} target records in {idx_time:.2f} s. Peak Worker RAM: {peak_ram_post_idx:.1f} MB.", flush=True)

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

    target_cache: dict[str, BusinessRecord] = {}
    all_query_results: dict[str, dict] = dict(existing_checkpoints)

    num_batches = (len(queries) + batch_size - 1) // batch_size
    t0_eval = time.perf_counter()

    for b_idx in range(num_batches):
        batch_q = queries[b_idx * batch_size : (b_idx + 1) * batch_size]

        if all(q.entity_id in all_query_results for q in batch_q):
            continue

        batch_pairs = []
        batch_meta = []
        batch_candidates_dict = {}
        batch_cross_script_flags = defaultdict(list)

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

        # Extract features and predict with all eval models
        sid_to_cand_scores = {m_key: defaultdict(dict) for m_key in eval_models}

        if batch_pairs:
            feats = extractor.extract_batch(batch_pairs)
            for m_key, model in eval_models.items():
                probs = model.predict_proba(feats)
                for (sid, tid), score in zip(batch_meta, probs):
                    sid_to_cand_scores[m_key][sid][tid] = float(score)

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
            for m_key in eval_models:
                scores_by_model[m_key] = [
                    round(sid_to_cand_scores[m_key][sid].get(tid, 0.0), 5)
                    for tid in cands
                ]

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

        if batch_checkpoint_records:
            tmp_path = cp_dir / f"batch_{b_idx:05d}.jsonl.tmp"
            final_path = cp_dir / f"batch_{b_idx:05d}.jsonl"
            with open(tmp_path, "w", encoding="utf-8") as f:
                for rec in batch_checkpoint_records:
                    f.write(json.dumps(rec) + "\n")
            if final_path.exists():
                final_path.unlink()
            tmp_path.rename(final_path)

        if len(target_cache) > 25000:
            target_cache.clear()
            gc.collect()

        processed_so_far = len(all_query_results)
        if (b_idx + 1) % 5 == 0 or (b_idx + 1) == num_batches:
            elapsed = time.perf_counter() - t0_eval
            qps = (processed_so_far - num_resumed) / elapsed if elapsed > 0 else 0
            cur_ram = get_process_ram_mb()
            sys_ram = psutil.virtual_memory().used / (1024 * 1024)
            print(f"  [{country}] Eval Progress: {processed_so_far:,}/{len(queries):,} queries ({(processed_so_far/len(queries))*100:.1f}%) | "
                  f"{qps:.1f} q/s | Worker RAM: {cur_ram:.1f} MB | Sys RAM: {sys_ram:.1f} MB", flush=True)

    total_eval_time = time.perf_counter() - t0_eval

    for fh in file_handles:
        fh.close()

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

    return all_query_results, profile


# ---------------------------------------------------------------------------
# Metric Aggregations Across Tested Thresholds [0.90 ... 0.99]
# ---------------------------------------------------------------------------

def compute_all_metrics(
    eval_results: dict[str, dict],
    gt: dict[str, frozenset[str]],
    model_keys: list[str],
) -> dict:
    """
    Computes comprehensive threshold sweeps across [0.90 ... 0.99], country breakdowns,
    script breakdowns, cardinality breakdowns, singleton score distributions, and error decompositions.
    """
    print("\n" + "=" * 80)
    print("COMPUTING COMPREHENSIVE BENCHMARK METRICS ACROSS THRESHOLD GRID [0.90 ... 0.99]...")
    print("=" * 80, flush=True)

    total_queries = len(eval_results)
    candidate_counts = [res["candidates_count"] for res in eval_results.values()]
    total_true_pairs = sum(len(gt.get(sid, frozenset())) for sid in eval_results)
    retrieved_true_pairs = 0
    for sid, res in eval_results.items():
        cands = set(res["candidates"])
        truth = gt.get(sid, frozenset())
        retrieved_true_pairs += len(truth.intersection(cands))

    cand_recall = (retrieved_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0

    metrics_out = {
        "blocker": {
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
        },
        "models": {},
    }

    for m_key in model_keys:
        m_dict = {
            "threshold_sweep": {},
            "country_sweep": {"India": {}, "US": {}},
            "error_decomposition": {},
            "singleton_score_distribution": {},
            "cardinality_analysis": {},
            "script_analysis": {},
            "cross_script_pairs": {},
        }

        # Analyze singleton scores
        top1_scores = []
        top2_scores = []
        margins = []

        for sid, res in eval_results.items():
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

        m_dict["singleton_score_distribution"] = {
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

        # Run threshold sweeps across [0.90 ... 0.99]
        for tau in THRESHOLDS_EVAL:
            tau_str = f"{tau:.3f}" if tau in [0.985] else f"{tau:.2f}"

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

            ctry_stats = {
                "India": {"f05": [], "p": [], "r": [], "tp": 0, "fp": 0, "fn": 0, "sing_tot": 0, "sing_cor": 0, "preds_tot": 0, "tot": 0},
                "US": {"f05": [], "p": [], "r": [], "tp": 0, "fp": 0, "fn": 0, "sing_tot": 0, "sing_cor": 0, "preds_tot": 0, "tot": 0},
            }
            card_stats = defaultdict(lambda: {"entities": 0, "f05_s": 0.0, "p_s": 0.0, "r_s": 0.0, "exact": 0})
            script_stats = defaultdict(lambda: {"entities": 0, "f05_s": 0.0, "p_s": 0.0, "r_s": 0.0, "fp": 0})

            cs_pairs_evaluated = 0
            cs_tp = 0
            cs_fp = 0
            cs_fn = 0

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

            for sid, res in eval_results.items():
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

                cs = ctry_stats[country]
                cs["tot"] += 1
                cs["f05"].append(f05_i)
                cs["p"].append(p_i)
                cs["r"].append(r_i)
                cs["tp"] += tp
                cs["fp"] += fp
                cs["fn"] += fn
                cs["preds_tot"] += len(preds)

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

                    gt_in_cands = truth & cands_set
                    if len(gt_in_cands) == 0:
                        decomp["zero_gt_retrieved"] += 1
                    elif len(gt_in_cands) < len(truth):
                        decomp["candidate_truncation"] += 1
                    else:
                        if preds == truth:
                            decomp["all_gt_retrieved_correct"] += 1
                        elif preds > truth:
                            decomp["all_gt_retrieved_false_positives"] += 1
                        elif preds < truth:
                            decomp["all_gt_retrieved_false_negatives"] += 1
                        else:
                            decomp["all_gt_retrieved_mixed_error"] += 1

                card = len(truth)
                bucket = str(card) if card < 8 else ">=8"
                c_bucket = card_stats[bucket]
                c_bucket["entities"] += 1
                c_bucket["f05_s"] += f05_i
                c_bucket["p_s"] += p_i
                c_bucket["r_s"] += r_i
                if preds == truth:
                    c_bucket["exact"] += 1

                s_bucket = script_stats[script]
                s_bucket["entities"] += 1
                s_bucket["f05_s"] += f05_i
                s_bucket["p_s"] += p_i
                s_bucket["r_s"] += r_i
                s_bucket["fp"] += fp

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
            m_dict["threshold_sweep"][tau_str] = thresh_res

            for ctry in ["India", "US"]:
                cs = ctry_stats[ctry]
                c_f05 = float(np.mean(cs["f05"]))
                c_p = float(np.mean(cs["p"]))
                c_r = float(np.mean(cs["r"]))
                c_pair_p = cs["tp"] / (cs["tp"] + cs["fp"]) if (cs["tp"] + cs["fp"]) > 0 else 0.0
                c_pair_r = cs["tp"] / (cs["tp"] + cs["fn"]) if (cs["tp"] + cs["fn"]) > 0 else 0.0
                c_sing_acc = cs["sing_cor"] / cs["sing_tot"] if cs["sing_tot"] > 0 else 1.0

                m_dict["country_sweep"][ctry][tau_str] = {
                    "macro_f05": c_f05,
                    "macro_precision": c_p,
                    "macro_recall": c_r,
                    "pair_precision": c_pair_p,
                    "pair_recall": c_pair_r,
                    "singleton_accuracy": c_sing_acc,
                    "mean_fp_per_query": cs["fp"] / cs["tot"],
                    "mean_predicted_matches": cs["preds_tot"] / cs["tot"],
                }

            cs_prec = cs_tp / (cs_tp + cs_fp) if (cs_tp + cs_fp) > 0 else 0.0
            cs_rec = cs_tp / (cs_tp + cs_fn) if (cs_tp + cs_fn) > 0 else 0.0
            cs_f05 = (1.25 * cs_prec * cs_rec) / (0.25 * cs_prec + cs_rec) if (0.25 * cs_prec + cs_rec) > 0 else 0.0
            m_dict["cross_script_pairs"][tau_str] = {
                "cross_script_pairs_evaluated": cs_pairs_evaluated,
                "cross_script_tp": cs_tp,
                "cross_script_fp": cs_fp,
                "cross_script_fn": cs_fn,
                "cross_script_precision": cs_prec,
                "cross_script_recall": cs_rec,
                "cross_script_f05": cs_f05,
            }

            m_dict["error_decomposition"][tau_str] = decomp

            m_dict["cardinality_analysis"][tau_str] = {
                b: {
                    "entities": card_stats[b]["entities"],
                    "macro_f05": card_stats[b]["f05_s"] / card_stats[b]["entities"] if card_stats[b]["entities"] > 0 else 0.0,
                    "macro_p": card_stats[b]["p_s"] / card_stats[b]["entities"] if card_stats[b]["entities"] > 0 else 0.0,
                    "macro_r": card_stats[b]["r_s"] / card_stats[b]["entities"] if card_stats[b]["entities"] > 0 else 0.0,
                    "exact_match_pct": (card_stats[b]["exact"] / card_stats[b]["entities"] * 100) if card_stats[b]["entities"] > 0 else 0.0,
                }
                for b in ["0", "1", "2", "3", "4", "5", "6", "7", ">=8"] if b in card_stats
            }

            m_dict["script_analysis"][tau_str] = {
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

        metrics_out["models"][m_key] = m_dict

    return metrics_out


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4.3C Full-Universe Hard-Negative Mining.")
    parser.add_argument("--data_dir", type=str, default="../6ab10eb3b23ba_student_resource/student_resource/dataset/train")
    parser.add_argument("--mining_ids_file", type=str, default="data/pairs/phase4_3c/mining_query_ids.txt")
    parser.add_argument("--val_ids_file", type=str, default="data/validation/phase4_3b_dev_val_10k_ids.txt")
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--mining_cp_dir", type=str, default="output/checkpoints_mining_4_3c")
    parser.add_argument("--eval_cp_dir", type=str, default="output/checkpoints_eval_4_3c")
    parser.add_argument("--output_metrics", type=str, default="reports/phase4_3c_full_universe_mining_metrics.json")
    parser.add_argument("--csv_sweep_output", type=str, default="reports/phase4_3c_threshold_sweep.csv")
    parser.add_argument("--pairs_dir", type=str, default="data/pairs/phase4_3c")
    parser.add_argument("--models_dir", type=str, default="models/phase4_3c")
    parser.add_argument("--reports_dir", type=str, default="reports")
    parser.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    data_dir = Path(args.data_dir)
    mining_ids_file = Path(args.mining_ids_file)
    val_ids_file = Path(args.val_ids_file)
    pairs_dir = Path(args.pairs_dir)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    pipeline = NormalizationPipeline()

    print("=" * 80)
    print("PHASE 4.3C — FULL-UNIVERSE HARD-NEGATIVE MINING AGAINST 10.32M TARGETS")
    print("=" * 80, flush=True)

    # 1. Strict Data Isolation Audit
    print("\n--- STEP 1: STRICT DATA ISOLATION AUDIT ---")
    with open("data/splits/val_ids.txt", "r", encoding="utf-8") as f:
        official_val_ids = set(line.strip() for line in f if line.strip())
    with open("data/splits/dev_val_ids.txt", "r", encoding="utf-8") as f:
        dev_val_ids = set(line.strip() for line in f if line.strip())
    with open("data/splits/dev_train_ids.txt", "r", encoding="utf-8") as f:
        dev_train_ids = set(line.strip() for line in f if line.strip())

    with open(mining_ids_file, "r", encoding="utf-8") as f:
        mining_sids = [line.strip() for line in f if line.strip()]
    mining_sids_set = set(mining_sids)

    with open(val_ids_file, "r", encoding="utf-8") as f:
        val_sids = [line.strip() for line in f if line.strip()]
    val_sids_set = set(val_sids)

    assert len(mining_sids_set.intersection(official_val_ids)) == 0, "LEAKAGE: mining queries overlap with official val!"
    assert len(mining_sids_set.intersection(dev_val_ids)) == 0, "LEAKAGE: mining queries overlap with dev_val!"
    assert len(val_sids_set.intersection(official_val_ids)) == 0, "LEAKAGE: eval queries overlap with official val!"
    assert len(val_sids_set.intersection(dev_train_ids)) == 0, "LEAKAGE: eval queries overlap with dev_train!"
    assert len(mining_sids_set.intersection(val_sids_set)) == 0, "LEAKAGE: mining queries overlap with eval queries!"
    print("Data isolation 100% verified clean: 0 overlap between dev_train mining and dev_val/official_val.", flush=True)

    # 2. Load Ground Truth for mining and validation queries
    all_needed_sids = mining_sids_set | val_sids_set
    gt: dict[str, frozenset[str]] = {}
    with open(data_dir / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0].strip() in all_needed_sids:
                targets = frozenset(t.strip() for t in row[1].split(",") if t.strip())
                gt[row[0].strip()] = targets

    for sid in all_needed_sids:
        if sid not in gt:
            gt[sid] = frozenset()

    # 3. Stream queries and partition by country
    mining_us: list[tuple[str, str, str, str]] = []
    mining_india: list[tuple[str, str, str, str]] = []
    val_us: list[tuple[str, str, str, str]] = []
    val_india: list[tuple[str, str, str, str]] = []

    print("Streaming queries from train_source1.tsv...", flush=True)
    with open(data_dir / "train_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                sid, name, addr, ctry = row[0].strip(), row[1], row[2], row[3].strip()
                if sid in mining_sids_set:
                    if ctry == "India":
                        mining_india.append((sid, name, addr, ctry))
                    elif ctry == "US":
                        mining_us.append((sid, name, addr, ctry))
                elif sid in val_sids_set:
                    if ctry == "India":
                        val_india.append((sid, name, addr, ctry))
                    elif ctry == "US":
                        val_us.append((sid, name, addr, ctry))

    print(f"Mining queries: India={len(mining_india):,}, US={len(mining_us):,} (Total={len(mining_india)+len(mining_us):,})")
    print(f"Val queries:    India={len(val_india):,}, US={len(val_us):,} (Total={len(val_india)+len(val_us):,})", flush=True)

    # Load Teacher Model and Extractor
    print("\nLoading teacher model (HardNeg 1x) and feature extractor...", flush=True)
    teacher_model = PairwiseLightGBMMatcher.load(Path("models/phase4_3/matcher_hardneg_1x.pkl"))
    cs_path = Path("data/pairs/cross_script_aligner_dev_train.json")
    cs_aligner = CrossScriptAligner.load(cs_path)
    extractor = PairwiseFeatureExtractor(cross_script_aligner=cs_aligner, include_cross_script=True)

    target_tsv_paths = [data_dir / "train_source2.tsv", data_dir / "train_source3.tsv"]
    mining_cp_dir = Path(args.mining_cp_dir)

    telemetry = SystemTelemetryMonitor(interval=2.0)
    telemetry.start()
    t0_global = time.perf_counter()

    # 4. STAGE 1: Run Full-Universe Mining for India and US
    india_mined, india_mine_prof = run_mining_partition(
        country="India",
        mining_queries_raw=mining_india,
        target_tsv_paths=target_tsv_paths,
        teacher_model=teacher_model,
        extractor=extractor,
        pipeline=pipeline,
        gt=gt,
        cp_base_dir=mining_cp_dir,
        batch_size=args.batch_size,
    )

    us_mined, us_mine_prof = run_mining_partition(
        country="US",
        mining_queries_raw=mining_us,
        target_tsv_paths=target_tsv_paths,
        teacher_model=teacher_model,
        extractor=extractor,
        pipeline=pipeline,
        gt=gt,
        cp_base_dir=mining_cp_dir,
        batch_size=args.batch_size,
    )

    all_mined_records = india_mined + us_mined
    assert len(all_mined_records) == 10000, f"Expected 10,000 mined query records, got {len(all_mined_records)}"

    # 5. STAGE 2: Assemble Training Dataset and Train Full-Universe Matchers
    base_pairs_path = Path("data/pairs/pairwise_expanded_dev_train.npz")
    trained_models, train_meta = assemble_and_train_models(
        mined_records_all=all_mined_records,
        base_pairs_path=base_pairs_path,
        out_dir=pairs_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        seed=args.seed,
    )

    # Also include existing models for direct benchmark comparison
    eval_models = {
        "baseline": PairwiseLightGBMMatcher.load(Path("models/matching_global/matcher_lightgbm.pkl")),
        "hardneg_1x": teacher_model,
        "fulluniverse_hardneg_1x": trained_models["fulluniverse_hardneg_1x"],
        "fulluniverse_hardneg_2x": trained_models["fulluniverse_hardneg_2x"],
    }

    # 6. STAGE 3: Full-Scale Development Evaluation on dev_val
    eval_cp_dir = Path(args.eval_cp_dir)
    india_eval, india_eval_prof = run_eval_partition(
        country="India",
        val_queries_raw=val_india,
        target_tsv_paths=target_tsv_paths,
        eval_models=eval_models,
        extractor=extractor,
        pipeline=pipeline,
        cp_base_dir=eval_cp_dir,
        batch_size=args.batch_size,
    )

    us_eval, us_eval_prof = run_eval_partition(
        country="US",
        val_queries_raw=val_us,
        target_tsv_paths=target_tsv_paths,
        eval_models=eval_models,
        extractor=extractor,
        pipeline=pipeline,
        cp_base_dir=eval_cp_dir,
        batch_size=args.batch_size,
    )

    all_eval_results = {**india_eval, **us_eval}
    assert len(all_eval_results) == 10000, f"Expected 10,000 eval results, got {len(all_eval_results)}"

    total_wall_sec = time.perf_counter() - t0_global
    sys_telemetry = telemetry.stop()

    # 7. STAGE 4: Compute All Metrics & Sweeps across [0.90 ... 0.99]
    final_metrics = compute_all_metrics(
        eval_results=all_eval_results,
        gt=gt,
        model_keys=list(eval_models.keys()),
    )

    final_metrics["experiment_meta"] = {
        "phase": "4.3C",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        "mining_queries": len(all_mined_records),
        "val_queries": len(all_eval_results),
        "total_target_records": india_mine_prof["indexed_count"] + us_mine_prof["indexed_count"],
        "india_targets": india_mine_prof["indexed_count"],
        "us_targets": us_mine_prof["indexed_count"],
        "total_wall_sec": total_wall_sec,
        "mining_profiles": {"India": india_mine_prof, "US": us_mine_prof},
        "eval_profiles": {"India": india_eval_prof, "US": us_eval_prof},
        "training_metadata": train_meta,
        "telemetry": sys_telemetry,
    }

    # Verify target count
    expected_targets = 10320219
    actual_targets = final_metrics["experiment_meta"]["total_target_records"]
    assert actual_targets == expected_targets, f"Target count mismatch! Expected {expected_targets}, got {actual_targets}"

    # 8. Save Final Metrics JSON
    metrics_path = Path(args.output_metrics)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved full validation metrics to: {metrics_path}")

    # 9. Save CSV Threshold Sweep
    csv_path = Path(args.csv_sweep_output)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "threshold", "macro_f05", "macro_precision", "macro_recall",
            "pair_precision", "pair_recall", "complete_set_recall", "singleton_accuracy",
            "mean_fp_per_query", "mean_predicted_matches", "delta_f05_vs_baseline"
        ])
        base_sweep = final_metrics["models"]["baseline"]["threshold_sweep"]
        for m_key in eval_models:
            m_sweep = final_metrics["models"][m_key]["threshold_sweep"]
            for tau in THRESHOLDS_EVAL:
                tau_str = f"{tau:.3f}" if tau in [0.985] else f"{tau:.2f}"
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

    # 10. Print Headline Comparison Table
    print("\n" + "=" * 105)
    print("PHASE 4.3C HEADLINE COMPARISON: FULL-UNIVERSE HARD-NEGATIVE MINING (10.32M REAL TARGET UNIVERSE)")
    print("=" * 105)
    print(f"{'Model':<26} {'Thresh':<8} {'Macro F0.5':<12} {'Macro Prec':<12} {'Macro Rec':<12} {'Pair Prec':<12} {'Sing Acc':<12} {'FP/Query':<10}")
    print("-" * 105)

    for m_key in eval_models:
        sweep = final_metrics["models"][m_key]["threshold_sweep"]
        # Print at 0.92
        if "0.92" in sweep:
            d92 = sweep["0.92"]
            print(f"{m_key:<26} {'0.92':<8} {d92['macro_f05']:<12.4f} {d92['macro_precision']:<12.4f} {d92['macro_recall']:<12.4f} {d92['pair_precision']:<12.4f} {d92['singleton_accuracy']:<12.4f} {d92['mean_fp_per_query']:<10.4f}")

        # Print at best tau
        best_tau = max(THRESHOLDS_EVAL, key=lambda t: sweep[f"{t:.3f}" if t in [0.985] else f"{t:.2f}"]["macro_f05"])
        best_str = f"{best_tau:.3f}" if best_tau in [0.985] else f"{best_tau:.2f}"
        bm = sweep[best_str]
        print(f"{m_key + ' (opt)':<26} {best_str:<8} {bm['macro_f05']:<12.4f} {bm['macro_precision']:<12.4f} {bm['macro_recall']:<12.4f} {bm['pair_precision']:<12.4f} {bm['singleton_accuracy']:<12.4f} {bm['mean_fp_per_query']:<10.4f}")
        print("-" * 105)

    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
