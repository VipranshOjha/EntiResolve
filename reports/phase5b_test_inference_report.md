# Phase 5B: Full Test Inference & Submission Generation Report
**Amazon ML Challenge 2026: Business Entity Resolution**  
**Project:** EntiResolve Engine  
**Execution Date:** September 27, 2026  
**Status:** Certified & Submission Package Ready

---

## 1. Executive Summary

Phase 5B represents the final single-shot test execution of the frozen production pipeline across all **1,732,544 test Source-1 queries** against the complete **~9.97 million target universe** (Source 2 + Source 3).

Operating under a single-process 6-stream parallel architecture with thread-safe shared target indexes (`ThreadSafeTargetStore`), the inference pipeline processed the entire test universe in **10.37 wall-clock hours** with **0 failures**, generated all required leaderboard submission files, and passed the official challenge validator with **exit code 0 (`PASS`)**.

---

## 2. Test Execution Configuration

* **Query Set:** `dataset/test/test_source1.tsv` (1,732,544 records)
* **Target Universe:** `dataset/test/test_source2.tsv` (7,237,045 records) + `dataset/test/test_source3.tsv` (2,732,544 records) = **9,969,589 total targets**
* **Parallel Concurrency:** 6 query streams (`ThreadPoolExecutor`)
* **Batch Size:** 250 queries per stream
* **Target Indexing:** Thread-safe persistent SQLite stores (`ThreadSafeTargetStore`) with single shared in-memory `CompositeBlocker(K=175)`
* **Frozen Models:**
  * Model A: `models/matcher_hardneg_1x.pkl` ($\alpha = 0.88$)
  * Model B: `models/matcher_fulluniverse_hardneg_1x.pkl` ($1 - \alpha = 0.12$)
* **Decision Gate:** Score $S = 0.88 \cdot P_A + 0.12 \cdot P_B \ge 0.915$
* **Determinism:** `PYTHONHASHSEED = 2026`

---

## 3. Country Partition Breakdown

Partitions were processed sequentially to guarantee strict memory bounding:
$$\text{France } (259\text{k}) \longrightarrow \text{US } (663\text{k}) \longrightarrow \text{India } (810\text{k})$$

| Partition | S1 Queries | Target Records | Target Index Sec | Inference Sec | Throughput (q/s) | Candidates Emitted | Peak Worker RAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **France** | 259,452 | 1,434,993 | 194.12 s (3.2m) | 2,079.60 s (34.7m) | 82.2 q/s | 25,666,001 | 3,724.9 MB |
| **US** | 663,106 | 3,817,031 | 570.24 s (9.5m) | 7,417.72 s (2.06h) | 89.4 q/s | 89.402,909 | 8,910.2 MB |
| **India** | 809,986 | 4,717,565 | 897.83 s (15.0m) | 25,608.84 s (7.11h) | 31.6 q/s | 111,492,769 | 10,187.0 MB |
| **Total** | **1,732,544** | **9,969,589** | **1,662.19 s** | **35,106.16 s** | **46.4 q/s** | **239,826,802** | **10,762.8 MB** |

*Note: France resumed 88,500 queries from earlier atomic checkpoints and completed the remaining 170,952 queries in 34.66 minutes.*

---

## 4. End-to-End Execution Totals

* **Total Evaluated Entities:** 1,732,544
* **Total Matches Emitted:** 5,462,050
* **Total Candidates Emitted:** 239,826,802
* **Total Singletons (0 Matches):** 159,947 (9.23%)
* **Total Wall Clock Time:** **37,345.7 seconds (10.37 hours)**
* **Peak System RAM:** 16,059.9 MB
* **Peak Worker RAM:** 10,762.8 MB (stable, bounded across entire run)
* **Failures / Retries:** **0**

---

## 5. Output Verification & Official Validation

Both required submission files were generated in the exact original order of `test_source1.tsv`:
* `output/matching_results.tsv`: **88.59 MB** (exactly 1,732,545 lines: 1 header + 1,732,544 records)
* `output/candidate_pairs.tsv`: **2,969.23 MB** (exactly 1,732,545 lines: 1 header + 1,732,544 records)

### Official Validator Certification:
```text
ML Challenge 2026 — submission validator
  test dir: 6ab10eb3b23ba_student_resource/student_resource/dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (159947 empty, 1572597 non-empty).
  candidate_pairs.tsv:  1732544 rows (32 empty, 1732512 non-empty).

PASS — no blocking issues found. Safe to submit.
Exit Code: 0
```

### Packaging:
* Primary Archive: `antigravity_er_submission.zip` (**1,273.89 MB**)
* All required directory hierarchies, models, code, and documentation verified.
