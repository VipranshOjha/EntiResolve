# Phase 4.5 Official Full-Scale Validation Report
## Single-Shot Benchmark of the Frozen Complementary Matcher Ensemble

**Date:** September 26–27, 2026  
**Status:** **PASSED / PRODUCTION CERTIFIED**  
**Query Population:** `data/splits/val_ids.txt` (331,022 official validation entities)  
**Target Universe:** Complete real Source 2 + Source 3 universe (10,320,219 records)  
**Architecture:** Frozen Phase 4.4B Complementary Dual-LightGBM Ensemble  
- Model A: `models/phase4_3/matcher_hardneg_1x.pkl` (Sampled HardNeg 1x, $\alpha = 0.88$)  
- Model B: `models/phase4_3c/matcher_fulluniverse_hardneg_1x.pkl` (Full-Universe HardNeg 1x, $1-\alpha = 0.12$)  
- Blocker: `CompositeBlocker(max_total_candidates=175)`  
- Decision Threshold: $\tau = 0.915$  
- Determinism: `PYTHONHASHSEED = 2026`  

---

## 1. Executive Summary

Phase 4.5 executed a single-shot, un-tuned official validation across all **331,022 official validation queries** evaluated against the entire **10,320,219-record real target universe**.

The frozen ensemble achieved a headline **Macro $F_{0.5}$ of 0.8270**, outperforming the Phase 4.4B independent holdout confirmation (**0.8199**) by **+0.0071**. This proves that the frozen complementary blend of high-diversity sampled hard negatives and full-universe scale-rejection models generalizes with zero overfitting across massive target universes.

### Official Headline Results

| Metric | Target / Benchmark | Phase 4.5 Result | Status |
|:---|:---:|:---:|:---:|
| **Macro $F_{0.5}$** | $\ge 0.8100$ | **0.8270** | **EXCEEDED (+0.0170)** |
| **Macro Precision** | $\ge 0.9000$ | **0.9172** | **EXCEEDED (+0.0172)** |
| **Macro Recall** | $\ge 0.7400$ | **0.7600** | **EXCEEDED (+0.0200)** |
| **Pair Precision** | $\ge 0.8900$ | **0.9123** | **EXCEEDED (+0.0223)** |
| **Pair Recall** | $\ge 0.7500$ | **0.7653** | **EXCEEDED (+0.0153)** |
| **Candidate Pair Recall ($K=175$)** | $\ge 85.0\%$ | **87.21%** | **EXCEEDED (+2.21%)** |
| **Singleton Accuracy** | $\ge 65.0\%$ | **69.06%** | **EXCEEDED (+4.06%)** |
| **Complete-Set Recall** | $\ge 32.0\%$ | **36.26%** | **EXCEEDED (+4.26%)** |
| **Mean False Positives / Query** | $\le 0.3000$ | **0.2548** | **OUTPERFORMED (-0.0452)** |
| **True Positives (TP)** | — | **877,040** | — |
| **False Positives (FP)** | — | **84,335** | — |
| **False Negatives (FN)** | — | **268,939** | — |

---

## 2. Methodology Progression Across Phases

The table below demonstrates the cumulative progression of our entity resolution architecture against the full 10.32M target universe:

| Phase | Architecture / Strategy | Operating Threshold ($\tau$) | Pair Precision | Pair Recall | Singleton Accuracy | Macro $F_{0.5}$ | $\Delta F_{0.5}$ vs Baseline |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **4.1** | Heuristic Baseline Matcher | 0.920 | 0.7321 | 0.6510 | 48.20% | 0.7070 | Baseline |
| **4.3B** | Sampled HardNeg 1x (Model A) | 0.970 | 0.8936 | 0.7447 | 64.71% | 0.8069 | +0.0999 |
| **4.3C** | Full-Universe HardNeg 1x (Model B) | 0.900 | 0.9245 | 0.6640 | 73.90% | 0.7770 | +0.0700 |
| **4.4B** | Locked Blend Holdout ($0.88/0.12$) | 0.915 | 0.9090 | 0.7561 | 67.46% | 0.8199 | +0.1129 |
| **4.5** | **Official Validation (331k Entities)** | **0.915** | **0.9123** | **0.7653** | **69.06%** | **0.8270** | **+0.1200** |

---

## 3. Country Partition Breakdown

The official validation queries were partitioned country-first to optimize cache locality and ensure strict geographical candidate gating:

| Partition | Queries | Indexed Targets | True Positives | False Positives | False Negatives | Pair Precision | Pair Recall | Singleton Acc. | Macro $F_{0.5}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **India** | 132,478 | 4,133,346 | 332,663 | 37,600 | 126,456 | **0.8985** | **0.7246** | **66.97%** | **0.7921** |
| **US** | 198,544 | 6,186,873 | 544,377 | 46,735 | 142,483 | **0.9209** | **0.7926** | **70.46%** | **0.8504** |
| **Overall** | **331,022** | **10,320,219** | **877,040** | **84,335** | **268,939** | **0.9123** | **0.7653** | **69.06%** | **0.8270** |

---

## 4. Cardinality Stratification Analysis

Performance across true match cardinality strata confirms consistent discriminative power across both singleton entities and high-cardinality clusters:

| Match Strata | Query Count | Percentage | TP | FP | FN | Pair Precision | Pair Recall | Macro $F_{0.5}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (Singletons)** | 18,487 | 5.58% | 0 | 7,155 | 0 | 0.00% | 0.00% | **0.6906** (Acc: 69.1%) |
| **1 match** | 17,873 | 5.40% | 13,432 | 4,469 | 4,441 | 75.03% | 75.15% | **0.6639** |
| **2 matches** | 56,331 | 17.02% | 85,521 | 14,178 | 27,141 | 85.78% | 75.91% | **0.7910** |
| **3 matches** | 79,577 | 24.04% | 182,685 | 20,038 | 56,046 | 90.12% | 76.52% | **0.8359** |
| **4 matches** | 72,303 | 21.84% | 221,847 | 17,872 | 67,365 | 92.54% | 76.71% | **0.8591** |
| **5 matches** | 48,608 | 14.68% | 186,587 | 11,644 | 56,453 | 94.13% | 76.77% | **0.8738** |
| **6 matches** | 24,798 | 7.49% | 113,952 | 5,970 | 34,836 | 95.02% | 76.59% | **0.8815** |
| **7 matches** | 9,554 | 2.89% | 51,272 | 2,244 | 15,606 | 95.81% | 76.66% | **0.8903** |
| **$\ge 8$ matches** | 3,491 | 1.05% | 21,744 | 765 | 7,051 | 96.60% | 75.51% | **0.8905** |

---

## 5. Cross-Script Alignment Performance

Cross-script evaluation examined non-Latin and mixed-script queries (e.g., Devanagari, Gurmukhi, Tamil, Gujarati) mapped across script boundaries:
- **Cross-Script Candidate Pairs Evaluated:** **1,690,590 pairs**
- **True Positives (TP):** **38,718**
- **False Positives (FP):** **1,133**
- **False Negatives (FN):** **16,732**
- **Cross-Script Precision:** **97.16%**
- **Cross-Script Recall:** **69.83%**
- **Cross-Script $F_{0.5}$:** **0.9010**

This demonstrates that our phonetic transliteration mapping and cross-script features maintain high precision even under script divergence.

---

## 6. Error Decomposition

Comprehensive error decomposition across all 331,022 entities:

| Error Category | Query Count | Percentage | Description |
|:---|:---:|:---:|:---|
| **All-GT Retrieved & Predicted Correct** | 107,260 | 32.40% | Exact match set predicted with zero FP and zero FN |
| **All-GT Retrieved with False Positives** | 44,248 | 13.37% | Full recall achieved but over-predicted extra candidates |
| **All-GT Retrieved with False Negatives** | 49,504 | 14.95% | Partial match set predicted with zero false positives |
| **All-GT Retrieved with Mixed Errors** | 16,928 | 5.11% | Both FP and FN present despite all GT present in candidates |
| **Partial GT Retrieval by Blocker** | 87,176 | 26.34% | Blocker retrieved subset of true matches within $K=175$ |
| **Zero GT Retrieved by Blocker** | 7,419 | 2.24% | Blocker missed all true targets (extreme lexical/address divergence) |
| **Singleton Correctly Detected** | 12,768 | 3.86% | True singletons correctly predicted as empty match set (69.06% of singletons) |
| **Singleton False Merges** | 5,719 | 1.73% | True singletons incorrectly matched with 1+ targets |

---

## 7. Telemetry & Computational Performance

| Metric | India Partition | US Partition | Total / Combined |
|:---|:---:|:---:|:---:|
| **Indexed Target Records** | 4,133,346 | 6,186,873 | **10,320,219 records** |
| **Index Construction Time** | 1,072.5 s (17.8 min) | 1,234.4 s (20.6 min) | **2,306.9 s (38.4 min)** |
| **Query Evaluation Time** | 12,244.3 s (3.40 hrs) | 16,124.6 s (4.48 hrs) | **28,368.9 s (7.88 hrs)** |
| **Total Partition Wall Time** | 13,316.8 s (3.70 hrs) | 17,359.0 s (4.82 hrs) | **32,135.3 s (8.93 hrs)** |
| **Average Query Throughput** | 10.8 queries/sec | 12.3 queries/sec | **11.7 queries/sec** |
| **Peak Worker RAM** | 6,176.0 MB | 10,888.3 MB | **10,888.3 MB (~10.6 GB)** |
| **Peak System RAM** | — | — | **16,073.5 MB (~15.7 GB)** |
| **Mean CPU Utilization** | — | — | **11.08% (single worker)** |

---

## 8. Official Gate Confirmation

Phase 4.5 official validation has strictly met every acceptance gate:
1. **Single-shot evaluation on 331,022 official queries**: Confirmed.
2. **Evaluation against all 10,320,219 real target records**: Confirmed.
3. **Macro $F_{0.5} \ge 0.8100$**: **0.8270** achieved (+0.0170 above target).
4. **Pair Precision $\ge 0.8900$**: **0.9123** achieved (+0.0223 above target).
5. **Singleton Accuracy $\ge 65.0\%$**: **69.06%** achieved (+4.06% above target).
6. **Zero data contamination**: Confirmed (val_ids.txt only, strictly isolated).
7. **Production architecture locked**: Confirmed (Model A, Model B, $K=175$, blend 0.88/0.12, $\tau=0.915$).

**THE FROZEN ENSEMBLE IS FORMALLY CERTIFIED FOR TEST INFERENCE AND FINAL COMPETITION SUBMISSION.**
