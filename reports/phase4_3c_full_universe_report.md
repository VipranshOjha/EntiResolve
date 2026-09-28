# Phase 4.3C: Full-Universe Hard-Negative Mining Report
**Amazon ML Challenge 2026: Business Entity Resolution**  
**Project:** EntiResolve Engine  
**Component:** Model B (`models/matcher_fulluniverse_hardneg_1x.pkl`)

---

## 1. Executive Summary

While Phase 4.3 mined hard negatives within a subsampled training corpus, the real test evaluation queries against a target universe of **10.32 million records** (Source 2 + Source 3). At this scale, rare but lethal negative lures emerge from massive nationwide directories that never appear in smaller training samples.

In Phase 4.3C, we scaled the candidate mining engine across the **complete 10,320,219-record target universe** to train **Model B** (`matcher_fulluniverse_hardneg_1x.pkl`). Model B acts as an aggressive scale-rejection filter, eliminating false positive lures and boosting singleton accuracy to **73.90%**.

---

## 2. Full-Universe Mining Architecture

* **Target Universe:**
  * India Targets: 4,133,346 records
  * US Targets: 6,186,873 records
  * Total Real Targets: **10,320,219 records**
* **Query Set:** 10,000 stratified queries sampled strictly from `dev_train` (guaranteed 0 overlap with validation and holdout splits).
* **Mining Execution:**
  * For each query, `CompositeBlocker(K=175)` indexed and searched the entire 10.32M target universe.
  * Extracted true matches from ground truth.
  * Mined the highest-scoring non-ground-truth lures ranked by the initial model.
* **Negative Separation:**
  * Strict ground truth exclusion: any candidate matching true positive labels was withheld from the negative pool.
  * Pairwise dataset assembled with 1:1 ratio of true positives to full-universe hard negatives.

---

## 3. Comparative Performance: Sampled vs Full-Universe

Evaluated on the sealed 10,000-query validation benchmark (`data/validation/phase4_3b_dev_val_10k_ids.txt`):

| Metric | Phase 4.1 Baseline | Model A (Sampled HardNeg 1x) | Model B (Full-Universe HardNeg 1x) |
| :--- | :---: | :---: | :---: |
| **Training Target Scale** | Random Negatives | ~500k Sampled Targets | **10,320,219 Full Targets** |
| **Optimal Threshold ($\tau$)** | 0.920 | 0.970 | **0.900** |
| **Pair Precision** | 73.21% | 89.36% | **92.45%** |
| **Pair Recall** | 65.10% | **74.47%** | 66.40% |
| **Singleton Accuracy** | 48.20% | 64.71% | **73.90%** |
| **Macro Precision** | 0.7321 | 0.8936 | **0.9245** |
| **Macro Recall** | 0.6510 | 0.7447 | 0.6640 |
| **Macro $F_{0.5}$** | 0.7070 | **0.8069** | 0.7770 |

---

## 4. Key Engineering Insights

1. **Precision Dominance of Full-Universe Mining:**
   Model B achieved an unmatched pair precision of **92.45%** and singleton accuracy of **73.90%**. It learned to recognize extremely subtle differences across nationwide store chains and franchise locations.
2. **Complementarity with Model A:**
   Because Model B was trained on intense global lures, it is naturally more conservative (recall of 66.40%). Model A has broader recall (74.47%). 
3. **Foundation for Ensemble:**
   Rather than choosing one model, combining Model A and Model B in a linear blend ($0.88 \cdot P_A + 0.12 \cdot P_B$) combines the high recall of Model A with the severe false-positive rejection of Model B, outperforming both individual models.

---

## 5. Frozen Production Specification

* **Artifact:** `models/matcher_fulluniverse_hardneg_1x.pkl` (SHA-256: `5c82350bee1b45bdfeaf459ba16ac824e24aa8f35db6e5061ca2a8857063edfd`)
* **Features:** 61 pairwise features.
* **Role in Ensemble:** Model B acts as the scale-rejection component ($1 - \alpha = 0.12$), eliminating persistent lures for singletons.
