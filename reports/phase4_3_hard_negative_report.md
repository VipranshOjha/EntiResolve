# Phase 4.3: Hard-Negative Mining & Model Training Report
**Amazon ML Challenge 2026: Business Entity Resolution**  
**Project:** EntiResolve Engine  
**Component:** Model A (`models/matcher_hardneg_1x.pkl`)

---

## 1. Executive Summary

Standard pairwise entity resolution classifiers trained on random negatives suffer severe precision degradation when deployed against massive search spaces because random pairs are trivially distinguishable. In Phase 4.3, we developed an active hard-negative mining framework that mines high-utility, deceptive non-matches directly from the candidate generation stage.

Evaluating ratios of $1\times, 2\times, 4\times,$ and $8\times$ hard negatives per positive pair, we determined that **$1\times$ Hard-Negative Mining** achieved the optimal operating balance for the Macro $F_{0.5}$ objective. This model forms **Model A** of our production ensemble, contributing high recall and sharp boundary discrimination.

---

## 2. Hard-Negative Mining Strategy

For each true positive pair $(q, t^+)$ in the development training partition, we queried our 6-channel composite blocker ($K=175$) against the training target universe. Any retrieved candidate $t^-$ such that $t^- \notin \text{GroundTruth}(q)$ was collected as an empirical hard negative.

Hard negatives were categorized across 9 distinct structural deception categories:
1. **Category A (Exact Name, Different Address):** Chains, franchises, independent stores sharing common commercial names.
2. **Category B (Compact / Token Name Hit, Different Address):** Partial brand or token overlap.
3. **Category C (High String Similarity $\ge 0.85$, Different Address):** Minor typos, lexical variations.
4. **Category D (Shared Postal Code / PIN, Different Name):** Co-located non-identical businesses.
5. **Category E (Shared Street / Locality, Different Name):** Neighbors in same commercial centers.
6. **Category F (Cross-Script Transliteration False Match):** Phonetically similar non-identical Indic entities.
7. **Category G (High Provenance Multi-Channel Hits):** Candidates triggering 3+ blocking channels simultaneously.
8. **Category H (Boundary Rank $1 \dots 10$ Non-Matches):** Highest-ranked non-matches output by blocker.
9. **Category I (Deep Rank $50 \dots 175$ Non-Matches):** Tail candidates preventing late false merges.

---

## 3. Experimental Results Across Mining Ratios

All models were evaluated on the sealed 10,000-query development validation cohort (`data/validation/phase4_3b_dev_val_10k_ids.txt`) against 4.13M India targets and 6.19M US targets:

| Model Configuration | Negative Ratio | Decision Threshold ($\tau$) | Pair Precision | Pair Recall | Singleton Accuracy | Macro $F_{0.5}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Heuristic Baseline | $0\times$ (Random) | 0.920 | 73.21% | 65.10% | 48.20% | 0.7070 |
| **Model A (HardNeg 1x)** | **$1\times$ Hard** | **0.970** | **89.36%** | **74.47%** | **64.71%** | **0.8069** |
| HardNeg 2x | $2\times$ Hard | 0.950 | 88.92% | 72.10% | 66.12% | 0.7981 |
| HardNeg 4x | $4\times$ Hard | 0.930 | 88.15% | 70.45% | 67.80% | 0.7890 |
| HardNeg 8x | $8\times$ Hard | 0.900 | 87.40% | 68.20% | 69.10% | 0.7785 |

### Key Findings:
* Hard-negative training boosted Macro $F_{0.5}$ by **+0.0999** over the random-negative baseline.
* The $1\times$ ratio provided the greatest discriminative flexibility, preserving high true-positive recall (**74.47%**) while increasing precision from 73.21% to **89.36%**.
* Larger negative ratios ($4\times, 8\times$) over-penalized borderline true positives, suppressing recall without generating commensurate precision gains.

---

## 4. Frozen Production Specification

* **Artifact:** `models/matcher_hardneg_1x.pkl` (SHA-256: `0c6a5c3a8de0ee14aa2527ff180e2e381dfcaa31efd9a8277084e6be333e11cb`)
* **Features:** 61 pairwise features (RapidFuzz string similarities, structural address agreement, cross-script transliteration alignment, and blocker provenance).
* **Role in Ensemble:** Model A provides high candidate recovery and strong discriminative features for non-singleton entities ($\alpha = 0.88$).
