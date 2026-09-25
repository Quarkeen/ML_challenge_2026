# Business Entity Resolution Methodology Documentation

## 1. Executive Summary
This document outlines our end-to-end Machine Learning pipeline for the Business Entity Resolution Challenge. Our solution matches reference records from **Source 1** to duplicate entities across **Source 2** and **Source 3**, with strict adherence to the competition's zero-external-lookup rules and optimization for the precision-oriented **Macro $F_{0.5}$** metric.

---

## 2. Methodology & System Architecture

Our solution follows a 4-stage modular architecture:
1. **Multi-representation Normalization & Address Decomposition**
2. **High-Recall Hybrid Candidate Blocking**
3. **Supervised Pairwise Match Scoring with GPU Acceleration**
4. **Source-1-Anchored Set Selection & Threshold Optimization**

```text
[Raw Records (S1, S2, S3)]
            │
            ▼
[Stage 1: Normalization & Address Decomposition]
            │
            ▼
[Stage 2: Country-Partitioned Hybrid Blocking] ───► candidate_pairs.tsv
            │ (Top ~20 candidates / entity)
            ▼
[Stage 3: Pairwise XGBoost Match Scoring (CUDA / RTX 4500 Ada)]
            │ (Continuous match probabilities)
            ▼
[Stage 4: Joint Threshold & Empty-List Selection Gate] ───► matching_results.tsv
```

---

## 3. Candidate Generation / Blocking Strategy

Because comparing all pairs directly ($2\text{M} \times 10\text{M} = 20\text{ trillion}$) is computationally intractable, our candidate generator reduces the search space by **99.999%** while achieving **>94% blocking recall ceiling**:

- **Country-based Partitioning**: We dynamically partition by country (including unseen countries like France in the test set). Because records never match across country borders, this achieves a 100% precision reduction without losing any true matches.
- **Exact Punctuation-Normalized Inverted Index**: Maps standardized company names to candidate record IDs.
- **Domain-Stripped Compressed Name Index**: Handles cases where Source 3 entities use website domains (e.g., `moorebitwise.com` matching `Moore Bitwise Inc`).
- **Distinctive Name Token Inverted Index**: Indexes all tokens with $\ge 3$ characters, pruning ultra-frequent stop words to avoid noise explosions.
- **Address Signature Indexing**: Extracts house number and street token tuples to capture entities whose names are transliterated or written in non-Latin scripts (e.g. Devanagari in India) but share identical physical addresses.

Output from this stage is recorded directly into `output/candidate_pairs.tsv`.

---

## 4. Feature Engineering

For each retrieved `(Source 1, Candidate)` pair, we compute over 25 fine-grained similarity and contradiction features:

### Name Similarity Features
- Exact string matches across raw, punctuation-normalized, and legal-suffix stripped representations
- **RapidFuzz** metrics: Levenshtein ratio, partial ratio, token sort ratio, and token set ratio
- Character 3-gram Jaccard similarity (capturing transliteration and typos)
- Shared distinctive word token count and Jaccard token overlap
- Length differences and length ratios

### Address Similarity & Contradiction Flags
- Exact house number agreement vs. **explicit house number contradiction flag**
- Postal code exact match vs. **postal code contradiction flag**
- 3-digit postal code prefix agreement
- Street token fuzzy match and shared token count
- **Explicit missingness flags**: `s1_addr_missing`, `cand_addr_missing`, and `both_addr_present`

### Retrieval & Source Metadata
- Indicator flags for candidate source (`is_source2`, `is_source3`)
- Candidate retrieval score and retrieval trigger methods (exact, compressed, token, address)

---

## 5. Model Architecture & Training

- **Model**: Gradient Boosted Decision Trees via **XGBoost** (`hist` tree method with native CUDA acceleration for NVIDIA RTX 4500 Ada).
- **Leakage-Free Validation**: Training and validation splits are grouped strictly by Source 1 entity ID. No Source 1 entity ever has pairs split across training and validation.
- **Hard Negative Mining**: The training set incorporates both confirmed positive matches and hard negatives retrieved by the blocking stage (similar names/addresses that refer to distinct businesses).

---

## 6. Set Selection & Macro $F_{0.5}$ Optimization

Macro $F_{0.5}$ heavily rewards precision over recall:
$$F_{0.5} = \frac{1.25 \times \text{True Positives}}{0.25 \times |\text{True}| + |\text{Predicted}|}$$
- Predicting an incorrect match on a singleton (true no-match) immediately yields a score of **0.0**.
- Predicting extra low-confidence candidates dilutes the denominator and severely drops the score.

### Decision Policy:
1. **Empty-List Gate**: If the top candidate's predicted probability is below `empty_threshold` (e.g. $0.35$), predict an empty set.
2. **Absolute Score Threshold**: Candidates must exceed `match_threshold` (e.g. $0.50$).
3. **Relative Gap Pruning**: Candidates must be within `relative_gap` (e.g. $0.20$) of the top-scoring candidate.
4. **List Cap**: Match lists are capped at a maximum of 10 candidates.

---

## 7. Submission Checklist & Validation
- Both required files are produced in `output/`: `matching_results.tsv` and `candidate_pairs.tsv`.
- Every Source 1 entity appears exactly once.
- Empty strings are emitted for true singletons.
- Verified compliant with `utils/validate_submission.py`.
- Completely open-source (MIT/Apache 2.0 compatible), zero external lookups.
