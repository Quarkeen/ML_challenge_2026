"""
Phase 1: Baselines Implementation and Scorer Benchmarking.
Evaluates:
1. All-empty baseline
2. Exact normalized name matching
3. Exact name + postal code matching
4. Conservative rule-based baseline
All benchmarked against the official macro F_0.5 metric.
"""

from collections import defaultdict
from typing import Dict, Set, Optional, Any
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT
from .normalize import punct_normalize, strip_legal_suffix, extract_postal_code
from .scorer import detailed_evaluation, load_ground_truth, _parse_ids


def evaluate_all_empty(ground_truth: Dict[str, Set[str]], countries: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Baseline 1: All-Empty prediction."""
    preds = {s1_id: set() for s1_id in ground_truth}
    return detailed_evaluation(ground_truth, preds, countries)


def run_baselines(
    sample_size: Optional[int] = 25000,
    s1_path: Optional[str] = None,
    s2_path: Optional[str] = None,
    s3_path: Optional[str] = None,
    gt_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run and compare all official baselines on training data.
    Uses memory-efficient streaming to handle arbitrary dataset scales.
    """
    p_s1 = s1_path or TRAIN_S1
    p_s2 = s2_path or TRAIN_S2
    p_s3 = s3_path or TRAIN_S3
    p_gt = gt_path or TRAIN_GT

    print(f"Loading S1 sample (sample_size={sample_size})...")
    s1_df = pd.read_csv(p_s1, sep="\t", nrows=sample_size, dtype=str, keep_default_na=False)
    s1_ids = set(s1_df["entity_id"].values)
    s1_countries = dict(zip(s1_df["entity_id"].values, s1_df["country"].values))

    print("Loading Ground Truth...")
    gt_full = pd.read_csv(p_gt, sep="\t", dtype=str, keep_default_na=False)
    gt_sub = gt_full[gt_full["source1_entity_id"].isin(s1_ids)]
    gt_dict = {row["source1_entity_id"]: _parse_ids(row["matched_entity_ids"]) for _, row in gt_sub.iterrows()}
    # Ensure all S1 entities appear in gt_dict
    for sid in s1_ids:
        if sid not in gt_dict:
            gt_dict[sid] = set()

    results = {}

    # 1. All-empty baseline
    print("\n--- Baseline 1: All-Empty Prediction ---")
    b1_metrics = evaluate_all_empty(gt_dict, s1_countries)
    results["all_empty"] = b1_metrics
    print(f"  Macro F_0.5: {b1_metrics['macro_f05']:.4f}")
    print(f"  True singletons: {b1_metrics['true_singletons_count']} ({b1_metrics['true_singletons_pct']:.2f}%)")
    print(f"  Empty-list accuracy: {b1_metrics['empty_list_accuracy']:.4f}")

    # Build streaming exact-name and postal-code indexes for S2 and S3
    print("\nIndexing S2 & S3 candidate records (streaming)...")
    name_index = defaultdict(list)
    name_postal_index = defaultdict(list)

    for path in (p_s2, p_s3):
        print(f"  Scanning {path}...")
        for chunk in pd.read_csv(path, sep="\t", chunksize=250000, dtype=str, keep_default_na=False):
            ids = chunk["entity_id"].values
            names = chunk["business_name"].values
            addrs = chunk["business_address"].values
            countries = chunk["country"].values if "country" in chunk.columns else [""] * len(chunk)

            for cid, n, a, c in zip(ids, names, addrs, countries):
                p_n = punct_normalize(n)
                if not p_n:
                    continue
                c_clean = str(c).strip()
                name_index[(c_clean, p_n)].append(cid)
                
                pc = extract_postal_code(a, c_clean)
                if pc:
                    name_postal_index[(c_clean, p_n, pc)].append(cid)

    # 2. Exact Normalized Name Matching
    print("\n--- Baseline 2: Exact Normalized Name Matching ---")
    b2_preds = {}
    for sid, n, c in zip(s1_df["entity_id"].values, s1_df["business_name"].values, s1_df["country"].values):
        p_n = punct_normalize(n)
        matches = name_index.get((str(c).strip(), p_n), [])
        b2_preds[sid] = set(matches)

    b2_metrics = detailed_evaluation(gt_dict, b2_preds, s1_countries)
    results["exact_name"] = b2_metrics
    print(f"  Macro F_0.5: {b2_metrics['macro_f05']:.4f}")
    print(f"  Empty-list accuracy: {b2_metrics['empty_list_accuracy']:.4f}")
    print(f"  False nonempty rate: {b2_metrics['false_nonempty_rate']:.4f}")

    # 3. Exact Name + Postal Code Matching
    print("\n--- Baseline 3: Exact Name + Postal Code Matching ---")
    b3_preds = {}
    for sid, n, a, c in zip(s1_df["entity_id"].values, s1_df["business_name"].values, s1_df["business_address"].values, s1_df["country"].values):
        c_clean = str(c).strip()
        p_n = punct_normalize(n)
        pc = extract_postal_code(a, c_clean)
        matches = name_postal_index.get((c_clean, p_n, pc), []) if pc else []
        b3_preds[sid] = set(matches)

    b3_metrics = detailed_evaluation(gt_dict, b3_preds, s1_countries)
    results["exact_name_postal"] = b3_metrics
    print(f"  Macro F_0.5: {b3_metrics['macro_f05']:.4f}")
    print(f"  Empty-list accuracy: {b3_metrics['empty_list_accuracy']:.4f}")

    # 4. Conservative Rule-Based Baseline
    print("\n--- Baseline 4: Conservative Rule-Based Baseline ---")
    b4_preds = {}
    for sid, n, a, c in zip(s1_df["entity_id"].values, s1_df["business_name"].values, s1_df["business_address"].values, s1_df["country"].values):
        c_clean = str(c).strip()
        p_n = punct_normalize(n)
        pc = extract_postal_code(a, c_clean)

        # Rule: Exact name matches that share postal code if present, capped at 8 candidates
        cands = name_index.get((c_clean, p_n), [])
        if len(cands) == 0:
            b4_preds[sid] = set()
        elif len(cands) <= 6:
            b4_preds[sid] = set(cands)
        else:
            # Too many name collisions -> filter strictly by postal code
            strict = name_postal_index.get((c_clean, p_n, pc), []) if pc else []
            b4_preds[sid] = set(strict) if strict else set(cands[:3])

    b4_metrics = detailed_evaluation(gt_dict, b4_preds, s1_countries)
    results["conservative_rule"] = b4_metrics
    print(f"  Macro F_0.5: {b4_metrics['macro_f05']:.4f}")
    print(f"  Empty-list accuracy: {b4_metrics['empty_list_accuracy']:.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase 1 ER Baselines")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of S1 records to evaluate")
    args = parser.parse_args()
    run_baselines(sample_size=args.sample_size)
