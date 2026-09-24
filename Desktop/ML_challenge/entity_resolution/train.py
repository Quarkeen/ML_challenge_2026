"""
Phase 5: Leakage-Free End-to-End Training Pipeline.
Groups validation strictly by Source 1 anchor (zero cross-anchor leakage).
Supports GPU acceleration (CUDA / RTX 4500 Ada) and saves model + selection policy.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Set, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DEFAULT_CONFIG,
    TRAIN_S1,
    TRAIN_S2,
    TRAIN_S3,
    TRAIN_GT,
    MODELS_DIR,
    OUTPUT_DIR,
    detect_device,
)
from .normalize import punct_normalize, compressed_name
from .blocking import CandidateRetriever
from .features import FeatureGenerator
from .model import PairwiseRanker
from .selection import SetSelector
from .scorer import detailed_evaluation, load_ground_truth, macro_f05


def run_training_pipeline(
    sample_size: int = 50000,
    val_size: float = 0.20,
    device: str = "auto",
    model_type: str = "xgboost",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute full leakage-free training pipeline.
    """
    t_start = time.time()
    out_dir = Path(output_dir or MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = dict(DEFAULT_CONFIG)
    config["device"] = detect_device(device)
    config["model_type"] = model_type

    print(f"============================================================")
    print(f"Starting Entity Resolution Training Pipeline")
    print(f"Device: {config['device']} | Model: {config['model_type']} | Sample: {sample_size}")
    print(f"============================================================")

    # 1. Load S1 data and Ground Truth
    print("\n[Step 1/7] Loading Source 1 and Ground Truth...")
    s1_full = pd.read_csv(TRAIN_S1, sep="\t", nrows=sample_size, dtype=str, keep_default_na=False)
    all_s1_ids = s1_full["entity_id"].tolist()
    
    gt_full = pd.read_csv(TRAIN_GT, sep="\t", dtype=str, keep_default_na=False)
    gt_sub = gt_full[gt_full["source1_entity_id"].isin(set(all_s1_ids))]
    gt_map = load_ground_truth(gt_sub)
    for sid in all_s1_ids:
        if sid not in gt_map:
            gt_map[sid] = set()

    # 2. Leakage-Free Train/Val Split by Source 1 Anchor
    print(f"\n[Step 2/7] Splitting S1 entities into Train / Val (val_size={val_size})...")
    stratify_col = s1_full["country"].values if "country" in s1_full.columns else None
    train_s1_ids, val_s1_ids = train_test_split(
        all_s1_ids, test_size=val_size, random_state=config["seed"], stratify=stratify_col
    )
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)

    train_s1_df = s1_full[s1_full["entity_id"].isin(train_s1_set)].copy()
    val_s1_df = s1_full[s1_full["entity_id"].isin(val_s1_set)].copy()

    val_gt_map = {sid: gt_map[sid] for sid in val_s1_ids}
    print(f"  Train S1 entities: {len(train_s1_df):,}")
    print(f"  Val S1 entities:   {len(val_s1_df):,}")

    # 3. Build Candidate Retriever Indexes from S2 and S3
    print("\n[Step 3/7] Building Candidate Retriever Indexes from S2 and S3...")
    retriever = CandidateRetriever(config=config)
    
    # Ingest S2/S3 candidate pools via memory-efficient chunks
    cand_records = {}
    for p, name in [(TRAIN_S2, "Source 2"), (TRAIN_S3, "Source 3")]:
        print(f"  Scanning and indexing {name}...")
        for chunk in pd.read_csv(p, sep="\t", chunksize=config["chunk_size"], dtype=str, keep_default_na=False):
            retriever.build_indexes(chunk, None)
            for _, row in chunk.iterrows():
                cand_records[row["entity_id"]] = row.to_dict()

    # 4. Candidate Retrieval for Train and Validation sets
    print("\n[Step 4/7] Retrieving candidate matches...")
    val_candidates = retriever.retrieve_candidates(val_s1_df, max_candidates=config["max_candidates_per_entity"])
    blocking_metrics = retriever.evaluate_blocking(val_candidates, val_gt_map)
    print(f"  Val Blocking Recall:  {blocking_metrics['blocking_recall']*100:.2f}%")
    print(f"  Val Blocking Ceiling F_0.5: {blocking_metrics['blocking_ceiling_f05']:.4f}")
    print(f"  Avg candidates/entity: {blocking_metrics['avg_candidates_per_entity']:.1f}")

    train_candidates = retriever.retrieve_candidates(train_s1_df, max_candidates=config["max_candidates_per_entity"])

    # 5. Form Training Pairs (Positives + Hard Negatives from blocking)
    print("\n[Step 5/7] Generating Pairwise Features...")
    fg = FeatureGenerator(config=config)

    def build_pairs_and_labels(s1_df, cand_map, is_train=True):
        pairs = []
        labels = []
        retrieval_meta = {}
        s1_lookup = s1_df.set_index("entity_id").to_dict(orient="index")

        for s1_id, cands in cand_map.items():
            true_matches = gt_map.get(s1_id, set())

            # Include true positives from ground truth even if missed by blocking (for training)
            if is_train:
                for true_cid in true_matches:
                    if true_cid in cand_records:
                        pairs.append((s1_id, true_cid))
                        labels.append(1)
                        retrieval_meta[(s1_id, true_cid)] = cands.get(true_cid, {"retrieval_score": 10.0, "retrieval_methods": "gt_injection"})

            # Include retrieved candidates
            for cid, info in cands.items():
                if (s1_id, cid) in retrieval_meta:
                    continue  # already added
                is_pos = 1 if cid in true_matches else 0
                pairs.append((s1_id, cid))
                labels.append(is_pos)
                retrieval_meta[(s1_id, cid)] = info

        # Compute features
        print(f"  Computing features for {len(pairs):,} pairs...")
        cand_df = pd.DataFrame(list(cand_records.values()))
        feats_df = fg.compute_batch_features(s1_df, cand_df, pairs, retrieval_info=retrieval_meta)
        feats_df["target"] = labels
        return feats_df

    train_pairs_df = build_pairs_and_labels(train_s1_df, train_candidates, is_train=True)
    val_pairs_df = build_pairs_and_labels(val_s1_df, val_candidates, is_train=False)

    print(f"  Training pairs:   {len(train_pairs_df):,} (Pos: {(train_pairs_df['target']==1).sum():,}, Neg: {(train_pairs_df['target']==0).sum():,})")
    print(f"  Validation pairs: {len(val_pairs_df):,} (Pos: {(val_pairs_df['target']==1).sum():,}, Neg: {(val_pairs_df['target']==0).sum():,})")

    # 6. Model Training with GPU / CPU Ranker
    print("\n[Step 6/7] Training Pairwise Match Ranker...")
    ranker = PairwiseRanker(config=config)
    y_train = train_pairs_df["target"].values
    y_val = val_pairs_df["target"].values

    ranker.fit(train_pairs_df, y_train, val_pairs_df, y_val)

    # Score validation pairs
    val_pairs_df["score"] = ranker.predict_proba(val_pairs_df)

    # 7. Joint Set Selection Policy & Threshold Tuning
    print("\n[Step 7/7] Tuning Set Selection Policy on Validation against Macro F_0.5...")
    selector = SetSelector()
    tuning_res = selector.tune_thresholds(val_pairs_df, val_gt_map)
    print(f"  Best Validation Macro F_0.5: {tuning_res['best_macro_f05']:.4f}")
    print(f"  Optimal Parameters: {tuning_res['best_params']}")

    # Final detailed validation evaluation
    val_preds = selector.predict_sets(val_pairs_df, all_s1_ids=val_s1_ids)
    val_countries = dict(zip(val_s1_df["entity_id"].values, val_s1_df["country"].values))
    final_metrics = detailed_evaluation(val_gt_map, val_preds, val_countries)

    print("\n================== FINAL VALIDATION METRICS ==================")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("==============================================================")

    # Save artifacts
    print(f"\nSaving model and configuration to {out_dir}...")
    model_path = out_dir / f"pairwise_ranker.{model_type}"
    ranker.save(model_path)

    policy_path = out_dir / "selection_policy.json"
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump({
            "thresholds": tuning_res["best_params"],
            "validation_metrics": final_metrics,
            "blocking_metrics": blocking_metrics,
            "elapsed_seconds": time.time() - t_start,
        }, f, indent=2)

    feat_imp = ranker.get_feature_importance()
    imp_path = out_dir / "feature_importance.json"
    with open(imp_path, "w", encoding="utf-8") as f:
        json.dump(feat_imp, f, indent=2)

    print(f"Training pipeline finished in {time.time() - t_start:.1f}s.")
    return {
        "final_metrics": final_metrics,
        "tuning": tuning_res,
        "blocking": blocking_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Entity Resolution System")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of S1 train records")
    parser.add_argument("--val-size", type=float, default=0.20, help="Fraction for validation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device (RTX 4500 Ada = cuda)")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "lightgbm"], help="Model algorithm")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save model")
    args = parser.parse_args()

    run_training_pipeline(
        sample_size=args.sample_size,
        val_size=args.val_size,
        device=args.device,
        model_type=args.model_type,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
