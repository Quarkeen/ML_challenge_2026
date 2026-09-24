"""
End-to-End Inference and Submission Generator.
Streams test records to ensure strict OOM safety.
Outputs exact required TSV format:
    source1_entity_id\tmatched_entity_ids
Every Source 1 ID appears exactly once, with empty strings for singletons.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_CONFIG,
    TEST_S1,
    TEST_S2,
    TEST_S3,
    OUTPUT_DIR,
    MODELS_DIR,
    detect_device,
)
from .blocking import CandidateRetriever
from .features import FeatureGenerator
from .model import PairwiseRanker
from .selection import SetSelector


def generate_predictions(
    model_path: Optional[Path] = None,
    policy_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    s1_path: Optional[Path] = None,
    s2_path: Optional[Path] = None,
    s3_path: Optional[Path] = None,
    batch_size: int = 50000,
    device: str = "auto",
) -> Tuple[Path, Path]:
    """
    Generate predictions for test dataset and save official submission TSV.
    """
    t_start = time.time()
    m_path = Path(model_path or (MODELS_DIR / "pairwise_ranker.xgboost"))
    if not m_path.exists():
        # Try lightgbm alternative
        m_alt = m_path.with_suffix(".lightgbm")
        if m_alt.exists():
            m_path = m_alt

    pol_path = Path(policy_path or (MODELS_DIR / "selection_policy.json"))
    out_dir = Path(output_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    matching_file = out_dir / "matching_results.tsv"
    candidate_file = out_dir / "candidate_pairs.tsv"

    p_s1 = Path(s1_path or TEST_S1)
    p_s2 = Path(s2_path or TEST_S2)
    p_s3 = Path(s3_path or TEST_S3)

    print(f"============================================================")
    print(f"Starting Entity Resolution Inference")
    print(f"Model: {m_path}")
    print(f"Test S1: {p_s1}")
    print(f"Matching Results Output:  {matching_file}")
    print(f"Candidate Pairs Output:   {candidate_file}")
    print(f"============================================================")

    # 1. Load Model and Policy
    print("\n[Step 1/4] Loading trained model and selection policy...")
    ranker = PairwiseRanker.load(m_path)
    ranker.device = detect_device(device)
    print(f"  Model loaded. Inference device: {ranker.device}")

    selector_params = DEFAULT_CONFIG
    if pol_path.exists():
        with open(pol_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            selector_params.update(meta.get("thresholds", {}))
    selector = SetSelector(
        match_threshold=selector_params.get("match_threshold", 0.50),
        empty_threshold=selector_params.get("empty_threshold", 0.35),
        relative_gap=selector_params.get("relative_score_gap", 0.20),
        max_matches=selector_params.get("max_matches_per_entity", 10),
    )
    print(f"  Selector parameters: match_th={selector.match_threshold}, empty_th={selector.empty_threshold}")

    # 2. Build Candidate Indexes from Test S2 and S3
    print("\n[Step 2/4] Indexing Test S2 and S3 records...")
    retriever = CandidateRetriever(config=DEFAULT_CONFIG)
    cand_records = {}

    for p, name in [(p_s2, "Test Source 2"), (p_s3, "Test Source 3")]:
        print(f"  Indexing {name} ({p})...")
        for chunk in pd.read_csv(p, sep="\t", chunksize=DEFAULT_CONFIG["chunk_size"], dtype=str, keep_default_na=False):
            retriever.build_indexes(chunk, None)
            for _, row in chunk.iterrows():
                cand_records[row["entity_id"]] = row.to_dict()

    cand_df = pd.DataFrame(list(cand_records.values()))
    print(f"  Candidate pool indexed. Total unique candidates: {len(cand_df):,}")

    # 3. Stream through Test S1 in Batches & Write Both Output TSVs
    print("\n[Step 3/4] Streaming Test S1 records and predicting matches...")
    fg = FeatureGenerator(config=DEFAULT_CONFIG)

    total_s1_processed = 0
    non_empty_predictions = 0

    with open(matching_file, "w", encoding="utf-8") as f_match, open(candidate_file, "w", encoding="utf-8") as f_cand:
        # Write exact required headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_chunk in pd.read_csv(p_s1, sep="\t", chunksize=batch_size, dtype=str, keep_default_na=False):
            chunk_s1_ids = s1_chunk["entity_id"].tolist()
            
            # Retrieve candidates for chunk
            cands_map = retriever.retrieve_candidates(s1_chunk, max_candidates=DEFAULT_CONFIG["max_candidates_per_entity"])

            # Flatten pairs for scoring
            pairs = []
            retrieval_meta = {}
            for s1_id, c_dict in cands_map.items():
                for cid, info in c_dict.items():
                    pairs.append((s1_id, cid))
                    retrieval_meta[(s1_id, cid)] = info

            if pairs:
                feats_df = fg.compute_batch_features(s1_chunk, cand_df, pairs, retrieval_info=retrieval_meta)
                scores = ranker.predict_proba(feats_df)
                feats_df["score"] = scores
                chunk_preds = selector.predict_sets(feats_df, all_s1_ids=chunk_s1_ids)
            else:
                chunk_preds = {sid: set() for sid in chunk_s1_ids}

            # Write batch results to both files
            for s1_id in chunk_s1_ids:
                c_dict = cands_map.get(s1_id, {})
                cands_list = list(c_dict.keys())
                cand_str = ",".join(cands_list) if cands_list else ""
                f_cand.write(f"{s1_id}\t{cand_str}\n")

                matched_set = chunk_preds.get(s1_id, set())
                # Enforce official validator constraint: matches must be subset of candidates
                valid_matched = [m for m in sorted(matched_set) if m in c_dict]
                if valid_matched:
                    matched_str = ",".join(valid_matched)
                    non_empty_predictions += 1
                else:
                    matched_str = ""
                f_match.write(f"{s1_id}\t{matched_str}\n")

            total_s1_processed += len(chunk_s1_ids)
            print(f"  Processed {total_s1_processed:,} S1 entities... ({non_empty_predictions:,} matched)")

    # 4. Summary & Verification
    print("\n[Step 4/4] Validating submission integrity...")
    print(f"  Total S1 rows written: {total_s1_processed:,}")
    print(f"  Non-empty predictions: {non_empty_predictions:,} ({(non_empty_predictions/max(1, total_s1_processed))*100:.2f}%)")
    print(f"  Singletons (empty):    {total_s1_processed - non_empty_predictions:,} ({((total_s1_processed - non_empty_predictions)/max(1, total_s1_processed))*100:.2f}%)")
    print(f"  Matching Results:      {matching_file.resolve()}")
    print(f"  Candidate Pairs:       {candidate_file.resolve()}")
    print(f"Inference completed in {time.time() - t_start:.1f}s.")

    return matching_file, candidate_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Official ER Predictions")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--policy-path", type=str, default=None, help="Path to selection policy JSON")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--batch-size", type=int, default=50000, help="Batch size for S1 streaming")
    args = parser.parse_args()

    generate_predictions(
        model_path=Path(args.model_path) if args.model_path else None,
        policy_path=Path(args.policy_path) if args.policy_path else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        device=args.device,
        batch_size=args.batch_size,
    )
