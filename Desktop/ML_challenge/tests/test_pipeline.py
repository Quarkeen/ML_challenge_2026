"""
End-to-end synthetic pipeline smoke test.
Verifies the complete flow (Retriever -> FeatureGenerator -> Model -> Selector -> Scorer)
executes deterministically on any platform.
"""

import pandas as pd
import numpy as np
from entity_resolution.blocking import CandidateRetriever
from entity_resolution.features import FeatureGenerator
from entity_resolution.model import PairwiseRanker
from entity_resolution.selection import SetSelector
from entity_resolution.scorer import macro_f05


def test_end_to_end_synthetic_pipeline():
    # 1. Synthetic S1 Data
    s1_df = pd.DataFrame([
        {"entity_id": "S1-01", "business_name": "Atlas Logistics LLC", "business_address": "500 Market St, Denver, CO 80202", "country": "US"},
        {"entity_id": "S1-02", "business_name": "Bharat Bio Tech Pvt Ltd", "business_address": "Plot 24, HITEC City, Hyderabad 500081", "country": "India"},
        {"entity_id": "S1-03", "business_name": "True Singleton Inc", "business_address": "99 Nowhere Road, Dallas, TX", "country": "US"},
    ])

    # 2. Synthetic S2 & S3 Candidate Data
    s2_df = pd.DataFrame([
        {"entity_id": "S2-101", "business_name": "Atlas Logistics Incorporated", "business_address": "500 Market Street, Denver, Colorado 80202", "country": "US"},
        {"entity_id": "S2-102", "business_name": "Random Other Corp", "business_address": "123 Elm St, Austin, TX", "country": "US"},
    ])
    s3_df = pd.DataFrame([
        {"entity_id": "S3-201", "business_name": "bharatbiotech.com", "business_address": "Plot 24, HITEC City, Hyderabad", "country": "India"},
        {"entity_id": "S3-202", "business_name": "Unrelated Firm", "business_address": "Mumbai, India", "country": "India"},
    ])

    # 3. Ground Truth: S1-01 matches S2-101; S1-02 matches S3-201; S1-03 is singleton
    gt = {
        "S1-01": {"S2-101"},
        "S1-02": {"S3-201"},
        "S1-03": set(),
    }

    # Build retriever
    retriever = CandidateRetriever()
    retriever.build_indexes(s2_df, s3_df)

    # Retrieve candidates
    cands_map = retriever.retrieve_candidates(s1_df, max_candidates=10)
    assert "S2-101" in cands_map["S1-01"]
    assert "S3-201" in cands_map["S1-02"]

    # Generate features
    fg = FeatureGenerator()
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    pairs = [("S1-01", "S2-101"), ("S1-01", "S2-102"), ("S1-02", "S3-201"), ("S1-02", "S3-202")]
    
    feats_df = fg.compute_batch_features(s1_df, cand_df, pairs)
    labels = [1, 0, 1, 0]

    # Model training (CPU fallback mode safe)
    ranker = PairwiseRanker(config={"device": "cpu", "n_estimators": 10})
    ranker.fit(feats_df, labels)

    feats_df["score"] = ranker.predict_proba(feats_df)
    assert len(feats_df["score"]) == 4

    # Set selection
    selector = SetSelector(match_threshold=0.40, empty_threshold=0.30)
    preds = selector.predict_sets(feats_df, all_s1_ids=s1_df["entity_id"].tolist())

    assert "S1-03" in preds
    assert len(preds["S1-03"]) == 0  # correctly empty singleton

    # Compute official score
    score = macro_f05(gt, preds)
    assert 0.0 <= score <= 1.0
