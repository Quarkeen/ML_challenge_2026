"""
Unit tests for pairwise feature extraction.
"""

import pandas as pd
from entity_resolution.features import FeatureGenerator


def test_feature_generation_exact_pair():
    fg = FeatureGenerator()
    s1_row = {"business_name": "Acme Widgets LLC", "business_address": "123 Main St, Austin, TX 78701", "country": "US"}
    cand_row = {"entity_id": "S2-10", "business_name": "Acme Widgets Inc", "business_address": "123 Main Street, Austin, TX 78701", "country": "US"}

    feats = fg.compute_pair_features(s1_row, cand_row)
    assert feats["name_fuzz_ratio"] > 0.80
    assert feats["hn_exact_match"] == 1.0
    assert feats["postal_exact_match"] == 1.0
    assert feats["is_source2"] == 1.0
    assert feats["is_source3"] == 0.0
    assert feats["s1_addr_missing"] == 0.0


def test_feature_generation_missing_address():
    fg = FeatureGenerator()
    s1_row = {"business_name": "Solo Corp", "business_address": "", "country": "US"}
    cand_row = {"entity_id": "S3-25", "business_name": "Solo Corp", "business_address": None, "country": "US"}

    feats = fg.compute_pair_features(s1_row, cand_row)
    assert feats["name_exact_punct"] == 1.0
    assert feats["s1_addr_missing"] == 1.0
    assert feats["cand_addr_missing"] == 1.0
    assert feats["hn_exact_match"] == 0.0
    assert feats["hn_contradiction"] == 0.0
