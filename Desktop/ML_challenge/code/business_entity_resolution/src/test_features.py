import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FeatureGenerator

def test_feature_generator_initialization():
    fg = FeatureGenerator()
    assert len(fg.feature_names) == 54
    assert 'name_exact_match' in fg.feature_names
    assert 'addr_exact_match' in fg.feature_names
    assert 'country_match' in fg.feature_names
    assert 's1_addr_missing' in fg.feature_names

def test_exact_match_pair():
    fg = FeatureGenerator()
    s1_row = {
        'entity_id': 'S1-001',
        'business_name': 'Walmart Supercenter Inc',
        'business_address': '100 Main Street, Bentonville, AR 72712',
        'country': 'US'
    }
    cand_row = {
        'entity_id': 'S2-001',
        'business_name': 'Walmart Supercenter',
        'business_address': '100 Main Street, Bentonville, AR 72712',
        'country': 'US'
    }
    feats = fg.compute_pair_features(s1_row, cand_row)
    assert feats['name_exact_match_no_suffix'] is True
    assert feats['name_levenshtein'] > 85.0
    assert feats['country_match'] is True
    assert feats['country_is_US'] is True
    assert feats['source_is_S2'] is True
    assert feats['source_is_S3'] is False
    assert feats['addr_exact_match'] is True
    assert feats['s1_addr_missing'] is False
    assert feats['cand_addr_missing'] is False

def test_missing_address_handling():
    fg = FeatureGenerator()
    s1_row = {
        'entity_id': 'S1-002',
        'business_name': 'Acme Corp',
        'business_address': '',
        'country': 'India'
    }
    cand_row = {
        'entity_id': 'S3-002',
        'business_name': 'Acme Corporation',
        'business_address': 'NaN',
        'country': 'India'
    }
    feats = fg.compute_pair_features(s1_row, cand_row)
    assert feats['s1_addr_missing'] is True
    assert feats['cand_addr_missing'] is True
    assert feats['both_addr_missing'] is True
    assert np.isnan(feats['addr_exact_match'])
    assert np.isnan(feats['addr_levenshtein'])
    assert feats['source_is_S3'] is True
    assert feats['country_is_India'] is True

def test_batch_feature_generation():
    fg = FeatureGenerator()
    s1_df = pd.DataFrame([
        {'entity_id': 'S1-1', 'business_name': 'Target', 'business_address': '100 1st St', 'country': 'US'},
        {'entity_id': 'S1-2', 'business_name': 'Best Buy', 'business_address': '200 2nd St', 'country': 'US'}
    ])
    cand_df = pd.DataFrame([
        {'entity_id': 'S2-1', 'business_name': 'Target Store', 'business_address': '100 1st St', 'country': 'US'},
        {'entity_id': 'S3-2', 'business_name': 'Best Buy Co', 'business_address': '200 2nd St', 'country': 'US'}
    ])
    pairs = [('S1-1', 'S2-1'), ('S1-2', 'S3-2')]
    df_feats = fg.compute_batch_features(s1_df, cand_df, pairs)
    assert len(df_feats) == 2
    assert 'name_exact_match' in df_feats.columns
    assert 'addr_exact_match' in df_feats.columns
