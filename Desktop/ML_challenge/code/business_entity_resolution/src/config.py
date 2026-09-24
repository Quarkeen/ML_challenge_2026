"""
Configuration module for the business entity resolution pipeline.
Platform-agnostic path management, hyperparameters, and defaults.
Supports environment variables and relative path resolution.
"""

import os
import sys

# Dynamic root detection: find project root from this file's location
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# code/business_entity_resolution
PROJECT_DIR = os.path.dirname(SRC_DIR)
# Repository root (two levels above code/)
REPO_ROOT = os.path.dirname(os.path.dirname(PROJECT_DIR))
BASE_DIR = os.environ.get('PROJECT_ROOT', REPO_ROOT)

# Directory configurations (configurable via environment variables)
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(BASE_DIR, 'student_resource', 'dataset'))
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.join(BASE_DIR, 'output'))
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(BASE_DIR, 'models'))
CACHE_DIR = os.environ.get('CACHE_DIR', os.path.join(BASE_DIR, 'cache'))

# Training data files
TRAIN_SOURCE1 = os.path.join(TRAIN_DIR, 'train_source1.tsv')
TRAIN_SOURCE2 = os.path.join(TRAIN_DIR, 'train_source2.tsv')
TRAIN_SOURCE3 = os.path.join(TRAIN_DIR, 'train_source3.tsv')
TRAIN_GROUND_TRUTH = os.path.join(TRAIN_DIR, 'train_ground_truth.tsv')

TRAIN_S1 = TRAIN_SOURCE1
TRAIN_S2 = TRAIN_SOURCE2
TRAIN_S3 = TRAIN_SOURCE3
TRAIN_GT = TRAIN_GROUND_TRUTH

# Test data files
TEST_SOURCE1 = os.path.join(TEST_DIR, 'test_source1.tsv')
TEST_SOURCE2 = os.path.join(TEST_DIR, 'test_source2.tsv')
TEST_SOURCE3 = os.path.join(TEST_DIR, 'test_source3.tsv')

TEST_S1 = TEST_SOURCE1
TEST_S2 = TEST_SOURCE2
TEST_S3 = TEST_SOURCE3

# Output files
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

# Python executable (platform-agnostic)
PYTHON = sys.executable

# Random seed
RANDOM_SEED = 42

# Blocking parameters
BLOCKING_CONFIG = {
    'tfidf_ngram_range': (3, 5),
    'tfidf_max_features': 100000,
    'tfidf_top_k': 30,
    'max_candidates_per_entity': 25,
    'min_candidates_per_entity': 3,
}

# Model parameters
MODEL_CONFIG = {
    'lgbm_params': {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'verbose': -1,
        'n_jobs': -1,
    }
}

# Selection policy parameters
SELECTION_CONFIG = {
    'global_threshold': 0.5,
    'max_list_size': 10,
    'min_score_for_inclusion': 0.35,
    'empty_list_top_score_threshold': 0.35,
}

# Ensure local runtime directories exist
for d in [OUTPUT_DIR, MODEL_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)
