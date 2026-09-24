"""
Platform-agnostic configuration for Entity Resolution pipeline.
Handles paths across Windows/Linux/macOS, auto-detects CUDA / RTX 4500 Ada GPU,
and supports environment variables and CLI overrides.
"""

import os
import sys
from pathlib import Path

# Resolve base directory relative to package location or environment
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("ER_PROJECT_DIR", PACKAGE_DIR.parent))

# Data directories (platform agnostic via Path / os.environ)
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", PROJECT_DIR / "dataset"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Fallback detection for student_resource directory layout
if not TRAIN_DIR.exists():
    alt_train = PROJECT_DIR / "student_resource" / "dataset" / "train"
    if alt_train.exists():
        DATA_DIR = PROJECT_DIR / "student_resource" / "dataset"
        TRAIN_DIR = DATA_DIR / "train"
        TEST_DIR = DATA_DIR / "test"

# Default file paths
TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

# Output directories
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", PROJECT_DIR / "output"))
MODELS_DIR = Path(os.environ.get("ER_MODELS_DIR", PROJECT_DIR / "models"))
LOGS_DIR = Path(os.environ.get("ER_LOGS_DIR", PROJECT_DIR / "logs"))

# Ensure output directories exist safely across platforms
for d in (OUTPUT_DIR, MODELS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def detect_device(preferred_device=None):
    """
    Detect whether CUDA GPU (e.g. NVIDIA RTX 4500 Ada) or CPU is available.
    Returns: 'cuda' or 'cpu'
    """
    if preferred_device in ("cuda", "gpu"):
        return "cuda"
    if preferred_device == "cpu":
        return "cpu"

    # Check via torch if installed
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # Check via environment or nvidia-smi presence
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_env and cuda_env != "-1":
        return "cuda"

    return "cpu"


# Model hyperparameter defaults
DEFAULT_CONFIG = {
    # System & Execution
    "device": detect_device(),
    "n_jobs": -1 if os.name != "nt" else max(1, os.cpu_count() or 1),
    "seed": 42,
    "chunk_size": 250000,
    
    # Candidate Blocking
    "max_candidates_per_entity": 50,
    "min_name_token_length": 3,
    "max_token_df_filter": 500,  # exclude ultra-frequent name tokens
    "tfidf_char_ngram_range": (3, 5),
    "top_k_tfidf": 30,
    
    # Pairwise Model
    "model_type": "xgboost",  # 'xgboost' or 'lightgbm'
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "neg_to_pos_ratio": 7,  # negative sampling ratio for training
    
    # Selection Policy (Phase 6)
    "match_threshold": 0.50,
    "empty_threshold": 0.35,  # if top candidate score < empty_threshold -> return empty
    "max_matches_per_entity": 10,
    "relative_score_gap": 0.20,  # drop candidates whose score is < (top_score - gap)
}
