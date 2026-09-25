# High-Precision Multi-Source Entity Resolution System

A modular, platform-agnostic, GPU-accelerated Entity Resolution system designed specifically to optimize the official **Macro $F_{0.5}$** objective across Source 1 entities matching Source 2 and Source 3 records.

---

## Key Highlights

- **Platform Agnostic**: Runs identically on Windows, Linux, and macOS without hardcoded file paths or OS-specific commands. Uses `pathlib.Path` and environment variable overrides (`ER_DATA_DIR`, `ER_OUTPUT_DIR`, etc.).
- **NVIDIA GPU Acceleration (RTX 4500 Ada)**: Auto-detects CUDA hardware and leverages GPU histogram building (`tree_method='hist'`, `device='cuda'` for XGBoost / LightGBM) with seamless fallback to multi-core CPU.
- **Memory-Safe Streaming**: Employs batched and chunked reading so that datasets exceeding available RAM (10M+ records) stream reliably without Out-Of-Memory (OOM) crashes.
- **Exact Official Macro $F_{0.5}$ Optimization**: Jointly optimizes candidate ranking thresholds, relative score gaps, and the empty-list gate for true singletons.

---

## Project Structure

```text
ML_challenge/
├── entity_resolution/
│   ├── __init__.py           # Package definition
│   ├── config.py             # Platform-agnostic paths, env vars & GPU auto-detection
│   ├── scorer.py             # Exact Macro F_0.5 scorer & breakdown metrics
│   ├── normalize.py          # 7-level text representations & address parser
│   ├── blocking.py           # High-recall candidate retriever & blocking ceiling
│   ├── features.py           # RapidFuzz & address pairwise feature generator
│   ├── model.py              # Pairwise ranker (XGBoost/LightGBM with CUDA support)
│   ├── selection.py          # Source-1-anchored set selection policy
│   ├── baselines.py          # Phase 1 official baselines evaluation
│   ├── train.py              # Leakage-free training pipeline (grouped by S1 ID)
│   └── predict.py            # Streaming test inference & official submission output
├── tests/
│   ├── test_scorer.py        # Unit tests for official Macro F_0.5 & edge cases
│   ├── test_normalize.py     # Unit tests for text normalization & address parser
│   ├── test_features.py      # Unit tests for pairwise feature generation
│   └── test_pipeline.py      # End-to-end synthetic pipeline smoke test
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation file
└── README.md                 # Complete system documentation
```

---

## Setup & Deployment on RTX 4500 Ada Server

### 1. Transfer Code to the Target Server
Pack the directory and transfer via `scp` or `rsync`:
```bash
# On your local machine:
tar -czvf entity_resolution.tar.gz entity_resolution tests requirements.txt setup.py

# Transfer to remote server:
scp entity_resolution.tar.gz user@your-server-ip:/path/to/destination/
```

### 2. Environment Setup on Remote Server
On the remote machine (with NVIDIA RTX 4500 Ada):
```bash
# Unpack
tar -xzvf entity_resolution.tar.gz
cd ML_challenge

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

To enable GPU acceleration for the RTX 4500 Ada, ensure NVIDIA drivers and CUDA toolkit are installed (`nvidia-smi`), then install CUDA-enabled XGBoost or PyTorch:
```bash
pip install xgboost torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage Guide

### Phase 1: Run Official Baselines
Evaluate All-Empty, Exact Name, Exact Name + Postal Code, and Conservative Rule baselines:
```bash
python -m entity_resolution.baselines --sample-size 25000
```

### Phase 2 - 6: Train Model & Optimize Macro $F_{0.5}$
Runs leakage-free train/val split, candidate blocking, pairwise feature extraction, GPU-accelerated model training, and selection policy threshold tuning:
```bash
# Auto-detects RTX 4500 Ada GPU (CUDA)
python -m entity_resolution.train --sample-size 100000 --device cuda --model-type xgboost

# Or run with CPU fallback
python -m entity_resolution.train --sample-size 50000 --device cpu
```

### Phase 7: Generate Submission
Streams through `test_source1.tsv`, retrieves candidates from `test_source2.tsv` and `test_source3.tsv`, scores with the trained ranker on GPU, applies the tuned selection policy, and writes the submission TSV:
```bash
python -m entity_resolution.predict --device cuda --batch-size 50000
```
Output is saved to `output/submission.tsv` with the exact required header:
```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-12345,S3-67890
S1-00002	
S1-00003	S3-99999
```

---

## Running Unit Tests

Run test suite across scorer, normalization, features, and end-to-end pipeline:
```bash
pytest tests/ -v
```
All tests are completely standalone with synthetic data and require no external dataset files.
