import sys
import os
import argparse
import json
import logging
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    MODEL_DIR, OUTPUT_DIR, RANDOM_SEED, PYTHON
)
from scorer import detailed_evaluation, _parse_ids, macro_f05
from normalize import normalize_dataframe, punct_normalize
from blocking import CandidateRetriever
from features import FeatureGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Entity Resolution Matching Model")
    parser.add_argument("--sample-fraction", type=float, default=0.05,
                        help="Fraction of S1 training entities to use (default: 0.05 for fast iteration, 1.0 for full)")
    parser.add_argument("--max-candidates", type=int, default=20,
                        help="Max candidates per S1 entity during blocking (default: 20)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device for training: 'cpu', 'cuda' (RTX 4500 Ada / GPU), or 'auto'")
    parser.add_argument("--model-type", type=str, default="lightgbm", choices=["lightgbm", "xgboost"],
                        help="Classifier type: lightgbm or xgboost")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR,
                        help="Directory to save model artifacts")
    return parser.parse_args()

def tune_selection_thresholds(val_pairs_df, gt_dict):
    """Grid search over inclusion and empty-list thresholds directly maximizing Macro F_0.5."""
    logger.info("Tuning selection policy on validation set...")
    val_pairs_df = val_pairs_df.sort_values(['source1_entity_id', 'score'], ascending=[True, False])
    grouped = val_pairs_df.groupby('source1_entity_id')
    s1_candidates = {s1: list(zip(grp['candidate_id'], grp['score'])) for s1, grp in grouped}
    
    # Ensure all S1 entities in gt_dict are accounted for
    for s1 in gt_dict:
        if s1 not in s1_candidates:
            s1_candidates[s1] = []

    best_score = -1.0
    best_thresh = 0.5
    best_empty_thresh = 0.4
    
    thresholds = [0.25, 0.35, 0.45, 0.50, 0.55, 0.60, 0.70]
    empty_thresholds = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50]

    for thresh in thresholds:
        for ethresh in empty_thresholds:
            preds = {}
            for s1_id, cand_scores in s1_candidates.items():
                if not cand_scores or cand_scores[0][1] < ethresh:
                    preds[s1_id] = set()
                else:
                    preds[s1_id] = {c for c, s in cand_scores if s >= thresh}
            
            score = macro_f05(gt_dict, preds)
            if score > best_score:
                best_score = score
                best_thresh = thresh
                best_empty_thresh = ethresh

    logger.info(f"Optimal Thresholds -> Inclusion: {best_thresh:.2f}, Empty-gate: {best_empty_thresh:.2f} | Val Macro F_0.5: {best_score:.4f}")
    return best_thresh, best_empty_thresh, best_score

def train_model():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    t0 = time.time()
    
    logger.info(f"Starting Entity Resolution Training (Device: {args.device}, Sample: {args.sample_fraction})")
    
    # 1. Load S1 data
    logger.info("Loading S1 entities and Ground Truth...")
    s1_full = pd.read_csv(TRAIN_S1, sep='\t', dtype=str)
    s1_full['business_name'] = s1_full['business_name'].fillna('')
    s1_full['business_address'] = s1_full['business_address'].fillna('')
    s1_full['country'] = s1_full['country'].fillna('')
    
    if args.sample_fraction < 1.0:
        sample_n = max(1000, int(len(s1_full) * args.sample_fraction))
        s1_df = s1_full.sample(n=sample_n, random_state=RANDOM_SEED).reset_index(drop=True)
        logger.info(f"Sampled {len(s1_df)} S1 entities for training & validation.")
    else:
        s1_df = s1_full
        
    s1_ids = set(s1_df['entity_id'])
    
    # Load Ground Truth
    gt_df = pd.read_csv(TRAIN_GT, sep='\t', dtype=str)
    gt_df = gt_df[gt_df['source1_entity_id'].isin(s1_ids)]
    gt_dict = {row['source1_entity_id']: _parse_ids(row['matched_entity_ids']) for _, row in gt_df.iterrows()}
    for s1 in s1_ids:
        if s1 not in gt_dict:
            gt_dict[s1] = set()

    # 2. Leakage-free Grouped Train/Validation Split (by S1 entity ID)
    s1_train, s1_val = train_test_split(s1_df, test_size=0.2, random_state=RANDOM_SEED)
    s1_train = s1_train.reset_index(drop=True)
    s1_val = s1_val.reset_index(drop=True)
    val_gt_dict = {s1: gt_dict[s1] for s1 in s1_val['entity_id']}
    logger.info(f"Train split: {len(s1_train)} entities | Val split: {len(s1_val)} entities")

    # 3. Load candidate pool (S2 and S3)
    logger.info("Loading candidate pool from S2 and S3...")
    needed_pos_ids = set()
    for m in gt_df['matched_entity_ids'].dropna():
        needed_pos_ids.update(x.strip() for x in m.split(',') if x.strip())
        
    s2_rows, s3_rows = [], []
    for path, target_list in [(TRAIN_S2, s2_rows), (TRAIN_S3, s3_rows)]:
        chunk_iter = pd.read_csv(path, sep='\t', chunksize=250000, dtype=str,
                                 usecols=['entity_id', 'business_name', 'business_address', 'country'])
        for chunk in chunk_iter:
            chunk['business_name'] = chunk['business_name'].fillna('')
            chunk['business_address'] = chunk['business_address'].fillna('')
            chunk['country'] = chunk['country'].fillna('')
            target_list.append(chunk)
            
    s2_df = pd.concat(s2_rows, ignore_index=True)
    s3_df = pd.concat(s3_rows, ignore_index=True)
    logger.info(f"Loaded {len(s2_df):,} S2 records and {len(s3_df):,} S3 records.")

    # 4. Candidate Retrieval / Blocking
    logger.info("Building candidate indexes...")
    retriever = CandidateRetriever(config={'max_candidates': args.max_candidates, 'top_k_tfidf': 20})
    retriever.build_indexes(s2_df, s3_df)
    
    logger.info("Retrieving candidates for training set...")
    train_cands_dict = retriever.retrieve_candidates(s1_train, max_candidates=args.max_candidates)
    logger.info("Retrieving candidates for validation set...")
    val_cands_dict = retriever.retrieve_candidates(s1_val, max_candidates=args.max_candidates)

    # Convert to candidate pairs
    def to_pairs(s1_subset, cand_mapping):
        pairs = []
        for s1_id in s1_subset['entity_id']:
            for cid in cand_mapping.get(s1_id, {}):
                pairs.append((s1_id, cid))
        return pairs

    train_pairs = to_pairs(s1_train, train_cands_dict)
    val_pairs = to_pairs(s1_val, val_cands_dict)
    logger.info(f"Generated {len(train_pairs):,} training pairs and {len(val_pairs):,} validation pairs.")

    # 5. Feature Engineering
    logger.info("Generating pairwise features...")
    fg = FeatureGenerator()
    s23_combined = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=['entity_id'])
    
    # Subsample training negatives if ratio > 8:1
    train_labels = [1 if cid in gt_dict.get(s1, set()) else 0 for s1, cid in train_pairs]
    train_labels = np.array(train_labels)
    
    pos_idx = np.where(train_labels == 1)[0]
    neg_idx = np.where(train_labels == 0)[0]
    logger.info(f"Raw training labels -> Positive: {len(pos_idx):,}, Negative: {len(neg_idx):,}")
    
    if len(neg_idx) > len(pos_idx) * 8:
        keep_neg = np.random.choice(neg_idx, size=len(pos_idx) * 8, replace=False)
        selected_idx = np.sort(np.concatenate([pos_idx, keep_neg]))
        train_pairs = [train_pairs[i] for i in selected_idx]
        train_labels = train_labels[selected_idx]
        logger.info(f"Subsampled negatives to 8:1 ratio -> Total train pairs: {len(train_pairs):,}")

    X_train = fg.compute_batch_features(s1_train, s23_combined, train_pairs)
    y_train = train_labels

    val_labels = np.array([1 if cid in gt_dict.get(s1, set()) else 0 for s1, cid in val_pairs])
    X_val = fg.compute_batch_features(s1_val, s23_combined, val_pairs)
    y_val = val_labels

    # 6. Train Classifier (with GPU acceleration if available)
    feature_cols = [c for c in X_train.columns if c not in ['source1_entity_id', 'candidate_id']]
    logger.info(f"Training on {len(feature_cols)} features...")

    use_gpu = False
    if args.device in ["cuda", "auto"]:
        try:
            import torch
            if torch.cuda.is_available():
                use_gpu = True
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA GPU detected: {gpu_name}")
        except Exception:
            pass

    if args.model_type == "xgboost":
        import xgboost as xgb
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 400,
            'random_state': RANDOM_SEED,
            'tree_method': 'hist',
            'device': 'cuda' if use_gpu else 'cpu'
        }
        logger.info(f"Training XGBoost on {xgb_params['device']}...")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train[feature_cols], y_train, eval_set=[(X_val[feature_cols], y_val)], verbose=50)
        val_scores = model.predict_proba(X_val[feature_cols])[:, 1]
        model_save_path = os.path.join(args.model_dir, 'model.xgb')
        model.save_model(model_save_path)
    else:
        import lightgbm as lgb
        lgb_params = {
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
            'random_state': RANDOM_SEED
        }
        if use_gpu:
            lgb_params['device_type'] = 'cuda'
        
        logger.info(f"Training LightGBM (GPU: {use_gpu})...")
        dtrain = lgb.Dataset(X_train[feature_cols], label=y_train)
        dval = lgb.Dataset(X_val[feature_cols], label=y_val, reference=dtrain)
        
        try:
            booster = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=400,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(period=50)]
            )
        except Exception as e:
            if use_gpu:
                logger.warning(f"GPU training failed ({e}), falling back to CPU...")
                lgb_params.pop('device_type', None)
                booster = lgb.train(
                    lgb_params,
                    dtrain,
                    num_boost_round=400,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(period=50)]
                )
            else:
                raise e
                
        val_scores = booster.predict(X_val[feature_cols])
        model_save_path = os.path.join(args.model_dir, 'lgbm_model.txt')
        booster.save_model(model_save_path)

    # 7. Tune Selection Policy on Validation Set
    val_scored_df = pd.DataFrame({
        'source1_entity_id': [s1 for s1, _ in val_pairs],
        'candidate_id': [cid for _, cid in val_pairs],
        'score': val_scores
    })
    
    thresh, empty_thresh, best_val_f05 = tune_selection_thresholds(val_scored_df, val_gt_dict)
    
    thresholds_data = {
        'threshold': thresh,
        'empty_threshold': empty_thresh,
        'best_val_f05': best_val_f05,
        'model_type': args.model_type,
        'feature_cols': feature_cols
    }
    with open(os.path.join(args.model_dir, 'thresholds.json'), 'w') as f:
        json.dump(thresholds_data, f, indent=4)
        
    logger.info(f"Training complete in {time.time() - t0:.1f}s. Artifacts saved in {args.model_dir}.")

if __name__ == '__main__':
    train_model()
