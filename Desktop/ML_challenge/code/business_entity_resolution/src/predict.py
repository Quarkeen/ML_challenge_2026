import sys, os
import pandas as pd
import numpy as np
import json
import logging
import lightgbm as lgb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from normalize import normalize_dataframe
from blocking import CandidateRetriever
from features import FeatureGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(dir_path):
    s1 = pd.read_csv(os.path.join(dir_path, 'test_source1.tsv'), sep='\t')
    s2 = pd.read_csv(os.path.join(dir_path, 'test_source2.tsv'), sep='\t')
    s3 = pd.read_csv(os.path.join(dir_path, 'test_source3.tsv'), sep='\t')
    return s1, s2, s3

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logging.info("Loading test data...")
    s1, s2, s3 = load_data(TEST_DIR)
    
    logging.info("Normalizing data...")
    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    
    # Prepare combined s23 for feature generation
    s23_combined = pd.concat([s2, s3], ignore_index=True)
    
    logging.info("Building blocking indexes...")
    retriever = CandidateRetriever()
    retriever.fit(s2, s3)
    
    logging.info("Loading model and thresholds...")
    model_path = os.path.join(MODEL_DIR, 'lgbm_model.txt')
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}. Run train.py first.")
        return
        
    model = lgb.Booster(model_file=model_path)
    
    thresholds_path = os.path.join(MODEL_DIR, 'thresholds.json')
    if os.path.exists(thresholds_path):
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)
    else:
        logging.warning("Thresholds file not found. Using default 0.5 threshold.")
        thresholds = {'global': 0.5}
        
    global_threshold = thresholds.get('global', 0.5)
    
    fg = FeatureGenerator()
    
    matching_results = []
    candidate_results = []
    
    batch_size = 10000
    for i in tqdm(range(0, len(s1), batch_size), desc="Predicting in batches"):
        batch_s1 = s1.iloc[i:i+batch_size]
        
        # Retrieve candidates
        candidates_dict = retriever.retrieve(batch_s1, top_k=20)
        
        # Build pairs for feature generation
        pairs = []
        for s1_idx, row in batch_s1.iterrows():
            s1_id = row['entity_id']
            cand_list = candidates_dict.get(s1_id, [])
            for cand in cand_list:
                # cand could be a dict or a string depending on CandidateRetriever implementation
                cand_id = cand['id'] if isinstance(cand, dict) else cand
                pairs.append((s1_id, cand_id))
                
        if not pairs:
            for s1_idx, row in batch_s1.iterrows():
                s1_id = row['entity_id']
                matching_results.append(f"{s1_id}\t")
                candidate_results.append(f"{s1_id}\t")
            continue
            
        pairs_df = pd.DataFrame(pairs, columns=['source1_entity_id', 'candidate_entity_id'])
        
        # Generate features
        features_df = fg.generate_features(pairs_df, batch_s1, s23_combined)
        
        # Score candidates
        feature_cols = [c for c in features_df.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label']]
        X = features_df[feature_cols]
        preds = model.predict(X)
        features_df['score'] = preds
        
        # Apply selection policy
        for s1_idx, row in batch_s1.iterrows():
            s1_id = row['entity_id']
            country = row['country'] if not pd.isna(row['country']) else 'global'
            
            thresh = thresholds.get(country, global_threshold)
            
            s1_preds = features_df[features_df['source1_entity_id'] == s1_id]
            matches = s1_preds[s1_preds['score'] >= thresh]['candidate_entity_id'].tolist()
            cands = s1_preds['candidate_entity_id'].tolist()
            
            matching_results.append(f"{s1_id}\t{','.join(matches)}")
            candidate_results.append(f"{s1_id}\t{','.join(cands)}")
            
    logging.info("Writing output...")
    with open(os.path.join(OUTPUT_DIR, 'matching_results.tsv'), 'w') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("\n".join(matching_results) + "\n")
        
    with open(os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv'), 'w') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("\n".join(candidate_results) + "\n")
        
    logging.info("Running validator...")
    cmd = f'{PYTHON} student_resource/utils/validate_submission.py --matching {OUTPUT_DIR}/matching_results.tsv --candidate {OUTPUT_DIR}/candidate_pairs.tsv --test-dir student_resource/dataset/test'
    os.system(cmd)
    
if __name__ == '__main__':
    main()
