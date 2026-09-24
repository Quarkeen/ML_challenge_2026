import sys, os
import pandas as pd
import numpy as np
from collections import defaultdict
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT
from scorer import detailed_evaluation, _parse_ids
from normalize import punct_normalize, strip_legal_suffix, extract_postal_code

def run_baselines(sample_size=50000):
    t0 = time.time()
    print(f"=== PHASE 1: BASELINES & SCORER EVALUATION (Sample: {sample_size}) ===")
    
    # Load S1 sample
    print(f"Loading {sample_size} S1 records...")
    s1_df = pd.read_csv(TRAIN_S1, sep='\t', nrows=sample_size, dtype=str)
    s1_df['business_name'] = s1_df['business_name'].fillna('')
    s1_df['business_address'] = s1_df['business_address'].fillna('')
    s1_df['country'] = s1_df['country'].fillna('')
    s1_ids = set(s1_df['entity_id'])
    
    # Load Ground Truth for this sample
    print("Loading Ground Truth...")
    gt_df = pd.read_csv(TRAIN_GT, sep='\t', dtype=str)
    gt_df = gt_df[gt_df['source1_entity_id'].isin(s1_ids)]
    gt_dict = {row['source1_entity_id']: _parse_ids(row['matched_entity_ids']) for _, row in gt_df.iterrows()}
    # Ensure all s1_ids are present
    for s1_id in s1_ids:
        if s1_id not in gt_dict:
            gt_dict[s1_id] = set()
            
    s1_country = dict(zip(s1_df['entity_id'], s1_df['country']))

    # Baseline 1: All-Empty
    print("\n--- Baseline 1: All-Empty Prediction ---")
    preds_empty = {s1_id: set() for s1_id in s1_ids}
    metrics_empty = detailed_evaluation(gt_dict, preds_empty, s1_country)
    for k, v in metrics_empty.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # To evaluate matching baselines, we need candidate pools from S2 and S3
    print("\nLoading S2 and S3 candidate records (optimized fast streaming)...")
    name_to_s23 = defaultdict(list)
    name_addr_to_s23 = defaultdict(list)
    
    for path, source in [(TRAIN_S2, 'S2'), (TRAIN_S3, 'S3')]:
        t_src = time.time()
        print(f"Indexing {source} ({os.path.basename(path)})...")
        chunk_iter = pd.read_csv(path, sep='\t', chunksize=500000, dtype=str,
                                 usecols=['entity_id', 'business_name', 'business_address', 'country'])
        for chunk in chunk_iter:
            ids = chunk['entity_id'].values
            names = chunk['business_name'].fillna('').values
            addrs = chunk['business_address'].fillna('').values
            countries = chunk['country'].fillna('').values
            
            for cid, name, addr, c in zip(ids, names, addrs, countries):
                if not name:
                    continue
                norm_n = punct_normalize(name)
                if norm_n:
                    name_to_s23[(c, norm_n)].append(cid)
                    postal = extract_postal_code(addr, c)
                    if postal:
                        name_addr_to_s23[(c, norm_n, postal)].append(cid)
                        
        print(f"  Indexed {source} in {time.time() - t_src:.1f}s.")

    print(f"Total unique (country, name) keys: {len(name_to_s23):,}")

    # Baseline 2: Simple Exact Normalized Name Matching (within Country)
    print("\n--- Baseline 2: Exact Normalized Name Matching ---")
    preds_exact_name = {}
    for s1_id, name, c in zip(s1_df['entity_id'].values, s1_df['business_name'].values, s1_df['country'].values):
        norm_n = punct_normalize(name)
        matches = name_to_s23.get((c, norm_n), [])
        preds_exact_name[s1_id] = set(matches)

    metrics_exact = detailed_evaluation(gt_dict, preds_exact_name, s1_country)
    for k, v in metrics_exact.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Baseline 3: Exact Normalized Name + Postal Code
    print("\n--- Baseline 3: Exact Name + Postal Code ---")
    preds_name_postal = {}
    for s1_id, name, addr, c in zip(s1_df['entity_id'].values, s1_df['business_name'].values, s1_df['business_address'].values, s1_df['country'].values):
        norm_n = punct_normalize(name)
        postal = extract_postal_code(addr, c)
        if norm_n and postal:
            matches = name_addr_to_s23.get((c, norm_n, postal), [])
        else:
            matches = []
        preds_name_postal[s1_id] = set(matches)

    metrics_np = detailed_evaluation(gt_dict, preds_name_postal, s1_country)
    for k, v in metrics_np.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Baseline 4: Conservative Rule-Based Baseline
    # Rule: If exact name has 1-4 matches in same country, accept them.
    # If >4 matches, require matching postal code.
    print("\n--- Baseline 4: Conservative Rule-Based Baseline ---")
    preds_conservative = {}
    for s1_id, name, addr, c in zip(s1_df['entity_id'].values, s1_df['business_name'].values, s1_df['business_address'].values, s1_df['country'].values):
        norm_n = punct_normalize(name)
        name_matches = name_to_s23.get((c, norm_n), [])
        if len(name_matches) == 0:
            preds_conservative[s1_id] = set()
        elif len(name_matches) <= 4:
            preds_conservative[s1_id] = set(name_matches)
        else:
            postal = extract_postal_code(addr, c)
            if postal:
                np_matches = name_addr_to_s23.get((c, norm_n, postal), [])
                preds_conservative[s1_id] = set(np_matches)
            else:
                preds_conservative[s1_id] = set()

    metrics_cons = detailed_evaluation(gt_dict, preds_conservative, s1_country)
    for k, v in metrics_cons.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nAll baselines completed in {time.time() - t0:.1f}s.")
    return {
        'all_empty': metrics_empty,
        'exact_name': metrics_exact,
        'name_postal': metrics_np,
        'conservative': metrics_cons
    }

if __name__ == '__main__':
    run_baselines(50000)
