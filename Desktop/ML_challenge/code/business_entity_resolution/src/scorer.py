import pandas as pd
from typing import Dict, Set, Optional

def entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> float:
    """Calculate F0.5 score for a single entity."""
    if not true_ids and not pred_ids:
        return 1.0
    tp = len(true_ids.intersection(pred_ids))
    return 1.25 * tp / (0.25 * len(true_ids) + len(pred_ids))

def macro_f05(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> float:
    """Calculate macro F0.5 score over all entities."""
    if not ground_truth:
        return 0.0
    
    total_score = 0.0
    for s1_id, t in ground_truth.items():
        s = predictions.get(s1_id, set())
        total_score += entity_f05(t, s)
        
    return total_score / len(ground_truth)

def _parse_ids(val) -> Set[str]:
    if pd.isna(val) or not str(val).strip():
        return set()
    return set(x.strip() for x in str(val).split(',') if x.strip())

def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    """Load ground truth from TSV."""
    df = pd.read_csv(path, sep='\t')
    gt = {}
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        matched_str = row['matched_entity_ids']
        gt[s1_id] = _parse_ids(matched_str)
    return gt

def load_predictions(path: str) -> Dict[str, Set[str]]:
    """Load predictions from TSV."""
    df = pd.read_csv(path, sep='\t')
    preds = {}
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        matched_str = row['matched_entity_ids']
        preds[s1_id] = _parse_ids(matched_str)
    return preds

def detailed_evaluation(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]], s1_countries: Optional[Dict[str, str]] = None) -> dict:
    total = len(ground_truth)
    if total == 0:
        return {}

    overall_score = 0.0
    singletons = 0
    empty_list_correct = 0
    false_nonempty = 0
    
    len_1_score = 0.0
    len_1_count = 0
    len_2plus_score = 0.0
    len_2plus_count = 0
    
    s2_only_score = 0.0
    s2_only_count = 0
    s3_only_score = 0.0
    s3_only_count = 0
    mixed_score = 0.0
    mixed_count = 0
    
    country_scores = {}
    country_counts = {}
    
    for s1_id, t in ground_truth.items():
        s = predictions.get(s1_id, set())
        score = entity_f05(t, s)
        overall_score += score
        
        if not t:
            singletons += 1
            if not s:
                empty_list_correct += 1
            else:
                false_nonempty += 1
        
        if len(t) == 1:
            len_1_score += score
            len_1_count += 1
        elif len(t) >= 2:
            len_2plus_score += score
            len_2plus_count += 1
            
        if t:
            has_s2 = any(x.startswith('S2-') for x in t)
            has_s3 = any(x.startswith('S3-') for x in t)
            if has_s2 and not has_s3:
                s2_only_score += score
                s2_only_count += 1
            elif has_s3 and not has_s2:
                s3_only_score += score
                s3_only_count += 1
            elif has_s2 and has_s3:
                mixed_score += score
                mixed_count += 1
                
        if s1_countries:
            country = s1_countries.get(s1_id, 'UNKNOWN')
            country_scores[country] = country_scores.get(country, 0.0) + score
            country_counts[country] = country_counts.get(country, 0) + 1

    res = {
        'macro_f05': overall_score / total,
        'true_singletons_count': singletons,
        'true_singletons_pct': singletons / total * 100 if total > 0 else 0,
        'empty_list_accuracy': empty_list_correct / singletons if singletons > 0 else 0,
        'false_nonempty_rate': false_nonempty / singletons if singletons > 0 else 0,
        'f05_1_match': len_1_score / len_1_count if len_1_count > 0 else 0,
        'f05_2plus_matches': len_2plus_score / len_2plus_count if len_2plus_count > 0 else 0,
        'f05_s2_only': s2_only_score / s2_only_count if s2_only_count > 0 else 0,
        'f05_s3_only': s3_only_score / s3_only_count if s3_only_count > 0 else 0,
        'f05_mixed': mixed_score / mixed_count if mixed_count > 0 else 0,
    }
    
    if s1_countries:
        for c in country_scores:
            res[f'f05_country_{c}'] = country_scores[c] / country_counts[c]
            
    return res

def evaluate(gt_path: str, pred_path: str, s1_countries: Optional[Dict[str, str]] = None) -> dict:
    gt = load_ground_truth(gt_path)
    preds = load_predictions(pred_path)
    return detailed_evaluation(gt, preds, s1_countries)
