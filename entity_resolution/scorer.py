"""
Official Macro F_0.5 Scorer for Entity Resolution.
Evaluates Source 1 entities matching Source 2 / Source 3 records.
"""

from typing import Dict, Set, Union, List, Optional
import pandas as pd


def _parse_ids(val: Union[str, float, None, Set, List]) -> Set[str]:
    """Parse comma-separated ID string into a set of clean IDs."""
    if val is None or pd.isna(val):
        return set()
    if isinstance(val, (set, list, tuple)):
        return {str(x).strip() for x in val if str(x).strip() and str(x).strip().lower() != "nan"}
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return set()
    return {x.strip() for x in s.split(",") if x.strip() and x.strip().lower() != "nan"}


def entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> float:
    """
    Calculate F_0.5 for a single Source 1 entity according to the official formula:
    
        T = set of true matching IDs
        S = set of predicted matching IDs
        TP = len(T intersection S)
        
        F_0.5 = 1.0                                      if len(T) == 0 and len(S) == 0
                0.0                                      if len(T) == 0 and len(S) > 0
                0.0                                      if len(T) > 0 and TP == 0
                1.25 * TP / (0.25 * len(T) + len(S))     otherwise
    """
    t_len = len(true_ids)
    s_len = len(pred_ids)

    if t_len == 0 and s_len == 0:
        return 1.0
    if t_len == 0 or s_len == 0:
        return 0.0

    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0

    denom = 0.25 * t_len + s_len
    return (1.25 * tp) / denom


def macro_f05(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> float:
    """
    Compute arithmetic mean of F_0.5 across all Source 1 entities in ground truth.
    Every S1 ID in ground_truth is evaluated. Missing predictions are treated as empty sets.
    """
    if not ground_truth:
        return 0.0

    total_score = 0.0
    for s1_id, t_set in ground_truth.items():
        s_set = predictions.get(s1_id, set())
        total_score += entity_f05(t_set, s_set)

    return total_score / len(ground_truth)


def detailed_evaluation(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_countries: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[float, int]]:
    """
    Comprehensive evaluation returning overall macro F_0.5 and sub-group breakdowns:
    - true singleton metrics (empty list accuracy, false nonempty rate)
    - 1 match vs 2+ matches
    - S2 only, S3 only, mixed matches
    - country breakdowns if provided
    """
    n_total = len(ground_truth)
    if n_total == 0:
        return {}

    total_f05 = 0.0
    true_singletons = 0
    empty_list_correct = 0
    false_nonempty = 0

    scores_1_match = []
    scores_2plus_matches = []
    scores_s2_only = []
    scores_s3_only = []
    scores_mixed = []
    country_scores = {}

    for s1_id, t_set in ground_truth.items():
        s_set = predictions.get(s1_id, set())
        score = entity_f05(t_set, s_set)
        total_f05 += score

        # Singleton analysis
        if len(t_set) == 0:
            true_singletons += 1
            if len(s_set) == 0:
                empty_list_correct += 1
            else:
                false_nonempty += 1
        elif len(t_set) == 1:
            scores_1_match.append(score)
        else:
            scores_2plus_matches.append(score)

        # Source composition
        if len(t_set) > 0:
            has_s2 = any(x.startswith("S2-") for x in t_set)
            has_s3 = any(x.startswith("S3-") for x in t_set)
            if has_s2 and not has_s3:
                scores_s2_only.append(score)
            elif has_s3 and not has_s2:
                scores_s3_only.append(score)
            else:
                scores_mixed.append(score)

        # Country
        if s1_countries and s1_id in s1_countries:
            c = s1_countries[s1_id]
            if c not in country_scores:
                country_scores[c] = []
            country_scores[c].append(score)

    metrics = {
        "macro_f05": total_f05 / n_total,
        "total_entities": n_total,
        "true_singletons_count": true_singletons,
        "true_singletons_pct": (true_singletons / n_total) * 100.0,
        "empty_list_accuracy": (empty_list_correct / true_singletons) if true_singletons > 0 else 1.0,
        "false_nonempty_rate": (false_nonempty / true_singletons) if true_singletons > 0 else 0.0,
        "f05_1_match": (sum(scores_1_match) / len(scores_1_match)) if scores_1_match else 0.0,
        "f05_2plus_matches": (sum(scores_2plus_matches) / len(scores_2plus_matches)) if scores_2plus_matches else 0.0,
        "f05_s2_only": (sum(scores_s2_only) / len(scores_s2_only)) if scores_s2_only else 0.0,
        "f05_s3_only": (sum(scores_s3_only) / len(scores_s3_only)) if scores_s3_only else 0.0,
        "f05_mixed": (sum(scores_mixed) / len(scores_mixed)) if scores_mixed else 0.0,
    }

    for c, c_list in country_scores.items():
        metrics[f"f05_country_{c}"] = sum(c_list) / len(c_list)

    return metrics


def load_ground_truth(tsv_path: Union[str, pd.DataFrame]) -> Dict[str, Set[str]]:
    """Load ground truth mapping from TSV file or DataFrame."""
    if isinstance(tsv_path, pd.DataFrame):
        df = tsv_path
    else:
        df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    
    col_s1 = "source1_entity_id" if "source1_entity_id" in df.columns else df.columns[0]
    col_matches = "matched_entity_ids" if "matched_entity_ids" in df.columns else df.columns[1]
    
    gt = {}
    for s1_id, match_str in zip(df[col_s1].values, df[col_matches].values):
        gt[str(s1_id).strip()] = _parse_ids(match_str)
    return gt


def load_predictions(tsv_path: Union[str, pd.DataFrame]) -> Dict[str, Set[str]]:
    """Load predictions mapping from TSV file or DataFrame."""
    return load_ground_truth(tsv_path)
