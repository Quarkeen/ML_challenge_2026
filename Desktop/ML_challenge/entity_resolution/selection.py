"""
Phase 6: Source-1-Anchored Set Selection Policy.
Jointly optimizes the empty-list decision and the multi-match candidate cutoff
specifically against the official macro F_0.5 objective.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
from .scorer import macro_f05


class SetSelector:
    """
    Decides the final predicted set of candidate IDs for each Source 1 entity:
    - Empty-list gate: if max candidate score is below threshold, output empty set
    - Multi-match cutoff: score threshold, relative gap from top score, max candidates limit
    - Deduplication: ensures unique IDs per Source 1 entity
    """

    def __init__(
        self,
        match_threshold: float = 0.50,
        empty_threshold: float = 0.35,
        relative_gap: float = 0.20,
        max_matches: int = 10,
    ):
        self.match_threshold = match_threshold
        self.empty_threshold = empty_threshold
        self.relative_gap = relative_gap
        self.max_matches = max_matches

    def select_for_entity(self, candidate_scores: List[Tuple[str, float]]) -> List[str]:
        """
        Apply decision policy to candidate list [(candidate_id, score), ...]
        sorted descending by score.
        Returns: list of predicted candidate IDs.
        """
        if not candidate_scores:
            return []

        # Sort descending by score
        sorted_cands = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        top_id, top_score = sorted_cands[0]

        # 1. Empty-list gate: if even top candidate is weak, return empty list
        if top_score < self.empty_threshold:
            return []

        selected = []
        seen = set()

        for cid, score in sorted_cands:
            # Must satisfy absolute threshold
            if score < self.match_threshold:
                break

            # Must satisfy relative gap from top match
            if (top_score - score) > self.relative_gap:
                break

            # Deduplicate
            if cid not in seen:
                seen.add(cid)
                selected.append(cid)

            # Cap maximum match list size
            if len(selected) >= self.max_matches:
                break

        return selected

    def predict_sets(
        self,
        scored_pairs_df: pd.DataFrame,
        all_s1_ids: Optional[List[str]] = None,
    ) -> Dict[str, Set[str]]:
        """
        Generate predicted match sets for all Source 1 entities.
        Guarantees that every S1 entity in all_s1_ids appears in output.
        """
        predictions: Dict[str, Set[str]] = {}
        if all_s1_ids:
            for s1_id in all_s1_ids:
                predictions[str(s1_id).strip()] = set()

        # Group scored candidates by S1 ID
        if len(scored_pairs_df) > 0:
            c_s1 = "source1_entity_id"
            c_cand = "candidate_id"
            c_score = "score"

            for s1_id, group in scored_pairs_df.groupby(c_s1):
                cands = list(zip(group[c_cand].values, group[c_score].values))
                chosen = self.select_for_entity(cands)
                predictions[str(s1_id).strip()] = set(chosen)

        return predictions

    def tune_thresholds(
        self,
        scored_val_df: pd.DataFrame,
        val_ground_truth: Dict[str, Set[str]],
        match_grid: Optional[List[float]] = None,
        empty_grid: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Grid search optimal threshold and empty_threshold parameters directly against macro F_0.5.
        """
        match_grid = match_grid or [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        empty_grid = empty_grid or [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

        best_score = -1.0
        best_params = {
            "match_threshold": self.match_threshold,
            "empty_threshold": self.empty_threshold,
            "relative_gap": self.relative_gap,
            "max_matches": self.max_matches,
        }

        all_val_s1 = list(val_ground_truth.keys())

        # Pre-group candidates by s1_id for rapid grid evaluation
        grouped_cands: Dict[str, List[Tuple[str, float]]] = {}
        for s1_id in all_val_s1:
            grouped_cands[s1_id] = []

        if len(scored_val_df) > 0:
            for s1_id, group in scored_val_df.groupby("source1_entity_id"):
                s1_str = str(s1_id).strip()
                if s1_str in grouped_cands:
                    grouped_cands[s1_str] = sorted(
                        zip(group["candidate_id"].values, group["score"].values),
                        key=lambda x: x[1],
                        reverse=True,
                    )

        # Evaluate parameter grid
        for eth in empty_grid:
            for mth in match_grid:
                if mth < eth:
                    continue  # Match threshold should be >= empty threshold

                preds = {}
                for s1_id, cands in grouped_cands.items():
                    if not cands or cands[0][1] < eth:
                        preds[s1_id] = set()
                        continue

                    top_score = cands[0][1]
                    s_list = []
                    for cid, sc in cands:
                        if sc < mth or (top_score - sc) > self.relative_gap:
                            break
                        s_list.append(cid)
                        if len(s_list) >= self.max_matches:
                            break
                    preds[s1_id] = set(s_list)

                score = macro_f05(val_ground_truth, preds)
                if score > best_score:
                    best_score = score
                    best_params["match_threshold"] = mth
                    best_params["empty_threshold"] = eth

        # Update instance parameters
        self.match_threshold = best_params["match_threshold"]
        self.empty_threshold = best_params["empty_threshold"]

        return {
            "best_macro_f05": best_score,
            "best_params": best_params,
        }
