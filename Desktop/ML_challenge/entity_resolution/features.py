"""
Phase 4: Pairwise Feature Generation for Entity Matching.
Computes fine-grained name, address, source, and retrieval similarity features
using RapidFuzz and local deterministic token statistics without external lookups.
"""

import re
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .normalize import (
    punct_normalize,
    compressed_name,
    strip_legal_suffix,
    parse_address_fields,
    alphanum_only,
)


class FeatureGenerator:
    """
    Computes pairwise feature vectors between Source 1 entities and Candidate entities.
    Vectorized and cached for speed and memory efficiency.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @staticmethod
    def _char_ngrams(s: str, n: int = 3) -> set:
        if len(s) < n:
            return {s} if s else set()
        return {s[i:i+n] for i in range(len(s) - n + 1)}

    def compute_pair_features(
        self,
        s1_row: Union[pd.Series, Dict[str, Any]],
        cand_row: Union[pd.Series, Dict[str, Any]],
        retrieval_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive feature dictionary for a single (S1, Candidate) pair.
        """
        feats = {}

        # ----------------- Name Processing -----------------
        raw_n1 = s1_row.get("business_name", "")
        raw_n2 = cand_row.get("business_name", "")

        n1 = punct_normalize(raw_n1)
        n2 = punct_normalize(raw_n2)
        n1_comp = compressed_name(raw_n1)
        n2_comp = compressed_name(raw_n2)

        s1_stripped, s1_suf = strip_legal_suffix(raw_n1)
        s2_stripped, s2_suf = strip_legal_suffix(raw_n2)

        # Name Exact Similarities
        feats["name_exact_punct"] = float(n1 == n2 and bool(n1))
        feats["name_exact_compressed"] = float(n1_comp == n2_comp and len(n1_comp) >= 4)
        feats["name_exact_stripped"] = float(s1_stripped == s2_stripped and bool(s1_stripped))
        feats["legal_suffix_match"] = float(bool(s1_suf) and s1_suf == s2_suf)

        # RapidFuzz Name Similarities (0.0 to 1.0 scale)
        feats["name_fuzz_ratio"] = fuzz.ratio(n1, n2) / 100.0
        feats["name_fuzz_partial"] = fuzz.partial_ratio(n1, n2) / 100.0
        feats["name_token_sort"] = fuzz.token_sort_ratio(n1, n2) / 100.0
        feats["name_token_set"] = fuzz.token_set_ratio(n1, n2) / 100.0
        feats["name_stripped_ratio"] = fuzz.ratio(s1_stripped, s2_stripped) / 100.0

        # Character 3-gram Jaccard
        g1 = self._char_ngrams(n1, 3)
        g2 = self._char_ngrams(n2, 3)
        union_g = g1 | g2
        feats["name_ngram_jaccard"] = (len(g1 & g2) / len(union_g)) if union_g else 0.0

        # Word Token Overlap
        t1 = set(n1.split()) - {"inc", "llc", "ltd", "pvt", "private", "limited", "co", "corp", "the", "and"}
        t2 = set(n2.split()) - {"inc", "llc", "ltd", "pvt", "private", "limited", "co", "corp", "the", "and"}
        shared_t = t1 & t2
        total_t = t1 | t2
        feats["name_shared_token_count"] = float(len(shared_t))
        feats["name_token_jaccard"] = (len(shared_t) / len(total_t)) if total_t else 0.0
        feats["name_len_diff"] = float(abs(len(n1) - len(n2)))
        feats["name_len_ratio"] = min(len(n1), len(n2)) / max(1, max(len(n1), len(n2)))

        # ----------------- Address Processing -----------------
        raw_a1 = s1_row.get("business_address", "")
        raw_a2 = cand_row.get("business_address", "")
        c1 = s1_row.get("country", "")
        c2 = cand_row.get("country", "")

        a1_norm = punct_normalize(raw_a1)
        a2_norm = punct_normalize(raw_a2)

        # Missingness flags
        s1_addr_missing = not bool(a1_norm) or a1_norm in ("nan", "none")
        cand_addr_missing = not bool(a2_norm) or a2_norm in ("nan", "none")
        feats["s1_addr_missing"] = float(s1_addr_missing)
        feats["cand_addr_missing"] = float(cand_addr_missing)
        feats["both_addr_present"] = float(not s1_addr_missing and not cand_addr_missing)

        # Parsed Address Components
        p1 = parse_address_fields(raw_a1, c1)
        p2 = parse_address_fields(raw_a2, c2)

        hn1 = p1["house_number"]
        hn2 = p2["house_number"]
        if hn1 and hn2:
            feats["hn_exact_match"] = float(hn1 == hn2)
            feats["hn_contradiction"] = float(hn1 != hn2)
        else:
            feats["hn_exact_match"] = 0.0
            feats["hn_contradiction"] = 0.0

        pc1 = p1["postal_code"]
        pc2 = p2["postal_code"]
        if pc1 and pc2:
            feats["postal_exact_match"] = float(pc1 == pc2)
            feats["postal_prefix_match"] = float(pc1[:3] == pc2[:3])
            feats["postal_contradiction"] = float(pc1 != pc2)
        else:
            feats["postal_exact_match"] = 0.0
            feats["postal_prefix_match"] = 0.0
            feats["postal_contradiction"] = 0.0

        # Address Fuzzy Similarities (0.0 if either address missing)
        if not s1_addr_missing and not cand_addr_missing:
            feats["addr_fuzz_ratio"] = fuzz.ratio(a1_norm, a2_norm) / 100.0
            feats["addr_token_sort"] = fuzz.token_sort_ratio(a1_norm, a2_norm) / 100.0
            feats["addr_token_set"] = fuzz.token_set_ratio(a1_norm, a2_norm) / 100.0
        else:
            feats["addr_fuzz_ratio"] = 0.0
            feats["addr_token_sort"] = 0.0
            feats["addr_token_set"] = 0.0

        # ----------------- Source & Candidate Metadata -----------------
        cand_id = str(cand_row.get("entity_id", ""))
        feats["is_source2"] = float(cand_id.startswith("S2-"))
        feats["is_source3"] = float(cand_id.startswith("S3-"))
        feats["country_match"] = float(str(c1).strip().upper() == str(c2).strip().upper())

        # Retrieval Metadata features
        if retrieval_meta:
            feats["retrieval_score"] = float(retrieval_meta.get("retrieval_score", 0.0))
            methods = str(retrieval_meta.get("retrieval_methods", ""))
            feats["retrieved_by_exact"] = float("exact_name" in methods)
            feats["retrieved_by_comp"] = float("comp_name" in methods)
            feats["retrieved_by_token"] = float("token" in methods)
            feats["retrieved_by_addr"] = float("addr" in methods)
        else:
            feats["retrieval_score"] = 0.0
            feats["retrieved_by_exact"] = 0.0
            feats["retrieved_by_comp"] = 0.0
            feats["retrieved_by_token"] = 0.0
            feats["retrieved_by_addr"] = 0.0

        return feats

    def compute_batch_features(
        self,
        s1_df: pd.DataFrame,
        cand_df: pd.DataFrame,
        pairs: List[Tuple[str, str]],
        retrieval_info: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Compute features for a list of (s1_id, candidate_id) pairs.
        Returns a DataFrame of features ready for model training / scoring.
        """
        if not pairs:
            return pd.DataFrame()

        # Build fast row lookups
        s1_dict = s1_df.set_index("entity_id").to_dict(orient="index") if "entity_id" in s1_df.columns else {}
        cand_dict = cand_df.set_index("entity_id").to_dict(orient="index") if "entity_id" in cand_df.columns else {}

        rows = []
        for s1_id, cand_id in pairs:
            s1_row = s1_dict.get(s1_id, {})
            cand_row = cand_dict.get(cand_id, {})
            rmeta = retrieval_info.get((s1_id, cand_id)) if retrieval_info else None

            feat = self.compute_pair_features(s1_row, cand_row, retrieval_meta=rmeta)
            feat["source1_entity_id"] = s1_id
            feat["candidate_id"] = cand_id
            rows.append(feat)

        return pd.DataFrame(rows)

    def generate_features(
        self,
        pairs_df: Union[pd.DataFrame, List[Tuple[str, str]]],
        s1_df: pd.DataFrame,
        cand_df: pd.DataFrame,
        retrieval_info: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """Standard alias matching pipeline interface."""
        if isinstance(pairs_df, pd.DataFrame):
            c1 = "source1_entity_id" if "source1_entity_id" in pairs_df.columns else pairs_df.columns[0]
            c2 = "candidate_id" if "candidate_id" in pairs_df.columns else pairs_df.columns[1]
            pairs = list(zip(pairs_df[c1].values, pairs_df[c2].values))
        else:
            pairs = pairs_df
        return self.compute_batch_features(s1_df, cand_df, pairs, retrieval_info=retrieval_info)
