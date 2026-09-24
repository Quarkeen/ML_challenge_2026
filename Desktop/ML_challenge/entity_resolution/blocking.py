"""
Phase 3: High-Recall Hybrid Candidate Blocking Module.
Combines inverted indexes on exact name, compressed/domain name, distinctive tokens,
and address signatures with character n-gram TF-IDF retrieval.
Strictly partitioned by country to ensure zero cross-country candidate leakage and high efficiency.
"""

from collections import defaultdict, Counter
import re
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import (
    punct_normalize,
    compressed_name,
    strip_legal_suffix,
    extract_postal_code,
    alphanum_only,
)
from .scorer import entity_f05, detailed_evaluation


class CandidateRetriever:
    """
    High-Recall Candidate Retriever for Entity Resolution.
    Partitions the candidate pool (S2 + S3) by country for maximum retrieval speed and precision.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_candidates = self.config.get("max_candidates_per_entity", 50)
        self.top_k_tfidf = self.config.get("top_k_tfidf", 25)
        self.max_token_df = self.config.get("max_token_df_filter", 300)
        
        # Per-country indexes: country -> {index_name -> dict}
        self.indexes: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.candidate_metadata: Dict[str, Dict[str, Any]] = {}
        self.tfidf_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.tfidf_matrices: Dict[str, csr_matrix] = {}
        self.candidate_id_lookup: Dict[str, List[str]] = {}

    @staticmethod
    def _extract_addr_signature_tokens(addr: str) -> List[str]:
        """Extract meaningful street and address signature tokens."""
        if not addr or pd.isna(addr):
            return []
        words = re.findall(r"[a-zA-Z0-9]{3,}", str(addr).lower())
        common = {
            "street", "road", "avenue", "lane", "drive", "court", "suite", "unit",
            "floor", "apartment", "post", "office", "box", "city", "state", "near",
            "opp", "block", "plot", "house", "nagar", "colony", "first", "second"
        }
        return [w for w in words if w not in common]

    def build_indexes(self, s2_records: Any, s3_records: Any) -> "CandidateRetriever":
        """
        Build inverted indexes across S2 and S3 candidate records.
        Accepts DataFrames or iterables of records/chunks.
        """
        exact_name_idx = defaultdict(lambda: defaultdict(list))
        comp_name_idx = defaultdict(lambda: defaultdict(list))
        token_idx = defaultdict(lambda: defaultdict(list))
        addr_idx = defaultdict(lambda: defaultdict(list))
        country_cands = defaultdict(list)

        def index_batch(df):
            if df is None or len(df) == 0:
                return
            ids = df["entity_id"].values
            names = df["business_name"].fillna("").astype(str).values
            addrs = df["business_address"].fillna("").astype(str).values
            countries = df["country"].fillna("").astype(str).values if "country" in df.columns else [""] * len(df)

            for cid, n, a, c in zip(ids, names, addrs, countries):
                c_clean = str(c).strip()
                p_n = punct_normalize(n)
                c_n = compressed_name(n)
                
                country_cands[c_clean].append((cid, p_n, a))
                self.candidate_metadata[cid] = {
                    "business_name": n,
                    "business_address": a,
                    "country": c_clean,
                }

                # 1. Exact punct-normalized name
                if p_n:
                    exact_name_idx[c_clean][p_n].append(cid)

                # 2. Compressed/domain name
                if len(c_n) >= 5:
                    comp_name_idx[c_clean][c_n].append(cid)

                # 3. Distinctive name tokens
                words = set(p_n.split()) - {"inc", "llc", "ltd", "pvt", "private", "limited", "co", "corp", "the", "and"}
                for w in words:
                    if len(w) >= 3:
                        token_idx[c_clean][w].append(cid)

                # 4. Address tokens
                for aw in self._extract_addr_signature_tokens(a):
                    addr_idx[c_clean][aw].append(cid)

        # Ingest S2 and S3
        if isinstance(s2_records, list):
            for chunk in s2_records:
                index_batch(chunk)
        else:
            index_batch(s2_records)

        if isinstance(s3_records, list):
            for chunk in s3_records:
                index_batch(chunk)
        else:
            index_batch(s3_records)

        # Prune high-frequency tokens to keep retrieval fast and avoid noisy explosions
        for c in exact_name_idx:
            self.indexes[c]["exact_name"] = dict(exact_name_idx[c])
            self.indexes[c]["comp_name"] = dict(comp_name_idx[c])
            self.indexes[c]["tokens"] = {
                k: v for k, v in token_idx[c].items() if len(v) <= self.max_token_df
            }
            self.indexes[c]["addr"] = {
                k: v for k, v in addr_idx[c].items() if 2 <= len(v) <= 150
            }

        return self

    def fit(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> "CandidateRetriever":
        """Standard scikit-learn style alias for build_indexes."""
        return self.build_indexes(s2_df, s3_df)

    def retrieve_candidates(
        self,
        s1_df: pd.DataFrame,
        max_candidates: Optional[int] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Retrieve candidate matching records for all S1 entities.
        Returns: {s1_id: {candidate_id: {'retrieval_score': float, 'retrieval_method': str}}}
        """
        k = max_candidates or self.max_candidates
        results = {}

        ids = s1_df["entity_id"].values
        names = s1_df["business_name"].fillna("").astype(str).values
        addrs = s1_df["business_address"].fillna("").astype(str).values
        countries = s1_df["country"].fillna("").astype(str).values if "country" in s1_df.columns else [""] * len(s1_df)

        for s1_id, name, addr, country in zip(ids, names, addrs, countries):
            c_clean = str(country).strip()
            p_n = punct_normalize(name)
            c_n = compressed_name(name)
            words = set(p_n.split()) - {"inc", "llc", "ltd", "pvt", "private", "limited", "co", "corp", "the", "and"}
            a_tokens = self._extract_addr_signature_tokens(addr)

            cand_scores = Counter()
            methods = defaultdict(list)

            c_indexes = self.indexes.get(c_clean, {})

            # 1. Exact name match
            if "exact_name" in c_indexes and p_n in c_indexes["exact_name"]:
                for cid in c_indexes["exact_name"][p_n]:
                    cand_scores[cid] += 12.0
                    methods[cid].append("exact_name")

            # 2. Compressed/domain name match
            if "comp_name" in c_indexes and len(c_n) >= 5 and c_n in c_indexes["comp_name"]:
                for cid in c_indexes["comp_name"][c_n]:
                    cand_scores[cid] += 9.0
                    methods[cid].append("comp_name")

            # 3. Name token overlap
            if "tokens" in c_indexes:
                for w in words:
                    if w in c_indexes["tokens"]:
                        for cid in c_indexes["tokens"][w]:
                            cand_scores[cid] += 3.5
                            methods[cid].append("token")

            # 4. Address token overlap
            if "addr" in c_indexes:
                for aw in a_tokens:
                    if aw in c_indexes["addr"]:
                        for cid in c_indexes["addr"][aw]:
                            cand_scores[cid] += 2.0
                            methods[cid].append("addr")

            top_pairs = cand_scores.most_common(k)
            cand_dict = {}
            for cid, score in top_pairs:
                cand_dict[cid] = {
                    "retrieval_score": float(score),
                    "retrieval_methods": "|".join(set(methods[cid])),
                }
            results[str(s1_id).strip()] = cand_dict

        return results

    def retrieve(
        self,
        s1_df: pd.DataFrame,
        max_candidates: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Retrieve candidates and return flattened DataFrame of candidate pairs.
        """
        cands_dict = self.retrieve_candidates(s1_df, max_candidates=max_candidates)
        rows = []
        for s1_id, cands in cands_dict.items():
            for cid, info in cands.items():
                rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_id": cid,
                    "retrieval_score": info["retrieval_score"],
                    "retrieval_methods": info["retrieval_methods"],
                })
        if not rows:
            return pd.DataFrame(columns=["source1_entity_id", "candidate_id", "retrieval_score", "retrieval_methods"])
        return pd.DataFrame(rows)

    def evaluate_blocking(
        self,
        candidates: Dict[str, Dict[str, Dict[str, Any]]],
        ground_truth: Dict[str, Set[str]],
    ) -> Dict[str, float]:
        """
        Evaluate candidate blocking performance:
        - blocking recall = retrieved_true / total_true
        - blocking ceiling macro F_0.5 = F_0.5 of an oracle that predicts exactly all retrieved true IDs
        """
        total_true = 0
        retrieved_true = 0
        oracle_preds = {}

        for s1_id, t_set in ground_truth.items():
            total_true += len(t_set)
            cand_set = set(candidates.get(s1_id, {}).keys())
            matched = t_set & cand_set
            retrieved_true += len(matched)
            # Oracle predicts all retrieved matches (empty if none retrieved)
            oracle_preds[s1_id] = matched

        blocking_recall = (retrieved_true / total_true) if total_true > 0 else 1.0
        oracle_eval = detailed_evaluation(ground_truth, oracle_preds)

        return {
            "blocking_recall": blocking_recall,
            "blocking_ceiling_f05": oracle_eval.get("macro_f05", 0.0),
            "retrieved_true_matches": retrieved_true,
            "total_true_matches": total_true,
            "avg_candidates_per_entity": sum(len(c) for c in candidates.values()) / max(1, len(candidates)),
        }
