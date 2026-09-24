import pandas as pd
import numpy as np
import rapidfuzz
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import math
import re
from collections import Counter
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from normalize import *
except ImportError:
    pass

class FeatureGenerator:
    def __init__(self):
        self.feature_names = [
            'name_exact_match', 'name_exact_match_no_suffix', 'name_levenshtein', 
            'name_partial_ratio', 'name_token_sort', 'name_token_set', 'name_jaro_winkler', 
            'name_len_diff', 'name_len_ratio', 'name_shared_tokens', 'name_jaccard_tokens', 
            'name_prefix_match', 'name_contains', 'name_levenshtein_no_suffix', 
            'name_token_sort_no_suffix', 'name_char_ngram_sim', 'name_rare_token_overlap',
            'addr_exact_match', 'addr_levenshtein', 'addr_token_sort', 'addr_token_set', 
            'addr_shared_tokens', 'addr_jaccard_tokens', 'addr_house_num_match', 
            'addr_house_num_contradiction', 'addr_postal_match', 'addr_postal_prefix3', 
            'addr_city_match', 'addr_city_sim', 'addr_region_match', 'addr_shared_numerics', 
            'addr_numeric_jaccard', 's1_addr_missing', 'cand_addr_missing', 'both_addr_missing', 
            'either_addr_missing', 's1_has_postal', 'cand_has_postal', 'country_match', 
            'country_is_US', 'country_is_India', 'country_is_France', 'source_is_S2', 
            'source_is_S3', 'retrieval_name_tfidf_score', 'retrieval_addr_tfidf_score', 
            'retrieval_combined_tfidf_score', 'retrieval_token_overlap_score', 
            'retrieval_method_count', 'retrieval_rank', 'name_addr_product', 'max_name_sim', 
            'min_name_sim', 'name_sim_std'
        ]
        
    def _get_trigrams(self, text):
        if not text or len(text) < 3:
            return set([text]) if text else set()
        return set([text[i:i+3] for i in range(len(text)-2)])
        
    def _extract_numerics(self, text):
        if pd.isna(text) or not text:
            return set()
        return set(re.findall(r'\d+', text))
        
    def _extract_postal(self, text):
        if pd.isna(text) or not text:
            return None
        match = re.search(r'\b\d{5}(?:-\d{4})?\b', text)
        if match:
            return match.group(0)
        match = re.search(r'\b\d{6}\b', text)
        if match:
            return match.group(0)
        return None
        
    def compute_pair_features(self, s1_row, cand_row, retrieval_info=None):
        """Compute features for a single (S1, candidate) pair.
        Returns dict of feature_name -> value."""
        features = {}
        
        val_n1 = s1_row.get('business_name', '')
        val_n2 = cand_row.get('business_name', '')
        n1 = "" if (pd.isna(val_n1) or str(val_n1).strip().lower() in ('', 'nan', 'none')) else str(val_n1).strip().lower()
        n2 = "" if (pd.isna(val_n2) or str(val_n2).strip().lower() in ('', 'nan', 'none')) else str(val_n2).strip().lower()
        
        def strip_suffix(n):
            suffixes = r'\b(inc|llc|ltd|co|corp|corporation|company|limited|incorporated)\b\.?'
            return re.sub(suffixes, '', n).strip()
            
        n1_ns = strip_suffix(n1)
        n2_ns = strip_suffix(n2)
        
        features['name_exact_match'] = bool(n1 and n2 and n1 == n2)
        features['name_exact_match_no_suffix'] = bool(n1_ns and n2_ns and n1_ns == n2_ns)
        features['name_levenshtein'] = fuzz.ratio(n1, n2) if n1 and n2 else 0.0
        features['name_partial_ratio'] = fuzz.partial_ratio(n1, n2) if n1 and n2 else 0.0
        features['name_token_sort'] = fuzz.token_sort_ratio(n1, n2) if n1 and n2 else 0.0
        features['name_token_set'] = fuzz.token_set_ratio(n1, n2) if n1 and n2 else 0.0
        features['name_jaro_winkler'] = JaroWinkler.similarity(n1, n2) if n1 and n2 else 0.0
        
        features['name_len_diff'] = abs(len(n1) - len(n2))
        features['name_len_ratio'] = min(len(n1), len(n2)) / max(len(n1), len(n2)) if max(len(n1), len(n2)) > 0 else 0.0
        
        tok1, tok2 = set(n1.split()), set(n2.split())
        features['name_shared_tokens'] = len(tok1.intersection(tok2))
        union_len = len(tok1.union(tok2))
        features['name_jaccard_tokens'] = len(tok1.intersection(tok2)) / union_len if union_len > 0 else 0.0
        
        pref = 0
        for c1, c2 in zip(n1, n2):
            if c1 == c2: pref += 1
            else: break
        features['name_prefix_match'] = pref
        
        features['name_contains'] = bool(n1 and n2 and (n1 in n2 or n2 in n1))
        features['name_levenshtein_no_suffix'] = fuzz.ratio(n1_ns, n2_ns) if n1_ns and n2_ns else 0.0
        features['name_token_sort_no_suffix'] = fuzz.token_sort_ratio(n1_ns, n2_ns) if n1_ns and n2_ns else 0.0
        
        tri1, tri2 = self._get_trigrams(n1), self._get_trigrams(n2)
        tri_uni = len(tri1.union(tri2))
        features['name_char_ngram_sim'] = len(tri1.intersection(tri2)) / tri_uni if tri_uni > 0 else 0.0
        
        features['name_rare_token_overlap'] = features['name_shared_tokens'] 
        
        val_a1 = s1_row.get('business_address', '')
        val_a2 = cand_row.get('business_address', '')
        a1 = "" if (pd.isna(val_a1) or str(val_a1).strip().lower() in ('', 'nan', 'none')) else str(val_a1).strip().lower()
        a2 = "" if (pd.isna(val_a2) or str(val_a2).strip().lower() in ('', 'nan', 'none')) else str(val_a2).strip().lower()

        
        features['s1_addr_missing'] = not bool(a1)
        features['cand_addr_missing'] = not bool(a2)
        features['both_addr_missing'] = features['s1_addr_missing'] and features['cand_addr_missing']
        features['either_addr_missing'] = features['s1_addr_missing'] or features['cand_addr_missing']
        
        p1 = self._extract_postal(a1)
        p2 = self._extract_postal(a2)
        features['s1_has_postal'] = bool(p1)
        features['cand_has_postal'] = bool(p2)
        
        if features['either_addr_missing']:
            features['addr_exact_match'] = np.nan
            features['addr_levenshtein'] = np.nan
            features['addr_token_sort'] = np.nan
            features['addr_token_set'] = np.nan
            features['addr_shared_tokens'] = np.nan
            features['addr_jaccard_tokens'] = np.nan
            features['addr_house_num_match'] = np.nan
            features['addr_house_num_contradiction'] = np.nan
            features['addr_postal_match'] = np.nan
            features['addr_postal_prefix3'] = np.nan
            features['addr_city_match'] = np.nan
            features['addr_city_sim'] = np.nan
            features['addr_region_match'] = np.nan
            features['addr_shared_numerics'] = np.nan
            features['addr_numeric_jaccard'] = np.nan
        else:
            features['addr_exact_match'] = bool(a1 == a2)
            features['addr_levenshtein'] = fuzz.ratio(a1, a2)
            features['addr_token_sort'] = fuzz.token_sort_ratio(a1, a2)
            features['addr_token_set'] = fuzz.token_set_ratio(a1, a2)
            
            atok1, atok2 = set(a1.split()), set(a2.split())
            features['addr_shared_tokens'] = len(atok1.intersection(atok2))
            a_union = len(atok1.union(atok2))
            features['addr_jaccard_tokens'] = len(atok1.intersection(atok2)) / a_union if a_union > 0 else 0.0
            
            num1 = self._extract_numerics(a1)
            num2 = self._extract_numerics(a2)
            
            hn1 = next(iter(re.findall(r'^\D*(\d+)', a1)), None)
            hn2 = next(iter(re.findall(r'^\D*(\d+)', a2)), None)
            if hn1 and hn2:
                features['addr_house_num_match'] = bool(hn1 == hn2)
                features['addr_house_num_contradiction'] = bool(hn1 != hn2)
            else:
                features['addr_house_num_match'] = np.nan
                features['addr_house_num_contradiction'] = False
                
            if p1 and p2:
                features['addr_postal_match'] = bool(p1 == p2)
                features['addr_postal_prefix3'] = bool(p1[:3] == p2[:3])
            else:
                features['addr_postal_match'] = np.nan
                features['addr_postal_prefix3'] = np.nan
                
            features['addr_city_match'] = np.nan 
            features['addr_city_sim'] = np.nan
            features['addr_region_match'] = np.nan
            
            features['addr_shared_numerics'] = len(num1.intersection(num2))
            num_union = len(num1.union(num2))
            features['addr_numeric_jaccard'] = len(num1.intersection(num2)) / num_union if num_union > 0 else 0.0
            
        c1 = str(s1_row.get('country', '')).lower()
        c2 = str(cand_row.get('country', '')).lower()
        features['country_match'] = bool(c1 == c2)
        features['country_is_US'] = bool(c1 == 'us' or c2 == 'us')
        features['country_is_India'] = bool(c1 == 'india' or c2 == 'india')
        features['country_is_France'] = bool(c1 == 'france' or c2 == 'france')
        
        cid = str(cand_row.get('entity_id', ''))
        features['source_is_S2'] = cid.startswith('S2')
        features['source_is_S3'] = cid.startswith('S3')
        
        if retrieval_info:
            features['retrieval_name_tfidf_score'] = retrieval_info.get('name_tfidf_score', 0.0)
            features['retrieval_addr_tfidf_score'] = retrieval_info.get('addr_tfidf_score', 0.0)
            features['retrieval_combined_tfidf_score'] = retrieval_info.get('combined_tfidf_score', 0.0)
            features['retrieval_token_overlap_score'] = retrieval_info.get('token_overlap_score', 0.0)
            features['retrieval_method_count'] = retrieval_info.get('method_count', 0.0)
            features['retrieval_rank'] = retrieval_info.get('rank', 0.0)
        else:
            features['retrieval_name_tfidf_score'] = 0.0
            features['retrieval_addr_tfidf_score'] = 0.0
            features['retrieval_combined_tfidf_score'] = 0.0
            features['retrieval_token_overlap_score'] = 0.0
            features['retrieval_method_count'] = 0.0
            features['retrieval_rank'] = 0.0
            
        nl = features['name_levenshtein']
        al = features['addr_levenshtein'] if not pd.isna(features['addr_levenshtein']) else 0.0
        features['name_addr_product'] = (nl * al) / 10000.0
        
        name_sims = [
            features['name_levenshtein'],
            features['name_partial_ratio'],
            features['name_token_sort'],
            features['name_token_set'],
            features['name_jaro_winkler'] * 100
        ]
        features['max_name_sim'] = max(name_sims)
        features['min_name_sim'] = min(name_sims)
        features['name_sim_std'] = float(np.std(name_sims))
        
        return features

    def compute_batch_features(self, s1_df, cand_df, pairs, retrieval_info=None):
        """Compute features for a batch of pairs.
        pairs: list of (s1_id, cand_id) tuples
        Returns pd.DataFrame with feature columns."""
        s1_records = s1_df.set_index('entity_id').to_dict('index')
        cand_records = cand_df.set_index('entity_id').to_dict('index')
        
        feature_rows = []
        for i, (s1_id, cand_id) in enumerate(pairs):
            s1_row = s1_records.get(s1_id, {})
            cand_row = cand_records.get(cand_id, {})
            
            s1_row['entity_id'] = s1_id
            cand_row['entity_id'] = cand_id
            
            r_info = retrieval_info[i] if retrieval_info else None
            
            feats = self.compute_pair_features(s1_row, cand_row, r_info)
            feature_rows.append(feats)
            
        return pd.DataFrame(feature_rows)

    def generate_features(self, pairs_df, s1_df, cand_df, retrieval_info=None):
        """Generate features from pairs DataFrame or list."""
        if isinstance(pairs_df, pd.DataFrame):
            c1 = 'source1_entity_id' if 'source1_entity_id' in pairs_df.columns else pairs_df.columns[0]
            c2 = 'candidate_id' if 'candidate_id' in pairs_df.columns else ('candidate_entity_id' if 'candidate_entity_id' in pairs_df.columns else pairs_df.columns[1])
            pairs = list(zip(pairs_df[c1].values, pairs_df[c2].values))
        else:
            pairs = pairs_df
        return self.compute_batch_features(s1_df, cand_df, pairs, retrieval_info=retrieval_info)

