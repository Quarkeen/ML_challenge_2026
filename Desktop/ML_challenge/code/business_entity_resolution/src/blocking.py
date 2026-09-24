import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import logging
import gc
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, vstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Assuming normalize_dataframe or similar is defined in normalize.py
try:
    from normalize import normalize_text, extract_postal_code
except ImportError:
    # Basic fallbacks in case normalize module differs
    def normalize_text(text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_postal_code(text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        # Very basic regex for US/France/India postal codes
        matches = re.findall(r'\b\d{5,6}\b', text)
        if matches:
            return matches[-1]
        return ""

logger = logging.getLogger(__name__)

def sparse_top_k(query_matrix, corpus_matrix, top_k=50):
    """Compute top-k cosine similarities between query and corpus using sparse ops.
    Process in batches to avoid memory issues."""
    batch_size = 5000
    results = []
    
    # We transpose corpus_matrix once
    corpus_T = corpus_matrix.T.tocsr()
    
    for start in range(0, query_matrix.shape[0], batch_size):
        end = min(start + batch_size, query_matrix.shape[0])
        batch = query_matrix[start:end]
        sim = batch.dot(corpus_T)  # sparse * sparse.T -> sparse
        
        for i in range(sim.shape[0]):
            row = sim.getrow(i)
            if row.nnz == 0:
                results.append([])
                continue
            data = row.data
            indices = row.indices
            if len(data) > top_k:
                # Use argpartition for top-k
                top_idx = np.argpartition(data, -top_k)[-top_k:]
                # Sort the top-k in descending order
                top_idx_sorted = top_idx[np.argsort(-data[top_idx])]
                results.append(list(zip(indices[top_idx_sorted], data[top_idx_sorted])))
            else:
                # Sort all in descending order
                sort_idx = np.argsort(-data)
                results.append(list(zip(indices[sort_idx], data[sort_idx])))
    return results


class CandidateRetriever:
    def __init__(self, config=None):
        self.config = config or {}
        self.max_candidates = self.config.get('max_candidates', 50)
        self.top_k_tfidf = self.config.get('top_k_tfidf', 30)
        self.indexes = {}  # country -> {index_name -> index}
        
    def _extract_tokens(self, text, min_len=3):
        if not text:
            return []
        return [t for t in text.split() if len(t) >= min_len]
        
    def _extract_numeric_tokens(self, text):
        if not text:
            return []
        return [t for t in re.findall(r'\b\d+\b', text) if len(t) >= 1]
    
    def _prepare_data(self, df):
        """Prepare dataframe for indexing by adding normalized columns."""
        df_prep = df.copy()
        
        # Ensure string type
        df_prep['business_name'] = df_prep['business_name'].fillna('').astype(str)
        df_prep['business_address'] = df_prep['business_address'].fillna('').astype(str)
        
        # Normalize
        df_prep['norm_name'] = df_prep['business_name'].apply(normalize_text)
        df_prep['norm_addr'] = df_prep['business_address'].apply(normalize_text)
        df_prep['norm_combined'] = df_prep['norm_name'] + ' ' + df_prep['norm_addr']
        
        # Extract features
        df_prep['postal_code'] = df_prep['business_address'].apply(extract_postal_code)
        
        return df_prep

    def build_indexes(self, s2_df, s3_df):
        """Build all blocking indexes from S2+S3 data.
        Partition by country for efficiency."""
        
        logger.info("Preparing data for indexing...")
        # Combine S2 and S3 for candidate pool
        s2_df['source'] = 'S2'
        s3_df['source'] = 'S3'
        
        corpus_df = pd.concat([s2_df, s3_df], ignore_index=True)
        corpus_df = self._prepare_data(corpus_df)
        
        countries = corpus_df['country'].fillna('UNKNOWN').unique()
        
        for country in countries:
            logger.info(f"Building indexes for country: {country}")
            country_df = corpus_df[corpus_df['country'].fillna('UNKNOWN') == country].reset_index(drop=True)
            
            if len(country_df) == 0:
                continue
                
            self.indexes[country] = {}
            
            # Store IDs map (idx -> entity_id)
            self.indexes[country]['id_map'] = country_df['entity_id'].values
            
            # 1. Exact name index
            exact_name_idx = defaultdict(list)
            for idx, name in enumerate(country_df['norm_name']):
                if name:
                    exact_name_idx[name].append(idx)
            self.indexes[country]['exact_name'] = exact_name_idx
            
            # 2. Name token inverted index
            name_token_idx = defaultdict(list)
            for idx, name in enumerate(country_df['norm_name']):
                tokens = self._extract_tokens(name, min_len=3)
                for t in tokens:
                    name_token_idx[t].append(idx)
            self.indexes[country]['name_tokens'] = name_token_idx
            
            # 3. Postal code blocking
            postal_idx = defaultdict(list)
            for idx, pc in enumerate(country_df['postal_code']):
                if pc:
                    postal_idx[pc].append(idx)
            self.indexes[country]['postal_code'] = postal_idx
            
            # 4. Numeric token blocking
            numeric_idx = defaultdict(list)
            for idx, combined in enumerate(country_df['norm_combined']):
                tokens = self._extract_numeric_tokens(combined)
                for t in tokens:
                    numeric_idx[t].append(idx)
            self.indexes[country]['numeric'] = numeric_idx
            
            # 5. TF-IDF Models (Name Char, Addr Char, Combined Char, Name Word)
            logger.info(f"Building TF-IDF models for {country}...")
            
            # Name char ngram
            name_char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_df=0.8, dtype=np.float32)
            name_char_matrix = name_char_tfidf.fit_transform(country_df['norm_name'])
            self.indexes[country]['name_char_tfidf_model'] = name_char_tfidf
            self.indexes[country]['name_char_tfidf_matrix'] = name_char_matrix
            
            # Addr char ngram
            addr_char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_df=0.8, dtype=np.float32)
            addr_char_matrix = addr_char_tfidf.fit_transform(country_df['norm_addr'])
            self.indexes[country]['addr_char_tfidf_model'] = addr_char_tfidf
            self.indexes[country]['addr_char_tfidf_matrix'] = addr_char_matrix
            
            # Combined char ngram
            comb_char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_df=0.8, dtype=np.float32)
            comb_char_matrix = comb_char_tfidf.fit_transform(country_df['norm_combined'])
            self.indexes[country]['comb_char_tfidf_model'] = comb_char_tfidf
            self.indexes[country]['comb_char_tfidf_matrix'] = comb_char_matrix
            
            # Name word
            name_word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2, max_df=0.8, dtype=np.float32)
            name_word_matrix = name_word_tfidf.fit_transform(country_df['norm_name'])
            self.indexes[country]['name_word_tfidf_model'] = name_word_tfidf
            self.indexes[country]['name_word_tfidf_matrix'] = name_word_matrix
            
            # Run GC
            gc.collect()
            
        logger.info("Finished building indexes.")
        
    def retrieve_candidates(self, s1_df, max_candidates=100):
        """Retrieve candidates for all S1 entities.
        Returns dict: {s1_id: {candidate_id: score_info}}"""
        
        logger.info("Preparing S1 data for retrieval...")
        s1_prep = self._prepare_data(s1_df)
        
        all_candidates = {}
        
        countries = s1_prep['country'].fillna('UNKNOWN').unique()
        
        for country in countries:
            if country not in self.indexes:
                logger.warning(f"No indexes found for country: {country}")
                continue
                
            logger.info(f"Retrieving candidates for country: {country}")
            country_idx = self.indexes[country]
            id_map = country_idx['id_map']
            
            country_s1 = s1_prep[s1_prep['country'].fillna('UNKNOWN') == country].reset_index()
            if len(country_s1) == 0:
                continue
                
            # Dictionary to store candidates for this country: idx_in_s1 -> dict(idx_in_corpus -> features)
            country_candidates = defaultdict(lambda: defaultdict(dict))
            
            # 1-4. Exact, Token, Postal, Numeric (Rule-based)
            for i, row in tqdm(country_s1.iterrows(), total=len(country_s1), desc=f"Rule-based {country}"):
                cands = country_candidates[i]
                
                # Exact name
                if row['norm_name'] in country_idx['exact_name']:
                    for match_idx in country_idx['exact_name'][row['norm_name']]:
                        cands[match_idx]['exact_name'] = 1.0
                        
                # Name tokens
                tokens = self._extract_tokens(row['norm_name'], min_len=3)
                token_counts = defaultdict(int)
                for t in tokens:
                    if t in country_idx['name_tokens']:
                        for match_idx in country_idx['name_tokens'][t]:
                            token_counts[match_idx] += 1
                
                for match_idx, count in token_counts.items():
                    cands[match_idx]['shared_tokens'] = count / max(1, len(tokens))
                    
                # Postal code
                if row['postal_code'] and row['postal_code'] in country_idx['postal_code']:
                    for match_idx in country_idx['postal_code'][row['postal_code']]:
                        cands[match_idx]['same_postal'] = 1.0
                        
                # Numeric tokens
                num_tokens = self._extract_numeric_tokens(row['norm_combined'])
                for t in num_tokens:
                    if t in country_idx['numeric']:
                        for match_idx in country_idx['numeric'][t]:
                            cands[match_idx]['shared_numeric'] = 1.0
                            
            # 5. TF-IDF Retrieval
            logger.info(f"TF-IDF retrieval for {country} (Name Char)...")
            q_name_char = country_idx['name_char_tfidf_model'].transform(country_s1['norm_name'])
            name_char_results = sparse_top_k(q_name_char, country_idx['name_char_tfidf_matrix'], self.top_k_tfidf)
            
            for i, results in enumerate(name_char_results):
                cands = country_candidates[i]
                for match_idx, score in results:
                    cands[match_idx]['name_char_sim'] = float(score)
                    
            logger.info(f"TF-IDF retrieval for {country} (Addr Char)...")
            q_addr_char = country_idx['addr_char_tfidf_model'].transform(country_s1['norm_addr'])
            addr_char_results = sparse_top_k(q_addr_char, country_idx['addr_char_tfidf_matrix'], self.top_k_tfidf)
            
            for i, results in enumerate(addr_char_results):
                cands = country_candidates[i]
                for match_idx, score in results:
                    cands[match_idx]['addr_char_sim'] = float(score)
                    
            logger.info(f"TF-IDF retrieval for {country} (Comb Char)...")
            q_comb_char = country_idx['comb_char_tfidf_model'].transform(country_s1['norm_combined'])
            comb_char_results = sparse_top_k(q_comb_char, country_idx['comb_char_tfidf_matrix'], self.top_k_tfidf)
            
            for i, results in enumerate(comb_char_results):
                cands = country_candidates[i]
                for match_idx, score in results:
                    cands[match_idx]['comb_char_sim'] = float(score)
                    
            logger.info(f"TF-IDF retrieval for {country} (Name Word)...")
            q_name_word = country_idx['name_word_tfidf_model'].transform(country_s1['norm_name'])
            name_word_results = sparse_top_k(q_name_word, country_idx['name_word_tfidf_matrix'], self.top_k_tfidf)
            
            for i, results in enumerate(name_word_results):
                cands = country_candidates[i]
                for match_idx, score in results:
                    cands[match_idx]['name_word_sim'] = float(score)
                    
            # Consolidate and select top candidates
            logger.info(f"Consolidating candidates for {country}...")
            
            for i, row in tqdm(country_s1.iterrows(), total=len(country_s1), desc=f"Consolidating {country}"):
                s1_id = row['entity_id']
                cands = country_candidates[i]
                
                # Scoring heuristic for candidate reduction if needed
                scored_cands = []
                for match_idx, features in cands.items():
                    # Simple ensemble score to pick top candidates if we exceed max_candidates
                    score = (
                        features.get('exact_name', 0) * 3.0 + 
                        features.get('name_char_sim', 0) * 2.0 +
                        features.get('comb_char_sim', 0) * 2.0 +
                        features.get('name_word_sim', 0) * 1.5 +
                        features.get('addr_char_sim', 0) * 1.0 +
                        features.get('shared_tokens', 0) * 1.0 +
                        features.get('same_postal', 0) * 0.5 +
                        features.get('shared_numeric', 0) * 0.5
                    )
                    scored_cands.append((match_idx, score, features))
                    
                # Sort by heuristic score descending
                scored_cands.sort(key=lambda x: x[1], reverse=True)
                
                # Take top max_candidates
                top_cands = scored_cands[:max_candidates]
                
                final_cands = {}
                for match_idx, _, features in top_cands:
                    cand_id = id_map[match_idx]
                    final_cands[cand_id] = features
                    
                all_candidates[s1_id] = final_cands
                
            gc.collect()
            
        logger.info(f"Retrieved candidates for {len(all_candidates)} S1 entities.")
        return all_candidates
    
    def retrieve_for_entity(self, s1_row, country):
        """Retrieve candidates for a single S1 entity. (Useful for inference/API)"""
        # (Implementation can be added similar to the batched version but for a single record)
        pass
    
    def compute_blocking_recall(self, candidates, ground_truth):
        """Compute blocking recall and ceiling F0.5.
        ground_truth: {s1_id: set([s2_id, s3_id, ...])}"""
        
        total_gt = 0
        total_found = 0
        total_pred = 0
        
        for s1_id, gt_set in ground_truth.items():
            if not gt_set:
                continue
                
            total_gt += len(gt_set)
            
            cands = set(candidates.get(s1_id, {}).keys())
            total_pred += len(cands)
            
            found = len(gt_set.intersection(cands))
            total_found += found
            
        recall = total_found / total_gt if total_gt > 0 else 0
        precision_ceiling = total_found / total_pred if total_pred > 0 else 0
        
        f05_ceiling = 0
        if precision_ceiling > 0 and recall > 0:
            f05_ceiling = (1.25 * precision_ceiling * recall) / ((0.25 * precision_ceiling) + recall)
            
        return {
            'recall': recall,
            'precision_ceiling': precision_ceiling,
            'f05_ceiling': f05_ceiling,
            'total_gt': total_gt,
            'total_found': total_found,
            'total_candidates': total_pred
        }
    
    def save_indexes(self, path):
        """Save indexes to disk."""
        logger.info(f"Saving indexes to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'indexes': self.indexes
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_indexes(self, path):
        """Load indexes from disk."""
        logger.info(f"Loading indexes from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.config = data['config']
            self.indexes = data['indexes']
            self.max_candidates = self.config.get('max_candidates', 50)
            self.top_k_tfidf = self.config.get('top_k_tfidf', 30)

    def fit(self, s2_df, s3_df):
        """Alias for build_indexes."""
        return self.build_indexes(s2_df, s3_df)

    def retrieve(self, s1_df, max_candidates=None):
        """Retrieve candidate pairs and return as a DataFrame."""
        top_k = max_candidates or self.max_candidates
        cands_dict = self.retrieve_candidates(s1_df, max_candidates=top_k)
        rows = []
        for s1_id, cands in cands_dict.items():
            for cid, info in cands.items():
                rows.append({
                    'source1_entity_id': s1_id,
                    'candidate_id': cid,
                    **{f'retrieval_{k}': v for k, v in info.items()}
                })
        if not rows:
            return pd.DataFrame(columns=['source1_entity_id', 'candidate_id'])
        return pd.DataFrame(rows)

