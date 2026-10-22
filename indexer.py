"""
Hybrid Indexing Engine for Aviation Maintenance Records
Combines BM25 keyword matching and TF-IDF semantic vector similarity
using Reciprocal Rank Fusion (RRF) and an empirical refusal gate.
"""

import os
import csv
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from normalizer import AviationNormalizer

STOPWORDS = {
    "in", "the", "on", "at", "to", "for", "a", "an", "and", "or",
    "is", "are", "was", "were", "with", "by", "of", "it", "this",
    "that", "from", "as", "be", "been", "has", "have", "had"
}

class HybridIndex:
    def __init__(self, dataset_csv: str, abbreviation_csv: str = None):
        self.dataset_csv = dataset_csv
        self.abbreviation_csv = abbreviation_csv
        self.normalizer = AviationNormalizer(abbreviation_csv)
        
        self.records: List[Dict] = []
        self.record_by_id: Dict[str, Dict] = {}
        self.corpus_tokens: List[List[str]] = []
        self.corpus_texts: List[str] = []
        
        self.bm25: Optional[BM25Okapi] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        
        self._build_index()

    def _build_index(self):
        if not os.path.exists(self.dataset_csv):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_csv}")

        with open(self.dataset_csv, mode="r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ident = row.get("IDENT", "").strip()
                problem = row.get("PROBLEM", "").strip()
                action = row.get("ACTION", "").strip()
                
                norm_p = self.normalizer.normalize(problem)
                tokens = self.normalizer.tokenize(problem)
                
                record = {
                    "ident": ident,
                    "problem": problem,
                    "action": action,
                    "norm_problem": norm_p,
                    "tokens": tokens
                }
                
                self.records.append(record)
                self.record_by_id[ident] = record
                self.corpus_tokens.append(tokens)
                self.corpus_texts.append(norm_p)

        # Build BM25 keyword index
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
        # Build TF-IDF vector index (word & bigrams with sublinear scaling)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = self.tfidf.fit_transform(self.corpus_texts)

    def get_record(self, ident: str) -> Optional[Dict]:
        """Tool 2: get_record by IDENT"""
        clean_id = str(ident).strip().replace("IDENT:", "").strip()
        return self.record_by_id.get(clean_id)

    def search(self, query: str, top_k: int = 5) -> Tuple[List[Dict], bool, Dict]:
        """
        Tool 1: search_records
        Performs hybrid retrieval:
        1. BM25 on normalized query tokens
        2. TF-IDF Cosine Similarity on expanded query
        3. Reciprocal Rank Fusion (RRF) with k=60
        4. Verifies query coverage and calibrated refusal gate
        """
        norm_query = self.normalizer.normalize(query)
        q_tokens = [t for t in self.normalizer.tokenize(query) if t not in STOPWORDS]
        
        if not q_tokens or not self.records:
            return [], False, {
                "reason": "Query contained no informative tokens.",
                "confidence": 0.0,
                "coverage": 0.0,
                "norm_query": norm_query
            }

        # 1. BM25 Scores
        bm_scores = np.array(self.bm25.get_scores(q_tokens))
        
        # 2. Dense/TF-IDF Cosine Similarity
        q_vec = self.tfidf.transform([norm_query])
        sim_scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        # 3. Reciprocal Rank Fusion
        bm_order = np.argsort(-bm_scores)[:100]
        sim_order = np.argsort(-sim_scores)[:100]
        
        rrf = np.zeros(len(self.records))
        for rank, idx in enumerate(bm_order, start=1):
            if bm_scores[idx] > 0:
                rrf[idx] += 1.0 / (60.0 + rank)
                
        for rank, idx in enumerate(sim_order, start=1):
            if sim_scores[idx] > 0.05:
                rrf[idx] += 1.0 / (60.0 + rank)

        # Get top candidate indices by composite RRF score
        top_indices = np.argsort(-rrf)[:top_k]
        
        results = []
        for idx in top_indices:
            rec = self.records[idx]
            rec_tokens = set(rec["tokens"])
            matched_tokens = [t for t in q_tokens if t in rec_tokens]
            coverage = len(matched_tokens) / len(q_tokens) if q_tokens else 0.0
            cos_sim = float(sim_scores[idx])
            bm_score = float(bm_scores[idx])
            rrf_score = float(rrf[idx])
            confidence = 0.5 * cos_sim + 0.5 * coverage
            
            results.append({
                "ident": rec["ident"],
                "problem": rec["problem"],
                "action": rec["action"],
                "norm_problem": rec["norm_problem"],
                "bm25_score": round(bm_score, 2),
                "cos_sim": round(cos_sim, 3),
                "rrf_score": round(rrf_score, 4),
                "coverage": round(coverage, 2),
                "confidence": round(confidence, 3),
                "matched_tokens": matched_tokens
            })

        # Refusal Evaluation on best candidate
        best = results[0] if results else None
        if not best:
            return [], False, {"reason": "No matches found.", "confidence": 0.0}

        # Refusal criteria:
        # 1. Confidence >= 0.38 OR
        # 2. Token coverage >= 0.50 (at least half of search keywords exist in log) OR
        # 3. Cosine similarity >= 0.52
        is_confident = (
            (best["confidence"] >= 0.38) or
            (best["coverage"] >= 0.50 and best["bm25_score"] > 2.0) or
            (best["cos_sim"] >= 0.52)
        )
        
        metrics = {
            "query": query,
            "normalized_query": norm_query,
            "query_tokens": q_tokens,
            "top_confidence": best["confidence"],
            "top_coverage": best["coverage"],
            "top_cos_sim": best["cos_sim"],
            "top_rrf": best["rrf_score"],
            "is_confident": is_confident,
            "reason": "Sufficient evidence identified." if is_confident else "Insufficient evidence: query keywords do not match historical records with sufficient coverage or semantic similarity."
        }
        
        return results, is_confident, metrics


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "dataset", "Aircraft_Annotation_DataFile.csv")
    abbr_path = os.path.join(BASE_DIR, "dataset", "Aviation_Abbreviation_Dataset.csv")
    index = HybridIndex(data_path, abbr_path)
    
    print("Testing positive query:")
    res, conf, met = index.search("engine choke lose power on departure", top_k=3)
    print(f"Confident: {conf} | Metrics: {met}")
    for r in res:
        print(f"  [{r['ident']}] {r['problem']} -> {r['action']} (Conf: {r['confidence']})")
        
    print("\nTesting out-of-domain query:")
    res, conf, met = index.search("Boeing 777 toilet flush sensor broke in cabin", top_k=3)
    print(f"Confident: {conf} | Reason: {met['reason']}")
