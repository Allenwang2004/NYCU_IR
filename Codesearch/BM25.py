# BM25 implementation
import math
import json
from collections import defaultdict
import numpy as np

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.documents = []
        self.term_freqs = []
        self.doc_freqs = defaultdict(int)
        self.num_docs = 0
        self.vocabulary = set()
        self.doc_lengths = []
        self.avg_doc_len = 0
        self.k1 = k1
        self.b = b
        self._precomputed_idf = {}
        self._precomputed_norms = None

    def add_document(self, document_tokens):
        # 如果是字串格式的 JSON，先解析成列表
        if isinstance(document_tokens, str):
            try:
                terms = json.loads(document_tokens)
            except json.JSONDecodeError:
                # 如果不是 JSON 格式，就當作普通字串處理
                terms = document_tokens.split()
        else:
            # 如果已經是列表，直接使用
            terms = document_tokens
            
        self.documents.append(terms)
        self.doc_lengths.append(len(terms))
        self.num_docs += 1
        term_count = defaultdict(int)

        for term in terms:
            term_count[term] += 1
            self.vocabulary.add(term)

        self.term_freqs.append(term_count)

        for term in term_count.keys():
            self.doc_freqs[term] += 1
    
    def _precompute_idf(self):
        """預計算所有詞彙的 IDF 值"""
        self.avg_doc_len = sum(self.doc_lengths) / self.num_docs
        for term in self.vocabulary:
            df = self.doc_freqs[term]
            self._precomputed_idf[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5))
    
    def _precompute_normalization_factors(self):
        """預計算正規化因子"""
        self._precomputed_norms = np.array([
            self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            for doc_len in self.doc_lengths
        ])
    
    def compute_similarity_batch(self, query_tokens):
        if not self._precomputed_idf:
            self._precompute_idf()
        if self._precomputed_norms is None:
            self._precompute_normalization_factors()
            
        # 如果是字串格式的 JSON，先解析成列表
        if isinstance(query_tokens, str):
            try:
                query_terms = json.loads(query_tokens)
            except json.JSONDecodeError:
                # 如果不是 JSON 格式，就當作普通字串處理
                query_terms = query_tokens.split()
        else:
            # 如果已經是列表，直接使用
            query_terms = query_tokens
            
        similarities = np.zeros(self.num_docs)
        
        for doc_idx in range(self.num_docs):
            score = 0.0
            term_freq_doc = self.term_freqs[doc_idx]
            norm_factor = self._precomputed_norms[doc_idx]
            
            for term in query_terms:
                if term in self.vocabulary:
                    tf = term_freq_doc.get(term, 0)
                    if tf > 0:  # 只計算有出現的詞彙
                        idf = self._precomputed_idf[term]
                        normalized_tf = (tf * (self.k1 + 1)) / (tf + norm_factor)
                        score += idf * normalized_tf
            
            similarities[doc_idx] = score

        return similarities

    def get_top_k_similar_documents(self, query_tokens, k):
        similarities = self.compute_similarity_batch(query_tokens)

        if k >= len(similarities):
            top_k_indices = np.argsort(similarities)[::-1]      
        else:
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        return top_k_indices.tolist()
            