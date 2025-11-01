import math
import numpy as np
import json
from collections import defaultdict

class TFIDF:
    '''
    documents: List of documents (strings)
    term_freqs: List of term frequency dictionaries for each document ex: [{'term1': 2, 'term2': 1}, ...]
    doc_freqs: Document frequency dictionary for terms ex: {'term1': 3, 'term2': 5}
    num_docs: Total number of documents
    vocabulary: Set of unique terms across all documents
    idf: Inverse Document Frequency dictionary for terms
    tfidf_matrix: 2D numpy array storing TF-IDF scores for documents
    vocab_to_idx: Mapping from term to its index in the vocabulary
    '''
    def __init__(self):
        self.documents = []
        self.term_freqs = []
        self.doc_freqs = defaultdict(int)
        self.num_docs = 0
        self.vocabulary = set()
        self.idf = defaultdict(float)
        self.tfidf_matrix = None
        self.vocab_to_idx = defaultdict(int)

    def add_document(self, document_tokens):
        # 如果是字串格式的 JSON，先解析成列表
        if isinstance(document_tokens, str):
            try:
                tokens = json.loads(document_tokens)
            except json.JSONDecodeError:
                # 如果不是 JSON 格式，就當作普通字串處理
                tokens = document_tokens.split()
        else:
            # 如果已經是列表，直接使用
            tokens = document_tokens
            
        self.documents.append(tokens)
        self.num_docs += 1
        term_count = defaultdict(int)

        for term in tokens:
            term_count[term] += 1
            self.vocabulary.add(term)

        self.term_freqs.append(term_count)

        for term in term_count.keys():
            self.doc_freqs[term] += 1
    
    def _build_vocab_index(self):
        self.vocabulary = list(self.vocabulary)
        self.vocab_to_idx = {term: idx for idx, term in enumerate(self.vocabulary)}
        
    def compute_tfidf(self):
        if self.tfidf_matrix is not None:
            return self.tfidf_matrix
            
        self._build_vocab_index()
        vocab_size = len(self.vocabulary)
        
        for term in self.vocabulary:
            self.idf[term] = math.log((1 + (self.num_docs/self.doc_freqs[term])), 2)
        
        self.tfidf_matrix = np.zeros((self.num_docs, vocab_size))
        
        for doc_idx, term_count in enumerate(self.term_freqs):
            for term, count in term_count.items():
                if term in self.vocab_to_idx:
                    tf = 1 + math.log(count, 2)
                    tfidf_val = tf * self.idf[term]
                    self.tfidf_matrix[doc_idx][self.vocab_to_idx[term]] = tfidf_val
        
        return self.tfidf_matrix

    def get_query_vector(self, query_tokens):
        
        if self.tfidf_matrix is None:
            self.compute_tfidf()
        
        # 如果是字串格式的 JSON，先解析成列表
        if isinstance(query_tokens, str):
            try:
                tokens = json.loads(query_tokens)
            except json.JSONDecodeError:
                # 如果不是 JSON 格式，就當作普通字串處理
                tokens = query_tokens.split()
        else:
            # 如果已經是列表，直接使用
            tokens = query_tokens
        
        query_term_count = defaultdict(int)
        for term in tokens:
            query_term_count[term] += 1

        query_vector = np.zeros(len(self.vocabulary))
        
        for term, count in query_term_count.items():
            if term in self.vocab_to_idx:
                tf = 1 + math.log(count, 2)
                idf = self.idf.get(term, 0.0)
                query_vector[self.vocab_to_idx[term]] = tf * idf
                
        return query_vector
    
    def compute_similarity_batch(self, query_vector):
        
        dot_products = np.dot(self.tfidf_matrix, query_vector)
        
        # Document Length Normalization
        doc_norms = np.linalg.norm(self.tfidf_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        valid_mask = (doc_norms > 0) & (query_norm > 0)
        similarities = np.zeros(len(doc_norms))
        similarities[valid_mask] = dot_products[valid_mask] / (doc_norms[valid_mask] * query_norm)
        
        return similarities
    
    def get_top_k_similar_documents(self, query_tokens, k):
        
        query_vector = self.get_query_vector(query_tokens)
        similarities = self.compute_similarity_batch(query_vector)
        
        if k >= len(similarities):
            top_k_indices = np.argsort(similarities)[::-1]
        else:
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        
        return top_k_indices.tolist()
