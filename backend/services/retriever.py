import os
import re
import math
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

from backend.services.embedding import get_vector_store, get_embedding_model
from backend.database import SessionLocal
from backend.models import GlossaryTerm

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- CONFIGURATION VIA ENVIRONMENT VARIABLES ---
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L6-v2")


def get_dynamic_expansions() -> Dict[str, str]:
    """Dynamically fetches alias and acronym expansions from PostgreSQL."""
    db = SessionLocal()
    try:
        terms = db.query(GlossaryTerm).all()
        return {t.term.lower(): t.expansion.lower() for t in terms}
    except Exception as e:
        logging.warning(f"Could not load dynamic glossary: {e}")
        return {}
    finally:
        db.close()


def tokenize_text(text: str) -> List[str]:
    """Tokenizes alphanumeric words, properly isolating slash-separated terms (e.g., 'AI/ML' -> ['ai', 'ml'])."""
    return re.findall(r'[a-zA-Z0-9]+', text.lower())


def normalize_query_dynamic(query: str) -> str:
    """Normalizes query dynamically via DB expansions and sub-token splitting."""
    tokens = tokenize_text(query)
    expansions = get_dynamic_expansions()
    
    expanded_tokens = []
    for token in tokens:
        if token in expansions:
            expanded_tokens.append(f"{token} {expansions[token]}")
        else:
            expanded_tokens.append(token)
            
    return " ".join(expanded_tokens) if expanded_tokens else query.strip()


class HybridRetrievalEngine:
    def __init__(self):
        self.embeddings = get_embedding_model()
        self.vector_store = get_vector_store(self.embeddings)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)
        self._build_bm25_index()

    def _build_bm25_index(self):
        try:
            collection_data = self.vector_store._collection.get(include=["documents", "metadatas"])
            self.corpus_texts = collection_data.get("documents", [])
            self.corpus_metadatas = collection_data.get("metadatas", [])
            
            if self.corpus_texts:
                tokenized_corpus = [tokenize_text(doc) for doc in self.corpus_texts]
                self.bm25 = BM25Okapi(tokenized_corpus)
                logging.info(f"BM25 index built with {len(self.corpus_texts)} corpus chunks.")
            else:
                self.bm25 = None
        except Exception as e:
            logging.error(f"Failed to build BM25 index: {e}")
            self.bm25 = None

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        return [
            {
                "content": d.page_content,
                "metadata": d.metadata,
                "score": round(1.0 / (1.0 + float(s)), 4),
                "strategy": "dense"
            }
            for d, s in results
        ]

    def sparse_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.corpus_texts:
            return []
        tokenized_query = tokenize_text(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return [
            {
                "content": self.corpus_texts[i],
                "metadata": self.corpus_metadatas[i],
                "score": round(float(scores[i]) / max_score, 4),
                "strategy": "sparse"
            }
            for i in top_indices if scores[i] > 0
        ]

    def hybrid_search(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
        dense = self.dense_search(query, top_k=top_k * 2)
        sparse = self.sparse_search(query, top_k=top_k * 2)
        
        rrf_scores, doc_lookup = {}, {}
        
        for rank, item in enumerate(dense):
            text = item["content"]
            rrf_scores[text] = rrf_scores.get(text, 0.0) + (1.0 / (rrf_k + rank + 1))
            doc_lookup[text] = item
            
        for rank, item in enumerate(sparse):
            text = item["content"]
            rrf_scores[text] = rrf_scores.get(text, 0.0) + (1.0 / (rrf_k + rank + 1))
            if text not in doc_lookup:
                doc_lookup[text] = item
                
        sorted_texts = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)[:top_k]
        return [
            {
                "content": t,
                "metadata": doc_lookup[t]["metadata"],
                "score": round(rrf_scores[t], 4),
                "strategy": "hybrid"
            }
            for t in sorted_texts
        ]

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        
        pairs = [[query, doc["content"]] for doc in candidates]
        cross_scores = self.reranker.predict(pairs)
        
        for i, score in enumerate(cross_scores):
            # Sigmoid normalization maps logits to a 0.0 - 1.0 probability range
            sigmoid_score = 1.0 / (1.0 + math.exp(-float(score)))
            candidates[i]["rerank_score"] = round(sigmoid_score, 4)
            
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    def retrieve(self, query: str, strategy: str = "hybrid_rerank", top_k: int = 5) -> List[Dict[str, Any]]:
        norm_query = normalize_query_dynamic(query)
        if strategy == "dense":
            return self.dense_search(norm_query, top_k)
        if strategy == "sparse":
            return self.sparse_search(norm_query, top_k)
        if strategy == "hybrid":
            return self.hybrid_search(norm_query, top_k)
        if strategy == "hybrid_rerank":
            candidates = self.hybrid_search(norm_query, top_k=min(15, len(self.corpus_texts)))
            return self.rerank(norm_query, candidates, top_k)
        raise ValueError(f"Invalid strategy: {strategy}")