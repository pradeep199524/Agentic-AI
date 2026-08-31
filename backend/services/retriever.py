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
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
    """Tokenizes alphanumeric words, isolating slash-separated terms."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def normalize_query_dynamic(query: str) -> str:
    """Normalizes query dynamically via DB expansions."""
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
        # 1. Fixed the bug here (no arguments passed to get_vector_store)
        self.embedding_model = get_embedding_model()
        self.collection = get_vector_store() 
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Builds BM25 sparse index directly from ChromaDB documents."""
        try:
            # 2. Updated to native ChromaDB syntax
            collection_data = self.collection.get(include=["documents", "metadatas"])
            self.corpus_texts = collection_data.get("documents", []) or []
            self.corpus_metadatas = collection_data.get("metadatas", []) or []

            if self.corpus_texts:
                tokenized_corpus = [tokenize_text(doc) for doc in self.corpus_texts]
                self.bm25 = BM25Okapi(tokenized_corpus)
                logging.info(f"BM25 index built with {len(self.corpus_texts)} corpus chunks.")
            else:
                self.bm25 = None
        except Exception as e:
            logging.error(f"Failed to build BM25 index: {e}")
            self.bm25 = None
            self.corpus_texts = []
            self.corpus_metadatas = []

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries native ChromaDB using sentence embeddings."""
        if not self.corpus_texts:
            return []

        # 3. Updated to native ChromaDB query syntax (removed LangChain)
        query_vector = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, len(self.corpus_texts)),
            include=["documents", "metadatas", "distances"]
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        parsed_results = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            score = round(1.0 / (1.0 + float(dist)), 4)
            parsed_results.append({
                "content": doc,
                "metadata": meta,
                "score": score,
                "strategy": "dense"
            })
        return parsed_results

    def sparse_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries the corpus using BM25 keyword matching."""
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
        """Combines dense and sparse results using Reciprocal Rank Fusion (RRF)."""
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
        """Reranks candidates using a cross-encoder."""
        if not candidates:
            return []

        pairs = [[query, doc["content"]] for doc in candidates]
        cross_scores = self.reranker.predict(pairs)

        for i, score in enumerate(cross_scores):
            sigmoid_score = 1.0 / (1.0 + math.exp(-float(score)))
            candidates[i]["rerank_score"] = round(sigmoid_score, 4)

        return sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)[:top_k]

    def retrieve(self, query: str, strategy: str = "dense", top_k: int = 5) -> List[Dict[str, Any]]:
        """Main retrieval entrypoint supporting multiple strategies."""
        norm_query = normalize_query_dynamic(query)
        # 4. Changed default routing to dense/vector to save CPU load
        if strategy in ("dense", "vector"):
            return self.dense_search(norm_query, top_k)
        if strategy == "sparse":
            return self.sparse_search(norm_query, top_k)
        if strategy == "hybrid":
            return self.hybrid_search(norm_query, top_k)
        if strategy == "hybrid_rerank":
            candidates = self.hybrid_search(norm_query, top_k=min(15, len(self.corpus_texts)))
            return self.rerank(norm_query, candidates, top_k)
        raise ValueError(f"Invalid strategy: {strategy}")