import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.retriever import HybridRetrievalEngine

logger = logging.getLogger("api_logger")
router = APIRouter(tags=["Retrieval Pipeline"])

# Lazy-loaded singleton instance of the retrieval engine
_engine_instance: Optional[HybridRetrievalEngine] = None

def get_engine() -> HybridRetrievalEngine:
    global _engine_instance
    if _engine_instance is None:
        try:
            logger.info("Initializing Hybrid Retrieval Engine instance...")
            _engine_instance = HybridRetrievalEngine()
        except Exception as e:
            logger.error(f"Failed to initialize HybridRetrievalEngine: {e}")
            raise HTTPException(
                status_code=500,
                detail="Retrieval engine could not be initialized. Ensure ChromaDB has been built via embedding.py."
            )
    return _engine_instance

class SearchRequest(BaseModel):
    query: str = Field(..., description="The natural language query to retrieve chunks for")
    strategy: str = Field("hybrid_rerank", description="Retrieval strategy: 'dense', 'sparse', 'hybrid', 'hybrid_rerank'")
    top_k: int = Field(5, ge=1, le=20, description="Number of top chunks to return")

@router.post("/search", summary="Search documents using a specific strategy")
def search_documents(request: SearchRequest):
    engine = get_engine()
    try:
        results = engine.retrieve(
            query=request.query,
            strategy=request.strategy,
            top_k=request.top_k
        )
        return {
            "query": request.query,
            "strategy": request.strategy,
            "count": len(results),
            "results": results
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during search execution: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")