import logging
import asyncio
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.database import SessionLocal, engine
from backend.models import Document as DBDocument
from backend.services.retriever import HybridRetrievalEngine
from backend.services.embedding import process_and_embed_everything

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

# ============================================================================
# 1. SEARCH REQUEST SCHEMA & ENDPOINT (Existing)
# ============================================================================
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


# ============================================================================
# 2. FETCH AVAILABLE FILES FOR UI (New)
# ============================================================================
@router.get("/files", summary="Get all available files for chunking")
def get_available_files():
    db = SessionLocal()
    try:
        # CHANGED: Only fetch PDFs from the documents table to prevent duplicates!
        docs = db.query(DBDocument).filter(DBDocument.file_type == 'pdf').all()
        files = [{"filename": d.filename, "type": "PDF"} for d in docs]
        
        # Get dynamic CSV tables separately
        with engine.connect() as conn:
            tables_res = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'csv_data_%';"))
            for row in tables_res:
                files.append({"filename": row[0], "type": "CSV DATASET"})
                
        return {"files": files}
    except Exception as e:
        logger.error(f"Error fetching files: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch available files.")
    finally:
        db.close()


# ============================================================================
# 3. EMBED REQUEST WITH FILE SELECTION (New)
# ============================================================================
class EmbedRequest(BaseModel):
    strategy: str = Field("semantic", description="Chunking strategy: 'fixed', 'structured', or 'semantic'")
    filenames: List[str] = Field(default=[], description="List of specific filenames to embed. Leave empty for all.")

@router.post("/embed", summary="Trigger dynamic chunking pipeline")
async def trigger_embedding(request: EmbedRequest):
    global _engine_instance
    valid_strategies = ["fixed", "structured", "semantic"]
    
    if request.strategy.lower() not in valid_strategies:
        raise HTTPException(status_code=400, detail=f"Invalid strategy. Must be one of: {valid_strategies}")

    try:
        logger.info(f"Starting background embedding pipeline. Strategy: {request.strategy.upper()}")
        
        # Run CPU/GPU heavy embedding process in a worker thread so FastAPI remains non-blocking
        await asyncio.to_thread(
            process_and_embed_everything, 
            chunking_strategy=request.strategy.lower(),
            target_filenames=request.filenames
        )
        
        # Invalidate the singleton so the search engine re-binds to the newly built ChromaDB collection
        _engine_instance = None 
        logger.info("Hybrid Retrieval Engine singleton reset to reload new collection on next query.")
        
        target_msg = "all documents" if not request.filenames else f"{len(request.filenames)} selected files"
        return {
            "status": "success", 
            "message": f"Successfully processed {target_msg} using {request.strategy.upper()} chunking!"
        }
        
    except Exception as e:
        logger.error(f"Embedding pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")