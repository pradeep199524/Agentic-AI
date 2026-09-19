import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

# Import from existing modules
from backend.database import SessionLocal, engine
from backend.models import Document, Page
from backend.services.pipeline import process_pdf, process_csv
from backend.routers.retrieval import router as retrieval_router
from backend.routers import chat
from backend.routers.agent import router as agent_router
import chromadb

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(os.getcwd(), "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_knowledge_base")
# ==========================================
# 1. SETUP & LOGGING WITH CORRELATION IDs
# ==========================================
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = getattr(record, 'correlation_id', 'SYSTEM')
        return True

logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s: [%(correlation_id)s] - %(message)s'))
handler.addFilter(CorrelationIdFilter())
logger.addHandler(handler)

# Ensure the data folder exists for uploads
os.makedirs("data", exist_ok=True)

# FastAPI initialization
app = FastAPI(
    title="AI Data Engineering Pipeline API",
    description="API to ingest documents, trigger background extraction, and fetch structured data.",
    version="1.0.0"
)
app.include_router(retrieval_router)
app.include_router(chat.router)
app.include_router(agent_router)

# CORS Configuration for Next.js Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Correlation ID Middleware
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    corr_id = str(uuid.uuid4())
    request.state.correlation_id = corr_id
    
    logger_adapter = logging.LoggerAdapter(logger, {'correlation_id': corr_id})
    request.state.logger = logger_adapter
    
    logger_adapter.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger_adapter.info(f"Completed request with status: {response.status_code}")
    
    response.headers["X-Correlation-ID"] = corr_id
    return response

# Database Session Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# NEW: BULLETPROOF FILE CLEANUP
# ==========================================
def clean_orphaned_files(directory: str, base_filename: str):
    """Scans the directory and deletes any file matching the filename or ending with _filename"""
    if not os.path.exists(directory):
        return
    for f in os.listdir(directory):
        if f == base_filename or f.endswith(f"_{base_filename}"):
            file_path = os.path.join(directory, f)
            try:
                os.remove(file_path)
                logger.info(f"Aggressively cleaned up old file: {f}")
            except Exception as e:
                logger.error(f"Failed to delete old file {f}: {e}")

# ==========================================
# 2. BACKGROUND TASK WRAPPERS WITH STATUS UPDATE
# ==========================================
def background_process_pdf(doc_id: int, file_path: str, corr_id: str):
    task_logger = logging.LoggerAdapter(logger, {'correlation_id': corr_id})
    task_logger.info(f"Starting background PDF job for doc_id={doc_id}, file={file_path}")
    db = SessionLocal()
    try:
        process_pdf(file_path, db, doc_id=doc_id)
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = "completed"
            db.commit()
        task_logger.info(f"Finished background PDF job for doc_id={doc_id}. Status: completed")
    except Exception as e:
        task_logger.error(f"Error in background PDF processing for doc_id={doc_id}: {e}")
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
    finally:
        db.close()

def background_process_csv(doc_id: int, file_path: str, filename: str, corr_id: str):
    task_logger = logging.LoggerAdapter(logger, {'correlation_id': corr_id})
    task_logger.info(f"Starting background CSV job for doc_id={doc_id}, file={file_path}")
    db = SessionLocal()
    try:
        process_csv(file_path, filename, db, engine, doc_id=doc_id)
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = "completed"
            db.commit()
        task_logger.info(f"Finished background CSV job for doc_id={doc_id}. Status: completed")
    except Exception as e:
        task_logger.error(f"Error in background CSV processing for doc_id={doc_id}: {e}")
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
    finally:
        db.close()

# ==========================================
# 3. REST ENDPOINTS
# ==========================================

@app.post("/ingest/pdf", status_code=202, summary="Upload a PDF for extraction")
async def ingest_pdf(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Accepts a PDF, creates a Document record in 'processing' status, and triggers background extraction."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")
    
    # 0. Delete ALL old database records with this filename
    existing_docs = db.query(Document).filter(Document.filename == file.filename).all()
    for doc in existing_docs:
        db.delete(doc)
    db.commit()

    # 1. Aggressively wipe out all physical copies of this file in the data folder
    clean_orphaned_files("data", file.filename)
    
    # 2. Create document record in database immediately
    new_doc = Document(filename=file.filename, file_type="pdf", status="processing")
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    file_path = os.path.join("data", f"{new_doc.id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    # 3. Dispatch background task with document ID
    background_tasks.add_task(
        background_process_pdf,
        new_doc.id,
        file_path,
        request.state.correlation_id
    )

    return {
        "document_id": new_doc.id,
        "status": "processing",
        "filename": file.filename,
        "message": "PDF accepted. Processing in the background."
    }


@app.post("/ingest/csv", status_code=202, summary="Upload a CSV for dynamic table creation")
async def ingest_csv(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Accepts a CSV, creates a Document record in 'processing' status, and triggers dynamic table creation."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSVs are allowed.")
    
    # 0. Delete ALL old database records with this filename
    existing_docs = db.query(Document).filter(Document.filename == file.filename).all()
    for doc in existing_docs:
        db.delete(doc)
    db.commit()

    # 1. Aggressively wipe out all physical copies of this file in the data folder
    clean_orphaned_files("data", file.filename)
    
    # 2. Create document record in database immediately
    new_doc = Document(filename=file.filename, file_type="csv", status="processing")
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    file_path = os.path.join("data", f"{new_doc.id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    # 3. Dispatch background task with document ID
    background_tasks.add_task(
        background_process_csv,
        new_doc.id,
        file_path,
        file.filename,
        request.state.correlation_id
    )

    return {
        "document_id": new_doc.id,
        "status": "processing",
        "filename": file.filename,
        "message": "CSV accepted. Processing in the background."
    }


@app.get("/documents/{doc_id}/status", summary="Poll document ingestion status")
def get_document_status(doc_id: int, db: Session = Depends(get_db)):
    """Returns the current processing status of a specific document ('processing', 'completed', 'failed')."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "status": doc.status
    }


@app.get("/documents", summary="Fetch document metadata")
def get_documents(db: Session = Depends(get_db)):
    """Returns a list of all ingested documents in the system."""
    docs = db.query(Document).order_by(Document.id.desc()).all()
    return {"documents": docs}


@app.get("/documents/{doc_id}/pages", summary="Fetch extracted pages for a document")
def get_document_pages(doc_id: int, db: Session = Depends(get_db)):
    """Returns the structured JSON content (paragraphs/tables) extracted from a specific PDF."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc.file_type != "pdf":
        raise HTTPException(status_code=400, detail="Document is not a PDF. Try fetching from dynamic CSV tables instead.")
        
    pages = db.query(Page).filter(Page.document_id == doc_id).order_by(Page.page_number.asc()).all()
    return {"document": doc.filename, "pages": pages}


@app.get("/csv/{table_name}", summary="Fetch records from dynamic CSV tables")
def get_csv_records(table_name: str, db: Session = Depends(get_db)):
    """Query the dynamic tables created by the CSV ingestion."""
    try:
        query = text(f"SELECT * FROM {table_name}")
        result = db.execute(query)
        
        columns = list(result.keys())
        data = [dict(zip(columns, row)) for row in result.fetchall()]
        
        return {"table": table_name, "columns": columns, "records": data}
    except Exception as e:
        logger.error(f"Failed to query table {table_name}: {e}")
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found or invalid.")


@app.delete("/documents/{doc_id}", summary="Delete document and related resources")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    """Deletes the document from PostgreSQL, drops CSV tables, deletes ChromaDB embeddings, and removes the local file."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # 1. Aggressively clean up local files
    clean_orphaned_files("data", doc.filename)
            
    # 2. Drop dynamic CSV table if applicable
    table_name = None
    if doc.file_type == "csv":
        base_name = doc.filename.rsplit('.', 1)[0].lower().replace(" ", "_").replace("-", "_")
        table_name = f"csv_data_{base_name}"
        try:
            db.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))
        except Exception as e:
            logger.error(f"Failed to drop SQL table {table_name}: {e}")
            
    # 3. Surgically delete embeddings from ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        
        # Target the filename used in the vector metadata
        targets = [doc.filename]
        if table_name:
            targets.append(table_name)
            
        for target in targets:
            collection.delete(where={"filename": target})
            logger.info(f"Purged ChromaDB embeddings for: {target}")
    except Exception as e:
        logger.warning(f"Could not purge ChromaDB embeddings for {doc.filename}: {e}")

    # 4. Delete Document record from PostgreSQL (Pages cascade automatically)
    db.delete(doc)
    db.commit()
    
    return {"message": f"Document '{doc.filename}' and all associated database/vector records successfully deleted."}