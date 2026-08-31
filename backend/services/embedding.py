import os
import re
import uuid
import json
import logging
import chromadb
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Import both SessionLocal and engine from your database setup
from backend.database import SessionLocal, engine
from backend.models import Document as DBDocument, Page

# Initialize Environment and Logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Configuration Variables
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(os.getcwd(), "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_knowledge_base")
BATCH_SIZE = 500  # Safe batch size to prevent Out-Of-Memory (OOM) errors


def get_embedding_model() -> SentenceTransformer:
    """Returns the shared SentenceTransformer model instance."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_vector_store():
    """Returns the native ChromaDB collection instance."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def robust_overlapping_chunker(text: str, max_words: int = 600, overlap_words: int = 150) -> list:
    """
    Header-Aware Sliding Window Chunker:
    1. Identifies headers (short lines without ending punctuation) and emphasizes them.
    2. Groups text into max 600-word chunks.
    3. Uses a 150-word overlap for the next chunk so context is never lost.
    4. Respects sentence boundaries (never splits a sentence in half).
    """
    if not text:
        return []

    # 1. SPLIT BY RAW LINES (To preserve header structure before cleaning spaces)
    raw_lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    semantic_units = [] # Will store either a [HEADER] or a full complete sentence
    current_sentence_buffer = ""

    for line in raw_lines:
        # Heuristic to detect a Header: 
        # - Short length (<= 12 words)
        # - Does not end with sentence punctuation (. ! ?)
        # - Is not a bullet point (-, *, •)
        is_header = (
            len(line.split()) <= 12 
            and not line[-1] in '.!?' 
            and not line.startswith(('-', '*', '•'))
        )

        if is_header:
            # If we were building a sentence, save it first before adding the header
            if current_sentence_buffer:
                # Split any multiple sentences inside the buffer safely
                for s in re.split(r'(?<=[.!?])\s+', current_sentence_buffer.strip()):
                    if s.strip(): semantic_units.append(s.strip())
                current_sentence_buffer = ""
            
            # Make the header stand out for the LLM
            semantic_units.append(f"\n[{line.upper()}]")
        else:
            # It's normal text. Add to our sentence buffer.
            current_sentence_buffer += (" " if current_sentence_buffer else "") + line
            
            # If the buffer now ends with punctuation, it's a complete sentence. Save it.
            if current_sentence_buffer[-1] in '.!?':
                for s in re.split(r'(?<=[.!?])\s+', current_sentence_buffer.strip()):
                    if s.strip(): semantic_units.append(s.strip())
                current_sentence_buffer = ""

    # Flush any remaining text in the buffer
    if current_sentence_buffer:
        for s in re.split(r'(?<=[.!?])\s+', current_sentence_buffer.strip()):
            if s.strip(): semantic_units.append(s.strip())

    if not semantic_units:
        return []

    # 2. OVERLAPPING CHUNK LOGIC (600 Max Words, 150 Overlap)
    chunks = []
    current_chunk = []
    current_word_count = 0

    i = 0
    while i < len(semantic_units):
        unit = semantic_units[i]
        unit_word_count = len(unit.split())

        # If adding this unit exceeds our max word limit
        if current_word_count + unit_word_count > max_words and current_chunk:
            # Save the completed chunk
            chunks.append(" ".join(current_chunk).strip())
            
            # Create overlap for the next chunk (going back ~150 words)
            overlap_chunk = []
            overlap_count = 0
            
            for u in reversed(current_chunk):
                u_len = len(u.split())
                if overlap_count + u_len <= overlap_words:
                    overlap_chunk.insert(0, u)
                    overlap_count += u_len
                else:
                    if not overlap_chunk: # Ensure at least one unit overlaps
                        overlap_chunk.insert(0, u)
                        overlap_count += u_len
                    break
            
            current_chunk = overlap_chunk
            current_word_count = overlap_count
            
        current_chunk.append(unit)
        current_word_count += unit_word_count
        i += 1

    # Append any remaining units as the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def process_and_embed_everything():
    """
    Extracts text and tables from PostgreSQL (PDF pages & CSV tables), applies robust 
    chunking/formatting, and embeds everything into ChromaDB using memory-safe batching.
    """
    db = SessionLocal()

    try:
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        model = get_embedding_model()
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Clear existing collection to avoid duplicated data upon re-running
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            logging.info("Existing ChromaDB collection cleared.")
        except Exception:
            pass

        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

        all_texts = []
        all_metadatas = []
        all_ids = []

        # ==========================================
        # PHASE 1: FETCH AND PROCESS PDF DOCUMENTS
        # ==========================================
        logging.info("Extracting PDF documents from PostgreSQL 'pages' table...")
        docs = db.query(DBDocument).filter(
            DBDocument.file_type == "pdf",
            DBDocument.status == "completed"
        ).all()

        for doc in docs:
            logging.info(f"Processing PDF: {doc.filename}")
            pages = db.query(Page).filter(Page.document_id == doc.id).order_by(Page.page_number).all()

            for page in pages:
                # 1. Safely parse JSON content
                content = page.content or {}
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = {}
                
                # Extract Text and Tables independently
                raw_text = content.get("text", "")
                tables = content.get("tables", [])
                
                # -------------------------------------------------
                # STEP A: PROCESS PDF TEXT WITH OVERLAPPING CHUNKER
                # -------------------------------------------------
                chunks = robust_overlapping_chunker(raw_text, max_words=600, overlap_words=150)

                for idx, chunk in enumerate(chunks):
                    context_header = f"Source Document: {doc.filename} (Page {page.page_number})"
                    final_chunk_text = f"{context_header}\n\n{chunk}"
                    
                    all_texts.append(final_chunk_text)
                    all_metadatas.append({
                        "filename": doc.filename,
                        "type": "pdf_text",
                        "page_number": int(page.page_number),
                        "chunk_index": idx
                    })
                    all_ids.append(str(uuid.uuid4()))

                # -------------------------------------------------
                # STEP B: PROCESS PDF TABLES (Row by Row)
                # -------------------------------------------------
                if tables:
                    for table_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue # Skip empty tables or tables without data rows
                        
                        # Assume the first row contains the headers
                        headers = table[0]
                        
                        # Loop through the remaining rows and map them to headers
                        for row_idx, row in enumerate(table[1:]):
                            # Zip headers and row data together safely
                            row_dict = dict(zip(headers, row))
                            
                            # Format as Semantic Key-Value pairs (e.g., "Domain: Programming | Technologies: Python")
                            formatted_row = " | ".join([f"{str(k).strip()}: {str(v).strip()}" for k, v in row_dict.items() if k and v])
                            
                            table_chunk_text = (
                                f"Source Document: {doc.filename} (Page {page.page_number})\n"
                                f"Table Data (Row {row_idx + 1}): {formatted_row}"
                            )
                            
                            all_texts.append(table_chunk_text)
                            all_metadatas.append({
                                "filename": doc.filename,
                                "type": "pdf_table_row",
                                "page_number": int(page.page_number),
                                "table_index": table_idx,
                                "row_index": row_idx
                            })
                            all_ids.append(str(uuid.uuid4()))

        # ==========================================
        # PHASE 2: FETCH AND FORMAT CSV TABLES
        # ==========================================
        logging.info("Scanning PostgreSQL for dynamic CSV Tables...")
        with engine.connect() as conn:
            # Query the information schema to find all dynamic CSV tables
            tables_res = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'csv_data_%';"))
            csv_tables = [row[0] for row in tables_res]

            for table in csv_tables:
                logging.info(f"Extracting Database Table: {table}")
                rows_res = conn.execute(text(f'SELECT * FROM "{table}";'))
                rows = rows_res.fetchall()
                columns = list(rows_res.keys())

                for row_idx, row in enumerate(rows):
                    # Convert row to dictionary
                    row_dict = dict(zip(columns, row))
                    
                    # Convert row properties into a readable semantic string
                    row_text = " | ".join([f"{col}: {val}" for col, val in row_dict.items() if val is not None])
                    final_chunk_text = f"Dataset: {table}\nRow Record: {row_text}"
                    
                    # CSV rows are usually small, so we don't need to pass them through the chunker
                    all_texts.append(final_chunk_text)
                    all_metadatas.append({
                        "filename": table,
                        "type": "csv_row",
                        "row_index": row_idx
                    })
                    all_ids.append(str(uuid.uuid4()))

        # ==========================================
        # PHASE 3: BATCH EMBEDDING AND STORAGE
        # ==========================================
        total_chunks = len(all_texts)
        if total_chunks > 0:
            logging.info(f"Ready to embed {total_chunks} total chunks. Starting batch processing...")
            
            # Process in batches to prevent Out-Of-Memory (OOM) errors
            for i in range(0, total_chunks, BATCH_SIZE):
                batch_texts = all_texts[i : i + BATCH_SIZE]
                batch_metadatas = all_metadatas[i : i + BATCH_SIZE]
                batch_ids = all_ids[i : i + BATCH_SIZE]
                
                logging.info(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch_texts)} chunks)...")
                
                # Generate embeddings for the current batch
                batch_embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()

                # Insert batch into ChromaDB
                collection.add(
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
            logging.info("✅ Success! All data (Text + PDF Tables + CSVs) is properly chunked, embedded, and saved to ChromaDB.")
        else:
            logging.warning("⚠️ No text or CSV data found in the database to embed.")

    except Exception as e:
        logging.error(f"❌ Pipeline Execution Error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    process_and_embed_everything()