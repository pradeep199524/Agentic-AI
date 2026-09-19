import os
import re
import uuid
import json
import logging
import argparse
import chromadb
import tiktoken
from sqlalchemy import text
from sklearn.metrics.pairwise import cosine_similarity
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

# Initialize tiktoken for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")

def get_embedding_model() -> SentenceTransformer:
    """Returns the shared SentenceTransformer model instance."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def get_vector_store():
    """Returns the native ChromaDB collection instance."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def count_tokens(text: str) -> int:
    """Returns the exact number of LLM tokens in a string."""
    return len(tokenizer.encode(text))


# =========================================================================
# CHUNKING STRATEGY 1: FIXED (Rigid Token/Character Limits)
# =========================================================================
def fixed_size_chunker(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Splits text rigidly by an exact token count, with a slight overlap to prevent cutting words."""
    if not text:
        return []
    
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens).strip())
        
    return chunks


# =========================================================================
# CHUNKING STRATEGY 2: STRUCTURE-AWARE (Headings, Sections & Token Limits)
# =========================================================================
def structure_aware_chunker(text: str, max_tokens: int = 500) -> list:
    """
    Splits text along structural document boundaries (Headings, Sections, Paragraphs)
    and enforces strict token limits using tiktoken to prevent model truncation.
    """
    if not text:
        return []

    # Regex matches Markdown headers (# H1, ## H2), 'Section X', or double newlines
    section_pattern = r'(?=(\n#{1,6}\s+[^\n]+|\n(?:Section|\bPart|\bChapter)\s+[0-9IVXLCDM]+[^\n]*|\n\s*\n))'
    raw_sections = re.split(section_pattern, text.strip(), flags=re.IGNORECASE)
    
    sections = [s.strip() for s in raw_sections if s and s.strip()]

    final_chunks = []
    current_chunk = ""

    for section in sections:
        section_tokens = count_tokens(section)

        # If a single section is larger than max_tokens, split it with fixed token chunker
        if section_tokens > max_tokens:
            if current_chunk:
                final_chunks.append(current_chunk.strip())
                current_chunk = ""
            final_chunks.extend(fixed_size_chunker(section, chunk_size=max_tokens, overlap=50))
            continue

        # If adding the section exceeds the limit, push current and start new
        if count_tokens(current_chunk + "\n\n" + section) > max_tokens:
            if current_chunk:
                final_chunks.append(current_chunk.strip())
            current_chunk = section
        else:
            current_chunk = f"{current_chunk}\n\n{section}".strip()

    if current_chunk:
        final_chunks.append(current_chunk.strip())

    return final_chunks


# =========================================================================
# CHUNKING STRATEGY 3: SEMANTIC (Meaning-based similarity)
# =========================================================================
def true_semantic_chunker(text: str, embedding_model, max_tokens: int = 500, similarity_threshold: float = 0.45) -> list:
    """Splits text when the semantic topic changes, while strictly enforcing token limits."""
    if not text:
        return []

    # 1. Split text into individual sentences safely
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    # 2. Get embeddings for every single sentence to compare their meanings
    sentence_embeddings = embedding_model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    current_chunk_tokens = count_tokens(sentences[0])

    # 3. Iterate through sentences and calculate similarity
    for i in range(1, len(sentences)):
        sentence = sentences[i]
        sentence_tokens = count_tokens(sentence)

        # Calculate Cosine Similarity between the PREVIOUS sentence and the CURRENT one
        similarity = cosine_similarity(
            [sentence_embeddings[i - 1]], 
            [sentence_embeddings[i]]
        )[0][0]

        # Splitting condition: Topic changed OR token limit reached
        if similarity < similarity_threshold or (current_chunk_tokens + sentence_tokens > max_tokens):
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_chunk_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens

    # Flush the remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


# =========================================================================
# MAIN PIPELINE
# =========================================================================
def process_and_embed_everything(chunking_strategy: str = "semantic", target_filenames: list = None):
    """
    Extracts text and tables from PostgreSQL, applies the user-selected chunking strategy, 
    and embeds everything into ChromaDB. Supports targeted embedding for specific files.
    """
    db = SessionLocal()

    try:
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        logging.info(f"Using text chunking strategy: {chunking_strategy.upper()}")
        model = get_embedding_model()
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # -------------------------------------------------------------
        # STEP 1: Handle Database Wiping OR Targeted Deletion
        # -------------------------------------------------------------
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        
        if target_filenames and len(target_filenames) > 0:
            logging.info(f"Targeted embedding mode activated for: {target_filenames}")
            # Surgically remove old embeddings for ONLY the selected files
            for fname in target_filenames:
                try:
                    collection.delete(where={"filename": fname})
                    logging.info(f"Cleared old ChromaDB embeddings for: {fname}")
                except Exception as e:
                    logging.warning(f"Could not delete old embeddings for {fname}: {e}")
        else:
            # If no files selected, we do a full wipe and rebuild
            try:
                chroma_client.delete_collection(name=COLLECTION_NAME)
                logging.info("Existing ChromaDB collection cleared for full rebuild.")
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
        
        # Start base query (Removed strict status filter to match frontend UI)
        pdf_query = db.query(DBDocument).filter(
            DBDocument.file_type == "pdf"
        )
        
        # Filter query by specific files if requested
        if target_filenames and len(target_filenames) > 0:
            pdf_query = pdf_query.filter(DBDocument.filename.in_(target_filenames))
            
        docs = pdf_query.all()

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
                # STEP A: DYNAMIC ROUTING TO SELECTED CHUNKER
                # -------------------------------------------------
                if chunking_strategy == "fixed":
                    chunks = fixed_size_chunker(text=raw_text, chunk_size=500, overlap=50)
                elif chunking_strategy == "structured":
                    # FIX: Now using the upgraded structure_aware_chunker
                    chunks = structure_aware_chunker(text=raw_text, max_tokens=500)
                else: # Default to Semantic
                    chunks = true_semantic_chunker(
                        text=raw_text, 
                        embedding_model=model, 
                        max_tokens=500, 
                        similarity_threshold=0.45
                    )

                for idx, chunk in enumerate(chunks):
                    context_header = f"Source Document: {doc.filename} (Page {page.page_number})"
                    final_chunk_text = f"{context_header}\n\n{chunk}"
                    
                    all_texts.append(final_chunk_text)
                    all_metadatas.append({
                        "filename": doc.filename,
                        "type": "pdf_text",
                        "page_number": int(page.page_number),
                        "chunk_index": idx,
                        "strategy_used": chunking_strategy
                    })
                    all_ids.append(str(uuid.uuid4()))

                # -------------------------------------------------
                # STEP B: PROCESS PDF TABLES (Always Structured)
                # -------------------------------------------------
                if tables:
                    for table_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue 
                        
                        headers = table[0]
                        table_rows_formatted = []
                        
                        for row_idx, row in enumerate(table[1:]):
                            row_dict = dict(zip(headers, row))
                            formatted_row = " | ".join([f"{str(k).strip()}: {str(v).strip()}" for k, v in row_dict.items() if k and v])
                            table_rows_formatted.append(f"Row {row_idx + 1}: {formatted_row}")
                            
                        full_table_text = "\n".join(table_rows_formatted)
                        
                        table_chunk_text = (
                            f"Source Document: {doc.filename} (Page {page.page_number})\n"
                            f"Full Table Data:\n{full_table_text}"
                        )
                        
                        all_texts.append(table_chunk_text)
                        all_metadatas.append({
                            "filename": doc.filename,
                            "type": "pdf_full_table",
                            "page_number": int(page.page_number),
                            "table_index": table_idx
                        })
                        all_ids.append(str(uuid.uuid4()))

        # ==========================================
        # PHASE 2: FETCH AND FORMAT CSV TABLES
        # ==========================================
        logging.info("Scanning PostgreSQL for dynamic CSV Tables...")
        with engine.connect() as conn:
            tables_res = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'csv_data_%';"))
            csv_tables = [row[0] for row in tables_res]
            
            # Filter CSV tables by specific files if requested
            if target_filenames and len(target_filenames) > 0:
                csv_tables = [t for t in csv_tables if t in target_filenames]

            for table in csv_tables:
                logging.info(f"Extracting Database Table: {table}")
                rows_res = conn.execute(text(f'SELECT * FROM "{table}";'))
                rows = rows_res.fetchall()
                columns = list(rows_res.keys())

                # -------------------------------------------------
                # STRATEGY 1: STRUCTURED (Row-by-Row Chunking)
                # -------------------------------------------------
                if chunking_strategy == "structured":
                    for row_idx, row in enumerate(rows):
                        row_dict = dict(zip(columns, row))
                        row_text = " | ".join([f"{col}: {val}" for col, val in row_dict.items() if val is not None])
                        final_chunk_text = f"Dataset: {table}\nRow Record: {row_text}"
                        
                        all_texts.append(final_chunk_text)
                        all_metadatas.append({
                            "filename": table,
                            "type": "csv_row",
                            "row_index": row_idx,
                            "strategy_used": "structured"
                        })
                        all_ids.append(str(uuid.uuid4()))
                        
                # -------------------------------------------------
                # STRATEGY 2 & 3: FIXED OR SEMANTIC CHUNKING
                # -------------------------------------------------
                else:
                    # Combine all rows into a massive readable text block
                    combined_csv_text = ""
                    for row_idx, row in enumerate(rows):
                        row_dict = dict(zip(columns, row))
                        row_text = " | ".join([f"{col}: {val}" for col, val in row_dict.items() if val is not None])
                        combined_csv_text += f"Row {row_idx + 1} - {row_text}. "
                    
                    # Apply the chosen chunking algorithm to the combined text
                    if chunking_strategy == "fixed":
                        chunks = fixed_size_chunker(text=combined_csv_text, chunk_size=500, overlap=50)
                    else: # Semantic
                        chunks = true_semantic_chunker(
                            text=combined_csv_text, 
                            embedding_model=model, 
                            max_tokens=500, 
                            similarity_threshold=0.45
                        )
                        
                    for idx, chunk in enumerate(chunks):
                        final_chunk_text = f"Dataset: {table}\n\n{chunk}"
                        
                        all_texts.append(final_chunk_text)
                        all_metadatas.append({
                            "filename": table,
                            "type": "csv_text_block",
                            "chunk_index": idx,
                            "strategy_used": chunking_strategy
                        })
                        all_ids.append(str(uuid.uuid4()))

        # ==========================================
        # PHASE 3: BATCH EMBEDDING AND STORAGE
        # ==========================================
        total_chunks = len(all_texts)
        if total_chunks > 0:
            logging.info(f"Ready to embed {total_chunks} total chunks. Starting batch processing...")
            
            for i in range(0, total_chunks, BATCH_SIZE):
                batch_texts = all_texts[i : i + BATCH_SIZE]
                batch_metadatas = all_metadatas[i : i + BATCH_SIZE]
                batch_ids = all_ids[i : i + BATCH_SIZE]
                
                logging.info(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch_texts)} chunks)...")
                
                batch_embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()

                collection.add(
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
            logging.info(f"✅ Success! Data chunked using [{chunking_strategy.upper()}], embedded, and saved to ChromaDB.")
        else:
            logging.warning("⚠️ No text or CSV data found to embed for the selected files.")

    except Exception as e:
        logging.error(f"❌ Pipeline Execution Error: {e}")
        raise e  # FIX: Re-raise the exception so FastAPI properly triggers a 500 Error in the UI
    finally:
        db.close()


if __name__ == "__main__":
    # Allows the user to select the chunking technique directly from the command line!
    parser = argparse.ArgumentParser(description="Run the RAG Data Ingestion and Embedding Pipeline.")
    
    parser.add_argument(
        "--strategy", 
        type=str, 
        choices=["fixed", "structured", "semantic"], 
        default="semantic",
        help="Select the chunking technique for unstructured PDF text."
    )
    
    parser.add_argument(
        "--files", 
        type=str, 
        default="",
        help="Comma-separated list of filenames to target (e.g., 'policy.pdf,csv_data_sales'). Leave blank for all."
    )
    
    args = parser.parse_args()
    
    target_list = [f.strip() for f in args.files.split(",")] if args.files else None
    
    process_and_embed_everything(chunking_strategy=args.strategy, target_filenames=target_list)