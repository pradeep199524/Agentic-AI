import os
import re
import uuid
import json
import logging
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

def true_semantic_chunker(text: str, embedding_model, max_tokens: int = 500, similarity_threshold: float = 0.45) -> list:
    """
    Splits text when the semantic topic changes, while strictly enforcing token limits.
    """
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

def process_and_embed_everything():
    """
    Extracts text and tables from PostgreSQL, applies TRUE SEMANTIC chunking, 
    and embeds everything into ChromaDB using memory-safe batching.
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
                # STEP A: PROCESS PDF TEXT WITH TRUE SEMANTIC CHUNKER
                # -------------------------------------------------
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
                        "chunk_index": idx
                    })
                    all_ids.append(str(uuid.uuid4()))

                # -------------------------------------------------
                # STEP B: PROCESS PDF TABLES (Whole Table Chunking)
                # -------------------------------------------------
                if tables:
                    for table_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue # Skip empty tables or tables without data rows
                        
                        headers = table[0]
                        table_rows_formatted = []
                        
                        # Gather all rows into a single list
                        for row_idx, row in enumerate(table[1:]):
                            row_dict = dict(zip(headers, row))
                            formatted_row = " | ".join([f"{str(k).strip()}: {str(v).strip()}" for k, v in row_dict.items() if k and v])
                            table_rows_formatted.append(f"Row {row_idx + 1}: {formatted_row}")
                            
                        # Join all rows together into one massive string block
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

            for table in csv_tables:
                logging.info(f"Extracting Database Table: {table}")
                rows_res = conn.execute(text(f'SELECT * FROM "{table}";'))
                rows = rows_res.fetchall()
                columns = list(rows_res.keys())

                for row_idx, row in enumerate(rows):
                    row_dict = dict(zip(columns, row))
                    row_text = " | ".join([f"{col}: {val}" for col, val in row_dict.items() if val is not None])
                    final_chunk_text = f"Dataset: {table}\nRow Record: {row_text}"
                    
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
                
            logging.info("✅ Success! All data is properly chunked with TRUE semantics, embedded, and saved to ChromaDB.")
        else:
            logging.warning("⚠️ No text or CSV data found in the database to embed.")

    except Exception as e:
        logging.error(f"❌ Pipeline Execution Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    process_and_embed_everything()