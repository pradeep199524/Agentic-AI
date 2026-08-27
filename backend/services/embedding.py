import os
import re
import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

from backend.database import SessionLocal
from backend.models import Document as DBDocument, Page

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- DYNAMIC CONFIGURATION (NO HARDCODING) ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(os.getcwd(), "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_knowledge_base")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 600))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))


def is_heading(line: str) -> bool:
    """Detects if a single line acts as a section heading or title based on structural patterns."""
    line = line.strip()
    if not line or len(line) > 100:
        return False
    is_numbered = bool(re.match(r'^(\d+(\.\d+)*|[A-Z]\.)\s+[A-Z]', line))
    is_all_caps = line.isupper() and len(line.split()) <= 10
    is_title_case = bool(re.match(r'^[A-Z][a-zA-Z0-9\s,&/\-]{2,40}$', line)) and len(line.split()) <= 6
    return is_numbered or is_all_caps or is_title_case


def create_semantic_chunks(page_text: str, tables: list, metadata_base: dict) -> List[Document]:
    """Splits page text and tables while injecting hierarchical document and section context dynamically."""
    chunks = []
    
    # Text splitter driven by dynamic environment variables
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    
    doc_name = metadata_base.get("filename", "Unknown Document").replace(".pdf", "").replace("_", " ")
    current_section = "General Overview"
    section_buffer = []

    # 1. Process Text with Context Enrichment
    if page_text:
        lines = page_text.split('\n')
        for line in lines:
            if is_heading(line):
                if section_buffer:
                    full_text = "\n".join(section_buffer)
                    for idx, txt in enumerate(text_splitter.split_text(full_text)):
                        enriched_text = f"Document: {doc_name}\nSection: {current_section}\n\n{txt}"
                        meta = {**metadata_base, "section": current_section, "type": "text", "sub_index": idx}
                        chunks.append(Document(page_content=enriched_text, metadata=meta))
                    section_buffer = []
                current_section = line.strip()
            else:
                section_buffer.append(line)

        if section_buffer:
            full_text = "\n".join(section_buffer)
            for idx, txt in enumerate(text_splitter.split_text(full_text)):
                enriched_text = f"Document: {doc_name}\nSection: {current_section}\n\n{txt}"
                meta = {**metadata_base, "section": current_section, "type": "text", "sub_index": idx}
                chunks.append(Document(page_content=enriched_text, metadata=meta))

    # 2. Process Tables with Heading Context
    if tables:
        for t_idx, table in enumerate(tables):
            if not table or len(table) < 1:
                continue
            headers = " | ".join(table[0])
            rows = [" | ".join(row) for row in table[1:]]
            table_markdown = (
                f"Document: {doc_name}\nSection: {current_section}\n"
                f"Table {t_idx + 1}:\n| {headers} |\n| " + " | ".join(["---"] * len(table[0])) + " |\n"
            )
            table_markdown += "\n".join([f"| {r} |" for r in rows])
            
            meta = {**metadata_base, "section": current_section, "type": "table", "table_index": t_idx}
            chunks.append(Document(page_content=table_markdown, metadata=meta))

    return chunks


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Initializes and returns the Hugging Face embedding model based on env variables."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_vector_store(embeddings: HuggingFaceEmbeddings = None) -> Chroma:
    """Returns the Chroma vector store instance connected to the dynamic path."""
    if embeddings is None:
        embeddings = get_embedding_model()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )


def build_vector_store():
    """Reads processed data from SQL, applies context-enriched chunking, and persists vectors."""
    db = SessionLocal()
    try:
        logging.info(f"Initializing Hugging Face model: {EMBEDDING_MODEL_NAME}...")
        embeddings = get_embedding_model()
        vector_store = get_vector_store(embeddings)

        # Clear previous vector collection to avoid stale chunks
        try:
            logging.info("Resetting existing ChromaDB collection...")
            vector_store.delete_collection()
            vector_store = get_vector_store(embeddings)
        except Exception:
            pass

        docs = db.query(DBDocument).filter(
            DBDocument.file_type == 'pdf', 
            DBDocument.status == 'completed'
        ).all()

        if not docs:
            logging.warning("No completed PDF documents found in database to chunk.")
            return

        all_documents: List[Document] = []

        for doc in docs:
            logging.info(f"Processing structure-aware chunks for: {doc.filename}")
            pages = db.query(Page).filter(Page.document_id == doc.id).order_by(Page.page_number.asc()).all()

            for page in pages:
                content = page.content or {}
                text = content.get("text", "")
                tables = content.get("tables", [])

                base_meta = {
                    "document_id": str(doc.id),
                    "filename": doc.filename,
                    "page_number": page.page_number
                }

                page_chunks = create_semantic_chunks(text, tables, base_meta)
                all_documents.extend(page_chunks)

        if all_documents:
            logging.info(f"Adding {len(all_documents)} semantic chunks to ChromaDB at {CHROMA_PERSIST_DIR}...")
            vector_store.add_documents(documents=all_documents)
            logging.info("Embedding & storage process completed successfully!")
        else:
            logging.warning("No chunks generated from documents.")

    except Exception as e:
        logging.error(f"Error during chunking and vector storage: {e}")
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    build_vector_store()