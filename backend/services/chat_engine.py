import os
import json
import uuid
import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Any, List, Optional
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

from backend.services.retriever import HybridRetrievalEngine
from backend.agents.sql_agent import PostgresSQLAgent

# Load environment variables
load_dotenv()


SESSION_STORE: Dict[str, Dict[str, Any]] = {}

def create_new_session(title: Optional[str] = "New Conversation") -> Dict[str, Any]:
    """Initializes and returns a new chat session."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    session_data = {
        "session_id": session_id,
        "title": title or "New Conversation",
        "created_at": now,
        "updated_at": now,
        "messages": []
    }
    SESSION_STORE[session_id] = session_data
    return session_data

def get_all_sessions() -> List[Dict[str, Any]]:
    """Returns metadata for all available chat sessions, sorted by most recent."""
    sessions = [
        {
            "session_id": s["session_id"],
            "title": s.get("title", "New Conversation"),
            "created_at": s.get("created_at"),
            "updated_at": s.get("updated_at"),
            "message_count": len(s.get("messages", []))
        }
        for s in SESSION_STORE.values()
    ]
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions

def get_session_history(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full conversation history for a given session."""
    return SESSION_STORE.get(session_id)

def delete_session(session_id: str) -> bool:
    """Deletes a chat session and its history. Returns True if deleted."""
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
        return True
    return False

def clear_all_sessions() -> bool:
    """Clears all in-memory chat sessions."""
    SESSION_STORE.clear()
    return True


# =============================================================================
# AZURE OPENAI & RETRIEVAL ENGINE SETUP
# =============================================================================
DEFAULT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

llm_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    timeout=180.0
)

retrieval_engine = HybridRetrievalEngine()
sql_agent = PostgresSQLAgent()


# =============================================================================
# CHAT STREAMING ENGINE
# =============================================================================
async def stream_chat_response(
    query: str,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 5,
    model: str = DEFAULT_MODEL
) -> AsyncGenerator[str, None]:
    
    clean_query = query.strip()
    
    # 1. Initialize session if missing
    if not session_id or session_id not in SESSION_STORE:
        session_obj = create_new_session(title=clean_query[:40] + ("..." if len(clean_query) > 40 else ""))
        session_id = session_obj["session_id"]
    else:
        # If it was a default title, name it after the first user query
        if SESSION_STORE[session_id].get("title") in ("New Conversation", None):
            SESSION_STORE[session_id]["title"] = clean_query[:40] + ("..." if len(clean_query) > 40 else "")

    print(f"[INFO] 🚀 Dynamic Retrieval for Session [{session_id[:8]}...]: '{clean_query}'")

    # =========================================================================
    # PARALLEL RETRIEVAL (Zero Hardcoding)
    # =========================================================================
    async def fetch_db():
        print("[INFO] Checking PostgreSQL...")
        return await sql_agent.execute_query(clean_query, llm_client, model)

    async def fetch_docs():
        print("[INFO] Checking ChromaDB...")
        return await asyncio.to_thread(
            retrieval_engine.retrieve, 
            clean_query, 
            strategy="hybrid_rerank", 
            top_k=top_k
        )

    db_results, retrieved_docs = await asyncio.gather(fetch_db(), fetch_docs())

    # =========================================================================
    # CONTEXT STRUCTURING
    # =========================================================================
    db_context_str = "\n".join(db_results).strip() if db_results else ""

    doc_blocks = []
    for d in retrieved_docs:
        doc_name = d.get("metadata", {}).get("filename", "Document")
        page_no = d.get("metadata", {}).get("page_number", "")
        page_str = f" (Page {page_no})" if page_no else ""
        content = d.get("content", "").strip()
        if content:
            doc_blocks.append(f"[{doc_name}{page_str}]\n{content}")
    
    doc_context_str = "\n\n".join(doc_blocks).strip() if doc_blocks else ""

    # Empty Context Guard
    if not db_context_str and not doc_context_str:
        yield json.dumps({
            "token": "I do not know based on the provided data.",
            "done": True,
            "session_id": session_id
        }) + "\n"
        return

    context_sections = []
    if db_context_str:
        context_sections.append(f"=== DATABASE RECORDS (Structured Tabular Data) ===\n{db_context_str}")
    if doc_context_str:
        context_sections.append(f"=== DOCUMENT TEXT (Unstructured Text & PDFs) ===\n{doc_context_str}")

    context_block = "\n\n".join(context_sections)

    # =========================================================================
    # SYSTEM PROMPT
    # =========================================================================
    system_instruction = (
        "You are an enterprise AI assistant strictly grounded in the provided context.\n"
        "You will receive context from two distinct sources: DATABASE RECORDS (structured tabular data) and DOCUMENT TEXT (unstructured PDF/text data).\n\n"
        "SOURCE EVALUATION & CONFLICT RESOLUTION:\n"
        "1. Do not rely on specific keywords. Instead, evaluate the data provided in both sources against the user's intent.\n"
        "2. DATABASE TRUTH: 'DATABASE RECORDS' represent the absolute system of record for structured metrics, entity relationships, and raw tabular data. If answering a question requiring data comparison or exact metrics, prioritize this source.\n"
        "3. DOCUMENT TRUTH: 'DOCUMENT TEXT' represents the system of record for policies, guidelines, definitions, and unstructured narratives.\n"
        "4. CONTRADICTIONS: If both sources contain data that answers the query but the numeric values or entities conflict, 'DATABASE RECORDS' strictly supersedes 'DOCUMENT TEXT'.\n\n"
        "STRICT DIRECTIVES:\n"
        "1. Provide extremely direct, brief answers. Get straight to the point.\n"
        "2. NO META-TALK: Never say 'According to the database' or 'Based on the context'.\n"
        "3. NO MARKDOWN: Output raw plain text ONLY. No asterisks (**) or hashes (#).\n"
        "4. If neither source contains the answer, output ONLY: 'I do not know based on the provided data.'"
    )

    messages = [{"role": "system", "content": system_instruction}]

    # Multi-turn history: Use passed history or pull stored session messages (last 6 turns)
    session_messages = history or SESSION_STORE[session_id].get("messages", [])
    for msg in session_messages[-6:]:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })

    messages.append({
        "role": "user",
        "content": f"<context>\n{context_block}\n</context>\n\nQuestion: {clean_query}\nAnswer:"
    })

    # =========================================================================
    # STREAMING GENERATION
    # =========================================================================
    full_response = ""
    now_iso = datetime.now(timezone.utc).isoformat()
    
    try:
        response_stream = await llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
            stream=True
        )

        async for chunk in response_stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content or ""
            if token:
                full_response += token
                yield json.dumps({"token": token, "done": False, "session_id": session_id}) + "\n"

        # Update Session History
        SESSION_STORE[session_id]["updated_at"] = now_iso
        SESSION_STORE[session_id]["messages"].extend([
            {"role": "user", "content": clean_query, "timestamp": now_iso},
            {"role": "assistant", "content": full_response, "timestamp": now_iso}
        ])

        yield json.dumps({"token": "", "done": True, "session_id": session_id}) + "\n"

    except Exception as e:
        yield json.dumps({"token": f"\n[Error: {str(e)}]", "done": True, "session_id": session_id}) + "\n"