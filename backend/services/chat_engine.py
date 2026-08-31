import os
import json
import uuid
import asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

from backend.services.retriever import HybridRetrievalEngine
from backend.agents.sql_agent import PostgresSQLAgent

# Load environment variables
load_dotenv()

# In-memory chat history
SESSION_STORE: Dict[str, Dict[str, Any]] = {}

# Use Azure Deployment Name as the model
DEFAULT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Initialize Azure OpenAI Client (Zero Hardcoding)
llm_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    timeout=180.0
)

retrieval_engine = HybridRetrievalEngine()
sql_agent = PostgresSQLAgent()


async def stream_chat_response(
    query: str,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 5,
    model: str = DEFAULT_MODEL
) -> AsyncGenerator[str, None]:
    
    clean_query = query.strip()
    session_id = session_id or str(uuid.uuid4())

    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = {"session_id": session_id, "messages": []}

    print(f"[INFO] 🚀 Dynamic Parallel Retrieval Started for: '{clean_query}'")

    # =========================================================================
    # ZERO ROUTING / ZERO HARDCODING: Query both engines concurrently
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

    # Execute both searches concurrently
    db_results, retrieved_docs = await asyncio.gather(fetch_db(), fetch_docs())

    # =========================================================================
    # CONTEXT STRUCTURING: Map explicitly to System Prompt definitions
    # =========================================================================
    
    # 1. Format Database Records
    db_context_str = "\n".join(db_results).strip() if db_results else ""

    # 2. Format Document Text Chunks
    doc_blocks = []
    for d in retrieved_docs:
        doc_name = d.get("metadata", {}).get("filename", "Document")
        page_no = d.get("metadata", {}).get("page_number", "")
        page_str = f" (Page {page_no})" if page_no else ""
        content = d.get("content", "").strip()
        if content:
            doc_blocks.append(f"[{doc_name}{page_str}]\n{content}")
    
    doc_context_str = "\n\n".join(doc_blocks).strip() if doc_blocks else ""

    # 3. Handle Empty Context
    if not db_context_str and not doc_context_str:
        yield json.dumps({
            "token": "I do not know based on the provided data.",
            "done": True,
            "session_id": session_id
        }) + "\n"
        return

    # 4. Assemble the Distinct Structured Context
    context_sections = []
    if db_context_str:
        context_sections.append(f"=== DATABASE RECORDS (Structured Tabular Data) ===\n{db_context_str}")
    if doc_context_str:
        context_sections.append(f"=== DOCUMENT TEXT (Unstructured Text & PDFs) ===\n{doc_context_str}")

    context_block = "\n\n".join(context_sections)

    print(f"\n[DEBUG FINAL CONTEXT TO LLM]:\n{context_block}\n")

    # =========================================================================
    # EVALUATION SYSTEM PROMPT: Dynamic truth-hierarchy reasoning
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

    if history:
        for msg in history[-4:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    messages.append({
        "role": "user",
        "content": f"<context>\n{context_block}\n</context>\n\nQuestion: {clean_query}\nAnswer:"
    })

    # =========================================================================
    # STREAMING GENERATION
    # =========================================================================
    full_response = ""
    try:
        response_stream = await llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
            stream=True
        )

        async for chunk in response_stream:
            # Azure OpenAI sometimes returns empty choices arrays on the first/last chunk
            if not chunk.choices:
                continue
                
            token = chunk.choices[0].delta.content or ""
            if token:
                full_response += token
                yield json.dumps({"token": token, "done": False, "session_id": session_id}) + "\n"

        SESSION_STORE[session_id]["messages"].extend([
            {"role": "user", "content": clean_query},
            {"role": "assistant", "content": full_response}
        ])
        yield json.dumps({"token": "", "done": True, "session_id": session_id}) + "\n"

    except Exception as e:
        yield json.dumps({"token": f"\n[Error: {str(e)}]", "done": True, "session_id": session_id}) + "\n"