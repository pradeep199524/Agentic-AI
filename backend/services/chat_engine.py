import os
import json
import uuid
from typing import AsyncGenerator, Dict, Any, List, Optional
from datetime import datetime
from openai import AsyncOpenAI

from backend.services.retriever import HybridRetrievalEngine
from backend.agents.sql_agent import PostgresSQLAgent  # Updated import to Postgres

SESSION_STORE: Dict[str, Dict[str, Any]] = {}

# Fetch configurations from environment variables to prevent hardcoding
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen2.5-14b-instruct-1m")
FALLBACK_THRESHOLD = float(os.getenv("CHROMA_FALLBACK_THRESHOLD", "0.05"))

llm_client = AsyncOpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1"),
    api_key=os.getenv("LLM_API_KEY", "lm-studio"),
    timeout=180.0
)

# Initialize engines
retrieval_engine = HybridRetrievalEngine()
sql_agent = PostgresSQLAgent()


async def stream_chat_response(
    query: str,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 6,
    model: str = DEFAULT_MODEL
) -> AsyncGenerator[str, None]:
    """
    Retrieves information dynamically.
    Checks ChromaDB first. If confidence is low, falls back to PostgreSQL.
    Never touches, reads, or reloads raw source files during chat.
    """
    clean_query = query.strip()
    session_id = session_id or str(uuid.uuid4())

    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "messages": []
        }

    # 1. First, retrieve unstructured data from ChromaDB
    retrieved_docs = retrieval_engine.retrieve(clean_query, strategy="hybrid_rerank", top_k=top_k)
    
    unstructured_context: List[str] = []
    max_score = 0.0
    
    for doc in retrieved_docs:
        # Dynamically extract score regardless of metadata structure
        score = doc.get("score") or doc.get("metadata", {}).get("score", 0.0)
        
        # Safely convert score to float
        try:
            score_val = float(score)
        except (ValueError, TypeError):
            score_val = 0.0
            
        if score_val > max_score:
            max_score = score_val
            
        meta = doc.get("metadata", {})
        doc_name = meta.get("filename", "Document")
        page_no = meta.get("page_number", "N/A")
        content = doc.get("content", "").strip()
        unstructured_context.append(f"[{doc_name} - Page {page_no}]:\n{content}")

    structured_context: List[str] = []
    
    # 2. Dynamic Threshold Fallback: Trigger Postgres only if ChromaDB score is too low
    if max_score < FALLBACK_THRESHOLD:
        print(f"[INFO] ChromaDB max score ({max_score:.4f}) is below threshold ({FALLBACK_THRESHOLD}). Triggering DB Fallback...")
        structured_context = await sql_agent.execute_query(clean_query, llm_client, model)
    else:
        print(f"[INFO] ChromaDB score ({max_score:.4f}) meets threshold. Skipping DB fallback.")

    # 3. Unified In-Memory Context Pool
    all_context = unstructured_context + structured_context

    if not all_context:
        yield json.dumps({
            "token": "I do not know based on the provided data.",
            "done": True,
            "session_id": session_id
        })
        return

    context_block = "\n\n---\n\n".join(all_context)

    # 4. Synthesize Answer
    system_instruction = (
        "You are an enterprise AI assistant strictly grounded in the provided context.\n"
        "DIRECTIVES:\n"
        "1. FACTUAL GROUNDING: Base your answer strictly on the facts present in <context>.\n"
        "2. MULTI-SOURCE SYNTHESIS: If the context contains overlapping or differing information from multiple distinct sources, cleanly state the findings from both without explicitly citing file names or page numbers.\n"
        "3. PLAIN NATURAL TEXT ONLY: Answer in direct, standard conversational sentences. NEVER output JSON, function calls, parameters, or dictionary objects. Do NOT include citation brackets or file names in the text.\n"
        "4. NO QUESTION REPETITION: Do NOT repeat, paraphrase, or summarize the user's question.\n"
        "5. VERBATIM METRICS: Output specific entity names, values, metrics, and identifiers exactly as presented.\n"
        "6. UNKNOWN INFO: If the answer is not present in the context, output ONLY: 'I do not know based on the provided data.'"
    )

    messages = [{"role": "system", "content": system_instruction}]

    if history:
        for msg in history[-4:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    messages.append({
        "role": "user",
        "content": (
            f"<context>\n{context_block}\n</context>\n\n"
            f"Question: {clean_query}\n\n"
            "Provide the direct plain text answer:"
        )
    })

    full_response = ""
    try:
        response_stream = await llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
            stream=True
        )

        async for chunk in response_stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_response += token
                yield json.dumps({"token": token, "done": False, "session_id": session_id}) + "\n"

        SESSION_STORE[session_id]["messages"].append({"role": "user", "content": clean_query})
        SESSION_STORE[session_id]["messages"].append({"role": "assistant", "content": full_response})

        yield json.dumps({"token": "", "done": True, "session_id": session_id}) + "\n"

    except Exception as e:
        yield json.dumps({"token": f"\n[Error: {str(e)}]", "done": True, "session_id": session_id}) + "\n"