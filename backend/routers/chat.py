from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

# --- FIX: Import the NEW streaming function instead of the old one ---
from backend.services.chat_engine import stream_chat_response, SESSION_STORE

router = APIRouter(prefix="/api/v1/chat", tags=["Agentic Chat"])

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    history: List[ChatMessage] = []

# --- FIX: New /stream endpoint and StreamingResponse ---
@router.post("/stream")
async def chat_stream_endpoint(payload: ChatRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Convert Pydantic objects to dictionaries for the engine
    history_dicts = [{"role": m.role, "content": m.content} for m in payload.history]
    
    # Return the StreamingResponse
    return StreamingResponse(
        stream_chat_response(
            query=payload.query,
            session_id=payload.session_id,
            history=history_dicts,
            top_k=2
        ),
        media_type="application/x-ndjson"
    )

@router.get("/sessions")
async def list_sessions():
    """Returns a list of all chat sessions."""
    sessions = [
        {"session_id": s["session_id"], "title": s.get("title", "Chat Session"), "created_at": s["created_at"]}
        for s in SESSION_STORE.values()
    ]
    return sorted(sessions, key=lambda x: x["created_at"], reverse=True)

@router.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")
    return SESSION_STORE[session_id]