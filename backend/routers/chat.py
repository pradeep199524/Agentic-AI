import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

# --- FIX: Import the NEW streaming function and the store ---
from backend.services.chat_engine import stream_chat_response, SESSION_STORE

router = APIRouter(prefix="/api/v1/chat", tags=["Agentic Chat"])

# =====================================================================
# PYDANTIC MODELS
# =====================================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    history: List[ChatMessage] = []

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Conversation"


# =====================================================================
# CHAT STREAM ENDPOINT
# =====================================================================
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


# =====================================================================
# SESSION MANAGEMENT ENDPOINTS
# =====================================================================
@router.get("/sessions")
async def list_sessions():
    """Returns a list of all chat sessions for the frontend sidebar."""
    sessions = [
        {
            "session_id": s["session_id"], 
            "title": s.get("title", "New Conversation"), 
            "created_at": s.get("created_at", ""),
            "updated_at": s.get("updated_at", "")
        }
        for s in SESSION_STORE.values()
    ]
    # Sort newest to oldest. Default to created_at if updated_at is missing.
    sorted_sessions = sorted(
        sessions, 
        key=lambda x: x["updated_at"] if x["updated_at"] else x["created_at"], 
        reverse=True
    )
    # The Next.js frontend expects the data wrapped in a "sessions" key
    return {"sessions": sorted_sessions}


@router.post("/sessions")
async def start_new_session(req: CreateSessionRequest):
    """Start a brand new chat session from the UI."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    session_data = {
        "session_id": session_id,
        "title": req.title or "New Conversation",
        "created_at": now,
        "updated_at": now,
        "messages": []
    }
    SESSION_STORE[session_id] = session_data
    
    return {"status": "success", "session": session_data}


@router.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    """Load the message history of an old chat when clicked."""
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")
    return SESSION_STORE[session_id]


@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    """Delete a specific chat session."""
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del SESSION_STORE[session_id]
    return {"status": "success", "message": f"Session {session_id} deleted."}


@router.delete("/sessions")
async def remove_all_sessions():
    """Clear all chat sessions from memory."""
    SESSION_STORE.clear()
    return {"status": "success", "message": "All chat history cleared."}