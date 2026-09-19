import os
import uuid
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import text

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from backend.database import engine
from backend.services.agent_orchestrator import agent_app

logger = logging.getLogger("agent_router")
router = APIRouter(prefix="/api/v1/agent", tags=["Agentic Workflow (Module 6)"])

# In-memory session tracking for active workflows and approval checkpoints
SESSION_STORE: Dict[str, Dict[str, Any]] = {}


# ==========================================
# REQUEST / RESPONSE SCHEMAS
# ==========================================
class AgentChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class AgentApprovalRequest(BaseModel):
    session_id: str
    approved: bool


# ==========================================
# HELPER: PARSE TRACE FOR THE FRONTEND
# ==========================================
def extract_execution_trace(messages: list) -> List[Dict[str, Any]]:
    """Transforms raw LangChain messages into a structured trace of plans and tool calls."""
    trace = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                trace.append({
                    "type": "tool_call",
                    "tool": tc.get("name"),
                    "args": tc.get("args")
                })
        elif isinstance(msg, ToolMessage):
            trace.append({
                "type": "tool_result",
                "tool": msg.name,
                "output": msg.content
            })
    return trace


# ==========================================
# 1. RUN AGENT WORKFLOW ENDPOINT
# ==========================================
@router.post("/chat")
async def run_agent(request: AgentChatRequest):
    """Executes the agent workflow. Returns the final answer or pauses for approval."""
    session_id = request.session_id or str(uuid.uuid4())

    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "requires_approval": False,
        "pending_action": ""
    }

    try:
        # Run graph through the supervisor and nodes
        result_state = agent_app.invoke(initial_state)

        trace = extract_execution_trace(result_state.get("messages", []))
        last_message = result_state["messages"][-1].content if result_state.get("messages") else ""

        # Case A: Paused for Human Approval
        if result_state.get("requires_approval"):
            SESSION_STORE[session_id] = {
                "state": result_state,
                "pending_action": result_state.get("pending_action")
            }
            return {
                "session_id": session_id,
                "status": "awaiting_approval",
                "message": last_message,
                "pending_action": result_state.get("pending_action"),
                "trace": trace
            }

        # Case B: Execution Completed Normally
        return {
            "session_id": session_id,
            "status": "completed",
            "response": last_message,
            "trace": trace
        }

    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 2. HUMAN-IN-THE-LOOP APPROVAL ENDPOINT
# ==========================================
@router.post("/approve")
async def handle_human_approval(request: AgentApprovalRequest):
    """Resumes the paused agent workflow after human decision."""
    session_data = SESSION_STORE.get(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session expired or not found awaiting approval.")

    current_state = session_data.get("state")

    if not request.approved:
        # User Rejected
        del SESSION_STORE[request.session_id]
        return {
            "session_id": request.session_id,
            "status": "rejected",
            "response": "The requested action was rejected by the operator. No changes were made."
        }

    # --- FIX: Scan backwards to find the actual message containing the tool calls ---
    tool_call_message = None
    for msg in reversed(current_state["messages"]):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call_message = msg
            break

    if not tool_call_message:
        raise HTTPException(status_code=400, detail="No pending tool calls found in the current state.")

    # Dynamically extract the exact tool, arguments, and ID from the paused state
    target_tool = tool_call_message.tool_calls[0]
    tool_name = target_tool["name"]
    tool_args = target_tool["args"]
    actual_tool_call_id = target_tool["id"]

    try:
        # Execute based on the exact tool requested, rather than parsing string text
        if tool_name == "stage_email_draft":
            email = tool_args.get("email_address")
            logger.info(f"📧 SIMULATING EMAIL DISPATCH TO: {email}")
            action_result = f"Email successfully sent to {email} via SMTP API."
            
        elif tool_name == "stage_sql_update":
            sql_query = tool_args.get("sql_query")
            with engine.connect() as conn:
                conn.execute(text(sql_query))
                conn.commit()
            action_result = f"Successfully executed approved query: {sql_query}"
            
        else:
            action_result = f"Action for {tool_name} approved, but no execution logic is defined."

        # Resume state and complete the final synthesis
        followup_state = {
            "messages": current_state["messages"] + [
                ToolMessage(
                    content=action_result,
                    tool_call_id=actual_tool_call_id,
                    name=tool_name 
                ),
                HumanMessage(content="The action was officially approved and executed. Please give the final summary.")
            ],
            "requires_approval": False,
            "pending_action": ""
        }

        final_result = agent_app.invoke(followup_state)
        final_answer = final_result["messages"][-1].content
        trace = extract_execution_trace(final_result.get("messages", []))

        # Cleanup memory after successful execution
        del SESSION_STORE[request.session_id]

        return {
            "session_id": request.session_id,
            "status": "completed",
            "response": final_answer,
            "trace": trace
        }

    except Exception as e:
        logger.error(f"Failed to execute approved action: {e}")
        raise HTTPException(status_code=500, detail=f"Execution error: {e}")