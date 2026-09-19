import os
import json
import logging
from typing import TypedDict, Annotated, Sequence, Literal
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI  
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Read live database tables dynamically
from sqlalchemy import text, inspect 
from backend.database import engine
import chromadb

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ==========================================
# 1. DEFINE THE AGENT'S MEMORY (STATE)
# ==========================================
class AgentState(TypedDict):
    """This acts as the memory for our workflow as it moves from step to step."""
    messages: Annotated[Sequence[BaseMessage], list.__add__]
    requires_approval: bool       # Triggers Human-in-the-Loop
    pending_action: str           # The dangerous action waiting for approval


# ==========================================
# 2. DEFINE THE TOOLS (THE ACTIONS)
# ==========================================

@tool
def search_pdf_knowledge_base(query: str) -> str:
    """Use this tool to search the Vector Database (Chroma) for unstructured text, policies, and PDF documents."""
    try:
        from backend.services.embedding import get_embedding_model
        
        # Uses env variables so collections are never hardcoded in the script
        chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))
        collection = chroma_client.get_collection(name=os.getenv("COLLECTION_NAME", "enterprise_knowledge_base"))
        model = get_embedding_model()
        
        query_embedding = model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        
        if not results['documents'][0]:
            return "No relevant documents found in the Vector Database."
            
        return "\n\n".join(results['documents'][0])
    except Exception as e:
        return f"Error searching document database: {e}"


@tool
def execute_sql_read_query(sql_query: str) -> str:
    """Use this tool to run SELECT queries on PostgreSQL for CSV math, filtering, and structured data retrieval."""
    if "UPDATE" in sql_query.upper() or "DELETE" in sql_query.upper() or "DROP" in sql_query.upper() or "INSERT" in sql_query.upper():
        return "ERROR: You cannot use this tool to modify data."
        
    try:
        with engine.connect() as conn:
            cursor = conn.execute(text(sql_query))
            rows = cursor.fetchall()
            if not rows:
                return "Query executed successfully, but returned 0 rows."
                
            keys = list(cursor.keys())
            formatted_rows = [" | ".join([f"{k}: {v}" for k, v in dict(zip(keys, row)).items()]) for row in rows]
            return "DATABASE RESULTS:\n" + "\n".join(formatted_rows)
    except Exception as e:
        return f"SQL Error: {e}. Please rewrite the query and try again."


@tool
def stage_sql_update(sql_query: str) -> str:
    """
    Use this tool if the user asks you to update, flag, or modify structured database data.
    This tool DOES NOT run the query. It stages it for Human Approval.
    """
    if "UPDATE" not in sql_query.upper() and "DELETE" not in sql_query.upper():
        return "ERROR: This tool is only for UPDATE or DELETE queries."
        
    return json.dumps({"action": "PAUSE_FOR_APPROVAL", "query": sql_query})


@tool
def stage_email_draft(email_address: str, subject: str, body: str) -> str:
    """
    Use this tool to email reports, summaries, or data to a user.
    This DOES NOT send the email immediately. It stages it for Human Approval.
    """
    draft_details = f"To: {email_address} | Subject: {subject} | Body: {body}"
    return json.dumps({"action": "PAUSE_FOR_APPROVAL", "query": draft_details})


# Available tools array
tools = [search_pdf_knowledge_base, execute_sql_read_query, stage_sql_update, stage_email_draft]


# ==========================================
# 3. DEFINE THE GRAPH NODES
# ==========================================

def ai_reasoning_node(state: AgentState):
    """The brain of the agent. It looks at the history and decides which tool to use next."""
    
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0
    )
    
    # FIX 2: Disable parallel tool calling so the agent completes one task before hitting the approval pause
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
    
    # FIX 1: Dynamically fetch all real table names AND their columns to prevent schema guessing
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        schema_details = []
        for table in table_names:
            columns = [col['name'] for col in inspector.get_columns(table)]
            schema_details.append(f"- {table} (Columns: {', '.join(columns)})")
        db_context = "The following tables and columns exist in the PostgreSQL database:\n" + "\n".join(schema_details)
    except Exception:
        db_context = "Database schema unavailable."
    
    # <-- THE UPGRADED MULTI-SOURCE PROMPT -->
    system_msg = HumanMessage(content=f"""You are an advanced Autonomous Enterprise AI Orchestrator.
    Your goal is to fulfill user requests by dynamically combining data from available sources.

    AVAILABLE KNOWLEDGE BASES:
    1. Structured Data (PostgreSQL): Contains uploaded CSVs. 
       {db_context}
       - Use `execute_sql_read_query` to query this data. ONLY use the table and column names explicitly listed above.
    
    2. Unstructured Data (ChromaDB): Contains uploaded PDFs, policies, and text documents.
       - Use `search_pdf_knowledge_base` to query semantic information from these documents.

    WORKFLOW INSTRUCTIONS:
    - If a user asks for a complex report, you are ENCOURAGED to use multiple tools in sequence.
    - If the user asks to modify data, use `stage_sql_update`. Do not guess column names.
    - If the user asks to email or send a report, DO NOT WRITE SQL INSERTS. You MUST synthesize the data you collected and pass it into the `stage_email_draft` tool.
    
    Always synthesize your findings into a clear, professional final response.""")
    
    messages_to_send = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages_to_send)
    
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "human_approval", "__end__"]:
    """Traffic controller: Decides if we need to run a tool, pause for a human, or finish."""
    last_message = state["messages"][-1]
    
    if not last_message.tool_calls:
        return "__end__"
        
    for tool_call in last_message.tool_calls:
        if tool_call["name"] in ["stage_sql_update", "stage_email_draft"]:
            return "human_approval"
            
    return "tools"


def human_approval_node(state: AgentState):
    """This node represents the Human-in-the-Loop pause."""
    last_message = state["messages"][-1]
    dangerous_action = ""
    action_type = "perform a critical action"
    
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "stage_sql_update":
            dangerous_action = tool_call["args"]["sql_query"]
            action_type = "modify the database"
            break
        elif tool_call["name"] == "stage_email_draft":
            email = tool_call["args"]["email_address"]
            sub = tool_call["args"]["subject"]
            dangerous_action = f"Send Email to {email}\nSubject: '{sub}'"
            action_type = "send an email"
            break
            
    return {
        "requires_approval": True,
        "pending_action": dangerous_action,
        "messages": [AIMessage(content=f"⚠️ I need your approval to {action_type}. Requested action: `{dangerous_action}`")]
    }

# ==========================================
# 4. COMPILE THE WORKFLOW GRAPH
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("agent", ai_reasoning_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("human_approval", human_approval_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

agent_app = workflow.compile()