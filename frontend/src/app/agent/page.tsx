"use client";

import React, { useState } from "react";

interface TraceItem {
  type: "tool_call" | "tool_result";
  tool?: string;
  args?: Record<string, any>;
  output?: string;
}

export default function AgenticWorkflowPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "running" | "awaiting_approval" | "completed" | "rejected" | "error">("idle");
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [trace, setTrace] = useState<TraceItem[]>([]);
  const [finalResponse, setFinalResponse] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

  // Trigger the initial agent workflow
  const handleExecute = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    setStatus("running");
    setErrorMessage(null);
    setFinalResponse(null);
    setTrace([]);
    setPendingAction(null);

    try {
      const res = await fetch(`${API_BASE}/api/v1/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, session_id: sessionId || undefined }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Agent workflow execution failed.");
      }

      const data = await res.json();
      setSessionId(data.session_id);
      setTrace(data.trace || []);

      if (data.status === "awaiting_approval") {
        setStatus("awaiting_approval");
        setPendingAction(data.pending_action);
        setFinalResponse(data.message);
      } else {
        setStatus("completed");
        setFinalResponse(data.response);
      }
    } catch (err: any) {
      setStatus("error");
      setErrorMessage(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  // Human-in-the-Loop decision (Approve or Reject)
  const handleDecision = async (approved: boolean) => {
    if (!sessionId) return;
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/v1/agent/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, approved }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to submit approval decision.");
      }

      const data = await res.json();
      if (data.trace && data.trace.length > 0) {
        setTrace((prev) => [...prev, ...data.trace]);
      }
      setStatus(data.status);
      setFinalResponse(data.response);
      setPendingAction(null);
    } catch (err: any) {
      setStatus("error");
      setErrorMessage(err.message || "Failed during approval execution.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 min-h-screen bg-slate-50 text-slate-900 p-8">
      {/* Header */}
      <div className="max-w-6xl mx-auto mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">
              Module 6: Autonomous Agent Orchestrator
            </h1>
            <p className="text-sm text-slate-500 mt-1">
              Multi-Agent planning, dynamic database execution, and Human-in-the-Loop approval workflows.
            </p>
          </div>
          <span
            className={`px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider ${
              status === "awaiting_approval"
                ? "bg-amber-100 text-amber-800 border border-amber-300 animate-pulse"
                : status === "completed"
                ? "bg-emerald-100 text-emerald-800 border border-emerald-300"
                : status === "running"
                ? "bg-blue-100 text-blue-800 border border-blue-300"
                : "bg-slate-200 text-slate-700"
            }`}
          >
            Status: {status.replace("_", " ")}
          </span>
        </div>
      </div>

      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Columns: Input and Final Response */}
        <div className="lg:col-span-2 space-y-6">
          {/* Query Box */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
            <form onSubmit={handleExecute} className="space-y-4">
              <label className="block text-sm font-medium text-slate-700">
                Agent Goal or Task Command
              </label>
              
              {/* UPDATED TEXTAREA: Added onKeyDown for Enter-to-submit */}
              <textarea
                rows={3}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleExecute();
                  }
                }}
                placeholder="e.g. Which region handled the highest volume of operational requests?"
                className="w-full rounded-lg border border-slate-300 p-3 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
              />
              
              <div className="flex justify-between items-center">
                <div className="flex gap-2 flex-wrap">
                  
                  {/* BUTTON 1: PDF Knowledge Base Question */}
                  <button
                    type="button"
                    onClick={() =>
                      setQuery("Based on the Q2 operations report, which region handled the highest volume of requests and which had the most exceptions?")
                    }
                    className="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-2.5 py-1.5 rounded border border-slate-200"
                  >
                    📄 Test PDF Query
                  </button>
                  
                  {/* BUTTON 2: SQL Update on your Company Performance Dataset */}
                  <button
                    type="button"
                    onClick={() =>
                      setQuery("Find all companies in the performance dataset with revenue under 50,000 and update their status to 'REVIEW'")
                    }
                    className="text-xs bg-amber-50 hover:bg-amber-100 text-amber-800 px-2.5 py-1.5 rounded border border-amber-200"
                  >
                    ⚠️ Test SQL Update
                  </button>
                  
                  {/* BUTTON 3: Multi-Source Analytics & Email Task */}
                  <button
                    type="button"
                    onClick={() =>
                      setQuery("Find the top 3 performing companies by revenue from the database, cross-reference our technology infrastructure limits from the PDF, and email the report to boss@company.com")
                    }
                    className="text-xs bg-indigo-50 hover:bg-indigo-100 text-indigo-800 px-2.5 py-1.5 rounded border border-indigo-200 font-medium"
                  >
                    📧 Test Email HITL
                  </button>
                  
                </div>
                <button
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-5 py-2 rounded-lg text-sm transition disabled:opacity-50 shrink-0 ml-4"
                >
                  {loading ? "Orchestrating..." : "Execute Agent"}
                </button>
              </div>
            </form>
          </div>

          {/* HUMAN-IN-THE-LOOP APPROVAL CARD */}
          {status === "awaiting_approval" && (
            <div className="bg-amber-50 border-2 border-amber-400 rounded-xl p-6 shadow-sm">
              <div className="flex items-start gap-4">
                <span className="text-2xl">⚠️</span>
                <div className="flex-1">
                  <h3 className="text-base font-bold text-amber-950">
                    Human-in-the-Loop Checkpoint Required
                  </h3>
                  <p className="text-sm text-amber-900 mt-1">
                    The autonomous agent staged a critical action (database modification or outgoing email). Execution is currently frozen awaiting your decision.
                  </p>
                  {pendingAction && (
                    <div className="mt-3 bg-white border border-amber-300 rounded p-3 font-mono text-xs text-slate-800 overflow-x-auto whitespace-pre-wrap">
                      {pendingAction}
                    </div>
                  )}
                  <div className="mt-4 flex gap-3">
                    <button
                      onClick={() => handleDecision(true)}
                      disabled={loading}
                      className="bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg text-sm font-semibold shadow-sm transition"
                    >
                      ✓ Approve & Execute Action
                    </button>
                    <button
                      onClick={() => handleDecision(false)}
                      disabled={loading}
                      className="bg-rose-600 hover:bg-rose-700 text-white px-4 py-2 rounded-lg text-sm font-semibold shadow-sm transition"
                    >
                      ✕ Reject Action
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Error Message */}
          {errorMessage && (
            <div className="bg-rose-50 border border-rose-300 text-rose-800 p-4 rounded-xl text-sm">
              <strong>Execution Error:</strong> {errorMessage}
            </div>
          )}

          {/* Final Agent Synthesis */}
          {finalResponse && (
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 mb-3">
                Final Synthesis & Agent Response
              </h2>
              <div className="prose prose-sm text-slate-800 whitespace-pre-wrap leading-relaxed">
                {finalResponse}
              </div>
            </div>
          )}
        </div>

        {/* Right 1 Column: Trace / Observability Inspector */}
        <div className="space-y-4">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
            <h2 className="text-sm font-bold text-slate-800 flex items-center justify-between">
              <span>Execution Trace & Tools</span>
              <span className="text-xs text-slate-400 font-normal">
                {trace.length} action(s)
              </span>
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              Step-by-step audit of reasoning, tool calls, and data returns.
            </p>

            <div className="mt-4 space-y-3 max-h-[600px] overflow-y-auto pr-1">
              {trace.length === 0 ? (
                <div className="text-center py-10 text-xs text-slate-400 border border-dashed border-slate-200 rounded-lg">
                  No active tool calls yet. Run a prompt to inspect agent actions.
                </div>
              ) : (
                trace.map((item, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg border text-xs ${
                      item.type === "tool_call"
                        ? "bg-indigo-50/70 border-indigo-200 text-indigo-950"
                        : "bg-slate-50 border-slate-200 text-slate-800"
                    }`}
                  >
                    <div className="flex items-center justify-between font-semibold mb-1">
                      <span className="uppercase tracking-wider">
                        {item.type === "tool_call" ? "⚙️ Tool Call" : "📥 Tool Result"}
                      </span>
                      {item.tool && (
                        <span className="font-mono bg-white px-1.5 py-0.5 rounded border text-[11px]">
                          {item.tool}
                        </span>
                      )}
                    </div>
                    {item.args && (
                      <pre className="mt-1 bg-white p-2 rounded border border-indigo-100 font-mono text-[11px] overflow-x-auto whitespace-pre-wrap">
                        {JSON.stringify(item.args, null, 2)}
                      </pre>
                    )}
                    {item.output && (
                      <pre className="mt-1 bg-white p-2 rounded border border-slate-200 font-mono text-[11px] overflow-x-auto whitespace-pre-wrap max-h-32">
                        {item.output}
                      </pre>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}