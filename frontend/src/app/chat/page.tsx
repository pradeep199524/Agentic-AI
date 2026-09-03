"use client";

import { useState, useEffect, useRef, useCallback } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
  latency?: string;
  timestamp?: string;
}

interface SessionMeta {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  
  const chatBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/v1/chat/sessions");
      if (res.ok) {
        const data = await res.json();
        setSessions(data.sessions || []);
      }
    } catch (err) {
      console.warn("Could not fetch sessions yet. Backend might be starting.", err);
    }
  }, []);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const handleNewChat = async () => {
    if (loading) return;
    try {
      const res = await fetch("http://127.0.0.1:8000/api/v1/chat/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New Conversation" }),
      });
      if (res.ok) {
        const data = await res.json();
        setSessionId(data.session.session_id);
        setMessages([]);
        fetchSessions();
      }
    } catch (err) {
      console.error("Failed to start new chat:", err);
      setSessionId(null);
      setMessages([]);
    }
  };

  const handleSelectSession = async (id: string) => {
    if (loading || id === sessionId) return;
    setLoadingHistory(true);
    setSessionId(id);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/v1/chat/sessions/${id}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
      }
    } catch (err) {
      console.error("Failed to load session history:", err);
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleDeleteSession = async (idToDelete: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (loading) return;
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/v1/chat/sessions/${idToDelete}`, {
        method: "DELETE",
      });
      if (res.ok) {
        if (sessionId === idToDelete) {
          setSessionId(null);
          setMessages([]);
        }
        fetchSessions();
      }
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userText = input.trim();
    setInput("");
    const startTime = Date.now();

    const userMsg: Message = { role: "user", content: userText };
    const assistantMsgPlaceholder: Message = { role: "assistant", content: "" };

    setMessages((prev) => [...prev, userMsg, assistantMsgPlaceholder]);
    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/v1/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userText,
          session_id: sessionId,
          history: messages.map((m) => ({ role: m.role, content: m.content })),
        }),
      });

      if (!res.ok || !res.body) throw new Error("Failed to connect to streaming API.");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunkText = decoder.decode(value, { stream: true });
        const lines = chunkText.split("\n").filter((line) => line.trim() !== "");

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            
            if (data.session_id) {
              setSessionId(data.session_id);
            }
            
            if (data.token) {
              accumulatedText += data.token;
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  role: "assistant",
                  content: accumulatedText,
                };
                return updated;
              });
            }
          } catch (e) {
            // Partial JSON chunk arrived, ignore
          }
        }
      }

      const totalLatency = ((Date.now() - startTime) / 1000).toFixed(2);
      
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          latency: `${totalLatency}s`,
        };
        return updated;
      });

      fetchSessions();

    } catch (err: any) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: `⚠️ Error: ${err.message || "Failed to stream answer."}`,
        };
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    /* 
      FIX: Using fixed positioning (left-64 matches sidebar width) 
      to break out of the layout's padding box ONLY on the chat page.
    */
    <div className="fixed inset-y-0 right-0 left-64 z-50 flex overflow-hidden font-sans text-gray-800 bg-white">
      
      {/* SIDEBAR */}
      <aside className="w-[300px] shrink-0 bg-gray-50/50 border-r border-gray-200/80 flex flex-col">
        <div className="p-4 border-b border-gray-200/80">
          <button
            onClick={handleNewChat}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 bg-white border border-blue-600 text-blue-600 hover:bg-blue-50 font-medium py-2.5 rounded-lg transition-colors disabled:opacity-50 text-sm shadow-sm"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            New Chat
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-3 space-y-1.5 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-2 mb-3 mt-1">Chat History</h3>
          {sessions.length === 0 ? (
            <p className="text-xs text-gray-400 text-center py-4 italic">No history yet.</p>
          ) : (
            sessions.map((s) => (
              <div
                key={s.session_id}
                onClick={() => handleSelectSession(s.session_id)}
                className={`group flex justify-between items-center p-3 rounded-xl cursor-pointer text-sm transition-all ${
                  s.session_id === sessionId 
                    ? "bg-blue-50 text-blue-900 border border-blue-100" 
                    : "hover:bg-white hover:shadow-sm border border-transparent text-gray-600"
                }`}
              >
                <div className="truncate pr-2 flex-1">
                  <div className="font-medium truncate text-xs">{s.title || "New Chat"}</div>
                  <div className="text-[10px] text-gray-400 mt-1">
                    {new Date(s.updated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
                <button
                  onClick={(e) => handleDeleteSession(s.session_id, e)}
                  className="opacity-0 group-hover:opacity-100 text-gray-300 hover:text-red-500 hover:bg-red-50 p-1.5 rounded-md transition-all"
                  title="Delete"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            ))
          )}
        </div>
      </aside>

      {/* MAIN CHAT AREA */}
      <main className="flex-1 flex flex-col w-full bg-white relative">
        <header className="border-b border-gray-200/80 px-6 py-4 flex justify-between items-center bg-white/80 backdrop-blur-sm absolute top-0 w-full z-10">
          <div>
            <h1 className="text-lg font-bold text-gray-900 leading-tight">AI Enterprise Assistant</h1>
            <p className="text-xs text-gray-500 font-medium">Grounded Retrieval Engine</p>
          </div>
          {sessionId && (
            <span className="text-[10px] bg-gray-100/80 text-gray-500 px-3 py-1.5 rounded-full font-mono border border-gray-200/50">
              {sessionId.slice(0, 8)}
            </span>
          )}
        </header>

        <div className="flex-1 overflow-y-auto pt-24 pb-6 px-4 md:px-6 flex flex-col w-full [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
          <div className="w-full space-y-6">
            {loadingHistory ? (
              <div className="flex justify-center items-center py-20 text-sm text-gray-400 gap-2">
                <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                Loading history...
              </div>
            ) : messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24 text-center">
                <div className="w-16 h-16 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center mb-4 shadow-sm border border-blue-100">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                </div>
                <h2 className="text-lg font-semibold text-gray-800 mb-2">How can I help you today?</h2>
                <p className="text-sm text-gray-500 max-w-sm">
                  Ask any question regarding the documents or datasets indexed in your knowledge base.
                </p>
              </div>
            ) : (
              messages.map((m, idx) => (
                <div key={idx} className={`flex gap-3 md:gap-4 ${m.role === "user" ? "flex-row-reverse" : "flex-row"} w-full`}>
                  <div className="shrink-0 mt-1">
                    {m.role === "user" ? (
                      <div className="w-8 h-8 bg-gray-900 text-white rounded-full flex items-center justify-center text-xs font-bold shadow-sm">U</div>
                    ) : (
                      <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold shadow-sm ring-4 ring-blue-50">AI</div>
                    )}
                  </div>
                  <div className={`max-w-[98%] flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}>
                    <div className={`px-5 py-3.5 text-sm leading-relaxed shadow-sm w-full ${m.role === "user" ? "bg-gray-900 text-white rounded-2xl rounded-tr-sm" : "bg-white border border-gray-200 text-gray-800 rounded-2xl rounded-tl-sm"}`}>
                      <div className="whitespace-pre-wrap font-medium break-words">
                        {m.content || (loading && idx === messages.length - 1 ? (
                          <span className="flex items-center gap-1.5 text-gray-400">
                            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"></span>
                            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
                            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                          </span>
                        ) : "No response generated.")}
                      </div>
                    </div>
                    {m.role === "assistant" && m.latency && (
                      <div className="mt-1.5 text-[10px] text-gray-400 font-medium flex items-center gap-1 ml-1">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {m.latency}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            <div ref={chatBottomRef} />
          </div>
        </div>

        <div className="p-4 bg-white/80 backdrop-blur-md border-t border-gray-200/80 flex justify-center pb-6 px-4 md:px-6 w-full">
          <div className="w-full relative flex items-center">
            <input
              type="text"
              className="w-full border border-gray-300 rounded-xl pl-5 pr-24 py-4 text-gray-800 text-sm focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-50 transition-all bg-white shadow-sm placeholder-gray-400"
              placeholder="Ask a question about your documents or datasets..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              disabled={loading}
            />
            <button
              onClick={handleSend}
              disabled={loading || !input.trim()}
              className="absolute right-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold px-5 py-2.5 rounded-lg text-sm transition-colors disabled:opacity-50 disabled:hover:bg-blue-600 flex items-center gap-2"
            >
              {loading ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              ) : (
                <>
                  Send
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                </>
              )}
            </button>
          </div>
        </div>
      </main>
      
    </div>
  );
}