"use client";

import { useState, useEffect, useRef } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
  latency?: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll when messages update
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userText = input.trim();
    setInput("");
    const startTime = Date.now();

    // 1. Add user message and an empty placeholder for the assistant's streaming response
    const userMsg: Message = { role: "user", content: userText };
    const assistantMsgPlaceholder: Message = { role: "assistant", content: "" };

    setMessages((prev) => [...prev, userMsg, assistantMsgPlaceholder]);
    setLoading(true);

    try {
      // 2. Point to the new /stream endpoint
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

      // 3. Read the stream chunk-by-chunk
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode the incoming byte chunk into text
        const chunkText = decoder.decode(value, { stream: true });
        
        // Split by lines (in case multiple JSON objects arrive at once)
        const lines = chunkText.split("\n").filter((line) => line.trim() !== "");

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            
            if (data.session_id) setSessionId(data.session_id);
            
            if (data.token) {
              accumulatedText += data.token;
              
              // Update the UI in real-time
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
            // Partial JSON chunk arrived, ignore and wait for the rest
          }
        }
      }

      // 4. Calculate total latency once stream finishes
      const totalLatency = ((Date.now() - startTime) / 1000).toFixed(2);
      
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          latency: `${totalLatency}s`,
        };
        return updated;
      });

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
    <div className="flex flex-col h-[75vh] max-w-4xl mx-auto font-sans text-gray-800">
      
      <header className="border-b pb-3 mb-4 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">AI Enterprise Assistant</h1>
        </div>
        {sessionId && (
          <span className="text-xs bg-gray-100 text-gray-600 px-2.5 py-1 rounded font-mono">
            Session: {sessionId.slice(0, 8)}...
          </span>
        )}
      </header>

      {/* Message Feed - SCROLLBAR HIDDEN HERE */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-2 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
        {messages.length === 0 ? (
          <div className="text-center py-24 text-gray-400">
            Ask any question regarding the documents or datasets indexed in your knowledge base.
          </div>
        ) : (
          messages.map((m, idx) => (
            <div
              key={idx}
              className={`flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-lg p-4 text-sm leading-relaxed ${
                  m.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-900 border border-gray-200"
                }`}
              >
                {/* Show "Thinking..." if streaming hasn't started yet */}
                <div className="whitespace-pre-wrap">
                  {m.content || (loading && idx === messages.length - 1 ? "Thinking..." : "")}
                </div>
                
                {/* --- RESPONSE TIME BLOCK --- */}
                {m.role === "assistant" && m.latency && (
                  <div className="mt-2 text-[10px] text-gray-400 text-right opacity-80 border-t border-gray-200 pt-1">
                    ⏱️ {m.latency}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={chatBottomRef} />
      </div>

      {/* Input Form */}
      <div className="pt-4 border-t flex gap-2">
        <input
          type="text"
          className="flex-1 border border-gray-300 rounded-lg px-4 py-3 text-black text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Ask a question about your documents or datasets..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-3 rounded-lg text-sm transition-colors disabled:opacity-50"
        >
          {loading ? "Streaming..." : "Send"}
        </button>
      </div>
    </div>
  );
}