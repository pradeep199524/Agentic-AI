"use client";
import { useState } from "react";

export default function RetrievalPage() {
  const [query, setQuery] = useState("");
  const [strategy, setStrategy] = useState("hybrid_rerank");
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Point to your FastAPI backend
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query) return;

    setLoading(true);
    setError("");
    setResults([]);

    try {
      const res = await fetch(`${API_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, strategy, top_k: 5 }),
      });

      if (!res.ok) throw new Error("Search request failed. Is the backend running?");
      
      const data = await res.json();
      setResults(data.results);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Retrieval Engine</h1>
        <p className="text-sm text-slate-500 mt-2">
          Test semantic search, keyword search, and cross-encoder reranking (Module 4).
        </p>
      </div>

      {/* Search Controls */}
      <form onSubmit={handleSearch} className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 space-y-5">
        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-1">Natural Language Query</label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., What engineering tools are used for performance?"
            className="w-full p-3 border border-slate-300 rounded-lg text-sm text-slate-800 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            required
          />
        </div>
        
        <div className="flex flex-col sm:flex-row gap-4 items-end">
          <div className="flex-1 w-full">
            <label className="block text-sm font-semibold text-slate-700 mb-1">Retrieval Strategy</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="w-full p-3 border border-slate-300 rounded-lg text-sm text-slate-800 bg-white focus:ring-2 focus:ring-blue-500 outline-none"
            >
              <option value="dense">Dense Vector Search (ChromaDB Embeddings)</option>
              <option value="sparse">Sparse Keyword Search (BM25)</option>
              <option value="hybrid">Hybrid Search (Reciprocal Rank Fusion)</option>
              <option value="hybrid_rerank">Hybrid + Cross-Encoder Reranking (Best for LLMs)</option>
            </select>
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full sm:w-auto bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg text-sm disabled:opacity-50 transition-all shadow-sm"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </form>

      {/* Error Message */}
      {error && (
        <div className="p-4 bg-red-50 text-red-700 rounded-lg text-sm border border-red-200">
          <strong>Error: </strong>{error}
        </div>
      )}

      {/* Results Display */}
      <div className="space-y-4">
        {results.map((res, idx) => {
          // Rerank uses 'rerank_score', others use 'score'
          const score = res.rerank_score !== undefined ? res.rerank_score : res.score;
          
          return (
            <div key={idx} className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 transition-all hover:shadow-md">
              <div className="flex flex-col sm:flex-row justify-between items-start mb-4 gap-2">
                <div>
                  <span className="text-xs font-bold text-blue-700 bg-blue-100 px-3 py-1 rounded-full uppercase tracking-wide">
                    Rank #{idx + 1}
                  </span>
                  <span className="ml-3 text-sm text-slate-600">
                    Source: <span className="font-semibold text-slate-900">{res.metadata?.filename}</span> (Page {res.metadata?.page_number})
                  </span>
                </div>
                <div className="text-xs font-mono bg-slate-100 px-3 py-1 rounded-full text-slate-700 border border-slate-200">
                  Similarity Score: {score.toFixed(4)}
                </div>
              </div>
              
              {/* The Actual Chunk Content */}
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                <p className="text-sm text-slate-700 whitespace-pre-wrap font-serif leading-relaxed">
                  {res.content}
                </p>
              </div>
            </div>
          );
        })}
        
        {/* Empty State */}
        {!loading && results.length === 0 && !error && query && (
          <div className="text-center py-12 bg-white rounded-xl border border-dashed border-slate-300">
            <p className="text-sm text-slate-500">No results found for this query. Try a different strategy or keyword.</p>
          </div>
        )}
      </div>
    </div>
  );
}