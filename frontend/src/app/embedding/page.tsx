"use client";

import { useState, useEffect, useCallback } from "react";

interface FileItem {
  filename: string;
  type: string;
}

export default function ChunkingAndEmbeddingPage() {
  const [strategy, setStrategy] = useState("semantic");
  const [loading, setLoading] = useState(false);
  const [fetchingFiles, setFetchingFiles] = useState(true);
  const [availableFiles, setAvailableFiles] = useState<FileItem[]>([]);
  
  // CHANGED: Use an array for multiple selected files
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [result, setResult] = useState<{ type: "success" | "error"; message: string } | null>(null);

  const fetchFiles = useCallback(async () => {
    setFetchingFiles(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/files");
      if (res.ok) {
        const data = await res.json();
        setAvailableFiles(data.files || []);
      }
    } catch (err) {
      console.error("Failed to fetch files", err);
    } finally {
      setFetchingFiles(false);
    }
  }, []);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  const clearSelection = () => setSelectedFiles([]);
  
  // NEW: Select all files function
  const selectAll = () => {
    setSelectedFiles(availableFiles.map(file => file.filename));
  };

  const handleRunPipeline = async () => {
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/embed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          strategy,
          // CHANGED: Pass the array of selected files directly
          filenames: selectedFiles 
        }),
      });

      const data = await res.json();

      if (res.ok) {
        setResult({ type: "success", message: data.message });
      } else {
        setResult({ type: "error", message: data.detail || "Failed to run pipeline." });
      }
    } catch (err: any) {
      setResult({ type: "error", message: "Network error: Could not connect to the backend." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col w-full h-full relative overflow-y-auto bg-slate-50">
      <header className="border-b border-gray-200 px-8 py-6 bg-white shadow-sm sticky top-0 z-10">
        <h1 className="text-2xl font-bold text-gray-900 leading-tight">Data Chunking & Embedding</h1>
        <p className="text-sm text-gray-500 font-medium mt-1">Select specific files and configure how they are chunked into the vector database.</p>
      </header>

      <div className="p-8 max-w-5xl mx-auto w-full mt-4 flex flex-col md:flex-row gap-6">
        
        {/* LEFT COLUMN: FILE SELECTION */}
        <div className="w-full md:w-1/2 flex flex-col gap-4">
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden flex-1 flex flex-col">
            <div className="p-5 border-b border-gray-200 bg-gray-50/50 flex justify-between items-center">
              <div>
                <h2 className="text-md font-semibold text-gray-900">1. Select Target Files</h2>
                <p className="text-xs text-gray-500 mt-1">Leave empty to rebuild entire database.</p>
              </div>
              <div className="flex gap-3 items-center">
                <button onClick={fetchFiles} className="text-xs text-gray-600 hover:text-blue-600 transition-colors flex items-center gap-1" title="Refresh Files">
                  <svg xmlns="http://www.w3.org/2000/svg" className={`h-4 w-4 ${fetchingFiles ? "animate-spin" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
                <span className="text-gray-300">|</span>
                {/* CHANGED: Added Select All button */}
                <button onClick={selectAll} className="text-xs text-gray-500 hover:text-blue-600 transition-colors font-medium">Select All</button>
                <span className="text-gray-300">|</span>
                <button onClick={clearSelection} className="text-xs text-gray-500 hover:text-red-500 transition-colors font-medium">Clear</button>
              </div>
            </div>
            
            <div className="p-3 max-h-[400px] overflow-y-auto flex-1">
              {fetchingFiles ? (
                <div className="p-8 text-center text-sm text-gray-500 animate-pulse">Loading documents...</div>
              ) : availableFiles.length === 0 ? (
                <div className="p-8 text-center flex flex-col items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-gray-300 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-sm text-gray-500 font-medium">No processed documents found.</p>
                  <p className="text-xs text-gray-400 mt-1">Go to 'Upload Data' to add files first.</p>
                </div>
              ) : (
                <div className="space-y-1">
                  {availableFiles.map((file) => {
                    const isSelected = selectedFiles.includes(file.filename);
                    return (
                      <label 
                        key={file.filename} 
                        className={`flex items-center p-3 hover:bg-blue-50/50 rounded-xl cursor-pointer transition-all border ${
                          isSelected ? "border-blue-300 bg-blue-50/30 shadow-sm" : "border-transparent"
                        }`}
                      >
                        {/* CHANGED: Changed input type to checkbox */}
                        <input 
                          type="checkbox" 
                          checked={isSelected}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedFiles([...selectedFiles, file.filename]);
                            } else {
                              setSelectedFiles(selectedFiles.filter(name => name !== file.filename));
                            }
                          }}
                          className="w-4 h-4 rounded text-blue-600 border-gray-300 focus:ring-blue-500"
                        />
                        <div className="ml-3 flex-1 overflow-hidden">
                          <span className={`block text-sm font-medium truncate ${isSelected ? "text-blue-900" : "text-gray-800"}`}>
                            {file.filename}
                          </span>
                        </div>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>
            {selectedFiles.length > 0 && (
              <div className="p-3 bg-blue-50 border-t border-blue-100 text-xs font-semibold text-blue-800 text-center">
                {selectedFiles.length} file(s) selected for targeted re-embedding.
              </div>
            )}
          </div>
        </div>

        {/* RIGHT COLUMN: STRATEGY & EXECUTION */}
        <div className="w-full md:w-1/2 flex flex-col gap-4">
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="p-5 border-b border-gray-200 bg-gray-50/50">
              <h2 className="text-md font-semibold text-gray-900">2. Choose Chunking Strategy</h2>
            </div>
            
            <div className="p-5 space-y-3">
              <label className={`flex p-4 border rounded-xl cursor-pointer transition-all ${ strategy === "semantic" ? "border-blue-600 bg-blue-50/30 ring-1 ring-blue-600" : "hover:border-gray-300" }`}>
                <input type="radio" value="semantic" checked={strategy === "semantic"} onChange={(e) => setStrategy(e.target.value)} className="mt-0.5 w-4 h-4 text-blue-600 focus:ring-blue-500" />
                <div className="ml-3">
                  <span className="block text-sm font-semibold text-gray-900">Semantic (Recommended)</span>
                  <span className="block text-xs text-gray-500 mt-1"><strong>PDFs:</strong> Splits by topic/meaning.<br/><strong>CSVs:</strong> Merges rows into semantic clusters.</span>
                </div>
              </label>

              <label className={`flex p-4 border rounded-xl cursor-pointer transition-all ${ strategy === "structured" ? "border-blue-600 bg-blue-50/30 ring-1 ring-blue-600" : "hover:border-gray-300" }`}>
                <input type="radio" value="structured" checked={strategy === "structured"} onChange={(e) => setStrategy(e.target.value)} className="mt-0.5 w-4 h-4 text-blue-600 focus:ring-blue-500" />
                <div className="ml-3">
                  <span className="block text-sm font-semibold text-gray-900">Structured</span>
                  <span className="block text-xs text-gray-500 mt-1"><strong>PDFs:</strong> Splits by paragraph breaks.<br/><strong>CSVs:</strong> Chunks strictly row-by-row.</span>
                </div>
              </label>

              <label className={`flex p-4 border rounded-xl cursor-pointer transition-all ${ strategy === "fixed" ? "border-blue-600 bg-blue-50/30 ring-1 ring-blue-600" : "hover:border-gray-300" }`}>
                <input type="radio" value="fixed" checked={strategy === "fixed"} onChange={(e) => setStrategy(e.target.value)} className="mt-0.5 w-4 h-4 text-blue-600 focus:ring-blue-500" />
                <div className="ml-3">
                  <span className="block text-sm font-semibold text-gray-900">Fixed Token</span>
                  <span className="block text-xs text-gray-500 mt-1">Rigidly cuts text every 500 tokens (applies to both PDFs and combined CSV data).</span>
                </div>
              </label>
            </div>

            <div className="p-5 bg-gray-50/50 border-t border-gray-200">
              <button
                onClick={handleRunPipeline}
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-xl text-sm transition-colors disabled:opacity-50 flex justify-center items-center gap-2 shadow-sm"
              >
                {loading ? (
                  <><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div> Processing...</>
                ) : (
                  // CHANGED: Update button text to reflect multiple selections
                  selectedFiles.length > 0 ? `Re-Embed ${selectedFiles.length} File(s)` : "Rebuild Entire Knowledge Base"
                )}
              </button>
            </div>
          </div>

          {result && (
            <div className={`p-4 rounded-xl border flex items-start gap-3 shadow-sm ${ result.type === "success" ? "bg-green-50 border-green-200 text-green-800" : "bg-red-50 border-red-200 text-red-800" }`}>
              {result.type === "success" ? (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 shrink-0 mt-0.5 text-green-600" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 shrink-0 mt-0.5 text-red-600" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              )}
              <div>
                <h3 className="text-sm font-bold">{result.type === "success" ? "Success" : "Failed"}</h3>
                <p className="text-xs mt-1">{result.message}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}