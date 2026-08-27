"use client";
import { useState, useEffect } from "react";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  
  // Upgraded status state to track exact background pipeline stages
  const [status, setStatus] = useState({ stage: "idle", message: "", error: false });

  // Dynamically grab the API URL from .env.local
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // NEW: Auto-reset the banner after 5 seconds of success or failure
  useEffect(() => {
    if (status.stage === "complete" || status.stage === "error") {
      const timer = setTimeout(() => {
        setStatus({ stage: "idle", message: "", error: false });
      }, 5000); // 5000 milliseconds = 5 seconds
      
      // Cleanup the timer if the component unmounts or status changes before 5s
      return () => clearTimeout(timer);
    }
  }, [status.stage]);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    const isPdf = file.name.toLowerCase().endsWith(".pdf");
    const isCsv = file.name.toLowerCase().endsWith(".csv");

    if (!isPdf && !isCsv) {
      setStatus({ stage: "error", message: "Please select a valid .pdf or .csv file.", error: true });
      return;
    }

    const endpoint = isPdf ? "/ingest/pdf" : "/ingest/csv";
    
    // STATE 1: Uploading the file to the server
    setStatus({ stage: "uploading", message: `Uploading ${isPdf ? "PDF" : "CSV"} to server...`, error: false });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");

      // STATE 2: Upload done, begin polling for background extraction completion
      setStatus({ stage: "extracting", message: "Upload successful! Extracting data in the background. Please wait...", error: false });
      const fileName = file.name;
      setFile(null); // Clear the file input visually

      // POLLING MECHANISM: Ping database every 3 seconds to see if extraction is done
      let attempts = 0;
      const pollInterval = setInterval(async () => {
        attempts++;
        
        // Safety timeout (stops checking after 2 minutes)
        if (attempts > 40) { 
          clearInterval(pollInterval);
          setStatus({ stage: "error", message: "Extraction is taking too long or failed. Check backend logs.", error: true });
          return;
        }

        try {
          const checkRes = await fetch(`${API_URL}/documents`);
          const checkData = await checkRes.json();
          
          // Check if our newly uploaded file has appeared in the DB yet
          const isFinished = checkData.documents?.some((doc: any) => doc.filename === fileName);
          
          if (isFinished) {
            clearInterval(pollInterval);
            // STATE 3: Complete
            setStatus({ stage: "complete", message: `✨ Extraction Complete! "${fileName}" has been fully processed. Check the Dashboard.`, error: false });
          }
        } catch (err) {
          // Ignore network blips during polling
        }
      }, 3000);

    } catch (err: any) {
      setStatus({ stage: "error", message: err.message, error: true });
    }
  };

  // Helper to change banner color based on current stage
  const getBannerStyle = () => {
    if (status.error) return "bg-red-50 text-red-700 border-red-200";
    if (status.stage === "complete") return "bg-green-50 text-green-700 border-green-200";
    if (status.stage === "uploading" || status.stage === "extracting") return "bg-blue-50 text-blue-700 border-blue-200 animate-pulse";
    return "";
  };

  return (
    <div className="max-w-xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-bold text-slate-900">Upload Data Source</h1>
        <p className="text-xs text-slate-500 mt-1">Upload unstructured PDFs or structured CSV files for ingestion.</p>
      </div>

      <form onSubmit={handleUpload} className="space-y-4 border border-dashed border-slate-300 rounded-lg p-6 text-center bg-slate-50">
        <input
          type="file"
          accept=".pdf,.csv"
          onChange={(e) => {
            setFile(e.target.files?.[0] || null);
            // Reset status when user selects a new file
            setStatus({ stage: "idle", message: "", error: false });
          }}
          className="block w-full text-xs text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />

        {file && (
          <p className="text-xs text-slate-600 font-medium">
            Selected: <span className="text-blue-600">{file.name}</span> ({file.name.endsWith(".pdf") ? "PDF" : "CSV"})
          </p>
        )}

        <button
          type="submit"
          disabled={!file || status.stage === "uploading" || status.stage === "extracting"}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 rounded text-xs disabled:opacity-50 transition-colors"
        >
          {status.stage === "uploading" ? "Uploading..." : 
           status.stage === "extracting" ? "Extracting Data..." : 
           "Ingest Document"}
        </button>
      </form>

      {status.message && (
        <div className={`p-3 rounded text-xs border ${getBannerStyle()}`}>
          {status.message}
        </div>
      )}
    </div>
  );
}