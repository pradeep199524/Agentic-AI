"use client";
import { useEffect, useState, useMemo } from "react";

export default function DashboardPage() {
  const [documents, setDocuments] = useState<any[]>([]);
  const [activeDoc, setActiveDoc] = useState<any>(null);
  const [pdfPages, setPdfPages] = useState<any[]>([]);
  
  // FIX: Added 'table: string;' to the state type definition here
  const [csvData, setCsvData] = useState<{ table: string; columns: string[]; records: any[] } | null>(null);
  
  const [loading, setLoading] = useState(false);
  const [visiblePagesCount, setVisiblePagesCount] = useState<number>(10);
  const [docToDelete, setDocToDelete] = useState<number | null>(null);

  // State to hold the user's search query for a specific page number
  const [pageSearchQuery, setPageSearchQuery] = useState<string>("");

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const fetchDocuments = () => {
    fetch(`${API_URL}/documents`)
      .then((res) => res.json())
      .then((data) => setDocuments(data.documents || []))
      .catch((err) => console.error("Error loading documents:", err));
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleSelectDoc = async (doc: any) => {
    setActiveDoc(doc);
    setPdfPages([]);
    setCsvData(null);
    setLoading(true);
    setVisiblePagesCount(10);
    setPageSearchQuery(""); // Reset search when switching documents

    try {
      if (doc.file_type === "pdf") {
        const res = await fetch(`${API_URL}/documents/${doc.id}/pages`);
        const data = await res.json();
        const sortedPages = (data.pages || []).sort((a: any, b: any) => a.page_number - b.page_number);
        setPdfPages(sortedPages);
      } else if (doc.file_type === "csv") {
        const tableName = `csv_data_${doc.filename.split(".")[0].toLowerCase()}`;
        const res = await fetch(`${API_URL}/csv/${tableName}`);
        const data = await res.json();
        setCsvData(data);
      }
    } catch (err) {
      alert("Failed to load document content.");
    }
    setLoading(false);
  };

  const handleDeleteClick = (id: number) => {
    setDocToDelete(id);
  };

  const confirmDelete = async () => {
    if (docToDelete === null) return;
    
    try {
      const res = await fetch(`${API_URL}/documents/${docToDelete}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Delete failed");
      
      if (activeDoc?.id === docToDelete) {
        setActiveDoc(null);
        setPdfPages([]);
        setCsvData(null);
      }
      fetchDocuments();
    } catch (err) {
      alert("Error deleting record");
    } finally {
      setDocToDelete(null);
    }
  };

  // Filter logic. If they typed a number, only show that exact page. Otherwise, show normal sliced array.
  const filteredPages = useMemo(() => {
    if (pageSearchQuery.trim() !== "") {
      const targetPageNum = parseInt(pageSearchQuery, 10);
      return pdfPages.filter(page => page.page_number === targetPageNum);
    }
    return pdfPages.slice(0, visiblePagesCount);
  }, [pdfPages, visiblePagesCount, pageSearchQuery]);

  return (
    <>
      <div className="space-y-6 h-[calc(100vh-8rem)] overflow-y-auto pb-24 pr-2 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Document Repository</h1>
          <p className="text-xs text-slate-500 mt-1">Select any file to inspect extracted content and relational records.</p>
        </div>

        {/* Document Catalog */}
        <div className="border border-slate-200 rounded-lg overflow-hidden divide-y divide-slate-200">
          {documents.length === 0 ? (
            <p className="p-4 text-xs text-slate-500">No documents ingested. Go to Upload to add files.</p>
          ) : (
            documents.map((doc) => (
              <div
                key={doc.id}
                className={`p-3 flex items-center justify-between transition-colors ${
                  activeDoc?.id === doc.id ? "bg-blue-50 border-l-4 border-l-blue-600" : "hover:bg-slate-50"
                }`}
              >
                <div>
                  <p className="text-sm font-semibold text-slate-800">{doc.filename}</p>
                  <div className="flex gap-2 text-xs text-slate-500 mt-0.5">
                    <span className="font-bold uppercase text-blue-600">{doc.file_type}</span>
                    <span>•</span>
                    <span>ID: {doc.id}</span>
                    <span>•</span>
                    <span>{new Date(doc.upload_date).toLocaleDateString()}</span>
                  </div>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => handleSelectDoc(doc)}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-xs font-medium"
                  >
                    Inspect Content
                  </button>
                  <button
                    onClick={() => handleDeleteClick(doc.id)}
                    className="bg-red-50 hover:bg-red-100 text-red-600 border border-red-200 px-3 py-1 rounded text-xs font-medium"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        {loading && <p className="text-xs text-slate-500 py-4">Loading extracted content...</p>}

        {/* PDF Structured View */}
        {activeDoc?.file_type === "pdf" && (
          <div className="border border-slate-200 rounded-lg p-5 bg-white space-y-6">
            
            {/* Header with Search Bar */}
            <div className="border-b border-slate-200 pb-3 flex items-end justify-between">
              <div>
                <h2 className="text-base font-bold text-slate-800">Document Reader</h2>
                <p className="text-xs text-slate-500">Source: {activeDoc.filename} ({pdfPages.length} Pages)</p>
              </div>
              
              {/* Page Number Filter Input */}
              <div className="w-48">
                <label className="block text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Jump to Page</label>
                <input
                  type="number"
                  min="1"
                  max={pdfPages.length}
                  placeholder="e.g. 10"
                  value={pageSearchQuery}
                  onChange={(e) => setPageSearchQuery(e.target.value)}
                  className="w-full px-3 py-1.5 text-sm border border-slate-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>

            {/* If search returns nothing, show a friendly message */}
            {filteredPages.length === 0 && pageSearchQuery !== "" && (
              <p className="text-sm text-slate-500 italic py-4 text-center">
                Page {pageSearchQuery} was not found in this document.
              </p>
            )}

            {/* Map through the filtered array instead of the raw array */}
            {filteredPages.map((page) => (
              <div key={page.id} className="border border-slate-200 rounded-lg p-4 bg-slate-50 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-blue-700 bg-blue-100 px-2 py-0.5 rounded">
                    Page {page.page_number}
                  </span>
                  <span className="text-xs text-slate-400">
                    {page.content.character_count} Characters
                  </span>
                </div>

                <div className="space-y-3">
                  <p className="text-xs font-semibold text-slate-700">Page Content:</p>
                  <div className="bg-white p-5 rounded border border-slate-200">
                    {page.content.text || (page.content.paragraphs && page.content.paragraphs.length > 0) ? (
                      <p className="text-sm leading-relaxed text-slate-800 whitespace-pre-wrap">
                        {page.content.text || page.content.paragraphs.join('\n\n')}
                      </p>
                    ) : (
                      <p className="text-xs text-slate-400 italic">No text extracted on this page.</p>
                    )}
                  </div>
                </div>

                {page.content.tables?.length > 0 && (
                  <div className="space-y-2 pt-2">
                    <p className="text-xs font-semibold text-slate-700">Extracted Tables:</p>
                    {page.content.tables.map((table: any[], tIdx: number) => (
                      <div key={tIdx} className="overflow-x-auto border border-slate-200 rounded bg-white">
                        <table className="min-w-full text-xs text-left divide-y divide-slate-200">
                          <tbody>
                            {table.map((row: any[], rIdx: number) => (
                              <tr key={rIdx} className={rIdx === 0 ? "bg-slate-100 font-semibold text-slate-800" : "divide-x divide-slate-100"}>
                                {row.map((cell: any, cIdx: number) => (
                                  <td key={cIdx} className="p-2 border-r last:border-r-0 border-slate-200">
                                    {cell || "-"}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {/* Only show load more button if NOT searching and there are more pages */}
            {pdfPages.length > visiblePagesCount && pageSearchQuery === "" && (
              <div className="flex justify-center pt-4">
                <button
                  onClick={() => setVisiblePagesCount((prev) => prev + 10)}
                  className="bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold py-2 px-6 rounded-full text-sm transition-colors border border-slate-300 shadow-sm"
                >
                  Load Next 10 Pages ({pdfPages.length - visiblePagesCount} remaining)
                </button>
              </div>
            )}
          </div>
        )}

        {/* CSV Tabular View */}
        {activeDoc?.file_type === "csv" && csvData && (
          <div className="border border-slate-200 rounded-lg p-5 bg-white space-y-4">
            <div className="border-b border-slate-200 pb-3">
              <h2 className="text-base font-bold text-slate-800">Dynamic CSV Relational View</h2>
              <p className="text-xs text-slate-500">SQL Table: {csvData.table}</p>
            </div>

            <div className="overflow-x-auto border border-slate-200 rounded max-h-96 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
              <table className="min-w-full text-xs text-left divide-y divide-slate-200">
                <thead className="bg-slate-100 sticky top-0">
                  <tr>
                    {csvData.columns.map((col) => (
                      <th key={col} className="p-2.5 font-bold text-slate-700 border-r last:border-r-0 border-slate-200">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 bg-white">
                  {csvData.records.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-50">
                      {csvData.columns.map((col) => (
                        <td key={col} className="p-2.5 border-r last:border-r-0 border-slate-200 text-slate-600 whitespace-nowrap">
                          {String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Custom Confirmation Modal Overlay */}
      {docToDelete !== null && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-md w-full shadow-2xl border border-slate-200 transform transition-all">
            <h3 className="text-lg font-bold text-slate-900 mb-2">Are you sure you want to delete?</h3>
            <p className="text-sm text-slate-500 mb-6 leading-relaxed">
              This will permanently remove the document, its extracted records, and the source file from the database. This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setDocToDelete(null)}
                className="px-4 py-2 text-sm font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors shadow-sm"
              >
                Yes, Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}