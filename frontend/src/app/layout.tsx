import "./globals.css";
import Link from "next/link";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-100 text-slate-900 font-sans min-h-screen flex">
        {/* Left Sidebar */}
        <aside className="w-64 bg-slate-900 text-white min-h-screen flex flex-col p-5 border-r border-slate-800 shrink-0">
          <div className="mb-8">
            <h1 className="text-lg font-bold text-white tracking-wide">Enterprise AI Pipeline</h1>
            <p className="text-xs text-slate-400 mt-1">Data Ingestion Hub</p>
          </div>

          <nav className="flex flex-col gap-2 text-xs font-semibold">
            <Link
              href="/dashboard"
              className="flex items-center gap-3 px-3 py-2.5 rounded text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
            >
              Dashboard & Reader
            </Link>
            
            <Link
              href="/upload"
              className="flex items-center gap-3 px-3 py-2.5 rounded text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
            >
              Upload Data (PDF/CSV)
            </Link>

            {/* --- NEW: Chunking & Embedding Link --- */}
            <Link
              href="/embedding"
              className="flex items-center gap-3 px-3 py-2.5 rounded text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
            >
              Data Chunking & Embedding
            </Link>

            <Link
              href="/retrieval"
              className="flex items-center gap-3 px-3 py-2.5 rounded text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
            >
              Retrieval Engine
            </Link>

            <div className="my-2 border-t border-slate-700"></div>

            {/* --- AI Chatbot Link --- */}
            <Link
              href="/chat"
              className="flex items-center gap-3 px-3 py-2.5 rounded text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
            >
              AI Assistant Chat
            </Link>

            {/* --- NEW: Agentic Workflow Link --- */}
            <Link
              href="/agent"
              className="flex items-center gap-3 px-3 py-2.5 rounded text-blue-400 bg-blue-900/20 hover:bg-slate-800 hover:text-blue-300 transition-colors border border-blue-900/50"
            >
              Agentic Workflow (Mod 6)
            </Link>
          </nav>
        </aside>

        {/* Content Area */}
        <main className="flex-1 p-8 overflow-y-auto">
          <div className="max-w-5xl mx-auto bg-white shadow-sm rounded-lg border border-slate-200 p-6">
            {children}
          </div>
        </main>
      </body>
    </html>
  );
}