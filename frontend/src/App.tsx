import { type ChangeEvent, type DragEvent, type FormEvent, type ReactNode, useState } from "react";
import "./App.css";

type ConversationMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
};

type AskResponse = { question: string; rewritten_query: string; answer: string; sources: string[] };
type UploadResponse = { success: boolean; filename: string; document_id: string; status: string };
type UploadStatusResponse = { document_id: string; status: "pending" | "processing" | "ready" | "failed" };

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentName, setDocumentName] = useState<string | null>(null);
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [chatError, setChatError] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  function selectFile(file: File | undefined) {
    if (!file) return;
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setSelectedFile(null);
      setUploadError("Please choose a PDF file.");
      return;
    }
    setSelectedFile(file);
    setUploadError("");
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0]);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files[0]);
  }

  async function handleUpload() {
    if (!selectedFile || isUploading) return;
    setIsUploading(true);
    setUploadError("");
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, { method: "POST", body: formData });
      if (!response.ok) throw new Error("upload failed");
      const data: UploadResponse = await response.json();
      if (!data.success) throw new Error("upload failed");

      let status = data.status;
      while (status === "pending" || status === "processing") {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        const statusResponse = await fetch(`${API_BASE_URL}/upload/${data.document_id}/status`);
        if (!statusResponse.ok) throw new Error("upload failed");
        const statusData: UploadStatusResponse = await statusResponse.json();
        status = statusData.status;
      }
      if (status !== "ready") throw new Error("upload failed");

      setDocumentName(data.filename || selectedFile.name);
      setDocumentId(data.document_id);
      setMessages([]);
      setQuestion("");
      setChatError("");
    } catch {
      setUploadError("We couldn't process that PDF. Please try again.");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!documentName || !documentId || !trimmedQuestion || isAsking) return;

    const userMessage: ConversationMessage = { id: crypto.randomUUID(), role: "user", content: trimmedQuestion };
    const history = messages.map(({ role, content }) => ({ role, content }));
    setMessages((currentMessages) => [...currentMessages, userMessage]);
    setQuestion("");
    setChatError("");
    setIsAsking(true);

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmedQuestion, document_id: documentId, history }),
      });
      if (!response.ok) throw new Error(response.status === 502 ? "generation unavailable" : "ask failed");
      const data: AskResponse = await response.json();
      setMessages((currentMessages) => [
        ...currentMessages,
        { id: crypto.randomUUID(), role: "assistant", content: data.answer, sources: data.sources },
      ]);
    } catch (error) {
      setChatError(
        error instanceof Error && error.message === "generation unavailable"
          ? "The answer service is temporarily unavailable. Please try again."
          : "We couldn't answer that question. Please try again."
      );
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand-mark" aria-hidden="true">P</div>
        <div><h1>PDF Assistant</h1><p>Ask questions about your documents.</p></div>
      </header>

      <section className="upload-panel" aria-labelledby="upload-title">
        <div className="section-heading">
          <div><span className="eyebrow">Document</span><h2 id="upload-title">Upload a PDF to begin</h2></div>
          {documentName && <span className="ready-badge">Ready to chat</span>}
        </div>
        <label className={`drop-zone ${isDragging ? "is-dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={handleDrop}>
          <input type="file" accept="application/pdf,.pdf" onChange={handleFileChange} />
          <span className="upload-icon" aria-hidden="true">↑</span>
          <span className="drop-zone-copy"><strong>{selectedFile ? selectedFile.name : "Drop your PDF here"}</strong><span>{selectedFile ? "Choose another file" : "or browse files from your computer"}</span></span>
        </label>
        <div className="upload-actions">
          <p className="upload-help">PDF files only. Uploading a new document starts a fresh conversation.</p>
          <button className="primary-button" type="button" onClick={handleUpload} disabled={!selectedFile || isUploading}>{isUploading ? "Processing document…" : "Process PDF"}</button>
        </div>
        {uploadError && <p className="error-message" role="alert">{uploadError}</p>}
      </section>

      <section className={`chat-panel ${!documentName ? "is-disabled" : ""}`} aria-labelledby="chat-title">
        <div className="section-heading"><div><span className="eyebrow">Conversation</span><h2 id="chat-title">{documentName ?? "Your document chat"}</h2></div></div>
        <div className="message-list" aria-live="polite">
          {!documentName ? <EmptyState icon="⌁" title="Upload a document first">Once it has been processed, you can ask questions grounded in its contents.</EmptyState>
            : messages.length === 0 ? <EmptyState icon="✦" title="What would you like to know?">Ask about details, concepts, or anything else in <strong>{documentName}</strong>.</EmptyState>
              : messages.map((message) => <article key={message.id} className={`message message-${message.role}`}>
                <span className="message-label">{message.role === "user" ? "You" : "PDF Assistant"}</span><p>{message.content}</p>
                {message.role === "assistant" && message.sources && message.sources.length > 0 && <details className="sources"><summary>Sources <span>{message.sources.length}</span></summary><ol>{message.sources.map((source, index) => <li key={index}>{source}</li>)}</ol></details>}
              </article>)}
          {isAsking && <div className="assistant-loading"><span /><span /><span /> Finding an answer</div>}
        </div>
        {chatError && <p className="error-message chat-error" role="alert">{chatError}</p>}
        <form className="question-form" onSubmit={handleAsk}>
          <label className="sr-only" htmlFor="question">Ask a question about the PDF</label>
          <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={documentName ? "Ask a question about this document…" : "Upload a PDF to start asking questions"} rows={2} disabled={!documentName || isAsking} />
          <button className="send-button" type="submit" disabled={!documentName || !question.trim() || isAsking} aria-label="Send question">{isAsking ? "…" : "→"}</button>
        </form>
      </section>
    </main>
  );
}

function EmptyState({ icon, title, children }: { icon: string; title: string; children: ReactNode }) {
  return <div className="empty-state"><span className="empty-icon" aria-hidden="true">{icon}</span><h3>{title}</h3><p>{children}</p></div>;
}

export default App;
