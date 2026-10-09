import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import {
  ApiError,
  type DocumentInfo,
  type DocumentType,
  type HistoryMessage,
  type UploadPhase,
  streamAnswer,
  uploadPdf,
} from "./api";
import { ChatScreen } from "./components/ChatScreen";
import { UploadScreen } from "./components/UploadScreen";
import type { AssistantMessage, ChatMessage } from "./types";

const TYPE_LABELS: Record<DocumentType, string> = {
  prose: "Document",
  slides: "Slides",
  resume: "Resume",
};

// Earlier turns sent with each question. Fewer turns means a shorter prompt
// and a faster answer; follow-ups rarely need more than the last few.
const HISTORY_MESSAGES = 6;

const OFFLINE_MESSAGE = "We couldn't reach the server. Check that the backend is running.";

function toHistory(messages: ChatMessage[]): HistoryMessage[] {
  return messages
    .filter((message) => message.role === "user" || message.status === "done")
    .slice(-HISTORY_MESSAGES)
    .map(({ role, content }) => ({ role, content }));
}

function App() {
  const [doc, setDoc] = useState<DocumentInfo | null>(null);
  const [uploadPhase, setUploadPhase] = useState<UploadPhase | null>(null);
  const [uploadFileName, setUploadFileName] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isAnswering, setIsAnswering] = useState(false);

  // Callbacks read the latest messages through a ref so they stay stable and
  // memoized messages are not re-rendered on every streamed token.
  const messagesRef = useRef(messages);
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const updateAnswer = useCallback(
    (id: string, patch: Partial<AssistantMessage> | ((message: AssistantMessage) => Partial<AssistantMessage>)) => {
      setMessages((current) =>
        current.map((message) =>
          message.id === id && message.role === "assistant"
            ? { ...message, ...(typeof patch === "function" ? patch(message) : patch) }
            : message,
        ),
      );
    },
    [],
  );

  const send = useCallback(
    async (question: string) => {
      if (!doc || abortRef.current) return;

      const history = toHistory(messagesRef.current);
      const answerId = crypto.randomUUID();
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: "user", content: question },
        { id: answerId, role: "assistant", content: "", sources: [], status: "searching", question },
      ]);

      const controller = new AbortController();
      abortRef.current = controller;
      setIsAnswering(true);

      // Batch streamed tokens into one state update per animation frame.
      let pending = "";
      let frame = 0;
      const flush = () => {
        frame = 0;
        if (!pending) return;
        const text = pending;
        pending = "";
        updateAnswer(answerId, (message) => ({ content: message.content + text, status: "streaming" }));
      };

      try {
        const answer = await streamAnswer(
          { question, documentId: doc.id, history },
          {
            onSources: (sources) => updateAnswer(answerId, { sources, status: "streaming" }),
            onToken: (text) => {
              pending += text;
              if (!frame) frame = window.requestAnimationFrame(flush);
            },
          },
          controller.signal,
        );
        window.cancelAnimationFrame(frame);
        updateAnswer(answerId, { content: answer, status: "done" });
      } catch (error) {
        window.cancelAnimationFrame(frame);
        flush();
        if (error instanceof DOMException && error.name === "AbortError") {
          updateAnswer(answerId, { status: "stopped" });
        } else {
          updateAnswer(answerId, {
            status: "error",
            error: error instanceof ApiError ? error.message : OFFLINE_MESSAGE,
          });
        }
      } finally {
        abortRef.current = null;
        setIsAnswering(false);
      }
    },
    [doc, updateAnswer],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  async function handleUpload(file: File) {
    setUploadError("");
    setUploadFileName(file.name);
    try {
      const info = await uploadPdf(file, setUploadPhase);
      setMessages([]);
      setDoc(info);
    } catch (error) {
      setUploadError(error instanceof ApiError ? error.message : OFFLINE_MESSAGE);
    } finally {
      setUploadPhase(null);
    }
  }

  function startOver() {
    abortRef.current?.abort();
    setDoc(null);
    setMessages([]);
    setUploadError("");
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
              <path d="M14 3v5h5M9 13h6M9 17h4" />
            </svg>
          </span>
          <span className="brand-name">PDF Assistant</span>
        </div>

        {doc && (
          <div className="doc-bar">
            <div className="doc-chip" title={doc.name}>
              <span className="doc-name">{doc.name}</span>
              <span className="doc-meta">
                {TYPE_LABELS[doc.documentType ?? "prose"]}
                {doc.pages ? ` · ${doc.pages} ${doc.pages === 1 ? "page" : "pages"}` : ""}
              </span>
            </div>
            <button type="button" className="secondary-button" onClick={startOver}>
              New document
            </button>
          </div>
        )}
      </header>

      <main className="workspace">
        {doc ? (
          <ChatScreen doc={doc} messages={messages} isAnswering={isAnswering} onSend={send} onStop={stop} />
        ) : (
          <UploadScreen phase={uploadPhase} fileName={uploadFileName} error={uploadError} onUpload={handleUpload} />
        )}
      </main>
    </div>
  );
}

export default App;
