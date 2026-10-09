import { memo, useMemo, useState } from "react";
import type { Source } from "../api";
import type { AssistantMessage, ChatMessage } from "../types";
import { AnswerText } from "./AnswerText";

type Props = {
  message: ChatMessage;
  onRetry: (question: string) => void;
};

/** Memoized so streaming tokens into one answer do not re-render the others. */
export const Message = memo(function Message({ message, onRetry }: Props) {
  if (message.role === "user") {
    return (
      <div className="message message-user">
        <p>{message.content}</p>
      </div>
    );
  }
  return <Answer message={message} onRetry={onRetry} />;
});

function Answer({ message, onRetry }: { message: AssistantMessage; onRetry: (question: string) => void }) {
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const sourceIds = useMemo(() => new Set(message.sources.map((source) => source.id)), [message.sources]);
  const isLive = message.status === "searching" || message.status === "streaming";

  function toggleSource(id: number) {
    setActiveSource((current) => (current === id ? null : id));
  }

  async function copyAnswer() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard access can be denied; copying is a convenience only.
    }
  }

  return (
    <article className="message message-assistant" aria-busy={isLive}>
      {message.status === "searching" && (
        <p className="answer-status">
          <span className="status-dot" aria-hidden="true" />
          Searching the document…
        </p>
      )}

      {message.content && (
        <div className={`answer-body ${message.status === "streaming" ? "is-streaming" : ""}`}>
          <AnswerText
            text={message.content}
            sourceIds={sourceIds}
            activeSource={activeSource}
            onCite={toggleSource}
          />
        </div>
      )}

      {message.status === "stopped" && <p className="answer-note">Stopped.</p>}

      {message.status === "error" && (
        <div className="answer-error" role="alert">
          <p>{message.error}</p>
          <button type="button" className="text-button" onClick={() => onRetry(message.question)}>
            Try again
          </button>
        </div>
      )}

      {message.sources.length > 0 && message.status !== "searching" && (
        <SourceList sources={message.sources} activeSource={activeSource} onToggle={toggleSource} />
      )}

      {message.status === "done" && message.content && (
        <div className="answer-actions">
          <button type="button" className="text-button" onClick={copyAnswer}>
            {copied ? "Copied" : "Copy answer"}
          </button>
        </div>
      )}
    </article>
  );
}

function SourceList({
  sources,
  activeSource,
  onToggle,
}: {
  sources: Source[];
  activeSource: number | null;
  onToggle: (id: number) => void;
}) {
  return (
    <section className="sources" aria-label="Sources">
      <h3 className="sources-title">Sources</h3>
      <ol className="source-list">
        {sources.map((source) => {
          const isOpen = activeSource === source.id;
          return (
            <li key={source.id} className={`source ${isOpen ? "is-open" : ""}`}>
              <button
                type="button"
                className="source-header"
                aria-expanded={isOpen}
                onClick={() => onToggle(source.id)}
              >
                <span className="source-number">{source.id}</span>
                <span className="source-location">
                  {source.page != null && <span className="source-page">p. {source.page}</span>}
                  <span className="source-section">{source.section || "Document excerpt"}</span>
                </span>
                <span className="source-chevron" aria-hidden="true" />
              </button>
              {isOpen && <p className="source-text">{source.text}</p>}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
