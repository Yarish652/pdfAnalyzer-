import { useEffect, useRef } from "react";
import type { DocumentInfo, DocumentType } from "../api";
import type { ChatMessage } from "../types";
import { Composer } from "./Composer";
import { Message } from "./Message";

type Props = {
  doc: DocumentInfo;
  messages: ChatMessage[];
  isAnswering: boolean;
  onSend: (question: string) => void;
  onStop: () => void;
};

const SUGGESTIONS: Record<DocumentType, string[]> = {
  prose: [
    "Summarize the main topics of this document",
    "What are the key definitions introduced?",
    "What limitations or open problems are discussed?",
  ],
  slides: [
    "Summarize the main ideas of this deck",
    "What are the key terms and their definitions?",
    "Walk me through the steps or algorithms covered",
  ],
  resume: [
    "Summarize this person's experience",
    "Which projects are listed, and what technologies do they use?",
    "What is their education background?",
  ],
};

// Distance from the bottom within which new content keeps the view pinned.
const PIN_THRESHOLD_PX = 120;

export function ChatScreen({ doc, messages, isAnswering, onSend, onStop }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const messageCountRef = useRef(messages.length);

  // Follow streaming text only while the reader is at the bottom, so
  // scrolling up to reread an answer is never interrupted. Sending a new
  // question always jumps to the bottom.
  useEffect(() => {
    if (messages.length > messageCountRef.current) pinnedRef.current = true;
    messageCountRef.current = messages.length;

    const element = scrollRef.current;
    if (element && pinnedRef.current) element.scrollTop = element.scrollHeight;
  }, [messages]);

  function handleScroll() {
    const element = scrollRef.current;
    if (!element) return;
    pinnedRef.current = element.scrollHeight - element.scrollTop - element.clientHeight < PIN_THRESHOLD_PX;
  }

  return (
    <section className="chat" aria-label={`Conversation about ${doc.name}`}>
      <div className="chat-scroll" ref={scrollRef} onScroll={handleScroll}>
        <div className="chat-column">
          {messages.length === 0 ? (
            <div className="chat-empty">
              <h2>What would you like to know?</h2>
              <p>Answers are grounded in <strong>{doc.name}</strong> and cite the pages they use.</p>
              <ul className="suggestions">
                {SUGGESTIONS[doc.documentType ?? "prose"].map((suggestion) => (
                  <li key={suggestion}>
                    <button type="button" className="suggestion" onClick={() => onSend(suggestion)}>
                      {suggestion}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="message-list" role="log" aria-live="polite" aria-relevant="additions">
              {messages.map((message) => (
                <Message key={message.id} message={message} onRetry={onSend} />
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="composer-dock">
        <div className="chat-column">
          <Composer isAnswering={isAnswering} onSend={onSend} onStop={onStop} />
          <p className="composer-hint">Enter to send · Shift+Enter for a new line</p>
        </div>
      </div>
    </section>
  );
}
