import { type FormEvent, type KeyboardEvent, useLayoutEffect, useRef, useState } from "react";

type Props = {
  isAnswering: boolean;
  onSend: (question: string) => void;
  onStop: () => void;
};

const MAX_HEIGHT_PX = 200;

export function Composer({ isAnswering, onSend, onStop }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Grow with the text, up to a limit, then scroll.
  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, [value]);

  function submit(event?: FormEvent) {
    event?.preventDefault();
    const question = value.trim();
    if (!question || isAnswering) return;
    onSend(question);
    setValue("");
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter adds a line. Ignore Enter while an IME is composing.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <form className="composer" onSubmit={submit}>
      <label className="sr-only" htmlFor="question">Ask a question about the document</label>
      <textarea
        id="question"
        ref={textareaRef}
        value={value}
        rows={1}
        placeholder="Ask a question about this document"
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        autoFocus
      />
      {isAnswering ? (
        <button type="button" className="composer-button is-stop" onClick={onStop} aria-label="Stop answering">
          <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><rect x="5" y="5" width="14" height="14" rx="2.5" fill="currentColor" /></svg>
        </button>
      ) : (
        <button type="submit" className="composer-button" disabled={!value.trim()} aria-label="Send question">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 19V5M5.5 11.5 12 5l6.5 6.5" />
          </svg>
        </button>
      )}
    </form>
  );
}
