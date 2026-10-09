import { type ChangeEvent, type DragEvent, useState } from "react";
import type { UploadPhase } from "../api";

type Props = {
  phase: UploadPhase | null;
  fileName: string | null;
  error: string;
  onUpload: (file: File) => void;
};

const STEPS: { phase: UploadPhase | "ready"; label: string }[] = [
  { phase: "uploading", label: "Upload" },
  { phase: "processing", label: "Read & index" },
  { phase: "ready", label: "Ready to chat" },
];

export function UploadScreen({ phase, fileName, error, onUpload }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [localError, setLocalError] = useState("");
  const isBusy = phase !== null;

  function choose(file: File | undefined) {
    if (!file || isBusy) return;
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setLocalError("That file isn't a PDF. Choose a .pdf file.");
      return;
    }
    setLocalError("");
    onUpload(file);
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    choose(event.target.files?.[0]);
    // Allow choosing the same file again after an error.
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    choose(event.dataTransfer.files[0]);
  }

  const shownError = localError || error;
  const activeStep = phase ? STEPS.findIndex((step) => step.phase === phase) : -1;

  return (
    <section className="upload-screen" aria-labelledby="upload-title">
      <div className="upload-intro">
        <h1 id="upload-title">Ask questions about any PDF</h1>
        <p>
          Books, papers, lecture slides, or resumes. Answers come only from your
          document, with citations to the pages they use.
        </p>
      </div>

      <label
        className={`drop-zone ${isDragging ? "is-dragging" : ""} ${isBusy ? "is-busy" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          if (!isBusy) setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <input type="file" accept="application/pdf,.pdf" onChange={handleChange} disabled={isBusy} />
        <span className="drop-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
            <path d="M14 3v5h5M12 17v-6M9.5 13.5 12 11l2.5 2.5" />
          </svg>
        </span>
        {isBusy ? (
          <span className="drop-copy">
            <strong>{fileName}</strong>
            <span>{phase === "uploading" ? "Uploading…" : "Reading and indexing. Long documents take a minute."}</span>
          </span>
        ) : (
          <span className="drop-copy">
            <strong>Drop a PDF here, or <span className="link-like">browse</span></strong>
            <span>Text-based PDFs up to 300 pages</span>
          </span>
        )}
      </label>

      {isBusy && (
        <ol className="steps" aria-label="Processing progress">
          {STEPS.map((step, index) => (
            <li
              key={step.phase}
              className={index < activeStep ? "is-done" : index === activeStep ? "is-active" : ""}
              aria-current={index === activeStep ? "step" : undefined}
            >
              <span className="step-marker" aria-hidden="true" />
              {step.label}
            </li>
          ))}
        </ol>
      )}

      {shownError && (
        <p className="upload-error" role="alert">
          {shownError}
        </p>
      )}
    </section>
  );
}
