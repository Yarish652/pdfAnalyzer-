const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export type DocumentType = "resume" | "slides" | "prose";

export type DocumentInfo = {
  id: string;
  name: string;
  documentType?: DocumentType;
  pages?: number;
  chunks?: number;
};

export type Source = {
  id: number;
  page: number | null;
  section: string;
  text: string;
};

export type HistoryMessage = { role: "user" | "assistant"; content: string };

export type UploadPhase = "uploading" | "processing";

type StatusResponse = {
  status: "pending" | "processing" | "ready" | "failed";
  reason?: string;
  document_type?: DocumentType;
  pages?: number;
  chunks?: number;
};

/** An error whose message is safe and useful to show to the user. */
export class ApiError extends Error {}

const POLL_INTERVAL_MS = 400;

async function errorDetail(response: Response, fallback: string): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    // The body was not JSON; use the fallback below.
  }
  return fallback;
}

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export async function uploadPdf(file: File, onPhase: (phase: UploadPhase) => void): Promise<DocumentInfo> {
  onPhase("uploading");
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload`, { method: "POST", body: formData });
  if (!response.ok) {
    throw new ApiError(await errorDetail(response, "We couldn't upload that file. Please try again."));
  }
  const { document_id: id, filename } = await response.json();

  onPhase("processing");
  for (;;) {
    const statusResponse = await fetch(`${API_BASE_URL}/upload/${id}/status`);
    if (!statusResponse.ok) throw new ApiError("We lost track of that upload. Please try again.");
    const status: StatusResponse = await statusResponse.json();

    if (status.status === "ready") {
      return {
        id,
        name: filename || file.name,
        documentType: status.document_type,
        pages: status.pages,
        chunks: status.chunks,
      };
    }
    if (status.status === "failed") {
      throw new ApiError(status.reason ?? "We couldn't process this PDF. Please try another file.");
    }
    await wait(POLL_INTERVAL_MS);
  }
}

type StreamHandlers = {
  onSources: (sources: Source[]) => void;
  onToken: (text: string) => void;
};

/**
 * Ask a question and stream the answer from /ask/stream (server-sent events).
 * Resolves with the final answer text. Rejects with ApiError on failure, or
 * with a DOMException named "AbortError" when the signal is aborted.
 */
export async function streamAnswer(
  request: { question: string; documentId: string; history: HistoryMessage[] },
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: request.question,
      document_id: request.documentId,
      history: request.history,
    }),
    signal,
  });

  if (!response.ok || !response.body) {
    const fallback = response.status === 502
      ? "The answer service is busy right now. Please try again in a moment."
      : "We couldn't answer that question. Please try again.";
    throw new ApiError(await errorDetail(response, fallback));
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;

    // Events are separated by a blank line.
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const event = parseEvent(block);
      if (!event) continue;

      if (event.name === "sources") handlers.onSources(event.data.sources ?? []);
      else if (event.name === "token") handlers.onToken(event.data.text ?? "");
      else if (event.name === "done") return event.data.answer ?? "";
      else if (event.name === "error") {
        throw new ApiError(event.data.detail ?? "The answer was interrupted. Please try again.");
      }
    }
  }

  throw new ApiError("The answer was interrupted. Please try again.");
}

type EventData = { sources?: Source[]; text?: string; answer?: string; detail?: string };

function parseEvent(block: string): { name: string; data: EventData } | null {
  let name = "message";
  const dataLines: string[] = [];

  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }

  if (dataLines.length === 0) return null;
  try {
    return { name, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}
