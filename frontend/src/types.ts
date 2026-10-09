import type { Source } from "./api";

export type UserMessage = { id: string; role: "user"; content: string };

export type AssistantStatus = "searching" | "streaming" | "done" | "stopped" | "error";

export type AssistantMessage = {
  id: string;
  role: "assistant";
  content: string;
  sources: Source[];
  status: AssistantStatus;
  error?: string;
  /** The question this answers, so a failed answer can be retried. */
  question: string;
};

export type ChatMessage = UserMessage | AssistantMessage;
