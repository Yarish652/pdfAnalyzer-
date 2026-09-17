"""Compare OpenRouter models on conversational query rewriting."""

import os
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from backend.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    
]


def _request_delay_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("BENCHMARK_REQUEST_DELAY_SECONDS", "5")))
    except ValueError:
        return 5.0

REWRITE_SYSTEM_PROMPT = """Rewrite only when necessary.

If the question is already standalone and retrieval-ready, return it unchanged.

For conversational questions:
- Resolve pronouns and references using the conversation.
- Preserve the user's original intent exactly.
- Preserve important entities and names.
- Do not add unnecessary details from the conversation.
- Do not infer information that is not explicitly established.
- Do not answer the question.
- Return exactly one standalone natural-language question."""

TEST_CASES = [
    {
        "question": "What was her role in the migration?",
        "history": [
            {"role": "user", "content": "Who was Maya Patel on the project?"},
            {"role": "assistant", "content": "Maya Patel was the migration lead."},
        ],
    },
    {
        "question": "And what were the results?",
        "history": [
            {"role": "user", "content": "Summarize the 2024 customer survey."},
            {"role": "assistant", "content": "The survey found higher satisfaction but slower onboarding."},
        ],
    },
    {
        "question": "How long did it take?",
        "history": [
            {"role": "user", "content": "When did the Phoenix rollout begin?"},
            {"role": "assistant", "content": "The Phoenix rollout began in March 2023."},
            {"role": "user", "content": "Was it completed before the end of the year?"},
            {"role": "assistant", "content": "Yes, the rollout was completed in November 2023."},
        ],
    },
    {
        "question": "Does it support offline use?",
        "history": [
            {"role": "user", "content": "Tell me about Atlas Mobile."},
            {"role": "assistant", "content": "Atlas Mobile is the field inspection application."},
        ],
    },
    {
        "question": "What did the second phase change?",
        "history": [
            {"role": "user", "content": "What were the phases of the Orion project?"},
            {"role": "assistant", "content": "Phase one covered data collection. Phase two covered automated validation."},
        ],
    },
    {
        "question": "Who approved that decision?",
        "history": [
            {"role": "user", "content": "Why was the storage architecture changed?"},
            {"role": "assistant", "content": "The team changed it to reduce recovery time and operating cost."},
            {"role": "user", "content": "Was the change documented?"},
            {"role": "assistant", "content": "Yes, it was documented in the architecture decision record."},
        ],
    },
    {
        "question": "Compare its limitations with the legacy service.",
        "history": [
            {"role": "user", "content": "What is the Nimbus service?"},
            {"role": "assistant", "content": "Nimbus is the replacement for the legacy reporting service."},
        ],
    },
    {
        "question": "What does the term mean in this document?",
        "history": [
            {"role": "user", "content": "The report mentions a 'trust boundary'."},
            {"role": "assistant", "content": "It describes a point where data access or security assumptions change."},
        ],
    },
    {
        "question": "When was the Redwood policy last updated?",
        "history": [
            {"role": "user", "content": "What does the Redwood policy cover?"},
            {"role": "assistant", "content": "It covers retention and deletion of customer records."},
        ],
    },
    {
        "question": "List the three security controls required for production deployment.",
        "history": [],
    },
    {
        "question": "What is the maximum upload size?",
        "history": [],
    },
    {
        "question": "How does the approval process differ between Europe and North America?",
        "history": [],
    },
]


def build_prompt(question: str, history: list[dict[str, str]]) -> str:
    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in history
        if message.get("role") in {"user", "assistant"} and message.get("content")
    )

    return f"""Conversation:
{history_text or "(No prior conversation.)"}

Latest question: {question}

"""


def rewrite_query(client: OpenAI, model: str, question: str, history: list[dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, history)},
        ],
        max_tokens=100,
        extra_body={"reasoning": {"enabled": False}},
    )

    content = response.choices[0].message.content if response.choices else None
    if not content:
        print("FULL EMPTY RESPONSE:")
        print(response.model_dump())
        return "(empty response)"

    return content.strip()


def print_history(history: list[dict[str, str]]) -> None:
    if not history:
        print("(No prior conversation.)")
        return

    for message in history:
        print(f"{message['role'].upper()}: {message['content']}")


def format_api_error(error: Exception) -> str:
    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return f"[timeout/connection error; continuing] {error}"

    if isinstance(error, APIStatusError):
        status_code = error.status_code
        if status_code == 429:
            return f"[rate limit (429); continuing] {error}"
        if status_code == 404:
            return f"[model unavailable (404); continuing] {error}"
        return f"[API error ({status_code}); continuing] {error}"

    return f"[API error; continuing] {type(error).__name__}: {error}"


def main() -> None:
    if not OPENROUTER_API_KEY:
        raise SystemExit("OPEN_ROUTER_API_KEY is not set.")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
    request_delay = _request_delay_seconds()

    for model in MODELS:
        print("\n" + "=" * 80)
        print(f"MODEL: {model}")
        print(f"REQUEST DELAY: {request_delay:g} seconds")

        for case_number, case in enumerate(TEST_CASES, start=1):
            question = case["question"]
            history = case["history"]

            print("\n" + "-" * 80)
            print(f"TEST CASE {case_number}")
            print(f"ORIGINAL QUESTION: {question}")
            print("CONVERSATION HISTORY:")
            print_history(history)
            print("REWRITTEN QUERY:")
            try:
                print(rewrite_query(client, model, question, history))
            except Exception as error:
                print(format_api_error(error))

            if case_number < len(TEST_CASES) and request_delay:
                time.sleep(request_delay)


if __name__ == "__main__":
    main()