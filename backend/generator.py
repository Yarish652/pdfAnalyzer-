from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY
)



def generate_answer(question, context, history):
    messages = [
        {
            "role": "system",
            "content": """Answer only from the supplied document context.
Use conversation history only to understand the user's question, not as a
source of facts. Return only the final answer. Never output reasoning,
analysis, chain-of-thought, or a thinking process. Do not infer unsupported
facts. If the context does not support the answer, say exactly:
I don't know based on the provided document."""
        }
    ]

    messages.extend(history)

    messages.append({
        "role": "user",
        "content": f"""
Context:
{context}

Question:
{question}
"""
    })

    response = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning:free",
        messages=messages,
        max_tokens=500,
        extra_body={
            "reasoning": {
                "enabled": False
            }
        }
    )

    content = None
    if response and response.choices:
        content = response.choices[0].message.content

    if not content or not content.strip():
        return "I don't know based on the provided document."

    return content.strip()

def rewrite_query(question, history):
    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in history
        if message.get("role") in {"user", "assistant"} and message.get("content")
    )

    prompt = f"""Conversation:
{history_text or "(No prior conversation.)"}

Latest question: {question}

Rewrite the latest question as exactly one standalone natural-language question
for document retrieval. Resolve pronouns and references from the conversation.
Preserve the user's intent and important names or project titles. Do not answer
the question, add facts, use keywords-only phrasing, explain the rewrite, or
output labels, JSON, history, or multiple queries.
"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning:free",
        messages=[
            {
                "role": "system",
                "content": "Return only the rewritten standalone question."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        extra_body={
            "reasoning": {
                "enabled": False
            }
        }
    )
    content = response.choices[0].message.content

    if not content:
        return question

    return content.strip()
