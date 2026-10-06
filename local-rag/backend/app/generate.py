"""Grounded answer generation with local Ollama."""

from __future__ import annotations

from typing import Any

import requests

from .config import CHAT_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT


def build_prompt(
    question: str,
    contexts: list[dict[str, Any]],
) -> str:
    """Build a grounded prompt from assembled parent contexts."""

    if not contexts:
        raise ValueError("At least one context is required")

    source_parts = []

    for i, context in enumerate(contexts, start=1):
        source_parts.append(
            f"[SOURCE {i}] "
            f"{context['filename']} "
            f"page={context['page']} "
            f"section={context.get('section_path') or ''}\n"
            f"{context['content']}"
        )

    sources = "\n\n".join(source_parts)

    return f"""Answer the question using only the supplied sources.

Question:
{question}

Sources:
{sources}

Rules:
- Do not invent facts not supported by the sources.
- Prefer direct evidence from the source text, OCR and visual descriptions.
- When useful, cite the source number and page in the form [SOURCE n, p.X].
- If the sources do not contain the answer, say so.
"""


def generate_answer(
    question: str,
    contexts: list[dict[str, Any]],
) -> str:
    """Generate a grounded answer using the local Ollama model."""

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if not contexts:
        return (
            "I could not find relevant information "
            "in the indexed documents."
        )

    prompt = build_prompt(
        question,
        contexts,
    )

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0,
            },
        },
        timeout=OLLAMA_TIMEOUT,
    )

    response.raise_for_status()

    answer = (
        response.json()
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not answer:
        raise ValueError("Ollama returned an empty answer")

    return answer