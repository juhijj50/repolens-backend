"""
Groq client wrapper.

Uses Llama 3.1 70B for reasoning. Groq's OpenAI-compatible API supports
JSON mode via response_format, which we use for all three LLM calls in
the analysis pipeline (file selection, overview, module graph).

All calls are synchronous — Groq is fast enough that async wouldn't buy
us much for the three sequential calls per analysis.
"""
import json
import logging
from typing import Any

from groq import Groq

from app.config import settings

logger = logging.getLogger(__name__)

# Model choice: Groq's migration target after the Llama 3.1/3.3 70B deprecations.
MODEL = "openai/gpt-oss-120b"

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def call_json(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """
    Call Groq with JSON mode enabled. Returns parsed dict.

    Raises ValueError if the model returns malformed JSON — callers should
    catch this and mark the analysis job as failed with a clear message.
    """
    client = _get_client()
    logger.warning("DEBUG: About to call Groq with MODEL=%s", MODEL)
    print(f"DEBUG STDOUT: MODEL={MODEL}", flush=True)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty response")
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("LLM returned invalid JSON: %s", content[:500])
        raise ValueError(f"LLM returned invalid JSON: {e}") from e


def call_text(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """Call Groq without JSON mode. Used by the chat endpoint (Phase 3)."""
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
