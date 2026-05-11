"""OpenRouter client. Phase 3 = single model, no fallback yet."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from openai import AsyncOpenAI

from app.config import get_settings


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers={
            "HTTP-Referer": settings.openrouter_http_referer,
            "X-Title": settings.openrouter_x_title,
        },
    )


@dataclass(frozen=True)
class LLMResult:
    model: str
    text: str
    input_tokens: int
    output_tokens: int


async def complete(model: str, messages: list[dict]) -> LLMResult:
    """Non-streaming completion. Used by phase 3 only — phase 4 replaces this
    with a streaming variant in `app/llm_stream.py`."""
    client = get_client()
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    choice = resp.choices[0].message.content or ""
    usage = resp.usage
    return LLMResult(
        model=resp.model,
        text=choice,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )
