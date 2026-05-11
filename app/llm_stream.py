"""Streaming LLM call. Yields events as they arrive.

Phase 4 = single model, no fallback yet. Phase 5 will wrap this with the
fallback chain by retrying on the next model when the first chunk fails.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)

from app.circuit_breaker import breaker
from app.concurrency import get_llm_semaphore
from app.config import get_settings
from app.llm import get_client

log = logging.getLogger("rag.llm")

# 401/403: auth problem — fixing the request will not help, the same error will
# happen on the next model with the same key. Surface to the user.
NON_RETRYABLE_STATUS = {401, 403}


def _is_retryable_before_first_token(exc: BaseException) -> bool:
    """Decide whether to fall back to the next model.

    Called BEFORE the first token has been yielded — meaning the client has
    not seen any output yet, so it is safe to try another model. We default
    to "yes, retry" for almost everything: timeouts, network errors, 5xx,
    rate limits, and even 400 (which OpenRouter returns for invalid model
    IDs — that is a per-model problem, the next model in the chain may work).
    """
    if isinstance(exc, (APITimeoutError, APIConnectionError, asyncio.TimeoutError, TimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code not in NON_RETRYABLE_STATUS
    return False


@dataclass
class StreamUsage:
    """Final usage block emitted after the stream completes."""
    model: str
    input_tokens: int
    output_tokens: int
    ttft_ms: int
    total_ms: int


@dataclass
class StreamEvent:
    """Either a token (text != None) or the final usage marker (usage != None).

    Generators in Python can't return a final value cleanly when consumed in
    an async for loop, so we use a tagged union instead.
    """
    text: str | None = None
    usage: StreamUsage | None = None


async def stream_complete(model: str, messages: list[dict]) -> AsyncIterator[StreamEvent]:
    """Async generator: yields token events, then exactly one usage event.

    Token usage from OpenRouter requires `stream_options={"include_usage": True}`
    — without it the final chunk has no usage and we'd have to estimate via
    tiktoken (less accurate).
    """
    client = get_client()
    started = time.perf_counter()
    ttft_ms: int | None = None
    output_text_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0
    final_model = model

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        # `chunk` is a ChatCompletionChunk
        if chunk.model:
            final_model = chunk.model

        if chunk.choices:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            if content:
                if ttft_ms is None:
                    ttft_ms = int((time.perf_counter() - started) * 1000)
                output_text_parts.append(content)
                yield StreamEvent(text=content)

        # The final chunk (after stop) carries usage when include_usage=True
        if chunk.usage is not None:
            input_tokens = chunk.usage.prompt_tokens
            output_tokens = chunk.usage.completion_tokens

    total_ms = int((time.perf_counter() - started) * 1000)
    yield StreamEvent(
        usage=StreamUsage(
            model=final_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms or total_ms,
            total_ms=total_ms,
        )
    )


@dataclass
class FallbackEvent(StreamEvent):
    """Adds `fallback_used` so the endpoint can mark the done payload."""
    fallback_used: bool = False


async def stream_with_fallback(
    models: list[str],
    messages: list[dict],
) -> AsyncIterator[FallbackEvent]:
    """Stream from the first model that works.

    For each model in order:
      - skip if its circuit breaker is open
      - acquire the global LLM semaphore
      - request a stream; wrap in a 15s timeout
      - if we manage to read the FIRST event without a retryable error,
        we are committed to this model. Stream the rest unconditionally.
      - on retryable failure, record it on the breaker and try the next model
      - on non-retryable failure (400/401/422/etc.), raise immediately

    Once the first token is yielded to the caller, we cannot fall back any
    longer — the user already saw partial output. Mid-stream errors are
    surfaced to the caller (the endpoint will emit an SSE error event).
    """
    settings = get_settings()
    timeout = settings.llm_request_timeout_seconds
    sem = get_llm_semaphore()
    last_exc: BaseException | None = None
    tried: list[str] = []

    for idx, model in enumerate(models):
        if await breaker.is_open(model):
            log.warning("circuit open for %s, skipping", model)
            tried.append(f"{model}(open)")
            continue
        tried.append(model)
        is_fallback = idx > 0

        async with sem:
            try:
                gen = stream_complete(model=model, messages=messages)
                # First chunk under timeout — this is where most failures show up
                first = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
            except (StopAsyncIteration, *(e for e in [])):
                # Empty stream — treat as failure
                await breaker.record_failure(model)
                last_exc = RuntimeError(f"empty stream from {model}")
                continue
            except BaseException as exc:
                if _is_retryable_before_first_token(exc):
                    log.warning("retryable failure on %s: %s", model, exc)
                    await breaker.record_failure(model)
                    last_exc = exc
                    continue
                # Auth-style errors (401/403): same key won't fare better
                # against the next model. Surface to the user.
                raise

            # We got a first event without error — commit to this model
            yield FallbackEvent(text=first.text, usage=first.usage, fallback_used=is_fallback)
            try:
                async for ev in gen:
                    yield FallbackEvent(text=ev.text, usage=ev.usage, fallback_used=is_fallback)
            except BaseException:
                # Mid-stream failure: don't fall back — user already saw output
                raise
            else:
                await breaker.record_success(model)
                return

    raise RuntimeError(f"all models failed; tried={tried}; last={last_exc!r}")
