"""Hardcoded API keys + tier metadata.

In production these would live in a secret store + database. For the homework
they are checked into the repo (intentionally — the assignment requires three
demo keys with documented tiers).

Tier shape:
    tokens_per_minute: int   — used by the Redis token bucket (phase 7)
    models: list[str]        — fallback chain. models[0] is primary; the rest
                               are tried in order on retryable failure (phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, status


@dataclass(frozen=True)
class Tier:
    name: str
    tokens_per_minute: int
    models: list[str]


# Models picked from openrouter.ai/models. We deliberately mix providers in
# each tier so a single provider outage does not kill the whole chain.
TIERS: dict[str, Tier] = {
    "demo-free": Tier(
        name="demo-free",
        tokens_per_minute=5_000,
        models=[
            "meta-llama/llama-3.1-8b-instruct",
            "google/gemini-flash-1.5",
            "meta-llama/llama-3.2-3b-instruct:free",
        ],
    ),
    "demo-pro": Tier(
        name="demo-pro",
        tokens_per_minute=20_000,
        models=[
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-haiku",
            "google/gemini-flash-1.5",
        ],
    ),
    "demo-enterprise": Tier(
        name="demo-enterprise",
        tokens_per_minute=100_000,
        models=[
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o-mini",
        ],
    ),
}


@dataclass(frozen=True)
class Principal:
    api_key: str
    tier: Tier


def require_api_key(x_api_key: str | None = Header(default=None)) -> Principal:
    """FastAPI dependency: validate X-API-Key header, return the Principal.

    Raises 401 if the header is missing or unknown.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    tier = TIERS.get(x_api_key)
    if tier is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unknown API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return Principal(api_key=x_api_key, tier=tier)
