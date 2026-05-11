"""OpenRouter model prices, $/1M tokens.

Single source of truth for cost calculation. Prices are per million tokens
in USD, sampled from openrouter.ai/models. Update when adding new models
to a tier in app/auth.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("rag.pricing")


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float


PRICING: dict[str, ModelPrice] = {
    # demo-free chain
    "meta-llama/llama-3.1-8b-instruct": ModelPrice(0.05, 0.05),
    "google/gemini-flash-1.5": ModelPrice(0.075, 0.30),
    "meta-llama/llama-3.2-3b-instruct:free": ModelPrice(0.0, 0.0),
    # demo-pro chain
    "openai/gpt-4o-mini": ModelPrice(0.15, 0.60),
    "anthropic/claude-3.5-haiku": ModelPrice(0.80, 4.00),
    # demo-enterprise chain
    "openai/gpt-4o": ModelPrice(2.50, 10.00),
    "anthropic/claude-3.5-sonnet": ModelPrice(3.00, 15.00),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute cost in USD. Unknown models fall back to 0 with a warning —
    we do not want to silently overcharge or block the request."""
    price = PRICING.get(model)
    if price is None:
        log.warning("no pricing entry for model %s — billing $0", model)
        return 0.0
    return (input_tokens * price.input_per_1m + output_tokens * price.output_per_1m) / 1_000_000
