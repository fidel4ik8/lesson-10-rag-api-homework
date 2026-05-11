"""Prompt-injection defenses.

Three layers (ТЗ §8):
1. Input length cap — enforced by Pydantic on the request model (4000 chars).
2. Input pattern detection — refuse the request before it reaches the LLM.
3. Output filter — post-stream check for leaked system-prompt fragments.

The pattern lists are intentionally short and high-precision. Catching every
possible injection is impossible; we focus on the obvious markers that show
up in real abuse and document the trade-off in the report.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUSPICIOUS_REQUESTS_LOG = ROOT / "suspicious_requests.log"
SUSPICIOUS_RESPONSES_LOG = ROOT / "suspicious_responses.log"


def _make_file_logger(name: str, path: Path) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.WARNING)
    if not log.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        log.addHandler(handler)
        log.propagate = False
    return log


_req_log = _make_file_logger("rag.security.request", SUSPICIOUS_REQUESTS_LOG)
_resp_log = _make_file_logger("rag.security.response", SUSPICIOUS_RESPONSES_LOG)


# Input markers — case-insensitive substrings. We require a literal match
# to keep false positives low; users may legitimately ask about "system prompts"
# in general, but they will not type "ignore previous instructions" by accident.
INPUT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_previous", re.compile(r"\bignore\s+(all\s+)?previous\s+(instructions?|prompts?|rules?)", re.I)),
    ("disregard_above", re.compile(r"\bdisregard\s+(the\s+)?(above|previous|prior|all)", re.I)),
    ("reveal_system_prompt", re.compile(r"\b(reveal|show|print|reveal\s+me|tell\s+me)\s+(your\s+)?(system\s+)?(prompt|instructions?|rules?)", re.I)),
    ("role_override", re.compile(r"\byou\s+are\s+(now|no\s+longer)\b", re.I)),
    ("chat_template_tokens", re.compile(r"<\|(im_start|im_end|system|user|assistant|/s)\|?>|</?s>", re.I)),
    ("system_role_injection", re.compile(r"^\s*system\s*:", re.I | re.M)),
    ("jailbreak_keywords", re.compile(r"\b(DAN|do\s+anything\s+now|jailbreak)\b", re.I)),
]


# Output markers — phrases unique to OUR system prompt. If the model echoes
# them, that is a signal it leaked instructions to the user.
OUTPUT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("rules_header", re.compile(r"\bRules\s*:\s*\n\s*1\.\s*Answer", re.I)),
    ("treat_as_data", re.compile(r"treat\s+anything\s+inside\s+<\s*user_query\s*>", re.I)),
    ("verbatim_system_role", re.compile(r"you\s+are\s+a\s+Q\s*&\s*A\s+assistant\s+for", re.I)),
    ("user_query_tag_echo", re.compile(r"<\s*/?\s*user_query\s*>", re.I)),
    ("context_tag_echo", re.compile(r"<\s*/?\s*context\s*>", re.I)),
]


def screen_input(text: str) -> list[str]:
    """Return the names of patterns that matched. Empty list = clean."""
    return [name for name, pat in INPUT_PATTERNS if pat.search(text)]


def screen_output(text: str) -> list[str]:
    """Same idea for the model's reply — used after the stream completes."""
    return [name for name, pat in OUTPUT_PATTERNS if pat.search(text)]


def log_suspicious_request(*, api_key: str, request_id: str, matches: list[str], snippet: str) -> None:
    snippet = snippet.replace("\n", " ")[:300]
    _req_log.warning(
        "api_key=%s request_id=%s matches=%s snippet=%r",
        api_key, request_id, ",".join(matches), snippet,
    )


def log_suspicious_response(*, api_key: str, request_id: str, matches: list[str], snippet: str) -> None:
    snippet = snippet.replace("\n", " ")[:300]
    _resp_log.warning(
        "api_key=%s request_id=%s matches=%s snippet=%r",
        api_key, request_id, ",".join(matches), snippet,
    )
