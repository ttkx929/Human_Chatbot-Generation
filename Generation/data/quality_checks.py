"""Lightweight dialogue quality checks for GPQA SFT export."""

from __future__ import annotations

import re

SPOILER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"correct\s+(choice|option|answer)\s+is",
        r"the\s+answer\s+is",
        r"option\s+[A-D]\s+(is|looks)\s+correct",
        r"so\s+(the\s+)?answer\s+is\s+[A-D]\b",
        r"matches\s+option\s+[A-D]\b",
        r"choose\s+option\s+[A-D]\b",
        r"option\s+[A-D]\s+is\s+the\s+(only\s+)?(right|correct|plausible)",
    )
]


def first_assistant_message(messages: list[dict]) -> str | None:
    for message in messages:
        role = message.get("role")
        if role in ("assistant", "bot"):
            return message.get("content", "")
    return None


def assistant_reveals_answer(text: str) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in SPOILER_PATTERNS)


def early_spoiler_in_messages(messages: list[dict]) -> bool:
    first = first_assistant_message(messages)
    return assistant_reveals_answer(first or "")
