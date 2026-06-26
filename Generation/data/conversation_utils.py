"""Helpers to keep multi-turn dialogues ending on an assistant/bot turn."""

from __future__ import annotations

from langchain.schema import AIMessage, HumanMessage, SystemMessage

CLOSING_SUFFIX = (
    "\n\nThe student has no further questions. "
    "Reply with a brief, supportive closing (2-4 sentences). "
    "Summarize the main takeaway without introducing new material. "
    "You may confirm the final option letter only if the student already derived it."
)


def ends_with_bot(conversation: list[dict]) -> bool:
    return bool(conversation) and conversation[-1]["role"] == "bot"


def strip_trailing_human_turns(conversation: list[dict]) -> list[dict]:
    """Drop trailing human turns when no bot reply follows (last-resort fallback)."""
    fixed = list(conversation)
    while fixed and fixed[-1]["role"] == "human":
        fixed.pop()
    return fixed


def append_closing_bot_turn(
    conversation: list[dict],
    responder_system_prompt: str,
    responder_llm,
) -> list[dict]:
    """Generate one closing tutor reply when the dialogue ends on a human turn."""
    if ends_with_bot(conversation):
        return conversation

    messages = []
    for turn in conversation:
        if turn["role"] == "human":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    closing_system = SystemMessage(content=responder_system_prompt + CLOSING_SUFFIX)
    response = responder_llm.invoke([closing_system] + messages)
    return conversation + [{"role": "bot", "content": response.content.strip()}]


def ensure_conversation_ends_with_bot(
    conversation: list[dict],
    responder_system_prompt: str | None = None,
    responder_llm=None,
    *,
    prefer_closing: bool = True,
) -> list[dict]:
    """
    Ensure the last turn is from the bot/assistant.

    If prefer_closing and responder_llm is provided, append a generated closing turn.
    Otherwise drop dangling human turns.
    """
    if ends_with_bot(conversation):
        return conversation
    if prefer_closing and responder_llm is not None and responder_system_prompt:
        return append_closing_bot_turn(conversation, responder_system_prompt, responder_llm)
    return strip_trailing_human_turns(conversation)
