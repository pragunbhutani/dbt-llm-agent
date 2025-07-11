from __future__ import annotations

"""Shared workflow utilities."""

import logging
from typing import Any, Dict, Type, TypeVar, Union, List

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def ensure_contract(
    model_cls: Type[T],
    data: Union[Dict[str, Any], T],
    *,
    component: str = "unknown",
    strict_mode: bool = False,
) -> T:
    """Validate *data* against *model_cls* and return a model instance.

    Parameters
    ----------
    model_cls: The Pydantic model that defines the contract.
    data: The incoming payload (dict or already a model instance).
    component: For logging/error context.
    strict_mode: In strict_mode=True we re-raise validation errors. In
        non-strict mode we return a best-effort instance using partial
        data and log the error.
    """

    if isinstance(data, model_cls):
        return data  # Already validated elsewhere.

    try:
        return model_cls(**data)  # type: ignore[arg-type]
    except ValidationError as err:
        logger.warning(
            "Contract validation failed in %s for %s: %s",
            component,
            model_cls.__name__,
            err,
        )
        if strict_mode:
            raise
        # Attempt salvage: include only fields the model knows.
        filtered: Dict[str, Any] = {
            k: v for k, v in data.items() if k in model_cls.model_fields
        }
        return model_cls(**filtered)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# Slack formatting helpers
# ------------------------------------------------------------------

import re

SMART_QUOTES = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}


def _replace_smart_quotes(text: str) -> str:
    for smart, normal in SMART_QUOTES.items():
        text = text.replace(smart, normal)
    return text


def _collapse_spaces(text: str) -> str:
    # Collapse runs of 3+ spaces; keep newlines intact
    return re.sub(r"[ \t]{3,}", "  ", text)


def _convert_list_markers(text: str) -> str:
    # Replace leading hyphen or asterisk with bullet dot
    return re.sub(r"^(?:\s*[-*])\s+", "• ", text, flags=re.MULTILINE)


def _ensure_double_space_linebreaks(text: str) -> str:
    # Add two trailing spaces at end of each line (Slack soft break)
    lines = text.split("\n")
    return "\n".join(
        [line if line.endswith("  ") or line == "" else f"{line}  " for line in lines]
    )


def _is_bullet(line: str) -> bool:
    return bool(re.match(r"^\s*[-•*]\s+", line))


def markdown_to_blocks(raw: str) -> List[Dict[str, Any]]:  # type: ignore[override]
    """Enhanced converter with proper bullet grouping and no stray dividers."""

    # Basic sanitisation first (reuse earlier helper chain but keep list markers)
    cleaned = _replace_smart_quotes(raw)
    cleaned = _collapse_spaces(cleaned)

    lines = cleaned.splitlines()

    blocks: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip leading blank lines
        if line.strip() == "":
            j = i + 1
            # If the next non-blank line starts a bullet list, skip divider
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and _is_bullet(lines[j]):
                i = j
                continue
            # else add divider if blocks exist and next line non blank
            if blocks and j < len(lines):
                blocks.append({"type": "divider"})
            i = j
            continue

        # Bullet list grouping -------------------------------------------------
        if _is_bullet(line):
            bullets: List[str] = []
            while i < len(lines) and (_is_bullet(lines[i]) or lines[i].strip() == ""):
                if _is_bullet(lines[i]):
                    bullets.append(re.sub(r"^\s*[-•*]\s+", "• ", lines[i]).strip())
                i += 1
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "\n".join(bullets)},
                }
            )
            continue

        # Fenced code blocks ---------------------------------------------------
        if line.strip().startswith("```"):
            fence_header = line.strip()
            language = fence_header[3:].strip()
            code_lines: List[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # skip closing fence
            code_text = f"```{language}\n" + "\n".join(code_lines) + "\n```"
            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": code_text}}
            )
            continue

        # Paragraph ------------------------------------------------------------
        para_lines: List[str] = []
        while i < len(lines) and lines[i].strip() != "":
            para_lines.append(lines[i].strip())
            i += 1
        paragraph_text = " ".join(para_lines)
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": paragraph_text}}
        )

    return blocks


# ------------------------------------------------------------------
# Legacy formatting helper (kept for compatibility)
# ------------------------------------------------------------------


def format_for_slack(raw: str) -> str:
    """Sanitise *raw* string for Slack markdown rendering.

    Steps: smart-quote normalisation, space collapsing, bullet conversion,
    and ensuring double-space line breaks.
    """

    cleaned = _replace_smart_quotes(raw)
    cleaned = _collapse_spaces(cleaned)
    cleaned = _convert_list_markers(cleaned)
    cleaned = _ensure_double_space_linebreaks(cleaned)
    return cleaned


# ------------------------------------------------------------------
# Slack Block Kit helpers (Option 3: rich messages)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Post markdown with automatic block conversion
# ------------------------------------------------------------------

import json
from typing import Dict, List

# These imports are late-bound to avoid circular dependencies during module import
from .blockkit_renderer import message_to_blockkit  # type: ignore
from .message_schema import Message  # type: ignore


async def post_markdown_message(  # noqa: D401 – simple helper
    slack_client: "AsyncWebClient",  # quoted to avoid hard dependency at import time
    *,
    channel: str,
    thread_ts: str,
    markdown: str,
    **kwargs: Any,
):
    """Post *markdown* content to Slack.

    1. We attempt ``rich_text_to_blocks`` first which may parse structured JSON.
    2. Fallback is markdown → blocks conversion.
    3. Always supply a plain-text fallback in the ``text`` field.
    """

    blocks = rich_text_to_blocks(markdown)

    # Plain-text fallback – collapse whitespace & truncate within safe limits
    fallback_text = _collapse_spaces(markdown)
    if len(fallback_text) > 2900:
        fallback_text = fallback_text[:2900] + "…"

    await slack_client.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=fallback_text or "(empty message)",
        blocks=blocks,
        **kwargs,
    )


# ------------------------------------------------------------------
# High-level helper: parse structured message or fallback to markdown
# ------------------------------------------------------------------


def rich_text_to_blocks(raw: str) -> List[Dict[str, Any]]:
    """Convert *raw* (possibly JSON) into Slack blocks.

    Workflow:
    • If *raw* looks like a JSON object with a ``blocks`` key, try to parse via
      the :class:`Message` schema and render to Block Kit.
    • On any failure, fall back to :func:`markdown_to_blocks` so we always return
      something usable.
    """

    cleaned = raw.strip()

    # Extract JSON if wrapped in ```json ... ``` or ``` ... ``` fences
    if cleaned.startswith("```"):
        # Remove leading/backtick lines
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            # drop first line (fence) and, if last line is closing fence, drop it
            if lines[-1].strip().startswith("```"):
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            cleaned = "\n".join(lines).strip()

    try:
        if cleaned.startswith("{"):
            data = json.loads(cleaned)
            if isinstance(data, dict) and "blocks" in data:
                message = Message.parse_obj(data)
                return message_to_blockkit(message)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug(
            "rich_text_to_blocks: JSON parse/validation failed (%s); falling back to markdown",
            exc,
        )

    return markdown_to_blocks(raw)
