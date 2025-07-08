from __future__ import annotations

"""Shared workflow utilities."""

import logging
from typing import Any, Dict, Type, TypeVar, Union

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
