from __future__ import annotations

"""Utilities to render our Message schema to Slack Block Kit structures."""

from typing import List, Dict, Any

from .message_schema import (
    Message,
    Paragraph,
    BulletList,
    CodeBlock,
    Divider,
    Button,
    SelectMenu,
)


# Small helpers ---------------------------------------------------------------


def _section(text: str, accessory: dict | None = None) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "type": "section",
        "text": {"type": "mrkdwn", "text": text},
    }
    if accessory is not None:
        block["accessory"] = accessory
    return block


def _button_block(button: Button) -> Dict[str, Any]:
    return {
        "type": "button",
        "text": {"type": "plain_text", "text": button.text, "emoji": True},
        "action_id": button.action_id,
        **({"url": button.url} if button.url else {}),
    }


def _select_block(menu: SelectMenu) -> Dict[str, Any]:
    return {
        "type": "static_select",
        "placeholder": {"type": "plain_text", "text": menu.placeholder, "emoji": True},
        "action_id": menu.action_id,
        "options": [
            {
                "text": {"type": "plain_text", "text": opt, "emoji": True},
                "value": opt,
            }
            for opt in menu.options
        ],
    }


# -----------------------------------------------------------------------------


def message_to_blockkit(message: Message) -> List[Dict[str, Any]]:
    """Convert validated Message object to Slack Block Kit list."""
    blocks: List[Dict[str, Any]] = []

    for blk in message.blocks:
        if isinstance(blk, Paragraph):
            blocks.append(_section(blk.text))
        elif isinstance(blk, BulletList):
            joined = "\n".join(f"• {item}" for item in blk.items)
            blocks.append(_section(joined))
        elif isinstance(blk, CodeBlock):
            code = f"```{blk.language or ''}\n{blk.text}\n```"
            blocks.append(_section(code))
        elif isinstance(blk, Divider):
            blocks.append({"type": "divider"})
        elif isinstance(blk, Button):
            blocks.append(_section(" ", accessory=_button_block(blk)))
        elif isinstance(blk, SelectMenu):
            blocks.append(_section(" ", accessory=_select_block(blk)))
        else:  # pragma: no cover
            # Unknown block types are ignored to avoid breaking message
            continue
    return blocks
