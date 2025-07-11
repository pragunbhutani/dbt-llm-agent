from __future__ import annotations

"""Structured message schema emitted by LLM and rendered to different channels.

We intentionally keep this minimal but extensible.  Each **block** represents one
logical chunk of content.  The union `MessageBlock` can be extended later (e.g.
tables, images).  Optional interactive blocks (Button, SelectMenu) are included
because Slack supports them and product wants future head-room.
"""

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


class Paragraph(BaseModel):
    type: Literal["paragraph"] = "paragraph"
    text: str = Field(..., description="Free-form mrkdwn paragraph.")


class BulletList(BaseModel):
    type: Literal["bullets"] = "bullets"
    items: List[str] = Field(..., description="Unordered list items.")


class CodeBlock(BaseModel):
    type: Literal["code"] = "code"
    language: Optional[str] = Field(
        None, description="Optional language tag, e.g. sql, python"
    )
    text: str = Field(..., description="Code contents **without** fenced markers.")


class Divider(BaseModel):
    type: Literal["divider"] = "divider"


# Optional interactive blocks --------------------------------------------------


class Button(BaseModel):
    type: Literal["button"] = "button"
    text: str = Field(..., description="Button label")
    action_id: str = Field(..., description="Slack action_id")
    url: Optional[str] = Field(None, description="Optional URL to open")


class SelectMenu(BaseModel):
    type: Literal["select"] = "select"
    placeholder: str = Field(..., description="Placeholder text")
    action_id: str = Field(..., description="Slack action_id")
    options: List[str] = Field(..., description="List of option labels")


MessageBlock = Union[Paragraph, BulletList, CodeBlock, Divider, Button, SelectMenu]


class Message(BaseModel):
    """Top-level container.

    The LLM should return a JSON object: {"blocks": [ ... ]}
    """

    blocks: List[MessageBlock] = Field(
        ..., description="Array of message blocks in order"
    )
