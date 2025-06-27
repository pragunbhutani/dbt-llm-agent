"""Resource definitions exposed to Claude via the Model Context Protocol.

For now we simply return an empty list.  When the backend adds explicit MCP
resource metadata we can proxy it the same way we do for tool calls.
"""

from typing import Any, Dict, List


def get_resources() -> List[Dict[str, Any]]:
    return []


__all__ = ["get_resources"]
