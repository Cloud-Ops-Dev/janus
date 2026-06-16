"""A fake downstream MCP server for client-manager integration tests.

Run as a stdio MCP server (``python tests/_fake_downstream.py``). The leading
underscore keeps pytest from collecting it as a test module. ``show_banner`` is
disabled so nothing pollutes the stdio JSON-RPC stream.
"""

from __future__ import annotations

import os

from fastmcp import FastMCP

mcp: FastMCP = FastMCP(os.environ.get("FAKE_NAME", "fake-downstream"))


@mcp.tool
def echo(text: str) -> str:
    """Echo the input text back unchanged."""
    return text


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
