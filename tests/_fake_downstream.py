"""A fake downstream MCP server for client-manager / discovery integration tests.

Run as a stdio MCP server (``python tests/_fake_downstream.py``). The leading
underscore keeps pytest from collecting it as a test module. ``show_banner`` is
disabled so nothing pollutes the stdio JSON-RPC stream.

Discovery/drift tests need a tool whose descriptor can change between two server
spawns. Because the MCP stdio client strips arbitrary parent env vars, the knobs
are passed as **command-line args** (the registry ``Server.args`` ARE forwarded):

    --desc-file PATH   read ``echo``'s description from PATH at startup (rewrite
                       the file + respawn to simulate descriptor drift)
    --extra-tool       expose an additional ``extra`` tool not in the registry
                       (to exercise unregistered-tool discovery)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastmcp import FastMCP

_DEFAULT_ECHO_DESC = "Echo the input text back unchanged."


def build() -> FastMCP:
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc-file")
    parser.add_argument("--extra-tool", action="store_true")
    ns, _ = parser.parse_known_args()

    echo_desc = _DEFAULT_ECHO_DESC
    if ns.desc_file:
        echo_desc = Path(ns.desc_file).read_text(encoding="utf-8").strip()

    mcp: FastMCP = FastMCP(os.environ.get("FAKE_NAME", "fake-downstream"))

    @mcp.tool(description=echo_desc)
    def echo(text: str) -> str:
        return text

    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two integers and return the sum."""
        return a + b

    if ns.extra_tool:

        @mcp.tool
        def extra(value: str) -> str:
            """An extra tool present only when --extra-tool is passed."""
            return value

    return mcp


if __name__ == "__main__":
    build().run(transport="stdio", show_banner=False)
