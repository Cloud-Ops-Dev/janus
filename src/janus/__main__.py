"""Janus gateway entrypoint.

    python -m janus --check     # validate environment, exit non-zero on problems
    python -m janus --stdio     # serve the MCP surface over stdio (per-client)
    python -m janus --serve     # serve the REST API (always-on networked surface)
    python -m janus --mcp-http  # serve the MCP surface over streamable-HTTP

``--check`` is the systemd ``ExecStartPre`` gate: it fails loudly when a required
endpoint or secret is missing, so the unit never starts in a half-configured,
silently-degraded state (constitution §12).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from janus.gateway import (
    GatewayConfig,
    check_environment,
    serve_mcp_http,
    serve_rest,
    serve_stdio,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="janus-gateway")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="validate environment and exit")
    mode.add_argument("--stdio", action="store_true", help="serve MCP over stdio")
    mode.add_argument("--serve", action="store_true", help="serve the REST API")
    mode.add_argument("--mcp-http", action="store_true", help="serve MCP over HTTP")
    ns = parser.parse_args(argv)

    config = GatewayConfig.from_env()

    if ns.check:
        problems = check_environment(config, os.environ)
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        if problems:
            return 1
        print("OK: environment valid", file=sys.stderr)
        return 0

    if ns.stdio:
        asyncio.run(serve_stdio(config))
    elif ns.serve:
        asyncio.run(serve_rest(config))
    elif ns.mcp_http:
        asyncio.run(serve_mcp_http(config))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
