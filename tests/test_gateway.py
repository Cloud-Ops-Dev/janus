"""Tests for the composition root + the end-to-end stdio MCP surface.

``test_end_to_end_stdio_phase1_acceptance`` spawns a real ``python -m janus
--stdio`` gateway, which itself connects to a fake downstream, and drives it as
an MCP client — directly verifying the Phase-1 acceptance criterion: an agent
with ONLY Janus loaded sees < 10 tools and can answer read questions.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import StdioServerParameters
from mcp.client.session_group import ClientSessionGroup
from mcp.types import CallToolResult, TextContent

from janus.gateway import Gateway, GatewayConfig, check_environment, parse_tokens
from janus.registry import EnvScope, load_registry

FAKE = str(Path(__file__).parent / "_fake_downstream.py")
SEED_CONFIG = Path(__file__).resolve().parents[1] / "config"


def _write_stdio_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "servers.yaml").write_text(
        "servers:\n"
        "  fake:\n"
        "    display_name: Fake\n"
        "    transport: stdio\n"
        f"    command: {sys.executable}\n"
        f"    args: ['{FAKE}']\n"
        "    risk_ceiling: read_only\n"
        "    default_env_scope: [dev, prod_safe]\n",
        encoding="utf-8",
    )
    (cfg / "capabilities.yaml").write_text(
        "capabilities:\n"
        "  fake.echo:\n"
        "    server_id: fake\n"
        "    downstream_tool_name: echo\n"
        "    title: Echo text\n"
        "    summary: Echo the text back.\n"
        "    risk: read_only\n"
        "    env_scope: [prod_safe]\n"
        "    approved: true\n"
        "  fake.add:\n"
        "    server_id: fake\n"
        "    downstream_tool_name: add\n"
        "    title: Add integers\n"
        "    summary: Add two integers.\n"
        "    risk: read_only\n"
        "    env_scope: [prod_safe]\n"
        "    approved: true\n",
        encoding="utf-8",
    )
    return cfg


def _stdio_config(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        config_dir=_write_stdio_config(tmp_path),
        data_dir=tmp_path / "data",
        default_env=EnvScope.PROD_SAFE,
    )


# --------------------------------------------------------------------------- #
# parse_tokens / check_environment
# --------------------------------------------------------------------------- #
def test_parse_tokens() -> None:
    tokens = parse_tokens({"JANUS_TOKENS": "ta=host-a:infra_operator;tb=host-b"})
    assert tokens["ta"].label == "host-a"
    assert tokens["ta"].profile == "infra_operator"
    assert tokens["tb"].label == "host-b"
    assert tokens["tb"].profile == "default_assistant"


def test_check_environment_flags_missing(tmp_path: Path) -> None:
    config = GatewayConfig(config_dir=SEED_CONFIG, data_dir=tmp_path / "data")
    problems = check_environment(config, {})
    assert any("endpoint env" in p for p in problems)
    assert any("secret env" in p for p in problems)
    assert any("JANUS_TOKENS" in p for p in problems)


def test_check_environment_ok_for_stdio(tmp_path: Path) -> None:
    config = GatewayConfig(
        config_dir=_write_stdio_config(tmp_path), data_dir=tmp_path / "data"
    )
    problems = check_environment(config, {"JANUS_TOKENS": "t=h:default_assistant"})
    assert problems == []


# --------------------------------------------------------------------------- #
# Gateway build + connect
# --------------------------------------------------------------------------- #
def test_gateway_build_and_connect(tmp_path: Path) -> None:
    async def body() -> None:
        config = _stdio_config(tmp_path)
        gateway = Gateway.build(config, environ={"JANUS_TOKENS": "t=h:default_assistant"})
        assert "fake" in gateway.deps.registry.servers
        assert gateway.tokens["t"].label == "h"
        connected = await gateway.connect()
        try:
            assert connected == ["fake"]
            assert gateway.manager.connected_servers == ["fake"]
        finally:
            await gateway.aclose()

    asyncio.run(body())


def test_seed_config_is_loadable() -> None:
    registry = load_registry(SEED_CONFIG)
    # open_brain, beads_readonly, beads_operator, paperclip (Phase-3 split).
    assert len(registry.servers) == 4


# --------------------------------------------------------------------------- #
# End-to-end: an agent with ONLY Janus loaded
# --------------------------------------------------------------------------- #
def _payload(result: CallToolResult) -> dict[str, Any]:
    sc = result.structuredContent
    if sc is not None:
        if set(sc) == {"result"} and isinstance(sc["result"], dict):
            return dict(sc["result"])
        return dict(sc)
    for block in result.content:
        if isinstance(block, TextContent):
            parsed = json.loads(block.text)
            assert isinstance(parsed, dict)
            return parsed
    return {}


def test_end_to_end_stdio_phase1_acceptance(tmp_path: Path) -> None:
    cfg = _write_stdio_config(tmp_path)
    env = {
        **os.environ,
        "JANUS_CONFIG_DIR": str(cfg),
        "JANUS_DATA_DIR": str(tmp_path / "data"),
        "JANUS_DEFAULT_ENV": "prod_safe",
        "JANUS_MCP_PROFILE": "default_assistant",
        "JANUS_TOKENS": "t=h:default_assistant",
    }
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "janus", "--stdio"], env=env
    )

    async def body() -> None:
        async with ClientSessionGroup() as group:
            session = await group.connect_to_server(params)

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            # core 7 + capability_expose/unexpose (Phase 6) = 9; still < 10.
            assert len(names) < 10
            assert "capability_search" in names

            search = _payload(await session.call_tool("capability_search", {"query": "echo"}))
            ids = {r["capability_id"] for r in search["results"]}
            assert "fake.echo" in ids

            servers = _payload(await session.call_tool("server_list", {}))
            assert servers["servers"][0]["server_id"] == "fake"
            assert servers["servers"][0]["connected"] is True

            called = _payload(
                await session.call_tool(
                    "capability_call",
                    {"capability_id": "fake.add", "arguments": {"a": 2, "b": 3},
                     "reason": "e2e"},
                )
            )
            assert called["status"] == "ok"

    asyncio.run(body())
