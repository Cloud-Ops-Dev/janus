"""Dynamic tool exposure (Phase 6, infra-lxt — optional, host-dependent).

After ``capability_search``, a client that handles ``notifications/tools/
list_changed`` can ask Janus to *temporarily* surface the top matches as native
MCP tools, each carrying the downstream's real input schema. The generic
``capability_call`` stays the universal fallback, so clients that ignore
list_changed are unaffected.

Every exposed tool proxies straight back through ``broker.capability_call`` — so
policy, the lethal-trifecta guard, and audit all still apply. A capability that
the current session may not call (policy ``deny``) is never exposed.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools.tool import Tool, ToolResult
from pydantic import PrivateAttr

from janus.broker import Broker
from janus.registry.registry import EnvScope

# ── Exposed-tool naming (infra-smy1) ──────────────────────────────────────────
# An exposed tool is named <stem><sep><capability_id with '.' -> sep>, e.g. the
# default "__" yields cap__open_brain__search_thoughts. "__" is the historical
# and still-default choice because capability ids already contain SINGLE
# underscores (open_brain, search_thoughts), so "_" cannot be parsed back out.
#
# Janus itself never parses the name back — DynamicToolExposer keeps a
# name -> capability_id dict — so ambiguity costs Janus nothing. It costs the
# CLIENT: Grok Build uses "__" as its OWN server/tool separator (its events.jsonl
# shows call_id "janus__capability_search"), so a tool called
# janus__cap__open_brain__search_thoughts is unparseable and Grok SILENTLY DROPS
# it with no warning. That is why Grok saw 9 janus tools where Claude saw 11:
# the two missing were exactly the JANUS_AUTO_EXPOSE natives, the only two whose
# names contain "__".
#
# So this is a per-client knob, not a global rename. DO NOT change the default:
# cap__* names are already baked into Claude and Codex hooks, docs, and memory
# (mcp__janus__cap__open_brain__search_thoughts). Point only Grok's stdio wrapper
# at JANUS_EXPOSED_SEPARATOR=_ and the other two agents keep byte-identical names.
#
# The stem is deliberately joined with the SAME separator rather than hardcoded
# as "cap__": a bare "cap__" prefix would smuggle a "__" back into every name and
# defeat the knob entirely, no matter what separator the ids were joined with.
_EXPOSED_STEM = "cap"
_DEFAULT_SEPARATOR = "__"
# MCP tool names are conventionally [A-Za-z0-9_-]; a separator outside that set
# would produce names some clients reject outright. Fail safe to the default
# rather than emit an invalid name.
_VALID_SEPARATOR = re.compile(r"^[A-Za-z0-9_-]{1,8}$")


def exposed_separator(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the separator used to build exposed-tool names.

    Reads ``JANUS_EXPOSED_SEPARATOR`` (default ``"__"``). An empty or
    structurally invalid value falls back to the default — a broken separator
    should degrade to today's behaviour, never to unnameable tools.
    """
    env = environ if environ is not None else os.environ
    sep = env.get("JANUS_EXPOSED_SEPARATOR", "").strip()
    if not sep or not _VALID_SEPARATOR.match(sep):
        return _DEFAULT_SEPARATOR
    return sep


def exposed_tool_name(capability_id: str, separator: str | None = None) -> str:
    """Build the native tool name for ``capability_id``."""
    sep = separator if separator is not None else exposed_separator()
    return _EXPOSED_STEM + sep + capability_id.replace(".", sep)


class _ProxyTool(Tool):  # type: ignore[misc]  # FastMCP Tool is untyped (Any)
    """A native MCP tool that forwards to ``broker.capability_call``.

    Carries the downstream's real input schema (``parameters``) but routes every
    invocation back through the broker, so policy / trifecta / audit still apply.
    """

    _broker: Broker = PrivateAttr()
    _capability_id: str = PrivateAttr()
    _env: EnvScope | None = PrivateAttr(default=None)

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        result = await self._broker.capability_call(
            self._capability_id,
            arguments,
            reason=f"dynamically exposed call to {self._capability_id}",
            env=self._env,
        )
        return ToolResult(structured_content=result)


class DynamicToolExposer:
    """Owns the set of dynamically-exposed native tools for one MCP session."""

    def __init__(self, mcp: FastMCP, broker: Broker, *, max_exposed: int = 16) -> None:
        self._mcp = mcp
        self._broker = broker
        self._max = max_exposed
        self._exposed: dict[str, str] = {}  # tool_name -> capability_id
        self._tools: dict[str, Tool] = {}

    @property
    def active(self) -> dict[str, str]:
        return dict(self._exposed)

    @property
    def tools(self) -> list[Tool]:
        """Session-owned proxy tools, independent of provider internals."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @staticmethod
    def tool_name(capability_id: str) -> str:
        return exposed_tool_name(capability_id)

    async def expose(
        self, capability_ids: list[str], env: EnvScope | None = None
    ) -> dict[str, Any]:
        exposed: list[str] = []
        skipped: list[dict[str, str]] = []
        for cid in capability_ids:
            if len(self._exposed) >= self._max and self.tool_name(cid) not in self._exposed:
                skipped.append({"capability_id": cid, "reason": "max exposed reached"})
                continue
            desc = await self._broker.capability_describe(cid, env)
            if "error" in desc:
                skipped.append({"capability_id": cid, "reason": desc["error"]})
                continue
            if desc.get("policy", {}).get("decision") == "deny":
                # Never surface something the session cannot call.
                skipped.append({"capability_id": cid, "reason": "policy denied"})
                continue
            name = self.tool_name(cid)
            schema = desc.get("input_schema") or {"type": "object", "properties": {}}
            tool = self._build_tool(cid, name, str(desc.get("summary", cid)), schema, env)
            if name in self._exposed:
                self._mcp.local_provider.remove_tool(name)  # refresh in place
            self._mcp.local_provider.add_tool(tool)
            self._exposed[name] = cid
            self._tools[name] = tool
            exposed.append(name)
        return {"exposed": exposed, "skipped": skipped, "active": sorted(self._exposed)}

    def unexpose(self, names: list[str] | None = None) -> dict[str, Any]:
        targets = list(self._exposed) if names is None else names
        removed: list[str] = []
        for name in targets:
            if name in self._exposed:
                self._mcp.local_provider.remove_tool(name)
                del self._exposed[name]
                self._tools.pop(name, None)
                removed.append(name)
        return {"unexposed": removed, "active": sorted(self._exposed)}

    def _build_tool(
        self,
        capability_id: str,
        name: str,
        summary: str,
        schema: dict[str, Any],
        env: EnvScope | None,
    ) -> _ProxyTool:
        tool = _ProxyTool(
            name=name,
            description=f"{summary} (exposed Janus capability '{capability_id}')",
            parameters=schema,
        )
        tool._broker = self._broker
        tool._capability_id = capability_id
        tool._env = env
        return tool
