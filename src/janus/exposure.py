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

from typing import Any

from fastmcp import FastMCP
from fastmcp.tools.tool import Tool, ToolResult
from pydantic import PrivateAttr

from janus.broker import Broker
from janus.registry.registry import EnvScope

_EXPOSED_PREFIX = "cap__"


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

    @property
    def active(self) -> dict[str, str]:
        return dict(self._exposed)

    @staticmethod
    def tool_name(capability_id: str) -> str:
        return _EXPOSED_PREFIX + capability_id.replace(".", "__")

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
            exposed.append(name)
        return {"exposed": exposed, "skipped": skipped, "active": sorted(self._exposed)}

    def unexpose(self, names: list[str] | None = None) -> dict[str, Any]:
        targets = list(self._exposed) if names is None else names
        removed: list[str] = []
        for name in targets:
            if name in self._exposed:
                self._mcp.local_provider.remove_tool(name)
                del self._exposed[name]
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
