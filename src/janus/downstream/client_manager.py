"""Downstream client manager — hold MCP sessions to many servers (design §5.5).

Wraps the python-sdk ``ClientSessionGroup`` to connect to downstream MCP servers
over stdio / streamable-HTTP / SSE, and exposes a uniform
``call(server_id, tool, args)`` primitive plus a health probe. Phase 1 connects
the ``always_on`` servers eagerly at startup; lazy lifecycle is Phase 4.

Connection details (endpoints, secrets, commands) are obtained through a
:class:`ConnectionResolver` so the credential broker (infra-22q.5) can later
supply ``op://`` resolution behind the same interface. The Phase-1
:class:`EnvConnectionResolver` reads the environment-variable *names* declared in
the registry. Secrets are never logged and never returned to callers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.session_group import (
    ClientSessionGroup,
    SseServerParameters,
    StreamableHttpParameters,
)
from mcp.types import CallToolResult, Implementation, TextContent, Tool

from janus.registry.registry import AuthType, Lifecycle, Server, Transport

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class DownstreamError(RuntimeError):
    """Base class for downstream connection/call failures."""


class DownstreamNotConnected(DownstreamError):
    def __init__(self, server_id: str) -> None:
        super().__init__(f"downstream server '{server_id}' is not connected")
        self.server_id = server_id


class DownstreamCallError(DownstreamError):
    def __init__(self, server_id: str, tool: str, cause: BaseException) -> None:
        super().__init__(f"call to {server_id}.{tool} failed: {cause}")
        self.server_id = server_id
        self.tool = tool
        self.cause = cause


# --------------------------------------------------------------------------- #
# Connection resolution (credential broker plugs in here later)
# --------------------------------------------------------------------------- #
@runtime_checkable
class ConnectionResolver(Protocol):
    """Resolves a server's runtime connection details from its declaration."""

    def resolve_endpoint(self, server: Server) -> str | None: ...

    def resolve_command(self, server: Server) -> str | None: ...

    def resolve_secret(self, server: Server) -> str | None: ...

    def resolve_header_secret(self, env_name: str) -> str | None: ...


class EnvConnectionResolver:
    """Phase-1 resolver: read endpoint/command/secret from named env vars.

    ``op://`` secret references are intentionally NOT resolved here — that is the
    credential broker's job (infra-22q.5). A server declaring ``secret_ref``
    (op://) with no ``secret_env`` resolves to ``None`` under this resolver.
    """

    def __init__(self, environ: dict[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else dict(os.environ)

    def resolve_endpoint(self, server: Server) -> str | None:
        if server.endpoint_env:
            return self._environ.get(server.endpoint_env)
        return None

    def resolve_command(self, server: Server) -> str | None:
        if server.command:
            return server.command
        if server.command_env:
            return self._environ.get(server.command_env)
        return None

    def resolve_secret(self, server: Server) -> str | None:
        if server.auth.secret_env:
            return self._environ.get(server.auth.secret_env)
        return None

    def resolve_header_secret(self, env_name: str) -> str | None:
        return self._environ.get(env_name)


# --------------------------------------------------------------------------- #
# Result / status value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DownstreamResult:
    """A tool-call result, before policy sanitization (infra-22q.6)."""

    is_error: bool
    text: str
    structured: dict[str, Any] | None

    @classmethod
    def from_call_result(cls, result: CallToolResult) -> DownstreamResult:
        text = "\n".join(
            block.text for block in result.content if isinstance(block, TextContent)
        )
        return cls(
            is_error=bool(result.isError),
            text=text,
            structured=result.structuredContent,
        )


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str | None
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class HealthStatus:
    server_id: str
    connected: bool
    tool_count: int | None
    error: str | None


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #
class DownstreamClientManager:
    """Connect to downstream MCP servers and broker tool calls to them.

    Use as an async context manager::

        async with DownstreamClientManager(registry) as mgr:
            await mgr.connect_all()
            result = await mgr.call("open_brain", "search_thoughts", {"query": "x"})
    """

    def __init__(
        self,
        servers: dict[str, Server],
        resolver: ConnectionResolver | None = None,
        *,
        call_timeout: float = 30.0,
        max_retries: int = 2,
        connect_retries: int = 4,
        connect_retry_delay: float = 3.0,
    ) -> None:
        self._servers = servers
        self._resolver: ConnectionResolver = resolver or EnvConnectionResolver()
        self._call_timeout = call_timeout
        self._max_retries = max_retries
        # Startup connect resilience (infra-xwx): retry a downstream that is not
        # yet reachable (e.g. a boot-time DNS race) before giving up on it.
        self._connect_retries = connect_retries
        self._connect_retry_delay = connect_retry_delay
        self._sessions: dict[str, ClientSession] = {}
        self._stack: AsyncExitStack | None = None
        self._group: ClientSessionGroup | None = None
        # Server ids that failed to connect on the last connect_all (id -> error).
        self._connect_failures: dict[str, str] = {}
        # Set transiently around each connect so the namespacing hook can tag
        # components with our registry server id (connects are sequential).
        self._connecting_server_id: str | None = None

    # -- lifecycle ---------------------------------------------------------- #
    async def __aenter__(self) -> DownstreamClientManager:
        self._stack = AsyncExitStack()
        self._group = await self._stack.enter_async_context(
            ClientSessionGroup(component_name_hook=self._component_name_hook)
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._group = None
        self._sessions.clear()

    def _component_name_hook(self, name: str, server_info: Implementation) -> str:
        prefix = self._connecting_server_id or server_info.name
        return f"{prefix}::{name}"

    @property
    def connected_servers(self) -> list[str]:
        return list(self._sessions)

    @property
    def connect_failures(self) -> dict[str, str]:
        """Server ids that failed to connect on the last connect_all (id -> error)."""
        return dict(self._connect_failures)

    # -- connecting --------------------------------------------------------- #
    def _build_params(
        self, server: Server
    ) -> StdioServerParameters | StreamableHttpParameters | SseServerParameters:
        if server.transport is Transport.STDIO:
            command = self._resolver.resolve_command(server)
            if not command:
                raise DownstreamError(
                    f"server '{server.id}': no command (check command/command_env)"
                )
            return StdioServerParameters(command=command, args=list(server.args))

        url = self._resolver.resolve_endpoint(server)
        if not url:
            raise DownstreamError(
                f"server '{server.id}': no endpoint (env '{server.endpoint_env}' unset)"
            )
        headers = self._build_auth_headers(server)
        if server.transport is Transport.SSE:
            return SseServerParameters(url=url, headers=headers)
        # HTTP and STREAMABLE_HTTP both use the streamable-HTTP client.
        return StreamableHttpParameters(url=url, headers=headers)

    def _build_auth_headers(self, server: Server) -> dict[str, str] | None:
        headers: dict[str, str] = {}
        if server.auth.type is AuthType.BEARER:
            secret = self._resolver.resolve_secret(server)
            if not secret:
                raise DownstreamError(
                    f"server '{server.id}': bearer auth declared but secret unresolved"
                )
            headers["Authorization"] = f"Bearer {secret}"
        # Additional static headers (e.g. Open Brain's x-brain-key). Each value
        # comes from a named env var; a declared-but-unset header is fatal (§12).
        for header_name, env_name in server.auth.extra_headers.items():
            value = self._resolver.resolve_header_secret(env_name)
            if not value:
                raise DownstreamError(
                    f"server '{server.id}': extra header '{header_name}' declared "
                    f"but env '{env_name}' is unset"
                )
            headers[header_name] = value
        return headers or None

    async def connect_server(self, server_id: str) -> None:
        if self._group is None:
            raise DownstreamError("manager not started (use 'async with')")
        if server_id in self._sessions:
            return
        server = self._servers.get(server_id)
        if server is None:
            raise DownstreamError(f"unknown server '{server_id}'")
        params = self._build_params(server)
        self._connecting_server_id = server_id
        try:
            session = await self._group.connect_to_server(params)
        finally:
            self._connecting_server_id = None
        self._sessions[server_id] = session

    async def _connect_server_with_retry(self, server_id: str) -> None:
        """Connect one server, retrying transient failures with a fixed backoff.

        A freshly-booted host may not have DNS / dependencies ready when the
        gateway starts (the infra-xwx boot race: a stdio wrapper's ``op`` call
        fails because the resolver is not up yet). Retrying lets that clear
        instead of failing the connect outright.
        """
        attempts = self._connect_retries + 1
        last_exc: BaseException = DownstreamError(
            f"server '{server_id}': connect not attempted"
        )
        for attempt in range(1, attempts + 1):
            try:
                await self.connect_server(server_id)
                return
            except Exception as exc:  # noqa: BLE001 — retry any connect failure
                last_exc = exc
                if attempt < attempts:
                    logger.info(
                        "downstream '%s' connect attempt %d/%d failed (%s); "
                        "retrying in %.1fs",
                        server_id,
                        attempt,
                        attempts,
                        exc,
                        self._connect_retry_delay,
                    )
                    await asyncio.sleep(self._connect_retry_delay)
        raise last_exc

    async def connect_all(self, *, only_always_on: bool = True) -> list[str]:
        """Connect declared servers (always-on by default), tolerantly.

        Each downstream is retried with backoff (to ride out a transient boot
        DNS / dependency race) and, if it still fails, logged and skipped rather
        than aborting startup. This is the Logout-Test fix for infra-xwx: one
        dead or not-yet-ready downstream must not take Janus down. Returns the
        ids that connected; failures are recorded in ``connect_failures``.
        """
        connected: list[str] = []
        self._connect_failures = {}
        for server_id, server in self._servers.items():
            if only_always_on and server.lifecycle is not Lifecycle.ALWAYS_ON:
                continue
            try:
                await self._connect_server_with_retry(server_id)
                connected.append(server_id)
            except Exception as exc:  # noqa: BLE001 — tolerate any single downstream
                self._connect_failures[server_id] = str(exc)
                logger.warning(
                    "downstream '%s' failed to connect after %d attempt(s); "
                    "skipping so the gateway can still serve: %s",
                    server_id,
                    self._connect_retries + 1,
                    exc,
                )
        if self._connect_failures:
            logger.warning(
                "connect_all: %d server(s) connected, %d failed (%s)",
                len(connected),
                len(self._connect_failures),
                ", ".join(sorted(self._connect_failures)),
            )
        return connected

    # -- calling ------------------------------------------------------------ #
    async def call(
        self, server_id: str, tool: str, arguments: dict[str, Any] | None = None
    ) -> DownstreamResult:
        session = self._sessions.get(server_id)
        if session is None:
            raise DownstreamNotConnected(server_id)
        last_exc: BaseException | None = None
        for _attempt in range(self._max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    session.call_tool(tool, arguments or {}),
                    timeout=self._call_timeout,
                )
                return DownstreamResult.from_call_result(result)
            except (TimeoutError, ConnectionError) as exc:
                last_exc = exc  # transient — retry
            except Exception as exc:  # noqa: BLE001 — wrap any downstream error
                raise DownstreamCallError(server_id, tool, exc) from exc
        # Retries exhausted on transient errors; last_exc is always set here.
        raise DownstreamCallError(
            server_id, tool, last_exc or RuntimeError("call failed")
        )

    async def list_tools(self, server_id: str) -> list[ToolInfo]:
        session = self._sessions.get(server_id)
        if session is None:
            raise DownstreamNotConnected(server_id)
        result = await session.list_tools()
        return [self._tool_info(tool) for tool in result.tools]

    @staticmethod
    def _tool_info(tool: Tool) -> ToolInfo:
        return ToolInfo(
            name=tool.name,
            description=tool.description,
            input_schema=dict(tool.inputSchema),
        )

    # -- health ------------------------------------------------------------- #
    async def health(self, server_id: str | None = None) -> dict[str, HealthStatus]:
        ids = [server_id] if server_id is not None else list(self._sessions)
        out: dict[str, HealthStatus] = {}
        for sid in ids:
            if sid not in self._sessions:
                out[sid] = HealthStatus(sid, connected=False, tool_count=None,
                                        error="not connected")
                continue
            try:
                tools = await self.list_tools(sid)
                out[sid] = HealthStatus(sid, connected=True, tool_count=len(tools),
                                        error=None)
            except Exception as exc:  # noqa: BLE001 — health must never raise
                out[sid] = HealthStatus(sid, connected=False, tool_count=None,
                                        error=str(exc))
        return out
