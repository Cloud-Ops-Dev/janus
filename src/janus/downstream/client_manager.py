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
import time
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
from mcp.client.stdio import get_default_environment
from mcp.types import CallToolResult, Implementation, TextContent, Tool

from janus.downstream.lifecycle import (
    BreakerState,
    CircuitBreaker,
    Clock,
    LifecycleState,
    ServerLifecycle,
)
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
    lifecycle_state: str | None = None


@dataclass
class _ConnRequest:
    """A connect/disconnect request handed to the connection-owner task.

    ``op`` is ``"connect"`` or ``"disconnect"``. ``future`` is resolved (or
    failed) by the owner task once the operation completes, so the submitting
    task awaits a normal result while the actual group mutation happens in the
    single owner task (infra-yvs.1.12).
    """

    op: str
    server_id: str
    params: StdioServerParameters | StreamableHttpParameters | SseServerParameters | None
    future: asyncio.Future[None]


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
        idle_after: float = 0.0,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 30.0,
        clock: Clock | None = None,
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
        self._group: ClientSessionGroup | None = None
        # Single owner task for the ClientSessionGroup (infra-yvs.1.12). EVERY
        # connect/disconnect/close runs inside this task so each stdio child's
        # anyio cancel scope is entered AND exited in the same task — the fix for
        # the lazy connect-on-demand hang (a lazy connect ran in the per-call task
        # while the group teardown ran in another, tripping anyio's "exit cancel
        # scope in a different task"). Requests are submitted over a queue; tool
        # CALLS still use the session directly from any task (stream I/O is safe).
        self._worker_task: asyncio.Task[None] | None = None
        self._request_queue: asyncio.Queue[_ConnRequest | None] | None = None
        self._worker_ready: asyncio.Event | None = None
        self._worker_stop: asyncio.Event | None = None
        self._worker_error: BaseException | None = None
        # Server ids that failed to connect on the last connect_all (id -> error).
        self._connect_failures: dict[str, str] = {}
        # Set transiently around each connect so the namespacing hook can tag
        # components with our registry server id (connects are sequential).
        self._connecting_server_id: str | None = None
        # Phase 4 — lazy lifecycle + circuit breaker. idle_after=0 disables idle
        # reaping (every server then behaves as before). The breaker still guards
        # connects/calls regardless. Clock is monotonic + injectable for tests.
        self._idle_after = idle_after
        self._clock: Clock = clock or time.monotonic
        self._lifecycle: dict[str, ServerLifecycle] = {
            sid: ServerLifecycle(
                breaker=CircuitBreaker(
                    failure_threshold=breaker_threshold,
                    cooldown_seconds=breaker_cooldown,
                )
            )
            for sid in servers
        }

    # -- lifecycle ---------------------------------------------------------- #
    async def __aenter__(self) -> DownstreamClientManager:
        queue: asyncio.Queue[_ConnRequest | None] = asyncio.Queue()
        ready = asyncio.Event()
        stop = asyncio.Event()
        self._request_queue = queue
        self._worker_ready = ready
        self._worker_stop = stop
        self._worker_error = None
        self._worker_task = asyncio.create_task(
            self._connection_worker(queue, ready)
        )
        await ready.wait()
        if self._worker_error is not None:
            err = self._worker_error
            await self._shutdown_worker()
            raise DownstreamError(f"connection worker failed to start: {err}")
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._shutdown_worker()
        self._sessions.clear()

    async def _connection_worker(
        self, queue: asyncio.Queue[_ConnRequest | None], ready: asyncio.Event
    ) -> None:
        """Own the ClientSessionGroup for the manager's whole lifetime.

        Connect/disconnect/close ALL happen here (infra-yvs.1.12), so a downstream
        opened on demand by a per-call task is torn down in this same task — never
        the cross-task cancel-scope teardown that hung lazy stdio connects.
        """
        try:
            async with AsyncExitStack() as stack:
                self._group = await stack.enter_async_context(
                    ClientSessionGroup(component_name_hook=self._component_name_hook)
                )
                ready.set()
                while True:
                    req = await queue.get()
                    if req is None:  # stop sentinel
                        break
                    await self._handle_request(req)
            # The group + every connection close HERE, inside this task.
        except Exception as exc:  # noqa: BLE001 — group failed to start; report it
            self._worker_error = exc
            if not ready.is_set():
                ready.set()
        finally:
            self._group = None

    async def _handle_request(self, req: _ConnRequest) -> None:
        try:
            if req.op == "connect":
                if req.server_id in self._sessions:  # idempotent
                    req.future.set_result(None)
                    return
                if self._group is None or req.params is None:
                    raise DownstreamError("connect request without an open group")
                self._connecting_server_id = req.server_id
                try:
                    session = await self._group.connect_to_server(req.params)
                finally:
                    self._connecting_server_id = None
                self._sessions[req.server_id] = session
            elif req.op == "disconnect" and req.server_id in self._sessions:
                session = self._sessions.pop(req.server_id)
                if self._group is not None:
                    await self._group.disconnect_from_server(session)
            if not req.future.done():
                req.future.set_result(None)
        except Exception as exc:  # noqa: BLE001 — surface to the submitting task
            if not req.future.done():
                req.future.set_exception(exc)

    async def _submit(
        self,
        op: str,
        server_id: str,
        params: StdioServerParameters
        | StreamableHttpParameters
        | SseServerParameters
        | None = None,
    ) -> None:
        if self._request_queue is None or self._worker_task is None:
            raise DownstreamError("manager not started (use 'async with')")
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._request_queue.put(_ConnRequest(op, server_id, params, future))
        await future

    async def _shutdown_worker(self) -> None:
        if self._worker_task is None:
            return
        if self._worker_stop is not None:
            self._worker_stop.set()
        if self._request_queue is not None:
            await self._request_queue.put(None)  # wake the worker to exit
        try:
            await self._worker_task
        finally:
            self._worker_task = None
            self._request_queue = None
            self._group = None

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
            return StdioServerParameters(
                command=command,
                args=list(server.args),
                env=self._build_child_env(server),
            )

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

    def _build_child_env(self, server: Server) -> dict[str, str] | None:
        """Environment for a stdio child (infra-b7g env-injection).

        ``None`` (the default when nothing is declared) lets the SDK inherit only
        its safe default set. When ``env``/``env_passthrough`` is declared we
        start from that same default set so the child keeps HOME/PATH, then layer
        passed-through process env vars (resolved + redaction-registered) and
        finally the static map. This lets a downstream run without a bespoke
        wrapper that re-exports the op token / mise PATH / BEADS_* itself.
        """
        if not (server.env or server.env_passthrough):
            return None
        child = get_default_environment()
        for name in server.env_passthrough:
            value = self._resolver.resolve_header_secret(name)
            if value is not None:
                child[name] = value
        child.update(server.env)
        return child

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
        if self._worker_task is None:
            raise DownstreamError("manager not started (use 'async with')")
        if server_id in self._sessions:
            return
        server = self._servers.get(server_id)
        if server is None:
            raise DownstreamError(f"unknown server '{server_id}'")
        # Build params in THIS task (so a missing command/secret raises to the
        # caller synchronously); the actual connect runs in the owner task.
        params = self._build_params(server)
        await self._submit("connect", server_id, params)

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
                lc = self._lifecycle[server_id]
                lc.state = LifecycleState.ACTIVE
                lc.note_used(self._clock())
                lc.breaker.record_success()
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

    # -- lazy lifecycle + circuit breaker (Phase 4) ------------------------- #
    async def ensure_ready(self, server_id: str) -> None:
        """Make a server ready to call, connecting LAZY ones on demand.

        Always-on servers are owned by ``connect_all`` — if one is not connected
        that is a real fault, so this leaves it untouched (the caller then raises
        ``DownstreamNotConnected``). LAZY servers connect on first use, gated by
        the circuit breaker: an OPEN breaker fails fast (``DEGRADED``) until the
        cooldown elapses, then one half-open trial decides recovery.
        """
        server = self._servers.get(server_id)
        if server is None:
            raise DownstreamError(f"unknown server '{server_id}'")
        lc = self._lifecycle[server_id]
        if server_id in self._sessions:
            lc.note_used(self._clock())
            return
        if server.lifecycle is not Lifecycle.LAZY:
            return  # always-on, not connected -> caller raises NotConnected
        now = self._clock()
        if not lc.breaker.allow(now):
            lc.state = LifecycleState.DEGRADED
            raise DownstreamError(
                f"server '{server_id}' is degraded (circuit breaker open) — failing fast"
            )
        lc.state = LifecycleState.WARMING
        try:
            await self.connect_server(server_id)
        except Exception as exc:  # noqa: BLE001 — any connect failure trips the breaker
            lc.breaker.record_failure(self._clock())
            lc.state = (
                LifecycleState.DEGRADED
                if lc.breaker.state is not BreakerState.CLOSED
                else LifecycleState.COLD
            )
            raise DownstreamError(f"server '{server_id}' connect failed: {exc}") from exc
        lc.breaker.record_success()
        lc.state = LifecycleState.ACTIVE
        lc.note_used(self._clock())

    async def disconnect_server(self, server_id: str) -> None:
        """Tear down one server's session (idle shutdown). Safe if not connected.

        Routed through the owner task so the stdio cancel scope is exited in the
        same task it was entered in (infra-yvs.1.12).
        """
        if server_id not in self._sessions or self._worker_task is None:
            return
        try:
            await self._submit("disconnect", server_id)
        except Exception as exc:  # noqa: BLE001 — never let a teardown error escape
            logger.warning("disconnect '%s' failed: %s", server_id, exc)

    async def reap_idle(self, now: float | None = None) -> list[str]:
        """Shut down LAZY servers idle past ``idle_after``; always-on are kept up."""
        if self._idle_after <= 0:
            return []
        stamp = now if now is not None else self._clock()
        reaped: list[str] = []
        for sid, server in list(self._servers.items()):
            if server.lifecycle is not Lifecycle.LAZY or sid not in self._sessions:
                continue
            lc = self._lifecycle[sid]
            if lc.is_idle(self._idle_after, stamp):
                await self.disconnect_server(sid)
                lc.state = LifecycleState.COLD
                reaped.append(sid)
        if reaped:
            logger.info("reaped idle lazy downstream(s): %s", ", ".join(reaped))
        return reaped

    async def run_idle_reaper(self, *, interval: float) -> None:
        """Background loop: reap idle lazy downstreams every ``interval`` seconds."""
        while True:
            await asyncio.sleep(interval)
            try:
                await self.reap_idle()
            except Exception as exc:  # noqa: BLE001 — the reaper must never crash serving
                logger.warning("idle reaper error: %s", exc)

    def lifecycle_state(self, server_id: str) -> LifecycleState:
        return self._lifecycle[server_id].state

    # -- calling ------------------------------------------------------------ #
    async def call(
        self, server_id: str, tool: str, arguments: dict[str, Any] | None = None
    ) -> DownstreamResult:
        await self.ensure_ready(server_id)
        session = self._sessions.get(server_id)
        if session is None:
            raise DownstreamNotConnected(server_id)
        lc = self._lifecycle[server_id]
        last_exc: BaseException | None = None
        for _attempt in range(self._max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    session.call_tool(tool, arguments or {}),
                    timeout=self._call_timeout,
                )
                lc.breaker.record_success()
                lc.note_used(self._clock())
                return DownstreamResult.from_call_result(result)
            except (TimeoutError, ConnectionError) as exc:
                last_exc = exc  # transient — retry
            except Exception as exc:  # noqa: BLE001 — wrap any downstream error
                self._note_call_failure(lc)
                raise DownstreamCallError(server_id, tool, exc) from exc
        # Retries exhausted on transient errors; last_exc is always set here.
        self._note_call_failure(lc)
        raise DownstreamCallError(
            server_id, tool, last_exc or RuntimeError("call failed")
        )

    def _note_call_failure(self, lc: ServerLifecycle) -> None:
        lc.breaker.record_failure(self._clock())
        if lc.breaker.state is not BreakerState.CLOSED:
            lc.state = LifecycleState.DEGRADED

    async def list_tools(self, server_id: str) -> list[ToolInfo]:
        await self.ensure_ready(server_id)
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
        # Passive probe: reports connected servers + lifecycle state, but never
        # lazily connects a cold server nor resets its idle clock.
        ids = [server_id] if server_id is not None else list(self._sessions)
        out: dict[str, HealthStatus] = {}
        for sid in ids:
            lc = self._lifecycle.get(sid)
            label = str(lc.state) if lc is not None else None
            session = self._sessions.get(sid)
            if session is None:
                out[sid] = HealthStatus(sid, connected=False, tool_count=None,
                                        error="not connected", lifecycle_state=label)
                continue
            try:
                result = await session.list_tools()
                out[sid] = HealthStatus(sid, connected=True,
                                        tool_count=len(result.tools),
                                        error=None, lifecycle_state=label)
            except Exception as exc:  # noqa: BLE001 — health must never raise
                out[sid] = HealthStatus(sid, connected=False, tool_count=None,
                                        error=str(exc), lifecycle_state=label)
        return out
