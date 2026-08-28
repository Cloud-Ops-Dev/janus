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
from typing import Any, NoReturn, Protocol, runtime_checkable

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
# Exception-context cycle (infra-gn2q livelock 1)
# --------------------------------------------------------------------------- #
def exception_context_is_cyclic(exc: BaseException, *, limit: int = 64) -> bool:
    """True if ``exc.__context__`` walks into a cycle (or exceeds ``limit``).

    ``contextlib._fix_exception_context`` is a ``while 1:`` walk of that chain.
    A cycle holds the GIL forever — SIGTERM never runs, systemd SIGKILLs.
    Bounded so tests can assert the old path WITHOUT hanging the suite.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    steps = 0
    while cur is not None:
        ident = id(cur)
        if ident in seen:
            return True
        seen.add(ident)
        steps += 1
        if steps > limit:
            return True
        cur = cur.__context__
    return False


def break_exception_context_cycle(exc: BaseException) -> BaseException:
    """Unlink ``__context__`` / ``__cause__`` so contextlib cannot spin.

    Used by the connect-retry reraise (and any other retry that re-raises a
    caught exception). ``raise ... from None`` alone is not enough if the
    object already carries a cyclic chain from earlier SDK/stack teardowns.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        nxt = cur.__context__
        cur.__context__ = None
        cur.__cause__ = None
        cur.__suppress_context__ = True
        cur = nxt
    return exc


def reraise_unlinked(exc: BaseException) -> NoReturn:
    """Re-raise ``exc`` with its context chain broken (``from None``)."""
    break_exception_context_cycle(exc)
    raise exc from None


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
    """A connect/disconnect request handed to one downstream's owner task."""

    op: str
    params: StdioServerParameters | StreamableHttpParameters | SseServerParameters | None
    future: asyncio.Future[None]


class _ServerWorker:
    """Own one ClientSessionGroup in one task (infra-yvs.1.12 + infra-gn2q).

    One worker per downstream so:

    * a failed connect's cancel-scope teardown cannot take siblings with it
      (anyio ``_deliver_cancellation`` self-recursion livelock);
    * a caller-side connect timeout can return WITHOUT cancelling this task
      (``asyncio.wait_for`` cancellation is what fed livelock 2).
    """

    def __init__(self, server_id: str, *, connect_timeout: float) -> None:
        self.server_id = server_id
        self._connect_timeout = connect_timeout
        self.session: ClientSession | None = None
        self._queue: asyncio.Queue[_ConnRequest | None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._start_error: BaseException | None = None

    def _component_name_hook(self, name: str, server_info: Implementation) -> str:
        prefix = self.server_id or server_info.name
        return f"{prefix}::{name}"

    async def start(self) -> None:
        if self._task is not None:
            return
        queue: asyncio.Queue[_ConnRequest | None] = asyncio.Queue()
        self._queue = queue
        self._ready = asyncio.Event()
        self._start_error = None
        self._task = asyncio.create_task(self._run(queue), name=f"janus-ds-{self.server_id}")
        await self._ready.wait()
        if self._start_error is not None:
            err = self._start_error
            await self.close()
            raise DownstreamError(
                f"connection worker for '{self.server_id}' failed to start: {err}"
            ) from None

    async def _run(self, queue: asyncio.Queue[_ConnRequest | None]) -> None:
        try:
            async with AsyncExitStack() as stack:
                group = await stack.enter_async_context(
                    ClientSessionGroup(component_name_hook=self._component_name_hook)
                )
                self._ready.set()
                while True:
                    req = await queue.get()
                    if req is None:
                        break
                    await self._handle(group, req)
        except Exception as exc:  # noqa: BLE001 — report start failure
            self._start_error = exc
            if not self._ready.is_set():
                self._ready.set()

    async def _handle(self, group: ClientSessionGroup, req: _ConnRequest) -> None:
        try:
            if req.op == "connect":
                if self.session is not None:
                    req.future.set_result(None)
                    return
                if req.params is None:
                    raise DownstreamError("connect request without params")
                session = await group.connect_to_server(req.params)
                self.session = session
            elif req.op == "disconnect" and self.session is not None:
                session = self.session
                self.session = None
                await group.disconnect_from_server(session)
            if not req.future.done():
                req.future.set_result(None)
        except Exception as exc:  # noqa: BLE001 — surface to the submitting task
            break_exception_context_cycle(exc)
            if not req.future.done():
                req.future.set_exception(exc)

    async def submit(
        self,
        op: str,
        params: StdioServerParameters
        | StreamableHttpParameters
        | SseServerParameters
        | None = None,
    ) -> None:
        if self._queue is None or self._task is None:
            raise DownstreamError("worker not started")
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._queue.put(_ConnRequest(op, params, future))
        timeout = self._connect_timeout
        if timeout <= 0:
            await future
            return
        # Caller-side timeout: wait on the Future, do NOT cancel the worker.
        # asyncio.wait_for() cancels its inner awaitable — that is the
        # cancel-scope storm. asyncio.wait() on a Future just times out.
        done, _pending = await asyncio.wait({future}, timeout=timeout)
        if not done:
            raise DownstreamError(
                f"server '{self.server_id}': {op} timed out after {timeout:.1f}s"
            ) from None
        await future  # already done — raise stored exception if any

    async def close(self) -> None:
        if self._task is None:
            return
        if self._queue is not None:
            await self._queue.put(None)
        timeout = max(self._connect_timeout, 0.1)
        done, _pending = await asyncio.wait({self._task}, timeout=timeout)
        if not done:
            logger.warning(
                "downstream '%s' worker did not exit within %.1fs; "
                "not cancelling (cancel-scope teardown is the livelock)",
                self.server_id,
                timeout,
            )
            return
        try:
            await self._task
        except Exception as exc:  # noqa: BLE001 — teardown noise
            logger.warning("downstream '%s' worker exited with error: %s", self.server_id, exc)
        finally:
            self._task = None
            self._queue = None
            self.session = None


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
        connect_timeout: float = 10.0,
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
        # Per-attempt caller-side timeout (infra-gn2q). 0 disables.
        self._connect_timeout = connect_timeout
        self._sessions: dict[str, ClientSession] = {}
        # One owner task + ClientSessionGroup PER downstream (infra-gn2q): a
        # hung/failed connect's cancel-scope teardown cannot take siblings with
        # it, and enter/exit still happen in the same task (infra-yvs.1.12).
        self._workers: dict[str, _ServerWorker] = {}
        self._started = False
        # Server ids that failed to connect on the last connect_all (id -> error).
        self._connect_failures: dict[str, str] = {}
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
        self._started = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._close_all_workers()
        self._sessions.clear()
        self._started = False

    async def _ensure_worker(self, server_id: str) -> _ServerWorker:
        worker = self._workers.get(server_id)
        if worker is None:
            worker = _ServerWorker(server_id, connect_timeout=self._connect_timeout)
            self._workers[server_id] = worker
        await worker.start()
        return worker

    async def _close_all_workers(self) -> None:
        workers = list(self._workers.values())
        self._workers.clear()
        if not workers:
            return
        # Isolate teardown: one hung cancel-scope must not block the others.
        results = await asyncio.gather(
            *(w.close() for w in workers), return_exceptions=True
        )
        for worker, result in zip(workers, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "downstream '%s' close raised: %s", worker.server_id, result
                )

    async def _submit(
        self,
        op: str,
        server_id: str,
        params: StdioServerParameters
        | StreamableHttpParameters
        | SseServerParameters
        | None = None,
    ) -> None:
        if not self._started:
            raise DownstreamError("manager not started (use 'async with')")
        worker = await self._ensure_worker(server_id)
        await worker.submit(op, params)
        if op == "connect":
            if worker.session is not None:
                self._sessions[server_id] = worker.session
        elif op == "disconnect":
            self._sessions.pop(server_id, None)

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
        if not self._started:
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
        reraise_unlinked(last_exc)

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
            break_exception_context_cycle(exc)
            raise DownstreamError(
                f"server '{server_id}' connect failed: {exc}"
            ) from None
        lc.breaker.record_success()
        lc.state = LifecycleState.ACTIVE
        lc.note_used(self._clock())

    async def disconnect_server(self, server_id: str) -> None:
        """Tear down one server's session (idle shutdown). Safe if not connected.

        Routed through the owner task so the stdio cancel scope is exited in the
        same task it was entered in (infra-yvs.1.12).
        """
        if server_id not in self._sessions or not self._started:
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
                last_exc = break_exception_context_cycle(exc)
            except Exception as exc:  # noqa: BLE001 — wrap any downstream error
                self._note_call_failure(lc)
                break_exception_context_cycle(exc)
                raise DownstreamCallError(server_id, tool, exc) from None
        # Retries exhausted on transient errors; last_exc is always set here.
        self._note_call_failure(lc)
        cause = last_exc or RuntimeError("call failed")
        break_exception_context_cycle(cause)
        raise DownstreamCallError(server_id, tool, cause) from None

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
    # Bound list_tools so one wedged downstream cannot freeze the REST event loop
    # (infra-6lip: hung open_brain list_tools made /v1/server/health and even
    # unauthenticated /v1/health unresponsive until janus.service restart).
    _HEALTH_LIST_TOOLS_TIMEOUT_S = 5.0

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
                result = await asyncio.wait_for(
                    session.list_tools(),
                    timeout=self._HEALTH_LIST_TOOLS_TIMEOUT_S,
                )
                out[sid] = HealthStatus(sid, connected=True,
                                        tool_count=len(result.tools),
                                        error=None, lifecycle_state=label)
            except TimeoutError:
                logger.warning(
                    "health list_tools timed out after %.1fs for server %s",
                    self._HEALTH_LIST_TOOLS_TIMEOUT_S,
                    sid,
                )
                out[sid] = HealthStatus(
                    sid,
                    connected=False,
                    tool_count=None,
                    error=f"list_tools timed out after {self._HEALTH_LIST_TOOLS_TIMEOUT_S:.0f}s",
                    lifecycle_state=label,
                )
            except Exception as exc:  # noqa: BLE001 — health must never raise
                out[sid] = HealthStatus(sid, connected=False, tool_count=None,
                                        error=str(exc), lifecycle_state=label)
        return out
