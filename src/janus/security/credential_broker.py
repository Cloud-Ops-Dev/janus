"""Credential broker — resolve op:// secret references at call time (design §5.4).

Janus is the only holder of downstream secrets. Server records reference secrets
as ``op://`` URIs or env-var names. The broker resolves them (env directly, or
``op read`` for op:// refs), caches resolved values in-memory with a short TTL,
registers each with a :class:`SecretRedactor`, and **never** writes them to disk,
logs them, or returns them to the model.

It implements the :class:`ConnectionResolver` interface so the downstream client
manager uses it transparently in place of the Phase-1 ``EnvConnectionResolver``.

Self-sufficiency (constitution §12): resolution fails LOUDLY. There is no
``op ... || true`` fallback that would silently produce an unauthenticated
connection — a missing token raises :class:`CredentialError`.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable

from janus.registry.registry import Server
from janus.security.secret_redactor import SecretRedactor

OpRunner = Callable[[str], str]
Clock = Callable[[], float]


class CredentialError(RuntimeError):
    """Raised when a secret cannot be resolved. Never contains the secret value."""


class CredentialBroker:
    def __init__(
        self,
        environ: dict[str, str] | None = None,
        *,
        op_path: str = "op",
        ttl_seconds: float = 300.0,
        redactor: SecretRedactor | None = None,
        op_runner: OpRunner | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._environ = environ if environ is not None else dict(os.environ)
        self._op_path = op_path
        self._ttl = ttl_seconds
        self.redactor = redactor or SecretRedactor()
        self._op_runner = op_runner or self._default_op_runner
        self._clock: Clock = clock or time.monotonic
        self._cache: dict[str, tuple[str, float]] = {}

    # -- ConnectionResolver interface -------------------------------------- #
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
        auth = server.auth
        if auth.secret_env:
            value = self._environ.get(auth.secret_env)
            self.redactor.register(value)
            return value
        if auth.secret_ref:
            return self._resolve_op_ref(auth.secret_ref)
        return None

    def resolve_header_secret(self, env_name: str) -> str | None:
        # Extra-header values (e.g. open_brain's x-brain-key) are secrets too:
        # register with the redactor so they never surface in logs.
        value = self._environ.get(env_name)
        self.redactor.register(value)
        return value

    # -- op:// resolution --------------------------------------------------- #
    def _resolve_op_ref(self, ref: str) -> str:
        if not ref.startswith("op://"):
            raise CredentialError(f"invalid secret reference (not op://): {ref}")
        now = self._clock()
        cached = self._cache.get(ref)
        if cached is not None and cached[1] > now:
            return cached[0]
        value = self._op_runner(ref)
        if not value:
            raise CredentialError(f"empty secret resolved for {ref}")
        self._cache[ref] = (value, now + self._ttl)
        self.redactor.register(value)
        return value

    def _default_op_runner(self, ref: str) -> str:
        try:
            proc = subprocess.run(  # noqa: S603 — fixed argv, ref is op:// validated
                [self._op_path, "read", ref],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CredentialError(
                f"1Password CLI '{self._op_path}' not found on PATH"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CredentialError(f"timed out resolving {ref}") from exc
        if proc.returncode != 0:
            # stderr only (stdout could carry a partial secret); trimmed.
            raise CredentialError(
                f"op read failed for {ref} (exit {proc.returncode}): "
                f"{proc.stderr.strip()[:200]}"
            )
        return proc.stdout.strip()

    def clear_cache(self) -> None:
        self._cache.clear()
