"""Tests for the credential broker + secret redactor.

The ``op`` CLI is never invoked: a fake op runner is injected so tests are
hermetic and no real secret is ever read.
"""

from __future__ import annotations

import pytest

from janus.downstream import ConnectionResolver
from janus.registry import AuthType, EnvScope, Server, ServerAuth, Transport
from janus.security import CredentialBroker, CredentialError, SecretRedactor

SAMPLE = "SUPERSECRETVALUE1234567890"


def _http_server(*, secret_env: str | None = None, secret_ref: str | None = None) -> Server:
    return Server(
        id="ob",
        display_name="OB",
        transport=Transport.STREAMABLE_HTTP,
        endpoint_env="OB_URL",
        auth=ServerAuth(
            type=AuthType.BEARER, secret_env=secret_env, secret_ref=secret_ref
        ),
        default_env_scope=[EnvScope.DEV],
    )


def test_resolve_secret_from_env_and_registers_redaction() -> None:
    broker = CredentialBroker({"OB_TOKEN": SAMPLE})
    server = _http_server(secret_env="OB_TOKEN")  # noqa: S106 — env var NAME
    assert broker.resolve_secret(server) == SAMPLE
    assert "«redacted»" in broker.redactor.redact(f"auth uses {SAMPLE} here")


def test_resolve_header_secret_from_env_and_registers_redaction() -> None:
    broker = CredentialBroker({"OB_BRAIN_KEY": SAMPLE})
    assert broker.resolve_header_secret("OB_BRAIN_KEY") == SAMPLE
    # extra-header values are secrets too -> registered for log redaction.
    assert "«redacted»" in broker.redactor.redact(f"x-brain-key: {SAMPLE}")
    assert broker.resolve_header_secret("MISSING") is None


def test_resolve_endpoint_and_command() -> None:
    broker = CredentialBroker({"OB_URL": "http://h:9/mcp"})
    assert broker.resolve_endpoint(_http_server(secret_env="X")) == "http://h:9/mcp"  # noqa: S106


def test_op_ref_resolution_caches_until_ttl() -> None:
    calls: list[str] = []

    def fake_op(ref: str) -> str:
        calls.append(ref)
        return SAMPLE

    now = [1000.0]
    broker = CredentialBroker(
        {},
        ttl_seconds=100.0,
        op_runner=fake_op,
        clock=lambda: now[0],
    )
    server = _http_server(secret_ref="op://System/ob/credential")  # noqa: S106 — op:// ref

    assert broker.resolve_secret(server) == SAMPLE
    assert broker.resolve_secret(server) == SAMPLE
    assert len(calls) == 1  # second call served from cache

    now[0] += 200.0  # past TTL
    assert broker.resolve_secret(server) == SAMPLE
    assert len(calls) == 2


def test_op_ref_value_is_redacted_after_resolution() -> None:
    broker = CredentialBroker({}, op_runner=lambda _ref: SAMPLE)
    server = _http_server(secret_ref="op://System/ob/credential")  # noqa: S106
    broker.resolve_secret(server)
    assert SAMPLE not in broker.redactor.redact(f"leaked {SAMPLE}")


def test_invalid_ref_raises() -> None:
    broker = CredentialBroker({}, op_runner=lambda _ref: SAMPLE)
    server = _http_server(secret_ref="vault://System/ob")  # noqa: S106
    with pytest.raises(CredentialError, match="not op://"):
        broker.resolve_secret(server)


def test_runner_failure_raises_loudly() -> None:
    def boom(_ref: str) -> str:
        raise CredentialError("op read failed for op://System/ob/credential (exit 1)")

    broker = CredentialBroker({}, op_runner=boom)
    server = _http_server(secret_ref="op://System/ob/credential")  # noqa: S106
    with pytest.raises(CredentialError):
        broker.resolve_secret(server)


def test_empty_secret_raises() -> None:
    broker = CredentialBroker({}, op_runner=lambda _ref: "")
    server = _http_server(secret_ref="op://System/ob/credential")  # noqa: S106
    with pytest.raises(CredentialError, match="empty secret"):
        broker.resolve_secret(server)


def test_broker_satisfies_connection_resolver_protocol() -> None:
    assert isinstance(CredentialBroker({}), ConnectionResolver)


# --------------------------------------------------------------------------- #
# SecretRedactor patterns
# --------------------------------------------------------------------------- #
def test_redactor_literal_and_token_shapes() -> None:
    redactor = SecretRedactor()
    redactor.register(SAMPLE)
    out = redactor.redact(
        f"literal={SAMPLE} bearer=Bearer abcdef1234567890 sa=ops_abcdefghijklmnop123"
    )
    assert SAMPLE not in out
    assert "abcdef1234567890" not in out
    assert "ops_abcdefghijklmnop123" not in out
    assert "Bearer «redacted»" in out


def test_redactor_ignores_short_values() -> None:
    redactor = SecretRedactor()
    redactor.register("abc")  # too short to register
    assert redactor.redact("abc def") == "abc def"
