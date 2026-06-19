"""Tests for the Janus capability registry: loading, validation, hashing, cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from janus.registry import (
    Capability,
    RegistryError,
    SchemaStore,
    hash_schema,
    hash_text,
    load_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_CONFIG = REPO_ROOT / "config"


def _write_config(
    tmp_path: Path, servers_yaml: str, capabilities_yaml: str
) -> Path:
    (tmp_path / "servers.yaml").write_text(servers_yaml, encoding="utf-8")
    (tmp_path / "capabilities.yaml").write_text(capabilities_yaml, encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- #
# Seed config
# --------------------------------------------------------------------------- #
def test_seed_config_loads_and_is_valid() -> None:
    registry = load_registry(SEED_CONFIG)
    assert set(registry.servers) == {
        "open_brain",
        "beads_readonly",
        "beads_operator",
        "paperclip",
    }
    assert len(registry.capabilities) >= 9
    # every seed capability is approved, not quarantined -> callable
    assert all(c.callable for c in registry.capabilities.values())
    assert len(registry.callable_capabilities()) == len(registry.capabilities)


def test_seed_capabilities_reference_real_servers() -> None:
    registry = load_registry(SEED_CONFIG)
    for cap in registry.capabilities.values():
        assert cap.server_id in registry.servers


def test_capabilities_for_server() -> None:
    registry = load_registry(SEED_CONFIG)
    ob = registry.capabilities_for_server("open_brain")
    # The FULL Open Brain surface is brokered (infra-rn6) so a client can drop its
    # direct open_brain MCP without losing any tool: 9 thought tools + 7 document
    # tools. If a tool is added to the open_brain MCP, mirror it here + in the seed.
    assert {c.downstream_tool_name for c in ob} == {
        # thoughts (semantic memory)
        "search_thoughts",
        "list_thoughts",
        "thought_stats",
        "capture_thought",
        "update_thought",
        "pin_thought",
        "unpin_thought",
        "archive_thought",
        "unarchive_thought",
        # documents (the Notion-replacement wiki)
        "search_documents",
        "get_document",
        "list_documents",
        "create_document",
        "update_document",
        "archive_document",
        "unarchive_document",
    }


# --------------------------------------------------------------------------- #
# Summary vs. raw description separation (the security invariant)
# --------------------------------------------------------------------------- #
def test_summary_is_model_visible_and_raw_description_is_not() -> None:
    fields = set(Capability.model_fields)
    # the model-visible, human-reviewed summary IS a field...
    assert "summary" in fields
    # ...but the untrusted raw downstream description is NEVER a model field;
    # only its hash is tracked.
    assert "description" not in fields
    assert "raw_description" not in fields
    assert "raw_description_hash" in fields


# --------------------------------------------------------------------------- #
# Cross-validation failures
# --------------------------------------------------------------------------- #
SERVER_OK = """
servers:
  s1:
    display_name: S1
    transport: streamable_http
    endpoint_env: S1_URL
    risk_ceiling: read_only
    default_env_scope: [dev]
"""


def test_unknown_server_id_rejected(tmp_path: Path) -> None:
    caps = """
capabilities:
  s1.bad:
    server_id: nonexistent
    downstream_tool_name: t
    title: T
    summary: s
    risk: read_only
    env_scope: [dev]
"""
    cfg = _write_config(tmp_path, SERVER_OK, caps)
    with pytest.raises(RegistryError, match="unknown server_id"):
        load_registry(cfg)


def test_risk_exceeds_server_ceiling_rejected(tmp_path: Path) -> None:
    caps = """
capabilities:
  s1.danger:
    server_id: s1
    downstream_tool_name: t
    title: T
    summary: s
    risk: destructive
    env_scope: [dev]
"""
    cfg = _write_config(tmp_path, SERVER_OK, caps)
    with pytest.raises(RegistryError, match="exceeds server"):
        load_registry(cfg)


def test_env_scope_outside_server_rejected(tmp_path: Path) -> None:
    caps = """
capabilities:
  s1.cap:
    server_id: s1
    downstream_tool_name: t
    title: T
    summary: s
    risk: read_only
    env_scope: [dev, prod]
"""
    cfg = _write_config(tmp_path, SERVER_OK, caps)
    with pytest.raises(RegistryError, match="outside server"):
        load_registry(cfg)


# --------------------------------------------------------------------------- #
# Transport validation
# --------------------------------------------------------------------------- #
def test_http_server_requires_endpoint_env(tmp_path: Path) -> None:
    servers = """
servers:
  s1:
    display_name: S1
    transport: streamable_http
    default_env_scope: [dev]
"""
    cfg = _write_config(tmp_path, servers, "capabilities: {}")
    with pytest.raises(RegistryError, match="requires endpoint_env"):
        load_registry(cfg)


def test_stdio_server_requires_command(tmp_path: Path) -> None:
    servers = """
servers:
  s1:
    display_name: S1
    transport: stdio
    default_env_scope: [dev]
"""
    cfg = _write_config(tmp_path, servers, "capabilities: {}")
    with pytest.raises(RegistryError, match="requires command"):
        load_registry(cfg)


def test_stdio_server_accepts_command(tmp_path: Path) -> None:
    servers = """
servers:
  s1:
    display_name: S1
    transport: stdio
    command: bd
    args: [mcp]
    default_env_scope: [dev]
"""
    cfg = _write_config(tmp_path, servers, "capabilities: {}")
    registry = load_registry(cfg)
    assert registry.servers["s1"].command == "bd"


def test_bearer_auth_requires_secret(tmp_path: Path) -> None:
    servers = """
servers:
  s1:
    display_name: S1
    transport: streamable_http
    endpoint_env: S1_URL
    auth: {type: bearer}
    default_env_scope: [dev]
"""
    cfg = _write_config(tmp_path, servers, "capabilities: {}")
    with pytest.raises(RegistryError, match="requires secret_env or secret_ref"):
        load_registry(cfg)


def test_seed_open_brain_declares_extra_header() -> None:
    registry = load_registry(SEED_CONFIG)
    auth = registry.servers["open_brain"].auth
    # header NAME -> env-var NAME (no secret values in the public seed).
    assert auth.extra_headers == {"x-brain-key": "JANUS_OPEN_BRAIN_KEY"}


def test_extra_headers_loaded(tmp_path: Path) -> None:
    servers = """
servers:
  s1:
    display_name: S1
    transport: streamable_http
    endpoint_env: S1_URL
    auth:
      type: bearer
      secret_env: S1_TOKEN
      extra_headers:
        x-brain-key: S1_BRAIN_KEY
    default_env_scope: [dev]
"""
    cfg = _write_config(tmp_path, servers, "capabilities: {}")
    registry = load_registry(cfg)
    assert registry.servers["s1"].auth.extra_headers == {"x-brain-key": "S1_BRAIN_KEY"}


def test_extra_headers_reject_empty_env_name(tmp_path: Path) -> None:
    servers = """
servers:
  s1:
    display_name: S1
    transport: streamable_http
    endpoint_env: S1_URL
    auth:
      type: bearer
      secret_env: S1_TOKEN
      extra_headers:
        x-brain-key: ""
    default_env_scope: [dev]
"""
    cfg = _write_config(tmp_path, servers, "capabilities: {}")
    with pytest.raises(RegistryError, match="non-empty"):
        load_registry(cfg)


def test_stdio_env_injection_loads(tmp_path: Path) -> None:
    servers = """
servers:
  bd:
    display_name: Beads
    transport: stdio
    command: bd
    env_passthrough: [OP_SERVICE_ACCOUNT_TOKEN, PATH]
    env:
      BEADS_ACTOR: janus
    default_env_scope: [dev]
"""
    cfg = _write_config(tmp_path, servers, "capabilities: {}")
    server = load_registry(cfg).servers["bd"]
    assert server.env == {"BEADS_ACTOR": "janus"}
    assert server.env_passthrough == ["OP_SERVICE_ACCOUNT_TOKEN", "PATH"]


def test_http_transport_rejects_env_injection(tmp_path: Path) -> None:
    servers = """
servers:
  ob:
    display_name: OB
    transport: streamable_http
    endpoint_env: OB_URL
    env: {X: "y"}
    default_env_scope: [dev]
"""
    cfg = _write_config(tmp_path, servers, "capabilities: {}")
    with pytest.raises(RegistryError, match="must not set env"):
        load_registry(cfg)


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="not found"):
        load_registry(tmp_path)


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #
def test_hash_text_deterministic_and_prefixed() -> None:
    h1 = hash_text("hello")
    h2 = hash_text("hello")
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert h1 != hash_text("world")


def test_hash_schema_is_key_order_invariant() -> None:
    a = {"type": "object", "properties": {"x": {"type": "string"}}}
    b = {"properties": {"x": {"type": "string"}}, "type": "object"}
    assert hash_schema(a) == hash_schema(b)
    assert hash_schema(a) != hash_schema({"type": "array"})


# --------------------------------------------------------------------------- #
# SQLite operational cache + drift detection
# --------------------------------------------------------------------------- #
def test_schema_store_sync_populates_cache(tmp_path: Path) -> None:
    registry = load_registry(SEED_CONFIG)
    with SchemaStore(tmp_path / "data" / "janus.db") as store:
        written = store.sync_from_registry(registry)
        assert written == len(registry.servers) + len(registry.capabilities)
        assert store.count("servers") == len(registry.servers)
        assert store.count("capabilities") == len(registry.capabilities)


def test_schema_store_drift_detection(tmp_path: Path) -> None:
    cap = Capability(
        id="s1.cap",
        server_id="s1",
        downstream_tool_name="t",
        title="T",
        summary="s",
        risk="read_only",  # type: ignore[arg-type]
        env_scope=["dev"],  # type: ignore[list-item]
        approved=True,
        raw_description_hash=hash_text("original description"),
    )
    with SchemaStore(tmp_path / "data" / "janus.db") as store:
        # need a server row for the FK
        from janus.registry import Server

        server = Server(
            id="s1",
            display_name="S1",
            transport="streamable_http",  # type: ignore[arg-type]
            endpoint_env="S1_URL",
            default_env_scope=["dev"],  # type: ignore[list-item]
        )
        store.upsert_server(server)
        store.upsert_capability(cap)

        # same hash -> no drift
        assert not store.detect_drift(
            "s1.cap",
            raw_description_hash=hash_text("original description"),
            input_schema_hash=None,
        )
        # changed hash -> drift
        assert store.detect_drift(
            "s1.cap",
            raw_description_hash=hash_text("POISONED description"),
            input_schema_hash=None,
        )


def test_schema_store_drift_unknown_capability(tmp_path: Path) -> None:
    with SchemaStore(tmp_path / "data" / "janus.db") as store:
        with pytest.raises(KeyError):
            store.detect_drift(
                "missing", raw_description_hash=None, input_schema_hash=None
            )
