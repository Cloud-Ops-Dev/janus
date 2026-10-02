"""Read-only public GitHub issue lookup.

Covers a valid allowlisted read, an unlisted repository, 404, rate limit,
a malformed issue number, and the absence of any write capability. HTTP is
stubbed; these tests do not contact GitHub.
"""

from __future__ import annotations

import asyncio
import io
import json
import urllib.error
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pytest

from janus.audit import InMemoryAuditSink
from janus.broker import Broker
from janus.discovery.crawler import DiscoveryCrawler
from janus.downstream import DownstreamCallError, DownstreamClientManager
from janus.downstream.github_issues import (
    HANDLER_ID,
    MAX_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    GithubIssueLookup,
    _RefuseRedirects,
)
from janus.gateway import GatewayConfig, check_environment
from janus.policy import (
    DEFAULT_PROFILES,
    AgentProfile,
    ProfilePolicyEngine,
    TrifectaGuard,
    TrifectaLeg,
    legs_for,
)
from janus.registry import (
    AuthType,
    EnvScope,
    RegistryError,
    RiskTier,
    SchemaStore,
    Transport,
    TrustLevel,
    load_registry,
)
from janus.registry.registry import NATIVE_HANDLERS, Capability, Registry, Server
from janus.security.output_sanitizer import OutputSanitizer
from janus.security.secret_redactor import SecretRedactor
from janus.server_rest import BrokerDeps, HostIdentity
from janus.session_mcp import McpSessionPool

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = REPO_ROOT / "config"
# Janus token label for the shared Retinue identity. Not a room id.
PRINCIPAL = "retinue"
PROFILE = "clay_blade_assistant"
CAP_ID = "github_public.issue_get"
BODY_SENTINEL = "DO_NOT_LEAK_ISSUE_BODY"


def _registry() -> Registry:
    server = Server(
        id="github_public",
        display_name="GitHub public issues",
        transport=Transport.NATIVE,
        trust_level=TrustLevel.THIRD_PARTY,
        risk_ceiling=RiskTier.READ_ONLY,
        default_env_scope=[EnvScope.DEV, EnvScope.TEST, EnvScope.PROD_SAFE],
    )
    cap = Capability(
        id=CAP_ID,
        server_id="github_public",
        downstream_tool_name="issue_get",
        handler=HANDLER_ID,
        title="Read a public GitHub issue",
        summary="Read title state updated URL of one public GitHub issue",
        risk=RiskTier.READ_ONLY,
        env_scope=[EnvScope.DEV, EnvScope.TEST, EnvScope.PROD_SAFE],
        approved=True,
        allowed_identities=[PRINCIPAL],
        repo_allowlist=["novique-ai/retinue"],
        tags=["github", "issues"],
    )
    return Registry(servers={server.id: server}, capabilities={cap.id: cap})


def _payload(number: int, state: str, title: str, updated_at: str) -> bytes:
    return json.dumps(
        {
            "title": title,
            "state": state,
            "updated_at": updated_at,
            "html_url": f"https://github.com/novique-ai/retinue/issues/{number}",
            "body": BODY_SENTINEL,
            "comments": 3,
        }
    ).encode()


class _Resp:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.headers: dict[str, str] = {}
        self._body = body

    def read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _http_error(url: str, code: int, body: bytes = b"", **headers: str) -> urllib.error.HTTPError:
    hdrs = EmailMessage()
    for key, value in headers.items():
        hdrs[key] = value
    return urllib.error.HTTPError(url, code, "err", hdrs, io.BytesIO(body))


def _broker(
    identity: str,
    urlopen: Any,
    *,
    profile: str = "default_assistant",
    env: EnvScope = EnvScope.PROD_SAFE,
    sanitizer: OutputSanitizer | None = None,
) -> tuple[Broker, list[tuple[str, str, float]]]:
    seen: list[tuple[str, str, float]] = []

    def opener(req: urllib.request.Request, timeout: float) -> Any:
        seen.append((req.full_url, req.get_method(), timeout))
        assert req.get_method() == "GET"
        names = {key.casefold() for key, _value in req.header_items()}
        assert "authorization" not in names
        return urlopen(req, timeout)

    registry = _registry()
    broker = Broker(
        registry,
        DownstreamClientManager(registry.servers),
        ProfilePolicyEngine(),
        InMemoryAuditSink(),
        sanitizer=sanitizer,
        session_id=identity,
        profile=profile,
        default_env=env,
        issue_lookup=GithubIssueLookup(opener),
    )
    return broker, seen


def _call(broker: Broker, arguments: dict[str, Any], env: EnvScope | None = None) -> dict[str, Any]:
    return asyncio.run(
        broker.capability_call(CAP_ID, arguments, reason="check issue state", env=env)
    )


def test_timeout_is_bounded() -> None:
    assert 0 < REQUEST_TIMEOUT_SECONDS <= MAX_TIMEOUT_SECONDS
    assert REQUEST_TIMEOUT_SECONDS == 10.0
    with pytest.raises(ValueError, match="timeout"):
        GithubIssueLookup(timeout=MAX_TIMEOUT_SECONDS + 1)


def test_seed_is_read_only_issue_get_for_one_identity_and_repo() -> None:
    registry = load_registry(SEED)
    assert HANDLER_ID in NATIVE_HANDLERS
    cap = registry.capabilities[CAP_ID]
    server = registry.servers["github_public"]
    assert cap.handler == HANDLER_ID
    assert cap.risk is RiskTier.READ_ONLY
    assert cap.approved and not cap.quarantined
    assert cap.allowed_identities == [PRINCIPAL]
    assert cap.repo_allowlist == ["novique-ai/retinue"]
    assert EnvScope.PROD not in cap.env_scope
    assert server.transport is Transport.NATIVE
    assert server.trust_level is TrustLevel.THIRD_PARTY
    assert server.risk_ceiling is RiskTier.READ_ONLY
    assert server.auth.type is AuthType.NONE
    assert server.endpoint_env is None
    github = registry.capabilities_for_server("github_public")
    assert [item.id for item in github] == [CAP_ID]
    assert github[0].downstream_tool_name == "issue_get"
    for item in registry.capabilities.values():
        if item.server_id == "github_public" or (item.handler or "").startswith("github"):
            assert item.risk is RiskTier.READ_ONLY
            assert item.downstream_tool_name == "issue_get"
    text = (REPO_ROOT / "src" / "janus" / "downstream" / "github_issues.py").read_text()
    assert text.count('method="GET"') == 1
    for verb in ("POST", "PUT", "PATCH", "DELETE"):
        assert verb not in text
    assert legs_for(cap, server) == frozenset({TrifectaLeg.UNTRUSTED_CONTENT})


def test_principal_can_search_and_describe_issue_get() -> None:
    def urlopen(_req: urllib.request.Request, _timeout: float) -> _Resp:
        raise AssertionError("describe and search must not open a request")

    broker, seen = _broker(PRINCIPAL, urlopen)
    found = broker.capability_search("public github issue")
    ids = [row["capability_id"] for row in found["results"]]
    assert CAP_ID in ids
    described = asyncio.run(broker.capability_describe(CAP_ID))
    assert described["schema_error"] is None
    assert set(described["input_schema"]["properties"]) == {"repository", "issue_number"}
    assert "url" not in described["input_schema"]["properties"]
    assert described["input_schema"]["additionalProperties"] is False
    assert described["policy"]["decision"] == "allow"
    assert described["risk"] == "read_only"
    explain = broker.policy_explain(CAP_ID)
    assert explain["decision"] == "allow"
    assert explain["identity"] == PRINCIPAL
    assert seen == []


def test_valid_lookup_reports_open_and_closed() -> None:
    def urlopen(req: urllib.request.Request, timeout: float) -> _Resp:
        assert timeout == REQUEST_TIMEOUT_SECONDS
        number = int(req.full_url.rsplit("/", 1)[-1])
        assert req.full_url == (
            f"https://api.github.com/repos/novique-ai/retinue/issues/{number}"
        )
        if number == 253:
            body = _payload(253, "open", "Inbox triage", "2026-09-28T15:04:05Z")
        else:
            assert number == 234
            body = _payload(234, "closed", "Shipped the cutover", "2026-08-01T00:00:00Z")
        return _Resp(200, body)

    broker, seen = _broker(PRINCIPAL, urlopen)
    opened = _call(
        broker, {"repository": "Novique-AI/Retinue", "issue_number": 253, "url": "https://evil.example"}
    )
    closed = _call(broker, {"repository": "novique-ai/retinue", "issue_number": 234})
    assert opened["status"] == "ok"
    assert opened["structured"] == {
        "repository": "novique-ai/retinue",
        "issue_number": 253,
        "title": "Inbox triage",
        "state": "OPEN",
        "updated_at": "2026-09-28T15:04:05Z",
        "html_url": "https://github.com/novique-ai/retinue/issues/253",
    }
    assert closed["structured"]["state"] == "CLOSED"
    assert closed["structured"]["title"] == "Shipped the cutover"
    assert closed["structured"]["updated_at"] == "2026-08-01T00:00:00Z"
    assert closed["structured"]["html_url"] == "https://github.com/novique-ai/retinue/issues/234"
    for result in (opened, closed):
        assert BODY_SENTINEL not in result["text"]
        assert BODY_SENTINEL not in json.dumps(result["structured"])
    assert all("evil.example" not in url for url, _method, _timeout in seen)
    assert [url for url, _method, _timeout in seen] == [
        "https://api.github.com/repos/novique-ai/retinue/issues/253",
        "https://api.github.com/repos/novique-ai/retinue/issues/234",
    ]
    entries = broker.audit_recent()["entries"]
    # Most recent first. The open lookup is the one that also carried an ignored url key.
    assert entries[1]["arg_keys"] == ["issue_number", "repository", "url"]
    assert entries[1]["decision"] == "allow"
    assert all("Inbox triage" not in json.dumps(entry) for entry in entries)


def test_unlisted_repository_is_denied_without_a_request() -> None:
    def urlopen(_req: urllib.request.Request, _timeout: float) -> _Resp:
        raise AssertionError("unlisted repository must not be requested")

    broker, seen = _broker(PRINCIPAL, urlopen)
    for repository in (
        "other/repo",
        "novique-ai/other",
        "https://github.com/novique-ai/retinue",
        "novique-ai/retinue.git",
        "novique-ai/retinue/issues/253",
        "../novique-ai/retinue",
        "",
    ):
        out = _call(broker, {"repository": repository, "issue_number": 253})
        assert out["status"] == "denied"
        assert out["error_code"] == "repo_denied"
        assert "allowlist" in out["reason"]
    assert seen == []


@pytest.mark.parametrize(
    "issue_number",
    [None, "253", 0, -1, True, 1.5, 1_000_000_001],
)
def test_malformed_issue_number_does_not_call_github(issue_number: object) -> None:
    def urlopen(_req: urllib.request.Request, _timeout: float) -> _Resp:
        raise AssertionError("malformed issue number must not be requested")

    broker, seen = _broker(PRINCIPAL, urlopen)
    arguments: dict[str, Any] = {"repository": "novique-ai/retinue"}
    if issue_number is not None:
        arguments["issue_number"] = issue_number
    out = _call(broker, arguments)
    assert out["status"] == "error"
    assert out["error_code"] == "invalid_issue"
    assert seen == []


def test_not_found_rate_limit_and_transport_are_explicit() -> None:
    cases: list[tuple[str, Any, str]] = []

    def not_found(req: urllib.request.Request, _timeout: float) -> Any:
        raise _http_error(req.full_url, 404, b'{"message":"Not Found"}')

    def rate_429(req: urllib.request.Request, _timeout: float) -> Any:
        raise _http_error(req.full_url, 429, b"")

    def rate_403(req: urllib.request.Request, _timeout: float) -> Any:
        raise _http_error(req.full_url, 403, b'{"message":"API rate limit exceeded"}',
                          **{"X-RateLimit-Remaining": "0"})

    def timed_out(_req: urllib.request.Request, _timeout: float) -> Any:
        raise TimeoutError()

    def broken(_req: urllib.request.Request, _timeout: float) -> Any:
        raise urllib.error.URLError("name resolution failed")

    def redirected(req: urllib.request.Request, _timeout: float) -> Any:
        raise _http_error(req.full_url, 302, b"")

    cases.extend(
        [
            ("not_found", not_found, "not found"),
            ("rate_limited", rate_429, "rate limit"),
            ("rate_limited", rate_403, "rate limit"),
            ("transport", timed_out, "timed out"),
            ("transport", broken, "failed"),
            ("transport", redirected, "redirect"),
        ]
    )
    for code, opener, snippet in cases:
        broker, _seen = _broker(PRINCIPAL, opener)
        out = _call(broker, {"repository": "novique-ai/retinue", "issue_number": 999})
        assert out["status"] == "error"
        assert out["error_code"] == code
        assert snippet in out["error"]
        assert BODY_SENTINEL not in out["error"]
        assert "structured" not in out


def test_other_identity_and_prod_are_denied() -> None:
    def urlopen(_req: urllib.request.Request, _timeout: float) -> _Resp:
        raise AssertionError("denied caller must not be requested")

    other, seen = _broker("some-other-host", urlopen)
    assert other.capability_search("github issue")["results"] == []
    described = asyncio.run(other.capability_describe(CAP_ID))
    assert described["policy"]["decision"] == "deny"
    denied = _call(other, {"repository": "novique-ai/retinue", "issue_number": 253})
    assert denied["status"] == "denied"
    assert "not permitted" in denied["reason"]
    room, room_seen = _broker(PRINCIPAL, urlopen)
    prod = _call(
        room, {"repository": "novique-ai/retinue", "issue_number": 253}, env=EnvScope.PROD
    )
    assert prod["status"] == "denied"
    assert "environment" in prod["reason"]
    assert seen == []
    assert room_seen == []


def _retinue_profiles() -> dict[str, AgentProfile]:
    """Host profile name, with only the read_only grant this lookup needs.

    ``clay_blade_assistant`` is declared on the Retinue deployment, not in the
    public seed. The test injects that name so the MCP path can reach the
    capability. It does not copy the deployment profile's other risk tiers.
    """
    profiles = dict(DEFAULT_PROFILES)
    profiles[PROFILE] = AgentProfile(
        name=PROFILE,
        allowed_env=frozenset({EnvScope.DEV, EnvScope.PROD_SAFE}),
        allow=frozenset({RiskTier.READ_ONLY}),
        confirm=frozenset(),
    )
    return profiles


def test_mcp_session_principal_reaches_issue_get_other_principal_denied() -> None:
    """Session registry authorizes the token label, not the MCP session key.

    ``McpSessionPool.state_for`` is given HostIdentity("retinue",
    profile="clay_blade_assistant") plus a session id. Search, describe, and
    call reach ``github_public.issue_get``. A second principal is denied, including
    one whose session id is the allowed label and one whose session key would
    prefix-parse as ``retinue``. Audit rows stay on ``retinue:mcp:<session-id>``.
    """
    seen: list[tuple[str, str, float]] = []

    def urlopen(req: urllib.request.Request, timeout: float) -> _Resp:
        seen.append((req.full_url, req.get_method(), timeout))
        assert req.get_method() == "GET"
        names = {key.casefold() for key, _value in req.header_items()}
        assert "authorization" not in names
        assert req.full_url == (
            "https://api.github.com/repos/novique-ai/retinue/issues/253"
        )
        body = _payload(253, "open", "Inbox triage", "2026-09-28T15:04:05Z")
        return _Resp(200, body)

    registry = load_registry(SEED)
    audit = InMemoryAuditSink()
    trifecta = TrifectaGuard()
    deps = BrokerDeps(
        registry=registry,
        manager=DownstreamClientManager(registry.servers),
        policy=ProfilePolicyEngine(_retinue_profiles()),
        audit=audit,
        trifecta=trifecta,
        default_env=EnvScope.PROD_SAFE,
        issue_lookup=GithubIssueLookup(urlopen),
    )
    pool = McpSessionPool(deps)
    retinue = HostIdentity(PRINCIPAL, profile=PROFILE)
    room_session = "obd-inbox-988cd9"
    room = pool.state_for(retinue, room_session)
    assert room is pool.state_for(retinue, room_session)
    assert room.key == f"{PRINCIPAL}:mcp:{room_session}"
    explain = room.broker.policy_explain(CAP_ID)
    assert explain["identity"] == PRINCIPAL
    assert explain["decision"] == "allow"

    found = room.broker.capability_search("public github issue", max_results=50)
    ids = [row["capability_id"] for row in found["results"]]
    assert CAP_ID in ids
    described = asyncio.run(room.broker.capability_describe(CAP_ID))
    assert described["schema_error"] is None
    assert described["policy"]["decision"] == "allow"
    assert set(described["input_schema"]["properties"]) == {"repository", "issue_number"}
    assert seen == []

    arguments = {"repository": "novique-ai/retinue", "issue_number": 253}
    opened = asyncio.run(
        room.broker.capability_call(CAP_ID, arguments, reason="check issue state")
    )
    assert opened["status"] == "ok"
    assert opened["structured"]["state"] == "OPEN"
    assert BODY_SENTINEL not in opened["text"]
    assert BODY_SENTINEL not in json.dumps(opened["structured"])
    assert trifecta.session_legs(room.key) == {TrifectaLeg.UNTRUSTED_CONTENT}
    assert trifecta.session_legs(PRINCIPAL) == frozenset()

    other_room = pool.state_for(retinue, "other-room")
    assert other_room.key == f"{PRINCIPAL}:mcp:other-room"
    assert other_room.broker is not room.broker
    shared = asyncio.run(
        other_room.broker.capability_call(
            CAP_ID, arguments, reason="shared identity, other room"
        )
    )
    assert shared["status"] == "ok"
    room_audit = room.broker.audit_recent()["entries"]
    other_audit = other_room.broker.audit_recent()["entries"]
    assert [entry["reason"] for entry in room_audit] == ["check issue state"]
    assert [entry["reason"] for entry in other_audit] == ["shared identity, other room"]
    assert {entry.session_id for entry in audit.recent(20)} == {room.key, other_room.key}
    assert trifecta.session_legs(other_room.key) == {TrifectaLeg.UNTRUSTED_CONTENT}
    assert trifecta.session_legs(PRINCIPAL) == frozenset()

    def denied(identity: HostIdentity, session_id: str) -> None:
        state = pool.state_for(identity, session_id)
        assert state.key == f"{identity.label}:mcp:{session_id}"
        denial = state.broker.policy_explain(CAP_ID)
        assert denial["identity"] == identity.label
        assert denial["decision"] == "deny"
        searched = state.broker.capability_search("public github issue", max_results=50)
        found_ids = [row["capability_id"] for row in searched["results"]]
        assert CAP_ID not in found_ids
        described_deny = asyncio.run(state.broker.capability_describe(CAP_ID))
        assert described_deny["policy"]["decision"] == "deny"
        assert "not permitted" in described_deny["policy"]["reason"]
        called = asyncio.run(
            state.broker.capability_call(CAP_ID, arguments, reason="must not run")
        )
        assert called["status"] == "denied"
        assert "not permitted" in called["reason"]
        assert state.broker.audit_recent()["entries"][0]["decision"] == "deny"
        assert trifecta.session_legs(state.key) == frozenset()

    requests_before_denies = len(seen)
    denied(HostIdentity("other-principal", profile=PROFILE), PRINCIPAL)
    # Splitting this session key on ":mcp:" would yield the allowed principal.
    denied(HostIdentity("retinue:mcp:evil", profile=PROFILE), "room-1")
    assert len(seen) == requests_before_denies
    assert [url for url, _method, _timeout in seen] == [
        "https://api.github.com/repos/novique-ai/retinue/issues/253",
        "https://api.github.com/repos/novique-ai/retinue/issues/253",
    ]

    rest = deps.broker_for(retinue)
    assert rest.policy_explain(CAP_ID)["identity"] == PRINCIPAL
    rest_out = asyncio.run(
        rest.capability_call(CAP_ID, arguments, reason="rest principal")
    )
    assert rest_out["status"] == "ok"
    rest_ids = {
        entry.session_id
        for entry in audit.recent(20)
        if entry.reason == "rest principal"
    }
    assert rest_ids == {PRINCIPAL}
    assert trifecta.session_legs(PRINCIPAL) == {TrifectaLeg.UNTRUSTED_CONTENT}
    rest_other = deps.broker_for(HostIdentity("other-principal", profile=PROFILE))
    rest_denied = asyncio.run(
        rest_other.capability_call(CAP_ID, arguments, reason="rest other")
    )
    assert rest_denied["status"] == "denied"
    assert "not permitted" in rest_denied["reason"]
    assert len(seen) == 3


def test_github_html_url_casing_is_accepted() -> None:
    def urlopen(_req: urllib.request.Request, _timeout: float) -> _Resp:
        body = json.dumps(
            {
                "title": "Inbox triage",
                "state": "open",
                "updated_at": "2026-09-28T15:04:05Z",
                "html_url": "https://github.com/Novique-AI/retinue/issues/253",
            }
        ).encode()
        return _Resp(200, body)

    broker, seen = _broker(PRINCIPAL, urlopen)
    out = _call(broker, {"repository": "novique-ai/retinue", "issue_number": 253})
    assert out["status"] == "ok"
    assert out["structured"]["html_url"] == "https://github.com/Novique-AI/retinue/issues/253"
    assert seen[0][0] == "https://api.github.com/repos/novique-ai/retinue/issues/253"


def test_third_party_title_is_wrapped_as_untrusted() -> None:
    def urlopen(_req: urllib.request.Request, _timeout: float) -> _Resp:
        return _Resp(200, _payload(253, "open", "Inbox triage", "2026-09-28T15:04:05Z"))

    broker, _seen = _broker(PRINCIPAL, urlopen, sanitizer=OutputSanitizer(SecretRedactor()))
    out = _call(broker, {"repository": "novique-ai/retinue", "issue_number": 253})
    assert "UNTRUSTED EXTERNAL CONTENT" in out["text"]
    assert "Inbox triage" in out["text"]
    assert out["structured"]["state"] == "OPEN"


def test_native_server_is_not_an_mcp_session(tmp_path: Path) -> None:
    async def body() -> None:
        registry = _registry()
        manager = DownstreamClientManager(registry.servers)
        store = SchemaStore(tmp_path / "registry.db")
        store.sync_from_registry(registry)
        try:
            async with manager:
                connected = await manager.connect_all()
                assert connected == ["github_public"]
                assert manager.connected_servers == ["github_public"]
                assert manager._sessions == {}
                health = await manager.health()
                assert health["github_public"].connected is True
                assert health["github_public"].error is None
                one = await manager.health("github_public")
                assert one["github_public"].connected is True
                assert one["github_public"].error is None
                with pytest.raises(DownstreamCallError, match="in-process handler"):
                    await manager.call("github_public", "issue_get", {})
                report = await DiscoveryCrawler(registry, manager, store).crawl()
            assert report.observations == []
            assert report.server_errors == {}
        finally:
            store.close()

    asyncio.run(body())


def test_native_server_rejects_connection_fields_and_writes(tmp_path: Path) -> None:
    native = """
servers:
  github_public:
    display_name: GitHub public issues
    transport: native
    endpoint_env: GITHUB_URL
    trust_level: third_party
    risk_ceiling: read_only
    default_env_scope: [dev]
"""
    with pytest.raises(RegistryError, match="must not declare a connection"):
        load_registry(_write(tmp_path / "endpoint", native, "capabilities: {}\n"))

    write_cap = """
capabilities:
  github_public.issue_create:
    server_id: github_public
    downstream_tool_name: issue_create
    title: Create a GitHub issue
    summary: Open an issue.
    risk: local_write
    env_scope: [dev]
    approved: true
"""
    server = """
servers:
  github_public:
    display_name: GitHub public issues
    transport: native
    trust_level: third_party
    risk_ceiling: read_only
    default_env_scope: [dev, test, prod_safe]
"""
    with pytest.raises(RegistryError, match="risk_ceiling"):
        load_registry(_write(tmp_path / "write", server, write_cap))

    unknown = """
capabilities:
  github_public.issue_get:
    server_id: github_public
    downstream_tool_name: issue_create
    handler: github_public.issue_create
    title: Create a GitHub issue
    summary: Open an issue.
    risk: read_only
    env_scope: [dev]
    approved: true
    allowed_identities: [retinue]
"""
    with pytest.raises(RegistryError, match="unknown handler"):
        load_registry(_write(tmp_path / "unknown", server, unknown))


def test_redirect_handler_does_not_return_a_new_request() -> None:
    handler = _RefuseRedirects()
    request = urllib.request.Request("https://api.github.com/repos/novique-ai/retinue/issues/1")
    with pytest.raises(urllib.error.HTTPError, match="redirect refused") as caught:
        handler.redirect_request(
            request, None, 302, "Found", EmailMessage(), "https://evil.example/steal"
        )
    assert caught.value.code == 302


def test_check_environment_does_not_require_a_github_secret(tmp_path: Path) -> None:
    config = GatewayConfig(config_dir=SEED, data_dir=tmp_path / "data")
    environ = {
        "JANUS_TOKENS": "t=retinue:clay_blade_assistant",
        "JANUS_OPEN_BRAIN_URL": "https://example.invalid/ob",
        "JANUS_OPEN_BRAIN_TOKEN": "token",
        "JANUS_OPEN_BRAIN_KEY": "key",
        "JANUS_BEADS_RO_URL": "https://example.invalid/beads",
        "JANUS_BEADS_RO_TOKEN": "token",
        "JANUS_BEADS_OP_URL": "https://example.invalid/beads-op",
        "JANUS_BEADS_OP_TOKEN": "token",
        "JANUS_PAPERCLIP_URL": "https://example.invalid/paperclip",
        "JANUS_PAPERCLIP_TOKEN": "token",
    }
    problems = check_environment(config, environ)
    assert problems == []


def _write(directory: Path, servers: str, capabilities: str) -> Path:
    directory.mkdir(parents=True)
    (directory / "servers.yaml").write_text(servers, encoding="utf-8")
    (directory / "capabilities.yaml").write_text(capabilities, encoding="utf-8")
    return directory
