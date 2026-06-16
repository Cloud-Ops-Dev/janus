"""Integration tests for the discovery crawler (infra-bpz.1).

These spin up the real ``_fake_downstream.py`` stdio server, crawl it, and assert
the descriptor observations / classifications / persistence. Async bodies run via
``asyncio.run`` (no pytest-asyncio dependency), matching the other suites.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from janus.discovery import DiscoveryCrawler, DiscoveryReport
from janus.downstream import DownstreamClientManager
from janus.registry import (
    Capability,
    EnvScope,
    Registry,
    RiskTier,
    SchemaStore,
    Server,
    Transport,
    hash_text,
)

FAKE = str(Path(__file__).parent / "_fake_downstream.py")


def _server(args: Sequence[str]) -> Server:
    return Server(
        id="fake",
        display_name="Fake",
        transport=Transport.STDIO,
        command=sys.executable,
        args=list(args),
        risk_ceiling=RiskTier.LOCAL_WRITE,
        default_env_scope=[EnvScope.DEV],
    )


def _cap(cid: str, tool: str, *, approved: bool = True) -> Capability:
    return Capability(
        id=cid,
        server_id="fake",
        downstream_tool_name=tool,
        title=cid,
        summary=cid,
        risk=RiskTier.READ_ONLY,
        env_scope=[EnvScope.DEV],
        approved=approved,
    )


def _run(
    registry: Registry,
    coro: Callable[[DownstreamClientManager], Awaitable[DiscoveryReport]],
    *,
    connect: Sequence[str] = ("fake",),
) -> DiscoveryReport:
    async def body() -> DiscoveryReport:
        mgr = DownstreamClientManager(registry.servers)
        async with mgr:
            for sid in connect:
                await mgr.connect_server(sid)
            return await coro(mgr)

    return asyncio.run(body())


def _crawl(
    registry: Registry, store: SchemaStore, *, connect: Sequence[str] = ("fake",)
) -> DiscoveryReport:
    return _run(
        registry,
        lambda mgr: DiscoveryCrawler(registry, mgr, store).crawl(),
        connect=connect,
    )


# --------------------------------------------------------------------------- #
# First crawl: discovery + persistence + TOFU baseline
# --------------------------------------------------------------------------- #
def test_first_crawl_discovers_stores_and_classifies_new(tmp_path: Path) -> None:
    registry = Registry(
        servers={"fake": _server([FAKE])},
        capabilities={
            "fake.echo": _cap("fake.echo", "echo"),
            "fake.add": _cap("fake.add", "add"),
        },
    )
    with SchemaStore(tmp_path / "data" / "janus.db") as store:
        store.sync_from_registry(registry)
        report = _crawl(registry, store)

        assert {o.capability_id for o in report.new} == {"fake.echo", "fake.add"}
        assert not report.changed and not report.missing

        # observed hashes + last_verified persisted; TOFU baseline locked.
        echo = store.get_state("fake.echo")
        assert echo is not None
        assert echo.present is True
        assert echo.observed_description_hash == hash_text("Echo the input text back unchanged.")
        assert echo.last_verified is not None
        assert echo.baseline_description_hash == echo.observed_description_hash
        assert echo.observed_schema_hash is not None


def test_pending_capability_gets_no_baseline(tmp_path: Path) -> None:
    """An unapproved capability is observed but its baseline stays unset."""
    registry = Registry(
        servers={"fake": _server([FAKE])},
        capabilities={"fake.echo": _cap("fake.echo", "echo", approved=False)},
    )
    with SchemaStore(tmp_path / "data" / "janus.db") as store:
        store.sync_from_registry(registry)
        report = _crawl(registry, store)
        assert report.new and not report.changed
        state = store.get_state("fake.echo")
        assert state is not None
        assert state.approved is False
        assert state.observed_description_hash is not None  # observed...
        assert state.baseline_description_hash is None  # ...but no trusted baseline


# --------------------------------------------------------------------------- #
# Second crawl: unchanged
# --------------------------------------------------------------------------- #
def test_second_crawl_is_unchanged(tmp_path: Path) -> None:
    registry = Registry(
        servers={"fake": _server([FAKE])},
        capabilities={"fake.echo": _cap("fake.echo", "echo")},
    )
    with SchemaStore(tmp_path / "data" / "janus.db") as store:
        store.sync_from_registry(registry)
        _crawl(registry, store)  # establishes baseline
        report = _crawl(registry, store)  # same descriptors
        assert {o.capability_id for o in report.unchanged} == {"fake.echo"}
        assert not report.new and not report.changed


# --------------------------------------------------------------------------- #
# Changed descriptor (two spawns via a rewritten desc-file)
# --------------------------------------------------------------------------- #
def test_changed_description_classified_as_changed(tmp_path: Path) -> None:
    desc_file = tmp_path / "echo_desc.txt"
    desc_file.write_text("Original echo description.", encoding="utf-8")
    registry = Registry(
        servers={"fake": _server([FAKE, "--desc-file", str(desc_file)])},
        capabilities={"fake.echo": _cap("fake.echo", "echo")},
    )
    with SchemaStore(tmp_path / "data" / "janus.db") as store:
        store.sync_from_registry(registry)
        first = _crawl(registry, store)
        assert {o.capability_id for o in first.new} == {"fake.echo"}

        # Poison the descriptor; a fresh spawn now advertises a new description.
        desc_file.write_text("POISONED echo description.", encoding="utf-8")
        second = _crawl(registry, store)
        assert {o.capability_id for o in second.changed} == {"fake.echo"}
        obs = second.changed[0]
        assert obs.observed_description_hash != obs.baseline_description_hash


# --------------------------------------------------------------------------- #
# Missing tool + unregistered tool + unreachable server
# --------------------------------------------------------------------------- #
def test_missing_downstream_tool(tmp_path: Path) -> None:
    registry = Registry(
        servers={"fake": _server([FAKE])},
        capabilities={"fake.gone": _cap("fake.gone", "does_not_exist")},
    )
    with SchemaStore(tmp_path / "data" / "janus.db") as store:
        store.sync_from_registry(registry)
        report = _crawl(registry, store)
        assert {o.capability_id for o in report.missing} == {"fake.gone"}
        state = store.get_state("fake.gone")
        assert state is not None and state.present is False


def test_unregistered_tool_reported(tmp_path: Path) -> None:
    registry = Registry(
        servers={"fake": _server([FAKE, "--extra-tool"])},
        capabilities={
            "fake.echo": _cap("fake.echo", "echo"),
            "fake.add": _cap("fake.add", "add"),
        },
    )
    with SchemaStore(tmp_path / "data" / "janus.db") as store:
        store.sync_from_registry(registry)
        report = _crawl(registry, store)
        # only the un-claimed downstream tool surfaces as unregistered.
        assert report.unregistered.get("fake") == ["extra"]


def test_unreachable_server_yields_error_and_missing(tmp_path: Path) -> None:
    registry = Registry(
        servers={"fake": _server([FAKE])},
        capabilities={"fake.echo": _cap("fake.echo", "echo")},
    )
    with SchemaStore(tmp_path / "data" / "janus.db") as store:
        store.sync_from_registry(registry)
        # connect nothing -> list_tools raises DownstreamNotConnected
        report = _crawl(registry, store, connect=())
        assert "fake" in report.server_errors
        assert {o.capability_id for o in report.missing} == {"fake.echo"}


def test_report_summary_is_hash_free(tmp_path: Path) -> None:
    registry = Registry(
        servers={"fake": _server([FAKE])},
        capabilities={"fake.echo": _cap("fake.echo", "echo")},
    )
    with SchemaStore(tmp_path / "data" / "janus.db") as store:
        store.sync_from_registry(registry)
        report = _crawl(registry, store)
        summary = report.summary()
        assert summary["counts"]["new"] == 1
        # the model-safe digest carries counts, never raw descriptor text.
        assert "Echo the input text" not in repr(summary)
