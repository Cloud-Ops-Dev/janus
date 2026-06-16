"""``janus-admin`` — operator CLI for the capability approval workflow.

Local, host-side control plane: it talks to the registry cache directly (the same
SQLite the live gateway reads), so approvals/quarantines take effect immediately
without a restart. Deliberately NOT a REST endpoint — approval is a human-only,
host-local action, never exposed over the network.

Commands:
    discover                          crawl downstreams, refresh observations
    list                              show every capability's lifecycle state
    pending                           show capabilities awaiting first approval
    approve <id>                      approve + lock observed descriptor as baseline
    quarantine-capability <id>        mark one capability uncallable
    quarantine-server <id>            mark every capability of a server uncallable
    diff <id> [--fetch]               show baseline-vs-observed hash delta; --fetch
                                      additionally prints the LIVE raw descriptor
                                      for the human operator to eyeball (never a model)

Config comes from the same env as the gateway (JANUS_CONFIG_DIR / JANUS_DATA_DIR).
Output is JSON on stdout; errors go to stderr with a non-zero exit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from typing import Any

from janus.admin.service import AdminError, AdminService
from janus.discovery import DiscoveryCrawler, DiscoveryReport
from janus.gateway import REGISTRY_DB_NAME, Gateway, GatewayConfig
from janus.registry.registry import load_registry
from janus.registry.schema_store import CapabilityState, SchemaStore


def _emit(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _state_dict(state: CapabilityState) -> dict[str, Any]:
    data = asdict(state)
    data["callable"] = state.callable
    return data


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="janus-admin", description="Janus approval workflow")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("discover", help="crawl downstreams and refresh observations")
    sub.add_parser("list", help="show every capability's lifecycle state")
    sub.add_parser("pending", help="show capabilities awaiting first approval")

    ap = sub.add_parser("approve", help="approve a capability + lock its baseline")
    ap.add_argument("capability_id")

    qc = sub.add_parser("quarantine-capability", help="mark a capability uncallable")
    qc.add_argument("capability_id")
    qc.add_argument("--reason", default="manual quarantine via janus-admin")

    qs = sub.add_parser("quarantine-server", help="mark a whole server uncallable")
    qs.add_argument("server_id")
    qs.add_argument("--reason", default="manual quarantine via janus-admin")

    df = sub.add_parser("diff", help="show baseline-vs-observed descriptor delta")
    df.add_argument("capability_id")
    df.add_argument(
        "--fetch",
        action="store_true",
        help="also fetch + print the live raw descriptor (human operator only)",
    )

    return p.parse_args(argv)


async def _discover(config: GatewayConfig) -> DiscoveryReport:
    gateway = Gateway.build(config)
    await gateway.connect()
    try:
        crawler = DiscoveryCrawler(
            gateway.deps.registry, gateway.manager, gateway.store
        )
        return await crawler.crawl()
    finally:
        await gateway.aclose()


async def _fetch_raw(config: GatewayConfig, server_id: str, tool_name: str) -> str | None:
    gateway = Gateway.build(config)
    await gateway.connect()
    try:
        for tool in await gateway.manager.list_tools(server_id):
            if tool.name == tool_name:
                return tool.description
        return None
    finally:
        await gateway.aclose()


def main(argv: list[str]) -> int:
    ns = _parse_args(argv)
    config = GatewayConfig.from_env()

    if ns.cmd == "discover":
        report = asyncio.run(_discover(config))
        _emit({"command": "discover", **report.summary()})
        return 0

    registry = load_registry(config.config_dir)
    store = SchemaStore(config.data_dir / REGISTRY_DB_NAME)
    try:
        store.sync_from_registry(registry)
        svc = AdminService(registry, store)
        if ns.cmd == "list":
            _emit({"command": "list",
                   "capabilities": [_state_dict(s) for s in svc.list_states()]})
        elif ns.cmd == "pending":
            _emit({"command": "pending",
                   "capabilities": [_state_dict(s) for s in svc.pending()]})
        elif ns.cmd == "approve":
            _emit({"command": "approve", "result": asdict(svc.approve(ns.capability_id))})
        elif ns.cmd == "quarantine-capability":
            svc.quarantine_capability(ns.capability_id, ns.reason)
            _emit({"command": "quarantine-capability",
                   "capability_id": ns.capability_id, "reason": ns.reason})
        elif ns.cmd == "quarantine-server":
            ids = svc.quarantine_server(ns.server_id, ns.reason)
            _emit({"command": "quarantine-server", "server_id": ns.server_id,
                   "reason": ns.reason, "quarantined": ids})
        elif ns.cmd == "diff":
            diff = svc.diff(ns.capability_id)
            payload: dict[str, Any] = {"command": "diff", "diff": asdict(diff)}
            if ns.fetch:
                cap = registry.capabilities.get(ns.capability_id)
                raw = (
                    asyncio.run(_fetch_raw(config, cap.server_id, cap.downstream_tool_name))
                    if cap is not None
                    else None
                )
                # Human-operator eyes only; clearly labeled untrusted, never a model.
                payload["live_raw_description"] = raw
                payload["_note"] = "live_raw_description is UNTRUSTED downstream text"
            _emit(payload)
    except AdminError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0
