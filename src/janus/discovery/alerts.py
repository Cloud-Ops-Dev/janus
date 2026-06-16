"""Alerting for supply-chain drift (design §5.8 / §10, infra-bpz.3).

When the discovery crawler auto-quarantines a drifted capability, the operator
must be told out-of-band. The channel is injectable (the :class:`Alerter`
protocol) so the drift monitor is testable without a network, and production uses
a best-effort Discord webhook.

Headless-safe + self-sufficient (constitution §12): the webhook URL comes from an
``EnvironmentFile`` value (``JANUS_DISCORD_WEBHOOK_URL``) — env-var NAME only in
this public repo, never a committed URL — and the POST uses the stdlib (no shell
session, no extra dependency). Alerting is best-effort: a failed/absent channel
never blocks the quarantine, which is the actual safety control.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

WEBHOOK_ENV = "JANUS_DISCORD_WEBHOOK_URL"
_DISCORD_CONTENT_LIMIT = 1990  # Discord caps message content at 2000 chars.


@runtime_checkable
class Alerter(Protocol):
    def send(self, message: str) -> bool:
        """Deliver an alert. Returns True on success; never raises."""
        ...


class NullAlerter:
    """No channel configured. Drift is still quarantined + logged, just not pinged."""

    def send(self, message: str) -> bool:
        return False


class WebhookAlerter:
    """Best-effort Discord webhook POST (stdlib only; headless-safe)."""

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    def send(self, message: str) -> bool:
        # Defense-in-depth: only ever open http(s); reject file:/custom schemes.
        if not self._url.startswith(("http://", "https://")):
            return False
        data = json.dumps({"content": message[:_DISCORD_CONTENT_LIMIT]}).encode("utf-8")
        # URL is an operator-configured webhook from the systemd EnvironmentFile,
        # and the scheme is guarded above.
        req = urllib.request.Request(  # noqa: S310 — scheme-guarded, trusted op-supplied webhook
            self._url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10):  # noqa: S310 — see above
                return True
        except (urllib.error.URLError, OSError):
            return False  # best-effort: never let a dead webhook block quarantine.


def build_alerter(environ: Mapping[str, str]) -> Alerter:
    """A WebhookAlerter when ``JANUS_DISCORD_WEBHOOK_URL`` is set, else NullAlerter."""
    url = environ.get(WEBHOOK_ENV, "").strip()
    return WebhookAlerter(url) if url else NullAlerter()
