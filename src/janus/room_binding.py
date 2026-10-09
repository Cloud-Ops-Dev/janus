"""Opaque per-session origin claim for ``room_approval`` grants.

The trusted Retinue gateway mints one value per (room, member, ACP MCP
session) with ``secrets.token_urlsafe(32)`` and sends it only as the MCP
HTTP header ``X-Retinue-Room-Binding``. That value is 43 characters from
the URL-safe base64 alphabet, with no padding and no whitespace.

Janus reads the header from the authenticated MCP request. Tool arguments,
model prose, and the REST call body cannot set it. A missing header is not
an error: in-process Hermes and Claude turns leave the claim unset so their
gateway can bind from the trusted tool result. Any other shape is rejected.
The rejection text is a fixed phrase and never includes the header value.
"""

from __future__ import annotations

import re
import secrets

from fastmcp.server.dependencies import get_http_request

# Starlette stores header names in lowercase. HTTP is case-insensitive.
ROOM_BINDING_HEADER = "x-retinue-room-binding"
ROOM_BINDING_LENGTH = 43
MALFORMED_ROOM_BINDING = "malformed room binding"
ROOM_BINDING_MISMATCH = "room binding does not match this MCP session"

_ROOM_BINDING_RE = re.compile(r"[A-Za-z0-9_-]{43}")
_REJECTIONS = frozenset({MALFORMED_ROOM_BINDING, ROOM_BINDING_MISMATCH})


class RoomBindingRejected(Exception):
    """The presented room-binding header cannot be used.

    ``str(exc)`` is one of the fixed phrases above. It does not contain the
    header value.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code if code in _REJECTIONS else MALFORMED_ROOM_BINDING)


def parse_room_binding(value: object) -> str:
    """Return a header value that matches the minted shape.

    Raises:
        RoomBindingRejected: the value is present but not exactly 43
            URL-safe base64 characters. The error text does not include it.
    """
    if isinstance(value, str) and _ROOM_BINDING_RE.fullmatch(value) is not None:
        return value
    raise RoomBindingRejected(MALFORMED_ROOM_BINDING) from None


def interpret_header_values(values: list[str]) -> str | None:
    """Map the raw header occurrences to one binding, or None if absent.

    Duplicate headers are rejected rather than joined, so a second value
    cannot hide behind a comma.
    """
    if not values:
        return None
    if len(values) != 1:
        raise RoomBindingRejected(MALFORMED_ROOM_BINDING) from None
    return parse_room_binding(values[0])


def room_bindings_equal(left: str, right: str) -> bool:
    """Constant-time equality for two binding strings."""
    left_b = left.encode()
    right_b = right.encode()
    if len(left_b) != len(right_b):
        secrets.compare_digest(left_b, left_b)
        return False
    return secrets.compare_digest(left_b, right_b)


def presented_room_binding() -> str | None:
    """Read ``X-Retinue-Room-Binding`` from the active MCP HTTP request.

    Returns None when this call is not inside an HTTP request (stdio and
    in-process turns) or the header is absent. A present but unusable value
    raises :class:`RoomBindingRejected`.

    FastMCP 3.4's streamable HTTP transport stores the Starlette request on
    the MCP request context for the duration of the tool call. ``get_http_request``
    returns that request, so the header is visible on the authenticated call
    and is independent of the tool arguments.
    """
    try:
        request = get_http_request()
    except RuntimeError:
        return None
    return interpret_header_values(request.headers.getlist(ROOM_BINDING_HEADER))
