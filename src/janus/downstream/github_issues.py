"""Read-only lookup of one public GitHub issue.

The only network operation is ``GET https://api.github.com/repos/{owner}/{repo}/issues/{n}``
for a repository on the capability's explicit allowlist. There is no URL
parameter, no credential, and no create/update/comment path. Callers receive
title, state (``OPEN`` or ``CLOSED``), ``updated_at``, and ``html_url``.

Redirects are refused. The request timeout is bounded. A repository that is
not on the allowlist is rejected before any socket is opened.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from email.message import Message
from typing import Any, NoReturn

from janus.downstream.client_manager import DownstreamResult
from janus.registry.registry import Capability

HANDLER_ID = "github_public.issue_get"
API_ORIGIN = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_TIMEOUT_SECONDS = 15.0
MAX_ISSUE_NUMBER = 1_000_000_000
MAX_BODY_BYTES = 65_536
_USER_AGENT = "janus-public-issue-lookup"
_UPDATED_AT_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)

ISSUE_GET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "repository": {
            "type": "string",
            "description": "Allowlisted public repository, as owner/name.",
        },
        "issue_number": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_ISSUE_NUMBER,
            "description": "GitHub issue number.",
        },
    },
    "required": ["repository", "issue_number"],
}

Urlopen = Callable[[urllib.request.Request, float], Any]


class IssueLookupError(Exception):
    """A failed issue read. ``denied`` is an allowlist rejection, not a transport failure."""

    def __init__(self, code: str, message: str, *, denied: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.denied = denied


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect. The issue URL is fixed; a redirect is a different request."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def _default_urlopen(req: urllib.request.Request, timeout: float) -> Any:
    # Host is the constant API origin; path segments were allowlist-checked.
    opener = urllib.request.build_opener(_RefuseRedirects)
    return opener.open(req, timeout=timeout)  # noqa: S310


class GithubIssueLookup:
    """GET one allowlisted public issue. Inject ``urlopen`` in tests; do not point it at a URL."""

    def __init__(
        self,
        urlopen: Urlopen | None = None,
        *,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if timeout <= 0 or timeout > MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"GitHub issue timeout must be in (0, {MAX_TIMEOUT_SECONDS}] seconds"
            )
        self._urlopen = urlopen or _default_urlopen
        self.timeout = timeout

    async def lookup(self, cap: Capability, arguments: Mapping[str, Any]) -> dict[str, Any]:
        repository, issue_number = _parse_arguments(cap, arguments)
        return await asyncio.to_thread(self._fetch, repository, issue_number)

    def _fetch(self, repository: str, issue_number: int) -> dict[str, Any]:
        owner, name = repository.split("/", 1)
        url = f"{API_ORIGIN}/repos/{owner}/{name}/issues/{issue_number}"
        request = urllib.request.Request(  # noqa: S310 — constant host, allowlisted path
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": _USER_AGENT,
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="GET",
        )
        try:
            with self._urlopen(request, self.timeout) as response:
                status = int(getattr(response, "status", 200))
                raw = _read_bounded(response)
        except urllib.error.HTTPError as exc:
            _http_error(exc, repository, issue_number)
        except TimeoutError:
            raise IssueLookupError("transport", "GitHub issue request timed out") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise IssueLookupError("transport", "GitHub issue request timed out") from None
            raise IssueLookupError(
                "transport", f"GitHub issue request failed ({type(exc.reason).__name__})"
            ) from None
        except OSError as exc:
            raise IssueLookupError(
                "transport", f"GitHub issue request failed ({type(exc).__name__})"
            ) from None
        if status != 200:
            raise IssueLookupError("transport", f"GitHub issue request failed: HTTP {status}")
        return _project(raw, repository, issue_number)


def handler_input_schema(cap: Capability) -> dict[str, Any]:
    """Static input schema for a native handler. Unknown handlers fail closed."""
    if cap.handler != HANDLER_ID:
        raise IssueLookupError(
            "unknown_handler", f"no in-process handler '{cap.handler}'"
        )
    return copy.deepcopy(ISSUE_GET_SCHEMA)


async def invoke_handler(
    lookup: GithubIssueLookup, cap: Capability, arguments: Mapping[str, Any]
) -> DownstreamResult:
    if cap.handler != HANDLER_ID:
        raise IssueLookupError(
            "unknown_handler", f"no in-process handler '{cap.handler}'"
        )
    payload = await lookup.lookup(cap, arguments)
    text = (
        f"{payload['repository']}#{payload['issue_number']} {payload['state']}\n"
        f"{payload['title']}\n"
        f"{payload['html_url']}"
    )
    return DownstreamResult(is_error=False, text=text, structured=payload)


def _parse_arguments(cap: Capability, arguments: Mapping[str, Any]) -> tuple[str, int]:
    if cap.handler != HANDLER_ID:
        raise IssueLookupError(
            "unknown_handler", f"no in-process handler '{cap.handler}'"
        )
    issue_number = arguments.get("issue_number")
    if (
        isinstance(issue_number, bool)
        or not isinstance(issue_number, int)
        or issue_number < 1
        or issue_number > MAX_ISSUE_NUMBER
    ):
        raise IssueLookupError("invalid_issue", "issue_number must be a positive integer")
    repository = arguments.get("repository")
    if not isinstance(repository, str):
        raise IssueLookupError(
            "repo_denied",
            "repository is not on the public issue allowlist",
            denied=True,
        )
    canonical = _canonical_repo(cap.repo_allowlist, repository)
    if canonical is None:
        raise IssueLookupError(
            "repo_denied",
            "repository is not on the public issue allowlist",
            denied=True,
        )
    return canonical, issue_number


def _canonical_repo(allowlist: list[str], requested: str) -> str | None:
    if not allowlist or "/" not in requested or ".." in requested:
        return None
    folded = requested.casefold()
    for entry in allowlist:
        if entry.casefold() == folded:
            return entry
    return None


def _read_bounded(response: Any) -> bytes:
    raw = response.read(MAX_BODY_BYTES + 1)
    if not isinstance(raw, bytes | bytearray):
        raise IssueLookupError("invalid_response", "GitHub issue response was incomplete")
    if len(raw) > MAX_BODY_BYTES:
        raise IssueLookupError("invalid_response", "GitHub issue response was too large")
    return bytes(raw)


def _http_error(exc: urllib.error.HTTPError, repository: str, issue_number: int) -> NoReturn:
    code = int(exc.code)
    if code in {301, 302, 303, 307, 308}:
        raise IssueLookupError(
            "transport", "GitHub issue request was redirected; refusing to follow"
        )
    headers = exc.headers
    raw = b""
    try:
        raw = _read_bounded(exc)
    except IssueLookupError:
        raw = b""
    if code == 404:
        raise IssueLookupError(
            "not_found", f"GitHub issue {repository}#{issue_number} was not found"
        )
    if code == 429 or _is_rate_limited(code, headers, raw):
        raise IssueLookupError("rate_limited", "GitHub API rate limit exceeded")
    raise IssueLookupError("transport", f"GitHub issue request failed: HTTP {code}")


def _is_rate_limited(code: int, headers: Message | None, raw: bytes) -> bool:
    if code != 403:
        return False
    remaining = _header(headers, "X-RateLimit-Remaining")
    if remaining == "0":
        return True
    if _header(headers, "Retry-After") is not None:
        return True
    text = raw.decode("utf-8", errors="replace").casefold()
    return "rate limit" in text


def _header(headers: Message | None, name: str) -> str | None:
    if headers is None:
        return None
    value = headers.get(name)
    if value is None:
        return None
    return str(value)


def _project(raw: bytes, repository: str, issue_number: int) -> dict[str, Any]:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise IssueLookupError("invalid_response", "GitHub issue response was incomplete") from None
    if not isinstance(body, dict):
        raise IssueLookupError("invalid_response", "GitHub issue response was incomplete")
    title = body.get("title")
    state = body.get("state")
    updated_at = body.get("updated_at")
    html_url = body.get("html_url")
    if not isinstance(title, str) or not title.strip() or len(title) > 512:
        raise IssueLookupError("invalid_response", "GitHub issue response was incomplete")
    if not isinstance(state, str):
        raise IssueLookupError("invalid_response", "GitHub issue response was incomplete")
    normalized = {"open": "OPEN", "closed": "CLOSED"}.get(state.casefold())
    if normalized is None:
        raise IssueLookupError("invalid_response", "GitHub issue response was incomplete")
    if not isinstance(updated_at, str) or not _UPDATED_AT_RE.fullmatch(updated_at):
        raise IssueLookupError("invalid_response", "GitHub issue response was incomplete")
    if not isinstance(html_url, str) or not _url_matches(html_url, repository, issue_number):
        raise IssueLookupError("invalid_response", "GitHub issue response was incomplete")
    safe_title = " ".join(title.split())
    return {
        "repository": repository,
        "issue_number": issue_number,
        "title": safe_title,
        "state": normalized,
        "updated_at": updated_at,
        "html_url": html_url,
    }


def _url_matches(html_url: str, repository: str, issue_number: int) -> bool:
    marker = "https://github.com/"
    if not html_url.startswith(marker):
        return False
    rest = html_url[len(marker):]
    expected = f"{repository}/"
    # Allowlist entries are ASCII, so casefold does not change the prefix length.
    if not rest.casefold().startswith(expected.casefold()):
        return False
    tail = rest[len(expected):]
    return tail in {f"issues/{issue_number}", f"pull/{issue_number}"}
