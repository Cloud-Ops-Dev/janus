"""Credential-bearing URL query parameters are redacted (infra-jymo).

A browser downstream returned a download URL whose ``access_token`` query
parameter was a bearer for the file. The redactor only knew fixed token
shapes, so the value reached the model and, via ``structuredContent``, the
calling agent's transcript. Every value here is synthetic.
"""

from __future__ import annotations

from janus.downstream import DownstreamResult
from janus.registry import TrustLevel
from janus.security import OutputSanitizer, SecretRedactor
from janus.security.secret_redactor import REDACTION_PLACEHOLDER

TOKEN = "synthjymo" + "A" * 24


def _r(text: str) -> str:
    return SecretRedactor().redact(text)


def test_access_token_param_redacted_other_params_kept() -> None:
    out = _r(f"https://files.example.test/dl/x.zip?type=zip&access_token={TOKEN}&download=1")
    assert TOKEN not in out
    assert f"access_token={REDACTION_PLACEHOLDER}&download=1" in out
    assert "?type=zip&" in out


def test_first_param_and_other_names() -> None:
    for name in ("token", "sig", "X-Amz-Signature", "api_key", "client_secret"):
        out = _r(f"http://host.example.test:8080/?{name}={TOKEN}")
        assert TOKEN not in out, name


def test_html_escaped_separator() -> None:
    out = _r(f"<a href='https://e.test/o?x=1&amp;access_token={TOKEN}'>f</a>")
    assert TOKEN not in out


def test_short_or_non_credential_values_untouched() -> None:
    text = "https://e.test/search?q=token&page=2&token=1&sort=signature"
    assert _r(text) == text


def test_structured_content_redacted_through_sanitizer() -> None:
    url = f"https://e.test/o?access_token={TOKEN}"
    res = DownstreamResult(is_error=False, text=url, structured={"text": url})
    out = OutputSanitizer(SecretRedactor()).sanitize(res, trust_level=TrustLevel.THIRD_PARTY)
    assert TOKEN not in out.text
    assert TOKEN not in str(out.structured)
