"""Secret redaction — scrub known + well-known-shaped secrets from text.

The credential broker registers every secret value it resolves; the redactor
then removes those literals (and a conservative set of token shapes) from any
text before it is logged or returned to the model.
"""

from __future__ import annotations

import re

REDACTION_PLACEHOLDER = "«redacted»"

# Conservative, high-confidence token shapes. The bearer pattern keeps the
# "Bearer " prefix and redacts only the credential.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}"), r"\1" + REDACTION_PLACEHOLDER),
    (re.compile(r"\bops_[A-Za-z0-9]{16,}\b"), REDACTION_PLACEHOLDER),  # 1Password SA
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), REDACTION_PLACEHOLDER),  # OpenAI-style
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"), REDACTION_PLACEHOLDER),  # GitHub
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), REDACTION_PLACEHOLDER),  # Slack
]


class SecretRedactor:
    def __init__(self) -> None:
        self._literals: set[str] = set()

    def register(self, secret: str | None) -> None:
        """Register a literal secret value to redact. Short/empty values ignored."""
        if secret and len(secret) >= 6:
            self._literals.add(secret)

    def redact(self, text: str) -> str:
        # Longest literals first, so a substring secret can't partially mask.
        for literal in sorted(self._literals, key=len, reverse=True):
            if literal in text:
                text = text.replace(literal, REDACTION_PLACEHOLDER)
        for pattern, replacement in _PATTERNS:
            text = pattern.sub(replacement, text)
        return text
