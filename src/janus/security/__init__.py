"""Security: credential broker (op:// resolution) + secret redaction.

The credential broker is the only holder of downstream secrets (design §5.4): it
resolves ``op://`` references at call time, injects them into connections, and
never returns them to the model. The redactor scrubs any secret value that might
otherwise leak into outputs or logs.
"""

from janus.security.credential_broker import CredentialBroker, CredentialError
from janus.security.output_sanitizer import (
    NullSanitizer,
    OutputSanitizer,
    ResultSanitizer,
)
from janus.security.secret_redactor import REDACTION_PLACEHOLDER, SecretRedactor

__all__ = [
    "REDACTION_PLACEHOLDER",
    "CredentialBroker",
    "CredentialError",
    "NullSanitizer",
    "OutputSanitizer",
    "ResultSanitizer",
    "SecretRedactor",
]
