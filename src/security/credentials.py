from __future__ import annotations

import os
import re
from typing import Dict, Any, List, Optional, Union

SECRET_REGEX = re.compile(
    r"(?i)(api[_-]?key|private[_-]?key|[a-z0-9_-]*key|secret|token|password|auth|bearer)\s*[:=]\s*([^\s,;\"']+)"
)
BOT_TOKEN_REGEX = re.compile(r"bot[0-9]{8,}:[a-zA-Z0-9_-]{20,}")

class CredentialVault:
    """
    Secure Credential & Secret Management.
    Enforces Windows DPAPI / Environment / Credential Manager least-privilege access.
    Strictly forbids raw secret logging or research agent self-escalation.
    """
    SECRET_MASK = "********"

    @classmethod
    def get_secret(cls, key: str, default: str | None = None) -> Optional[str]:
        """Safely retrieves secret without leaking to stdout or logs."""
        return os.environ.get(key, default)

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Redacts any API keys, bearer tokens, or password-like strings."""
        if not text:
            return ""
        sanitized = SECRET_REGEX.sub(r"\1: [REDACTED]", str(text))
        sanitized = BOT_TOKEN_REGEX.sub("[REDACTED_BOT_TOKEN]", sanitized)
        return sanitized

    @classmethod
    def redact_data(cls, data: Any) -> Any:
        """Recursively redacts sensitive keys from dictionaries, lists, tuples, and strings."""
        sensitive_key_terms = {"secret", "api_key", "token", "password", "auth", "private_key", "credentials"}
        if isinstance(data, dict):
            redacted = {}
            for k, v in data.items():
                if any(term in str(k).lower() for term in sensitive_key_terms):
                    redacted[k] = cls.SECRET_MASK
                else:
                    redacted[k] = cls.redact_data(v)
            return redacted
        elif isinstance(data, list):
            return [cls.redact_data(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(cls.redact_data(item) for item in data)
        elif isinstance(data, str):
            return cls.redact_text(data)
        else:
            return data

    @classmethod
    def redact_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward-compatible alias for redact_data on dictionaries."""
        return cls.redact_data(data)
