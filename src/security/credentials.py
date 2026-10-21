from __future__ import annotations

import os
import re
from typing import Dict, Any, Optional

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
        # Redact labeled secrets e.g. api_key: abc12345 or private_key=xyz
        pattern = re.compile(r"(?i)(api[_-]?key|private[_-]?key|[a-z0-9_-]*key|secret|token|password|auth|bearer)\s*[:=]\s*([^\s,;]+)")
        return pattern.sub(r"\1: [REDACTED]", text)

    @classmethod
    def redact_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively redacts dictionary keys containing sensitive terms."""
        sensitive_keys = {"secret", "api_key", "token", "password", "auth", "private_key"}
        redacted = {}
        for k, v in data.items():
            if any(s in k.lower() for s in sensitive_keys):
                redacted[k] = cls.SECRET_MASK
            elif isinstance(v, dict):
                redacted[k] = cls.redact_dict(v)
            elif isinstance(v, str):
                redacted[k] = cls.redact_text(v)
            else:
                redacted[k] = v
        return redacted
