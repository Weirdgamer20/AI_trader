from __future__ import annotations

import re
import requests
from typing import Dict, Any, Optional

# Secret patterns to automatically redact from outgoing notifications
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password|bearer|auth)\s*[:=]\s*([a-zA-Z0-9_\-\.]{8,})"),
    re.compile(r"bot[0-9]{8,}:[a-zA-Z0-9_-]{20,}"), # Telegram bot tokens
]

class TelegramNotifier:
    """
    Telegram Notification Dispatcher with strict automated secret redaction.
    """
    def __init__(self, bot_token: str | None = None, chat_id: str | None = None):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def sanitize_message(self, text: str) -> str:
        """Strips credentials, API keys, and sensitive tokens from message text."""
        sanitized = text
        # Redact labeled secrets e.g. api_key: xyz
        sanitized = re.sub(
            r"(?i)(api[_-]?key|secret|token|password|bearer|auth)\s*[:=]\s*([a-zA-Z0-9_\-\.]{8,})",
            r"\1: [REDACTED_SECRET]",
            sanitized,
        )
        # Redact raw bot tokens
        sanitized = re.sub(r"bot[0-9]{8,}:[a-zA-Z0-9_-]{20,}", "[REDACTED_BOT_TOKEN]", sanitized)
        return sanitized

    def send_notification(self, title: str, body: str, severity: str = "INFO") -> Dict[str, Any]:
        """Dispatches notification to Telegram (or simulates if no credentials configured)."""
        clean_title = self.sanitize_message(title)
        clean_body = self.sanitize_message(body)

        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨", "EMERGENCY": "🛑"}.get(severity, "ℹ️")
        formatted = f"{emoji} *{clean_title}*\n\n{clean_body}"

        if not self.bot_token or not self.chat_id:
            # Safe local log when bot credentials are not configured
            return {
                "sent": False,
                "reason": "Telegram credentials not configured; logged locally",
                "message_preview": formatted,
            }

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            resp = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": formatted, "parse_mode": "Markdown"},
                timeout=10,
            )
            return {"sent": resp.ok, "status_code": resp.status_code}
        except Exception as exc:
            return {"sent": False, "error": str(exc)}
