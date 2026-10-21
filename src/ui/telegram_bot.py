from __future__ import annotations

import re
import requests
from typing import Dict, Any, Optional

class TelegramNotifier:
    """
    Telegram Notification Dispatcher with strict automated secret redaction
    and safe Markdown escaping. Never leaks bot tokens or raw request URLs in error paths.
    """
    def __init__(self, bot_token: str | None = None, chat_id: str | None = None):
        self.bot_token = bot_token
        self.chat_id = chat_id

    @staticmethod
    def escape_markdown(text: str) -> str:
        """Escapes markdown v1 special characters outside formatting."""
        # Escape characters that break Telegram Markdown v1
        for char in ["_", "*", "`", "["]:
            text = text.replace(char, f"\\{char}")
        return text

    def sanitize_message(self, text: str) -> str:
        """Strips credentials, API keys, and sensitive tokens from message text."""
        sanitized = str(text)
        # Redact labeled secrets e.g. api_key: xyz
        sanitized = re.sub(
            r"(?i)(api[_-]?key|private[_-]?key|[a-z0-9_-]*key|secret|token|password|bearer|auth)\s*[:=]\s*([^\s,;\"']+)",
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
        safe_title = self.escape_markdown(clean_title)
        safe_body = self.escape_markdown(clean_body)
        formatted = f"{emoji} *{safe_title}*\n\n{safe_body}"

        if not self.bot_token or not self.chat_id:
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
            # Crucial: Sanitize exception so bot token in URL is NEVER returned
            safe_error_msg = self.sanitize_message(str(exc))
            return {"sent": False, "error": f"Telegram delivery failed: {safe_error_msg}"}
