from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from ..ledger.database import DatabaseManager

class EmergencyStopSystem:
    """
    Emergency Stop & Safe Mode Circuit Breaker.
    Monitors data integrity, balance reconciliations, and manual triggers.
    Forces system into safe mode and prevents new orders when an anomaly occurs.
    """
    def __init__(self, db: DatabaseManager, account_id: str = "default_spot"):
        self.db = db
        self.account_id = account_id
        self._manual_kill_active = False

    def trigger_emergency_stop(self, reason: str, severity: str = "CRITICAL", details: Dict[str, Any] | None = None) -> None:
        """Triggers emergency stop, halts account, and logs audit event."""
        now_iso = datetime.now(timezone.utc).isoformat()
        self._manual_kill_active = True

        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE accounts SET is_active = 0, updated_at = ? WHERE account_id = ?",
                (now_iso, self.account_id),
            )
            cur.execute(
                """
                INSERT INTO audit_events (
                    event_id, event_type, severity, message, details_json, timestamp
                ) VALUES (?, 'EMERGENCY_STOP', ?, ?, ?, ?)
                """,
                (
                    f"emerg-{int(datetime.now().timestamp())}",
                    severity,
                    reason,
                    json.dumps(details or {}),
                    now_iso,
                ),
            )

    def check_market_data_health(self, last_candle_timestamp_ms: int, interval_minutes: int = 15) -> bool:
        """Failsafe: If market data is older than 3 intervals, trigger emergency safe mode."""
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        max_allowed_delay_ms = 3 * interval_minutes * 60 * 1000
        if now_ms - last_candle_timestamp_ms > max_allowed_delay_ms:
            delay_min = (now_ms - last_candle_timestamp_ms) / (60 * 1000)
            self.trigger_emergency_stop(
                reason=f"Stale market data: Last candle is {delay_min:.1f} minutes old (threshold {3 * interval_minutes}m)",
                severity="CRITICAL",
                details={"last_ts": last_candle_timestamp_ms, "delay_min": delay_min},
            )
            return False
        return True

    def check_ledger_reconciliation(self, exchange_reported_balance: float) -> bool:
        """Failsafe: If internal ledger equity deviates from exchange balance by >1%, halt."""
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT available_balance, locked_balance FROM accounts WHERE account_id = ?", (self.account_id,))
            acc = cur.fetchone()
            if not acc:
                return False
            ledger_total = acc["available_balance"] + acc["locked_balance"]

        discrepancy = abs(ledger_total - exchange_reported_balance)
        discrepancy_pct = discrepancy / ledger_total if ledger_total > 0 else 1.0

        if discrepancy_pct > 0.01 and discrepancy > 5.0: # >1% and >₹5
            self.trigger_emergency_stop(
                reason=f"Ledger reconciliation mismatch: Ledger=₹{ledger_total:.2f}, Exchange=₹{exchange_reported_balance:.2f}",
                severity="EMERGENCY",
                details={"ledger_total": ledger_total, "exchange_balance": exchange_reported_balance, "discrepancy_pct": discrepancy_pct},
            )
            return False
        return True

    def is_safe_mode_active(self) -> bool:
        if self._manual_kill_active:
            return True
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT is_active FROM accounts WHERE account_id = ?", (self.account_id,))
            row = cur.fetchone()
            return not bool(row["is_active"]) if row else True

    def clear_emergency_stop(self, authorization_token: str) -> bool:
        if not authorization_token or len(authorization_token) < 8:
            return False
        now_iso = datetime.now(timezone.utc).isoformat()
        self._manual_kill_active = False
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE accounts SET is_active = 1, updated_at = ? WHERE account_id = ?", (now_iso, self.account_id))
            cur.execute(
                """
                INSERT INTO audit_events (
                    event_id, event_type, severity, message, details_json, timestamp
                ) VALUES (?, 'RECONCILIATION', 'INFO', 'Emergency safe mode cleared by operator', '{}', ?)
                """,
                (f"clear-{int(datetime.now().timestamp())}", now_iso),
            )
        return True
