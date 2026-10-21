from __future__ import annotations

import hmac
import os
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
        """Failsafe: If market data is older than 3 intervals or in the future, trigger emergency safe mode."""
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        max_allowed_delay_ms = 3 * interval_minutes * 60 * 1000

        # Disallow future timestamps (> 2 minutes ahead of system clock)
        if last_candle_timestamp_ms > now_ms + (2 * 60 * 1000):
            self.trigger_emergency_stop(
                reason=f"Corrupt market data: Candle timestamp is in the future ({last_candle_timestamp_ms} > {now_ms})",
                severity="CRITICAL",
                details={"last_ts": last_candle_timestamp_ms, "now_ms": now_ms},
            )
            return False

        if now_ms - last_candle_timestamp_ms > max_allowed_delay_ms:
            delay_min = (now_ms - last_candle_timestamp_ms) / (60 * 1000)
            self.trigger_emergency_stop(
                reason=f"Stale market data: Last candle is {delay_min:.1f} minutes old (threshold {3 * interval_minutes}m)",
                severity="CRITICAL",
                details={"last_ts": last_candle_timestamp_ms, "delay_min": delay_min},
            )
            return False
        return True

    def check_ledger_reconciliation(
        self,
        exchange_reported_cash: float,
        exchange_reported_holdings: Dict[str, float] | None = None,
    ) -> bool:
        """Failsafe: Reconciles internal cash and position quantities against exchange records."""
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT available_balance, locked_balance FROM accounts WHERE account_id = ?", (self.account_id,))
            acc = cur.fetchone()
            if not acc:
                return False
            ledger_cash_total = float(acc["available_balance"] + acc["locked_balance"])

            # Check positions
            cur.execute("SELECT market_pair, quantity FROM positions WHERE account_id = ? AND is_open = 1", (self.account_id,))
            db_positions = {r["market_pair"]: float(r["quantity"]) for r in cur.fetchall()}

        # 1. Cash discrepancy check
        discrepancy = abs(ledger_cash_total - exchange_reported_cash)
        discrepancy_pct = discrepancy / ledger_cash_total if ledger_cash_total > 0 else 1.0

        if discrepancy_pct > 0.01 and discrepancy > 5.0:
            self.trigger_emergency_stop(
                reason=f"Ledger cash mismatch: Ledger=₹{ledger_cash_total:.2f}, Exchange=₹{exchange_reported_cash:.2f}",
                severity="EMERGENCY",
                details={"ledger_cash": ledger_cash_total, "exchange_cash": exchange_reported_cash},
            )
            return False

        # 2. Portfolio holdings discrepancy check
        if exchange_reported_holdings is not None:
            for pair, db_qty in db_positions.items():
                ex_qty = exchange_reported_holdings.get(pair, 0.0)
                if abs(db_qty - ex_qty) > 1e-6:
                    self.trigger_emergency_stop(
                        reason=f"Portfolio asset mismatch for {pair}: Ledger={db_qty:.8f}, Exchange={ex_qty:.8f}",
                        severity="EMERGENCY",
                        details={"pair": pair, "db_qty": db_qty, "ex_qty": ex_qty},
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

    def clear_emergency_stop(self, authorization_token: str, operator_id: str = "operator") -> bool:
        expected_token = os.environ.get("EMERGENCY_CLEAR_TOKEN") or os.environ.get("LOSS_CEILING_RESET_TOKEN")
        if not expected_token or not authorization_token:
            return False

        if not hmac.compare_digest(authorization_token.strip(), expected_token.strip()):
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
                ) VALUES (?, 'RECONCILIATION', 'INFO', ?, '{}', ?)
                """,
                (f"clear-{int(datetime.now().timestamp())}", f"Emergency safe mode cleared by {operator_id}", now_iso),
            )
        return True
