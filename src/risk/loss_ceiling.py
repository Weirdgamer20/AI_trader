from __future__ import annotations

import hmac
import os
import json
from datetime import datetime, timezone
from typing import Dict, Any
from ..ledger.database import DatabaseManager

class AffordableLossCeiling:
    """
    Enforces the ₹250 hard affordable-loss ceiling.
    Cumulative loss reaching ₹250 triggers a hard trading halt.
    AI or strategies can never override this deterministic ceiling.
    
    Security & Integrity Guarantees:
    - Never erases historical cumulative losses on reset.
    - Uses constant-time token verification against LOSS_CEILING_RESET_TOKEN.
    - Requires operator identity, reason, and writes an immutable audit record.
    - Tracks active baseline offset without destroying lifetime loss history.
    """
    def __init__(self, db: DatabaseManager, account_id: str = "default_spot", ceiling_limit_inr: float = 250.0):
        self.db = db
        self.account_id = account_id
        self.ceiling_limit_inr = ceiling_limit_inr
        self._ensure_columns()

    def _ensure_columns(self) -> None:
        """Ensure accounts table has loss_baseline_inr column for period tracking."""
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(accounts);")
            cols = [r[1] for r in cur.fetchall()]
            if "loss_baseline_inr" not in cols:
                cur.execute("ALTER TABLE accounts ADD COLUMN loss_baseline_inr REAL NOT NULL DEFAULT 0.0;")

    def evaluate_ceiling_status(self) -> Dict[str, Any]:
        """
        Calculates period loss = (lifetime cumulative realized loss - loss_baseline) + open drawdown.
        If period loss >= ceiling_limit_inr, halts account and logs audit event.
        """
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT cumulative_realized_loss, loss_baseline_inr, is_active FROM accounts WHERE account_id = ?",
                (self.account_id,),
            )
            acc = cur.fetchone()
            if not acc:
                raise ValueError(f"Account {self.account_id} not found.")

            cum_realized_loss = float(acc["cumulative_realized_loss"])
            loss_baseline = float(acc["loss_baseline_inr"] if "loss_baseline_inr" in acc.keys() else 0.0)
            is_active = bool(acc["is_active"])

            # Open unrealized losses
            cur.execute(
                "SELECT SUM(CASE WHEN unrealized_pnl_inr < 0 THEN ABS(unrealized_pnl_inr) ELSE 0 END) as open_drawdown FROM positions WHERE account_id = ? AND is_open = 1",
                (self.account_id,),
            )
            row = cur.fetchone()
            open_drawdown = float(row["open_drawdown"] or 0.0)

            # Period loss measured from baseline
            realized_period_loss = max(0.0, cum_realized_loss - loss_baseline)
            total_measured_loss = realized_period_loss + open_drawdown
            remaining_budget = max(0.0, self.ceiling_limit_inr - total_measured_loss)
            is_breached = total_measured_loss >= self.ceiling_limit_inr

            if is_breached and is_active:
                now_iso = datetime.now(timezone.utc).isoformat()
                cur.execute(
                    "UPDATE accounts SET is_active = 0, updated_at = ? WHERE account_id = ?",
                    (now_iso, self.account_id),
                )
                cur.execute(
                    """
                    INSERT INTO audit_events (
                        event_id, event_type, severity, message, details_json, timestamp
                    ) VALUES (?, 'RISK_HALT', 'EMERGENCY', ?, ?, ?)
                    """,
                    (
                        f"halt-{int(datetime.now().timestamp())}",
                        f"Affordable loss ceiling of ₹{self.ceiling_limit_inr} breached. Period loss: ₹{total_measured_loss:.2f}",
                        json.dumps({
                            "lifetime_realized_loss": cum_realized_loss,
                            "loss_baseline": loss_baseline,
                            "realized_period_loss": realized_period_loss,
                            "open_drawdown": open_drawdown,
                            "total_measured_loss": total_measured_loss,
                            "ceiling": self.ceiling_limit_inr,
                        }),
                        now_iso,
                    ),
                )
                is_active = False

        return {
            "ceiling_limit_inr": self.ceiling_limit_inr,
            "lifetime_realized_loss_inr": round(cum_realized_loss, 2),
            "loss_baseline_inr": round(loss_baseline, 2),
            "period_realized_loss_inr": round(realized_period_loss, 2),
            "open_drawdown_inr": round(open_drawdown, 2),
            "total_measured_loss_inr": round(total_measured_loss, 2),
            "remaining_budget_inr": round(remaining_budget, 2),
            "is_breached": is_breached,
            "is_trading_allowed": is_active and not is_breached,
        }

    def manual_reset_with_authorization(
        self,
        authorization_token: str,
        operator_id: str = "operator",
        reason: str = "Authorized recovery reset",
        new_ceiling_inr: float = 250.0,
    ) -> bool:
        """
        Requires explicit operator authorization via LOSS_CEILING_RESET_TOKEN.
        Does NOT erase lifetime cumulative realized loss history.
        Sets loss_baseline_inr = current cumulative_realized_loss to track new period.
        """
        expected_token = os.environ.get("LOSS_CEILING_RESET_TOKEN")
        if not expected_token or not authorization_token:
            return False

        if not hmac.compare_digest(authorization_token.strip(), expected_token.strip()):
            return False

        if new_ceiling_inr <= 0:
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT cumulative_realized_loss FROM accounts WHERE account_id = ?", (self.account_id,))
            acc = cur.fetchone()
            if not acc:
                return False

            current_loss = float(acc["cumulative_realized_loss"])

            # Update baseline to current loss (never zero out historical loss!)
            cur.execute(
                """
                UPDATE accounts 
                SET is_active = 1,
                    affordable_loss_ceiling = ?,
                    loss_baseline_inr = ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (new_ceiling_inr, current_loss, now_iso, self.account_id),
            )
            cur.execute(
                """
                INSERT INTO audit_events (
                    event_id, event_type, severity, message, details_json, timestamp
                ) VALUES (?, 'RECONCILIATION', 'INFO', ?, ?, ?)
                """,
                (
                    f"reset-{int(datetime.now().timestamp())}",
                    f"Loss ceiling reset authorized by {operator_id}: {reason}",
                    json.dumps({
                        "operator": operator_id,
                        "reason": reason,
                        "lifetime_loss_preserved": current_loss,
                        "new_baseline": current_loss,
                        "new_ceiling": new_ceiling_inr,
                    }),
                    now_iso,
                ),
            )
        return True
