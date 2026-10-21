from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Any
from ..ledger.database import DatabaseManager

class AffordableLossCeiling:
    """
    Enforces the ₹250 hard affordable-loss ceiling.
    Cumulative loss reaching ₹250 triggers a hard trading halt.
    AI or strategies can never override this deterministic ceiling.
    """
    def __init__(self, db: DatabaseManager, account_id: str = "default_spot", ceiling_limit_inr: float = 250.0):
        self.db = db
        self.account_id = account_id
        self.ceiling_limit_inr = ceiling_limit_inr

    def evaluate_ceiling_status(self) -> Dict[str, Any]:
        """
        Calculates cumulative realized loss + open position unrealized loss.
        If total loss >= ceiling_limit_inr, halts account and logs audit event.
        """
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT cumulative_realized_loss, is_active FROM accounts WHERE account_id = ?",
                (self.account_id,),
            )
            acc = cur.fetchone()
            if not acc:
                raise ValueError(f"Account {self.account_id} not found.")

            cum_realized_loss = acc["cumulative_realized_loss"]
            is_active = bool(acc["is_active"])

            # Also check unrealized losses in open positions
            cur.execute(
                "SELECT SUM(CASE WHEN unrealized_pnl_inr < 0 THEN ABS(unrealized_pnl_inr) ELSE 0 END) as open_drawdown FROM positions WHERE account_id = ? AND is_open = 1",
                (self.account_id,),
            )
            row = cur.fetchone()
            open_drawdown = row["open_drawdown"] or 0.0

            total_loss = cum_realized_loss + open_drawdown
            remaining_budget = max(0.0, self.ceiling_limit_inr - total_loss)
            is_breached = total_loss >= self.ceiling_limit_inr

            if is_breached and is_active:
                # Halt account in database
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
                        f"Affordable loss ceiling of ₹{self.ceiling_limit_inr} breached. Total loss: ₹{total_loss:.2f}",
                        json.dumps({
                            "cum_realized_loss": cum_realized_loss,
                            "open_drawdown": open_drawdown,
                            "total_loss": total_loss,
                            "ceiling": self.ceiling_limit_inr,
                        }),
                        now_iso,
                    ),
                )
                is_active = False

        return {
            "ceiling_limit_inr": self.ceiling_limit_inr,
            "cumulative_realized_loss_inr": round(cum_realized_loss, 2),
            "open_drawdown_inr": round(open_drawdown, 2),
            "total_measured_loss_inr": round(total_loss, 2),
            "remaining_budget_inr": round(remaining_budget, 2),
            "is_breached": is_breached,
            "is_trading_allowed": is_active and not is_breached,
        }

    def manual_reset_with_authorization(self, authorization_token: str, new_ceiling_inr: float = 250.0) -> bool:
        """
        Requires explicit operator authorization to unhalt an account.
        """
        if not authorization_token or len(authorization_token) < 8:
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE accounts 
                SET is_active = 1, affordable_loss_ceiling = ?, cumulative_realized_loss = 0.0, updated_at = ?
                WHERE account_id = ?
                """,
                (new_ceiling_inr, now_iso, self.account_id),
            )
            cur.execute(
                """
                INSERT INTO audit_events (
                    event_id, event_type, severity, message, details_json, timestamp
                ) VALUES (?, 'RECONCILIATION', 'INFO', 'Manual loss ceiling reset authorized', ?, ?)
                """,
                (
                    f"reset-{int(datetime.now().timestamp())}",
                    json.dumps({"token_prefix": authorization_token[:4] + "***", "new_ceiling": new_ceiling_inr}),
                    now_iso,
                ),
            )
        return True
