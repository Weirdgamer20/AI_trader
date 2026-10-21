from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from .database import DatabaseManager
from .fee_tax_engine import FeeAndTaxEngine

class PortfolioLedger:
    """
    Authoritative Portfolio Accounting and Ledger.
    Maintains cash balance, locked funds, position cost basis,
    realized/unrealized P&L, tax records, and compounding tracking.
    """
    def __init__(self, db: DatabaseManager, fee_engine: FeeAndTaxEngine | None = None, account_id: str = "default_spot"):
        self.db = db
        self.fee_engine = fee_engine or FeeAndTaxEngine()
        self.account_id = account_id
        self._ensure_account()

    def _ensure_account(self) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT account_id FROM accounts WHERE account_id = ?", (self.account_id,))
            row = cur.fetchone()
            if not row:
                cur.execute(
                    """
                    INSERT INTO accounts (
                        account_id, account_type, currency, starting_balance,
                        available_balance, locked_balance, target_balance,
                        affordable_loss_ceiling, cumulative_realized_loss,
                        is_active, created_at, updated_at
                    ) VALUES (?, 'PAPER', 'INR', 500.0, 500.0, 0.0, 100000.0, 250.0, 0.0, 1, ?, ?)
                    """,
                    (self.account_id, now_iso, now_iso),
                )
                # Initial deposit ledger entry
                cur.execute(
                    """
                    INSERT INTO ledger_entries (
                        entry_id, account_id, entry_type, amount_inr,
                        balance_after_inr, reference_id, description, timestamp
                    ) VALUES (?, ?, 'DEPOSIT', 500.0, 500.0, 'INITIAL_SEED', 'Starting capital deposit', ?)
                    """,
                    (str(uuid.uuid4()), self.account_id, now_iso),
                )

    def get_account_summary(self) -> Dict[str, Any]:
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM accounts WHERE account_id = ?", (self.account_id,))
            acc = dict(cur.fetchone())

            cur.execute(
                "SELECT * FROM positions WHERE account_id = ? AND is_open = 1",
                (self.account_id,),
            )
            positions = [dict(r) for r in cur.fetchall()]

            cur.execute(
                """
                SELECT 
                    SUM(taxable_gain_vda_inr) as total_taxable_gain,
                    SUM(tax_liability_30pct_inr) as total_tax_liability,
                    SUM(tds_withheld_1pct_inr) as total_tds_withheld
                FROM tax_records WHERE account_id = ?
                """,
                (self.account_id,),
            )
            tax_row = cur.fetchone()

        total_position_cost = sum(p["cost_basis_inr"] for p in positions)
        total_unrealized_pnl = sum(p["unrealized_pnl_inr"] for p in positions)
        total_equity = acc["available_balance"] + acc["locked_balance"] + total_position_cost + total_unrealized_pnl

        return {
            "account_id": acc["account_id"],
            "starting_balance_inr": acc["starting_balance"],
            "available_balance_inr": acc["available_balance"],
            "locked_balance_inr": acc["locked_balance"],
            "total_equity_inr": round(total_equity, 2),
            "target_balance_inr": acc["target_balance"],
            "progress_to_target_pct": round((total_equity / acc["target_balance"]) * 100.0, 4),
            "affordable_loss_ceiling_inr": acc["affordable_loss_ceiling"],
            "cumulative_realized_loss_inr": acc["cumulative_realized_loss"],
            "remaining_loss_budget_inr": max(0.0, acc["affordable_loss_ceiling"] - acc["cumulative_realized_loss"]),
            "open_positions_count": len(positions),
            "total_taxable_vda_gain_inr": round(tax_row["total_taxable_gain"] or 0.0, 2),
            "estimated_tax_liability_inr": round(tax_row["total_tax_liability"] or 0.0, 2),
            "tds_withheld_inr": round(tax_row["total_tds_withheld"] or 0.0, 2),
            "is_active": bool(acc["is_active"]),
        }

    def lock_funds(self, amount: float) -> bool:
        """Lock funds when an order is placed."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT available_balance, locked_balance FROM accounts WHERE account_id = ?", (self.account_id,))
            row = cur.fetchone()
            if not row or row["available_balance"] < amount:
                return False
            cur.execute(
                """
                UPDATE accounts 
                SET available_balance = available_balance - ?,
                    locked_balance = locked_balance + ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (amount, amount, now_iso, self.account_id),
            )
            return True

    def unlock_funds(self, amount: float) -> None:
        """Unlock funds if an order is cancelled or rejected."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE accounts 
                SET available_balance = available_balance + ?,
                    locked_balance = MAX(0.0, locked_balance - ?),
                    updated_at = ?
                WHERE account_id = ?
                """,
                (amount, amount, now_iso, self.account_id),
            )

    def create_order(
        self,
        order_id: str,
        client_order_id: str,
        market_pair: str,
        side: str,
        order_type: str = "LIMIT",
        price: float | None = None,
        quantity: float = 0.0,
        planned_risk_inr: float = 0.0,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO orders (
                    order_id, client_order_id, account_id, market_pair, side,
                    order_type, status, price, quantity, filled_quantity,
                    planned_risk_inr, stop_loss_price, take_profit_price,
                    idempotency_key, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'PROPOSED', ?, ?, 0.0, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id, client_order_id, self.account_id, market_pair, side,
                    order_type, price, quantity, planned_risk_inr, stop_loss_price,
                    take_profit_price, idempotency_key or str(uuid.uuid4()), now_iso, now_iso,
                ),
            )
        return order_id

    def update_order_status(self, order_id: str, status: str, filled_qty: float | None = None, reason: str | None = None) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            cur = conn.cursor()
            if filled_qty is not None:
                cur.execute(
                    "UPDATE orders SET status = ?, filled_quantity = ?, rejection_reason = ?, updated_at = ? WHERE order_id = ?",
                    (status, filled_qty, reason, now_iso, order_id),
                )
            else:
                cur.execute(
                    "UPDATE orders SET status = ?, rejection_reason = ?, updated_at = ? WHERE order_id = ?",
                    (status, reason, now_iso, order_id),
                )

    def record_buy_fill(
        self,
        order_id: str,
        market_pair: str,
        price: float,
        quantity: float,
        is_maker: bool = False,
    ) -> Dict[str, Any]:
        costs = self.fee_engine.calculate_buy_costs(price, quantity, is_maker)
        now_iso = datetime.now(timezone.utc).isoformat()
        fill_id = str(uuid.uuid4())

        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO fills (
                    fill_id, order_id, market_pair, side, price, quantity,
                    fee_inr, gst_on_fee_inr, tds_deducted_inr, net_proceeds_inr,
                    timestamp, is_paper
                ) VALUES (?, ?, ?, 'BUY', ?, ?, ?, ?, 0.0, ?, ?, 1)
                """,
                (
                    fill_id, order_id, market_pair, price, quantity,
                    costs["brokerage_fee_inr"], costs["gst_on_fee_inr"],
                    costs["total_outlay_inr"], now_iso,
                ),
            )

            # Update account cash
            cur.execute(
                """
                UPDATE accounts 
                SET locked_balance = MAX(0.0, locked_balance - ?),
                    updated_at = ?
                WHERE account_id = ?
                """,
                (costs["total_outlay_inr"], now_iso, self.account_id),
            )

            cur.execute("SELECT available_balance, locked_balance FROM accounts WHERE account_id = ?", (self.account_id,))
            acc = cur.fetchone()
            new_balance = acc["available_balance"] + acc["locked_balance"]

            # Ledger entry
            cur.execute(
                """
                INSERT INTO ledger_entries (
                    entry_id, account_id, entry_type, amount_inr,
                    balance_after_inr, reference_id, description, timestamp
                ) VALUES (?, ?, 'BUY_COST', ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()), self.account_id, -costs["total_outlay_inr"],
                    new_balance, fill_id, f"Buy {quantity} {market_pair} @ {price}", now_iso,
                ),
            )

            # Update or create Position
            cur.execute(
                "SELECT * FROM positions WHERE account_id = ? AND market_pair = ? AND is_open = 1",
                (self.account_id, market_pair),
            )
            pos = cur.fetchone()
            if pos:
                new_qty = pos["quantity"] + quantity
                new_cost = pos["cost_basis_inr"] + costs["total_outlay_inr"]
                avg_price = new_cost / new_qty if new_qty > 0 else price
                cur.execute(
                    """
                    UPDATE positions 
                    SET quantity = ?, cost_basis_inr = ?, average_entry_price = ?,
                        current_price = ?, updated_at = ?
                    WHERE position_id = ?
                    """,
                    (new_qty, new_cost, avg_price, price, now_iso, pos["position_id"]),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO positions (
                        position_id, account_id, market_pair, quantity,
                        average_entry_price, cost_basis_inr, current_price,
                        unrealized_pnl_inr, realized_pnl_inr, is_open, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, 1, ?)
                    """,
                    (
                        str(uuid.uuid4()), self.account_id, market_pair, quantity,
                        price, costs["total_outlay_inr"], price, now_iso,
                    ),
                )

        return {"fill_id": fill_id, "costs": costs}

    def record_sell_fill(
        self,
        order_id: str,
        market_pair: str,
        price: float,
        quantity: float,
        is_maker: bool = False,
    ) -> Dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        fill_id = str(uuid.uuid4())

        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM positions WHERE account_id = ? AND market_pair = ? AND is_open = 1",
                (self.account_id, market_pair),
            )
            pos = cur.fetchone()
            if not pos or pos["quantity"] <= 0:
                raise ValueError(f"No open position found for {market_pair} to sell.")

            cost_basis_per_unit = pos["cost_basis_inr"] / pos["quantity"]
            sell_results = self.fee_engine.calculate_sell_proceeds(
                price, quantity, cost_basis_per_unit, is_maker
            )

            # Record Fill
            cur.execute(
                """
                INSERT INTO fills (
                    fill_id, order_id, market_pair, side, price, quantity,
                    fee_inr, gst_on_fee_inr, tds_deducted_inr, net_proceeds_inr,
                    timestamp, is_paper
                ) VALUES (?, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    fill_id, order_id, market_pair, price, quantity,
                    sell_results["brokerage_fee_inr"], sell_results["gst_on_fee_inr"],
                    sell_results["tds_withheld_1pct_inr"], sell_results["net_cash_received_inr"],
                    now_iso,
                ),
            )

            # Credit Cash Account
            cur.execute(
                """
                UPDATE accounts 
                SET available_balance = available_balance + ?,
                    cumulative_realized_loss = cumulative_realized_loss + ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (
                    sell_results["net_cash_received_inr"],
                    abs(sell_results["economic_pnl_inr"]) if sell_results["economic_pnl_inr"] < 0 else 0.0,
                    now_iso,
                    self.account_id,
                ),
            )

            cur.execute("SELECT available_balance, locked_balance FROM accounts WHERE account_id = ?", (self.account_id,))
            acc = cur.fetchone()
            new_balance = acc["available_balance"] + acc["locked_balance"]

            # Ledger Entry
            cur.execute(
                """
                INSERT INTO ledger_entries (
                    entry_id, account_id, entry_type, amount_inr,
                    balance_after_inr, reference_id, description, timestamp
                ) VALUES (?, ?, 'SELL_PROCEEDS', ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()), self.account_id, sell_results["net_cash_received_inr"],
                    new_balance, fill_id, f"Sell {quantity} {market_pair} @ {price}", now_iso,
                ),
            )

            # Record Indian Tax Record (Section 194S & 115BBH)
            cur.execute(
                """
                INSERT INTO tax_records (
                    record_id, account_id, financial_year, fill_id, market_pair,
                    sale_proceeds_inr, cost_basis_inr, economic_gain_inr,
                    taxable_gain_vda_inr, tax_liability_30pct_inr, tds_withheld_1pct_inr,
                    timestamp
                ) VALUES (?, ?, 'FY2026-27', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()), self.account_id, fill_id, market_pair,
                    sell_results["gross_sale_inr"], sell_results["cost_basis_inr"],
                    sell_results["economic_pnl_inr"], sell_results["taxable_vda_gain_inr"],
                    sell_results["vda_tax_liability_30pct_inr"], sell_results["tds_withheld_1pct_inr"],
                    now_iso,
                ),
            )

            # Update Position
            remaining_qty = max(0.0, pos["quantity"] - quantity)
            remaining_cost = cost_basis_per_unit * remaining_qty
            is_open = 1 if remaining_qty > 1e-8 else 0
            cur.execute(
                """
                UPDATE positions 
                SET quantity = ?, cost_basis_inr = ?, realized_pnl_inr = realized_pnl_inr + ?,
                    is_open = ?, updated_at = ?
                WHERE position_id = ?
                """,
                (
                    remaining_qty, remaining_cost, sell_results["economic_pnl_inr"],
                    is_open, now_iso, pos["position_id"],
                ),
            )

        return {"fill_id": fill_id, "sell_results": sell_results}
