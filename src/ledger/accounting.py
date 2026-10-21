from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from .database import DatabaseManager
from .fee_tax_engine import FeeAndTaxEngine

class PortfolioLedger:
    """
    Authoritative Portfolio Accounting and Double-Entry General Ledger.
    Guarantees:
    - True balanced double-entry journal postings (sum of debits == sum of credits).
    - Strict per-order reservation tracking for cash and asset positions.
    - Prevents duplicate fills via unique trade IDs.
    - Explicit tracking of TDS Receivable asset separate from cash.
    - Atomic execution across accounts, positions, orders, fills, and journal entries.
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
                        affordable_loss_ceiling, loss_baseline_inr, cumulative_realized_loss,
                        tds_receivable_inr, is_active, created_at, updated_at
                    ) VALUES (?, 'PAPER', 'INR', 500.0, 500.0, 0.0, 100000.0, 250.0, 0.0, 0.0, 0.0, 1, ?, ?)
                    """,
                    (self.account_id, now_iso, now_iso),
                )
                tx_id = str(uuid.uuid4())
                # Balanced initial seed journal: Debit CASH ₹500, Credit INITIAL_CAPITAL ₹500
                self._post_journal(
                    cur=cur,
                    tx_id=tx_id,
                    postings=[
                        ("CASH", 500.0, 0.0, "Initial seed capital"),
                        ("INITIAL_CAPITAL", 0.0, 500.0, "Initial equity reserve"),
                    ],
                    ref_id="INITIAL_SEED",
                    timestamp=now_iso,
                )
                cur.execute(
                    """
                    INSERT INTO ledger_entries (
                        entry_id, account_id, entry_type, amount_inr,
                        balance_after_inr, reference_id, description, timestamp
                    ) VALUES (?, ?, 'DEPOSIT', 500.0, 500.0, 'INITIAL_SEED', 'Starting capital deposit', ?)
                    """,
                    (str(uuid.uuid4()), self.account_id, now_iso),
                )

    def _post_journal(
        self,
        cur: Any,
        tx_id: str,
        postings: List[tuple[str, float, float, str]],
        ref_id: str,
        timestamp: str,
    ) -> None:
        """
        Enforces double-entry equation: Total Debits == Total Credits.
        """
        total_debits = sum(round(p[1], 4) for p in postings)
        total_credits = sum(round(p[2], 4) for p in postings)

        if abs(total_debits - total_credits) > 0.001:
            raise ValueError(
                f"Double-entry journal out of balance! Debits: {total_debits:.4f}, Credits: {total_credits:.4f}"
            )

        for acct_name, debit, credit, desc in postings:
            cur.execute(
                """
                INSERT INTO journal_entries (
                    entry_id, transaction_id, account_id, account_name,
                    debit_inr, credit_inr, reference_id, description, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), tx_id, self.account_id, acct_name, debit, credit, ref_id, desc, timestamp),
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

        total_position_cost = sum(float(p["cost_basis_inr"]) for p in positions)
        total_unrealized_pnl = sum(float(p["unrealized_pnl_inr"]) for p in positions)
        avail_cash = float(acc["available_balance"])
        locked_cash = float(acc["locked_balance"])
        tds_asset = float(acc.get("tds_receivable_inr", 0.0) or 0.0)

        # Total economic equity: Cash + Locked Cash + Position Market Value + TDS Credit
        total_equity = avail_cash + locked_cash + total_position_cost + total_unrealized_pnl + tds_asset

        return {
            "account_id": acc["account_id"],
            "starting_balance_inr": float(acc["starting_balance"]),
            "available_balance_inr": round(avail_cash, 2),
            "locked_balance_inr": round(locked_cash, 2),
            "tds_receivable_inr": round(tds_asset, 2),
            "total_equity_inr": round(total_equity, 2),
            "target_balance_inr": float(acc["target_balance"]),
            "progress_to_target_pct": round((total_equity / float(acc["target_balance"])) * 100.0, 4),
            "affordable_loss_ceiling_inr": float(acc["affordable_loss_ceiling"]),
            "cumulative_realized_loss_inr": float(acc["cumulative_realized_loss"]),
            "open_positions_count": len(positions),
            "total_taxable_vda_gain_inr": round(float(tax_row["total_taxable_gain"] or 0.0), 2),
            "estimated_tax_liability_inr": round(float(tax_row["total_tax_liability"] or 0.0), 2),
            "tds_withheld_inr": round(float(tax_row["total_tds_withheld"] or 0.0), 2),
            "is_active": bool(acc["is_active"]),
        }

    def lock_funds(self, amount: float, order_id: str | None = None) -> bool:
        """Alias for reserve_funds_for_order."""
        return self.reserve_funds_for_order(order_id=order_id or f"ord-lock-{uuid.uuid4().hex[:8]}", estimated_total_cost=amount)

    def unlock_funds(self, amount: float, order_id: str | None = None) -> bool:
        """Unlocks previously locked funds."""
        if amount <= 0:
            return False
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT locked_balance FROM accounts WHERE account_id = ?", (self.account_id,))
            acc = cur.fetchone()
            if not acc or float(acc["locked_balance"]) < amount:
                return False
            cur.execute(
                """
                UPDATE accounts 
                SET available_balance = available_balance + ?,
                    locked_balance = locked_balance - ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (amount, amount, now_iso, self.account_id),
            )
            tx_id = str(uuid.uuid4())
            self._post_journal(
                cur=cur,
                tx_id=tx_id,
                postings=[
                    ("CASH", amount, 0.0, f"Funds unlocked"),
                    ("LOCKED_CASH", 0.0, amount, f"Funds released"),
                ],
                ref_id=order_id or "unlock",
                timestamp=now_iso,
            )
            return True

    def reserve_funds_for_order(self, order_id: str, estimated_total_cost: float) -> bool:
        """
        Per-order reservation:
        Reserves the full estimated outlay (notional + fees + GST) for a BUY order.
        """
        if estimated_total_cost <= 0:
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT available_balance, locked_balance FROM accounts WHERE account_id = ?", (self.account_id,))
            acc = cur.fetchone()
            if not acc or float(acc["available_balance"]) < estimated_total_cost:
                return False

            # Atomically debit available cash, credit locked balance
            cur.execute(
                """
                UPDATE accounts 
                SET available_balance = available_balance - ?,
                    locked_balance = locked_balance + ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (estimated_total_cost, estimated_total_cost, now_iso, self.account_id),
            )
            cur.execute(
                "UPDATE orders SET reserved_funds_inr = ? WHERE order_id = ?",
                (estimated_total_cost, order_id),
            )
            # Post Journal
            tx_id = str(uuid.uuid4())
            self._post_journal(
                cur=cur,
                tx_id=tx_id,
                postings=[
                    ("LOCKED_CASH", estimated_total_cost, 0.0, f"Funds reserved for order {order_id}"),
                    ("CASH", 0.0, estimated_total_cost, f"Funds locked for order {order_id}"),
                ],
                ref_id=order_id,
                timestamp=now_iso,
            )
            return True

    def reserve_position_for_sell(self, order_id: str, market_pair: str, quantity: float) -> bool:
        """
        Reserves asset holdings for a SELL order to prevent concurrent overselling.
        """
        if quantity <= 0:
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT position_id, quantity, reserved_quantity FROM positions WHERE account_id = ? AND market_pair = ? AND is_open = 1",
                (self.account_id, market_pair),
            )
            pos = cur.fetchone()
            if not pos:
                return False

            unreserved_qty = float(pos["quantity"]) - float(pos["reserved_quantity"])
            if unreserved_qty < quantity - 1e-8:
                return False

            cur.execute(
                "UPDATE positions SET reserved_quantity = reserved_quantity + ?, updated_at = ? WHERE position_id = ?",
                (quantity, now_iso, pos["position_id"]),
            )
            cur.execute(
                "UPDATE orders SET reserved_quantity = ? WHERE order_id = ?",
                (quantity, order_id),
            )
            return True

    def release_order_reservation(self, order_id: str) -> None:
        """
        Releases only this order's still-reserved cash or asset quantity.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT side, market_pair, reserved_funds_inr, reserved_quantity FROM orders WHERE order_id = ?", (order_id,))
            order = cur.fetchone()
            if not order:
                return

            side = str(order["side"]).upper()
            reserved_cash = float(order["reserved_funds_inr"])
            reserved_qty = float(order["reserved_quantity"])

            if side == "BUY" and reserved_cash > 0:
                cur.execute(
                    """
                    UPDATE accounts 
                    SET available_balance = available_balance + ?,
                        locked_balance = MAX(0.0, locked_balance - ?),
                        updated_at = ?
                    WHERE account_id = ?
                    """,
                    (reserved_cash, reserved_cash, now_iso, self.account_id),
                )
                cur.execute("UPDATE orders SET reserved_funds_inr = 0.0 WHERE order_id = ?", (order_id,))
                # Journal
                tx_id = str(uuid.uuid4())
                self._post_journal(
                    cur=cur,
                    tx_id=tx_id,
                    postings=[
                        ("CASH", reserved_cash, 0.0, f"Unreserved cash refund for cancelled order {order_id}"),
                        ("LOCKED_CASH", 0.0, reserved_cash, f"Release locked funds for order {order_id}"),
                    ],
                    ref_id=order_id,
                    timestamp=now_iso,
                )

            elif side == "SELL" and reserved_qty > 0:
                pair = str(order["market_pair"])
                cur.execute(
                    """
                    UPDATE positions 
                    SET reserved_quantity = MAX(0.0, reserved_quantity - ?),
                        updated_at = ?
                    WHERE account_id = ? AND market_pair = ? AND is_open = 1
                    """,
                    (reserved_qty, now_iso, self.account_id, pair),
                )
                cur.execute("UPDATE orders SET reserved_quantity = 0.0 WHERE order_id = ?", (order_id,))

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
                    reserved_funds_inr, reserved_quantity,
                    planned_risk_inr, stop_loss_price, take_profit_price,
                    idempotency_key, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'PROPOSED', ?, ?, 0.0, 0.0, 0.0, ?, ?, ?, ?, ?, ?)
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
        exchange_trade_id: str | None = None,
        is_maker: bool = False,
    ) -> Dict[str, Any]:
        """
        Records a BUY fill atomically:
        - Validates order state and remaining quantity.
        - Deducts exact outlay from order's reserved funds and settles any remainder with available cash.
        - Posts balanced double-entry journal entries.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        fill_id = str(uuid.uuid4())
        trade_id = exchange_trade_id or f"fill-{uuid.uuid4().hex[:12]}"
        costs = self.fee_engine.calculate_buy_costs(price, quantity, is_maker)
        total_outlay = costs["total_outlay_inr"]

        with self.db.session() as conn:
            cur = conn.cursor()
            # 1. Verify order state and remaining unfulfilled quantity
            cur.execute("SELECT status, quantity, filled_quantity, reserved_funds_inr FROM orders WHERE order_id = ?", (order_id,))
            order = cur.fetchone()
            if not order:
                raise ValueError(f"Order {order_id} does not exist.")

            if order["status"] not in ["SUBMITTED", "PARTIAL", "PROPOSED", "AUTHORIZED"]:
                raise ValueError(f"Cannot fill order in status {order['status']}. Must be SUBMITTED or PARTIAL.")

            remaining_qty = float(order["quantity"]) - float(order["filled_quantity"])
            if quantity > remaining_qty + 1e-8:
                raise ValueError(f"Fill quantity {quantity} exceeds order remaining quantity {remaining_qty}")

            # 2. Settle against reserved funds
            reserved_cash = float(order["reserved_funds_inr"])
            cash_settlement_locked = min(reserved_cash, total_outlay)
            cash_settlement_extra = max(0.0, total_outlay - reserved_cash)
            leftover_refund = max(0.0, reserved_cash - total_outlay)

            # Insert Fill with UNIQUE exchange_trade_id constraint
            cur.execute(
                """
                INSERT INTO fills (
                    fill_id, order_id, exchange_trade_id, market_pair, side, price, quantity,
                    fee_inr, gst_on_fee_inr, tds_deducted_inr, net_proceeds_inr,
                    timestamp, is_paper
                ) VALUES (?, ?, ?, ?, 'BUY', ?, ?, ?, ?, 0.0, ?, ?, 1)
                """,
                (
                    fill_id, order_id, trade_id, market_pair, price, quantity,
                    costs["brokerage_fee_inr"], costs["gst_on_fee_inr"],
                    total_outlay, now_iso,
                ),
            )

            # Update account cash balances
            cur.execute(
                """
                UPDATE accounts 
                SET locked_balance = MAX(0.0, locked_balance - ?),
                    available_balance = available_balance - ? + ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (reserved_cash, cash_settlement_extra, leftover_refund, now_iso, self.account_id),
            )

            # Update order remaining reserved funds
            cur.execute(
                """
                UPDATE orders 
                SET filled_quantity = filled_quantity + ?,
                    reserved_funds_inr = 0.0,
                    updated_at = ?
                WHERE order_id = ?
                """,
                (quantity, now_iso, order_id),
            )

            # Update position
            cur.execute(
                "SELECT * FROM positions WHERE account_id = ? AND market_pair = ? AND is_open = 1",
                (self.account_id, market_pair),
            )
            pos = cur.fetchone()
            if pos:
                new_qty = float(pos["quantity"]) + quantity
                new_cost = float(pos["cost_basis_inr"]) + costs["gross_value_inr"]
                avg_price = new_cost / new_qty
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
                        position_id, account_id, market_pair, quantity, reserved_quantity,
                        average_entry_price, cost_basis_inr, current_price,
                        unrealized_pnl_inr, realized_pnl_inr, is_open, updated_at
                    ) VALUES (?, ?, ?, ?, 0.0, ?, ?, ?, 0.0, 0.0, 1, ?)
                    """,
                    (str(uuid.uuid4()), self.account_id, market_pair, quantity, price, costs["gross_value_inr"], price, now_iso),
                )

            # Post balanced double-entry journal entries:
            # Debit HOLDINGS: gross_value
            # Debit FEE_EXPENSE: brokerage_fee
            # Debit GST_EXPENSE: gst_on_fee
            # Credit CASH: total_outlay
            tx_id = str(uuid.uuid4())
            self._post_journal(
                cur=cur,
                tx_id=tx_id,
                postings=[
                    ("HOLDINGS", costs["gross_value_inr"], 0.0, f"Acquisition of {quantity} {market_pair}"),
                    ("FEE_EXPENSE", costs["brokerage_fee_inr"], 0.0, f"Brokerage fee for {order_id}"),
                    ("GST_EXPENSE", costs["gst_on_fee_inr"], 0.0, f"GST on brokerage for {order_id}"),
                    ("CASH", 0.0, total_outlay, f"Cash outlay for {quantity} {market_pair}"),
                ],
                ref_id=fill_id,
                timestamp=now_iso,
            )

            # Ledger balance snapshot
            cur.execute("SELECT available_balance, locked_balance FROM accounts WHERE account_id = ?", (self.account_id,))
            acc_cur = cur.fetchone()
            cur.execute(
                """
                INSERT INTO ledger_entries (
                    entry_id, account_id, entry_type, amount_inr,
                    balance_after_inr, reference_id, description, timestamp
                ) VALUES (?, ?, 'BUY_COST', ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()), self.account_id, -total_outlay,
                    float(acc_cur["available_balance"] + acc_cur["locked_balance"]), fill_id,
                    f"Buy {quantity} {market_pair} @ {price}", now_iso,
                ),
            )

        return {"fill_id": fill_id, "trade_id": trade_id, "costs": costs}

    def record_sell_fill(
        self,
        order_id: str,
        market_pair: str,
        price: float,
        quantity: float,
        exchange_trade_id: str | None = None,
        is_maker: bool = False,
    ) -> Dict[str, Any]:
        """
        Records a SELL fill atomically:
        - Reconciles held position, preventing overselling.
        - Credits net cash proceeds and records Section 194S TDS as an asset credit.
        - Calculates Section 115BBH VDA tax.
        - Posts balanced double-entry journal.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        fill_id = str(uuid.uuid4())
        trade_id = exchange_trade_id or f"fill-{uuid.uuid4().hex[:12]}"

        with self.db.session() as conn:
            cur = conn.cursor()
            # Verify order state
            cur.execute("SELECT status, quantity, filled_quantity, reserved_quantity FROM orders WHERE order_id = ?", (order_id,))
            order = cur.fetchone()
            if not order:
                raise ValueError(f"Order {order_id} does not exist.")

            if order["status"] not in ["SUBMITTED", "PARTIAL", "PROPOSED", "AUTHORIZED"]:
                raise ValueError(f"Cannot fill order in status {order['status']}. Must be SUBMITTED or PARTIAL.")

            # Verify position exists and has sufficient quantity
            cur.execute(
                "SELECT * FROM positions WHERE account_id = ? AND market_pair = ? AND is_open = 1",
                (self.account_id, market_pair),
            )
            pos = cur.fetchone()
            if not pos or float(pos["quantity"]) < quantity - 1e-8:
                raise ValueError(f"Cannot sell {quantity} {market_pair}. Held position quantity is only {pos['quantity'] if pos else 0.0}")

            pos_qty = float(pos["quantity"])
            pos_cost = float(pos["cost_basis_inr"])
            cost_basis_per_unit = pos_cost / pos_qty

            sell_results = self.fee_engine.calculate_sell_proceeds(
                sell_price=price,
                quantity=quantity,
                pure_cost_basis_per_unit=cost_basis_per_unit,
                is_maker=is_maker,
            )

            # Insert Fill with UNIQUE exchange_trade_id
            cur.execute(
                """
                INSERT INTO fills (
                    fill_id, order_id, exchange_trade_id, market_pair, side, price, quantity,
                    fee_inr, gst_on_fee_inr, tds_deducted_inr, net_proceeds_inr,
                    timestamp, is_paper
                ) VALUES (?, ?, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    fill_id, order_id, trade_id, market_pair, price, quantity,
                    sell_results["brokerage_fee_inr"], sell_results["gst_on_fee_inr"],
                    sell_results["tds_withheld_1pct_inr"], sell_results["net_cash_received_inr"],
                    now_iso,
                ),
            )

            # Credit Cash and TDS Receivable
            cur.execute(
                """
                UPDATE accounts 
                SET available_balance = available_balance + ?,
                    tds_receivable_inr = tds_receivable_inr + ?,
                    cumulative_realized_loss = cumulative_realized_loss + ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (
                    sell_results["net_cash_received_inr"],
                    sell_results["tds_withheld_1pct_inr"],
                    abs(sell_results["economic_pnl_inr"]) if sell_results["economic_pnl_inr"] < 0 else 0.0,
                    now_iso,
                    self.account_id,
                ),
            )

            # Update Order filled quantity
            cur.execute(
                """
                UPDATE orders 
                SET filled_quantity = filled_quantity + ?,
                    reserved_quantity = MAX(0.0, reserved_quantity - ?),
                    updated_at = ?
                WHERE order_id = ?
                """,
                (quantity, quantity, now_iso, order_id),
            )

            # Update Position
            remaining_qty = max(0.0, pos_qty - quantity)
            remaining_cost = cost_basis_per_unit * remaining_qty
            is_open = 1 if remaining_qty > 1e-8 else 0
            cur.execute(
                """
                UPDATE positions 
                SET quantity = ?,
                    reserved_quantity = MAX(0.0, reserved_quantity - ?),
                    cost_basis_inr = ?,
                    realized_pnl_inr = realized_pnl_inr + ?,
                    is_open = ?,
                    updated_at = ?
                WHERE position_id = ?
                """,
                (
                    remaining_qty, quantity, remaining_cost, sell_results["economic_pnl_inr"],
                    is_open, now_iso, pos["position_id"],
                ),
            )

            # Record Statutory Indian Tax Record (derived FY)
            fy = self.fee_engine.get_indian_financial_year(now_iso)
            cur.execute(
                """
                INSERT INTO tax_records (
                    record_id, account_id, financial_year, fill_id, market_pair,
                    sale_proceeds_inr, cost_basis_inr, economic_gain_inr,
                    taxable_gain_vda_inr, tax_liability_30pct_inr, tds_withheld_1pct_inr,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()), self.account_id, fy, fill_id, market_pair,
                    sell_results["gross_sale_inr"], sell_results["tax_cost_basis_inr"],
                    sell_results["economic_pnl_inr"], sell_results["taxable_vda_gain_inr"],
                    sell_results["vda_tax_liability_30pct_inr"], sell_results["tds_withheld_1pct_inr"],
                    now_iso,
                ),
            )

            # Post balanced double-entry journal entries:
            # Debit CASH: net_cash_received
            # Debit TDS_RECEIVABLE: tds_withheld
            # Debit FEE_EXPENSE: fee
            # Debit GST_EXPENSE: gst
            # Debit/Credit REALIZED_PNL: economic_pnl balance
            # Credit HOLDINGS: sold_cost_basis
            sold_cost_basis = round(cost_basis_per_unit * quantity, 4)
            tx_id = str(uuid.uuid4())
            # Accounting Gain on asset = Gross Sale - Sold Cost Basis
            asset_gain = round(sell_results["gross_sale_inr"] - sold_cost_basis, 4)
            gain_postings = (
                [("REALIZED_PNL", abs(asset_gain), 0.0, "Loss on asset sale")]
                if asset_gain < 0
                else [("REALIZED_PNL", 0.0, asset_gain, "Gain on asset sale")]
            )

            # Debits: Net cash + TDS withheld + Brokerage fee + GST on fee == Gross Sale
            # Credits: Sold asset cost basis + Asset gain == Gross Sale
            self._post_journal(
                cur=cur,
                tx_id=tx_id,
                postings=[
                    ("CASH", sell_results["net_cash_received_inr"], 0.0, f"Net cash proceeds from {market_pair}"),
                    ("TDS_RECEIVABLE", sell_results["tds_withheld_1pct_inr"], 0.0, f"TDS credit withheld u/s 194S"),
                    ("FEE_EXPENSE", sell_results["brokerage_fee_inr"], 0.0, f"Brokerage fee for {order_id}"),
                    ("GST_EXPENSE", sell_results["gst_on_fee_inr"], 0.0, f"GST on fee for {order_id}"),
                    ("HOLDINGS", 0.0, sold_cost_basis, f"Cost basis of sold {quantity} {market_pair}"),
                ] + gain_postings,
                ref_id=fill_id,
                timestamp=now_iso,
            )

            cur.execute("SELECT available_balance, locked_balance FROM accounts WHERE account_id = ?", (self.account_id,))
            acc_cur = cur.fetchone()
            cur.execute(
                """
                INSERT INTO ledger_entries (
                    entry_id, account_id, entry_type, amount_inr,
                    balance_after_inr, reference_id, description, timestamp
                ) VALUES (?, ?, 'SELL_PROCEEDS', ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()), self.account_id, sell_results["net_cash_received_inr"],
                    float(acc_cur["available_balance"] + acc_cur["locked_balance"]), fill_id,
                    f"Sell {quantity} {market_pair} @ {price}", now_iso,
                ),
            )

        return {"fill_id": fill_id, "trade_id": trade_id, "sell_results": sell_results}
