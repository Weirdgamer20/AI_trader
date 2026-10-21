from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional
from ..ledger.database import DatabaseManager
from ..ledger.accounting import PortfolioLedger
from ..risk.risk_engine import DeterministicRiskEngine
from ..risk.emergency import EmergencyStopSystem

class OrderState(str, Enum):
    PROPOSED = "PROPOSED"
    RISK_CHECKED = "RISK_CHECKED"
    AUTHORIZED = "AUTHORIZED"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    RECONCILED = "RECONCILED"

class ExecutionStateMachine:
    """
    Strict State Machine for order execution:
    PROPOSED -> RISK_CHECKED -> AUTHORIZED -> SUBMITTED -> PARTIAL / FILLED / REJECTED -> RECONCILED.
    Guarantees idempotency, atomic fund reservation, and partial fill safety.
    """
    VALID_TRANSITIONS = {
        OrderState.PROPOSED: {OrderState.RISK_CHECKED, OrderState.REJECTED},
        OrderState.RISK_CHECKED: {OrderState.AUTHORIZED, OrderState.REJECTED},
        OrderState.AUTHORIZED: {OrderState.SUBMITTED, OrderState.CANCELLED, OrderState.REJECTED},
        OrderState.SUBMITTED: {OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED},
        OrderState.PARTIAL: {OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED},
        OrderState.FILLED: {OrderState.RECONCILED},
        OrderState.CANCELLED: {OrderState.RECONCILED},
        OrderState.REJECTED: {OrderState.RECONCILED},
        OrderState.RECONCILED: set(),
    }

    def __init__(
        self,
        db: DatabaseManager,
        ledger: PortfolioLedger,
        risk_engine: DeterministicRiskEngine,
        emergency_system: EmergencyStopSystem,
    ):
        self.db = db
        self.ledger = ledger
        self.risk_engine = risk_engine
        self.emergency = emergency_system

    def propose_order(
        self,
        market_pair: str,
        side: str,
        current_price: float,
        proposed_stop_loss: float = 0.0,
        sell_quantity: float = 0.0,
        bid_price: float | None = None,
        ask_price: float | None = None,
        forecast_uncertainty: float = 0.0,
        idempotency_key: str | None = None,
    ) -> Dict[str, Any]:
        """Creates PROPOSED order, runs risk check, and transitions to AUTHORIZED or REJECTED."""
        if self.emergency.is_safe_mode_active():
            return {
                "success": False,
                "order_id": None,
                "state": OrderState.REJECTED,
                "reason": "Safe mode active: All new orders blocked",
            }

        order_id = f"ord-{uuid.uuid4().hex[:12]}"
        client_id = f"cli-{uuid.uuid4().hex[:12]}"
        idemp = idempotency_key or f"idemp-{uuid.uuid4().hex[:16]}"

        # Prevent duplicate idempotency key atomically
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT order_id, status FROM orders WHERE idempotency_key = ?", (idemp,))
            existing = cur.fetchone()
            if existing:
                return {
                    "success": False,
                    "order_id": existing["order_id"],
                    "state": existing["status"],
                    "reason": "Duplicate order request: Idempotency key already submitted",
                }

        # 1. State: PROPOSED
        self.ledger.create_order(
            order_id=order_id,
            client_order_id=client_id,
            market_pair=market_pair,
            side=side,
            order_type="LIMIT",
            price=current_price,
            quantity=sell_quantity if side.upper() == "SELL" else 0.0,
            planned_risk_inr=0.0,
            stop_loss_price=proposed_stop_loss,
            idempotency_key=idemp,
        )

        # 2. Risk Evaluation
        eval_result = self.risk_engine.evaluate_order_proposal(
            market_pair=market_pair,
            side=side,
            current_price=current_price,
            proposed_stop_loss=proposed_stop_loss,
            sell_quantity=sell_quantity,
            bid_price=bid_price,
            ask_price=ask_price,
            forecast_uncertainty=forecast_uncertainty,
        )

        if not eval_result["approved"]:
            self._transition(order_id, OrderState.REJECTED, reason=eval_result["rejection_reason"])
            return {
                "success": False,
                "order_id": order_id,
                "state": OrderState.REJECTED,
                "reason": eval_result["rejection_reason"],
            }

        # Update order with sized quantity and planned risk
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE orders 
                SET quantity = ?, planned_risk_inr = ?
                WHERE order_id = ?
                """,
                (eval_result["quantity"], eval_result["planned_risk_inr"], order_id),
            )

        self._transition(order_id, OrderState.RISK_CHECKED)

        # 3. Transition to AUTHORIZED with strict per-order reservation
        if side.upper() == "BUY":
            est_cost = eval_result["estimated_total_cost_inr"]
            if not self.ledger.reserve_funds_for_order(order_id, est_cost):
                self._transition(order_id, OrderState.REJECTED, reason="Failed to reserve funds for order")
                return {
                    "success": False,
                    "order_id": order_id,
                    "state": OrderState.REJECTED,
                    "reason": "Failed to reserve funds",
                }
        else: # SELL
            if not self.ledger.reserve_position_for_sell(order_id, market_pair, eval_result["quantity"]):
                self._transition(order_id, OrderState.REJECTED, reason="Failed to reserve asset position for sell")
                return {
                    "success": False,
                    "order_id": order_id,
                    "state": OrderState.REJECTED,
                    "reason": "Failed to reserve asset position",
                }

        self._transition(order_id, OrderState.AUTHORIZED)

        return {
            "success": True,
            "order_id": order_id,
            "state": OrderState.AUTHORIZED,
            "quantity": eval_result["quantity"],
            "notional_inr": eval_result["position_notional_inr"],
            "planned_risk_inr": eval_result["planned_risk_inr"],
        }

    def mark_submitted(self, order_id: str) -> bool:
        return self._transition(order_id, OrderState.SUBMITTED)

    def mark_filled(
        self,
        order_id: str,
        price: float,
        quantity: float,
        exchange_trade_id: str | None = None,
        is_maker: bool = False,
    ) -> Dict[str, Any]:
        """
        Validates state machine transition and processes fill atomically.
        """
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT status, market_pair, side, quantity, filled_quantity FROM orders WHERE order_id = ?", (order_id,))
            order = cur.fetchone()
            if not order:
                raise ValueError(f"Order {order_id} not found.")

        current_status = order["status"]
        if current_status not in [OrderState.SUBMITTED.value, OrderState.PARTIAL.value]:
            raise ValueError(f"Cannot record fill for order in state {current_status}. Must be SUBMITTED or PARTIAL.")

        side = order["side"].upper()
        pair = order["market_pair"]
        tot_qty = float(order["quantity"])
        prev_filled = float(order["filled_quantity"])
        new_total_filled = prev_filled + quantity

        # Atomic fill recording in ledger
        if side == "BUY":
            fill_info = self.ledger.record_buy_fill(order_id, pair, price, quantity, exchange_trade_id, is_maker)
        else:
            fill_info = self.ledger.record_sell_fill(order_id, pair, price, quantity, exchange_trade_id, is_maker)

        # Transition state based on completion
        if new_total_filled >= tot_qty - 1e-8:
            self._transition(order_id, OrderState.FILLED, filled_qty=new_total_filled)
            self._transition(order_id, OrderState.RECONCILED)
        else:
            self._transition(order_id, OrderState.PARTIAL, filled_qty=new_total_filled)

        return fill_info

    def cancel_order(self, order_id: str, reason: str = "User cancelled") -> bool:
        """
        Cancels an order and releases any remaining per-order reservations.
        """
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,))
            order = cur.fetchone()
            if not order or order["status"] in [OrderState.FILLED.value, OrderState.CANCELLED.value, OrderState.RECONCILED.value]:
                return False

        # Release unconsumed reservations
        self.ledger.release_order_reservation(order_id)
        self._transition(order_id, OrderState.CANCELLED, reason=reason)
        self._transition(order_id, OrderState.RECONCILED)
        return True

    def _transition(
        self,
        order_id: str,
        new_state: OrderState,
        filled_qty: float | None = None,
        reason: str | None = None,
    ) -> bool:
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,))
            row = cur.fetchone()
            if not row:
                return False

            current_state = OrderState(row["status"])
            valid_targets = self.VALID_TRANSITIONS.get(current_state, set())

            if new_state not in valid_targets:
                now_iso = datetime.now(timezone.utc).isoformat()
                cur.execute(
                    """
                    INSERT INTO audit_events (event_id, event_type, severity, message, details_json, timestamp)
                    VALUES (?, 'ORDER_TRANSITION', 'CRITICAL', 'Illegal order state transition attempted', ?, ?)
                    """,
                    (f"audit-{uuid.uuid4().hex[:8]}", f'{{"order_id": "{order_id}", "from": "{current_state}", "to": "{new_state}"}}', now_iso),
                )
                return False

        self.ledger.update_order_status(order_id, new_state.value, filled_qty=filled_qty, reason=reason)
        return True
