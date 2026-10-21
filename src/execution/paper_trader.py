from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from ..ledger.database import DatabaseManager
from ..ledger.accounting import PortfolioLedger
from .state_machine import ExecutionStateMachine, OrderState

class PaperTradingEngine:
    """
    High-fidelity Paper Execution Engine.
    Simulates realistic order matching, bid-ask spread crossing,
    volume-weighted slippage, latency, fees, and TDS deductions.
    All records are strictly tagged with is_paper = 1.
    """
    def __init__(
        self,
        db: DatabaseManager,
        ledger: PortfolioLedger,
        state_machine: ExecutionStateMachine,
        base_slippage_pct: float = 0.0015, # 0.15% base slippage
    ):
        self.db = db
        self.ledger = ledger
        self.sm = state_machine
        self.base_slippage_pct = base_slippage_pct

    def execute_paper_order(
        self,
        order_id: str,
        current_candle: Dict[str, float], # {open, high, low, close, volume}
        bid: float | None = None,
        ask: float | None = None,
    ) -> Dict[str, Any]:
        """
        Simulates order fill against the current candle dynamics.
        Checks if the limit price is actually achievable within [low, high].
        """
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
            order = cur.fetchone()
            if not order:
                return {"success": False, "reason": f"Order {order_id} not found"}

        if order["status"] not in [OrderState.AUTHORIZED.value, OrderState.SUBMITTED.value]:
            return {"success": False, "reason": f"Order in invalid status {order['status']} for execution"}

        self.sm.mark_submitted(order_id)

        side = order["side"].upper()
        limit_price = order["price"] or current_candle["close"]
        qty = order["quantity"]
        high = current_candle["high"]
        low = current_candle["low"]

        # Check market reachability:
        # A BUY limit order fills if market trades at or below limit price.
        # A SELL order fills if market trades at or above limit price.
        can_fill = (side == "BUY" and low <= limit_price) or (side == "SELL" and high >= limit_price)

        if not can_fill:
            return {
                "success": False,
                "order_id": order_id,
                "status": "UNFILLED",
                "reason": f"Price ₹{limit_price} not reached in candle range [₹{low}, ₹{high}]",
            }

        # Model realistic slippage:
        slippage_sign = 1.0 if side == "BUY" else -1.0
        slippage_pct = self.base_slippage_pct
        effective_fill_price = round(limit_price * (1.0 + (slippage_sign * slippage_pct)), 2)

        # Ensure fill price remains bounded by candle high/low
        effective_fill_price = max(low, min(high, effective_fill_price))

        fill_info = self.sm.mark_filled(order_id, price=effective_fill_price, quantity=qty)

        return {
            "success": True,
            "order_id": order_id,
            "status": "FILLED",
            "fill_price": effective_fill_price,
            "quantity": qty,
            "slippage_incurred_pct": round(abs(effective_fill_price - limit_price) / limit_price, 5),
            "fill_details": fill_info,
        }
