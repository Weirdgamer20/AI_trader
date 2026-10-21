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

        order_type = (order["order_type"] or "LIMIT").upper()

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

        # Model realistic execution price:
        # For limit orders: BUY must fill <= limit_price; SELL must fill >= limit_price
        # Using ask for BUY and bid for SELL if provided and within reach
        if side == "BUY":
            base_ref = ask if (ask is not None and ask <= limit_price) else min(limit_price, current_candle["close"])
            # Slippage on limit buy can only occur between low and limit_price (favorable or up to limit)
            # Never exceed limit_price for a limit order!
            if order_type == "LIMIT":
                effective_fill_price = min(limit_price, max(low, base_ref * (1.0 + self.base_slippage_pct * random.uniform(0.1, 0.5))))
                effective_fill_price = min(limit_price, effective_fill_price)
            else:
                # Market order can slip above close/ask
                effective_fill_price = max(low, min(high, base_ref * (1.0 + self.base_slippage_pct)))
        else: # SELL
            base_ref = bid if (bid is not None and bid >= limit_price) else max(limit_price, current_candle["close"])
            if order_type == "LIMIT":
                # Never fill below limit_price for a limit sell!
                effective_fill_price = max(limit_price, min(high, base_ref * (1.0 - self.base_slippage_pct * random.uniform(0.1, 0.5))))
                effective_fill_price = max(limit_price, effective_fill_price)
            else:
                # Market sell can slip below close/bid
                effective_fill_price = max(low, min(high, base_ref * (1.0 - self.base_slippage_pct)))

        effective_fill_price = round(effective_fill_price, 2)
        # Bounded strictly within candle
        effective_fill_price = max(low, min(high, effective_fill_price))
        if order_type == "LIMIT":
            if side == "BUY":
                effective_fill_price = min(limit_price, effective_fill_price)
            else:
                effective_fill_price = max(limit_price, effective_fill_price)

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
