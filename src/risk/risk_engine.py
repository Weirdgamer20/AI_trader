from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional
from ..ledger.database import DatabaseManager
from ..ledger.accounting import PortfolioLedger
from .loss_ceiling import AffordableLossCeiling

@dataclass(frozen=True)
class RiskParameters:
    max_risk_per_trade_pct: float = 0.02   # Strictly 2% of capital max planned loss
    max_portfolio_allocation_per_coin_pct: float = 0.30  # Max 30% capital in single coin
    max_open_positions: int = 5
    max_spread_pct: float = 0.015          # Max 1.5% bid-ask spread
    min_order_value_inr: float = 100.0     # CoinDCX minimum spot order ~₹100
    slippage_buffer_pct: float = 0.003     # 0.3% expected slippage buffer
    fee_buffer_pct: float = 0.0059         # 0.50% fee + 18% GST buffer

class DeterministicRiskEngine:
    """
    Deterministic Risk Engine with absolute authority.
    No ML model or AI heuristic can override its rules.
    
    Invariants Enforced:
    1. Planned risk can NEVER exceed strict 2% of capital under any circumstance.
    2. Only 'BUY' and 'SELL' sides permitted; all others rejected.
    3. All prices, stops, quantities, and uncertainties must be positive, finite numbers.
    4. For BUY: Stop-loss must be < current_price.
    5. For SELL: Position must exist, have positive quantity, and unreserved quantity >= sell quantity.
    6. If minimum exchange order size would cause planned risk to exceed 2%, the order is REJECTED.
    """
    def __init__(
        self,
        ledger: PortfolioLedger,
        ceiling: AffordableLossCeiling,
        params: RiskParameters | None = None,
    ):
        self.ledger = ledger
        self.ceiling = ceiling
        self.params = params or RiskParameters()

    def evaluate_order_proposal(
        self,
        market_pair: str,
        side: str,
        current_price: float,
        proposed_stop_loss: float = 0.0,
        sell_quantity: float = 0.0,
        bid_price: float | None = None,
        ask_price: float | None = None,
        forecast_uncertainty: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Validates proposed order against strict deterministic risk rules.
        """
        # 1. Validate Side strictly
        normalized_side = str(side).strip().upper()
        if normalized_side not in ("BUY", "SELL"):
            return {
                "approved": False,
                "rejection_reason": f"Invalid order side '{side}'. Must be 'BUY' or 'SELL'.",
                "planned_risk_inr": 0.0,
                "quantity": 0.0,
                "position_notional_inr": 0.0,
                "estimated_total_cost_inr": 0.0,
            }

        # 2. Validate numeric finiteness
        for val_name, val in [
            ("current_price", current_price),
            ("proposed_stop_loss", proposed_stop_loss),
            ("sell_quantity", sell_quantity),
            ("forecast_uncertainty", forecast_uncertainty),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                return {
                    "approved": False,
                    "rejection_reason": f"Invalid non-finite or negative numeric value for {val_name}: {val}",
                    "planned_risk_inr": 0.0,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                    "estimated_total_cost_inr": 0.0,
                }

        if current_price <= 0:
            return {
                "approved": False,
                "rejection_reason": f"current_price must be strictly positive, got {current_price}",
                "planned_risk_inr": 0.0,
                "quantity": 0.0,
                "position_notional_inr": 0.0,
                "estimated_total_cost_inr": 0.0,
            }

        # 3. Check Loss Ceiling Status
        ceiling_status = self.ceiling.evaluate_ceiling_status()
        if not ceiling_status["is_trading_allowed"]:
            return {
                "approved": False,
                "rejection_reason": "Trading halted: Affordable loss ceiling breached or account inactive",
                "planned_risk_inr": 0.0,
                "quantity": 0.0,
                "position_notional_inr": 0.0,
                "estimated_total_cost_inr": 0.0,
            }

        # 4. Check Bid/Ask Spread (Reject missing, crossed, or excessive spread)
        if bid_price is not None or ask_price is not None:
            if bid_price is None or ask_price is None or bid_price <= 0 or ask_price <= 0:
                return {
                    "approved": False,
                    "rejection_reason": "Order book integrity failure: Missing or non-positive bid/ask prices",
                    "planned_risk_inr": 0.0,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                    "estimated_total_cost_inr": 0.0,
                }
            if bid_price >= ask_price:
                return {
                    "approved": False,
                    "rejection_reason": f"Order book crossed or locked: bid {bid_price} >= ask {ask_price}",
                    "planned_risk_inr": 0.0,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                    "estimated_total_cost_inr": 0.0,
                }
            midpoint = (ask_price + bid_price) / 2.0
            spread_pct = (ask_price - bid_price) / midpoint
            if spread_pct > self.params.max_spread_pct:
                return {
                    "approved": False,
                    "rejection_reason": f"Spread {spread_pct*100:.2f}% exceeds maximum allowable {self.params.max_spread_pct*100:.2f}%",
                    "planned_risk_inr": 0.0,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                    "estimated_total_cost_inr": 0.0,
                }

        account = self.ledger.get_account_summary()
        current_capital = account["total_equity_inr"]
        available_cash = account["available_balance_inr"]

        # ==========================================
        # BUY ORDER VALIDATION & POSITION SIZING
        # ==========================================
        if normalized_side == "BUY":
            if account["open_positions_count"] >= self.params.max_open_positions:
                return {
                    "approved": False,
                    "rejection_reason": f"Maximum concurrent positions ({self.params.max_open_positions}) reached",
                    "planned_risk_inr": 0.0,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                    "estimated_total_cost_inr": 0.0,
                }

            if proposed_stop_loss >= current_price or proposed_stop_loss <= 0:
                return {
                    "approved": False,
                    "rejection_reason": f"Invalid stop loss price ₹{proposed_stop_loss} for buy at ₹{current_price}. Must be 0 < stop < price.",
                    "planned_risk_inr": 0.0,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                    "estimated_total_cost_inr": 0.0,
                }

            # Downside risk fraction including slippage and uncertainty penalty
            raw_downside_pct = (current_price - proposed_stop_loss) / current_price
            effective_downside_pct = (
                raw_downside_pct + self.params.slippage_buffer_pct + self.params.fee_buffer_pct + (0.5 * forecast_uncertainty)
            )

            # Strict 2.0% planned risk cap
            hard_max_risk_inr = round(current_capital * self.params.max_risk_per_trade_pct, 4)
            allowed_planned_risk_inr = min(hard_max_risk_inr, ceiling_status["remaining_budget_inr"])

            if allowed_planned_risk_inr <= 0:
                return {
                    "approved": False,
                    "rejection_reason": "No remaining risk budget under affordable loss ceiling",
                    "planned_risk_inr": 0.0,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                    "estimated_total_cost_inr": 0.0,
                }

            # Sizing formula: Notional = Planned Risk / Downside %
            desired_notional = allowed_planned_risk_inr / effective_downside_pct
            max_coin_allocation = current_capital * self.params.max_portfolio_allocation_per_coin_pct
            allowed_notional = min(desired_notional, max_coin_allocation, available_cash)

            # Check minimum exchange order constraint
            if allowed_notional < self.params.min_order_value_inr:
                # Invariant: NEVER permit minimum order size if its risk exceeds strict 2.0% limit
                implied_risk = self.params.min_order_value_inr * effective_downside_pct
                if implied_risk > hard_max_risk_inr:
                    return {
                        "approved": False,
                        "rejection_reason": (
                            f"Order rejected: Minimum order value ₹{self.params.min_order_value_inr} would incur "
                            f"planned risk of ₹{implied_risk:.2f}, strictly exceeding the 2% hard risk limit of ₹{hard_max_risk_inr:.2f}."
                        ),
                        "planned_risk_inr": round(implied_risk, 2),
                        "quantity": 0.0,
                        "position_notional_inr": 0.0,
                        "estimated_total_cost_inr": 0.0,
                    }
                if available_cash < self.params.min_order_value_inr:
                    return {
                        "approved": False,
                        "rejection_reason": f"Insufficient available cash (₹{available_cash:.2f}) for minimum order ₹{self.params.min_order_value_inr}",
                        "planned_risk_inr": 0.0,
                        "quantity": 0.0,
                        "position_notional_inr": 0.0,
                        "estimated_total_cost_inr": 0.0,
                    }
                allowed_notional = self.params.min_order_value_inr

            quantity = round(allowed_notional / current_price, 8)
            final_notional = round(quantity * current_price, 4)
            final_planned_risk = round(final_notional * effective_downside_pct, 4)

            # Absolute final assertion: Planned risk CANNOT exceed hard limit
            if final_planned_risk > hard_max_risk_inr + 1e-4:
                return {
                    "approved": False,
                    "rejection_reason": f"Planned risk ₹{final_planned_risk:.2f} strictly exceeds hard 2% limit of ₹{hard_max_risk_inr:.2f}",
                    "planned_risk_inr": final_planned_risk,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                    "estimated_total_cost_inr": 0.0,
                }

            # Calculate full outlay reservation: notional + 0.50% fee + 18% GST
            est_fee = final_notional * 0.0050
            est_gst = est_fee * 0.18
            estimated_total_cost = round(final_notional + est_fee + est_gst, 4)

            if estimated_total_cost > available_cash:
                return {
                    "approved": False,
                    "rejection_reason": f"Available cash ₹{available_cash:.2f} insufficient to cover full outlay including fees ₹{estimated_total_cost:.2f}",
                    "planned_risk_inr": final_planned_risk,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                    "estimated_total_cost_inr": 0.0,
                }

            return {
                "approved": True,
                "rejection_reason": None,
                "planned_risk_inr": final_planned_risk,
                "quantity": quantity,
                "position_notional_inr": final_notional,
                "estimated_total_cost_inr": estimated_total_cost,
                "effective_downside_pct": round(effective_downside_pct, 4),
            }

        # ==========================================
        # SELL ORDER VALIDATION
        # ==========================================
        else: # SELL
            if sell_quantity <= 0:
                return {
                    "approved": False,
                    "rejection_reason": f"sell_quantity must be strictly positive, got {sell_quantity}",
                    "planned_risk_inr": 0.0,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                    "estimated_total_cost_inr": 0.0,
                }

            # Check that position exists and has unreserved quantity
            with self.ledger.db.session() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT quantity, reserved_quantity FROM positions WHERE account_id = ? AND market_pair = ? AND is_open = 1",
                    (self.ledger.account_id, market_pair),
                )
                pos = cur.fetchone()
                if not pos:
                    return {
                        "approved": False,
                        "rejection_reason": f"Cannot sell {market_pair}: No open position held.",
                        "planned_risk_inr": 0.0,
                        "quantity": 0.0,
                        "position_notional_inr": 0.0,
                        "estimated_total_cost_inr": 0.0,
                    }

                unreserved = float(pos["quantity"]) - float(pos["reserved_quantity"])
                if unreserved < sell_quantity - 1e-8:
                    return {
                        "approved": False,
                        "rejection_reason": f"Cannot sell {sell_quantity} {market_pair}: Available unreserved quantity is only {unreserved:.8f} (Holding: {pos['quantity']})",
                        "planned_risk_inr": 0.0,
                        "quantity": 0.0,
                        "position_notional_inr": 0.0,
                        "estimated_total_cost_inr": 0.0,
                    }

            return {
                "approved": True,
                "rejection_reason": None,
                "planned_risk_inr": 0.0,
                "quantity": sell_quantity,
                "position_notional_inr": round(sell_quantity * current_price, 4),
                "estimated_total_cost_inr": 0.0,
                "effective_downside_pct": 0.0,
            }
