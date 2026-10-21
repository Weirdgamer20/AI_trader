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
        require_order_book: bool = False,
    ) -> Dict[str, Any]:
        """
        Validates proposed order against strict deterministic risk rules:
        - Strict 2.0% risk cap and remaining affordable loss ceiling budget (P0-6).
        - Accounts for existing positions AND pending authorized/submitted BUY orders (P1-1).
        - Validates order book spread and bid/ask integrity (P1-2).
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

        # 4. Check Bid/Ask Spread (Reject missing, crossed, or excessive spread) (P1-2)
        if require_order_book or (bid_price is not None or ask_price is not None):
            if bid_price is None or ask_price is None or bid_price <= 0 or ask_price <= 0:
                return {
                    "approved": False,
                    "rejection_reason": "Order book integrity failure: Missing or non-positive bid/ask prices required for execution",
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

        # P1-1: Account for existing open positions and pending BUY orders
        with self.ledger.db.session() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT market_pair, SUM(quantity * price) as pending_notional, COUNT(*) as pending_count
                FROM orders
                WHERE account_id = ? AND status IN ('AUTHORIZED', 'SUBMITTED', 'PARTIAL') AND side = 'BUY'
                GROUP BY market_pair
                """,
                (self.ledger.account_id,),
            )
            pending_orders = {r["market_pair"]: (float(r["pending_notional"] or 0.0), int(r["pending_count"] or 0)) for r in cur.fetchall()}

            cur.execute("SELECT DISTINCT market_pair FROM positions WHERE account_id = ? AND is_open = 1", (self.ledger.account_id,))
            open_pos_pairs = {r[0] for r in cur.fetchall()}
            all_active_pairs = open_pos_pairs.union(set(pending_orders.keys()))

            cur.execute(
                "SELECT quantity, current_price FROM positions WHERE account_id = ? AND market_pair = ? AND is_open = 1",
                (self.ledger.account_id, market_pair),
            )
            pos_cur = cur.fetchone()
            cur_pos_val = float(pos_cur["quantity"] * pos_cur["current_price"]) if pos_cur else 0.0
            pending_coin_notional = pending_orders.get(market_pair, (0.0, 0))[0]
            existing_coin_exposure = cur_pos_val + pending_coin_notional

        # ==========================================
        # BUY ORDER VALIDATION & POSITION SIZING
        # ==========================================
        if normalized_side == "BUY":
            if market_pair not in all_active_pairs and len(all_active_pairs) >= self.params.max_open_positions:
                return {
                    "approved": False,
                    "rejection_reason": f"Maximum concurrent positions/slots ({self.params.max_open_positions}) reached including pending orders",
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

            # Strict 2.0% planned risk cap AND remaining loss ceiling budget (P0-6)
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
            remaining_coin_budget = max(0.0, max_coin_allocation - existing_coin_exposure)
            if remaining_coin_budget <= 0:
                return {
                    "approved": False,
                    "rejection_reason": (
                        f"Maximum portfolio allocation ({self.params.max_portfolio_allocation_per_coin_pct*100:.1f}%) "
                        f"for coin {market_pair} exceeded (existing exposure ₹{existing_coin_exposure:.2f} >= max ₹{max_coin_allocation:.2f})"
                    ),
                    "planned_risk_inr": 0.0,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                    "estimated_total_cost_inr": 0.0,
                }

            allowed_notional = min(desired_notional, remaining_coin_budget, available_cash)

            # Check minimum exchange order constraint (P0-6)
            if allowed_notional < self.params.min_order_value_inr:
                implied_risk = self.params.min_order_value_inr * effective_downside_pct
                max_tolerable_risk = min(hard_max_risk_inr, allowed_planned_risk_inr)
                if implied_risk > max_tolerable_risk:
                    return {
                        "approved": False,
                        "rejection_reason": (
                            f"Order rejected: Minimum order value ₹{self.params.min_order_value_inr} would incur "
                            f"planned risk of ₹{implied_risk:.2f}, strictly exceeding the 2% hard risk limit of ₹{hard_max_risk_inr:.2f} "
                            f"(allowable limit ₹{max_tolerable_risk:.2f}, remaining loss budget ₹{ceiling_status['remaining_budget_inr']:.2f})."
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
                if self.params.min_order_value_inr > remaining_coin_budget:
                    return {
                        "approved": False,
                        "rejection_reason": f"Minimum order value ₹{self.params.min_order_value_inr} exceeds remaining coin allocation budget ₹{remaining_coin_budget:.2f}",
                        "planned_risk_inr": 0.0,
                        "quantity": 0.0,
                        "position_notional_inr": 0.0,
                        "estimated_total_cost_inr": 0.0,
                    }
                allowed_notional = self.params.min_order_value_inr

            quantity = round(allowed_notional / current_price, 8)
            final_notional = round(quantity * current_price, 4)
            final_planned_risk = round(final_notional * effective_downside_pct, 4)

            # Absolute final assertion: Planned risk CANNOT exceed either hard limit or remaining budget (P0-6)
            max_tolerable_risk = min(hard_max_risk_inr, allowed_planned_risk_inr)
            if final_planned_risk > max_tolerable_risk + 1e-4:
                return {
                    "approved": False,
                    "rejection_reason": (
                        f"Planned risk ₹{final_planned_risk:.2f} strictly exceeds the 2% hard risk limit of ₹{hard_max_risk_inr:.2f} "
                        f"(allowable limit ₹{max_tolerable_risk:.2f}, remaining loss budget ₹{ceiling_status['remaining_budget_inr']:.2f})."
                    ),
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
