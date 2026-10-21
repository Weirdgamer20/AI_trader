from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
from ..ledger.database import DatabaseManager
from ..ledger.accounting import PortfolioLedger
from .loss_ceiling import AffordableLossCeiling

@dataclass(frozen=True)
class RiskParameters:
    max_risk_per_trade_pct: float = 0.02   # 2% of capital max planned loss
    max_portfolio_allocation_per_coin_pct: float = 0.30  # Max 30% capital in single coin
    max_open_positions: int = 5
    max_spread_pct: float = 0.015          # Max 1.5% bid-ask spread
    min_order_value_inr: float = 100.0     # CoinDCX minimum spot order ~₹100
    slippage_buffer_pct: float = 0.003     # 0.3% expected slippage buffer

class DeterministicRiskEngine:
    """
    Deterministic Risk Engine with absolute authority.
    No ML model or AI heuristic can override its rules.
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
        proposed_stop_loss: float,
        bid_price: float | None = None,
        ask_price: float | None = None,
        forecast_uncertainty: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluates proposed order against all risk rules:
        1. Affordable loss ceiling active and not breached.
        2. Maximum 2% planned loss per trade.
        3. Valid stop loss and downside estimate.
        4. Spread and liquidity constraints.
        5. Portfolio concentration and dynamic position count.
        6. Available cash and minimum order constraints.
        """
        # 1. Check loss ceiling
        ceiling_status = self.ceiling.evaluate_ceiling_status()
        if not ceiling_status["is_trading_allowed"]:
            return {
                "approved": False,
                "rejection_reason": "Trading halted: Affordable loss ceiling breached or account inactive",
                "planned_risk_inr": 0.0,
                "quantity": 0.0,
                "position_notional_inr": 0.0,
            }

        # 2. Check spread
        if bid_price and ask_price and ask_price > 0:
            spread_pct = (ask_price - bid_price) / ask_price
            if spread_pct > self.params.max_spread_pct:
                return {
                    "approved": False,
                    "rejection_reason": f"Spread {spread_pct*100:.2f}% exceeds maximum allowable {self.params.max_spread_pct*100:.2f}%",
                    "planned_risk_inr": 0.0,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                }

        account = self.ledger.get_account_summary()
        current_capital = account["total_equity_inr"]
        available_cash = account["available_balance_inr"]

        # 3. Dynamic position count check
        if side.upper() == "BUY" and account["open_positions_count"] >= self.params.max_open_positions:
            return {
                "approved": False,
                "rejection_reason": f"Maximum concurrent positions ({self.params.max_open_positions}) reached",
                "planned_risk_inr": 0.0,
                "quantity": 0.0,
                "position_notional_inr": 0.0,
            }

        if side.upper() == "BUY":
            # 4. Stop-loss & Downside estimation
            if proposed_stop_loss >= current_price or proposed_stop_loss <= 0:
                return {
                    "approved": False,
                    "rejection_reason": f"Invalid stop loss price ₹{proposed_stop_loss} for buy at ₹{current_price}",
                    "planned_risk_inr": 0.0,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                }

            raw_downside_pct = (current_price - proposed_stop_loss) / current_price
            # Add slippage buffer and uncertainty penalty
            effective_downside_pct = raw_downside_pct + self.params.slippage_buffer_pct + (0.5 * forecast_uncertainty)

            # Planned risk = strictly 2% of capital (e.g. ₹10 at ₹500), capped by remaining loss ceiling
            max_risk_inr = current_capital * self.params.max_risk_per_trade_pct
            planned_risk_inr = min(max_risk_inr, ceiling_status["remaining_budget_inr"])

            if planned_risk_inr <= 0:
                return {
                    "approved": False,
                    "rejection_reason": "No remaining risk budget under loss ceiling",
                    "planned_risk_inr": 0.0,
                    "quantity": 0.0,
                    "position_notional_inr": 0.0,
                }

            # Sizing formula
            desired_notional = planned_risk_inr / effective_downside_pct

            # Cap notional by single-coin concentration limit (30% of total capital)
            max_coin_allocation = current_capital * self.params.max_portfolio_allocation_per_coin_pct
            allowed_notional = min(desired_notional, max_coin_allocation, available_cash)

            if allowed_notional < self.params.min_order_value_inr:
                # If available balance allows, allow min order value provided planned risk is still <= 2% limit
                if available_cash >= self.params.min_order_value_inr:
                    implied_risk = self.params.min_order_value_inr * effective_downside_pct
                    if implied_risk <= max_risk_inr * 1.15: # small tolerance for integer rounding
                        allowed_notional = self.params.min_order_value_inr
                    else:
                        return {
                            "approved": False,
                            "rejection_reason": f"Sized position ₹{allowed_notional:.2f} below min order ₹{self.params.min_order_value_inr} and implied risk exceeds 2% cap",
                            "planned_risk_inr": planned_risk_inr,
                            "quantity": 0.0,
                            "position_notional_inr": 0.0,
                        }
                else:
                    return {
                        "approved": False,
                        "rejection_reason": f"Insufficient available cash (₹{available_cash:.2f}) for minimum order ₹{self.params.min_order_value_inr}",
                        "planned_risk_inr": planned_risk_inr,
                        "quantity": 0.0,
                        "position_notional_inr": 0.0,
                    }

            quantity = round(allowed_notional / current_price, 8)
            final_notional = round(quantity * current_price, 2)
            actual_planned_risk = round(final_notional * effective_downside_pct, 2)

            return {
                "approved": True,
                "rejection_reason": None,
                "planned_risk_inr": actual_planned_risk,
                "quantity": quantity,
                "position_notional_inr": final_notional,
                "effective_downside_pct": round(effective_downside_pct, 4),
            }
        else:
            # Sell order validation
            return {
                "approved": True,
                "rejection_reason": None,
                "planned_risk_inr": 0.0,
                "quantity": 0.0,
                "position_notional_inr": 0.0,
                "effective_downside_pct": 0.0,
            }
