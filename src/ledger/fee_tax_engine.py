from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any

@dataclass(frozen=True)
class FeeSchedule:
    """Configurable, effective-dated fee schedule for CoinDCX Spot pairs."""
    maker_rate: float = 0.0050   # 0.50% maker
    taker_rate: float = 0.0050   # 0.50% taker
    gst_rate: float = 0.18       # 18% GST on brokerage fee
    tds_rate: float = 0.01       # 1% TDS under Section 194S on sell transfers
    vda_tax_rate: float = 0.30   # 30% flat tax under Section 115BBH on VDA gains

class FeeAndTaxEngine:
    """
    Computes exact fees, GST, Section 194S TDS, economic P&L,
    and statutory Indian VDA tax under Section 115BBH.
    """
    def __init__(self, schedule: FeeSchedule | None = None):
        self.schedule = schedule or FeeSchedule()

    @staticmethod
    def get_indian_financial_year(ts: str | datetime | None = None) -> str:
        """
        Derives Indian Financial Year (April 1 to March 31).
        E.g. 2026-05-15 -> 'FY2026-27'
             2027-02-10 -> 'FY2026-27'
             2025-11-20 -> 'FY2025-26'
        """
        if ts is None:
            dt = datetime.now(timezone.utc)
        elif isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                dt = datetime.now(timezone.utc)
        else:
            dt = ts

        year = dt.year
        if dt.month >= 4:
            # April to December: Current year to next year
            next_yy = str(year + 1)[-2:]
            return f"FY{year}-{next_yy}"
        else:
            # January to March: Previous year to current year
            prev_year = year - 1
            curr_yy = str(year)[-2:]
            return f"FY{prev_year}-{curr_yy}"

    def calculate_buy_costs(self, price: float, quantity: float, is_maker: bool = False) -> Dict[str, float]:
        """
        Pure acquisition cost = price * quantity.
        Brokerage fee = gross * rate.
        GST = brokerage * 18%.
        Total cash required = gross + fee + GST.
        """
        if price <= 0 or quantity <= 0:
            raise ValueError(f"Price and quantity must be positive, got price={price}, qty={quantity}")

        gross_value = round(price * quantity, 4)
        fee_rate = self.schedule.maker_rate if is_maker else self.schedule.taker_rate
        brokerage_fee = round(gross_value * fee_rate, 4)
        gst_on_fee = round(brokerage_fee * self.schedule.gst_rate, 4)
        total_outlay = round(gross_value + brokerage_fee + gst_on_fee, 4)

        return {
            "gross_value_inr": gross_value,
            "brokerage_fee_inr": brokerage_fee,
            "gst_on_fee_inr": gst_on_fee,
            "total_outlay_inr": total_outlay,
            "effective_cost_per_unit": round(total_outlay / quantity, 4),
        }

    def calculate_sell_proceeds(
        self,
        sell_price: float,
        quantity: float,
        pure_cost_basis_per_unit: float | None = None,
        total_cost_basis_per_unit: float | None = None,
        is_maker: bool = False,
        cost_basis_per_unit: float | None = None,
    ) -> Dict[str, float]:
        """
        Calculates proceeds, fees, GST, Section 194S TDS, economic P&L,
        and Section 115BBH tax liability.
        
        Section 115BBH Rule:
        - Gain = Gross Sale - Pure Cost of Acquisition (brokerage fees NOT deductible).
        - Tax = Gain * 30% if Gain > 0 else 0.
        
        Economic P&L Rule:
        - Realized economic cash gain factoring in all fees and GST.
        """
        if pure_cost_basis_per_unit is None:
            pure_cost_basis_per_unit = cost_basis_per_unit if cost_basis_per_unit is not None else 0.0
        if sell_price <= 0 or quantity <= 0:
            raise ValueError(f"Sell price and quantity must be positive, got price={sell_price}, qty={quantity}")

        gross_sale = round(sell_price * quantity, 4)
        fee_rate = self.schedule.maker_rate if is_maker else self.schedule.taker_rate
        brokerage_fee = round(gross_sale * fee_rate, 4)
        gst_on_fee = round(brokerage_fee * self.schedule.gst_rate, 4)
        tds_withheld = round(gross_sale * self.schedule.tds_rate, 4)

        net_cash_received = round(gross_sale - brokerage_fee - gst_on_fee - tds_withheld, 4)

        # Tax acquisition cost (pure) vs Economic cost (all-inclusive)
        tax_cost_basis = round(pure_cost_basis_per_unit * quantity, 4)
        econ_cost_unit = total_cost_basis_per_unit if total_cost_basis_per_unit is not None else pure_cost_basis_per_unit
        econ_cost_basis = round(econ_cost_unit * quantity, 4)

        economic_pnl = round(gross_sale - econ_cost_basis - brokerage_fee - gst_on_fee, 4)

        # Section 115BBH:
        raw_gain = gross_sale - tax_cost_basis
        taxable_vda_gain = round(max(0.0, raw_gain), 4)
        vda_tax_liability = round(taxable_vda_gain * self.schedule.vda_tax_rate, 4)

        return {
            "gross_sale_inr": gross_sale,
            "brokerage_fee_inr": brokerage_fee,
            "gst_on_fee_inr": gst_on_fee,
            "tds_withheld_1pct_inr": tds_withheld,
            "net_cash_received_inr": net_cash_received,
            "tax_cost_basis_inr": tax_cost_basis,
            "economic_cost_basis_inr": econ_cost_basis,
            "economic_pnl_inr": economic_pnl,
            "taxable_vda_gain_inr": taxable_vda_gain,
            "vda_tax_liability_30pct_inr": vda_tax_liability,
            "net_after_tax_pnl_inr": round(economic_pnl - vda_tax_liability, 4),
        }
