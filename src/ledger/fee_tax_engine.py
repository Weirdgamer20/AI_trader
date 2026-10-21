from __future__ import annotations

from dataclasses import dataclass
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

    def calculate_buy_costs(self, price: float, quantity: float, is_maker: bool = False) -> Dict[str, float]:
        """
        Gross purchase cost = price * quantity.
        Fee = gross * maker/taker rate.
        GST = fee * 18%.
        Total cash required = gross + fee + GST.
        """
        gross_value = price * quantity
        fee_rate = self.schedule.maker_rate if is_maker else self.schedule.taker_rate
        brokerage_fee = gross_value * fee_rate
        gst_on_fee = brokerage_fee * self.schedule.gst_rate
        total_outlay = gross_value + brokerage_fee + gst_on_fee

        return {
            "gross_value_inr": round(gross_value, 4),
            "brokerage_fee_inr": round(brokerage_fee, 4),
            "gst_on_fee_inr": round(gst_on_fee, 4),
            "total_outlay_inr": round(total_outlay, 4),
            "effective_cost_per_unit": round(total_outlay / quantity if quantity > 0 else price, 4),
        }

    def calculate_sell_proceeds(
        self,
        sell_price: float,
        quantity: float,
        cost_basis_per_unit: float,
        is_maker: bool = False,
    ) -> Dict[str, float]:
        """
        Calculates proceeds, fees, GST, Section 194S TDS, economic P&L,
        and Section 115BBH tax liability.
        
        Section 115BBH Rule:
        - Gain = Gross Sale Value - Cost of Acquisition (brokerage fees NOT deductible from VDA gains under 115BBH).
        - If Gain > 0: Tax = Gain * 30%.
        - If Gain <= 0: Tax = 0 (Losses cannot be set off against other VDA gains).
        
        Economic P&L Rule:
        - Economic P&L = Gross Sale - Gross Buy - All Brokerage Fees - All GST.
        """
        gross_sale = sell_price * quantity
        fee_rate = self.schedule.maker_rate if is_maker else self.schedule.taker_rate
        brokerage_fee = gross_sale * fee_rate
        gst_on_fee = brokerage_fee * self.schedule.gst_rate
        tds_withheld = gross_sale * self.schedule.tds_rate

        net_cash_received = gross_sale - brokerage_fee - gst_on_fee - tds_withheld
        acquisition_cost = cost_basis_per_unit * quantity

        # Economic P&L represents actual money made or lost
        economic_pnl = gross_sale - acquisition_cost - brokerage_fee - gst_on_fee

        # Indian Tax Law (Section 115BBH):
        raw_gain = gross_sale - acquisition_cost
        taxable_vda_gain = max(0.0, raw_gain)
        vda_tax_liability = taxable_vda_gain * self.schedule.vda_tax_rate

        return {
            "gross_sale_inr": round(gross_sale, 4),
            "brokerage_fee_inr": round(brokerage_fee, 4),
            "gst_on_fee_inr": round(gst_on_fee, 4),
            "tds_withheld_1pct_inr": round(tds_withheld, 4),
            "net_cash_received_inr": round(net_cash_received, 4),
            "cost_basis_inr": round(acquisition_cost, 4),
            "economic_pnl_inr": round(economic_pnl, 4),
            "taxable_vda_gain_inr": round(taxable_vda_gain, 4),
            "vda_tax_liability_30pct_inr": round(vda_tax_liability, 4),
            "net_after_tax_pnl_inr": round(economic_pnl - vda_tax_liability, 4),
        }
