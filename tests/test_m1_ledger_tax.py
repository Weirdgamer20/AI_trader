import pytest
from pathlib import Path
from src.ledger.database import DatabaseManager
from src.ledger.fee_tax_engine import FeeAndTaxEngine, FeeSchedule
from src.ledger.accounting import PortfolioLedger

def test_fee_and_tax_engine_buy_and_sell():
    engine = FeeAndTaxEngine(FeeSchedule(maker_rate=0.005, taker_rate=0.005, gst_rate=0.18, tds_rate=0.01, vda_tax_rate=0.30))
    
    # Buy 1 BTC at ₹10,000
    buy = engine.calculate_buy_costs(price=10000.0, quantity=1.0)
    assert buy["gross_value_inr"] == 10000.0
    assert buy["brokerage_fee_inr"] == 50.0 # 0.50%
    assert buy["gst_on_fee_inr"] == 9.0    # 18% of 50
    assert buy["total_outlay_inr"] == 10059.0

    # Profitable Sell 1 BTC at ₹12,000 (Cost basis ₹10,059)
    sell_profit = engine.calculate_sell_proceeds(sell_price=12000.0, quantity=1.0, cost_basis_per_unit=10059.0)
    assert sell_profit["gross_sale_inr"] == 12000.0
    assert sell_profit["brokerage_fee_inr"] == 60.0 # 0.50% of 12000
    assert sell_profit["gst_on_fee_inr"] == 10.8   # 18% of 60
    assert sell_profit["tds_withheld_1pct_inr"] == 120.0 # 1% of 12000
    # Net cash received = 12000 - 60 - 10.8 - 120 = 11809.2
    assert sell_profit["net_cash_received_inr"] == 11809.2
    # Section 115BBH VDA gain: 12000 - 10059 = 1941.0
    assert sell_profit["taxable_vda_gain_inr"] == 1941.0
    # 30% VDA tax = 582.3
    assert sell_profit["vda_tax_liability_30pct_inr"] == 582.3

    # Loss Sell 1 BTC at ₹8,000
    sell_loss = engine.calculate_sell_proceeds(sell_price=8000.0, quantity=1.0, cost_basis_per_unit=10059.0)
    assert sell_loss["gross_sale_inr"] == 8000.0
    assert sell_loss["taxable_vda_gain_inr"] == 0.0 # No tax on loss, and no loss set-off
    assert sell_loss["vda_tax_liability_30pct_inr"] == 0.0
    assert sell_loss["economic_pnl_inr"] < 0.0

def test_portfolio_ledger_lifecycle(tmp_path):
    db_file = tmp_path / "test_ledger.db"
    db = DatabaseManager(db_path=db_file)
    ledger = PortfolioLedger(db=db, account_id="test_acc")

    summary = ledger.get_account_summary()
    assert summary["starting_balance_inr"] == 500.0
    assert summary["available_balance_inr"] == 500.0
    assert summary["target_balance_inr"] == 100000.0
    assert summary["affordable_loss_ceiling_inr"] == 250.0

    # Lock funds for ₹100 order
    assert ledger.lock_funds(100.0) is True
    summary2 = ledger.get_account_summary()
    assert summary2["available_balance_inr"] == 400.0
    assert summary2["locked_balance_inr"] == 100.0

    # Cannot lock more than available
    assert ledger.lock_funds(450.0) is False

    # Create buy order first
    ledger.create_order(
        order_id="ord-1", client_order_id="client-1", market_pair="I-BTC_INR",
        side="BUY", price=50000.0, quantity=0.001, planned_risk_inr=10.0
    )
    # Buy fill: 0.001 units at ₹50,000 (gross ₹50, fee+GST = 0.25 + 0.045 = 0.295, total outlay ~50.295)
    buy_res = ledger.record_buy_fill(
        order_id="ord-1", market_pair="I-BTC_INR", price=50000.0, quantity=0.001
    )
    assert "fill_id" in buy_res
    summary3 = ledger.get_account_summary()
    assert summary3["open_positions_count"] == 1

    # Create sell order
    ledger.create_order(
        order_id="ord-2", client_order_id="client-2", market_pair="I-BTC_INR",
        side="SELL", price=60000.0, quantity=0.001
    )
    # Sell fill: 0.001 units at ₹60,000 (profitable)
    sell_res = ledger.record_sell_fill(
        order_id="ord-2", market_pair="I-BTC_INR", price=60000.0, quantity=0.001
    )
    assert "fill_id" in sell_res
    summary4 = ledger.get_account_summary()
    assert summary4["open_positions_count"] == 0
    assert summary4["total_taxable_vda_gain_inr"] > 0
    assert summary4["tds_withheld_inr"] > 0
