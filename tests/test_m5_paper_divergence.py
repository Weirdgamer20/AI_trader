import pytest
from src.ledger.database import DatabaseManager
from src.ledger.accounting import PortfolioLedger
from src.risk.loss_ceiling import AffordableLossCeiling
from src.risk.risk_engine import DeterministicRiskEngine
from src.risk.emergency import EmergencyStopSystem
from src.execution.state_machine import ExecutionStateMachine, OrderState
from src.execution.paper_trader import PaperTradingEngine
from src.execution.divergence import DivergenceMonitor

def test_paper_trading_execution_within_and_outside_candle_bounds(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "paper_test.db")
    ledger = PortfolioLedger(db=db, account_id="paper_acc")
    ceiling = AffordableLossCeiling(db=db, account_id="paper_acc")
    engine = DeterministicRiskEngine(ledger=ledger, ceiling=ceiling)
    emerg = EmergencyStopSystem(db=db, account_id="paper_acc")
    sm = ExecutionStateMachine(db=db, ledger=ledger, risk_engine=engine, emergency_system=emerg)
    trader = PaperTradingEngine(db=db, ledger=ledger, state_machine=sm)

    # Propose buy order at ₹50,000 (stop loss ₹47,600 -> ~5% downside, within ₹10 (2%) risk limit)
    prop = sm.propose_order(
        market_pair="I-BTC_INR", side="BUY", current_price=50000.0, proposed_stop_loss=47600.0,
        bid_price=49990.0, ask_price=50010.0,
    )
    assert prop["success"] is True
    order_id = prop["order_id"]

    # 1. Market candle does NOT reach buy limit: low is 51,000 (higher than 50,000)
    high_candle = {"open": 52000.0, "high": 53000.0, "low": 51000.0, "close": 52500.0, "volume": 10.0}
    res_unfilled = trader.execute_paper_order(order_id, high_candle)
    assert res_unfilled["success"] is False
    assert res_unfilled["status"] == "UNFILLED"

    # 2. Market candle DOES reach buy limit: low is 49,500
    reachable_candle = {"open": 50500.0, "high": 51000.0, "low": 49500.0, "close": 49800.0, "volume": 10.0}
    res_filled = trader.execute_paper_order(order_id, reachable_candle)
    assert res_filled["success"] is True
    assert res_filled["status"] == "FILLED"
    assert res_filled["fill_price"] >= reachable_candle["low"]
    assert res_filled["fill_price"] <= reachable_candle["high"]

    # Verify order is marked RECONCILED and fill is marked is_paper=1
    with db.session() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,))
        assert cur.fetchone()["status"] == OrderState.RECONCILED.value

        cur.execute("SELECT is_paper, price FROM fills WHERE order_id = ?", (order_id,))
        fill = cur.fetchone()
        assert fill["is_paper"] == 1

def test_divergence_monitor(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "div_test.db")
    monitor = DivergenceMonitor(db=db)

    # Under-sampled
    assert monitor.check_divergence(0.60, 1.8, [{"net_proceeds_inr": 100, "cost_basis_inr": 90}])["status"] == "INSUFFICIENT_SAMPLE"

    # Concordant paper fills (win rate ~ 60%)
    concordant_fills = [
        {"net_proceeds_inr": 110, "cost_basis_inr": 100},
        {"net_proceeds_inr": 110, "cost_basis_inr": 100},
        {"net_proceeds_inr": 110, "cost_basis_inr": 100},
        {"net_proceeds_inr": 90, "cost_basis_inr": 100},
        {"net_proceeds_inr": 90, "cost_basis_inr": 100},
    ]
    res1 = monitor.check_divergence(backtest_win_rate=0.62, backtest_profit_factor=1.5, paper_fills=concordant_fills)
    assert res1["is_divergence_detected"] is False
    assert res1["status"] == "CONCORDANT"

    # Divergent paper fills: 5 losses, 0 wins (win rate 0% vs 65% backtest)
    divergent_fills = [{"net_proceeds_inr": 80, "cost_basis_inr": 100} for _ in range(5)]
    res2 = monitor.check_divergence(backtest_win_rate=0.65, backtest_profit_factor=1.8, paper_fills=divergent_fills)
    assert res2["is_divergence_detected"] is True
    assert res2["status"] == "DIVERGENCE_ALERT"
