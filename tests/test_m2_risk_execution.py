import pytest
from src.ledger.database import DatabaseManager
from src.ledger.accounting import PortfolioLedger
from src.risk.loss_ceiling import AffordableLossCeiling
from src.risk.risk_engine import DeterministicRiskEngine, RiskParameters
from src.risk.emergency import EmergencyStopSystem
from src.execution.state_machine import ExecutionStateMachine, OrderState
from src.execution.live_guard import LiveExecutionGuard

def test_risk_engine_enforces_two_percent_risk(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "test.db")
    ledger = PortfolioLedger(db=db, account_id="acc1")
    ceiling = AffordableLossCeiling(db=db, account_id="acc1")
    engine = DeterministicRiskEngine(ledger=ledger, ceiling=ceiling)

    # Initial balance is ₹500. 2% risk limit = ₹10.
    # Current price = ₹1000, stop loss = ₹900 (10% downside).
    # Desired notional = 10 / (0.10 + 0.003) ~ ₹97.08
    # With min order ₹100, notional is ₹100, implied risk ~ ₹10.3 (within 15% tolerance).
    res = engine.evaluate_order_proposal(
        market_pair="I-BTC_INR",
        side="BUY",
        current_price=1000.0,
        proposed_stop_loss=900.0,
        bid_price=999.0,
        ask_price=1001.0,
    )
    assert res["approved"] is True
    assert res["planned_risk_inr"] <= 12.0 # Enforces ~2% planned loss
    assert res["quantity"] > 0

def test_risk_engine_rejects_excessive_spread(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "test.db")
    ledger = PortfolioLedger(db=db, account_id="acc2")
    ceiling = AffordableLossCeiling(db=db, account_id="acc2")
    engine = DeterministicRiskEngine(ledger=ledger, ceiling=ceiling)

    # Spread = (1030 - 1000)/1030 = 2.9% > 1.5% max
    res = engine.evaluate_order_proposal(
        market_pair="I-BTC_INR",
        side="BUY",
        current_price=1000.0,
        proposed_stop_loss=950.0,
        bid_price=1000.0,
        ask_price=1030.0,
    )
    assert res["approved"] is False
    assert "Spread" in res["rejection_reason"]

def test_loss_ceiling_halts_trading(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "test.db")
    ledger = PortfolioLedger(db=db, account_id="acc3")
    ceiling = AffordableLossCeiling(db=db, account_id="acc3", ceiling_limit_inr=250.0)

    # Simulate ₹260 cumulative loss
    with db.session() as conn:
        conn.execute("UPDATE accounts SET cumulative_realized_loss = 260.0 WHERE account_id = 'acc3'")

    status = ceiling.evaluate_ceiling_status()
    assert status["is_breached"] is True
    assert status["is_trading_allowed"] is False

    engine = DeterministicRiskEngine(ledger=ledger, ceiling=ceiling)
    res = engine.evaluate_order_proposal(
        market_pair="I-BTC_INR", side="BUY", current_price=1000.0, proposed_stop_loss=950.0
    )
    assert res["approved"] is False
    assert "Affordable loss ceiling breached" in res["rejection_reason"]

def test_execution_state_machine_happy_path(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "test.db")
    ledger = PortfolioLedger(db=db, account_id="acc4")
    ceiling = AffordableLossCeiling(db=db, account_id="acc4")
    engine = DeterministicRiskEngine(ledger=ledger, ceiling=ceiling)
    emerg = EmergencyStopSystem(db=db, account_id="acc4")
    sm = ExecutionStateMachine(db=db, ledger=ledger, risk_engine=engine, emergency_system=emerg)

    # Propose order
    prop = sm.propose_order(
        market_pair="I-BTC_INR", side="BUY", current_price=50000.0, proposed_stop_loss=45000.0,
        bid_price=49990.0, ask_price=50010.0, idempotency_key="key-1"
    )
    assert prop["success"] is True
    assert prop["state"] == OrderState.AUTHORIZED
    order_id = prop["order_id"]

    # Prevent duplicate
    dup = sm.propose_order(
        market_pair="I-BTC_INR", side="BUY", current_price=50000.0, proposed_stop_loss=45000.0,
        idempotency_key="key-1"
    )
    assert dup["success"] is False
    assert "Duplicate order" in dup["reason"]

    # Mark submitted
    assert sm.mark_submitted(order_id) is True

    # Mark filled
    fill = sm.mark_filled(order_id, price=50000.0, quantity=prop["quantity"])
    assert "fill_id" in fill

    # Verify terminal status in DB
    with db.session() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,))
        assert cur.fetchone()["status"] == OrderState.RECONCILED.value

def test_live_guard_rejects_unauthorized_activation():
    guard = LiveExecutionGuard()
    assert guard.is_live_trading_enabled() is False

    # Missing gates
    res1 = guard.authorize_live_trading(
        token="AUTHORIZED_LIVE_USER_CONFIRMED_2026",
        gates_checklist={"m1_data_integrity_passed": True},
    )
    assert res1["success"] is False
    assert guard.is_live_trading_enabled() is False

    # Bad token
    all_gates = {
        "m1_data_integrity_passed": True,
        "m2_risk_engine_verified": True,
        "m3_baseline_evaluation_passed": True,
        "m4_meta_learning_validated": True,
        "m5_paper_trading_divergence_cleared": True,
        "loss_ceiling_active": True,
        "no_open_emergency_stop": True,
    }
    res2 = guard.authorize_live_trading(token="WRONG_TOKEN", gates_checklist=all_gates)
    assert res2["success"] is False
    assert guard.is_live_trading_enabled() is False

    # Valid authorization
    res3 = guard.authorize_live_trading(token="AUTHORIZED_LIVE_USER_CONFIRMED_2026", gates_checklist=all_gates)
    assert res3["success"] is True
    assert guard.is_live_trading_enabled() is True
