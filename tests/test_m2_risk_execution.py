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

    # Initial balance is ₹500. Strict 2% risk limit = ₹10.00.
    # Current price = ₹1000, stop loss = ₹900 (10% downside).
    # Desired notional = 10 / (0.10 + 0.003) ~ ₹97.08
    # With min order ₹100, if notional ₹100 produces risk > ₹10.00, it must be rejected!
    # Here with current_price=1000, stop_loss=950 (5% downside):
    # notional = 10 / (0.05 + 0.003) = ₹188.67 >= 100, risk is strictly <= 10.0
    res = engine.evaluate_order_proposal(
        market_pair="I-BTC_INR",
        side="BUY",
        current_price=1000.0,
        proposed_stop_loss=950.0,
        bid_price=999.0,
        ask_price=1001.0,
    )
    assert res["approved"] is True
    assert res["planned_risk_inr"] <= 10.0001 # Strict 2% hard risk limit
    assert res["quantity"] > 0

    # P0-2: Verify rejection if planned risk exceeds 2% cap
    res_excess = engine.evaluate_order_proposal(
        market_pair="I-BTC_INR",
        side="BUY",
        current_price=1000.0,
        proposed_stop_loss=800.0, # 20% downside -> min order 100 produces risk 20.3 > 10.0
        bid_price=999.0,
        ask_price=1001.0,
    )
    assert res_excess["approved"] is False
    assert "strictly exceeding the 2% hard risk limit" in res_excess["rejection_reason"]

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

    # Propose order (price ₹50,000, stop loss ₹47,600 = ~5% downside, fits within ₹10 (2%) risk cap)
    prop = sm.propose_order(
        market_pair="I-BTC_INR", side="BUY", current_price=50000.0, proposed_stop_loss=47600.0,
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

    # Missing environment variable / unconfigured secret
    res1 = guard.authorize_live_trading(
        token="SOME_TOKEN",
        gates_checklist={"m1_data_integrity_passed": True},
    )
    assert res1["success"] is False
    assert guard.is_live_trading_enabled() is False

    # All gates True but missing secret or execution adapter
    all_gates = {
        "m1_data_integrity_passed": True,
        "m2_risk_engine_verified": True,
        "m3_baseline_evaluation_passed": True,
        "m4_meta_learning_validated": True,
        "m5_paper_trading_divergence_cleared": True,
        "loss_ceiling_active": True,
        "no_open_emergency_stop": True,
    }
    # P0-1: With no LIVE_TRADING_AUTHORIZATION_TOKEN configured, fail closed
    res2 = guard.authorize_live_trading(token="AUTHORIZED_LIVE_USER_CONFIRMED_2026", gates_checklist=all_gates)
    assert res2["success"] is False
    assert guard.is_live_trading_enabled() is False

    # Even if environment variable is set, live trading remains disabled without verified qualification evidence & live execution adapter
    import os
    os.environ["LIVE_TRADING_AUTHORIZATION_TOKEN"] = "SUPER_SECURE_TOKEN_2026"
    res3 = guard.authorize_live_trading(token="SUPER_SECURE_TOKEN_2026", gates_checklist=all_gates)
    assert res3["success"] is False
    assert ("Live exchange adapter not connected" in res3["reason"] or "qualification gates" in res3["reason"])
    assert guard.is_live_trading_enabled() is False
    del os.environ["LIVE_TRADING_AUTHORIZATION_TOKEN"]
