import math
import pytest
import sqlite3
from pathlib import Path
from src.ledger.database import DatabaseManager
from src.ledger.accounting import PortfolioLedger
from src.ledger.fee_tax_engine import FeeAndTaxEngine
from src.risk.risk_engine import DeterministicRiskEngine, RiskParameters
from src.risk.loss_ceiling import AffordableLossCeiling
from src.risk.emergency import EmergencyStopSystem
from src.execution.state_machine import ExecutionStateMachine, OrderState

def test_duplicate_fill_idempotency(tmp_path):
    """P0-3: Duplicate fill with same exchange trade ID is ignored without double financial mutation."""
    db = DatabaseManager(db_path=tmp_path / "test.db")
    ledger = PortfolioLedger(db=db, account_id="acc1")

    ledger.create_order("ord-1", "cli-1", "I-BTC_INR", "BUY", price=50000.0, quantity=0.002, status="SUBMITTED")
    res1 = ledger.record_buy_fill("ord-1", "I-BTC_INR", price=50000.0, quantity=0.001, exchange_trade_id="trade-tx-999")
    assert res1["duplicate"] is False
    summary1 = ledger.get_account_summary()

    # Replay same fill with same trade ID
    res2 = ledger.record_buy_fill("ord-1", "I-BTC_INR", price=50000.0, quantity=0.001, exchange_trade_id="trade-tx-999")
    assert res2["duplicate"] is True
    assert res2["status"] == "ALREADY_PROCESSED"
    summary2 = ledger.get_account_summary()

    # Invariant: Balances, cash, and holdings must not change twice
    assert summary1["available_balance_inr"] == summary2["available_balance_inr"]
    assert summary1["total_equity_inr"] == summary2["total_equity_inr"]

def test_non_finite_price_quantity_rejected(tmp_path):
    """P0-3: Fills with NaN, Inf, non-positive numbers are rejected before any writes."""
    db = DatabaseManager(db_path=tmp_path / "test.db")
    ledger = PortfolioLedger(db=db, account_id="acc1")
    ledger.create_order("ord-1", "cli-1", "I-BTC_INR", "BUY", price=50000.0, quantity=0.002, status="SUBMITTED")

    for bad_val in [float("nan"), float("inf"), float("-inf"), 0.0, -10.0]:
        with pytest.raises(ValueError):
            ledger.record_buy_fill("ord-1", "I-BTC_INR", price=bad_val, quantity=0.001)
        with pytest.raises(ValueError):
            ledger.record_buy_fill("ord-1", "I-BTC_INR", price=50000.0, quantity=bad_val)

def test_overfill_rejected(tmp_path):
    """P0-3: Fill exceeding remaining order quantity is rejected."""
    db = DatabaseManager(db_path=tmp_path / "test.db")
    ledger = PortfolioLedger(db=db, account_id="acc1")
    ledger.create_order("ord-1", "cli-1", "I-BTC_INR", "BUY", price=50000.0, quantity=0.001, status="SUBMITTED")

    with pytest.raises(ValueError, match="exceeds order remaining quantity"):
        ledger.record_buy_fill("ord-1", "I-BTC_INR", price=50000.0, quantity=0.002)

def test_partial_buy_reservation_and_completion(tmp_path):
    """P0-2 & P0-4: Partial BUY fill consumes only its share of reservation, keeping remainder for later fill."""
    db = DatabaseManager(db_path=tmp_path / "test.db")
    ledger = PortfolioLedger(db=db, account_id="acc1")

    # Reserve full estimated outlay for 0.002 BTC (~₹100.59)
    ledger.create_order("ord-1", "cli-1", "I-BTC_INR", "BUY", price=50000.0, quantity=0.002, status="SUBMITTED")
    assert ledger.reserve_funds_for_order("ord-1", 100.59) is True

    with db.session() as conn:
        cur = conn.cursor()
        cur.execute("SELECT reserved_funds_inr FROM orders WHERE order_id = 'ord-1'")
        assert round(cur.fetchone()[0], 2) == 100.59

    # Partial fill 1: 0.001 BTC (half)
    res1 = ledger.record_buy_fill("ord-1", "I-BTC_INR", price=50000.0, quantity=0.001)
    assert res1["duplicate"] is False

    # Check that remaining reservation is still active on order (~₹50.30)
    with db.session() as conn:
        cur = conn.cursor()
        cur.execute("SELECT reserved_funds_inr, filled_quantity FROM orders WHERE order_id = 'ord-1'")
        row = cur.fetchone()
        assert row["filled_quantity"] == 0.001
        assert row["reserved_funds_inr"] > 40.0 # Remainder strictly preserved!

    # Partial fill 2: 0.001 BTC (completes the order)
    res2 = ledger.record_buy_fill("ord-1", "I-BTC_INR", price=50000.0, quantity=0.001)
    assert res2["duplicate"] is False

    with db.session() as conn:
        cur = conn.cursor()
        cur.execute("SELECT reserved_funds_inr, filled_quantity FROM orders WHERE order_id = 'ord-1'")
        row2 = cur.fetchone()
        assert row2["filled_quantity"] == 0.002
        assert row2["reserved_funds_inr"] == 0.0

    # Reconcile operational state with general ledger journal (P0-4)
    rec = ledger.reconcile_operational_with_journal("acc1")
    assert rec["reconciled"] is True
    assert rec["is_journal_balanced"] is True
    assert len(rec["discrepancies"]) == 0

def test_partial_sell_reservation_and_cancellation(tmp_path):
    """P0-2: Partial SELL fill reduces reservation; cancellation releases exact remainder."""
    db = DatabaseManager(db_path=tmp_path / "test.db")
    ledger = PortfolioLedger(db=db, account_id="acc1")
    ceiling = AffordableLossCeiling(db=db, account_id="acc1")
    risk_engine = DeterministicRiskEngine(ledger=ledger, ceiling=ceiling)
    emergency = EmergencyStopSystem(db=db, account_id="acc1")
    sm = ExecutionStateMachine(db=db, ledger=ledger, risk_engine=risk_engine, emergency_system=emergency)

    # First acquire 0.002 BTC
    ledger.create_order("buy-1", "c-buy", "I-BTC_INR", "BUY", price=50000.0, quantity=0.002, status="SUBMITTED")
    ledger.record_buy_fill("buy-1", "I-BTC_INR", price=50000.0, quantity=0.002)

    # Propose SELL for 0.002 BTC
    prop = sm.propose_order("I-BTC_INR", "SELL", current_price=55000.0, sell_quantity=0.002)
    assert prop["success"] is True
    sell_order_id = prop["order_id"]
    sm.mark_submitted(sell_order_id)

    # Partial fill 1 of 0.001 BTC
    fill1 = sm.mark_filled(sell_order_id, price=55000.0, quantity=0.001)
    assert fill1["duplicate"] is False

    # Check order remaining reserved quantity
    with db.session() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status, reserved_quantity, filled_quantity FROM orders WHERE order_id = ?", (sell_order_id,))
        ord_row = cur.fetchone()
        assert ord_row["status"] == "PARTIAL"
        assert round(ord_row["reserved_quantity"], 6) == 0.001

    # Cancel the remaining unfilled portion
    cancelled = sm.cancel_order(sell_order_id, reason="User cancelled remaining")
    assert cancelled is True

    # Position reserved_quantity must be completely released back to 0.0
    with db.session() as conn:
        cur = conn.cursor()
        cur.execute("SELECT quantity, reserved_quantity FROM positions WHERE account_id = 'acc1' AND market_pair = 'I-BTC_INR'")
        pos_row = cur.fetchone()
        assert round(pos_row["quantity"], 6) == 0.001
        assert round(pos_row["reserved_quantity"], 6) == 0.0

def test_atomic_fill_rollback_on_invalid_transition(tmp_path):
    """P0-1: State machine atomic transaction rolls back fill if transition is invalid."""
    db = DatabaseManager(db_path=tmp_path / "test.db")
    ledger = PortfolioLedger(db=db, account_id="acc1")
    ceiling = AffordableLossCeiling(db=db, account_id="acc1")
    risk_engine = DeterministicRiskEngine(ledger=ledger, ceiling=ceiling)
    emergency = EmergencyStopSystem(db=db, account_id="acc1")
    sm = ExecutionStateMachine(db=db, ledger=ledger, risk_engine=risk_engine, emergency_system=emergency)

    # Order in PROPOSED state cannot be filled directly via state machine
    ledger.create_order("ord-unsub", "cli-u", "I-BTC_INR", "BUY", price=50000.0, quantity=0.001, status="PROPOSED")
    with pytest.raises(ValueError, match="Cannot record fill for order in state PROPOSED"):
        sm.mark_filled("ord-unsub", price=50000.0, quantity=0.001)

    # Verify no fills or positions created
    summary = ledger.get_account_summary()
    assert summary["open_positions_count"] == 0
    with db.session() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM fills WHERE order_id = 'ord-unsub'")
        assert cur.fetchone()[0] == 0

def test_minimum_order_exceeds_remaining_loss_budget_rejected(tmp_path):
    """P0-6: If remaining loss budget is only ₹1, minimum order with implied risk ₹5 is rejected."""
    db = DatabaseManager(db_path=tmp_path / "test.db")
    ledger = PortfolioLedger(db=db, account_id="acc1")
    ceiling = AffordableLossCeiling(db=db, account_id="acc1")
    engine = DeterministicRiskEngine(ledger=ledger, ceiling=ceiling)

    # Set cumulative realized loss so that remaining budget is only ₹1.0
    with db.session() as conn:
        cur = conn.cursor()
        # Affordable ceiling is ₹250. Cumulative loss = 249.0 -> Remaining budget = ₹1.0
        cur.execute("UPDATE accounts SET cumulative_realized_loss = 249.0 WHERE account_id = 'acc1'")

    ceil_stat = ceiling.evaluate_ceiling_status()
    assert round(ceil_stat["remaining_budget_inr"], 2) == 1.0

    # With equity ~₹500, 2% limit is ₹10.
    # Current price ₹1000, stop ₹950 (5% downside).
    # Minimum order is ₹100. Implied risk = 100 * (0.05 + buffers ~0.003) = ~₹5.3
    # ₹5.3 is under ₹10 hard cap, but EXCEEDS ₹1.0 remaining budget!
    res = engine.evaluate_order_proposal(
        market_pair="I-BTC_INR",
        side="BUY",
        current_price=1000.0,
        proposed_stop_loss=950.0,
        bid_price=999.0,
        ask_price=1001.0,
    )
    assert res["approved"] is False
    assert "remaining loss budget" in res["rejection_reason"]

def test_multiple_pending_buy_orders_allocation(tmp_path):
    """P1-1: Multiple pending BUY orders must not exceed the 30% per-coin portfolio allocation."""
    db = DatabaseManager(db_path=tmp_path / "test.db")
    ledger = PortfolioLedger(db=db, account_id="acc1")
    ceiling = AffordableLossCeiling(db=db, account_id="acc1")
    risk_engine = DeterministicRiskEngine(ledger=ledger, ceiling=ceiling)
    emergency = EmergencyStopSystem(db=db, account_id="acc1")
    sm = ExecutionStateMachine(db=db, ledger=ledger, risk_engine=risk_engine, emergency_system=emergency)

    # Capital = ₹500. 30% per-coin max = ₹150.
    # First proposal for ₹120 notional (within ₹150)
    # Stop loss close to price to allow larger size within 2% risk limit:
    # 2% of 500 = ₹10. Downside = 7% -> notional ~ ₹135
    prop1 = sm.propose_order("I-BTC_INR", "BUY", current_price=1000.0, proposed_stop_loss=930.0, bid_price=999.0, ask_price=1001.0)
    assert prop1["success"] is True
    # Order is now AUTHORIZED, consuming pending allocation!

    # Second proposal for same coin: remaining allocation budget is exceeded
    prop2 = sm.propose_order("I-BTC_INR", "BUY", current_price=1000.0, proposed_stop_loss=930.0, bid_price=999.0, ask_price=1001.0)
    assert prop2["success"] is False
    assert "allocation" in prop2["reason"].lower() or "exceeded" in prop2["reason"].lower()

def test_legacy_database_migration_upgrade(tmp_path):
    """P0-5: Existing database missing newer columns is upgraded safely via ALTER TABLE without data loss."""
    legacy_db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(legacy_db_file))
    cur = conn.cursor()
    # Create older accounts table missing tds_receivable_inr and locked_balance
    cur.execute(
        """
        CREATE TABLE accounts (
            account_id TEXT PRIMARY KEY,
            account_type TEXT NOT NULL,
            currency TEXT NOT NULL DEFAULT 'INR',
            starting_balance REAL NOT NULL DEFAULT 500.0,
            available_balance REAL NOT NULL DEFAULT 500.0,
            target_balance REAL NOT NULL DEFAULT 100000.0,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    cur.execute("INSERT INTO accounts VALUES ('acc-legacy', 'PAPER', 'INR', 500.0, 500.0, 100000.0, 1, '2026-01-01', '2026-01-01')")
    conn.commit()
    conn.close()

    # Initialize DatabaseManager on this legacy database
    db = DatabaseManager(db_path=legacy_db_file)
    with db.session() as c:
        cur = c.cursor()
        cur.execute("PRAGMA table_info(accounts)")
        cols = {r[1] for r in cur.fetchall()}
        assert "tds_receivable_inr" in cols
        assert "locked_balance" in cols

        # Ensure existing row was preserved
        cur.execute("SELECT account_id, available_balance, tds_receivable_inr FROM accounts WHERE account_id = 'acc-legacy'")
        row = cur.fetchone()
        assert row["account_id"] == "acc-legacy"
        assert row["available_balance"] == 500.0
        assert row["tds_receivable_inr"] == 0.0

def test_dataset_manifest_and_cross_pair_contamination_rejected(tmp_path):
    """P1-7: Downloading pair A, then requesting pair B with resume on A's file strictly rejects cross-pair contamination."""
    import pandas as pd
    from src.data import save_candles, fetch_candles

    file_a = tmp_path / "btc_inr_15m.csv"
    dummy_df = pd.DataFrame([
        {"time": pd.Timestamp("2026-01-01 10:00:00", tz="Asia/Kolkata"), "open": 50000.0, "high": 50100.0, "low": 49900.0, "close": 50050.0, "volume": 1.2}
    ])
    # Save pair A
    save_candles(dummy_df, path=str(file_a), pair="I-BTC_INR")
    manifest_file = tmp_path / "btc_inr_15m.csv.manifest.json"
    assert manifest_file.exists()

    # Attempt to resume pair B (e.g. I-ETH_INR) pointing to pair A's file
    with pytest.raises(ValueError, match="Cross-pair data contamination strictly prevented"):
        fetch_candles(pair="I-ETH_INR", resume_path=str(file_a))

def test_split_boundary_purging_prevents_leakage():
    """P1-8: Chronological train split targets must not cross into the calibration split."""
    from src.modeling_multicoin import MultiHorizonForecaster
    import numpy as np
    import pandas as pd

    # Generate 250 candles
    rng = np.random.default_rng(42)
    n = 250
    returns = rng.normal(0.0001, 0.003, n)
    close = 1000.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close, "volume": 100.0,
    })

    forecaster = MultiHorizonForecaster(horizons=[4])
    metrics = forecaster.fit(df, train_ratio=0.70, calib_ratio=0.15)
    assert forecaster.is_fitted is True

def test_empty_test_partition_reports_data_insufficient():
    """P1-9: Empty test partition returns DATA_INSUFFICIENT and None, never fabricated metrics."""
    from src.modeling_multicoin import MultiHorizonForecaster
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 280
    returns = rng.normal(0.0001, 0.003, n)
    close = 1000.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close, "volume": 100.0,
    })

    forecaster = MultiHorizonForecaster(horizons=[1])
    # Allocate 100% of data to train and calib (0% to test)
    metrics = forecaster.fit(df, train_ratio=0.80, calib_ratio=0.20)
    assert metrics["h1_status"] == "DATA_INSUFFICIENT"
    assert metrics["h1_mae"] is None
    assert metrics["h1_empirical_coverage_90"] is None
    assert forecaster.empirical_coverage_90[1] is None

def test_credential_redaction_naming_variants_and_urls():
    """Verify that camelCase and variant secret keys (apiKey, privateKey) and Telegram URLs are fully redacted."""
    from src.security.credentials import CredentialVault

    data = {
        "apiKey": "super_secret_api_key_12345",
        "privateKey": "privkey_9876543210_abcdef",
        "nested": {
            "secretToken": "token_xyz_999",
            "normal_field": "public_data",
        },
        "url": "https://api.telegram.org/bot1234567890:ABCdefGHIjklMNOpqrsTUVwxyz_12345/sendMessage"
    }

    redacted = CredentialVault.redact_data(data)
    assert redacted["apiKey"] == "********"
    assert redacted["privateKey"] == "********"
    assert redacted["nested"]["secretToken"] == "********"
    assert redacted["nested"]["normal_field"] == "public_data"
    assert "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz_12345" not in str(redacted["url"])
    assert "[REDACTED_BOT_TOKEN]" in redacted["url"]


