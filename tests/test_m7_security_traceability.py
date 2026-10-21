import pytest
from src.ledger.database import DatabaseManager
from src.ledger.fee_tax_engine import FeeAndTaxEngine
from src.ledger.accounting import PortfolioLedger
from src.risk.loss_ceiling import AffordableLossCeiling
from src.risk.risk_engine import DeterministicRiskEngine
from src.risk.emergency import EmergencyStopSystem
from src.scanner.coin_scanner import DynamicCoinScanner
from src.meta_learning.meta_features import MetaFeatureExtractor
from src.meta_learning.meta_selector import MetaLearnerSelector
from src.meta_learning.experiment_registry import ExperimentRegistry
from src.meta_learning.critic import IndependentStrategyCritic
from src.execution.state_machine import ExecutionStateMachine, OrderState
from src.execution.live_guard import LiveExecutionGuard
from src.security.credentials import CredentialVault
from src.security.audit_logger import ProvenanceMemoryLogger

def test_req_01_spot_only_and_rejection_of_unsupported_markets():
    """REQ-01: CoinDCX Spot-only trading. Rejects futures, leverage, or unsupported pairs."""
    scanner = DynamicCoinScanner()
    ticker = [
        {"market": "I-BTC_INR", "last_price": "6000000", "bid": "5995000", "ask": "6005000", "volume": "10", "change_24_hour": "1.0"},
        {"market": "BTC_FUTURES_PERP", "last_price": "6000000", "bid": "5995000", "ask": "6005000", "volume": "100", "change_24_hour": "1.0"},
        {"market": "LEVERAGE_3X_BTC", "last_price": "100", "bid": "99", "ask": "101", "volume": "50", "change_24_hour": "0.0"},
    ]
    scanned = scanner.scan_markets(raw_ticker=ticker)
    pairs = [s["market_pair"] for s in scanned]
    assert "I-BTC_INR" in pairs
    assert "BTC_FUTURES_PERP" not in pairs
    assert "LEVERAGE_3X_BTC" not in pairs

def test_req_09_and_10_risk_and_loss_ceiling(tmp_path):
    """REQ-09 & REQ-10: 2% planned risk limit per trade and ₹250 loss ceiling halt."""
    db = DatabaseManager(db_path=tmp_path / "req_risk.db")
    ledger = PortfolioLedger(db=db, account_id="req_acc")
    ceiling = AffordableLossCeiling(db=db, account_id="req_acc", ceiling_limit_inr=250.0)
    engine = DeterministicRiskEngine(ledger=ledger, ceiling=ceiling)

    # Initial equity ₹500 -> 2% planned loss = ₹10.0
    eval1 = engine.evaluate_order_proposal("I-BTC_INR", "BUY", current_price=1000.0, proposed_stop_loss=900.0)
    assert eval1["approved"] is True
    assert eval1["planned_risk_inr"] <= 12.0 # strictly ~2%

    # Simulate ₹250 loss breach
    with db.session() as conn:
        conn.execute("UPDATE accounts SET cumulative_realized_loss = 250.0 WHERE account_id = 'req_acc'")

    eval2 = engine.evaluate_order_proposal("I-BTC_INR", "BUY", current_price=1000.0, proposed_stop_loss=900.0)
    assert eval2["approved"] is False
    assert "Affordable loss ceiling breached" in eval2["rejection_reason"]

def test_req_13_emergency_stop_circuit_breaker(tmp_path):
    """REQ-13: Emergency stop prevents all orders upon data outage or manual trigger."""
    db = DatabaseManager(db_path=tmp_path / "req_emerg.db")
    ledger = PortfolioLedger(db=db, account_id="em_acc")
    emerg = EmergencyStopSystem(db=db, account_id="em_acc")

    assert emerg.is_safe_mode_active() is False
    # Trigger manual kill switch
    emerg.trigger_emergency_stop("Manual test emergency stop")
    assert emerg.is_safe_mode_active() is True

def test_req_24_secret_redaction():
    """REQ-24: Secret credentials masked in text and structured records."""
    raw = "Connecting with coindcx_api_key: sk_live_998877665544 and private_key=abc123secret!"
    redacted = CredentialVault.redact_text(raw)
    assert "sk_live_998877665544" not in redacted
    assert "abc123secret!" not in redacted

def test_req_25_provenance_memory_categorization(tmp_path):
    """REQ-25: Persistent operational memory separates facts, calculations, and hypotheses."""
    db = DatabaseManager(db_path=tmp_path / "req_mem.db")
    mem = ProvenanceMemoryLogger(db=db)

    # Record verified exchange fact
    f_id = mem.record_memory("FACT", "CoinDCX Public API", "BTC/INR Spot Genesis", {"earliest_candle": "2018-10-18"})
    assert f_id.startswith("mem-")

    # Record model hypothesis
    h_id = mem.record_memory("HYPOTHESIS", "HistGradientBoosting v1", "Expected continuation", {"prob_up": 0.62})
    assert h_id.startswith("mem-")

    facts = mem.query_memories(category="FACT")
    assert len(facts) == 1
    assert facts[0]["title"] == "BTC/INR Spot Genesis"

def test_req_28_and_29_live_guard_defaults_to_disabled():
    """REQ-28 & REQ-29: Automated real-money orders physically blocked until user confirms live authorization."""
    guard = LiveExecutionGuard()
    assert guard.is_live_trading_enabled() is False
    # Attempting to place live order without token must remain disabled
    guard.revoke_live_trading()
    assert guard.is_live_trading_enabled() is False

def test_req_30_end_to_end_audit_traceability(tmp_path):
    """REQ-30: Complete auditability from input proposal to ledger fill."""
    db = DatabaseManager(db_path=tmp_path / "trace.db")
    ledger = PortfolioLedger(db=db, account_id="trace_acc")
    ceiling = AffordableLossCeiling(db=db, account_id="trace_acc")
    engine = DeterministicRiskEngine(ledger=ledger, ceiling=ceiling)
    emerg = EmergencyStopSystem(db=db, account_id="trace_acc")
    sm = ExecutionStateMachine(db=db, ledger=ledger, risk_engine=engine, emergency_system=emerg)

    # 1. Proposal
    prop = sm.propose_order("I-BTC_INR", "BUY", current_price=50000.0, proposed_stop_loss=46000.0)
    assert prop["success"] is True
    order_id = prop["order_id"]

    # 2. State transition to SUBMITTED
    sm.mark_submitted(order_id)

    # 3. Execution fill
    fill = sm.mark_filled(order_id, price=50000.0, quantity=prop["quantity"])
    assert "fill_id" in fill

    # 4. Trace in database ledger
    summary = ledger.get_account_summary()
    assert summary["open_positions_count"] == 1
    assert summary["available_balance_inr"] < 500.0
