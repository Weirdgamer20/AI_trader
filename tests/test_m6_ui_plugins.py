import pytest
from src.ledger.database import DatabaseManager
from src.ledger.accounting import PortfolioLedger
from src.risk.loss_ceiling import AffordableLossCeiling
from src.risk.emergency import EmergencyStopSystem
from src.execution.live_guard import LiveExecutionGuard
from src.ui.plugin_system import PluginSandbox
from src.ui.telegram_bot import TelegramNotifier
from src.ui.terminal_dashboard import TerminalDashboard

def test_plugin_sandbox_permissions_and_crash_isolation():
    sandbox = PluginSandbox()

    # 1. Valid plugin registration
    valid_manifest = {
        "plugin_id": "custom-scanner-widget",
        "name": "Custom Scanner Widget",
        "version": "1.0.0",
        "author": "Trader",
        "requested_permissions": ["read_market_scanner", "render_custom_widget"],
    }
    res1 = sandbox.register_plugin(valid_manifest)
    assert res1["success"] is True

    # 2. Rogue plugin requesting forbidden permission: MUST REJECT
    rogue_manifest = {
        "plugin_id": "malicious-key-stealer",
        "name": "Rogue Plugin",
        "version": "1.0.0",
        "author": "Attacker",
        "requested_permissions": ["access_credentials", "submit_live_orders"],
    }
    res2 = sandbox.register_plugin(rogue_manifest)
    assert res2["success"] is False
    assert "Security violation" in res2["reason"]

    # 3. Crash isolation: Crashing plugin must be isolated without killing host
    def crashing_handler():
        raise RuntimeError("Division by zero in third-party widget calculation")

    exec_res = sandbox.execute_plugin_safely("custom-scanner-widget", crashing_handler)
    assert exec_res["success"] is False
    assert "Plugin crashed and was automatically isolated" in exec_res["error"]
    assert sandbox.active_plugins["custom-scanner-widget"] is False

    # 4. Workspace layouts
    sandbox.save_workspace_layout("ProTrader", {"layout": ["scanner", "charts", "critic"], "density": "ultra"})
    retrieved = sandbox.get_workspace_layout("ProTrader")
    assert retrieved["density"] == "ultra"

def test_telegram_notifier_secret_redaction():
    notifier = TelegramNotifier()
    raw_message = "ALERT: Error authenticating with api_key: AIzaSyD9876543210secretToken! Password=SuperSecret123"
    sanitized = notifier.sanitize_message(raw_message)
    assert "AIzaSyD9876543210secretToken" not in sanitized
    assert "SuperSecret123" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized

def test_terminal_dashboard_rendering(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "ui_test.db")
    ledger = PortfolioLedger(db=db, account_id="ui_acc")
    ceiling = AffordableLossCeiling(db=db, account_id="ui_acc")
    emergency = EmergencyStopSystem(db=db, account_id="ui_acc")
    live_guard = LiveExecutionGuard()

    dashboard = TerminalDashboard(ledger=ledger, ceiling=ceiling, emergency=emergency, live_guard=live_guard)

    mock_scanner = [
        {"market_pair": "I-BTC_INR", "price": 6000000.0, "change_24h_pct": 2.5, "spread_pct": 0.002, "downside_risk_pct": 0.03, "eligibility_status": "ELIGIBLE", "strategy_tags": ["TRENDING_BULLISH"]}
    ]
    mock_champ = {"model_id": "champ-v1", "algorithm": "hist_gradient_boosting", "val_accuracy": 0.56, "val_mae": 1200.0}

    output = dashboard.render_summary(mock_scanner, mock_champ)
    assert "COINDCX SPOT AI TRADING PLATFORM" in output
    assert "Progress to Goal" in output
    assert "Starting Capital : ₹500.00" in output
    assert "Target Balance   : ₹100,000.00" in output
    assert "Affordable Loss Ceiling : ₹250.00" in output
    assert "I-BTC_INR" in output
    assert "champ-v1" in output
