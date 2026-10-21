#!/usr/bin/env python3
"""
CoinDCX Spot AI Trading Platform — Master Control CLI (M1–M8)
"""
import sys
import argparse
from src.ledger.database import DatabaseManager
from src.ledger.accounting import PortfolioLedger
from src.risk.loss_ceiling import AffordableLossCeiling
from src.risk.emergency import EmergencyStopSystem
from src.execution.live_guard import LiveExecutionGuard
from src.scanner.coin_scanner import DynamicCoinScanner
from src.ui.terminal_dashboard import TerminalDashboard
from src.meta_learning.experiment_registry import ExperimentRegistry

def main():
    parser = argparse.ArgumentParser(description="CoinDCX Spot AI Trading Platform")
    parser.add_argument("--status", action="store_true", help="Display full terminal dashboard status")
    parser.add_argument("--scan", action="store_true", help="Scan CoinDCX Spot pairs (up to 50 candidates)")
    parser.add_argument("--emergency-stop", action="store_true", help="Trigger emergency stop circuit breaker")
    parser.add_argument("--clear-stop", type=str, metavar="TOKEN", help="Clear emergency stop with authorization token")
    args = parser.parse_args()

    db = DatabaseManager()
    ledger = PortfolioLedger(db=db)
    ceiling = AffordableLossCeiling(db=db)
    emergency = EmergencyStopSystem(db=db)
    live_guard = LiveExecutionGuard()
    registry = ExperimentRegistry(db=db)

    if args.emergency_stop:
        emergency.trigger_emergency_stop("Triggered via CLI operator command")
        print("🛑 EMERGENCY STOP TRIGGERED. All trading halted and safe mode active.")
        return

    if args.clear_stop:
        if ceiling.manual_reset_with_authorization(args.clear_stop) and emergency.clear_emergency_stop(args.clear_stop):
            print("🟢 EMERGENCY STOP CLEARED and loss ceiling reset.")
        else:
            print("❌ Invalid authorization token.")
        return

    scanner = DynamicCoinScanner(max_scan_candidates=50)
    scanned_results = []
    if args.scan or args.status or len(sys.argv) == 1:
        print("Scanning active CoinDCX Spot markets...")
        scanned_results = scanner.scan_markets()

    dashboard = TerminalDashboard(
        ledger=ledger,
        ceiling=ceiling,
        emergency=emergency,
        live_guard=live_guard,
    )

    print(dashboard.render_summary(
        scanner_results=scanned_results,
        champion_model_info=registry.active_champion,
    ))

if __name__ == "__main__":
    main()
