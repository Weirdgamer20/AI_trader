from __future__ import annotations

from typing import Dict, Any, List
from ..ledger.accounting import PortfolioLedger
from ..risk.loss_ceiling import AffordableLossCeiling
from ..risk.emergency import EmergencyStopSystem
from ..execution.live_guard import LiveExecutionGuard

class TerminalDashboard:
    """
    Renders terminal-based trading terminal and status summary.
    Emphasizes actual metrics, balance vs ₹100k target, and strict risk limits.
    """
    def __init__(
        self,
        ledger: PortfolioLedger,
        ceiling: AffordableLossCeiling,
        emergency: EmergencyStopSystem,
        live_guard: LiveExecutionGuard,
    ):
        self.ledger = ledger
        self.ceiling = ceiling
        self.emergency = emergency
        self.live_guard = live_guard

    def render_summary(
        self,
        scanner_results: List[Dict[str, Any]] | None = None,
        champion_model_info: Dict[str, Any] | None = None,
    ) -> str:
        acc = self.ledger.get_account_summary()
        ceil = self.ceiling.evaluate_ceiling_status()
        is_safe = self.emergency.is_safe_mode_active()
        is_live = self.live_guard.is_live_trading_enabled()

        lines = []
        lines.append("=" * 80)
        lines.append("  COINDCX SPOT AI TRADING PLATFORM — TERMINAL DASHBOARD (M6)")
        lines.append("=" * 80)

        # Operating Mode
        mode_str = "LIVE REAL-MONEY (AUTHORIZED)" if is_live else "PAPER TRADING (LIVE GUARD ACTIVE)"
        status_str = "🛑 SAFE MODE / HALTED" if is_safe or ceil["is_breached"] else "🟢 NORMAL OPERATION"
        lines.append(f"  Execution Mode : {mode_str}")
        lines.append(f"  System Status  : {status_str}")
        lines.append("-" * 80)

        # Capital & Progress to Target
        balance = acc["total_equity_inr"]
        target = acc["target_balance_inr"]
        pct_target = acc["progress_to_target_pct"]
        lines.append("  CAPITAL & TARGET TRACKING:")
        lines.append(f"    Starting Capital : ₹{acc['starting_balance_inr']:.2f}")
        lines.append(f"    Current Equity   : ₹{balance:.2f} (Available: ₹{acc['available_balance_inr']:.2f}, Locked: ₹{acc['locked_balance_inr']:.2f})")
        lines.append(f"    Target Balance   : ₹{target:,.2f}")
        lines.append(f"    Progress to Goal : {pct_target:.3f}% [{'#' * int(min(20, pct_target / 5))}{'.' * (20 - int(min(20, pct_target / 5)))}]")
        lines.append("-" * 80)

        # Risk Controls
        lines.append("  RISK CONTROLS (DETERMINISTIC):")
        lines.append(f"    Max Planned Loss/Trade : 2.0% of Capital (Current: ₹{balance * 0.02:.2f})")
        lines.append(f"    Affordable Loss Ceiling : ₹{ceil['ceiling_limit_inr']:.2f}")
        lines.append(f"    Cumulative Loss         : ₹{ceil['total_measured_loss_inr']:.2f}")
        lines.append(f"    Remaining Loss Budget   : ₹{ceil['remaining_budget_inr']:.2f}")
        lines.append(f"    Ceiling Breach State    : {'BREACHED / HALTED' if ceil['is_breached'] else 'OK'}")
        lines.append("-" * 80)

        # Indian Tax & Ledger
        lines.append("  PORTFOLIO ACCOUNTING & INDIAN TAX (SECTION 115BBH & 194S):")
        lines.append(f"    Total Taxable VDA Gain  : ₹{acc['total_taxable_vda_gain_inr']:.2f}")
        lines.append(f"    Est. VDA Tax (30% flat) : ₹{acc['estimated_tax_liability_inr']:.2f}")
        lines.append(f"    TDS Withheld (1% u/s 194S): ₹{acc['tds_withheld_inr']:.2f}")
        lines.append(f"    Open Positions Count    : {acc['open_positions_count']}")
        lines.append("-" * 80)

        # Scanner Table (Top 5 if available)
        if scanner_results:
            lines.append("  LIVE SPOT SCANNER (TOP QUALIFIED CANDIDATES):")
            lines.append(f"    {'PAIR':<12} {'PRICE (INR)':<14} {'24H %':<8} {'SPREAD':<8} {'DOWNSIDE':<10} {'STATUS':<12} {'TAGS'}")
            lines.append(f"    {'-'*12} {'-'*14} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*15}")
            for item in scanner_results[:6]:
                tags_str = ",".join(item["strategy_tags"][:2])
                lines.append(
                    f"    {item['market_pair']:<12} ₹{item['price']:<13.2f} {item['change_24h_pct']:<+7.2f}% "
                    f"{item['spread_pct']*100:<7.2f}% {item['downside_risk_pct']*100:<9.2f}% {item['eligibility_status']:<12} {tags_str}"
                )
            lines.append("-" * 80)

        # Model Champion
        if champion_model_info:
            lines.append("  ACTIVE AI CHAMPION MODEL & META-LEARNER:")
            lines.append(f"    Model ID      : {champion_model_info.get('model_id', 'N/A')}")
            lines.append(f"    Algorithm     : {champion_model_info.get('algorithm', 'N/A')}")
            lines.append(f"    Val Accuracy  : {champion_model_info.get('val_accuracy', 0.0)*100:.2f}%")
            lines.append(f"    Val MAE (INR) : ₹{champion_model_info.get('val_mae', 0.0):.2f}")
            lines.append("-" * 80)

        lines.append("=" * 80)
        return "\n".join(lines)
