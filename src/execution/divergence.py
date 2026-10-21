from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from ..ledger.database import DatabaseManager

class DivergenceMonitor:
    """
    Monitors deployment divergence between backtested model metrics
    and live paper-trading outcomes.
    Triggers diagnostic alerts if paper results degrade beyond tolerance.
    """
    def __init__(
        self,
        db: DatabaseManager,
        max_acceptable_slippage_pct: float = 0.005, # Max 0.5% average slippage
        max_drawdown_divergence_pct: float = 0.15,   # Max 15% worse drawdown than backtest
    ):
        self.db = db
        self.max_slippage = max_acceptable_slippage_pct
        self.max_dd_div = max_drawdown_divergence_pct

    def check_divergence(
        self,
        backtest_win_rate: float,
        backtest_profit_factor: float,
        paper_fills: List[Dict[str, Any]],
        backtest_max_drawdown_pct: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Compares paper fills against expected backtest baseline.
        Evaluates win rate, profit factor, slippage, and drawdown divergence.
        """
        if len(paper_fills) < 5:
            return {
                "status": "INSUFFICIENT_SAMPLE",
                "sample_count": len(paper_fills),
                "is_divergence_detected": False,
                "notes": "Need at least 5 paper fills to evaluate divergence.",
            }

        # Calculate realized win rate & profit factor from paper fills
        gross_profit = 0.0
        gross_loss = 0.0
        wins = 0
        total_slippage = 0.0
        equity_curve = [0.0]
        cumulative_pnl = 0.0

        for f in paper_fills:
            proceeds = float(f.get("net_proceeds_inr", 0.0))
            cost = float(f.get("cost_basis_inr", 0.0))
            pnl = proceeds - cost
            cumulative_pnl += pnl
            equity_curve.append(cumulative_pnl)

            if pnl > 0:
                wins += 1
                gross_profit += pnl
            elif pnl < 0:
                gross_loss += abs(pnl)

            total_slippage += float(f.get("slippage_pct", f.get("slippage_incurred_pct", 0.0)))

        paper_win_rate = wins / len(paper_fills)
        paper_profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)
        avg_slippage = total_slippage / len(paper_fills)

        # Calculate paper max drawdown
        peak = 0.0
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq)
            if dd > max_dd:
                max_dd = dd
        # Normalize drawdown against assumed baseline initial capital ₹500
        paper_dd_pct = max_dd / 500.0

        win_rate_gap = backtest_win_rate - paper_win_rate
        profit_factor_gap = backtest_profit_factor - paper_profit_factor
        dd_gap = paper_dd_pct - backtest_max_drawdown_pct

        reasons = []
        if win_rate_gap > 0.15:
            reasons.append(f"Win rate gap {win_rate_gap:.1%} > 15%")
        if profit_factor_gap > 0.5:
            reasons.append(f"Profit factor degraded from {backtest_profit_factor:.2f} to {paper_profit_factor:.2f}")
        if avg_slippage > self.max_slippage:
            reasons.append(f"Average slippage {avg_slippage:.2%} exceeds max tolerance {self.max_slippage:.2%}")
        if dd_gap > self.max_dd_div:
            reasons.append(f"Drawdown {paper_dd_pct:.1%} exceeds backtest {backtest_max_drawdown_pct:.1%} by > {self.max_dd_div:.1%}")

        is_divergent = len(reasons) > 0

        if is_divergent:
            now_iso = datetime.now(timezone.utc).isoformat()
            with self.db.session() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO audit_events (event_id, event_type, severity, message, details_json, timestamp)
                    VALUES (?, 'RECONCILIATION', 'WARNING', 'Backtest to paper trading divergence detected', ?, ?)
                    """,
                    (
                        f"div-{int(datetime.now().timestamp())}",
                        json.dumps({
                            "reasons": reasons,
                            "backtest_win_rate": backtest_win_rate,
                            "paper_win_rate": paper_win_rate,
                            "backtest_profit_factor": backtest_profit_factor,
                            "paper_profit_factor": paper_profit_factor,
                            "avg_slippage": avg_slippage,
                            "sample_size": len(paper_fills),
                        }),
                        now_iso,
                    ),
                )

        return {
            "status": "DIVERGENCE_ALERT" if is_divergent else "CONCORDANT",
            "is_divergence_detected": is_divergent,
            "reasons": reasons,
            "backtest_win_rate": round(backtest_win_rate, 4),
            "paper_win_rate": round(paper_win_rate, 4),
            "win_rate_gap": round(win_rate_gap, 4),
            "backtest_profit_factor": round(backtest_profit_factor, 4),
            "paper_profit_factor": round(paper_profit_factor, 4),
            "average_slippage": round(avg_slippage, 5),
            "paper_max_drawdown_pct": round(paper_dd_pct, 4),
            "sample_size": len(paper_fills),
        }
