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
    ) -> Dict[str, Any]:
        """
        Compares paper fills against expected backtest baseline.
        """
        if len(paper_fills) < 5:
            return {
                "status": "INSUFFICIENT_SAMPLE",
                "sample_count": len(paper_fills),
                "is_divergence_detected": False,
                "notes": "Need at least 5 paper fills to evaluate divergence.",
            }

        # Calculate realized win rate from paper fills
        wins = [f for f in paper_fills if f.get("net_proceeds_inr", 0) > f.get("cost_basis_inr", 0)]
        paper_win_rate = len(wins) / len(paper_fills)

        win_rate_gap = backtest_win_rate - paper_win_rate
        is_divergent = win_rate_gap > 0.15 # Paper win rate >15% lower than backtest

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
                            "backtest_win_rate": backtest_win_rate,
                            "paper_win_rate": paper_win_rate,
                            "gap": win_rate_gap,
                            "sample_size": len(paper_fills),
                        }),
                        now_iso,
                    ),
                )

        return {
            "status": "DIVERGENCE_ALERT" if is_divergent else "CONCORDANT",
            "is_divergence_detected": is_divergent,
            "backtest_win_rate": round(backtest_win_rate, 4),
            "paper_win_rate": round(paper_win_rate, 4),
            "win_rate_gap": round(win_rate_gap, 4),
            "sample_size": len(paper_fills),
        }
