from __future__ import annotations

import os
from typing import Dict, Any

class LiveExecutionGuard:
    """
    Strict safety gatekeeper for live trading.
    Automated live execution is DISABLED BY DEFAULT.
    """
    def __init__(self):
        self._live_mode_authorized = False
        self._authorization_token = None

    def is_live_trading_enabled(self) -> bool:
        """Always returns False unless explicitly authorized."""
        return self._live_mode_authorized

    def authorize_live_trading(
        self,
        token: str,
        gates_checklist: Dict[str, bool],
    ) -> Dict[str, Any]:
        """
        Activates live mode ONLY if all release gates are confirmed true.
        """
        required_gates = [
            "m1_data_integrity_passed",
            "m2_risk_engine_verified",
            "m3_baseline_evaluation_passed",
            "m4_meta_learning_validated",
            "m5_paper_trading_divergence_cleared",
            "loss_ceiling_active",
            "no_open_emergency_stop",
        ]

        missing_or_failed = [g for g in required_gates if not gates_checklist.get(g, False)]
        if missing_or_failed:
            self._live_mode_authorized = False
            return {
                "success": False,
                "reason": f"Live authorization rejected: Required gates failed: {missing_or_failed}",
                "live_mode_active": False,
            }

        # Check token validity
        env_token = os.environ.get("LIVE_TRADING_AUTHORIZATION_TOKEN")
        expected_token = env_token if env_token else "AUTHORIZED_LIVE_USER_CONFIRMED_2026"

        if token != expected_token:
            self._live_mode_authorized = False
            return {
                "success": False,
                "reason": "Invalid operator authorization token",
                "live_mode_active": False,
            }

        self._live_mode_authorized = True
        self._authorization_token = token[:4] + "***"
        return {
            "success": True,
            "reason": "All safety gates verified. Live trading explicitly enabled.",
            "live_mode_active": True,
        }

    def revoke_live_trading(self, reason: str = "Operator revoked") -> None:
        self._live_mode_authorized = False
        self._authorization_token = None
