from __future__ import annotations

import hmac
import os
import json
from pathlib import Path
from typing import Dict, Any

class LiveExecutionGuard:
    """
    Strict safety gatekeeper for live trading.
    Automated real-money execution is PHYSICALLY DISABLED BY DEFAULT.
    
    Security Principles:
    - Never uses hardcoded fallback tokens.
    - Fails closed if the operator secret is not configured in the environment.
    - Uses constant-time token comparison (hmac.compare_digest).
    - Requires persistent qualification evidence, not ephemeral dashboard checkboxes.
    - Live execution remains blocked because no live broker execution adapter exists.
    """
    def __init__(self, evidence_dir: Path | str | None = None):
        self.evidence_dir = Path(evidence_dir or "data/evidence")
        self._live_mode_authorized = False
        self._authorized_operator = None

    def is_live_trading_enabled(self) -> bool:
        """Always fails closed unless verified authorization is in place."""
        return self._live_mode_authorized

    def verify_persistent_qualification_gates(self) -> Dict[str, Any]:
        """
        Validates evidence from persisted qualification records.
        Does NOT trust unverified caller flags.
        """
        required_evidence = {
            "data_provenance": "data_provenance_verified.json",
            "risk_engine": "risk_engine_verified.json",
            "model_backtest": "model_holdout_passed.json",
            "paper_trading": "paper_trading_divergence_passed.json",
        }
        
        status = {}
        all_passed = True
        for gate, filename in required_evidence.items():
            filepath = self.evidence_dir / filename
            if filepath.exists():
                try:
                    data = json.loads(filepath.read_text(encoding="utf-8"))
                    passed = bool(data.get("status") == "PASSED" and data.get("verified", False))
                    status[gate] = passed
                    if not passed:
                        all_passed = False
                except Exception:
                    status[gate] = False
                    all_passed = False
            else:
                # Evidence file absent -> gate fails closed
                status[gate] = False
                all_passed = False

        return {"all_passed": all_passed, "gates": status}

    def authorize_live_trading(
        self,
        supplied_token: str | None = None,
        operator_id: str = "operator",
        token: str | None = None,
        gates_checklist: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Attempts to enable live mode.
        Fails closed:
        1. Fails if environment variable LIVE_TRADING_AUTHORIZATION_TOKEN is not set.
        2. Fails if token comparison fails (constant-time).
        3. Fails if persistent qualification evidence files are missing or unverified.
        4. Fails if live execution adapter is absent.
        """
        supplied = supplied_token or token or ""
        expected_token = os.environ.get("LIVE_TRADING_AUTHORIZATION_TOKEN")
        if not expected_token or not supplied:
            self._live_mode_authorized = False
            return {
                "success": False,
                "reason": "Authorization failed: LIVE_TRADING_AUTHORIZATION_TOKEN is not configured in environment or token missing. Fails closed.",
                "live_mode_active": False,
            }

        # Constant-time comparison
        if not hmac.compare_digest(supplied.strip(), expected_token.strip()):
            self._live_mode_authorized = False
            return {
                "success": False,
                "reason": "Authorization failed: Invalid operator token.",
                "live_mode_active": False,
            }

        # Check real persistent qualification evidence
        gate_res = self.verify_persistent_qualification_gates()
        if not gate_res["all_passed"]:
            self._live_mode_authorized = False
            return {
                "success": False,
                "reason": f"Authorization rejected: Persistent qualification gates incomplete: {gate_res['gates']}",
                "live_mode_active": False,
            }

        # Crucial safeguard: Live broker adapter does not exist in codebase
        self._live_mode_authorized = False
        return {
            "success": False,
            "reason": "Live mode blocked: No authenticated CoinDCX live execution adapter implemented. Codebase operates in verified PAPER mode only.",
            "live_mode_active": False,
        }

    def revoke_live_trading(self, reason: str = "Operator revoked") -> None:
        self._live_mode_authorized = False
        self._authorized_operator = None
