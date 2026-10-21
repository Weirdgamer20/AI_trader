from __future__ import annotations

from typing import Dict, Any, List

class IndependentStrategyCritic:
    """
    Independent Critic and Judge:
    Provides objective, independent review of proposed strategy signals.
    Examines model agreement/disagreement, forecast uncertainty width,
    transaction cost feasibility, and out-of-distribution conditions.
    """
    def __init__(
        self,
        max_uncertainty_width_pct: float = 0.045,  # Max 4.5% forecast uncertainty width
        min_expected_return_pct: float = 0.012,     # Minimum 1.2% net expected return after costs
        min_model_confidence: float = 0.54,         # Directional confidence threshold
    ):
        self.max_uncertainty_width_pct = max_uncertainty_width_pct
        self.min_expected_return_pct = min_expected_return_pct
        self.min_model_confidence = min_model_confidence

    def evaluate_proposal(
        self,
        market_pair: str,
        side: str,
        forecast_projection: Dict[str, Any],
        model_agreement_pct: float = 1.0,
        is_out_of_distribution: bool = False,
    ) -> Dict[str, Any]:
        """
        Independently audits a proposed trade action before risk engine check.
        """
        rejection_reasons = []

        direction = forecast_projection.get("direction", "DOWN")
        prob_up = forecast_projection.get("probability_up", 0.5)
        expected_ret = forecast_projection.get("expected_return_pct", 0.0)
        uncertainty_width = forecast_projection.get("uncertainty_width_pct", 0.0)

        # 1. Directional alignment
        if side.upper() == "BUY":
            if direction != "UP" or prob_up < self.min_model_confidence:
                rejection_reasons.append(
                    f"Model confidence ({prob_up*100:.1f}%) below minimum required {self.min_model_confidence*100:.1f}%"
                )

            # 2. Expected return vs Transaction costs hurdle
            # CoinDCX roundtrip cost: 0.50% buy fee + 18% GST + 0.50% sell fee + 18% GST + 1% TDS ~ 2.18%
            if expected_ret < self.min_expected_return_pct:
                rejection_reasons.append(
                    f"Expected return ({expected_ret*100:.2f}%) insufficient to clear roundtrip cost hurdle ({self.min_expected_return_pct*100:.2f}%)"
                )

        # 3. Forecast uncertainty check
        if uncertainty_width > self.max_uncertainty_width_pct:
            rejection_reasons.append(
                f"Forecast uncertainty band width ({uncertainty_width*100:.2f}%) exceeds safety ceiling ({self.max_uncertainty_width_pct*100:.2f}%)"
            )

        # 4. Model consensus
        if model_agreement_pct < 0.65:
            rejection_reasons.append(
                f"Model consensus disagreement: agreement score {model_agreement_pct*100:.1f}% is below 65%"
            )

        # 5. Out of distribution / market anomaly
        if is_out_of_distribution:
            rejection_reasons.append("Market conditions flagged as out-of-distribution / anomalous")

        is_approved = len(rejection_reasons) == 0
        critic_score = round(max(0.0, 1.0 - (len(rejection_reasons) * 0.3)), 2)

        return {
            "approved": is_approved,
            "critic_score": critic_score,
            "rejection_reasons": rejection_reasons,
            "audit_summary": "Passed independent review" if is_approved else f"Rejected: {'; '.join(rejection_reasons)}",
        }
