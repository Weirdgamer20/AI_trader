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

        direction = forecast_projection.get("direction")
        prob_up = forecast_projection.get("probability_up")
        expected_ret = forecast_projection.get("expected_return_pct")
        uncertainty_width = forecast_projection.get("uncertainty_width_pct")

        # P1-5: Require explicit, valid uncertainty estimate; never substitute zero
        if uncertainty_width is None or not isinstance(uncertainty_width, (int, float)):
            rejection_reasons.append("Missing or invalid forecast uncertainty estimate (fail-closed)")
        elif uncertainty_width > self.max_uncertainty_width_pct:
            rejection_reasons.append(
                f"Forecast uncertainty band width ({uncertainty_width*100:.2f}%) exceeds safety ceiling ({self.max_uncertainty_width_pct*100:.2f}%)"
            )

        if expected_ret is None or not isinstance(expected_ret, (int, float)):
            rejection_reasons.append("Missing expected return estimate")
        if prob_up is None or not isinstance(prob_up, (int, float)):
            rejection_reasons.append("Missing directional probability estimate")

        # Dynamic roundtrip hurdle: 0.5% buy + 18% GST (0.09%) + 0.5% sell + 18% GST (0.09%) + 1.0% TDS + 0.3% slippage ~ 2.48%
        cost_hurdle = 0.0248

        # 1. Directional alignment & expected return check
        side_norm = side.upper()
        if side_norm == "BUY":
            if direction != "UP" or (prob_up is not None and prob_up < self.min_model_confidence):
                rejection_reasons.append(
                    f"BUY directional confidence ({prob_up*100:.1f}%) below minimum required {self.min_model_confidence*100:.1f}%"
                )
            if expected_ret is not None and expected_ret < cost_hurdle:
                rejection_reasons.append(
                    f"Expected return ({expected_ret*100:.2f}%) insufficient to clear roundtrip cost hurdle ({cost_hurdle*100:.2f}%)"
                )
        elif side_norm == "SELL":
            # For sell, expected return (or downside) should align with exiting
            if direction == "UP" and prob_up is not None and prob_up > 0.70:
                rejection_reasons.append(
                    f"Strong upward forecast ({prob_up*100:.1f}%) conflicts with proposed discretionary SELL"
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
