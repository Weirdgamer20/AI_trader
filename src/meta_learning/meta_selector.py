from __future__ import annotations

from typing import Dict, Any, List
from .meta_features import MetaFeatureExtractor

class MetaLearnerSelector:
    """
    Mandatory Meta-Learning Subsystem:
    Dynamically selects the appropriate base learner, horizon, and strategy configuration
    based on validated meta-features of market conditions.
    """
    def __init__(self, extractor: MetaFeatureExtractor | None = None):
        self.extractor = extractor or MetaFeatureExtractor()

    def select_optimal_configuration(
        self,
        meta_features: Dict[str, float],
        active_tags: List[str],
    ) -> Dict[str, Any]:
        """
        Maps meta-features to the highest expected utility configuration.
        """
        if meta_features.get("status") == "DATA_INSUFFICIENT" or meta_features.get("is_valid") is False:
            return {
                "status": "DATA_INSUFFICIENT",
                "can_trade": False,
                "selected_algorithm": None,
                "selected_horizon_candles": None,
                "entry_confidence_threshold": None,
                "rational": "Insufficient candle data to establish market regime. Trade proposals blocked.",
                "meta_features_snapshot": meta_features,
            }

        vol = meta_features.get("volatility_regime") or 0.01
        trend = meta_features.get("trend_strength") or 0.0
        expansion = meta_features.get("range_expansion") or 1.0

        # Meta-selection heuristics grounded in walk-forward empirical stability
        if "HIGH_VOLATILITY" in active_tags or vol > 0.018:
            # High volatility: Short horizon (1 candle) and robust tree-based model
            selected_algorithm = "hist_gradient_boosting"
            selected_horizon = 1
            entry_confidence_threshold = 0.58
            rational = "High volatility regime: selecting conservative 1-step horizon with tree regularization."
        elif "TRENDING_BULLISH" in active_tags and trend > 0.005:
            # Strong trend: Can extrapolate slightly further (2 candles)
            selected_algorithm = "hist_gradient_boosting"
            selected_horizon = 2
            entry_confidence_threshold = 0.54
            rational = "Bullish trend regime: selecting 2-step horizon to capture momentum continuation."
        elif "RANGE_BOUND" in active_tags or abs(trend) < 0.001:
            # Range bound: favor mean reversion / linear models with 1-candle horizon
            selected_algorithm = "logistic_regression"
            selected_horizon = 1
            entry_confidence_threshold = 0.55
            rational = "Range-bound regime: selecting 1-step mean reversion with linear regularized baseline."
        else:
            # Default / Neutral
            selected_algorithm = "random_forest"
            selected_horizon = 1
            entry_confidence_threshold = 0.55
            rational = "Neutral market: balanced ensemble random forest with standard 1-step horizon."

        return {
            "selected_algorithm": selected_algorithm,
            "selected_horizon_candles": selected_horizon,
            "entry_confidence_threshold": entry_confidence_threshold,
            "rational": rational,
            "meta_features_snapshot": meta_features,
        }
