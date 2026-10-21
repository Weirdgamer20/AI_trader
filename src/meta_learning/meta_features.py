from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List

class MetaFeatureExtractor:
    """
    Extracts time-consistent, non-leaking meta-features to characterize market regime
    and recent predictive capability across candidate learning algorithms.
    """
    def __init__(self, lookback_window: int = 30):
        self.lookback_window = lookback_window

    def extract_meta_features(
        self,
        candle_df: pd.DataFrame,
        recent_learner_scores: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """
        Calculates meta-features from historical candles and past validated learner scores.
        """
        if len(candle_df) < 15:
            return {
                "volatility_regime": 0.01,
                "trend_strength": 0.0,
                "range_expansion": 0.0,
                "volume_ratio": 1.0,
                "learner_consensus": 0.5,
            }

        close = candle_df["close"].to_numpy(dtype=float)
        high = candle_df["high"].to_numpy(dtype=float)
        low = candle_df["low"].to_numpy(dtype=float)
        volume = candle_df["volume"].to_numpy(dtype=float)

        # 1. Volatility regime
        rets = np.diff(close[-self.lookback_window:]) / close[-self.lookback_window:-1]
        volatility_regime = float(np.std(rets))

        # 2. Trend strength (direction of EMA-8 vs EMA-21)
        ema8 = float(candle_df["close"].ewm(span=8).mean().iloc[-1])
        ema21 = float(candle_df["close"].ewm(span=21).mean().iloc[-1])
        trend_strength = (ema8 - ema21) / close[-1]

        # 3. Range expansion
        recent_range = float(np.mean((high[-5:] - low[-5:]) / close[-5:]))
        hist_range = float(np.mean((high[-self.lookback_window:] - low[-self.lookback_window:]) / close[-self.lookback_window:]))
        range_expansion = recent_range / hist_range if hist_range > 0 else 1.0

        # 4. Volume ratio
        recent_vol = float(np.mean(volume[-5:]))
        hist_vol = float(np.mean(volume[-self.lookback_window:]))
        volume_ratio = recent_vol / hist_vol if hist_vol > 0 else 1.0

        # 5. Baseline learner performance meta-feature
        scores = recent_learner_scores or {}
        mean_learner_score = float(np.mean(list(scores.values()))) if scores else 0.5

        return {
            "volatility_regime": round(volatility_regime, 5),
            "trend_strength": round(trend_strength, 5),
            "range_expansion": round(range_expansion, 4),
            "volume_ratio": round(volume_ratio, 4),
            "learner_consensus": round(mean_learner_score, 4),
        }
