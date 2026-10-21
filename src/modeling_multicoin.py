from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from .features import FEATURES, make_features

class MultiHorizonForecaster:
    """
    Forecasting Engine:
    - Forecasts next OHLC candle, wicks (upper/lower), direction, and expected return.
    - Multi-candle projections (1-step, 2-step, 4-step).
    - Downside risk estimation.
    - Calibrated uncertainty intervals (residual error percentiles).
    """
    def __init__(self, horizons: List[int] | None = None):
        self.horizons = horizons or [1, 2, 4]
        self.models_close: Dict[int, Any] = {}
        self.models_high: Dict[int, Any] = {}
        self.models_low: Dict[int, Any] = {}
        self.models_dir: Dict[int, Any] = {}
        self.residuals_p10: Dict[int, float] = {}
        self.residuals_p90: Dict[int, float] = {}
        self.is_fitted = False

    def fit(self, raw_df: pd.DataFrame, train_ratio: float = 0.85) -> Dict[str, Any]:
        """Fit models on chronological train split, calibrate uncertainty on holdout."""
        df = make_features(raw_df, include_unlabeled=True)
        if len(df) < 200:
            raise ValueError(f"Need at least 200 candles to train multi-horizon forecaster, got {len(df)}")

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)

        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        eval_metrics = {}

        for h in self.horizons:
            # Targets for horizon h
            df[f"target_close_{h}"] = df["close"].shift(-h)
            df[f"target_high_{h}"] = df["high"].rolling(h).max().shift(-h)
            df[f"target_low_{h}"] = df["low"].rolling(h).min().shift(-h)
            df[f"target_dir_{h}"] = (df[f"target_close_{h}"] > df["close"]).astype(float)

            clean_train = df.iloc[:split_idx].dropna(subset=FEATURES + [f"target_close_{h}", f"target_high_{h}", f"target_low_{h}"])
            clean_val = df.iloc[split_idx:].dropna(subset=FEATURES + [f"target_close_{h}", f"target_high_{h}", f"target_low_{h}"])

            if len(clean_train) < 50 or len(clean_val) < 20:
                continue

            X_tr, y_close_tr = clean_train[FEATURES], clean_train[f"target_close_{h}"]
            y_high_tr = clean_train[f"target_high_{h}"]
            y_low_tr = clean_train[f"target_low_{h}"]
            y_dir_tr = clean_train[f"target_dir_{h}"]

            X_val, y_close_val = clean_val[FEATURES], clean_val[f"target_close_{h}"]

            # Train models
            m_close = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, random_state=42)
            m_high = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.08, random_state=42)
            m_low = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.08, random_state=42)
            m_dir = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, random_state=42)

            m_close.fit(X_tr, y_close_tr)
            m_high.fit(X_tr, y_high_tr)
            m_low.fit(X_tr, y_low_tr)
            m_dir.fit(X_tr, y_dir_tr)

            self.models_close[h] = m_close
            self.models_high[h] = m_high
            self.models_low[h] = m_low
            self.models_dir[h] = m_dir

            # Calibrate uncertainty bounds on holdout
            preds_val = m_close.predict(X_val)
            residuals = (y_close_val - preds_val) / y_close_val
            self.residuals_p10[h] = float(np.percentile(residuals, 10))
            self.residuals_p90[h] = float(np.percentile(residuals, 90))

            mae = float(np.mean(np.abs(y_close_val - preds_val)))
            dir_acc = float(np.mean(m_dir.predict(X_val) == clean_val[f"target_dir_{h}"]))
            eval_metrics[f"h{h}_mae"] = round(mae, 2)
            eval_metrics[f"h{h}_dir_acc"] = round(dir_acc, 4)

        self.is_fitted = True
        return eval_metrics

    def predict_latest(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate full multi-horizon forecast from latest candle."""
        if not self.is_fitted:
            raise RuntimeError("Forecaster must be fitted before calling predict_latest.")

        df = make_features(raw_df, include_unlabeled=True)
        latest_row = df.iloc[-1:]
        current_close = float(latest_row["close"].iloc[0])

        projections = []
        for h in self.horizons:
            if h not in self.models_close:
                continue

            pred_close = float(self.models_close[h].predict(latest_row[FEATURES])[0])
            pred_high = float(max(pred_close, self.models_high[h].predict(latest_row[FEATURES])[0]))
            pred_low = float(min(pred_close, self.models_low[h].predict(latest_row[FEATURES])[0]))
            prob_up = float(self.models_dir[h].predict_proba(latest_row[FEATURES])[0][1])

            # Expected wicks
            upper_wick = max(0.0, pred_high - max(current_close, pred_close))
            lower_wick = max(0.0, min(current_close, pred_close) - pred_low)

            # Uncertainty interval (90% calibrated)
            band_low = pred_close * (1.0 + self.residuals_p10.get(h, -0.02))
            band_high = pred_close * (1.0 + self.residuals_p90.get(h, 0.02))
            uncertainty_width_pct = (band_high - band_low) / pred_close

            expected_return_pct = (pred_close - current_close) / current_close
            downside_risk_pct = max(0.005, (current_close - pred_low) / current_close)

            projections.append({
                "horizon_candles": h,
                "horizon_minutes": h * 15,
                "predicted_close": round(pred_close, 2),
                "predicted_high": round(pred_high, 2),
                "predicted_low": round(pred_low, 2),
                "predicted_open": round(current_close, 2),
                "predicted_upper_wick": round(upper_wick, 2),
                "predicted_lower_wick": round(lower_wick, 2),
                "direction": "UP" if prob_up >= 0.5 else "DOWN",
                "probability_up": round(prob_up, 4),
                "expected_return_pct": round(expected_return_pct, 4),
                "downside_risk_pct": round(downside_risk_pct, 4),
                "uncertainty_band_low": round(band_low, 2),
                "uncertainty_band_high": round(band_high, 2),
                "uncertainty_width_pct": round(uncertainty_width_pct, 4),
            })

        return {
            "current_price": current_close,
            "candle_timestamp": str(latest_row["time"].iloc[0]),
            "projections": projections,
        }
