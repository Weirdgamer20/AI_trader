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
        self.residuals_p05: Dict[int, float] = {}
        self.residuals_p95: Dict[int, float] = {}
        self.empirical_coverage_90: Dict[int, float] = {}
        self.is_fitted = False

    def fit(self, raw_df: pd.DataFrame, train_ratio: float = 0.70, calib_ratio: float = 0.15) -> Dict[str, Any]:
        """
        Fit models on chronological train split, calibrate 90% empirical bounds
        (5th and 95th percentiles) on an independent calibration split, and test coverage on test split.
        """
        df = make_features(raw_df, include_unlabeled=True)
        if len(df) < 200:
            raise ValueError(f"Need at least 200 candles to train multi-horizon forecaster, got {len(df)}")

        n = len(df)
        train_end = int(n * train_ratio)
        calib_end = int(n * (train_ratio + calib_ratio))

        eval_metrics = {}

        for h in self.horizons:
            # Targets for horizon h: log return for close, ratio for high, ratio for low
            df[f"target_close_{h}"] = df["close"].shift(-h)
            df[f"target_log_ret_{h}"] = np.log(df[f"target_close_{h}"] / df["close"])
            df[f"target_high_{h}"] = df["high"].rolling(h).max().shift(-h)
            df[f"target_high_ratio_{h}"] = df[f"target_high_{h}"] / df["close"]
            df[f"target_low_{h}"] = df["low"].rolling(h).min().shift(-h)
            df[f"target_low_ratio_{h}"] = df[f"target_low_{h}"] / df["close"]
            df[f"target_dir_{h}"] = (df[f"target_close_{h}"] > df["close"]).astype(float)

            req_cols = FEATURES + [
                f"target_log_ret_{h}", f"target_high_ratio_{h}", f"target_low_ratio_{h}", f"target_dir_{h}"
            ]
            # P1-8: Purge the last h rows from train and calib partitions to prevent target leakage across split boundaries
            train_purge_end = max(0, train_end - h)
            calib_purge_end = max(train_end, calib_end - h)

            clean_train = df.iloc[:train_purge_end].dropna(subset=req_cols)
            clean_calib = df.iloc[train_end:calib_purge_end].dropna(subset=req_cols)
            clean_test = df.iloc[calib_end:].dropna(subset=req_cols)

            if len(clean_train) < 50 or len(clean_calib) < 20:
                continue

            X_tr, y_ret_tr = clean_train[FEATURES], clean_train[f"target_log_ret_{h}"]
            y_high_tr = clean_train[f"target_high_ratio_{h}"]
            y_low_tr = clean_train[f"target_low_ratio_{h}"]
            y_dir_tr = clean_train[f"target_dir_{h}"]

            # Train models on train split
            m_close = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.06, random_state=42)
            m_high = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.06, random_state=42)
            m_low = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.06, random_state=42)
            m_dir = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.06, random_state=42)

            m_close.fit(X_tr, y_ret_tr)
            m_high.fit(X_tr, y_high_tr)
            m_low.fit(X_tr, y_low_tr)
            m_dir.fit(X_tr, y_dir_tr)

            self.models_close[h] = m_close
            self.models_high[h] = m_high
            self.models_low[h] = m_low
            self.models_dir[h] = m_dir

            # Calibrate 90% interval on separate out-of-sample calibration split
            calib_pred_ret = m_close.predict(clean_calib[FEATURES])
            calib_pred_close = clean_calib["close"] * np.exp(calib_pred_ret)
            calib_residuals = (clean_calib[f"target_close_{h}"] - calib_pred_close) / calib_pred_close

            p05 = float(np.percentile(calib_residuals, 5))
            p95 = float(np.percentile(calib_residuals, 95))
            self.residuals_p05[h] = p05
            self.residuals_p95[h] = p95

            # Test empirical coverage on untouched final test split (P1-9)
            if len(clean_test) > 0:
                test_pred_ret = m_close.predict(clean_test[FEATURES])
                test_pred_close = clean_test["close"] * np.exp(test_pred_ret)
                test_band_low = test_pred_close * (1.0 + p05)
                test_band_high = test_pred_close * (1.0 + p95)
                actual_in_band = (clean_test[f"target_close_{h}"] >= test_band_low) & (
                    clean_test[f"target_close_{h}"] <= test_band_high
                )
                coverage = float(actual_in_band.mean())
                self.empirical_coverage_90[h] = round(coverage, 4)
                test_mae = float(np.mean(np.abs(clean_test[f"target_close_{h}"] - test_pred_close)))
                eval_metrics[f"h{h}_mae"] = round(test_mae, 2)
                eval_metrics[f"h{h}_test_mae"] = round(test_mae, 2)
                eval_metrics[f"h{h}_empirical_coverage_90"] = round(coverage, 4)
                eval_metrics[f"h{h}_status"] = "EVALUATED"
            else:
                # Never fabricate coverage or zero error when test split is empty (P1-9)
                self.empirical_coverage_90[h] = None
                eval_metrics[f"h{h}_mae"] = None
                eval_metrics[f"h{h}_test_mae"] = None
                eval_metrics[f"h{h}_empirical_coverage_90"] = None
                eval_metrics[f"h{h}_status"] = "DATA_INSUFFICIENT"

        if not self.models_close:
            raise RuntimeError("Failed to fit forecaster on any horizon.")

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

            # P1-10: Require fitted model and valid calibration artifact for this horizon
            has_calib = (h in self.residuals_p05) and (h in self.residuals_p95)
            is_tradable = has_calib

            pred_log_ret = float(self.models_close[h].predict(latest_row[FEATURES])[0])
            pred_close = float(current_close * np.exp(pred_log_ret))

            pred_high_ratio = float(self.models_high[h].predict(latest_row[FEATURES])[0])
            pred_high = float(max(pred_close, current_close * pred_high_ratio))

            pred_low_ratio = float(self.models_low[h].predict(latest_row[FEATURES])[0])
            pred_low = float(min(pred_close, current_close * pred_low_ratio))
            # Validate positive finite bounds
            pred_low = max(0.01, pred_low)
            pred_high = max(pred_low, pred_high)

            prob_up = float(self.models_dir[h].predict_proba(latest_row[FEATURES])[0][1])

            # Expected wicks
            upper_wick = max(0.0, pred_high - max(current_close, pred_close))
            lower_wick = max(0.0, min(current_close, pred_close) - pred_low)

            # Uncertainty interval (90% calibrated from 5th to 95th percentiles)
            if has_calib:
                band_low = max(0.01, pred_close * (1.0 + self.residuals_p05[h]))
                band_high = max(band_low, pred_close * (1.0 + self.residuals_p95[h]))
                uncertainty_width_pct = (band_high - band_low) / pred_close
                uncertainty_calibrated = True
            else:
                band_low = None
                band_high = None
                uncertainty_width_pct = None
                uncertainty_calibrated = False

            expected_return_pct = (pred_close - current_close) / current_close
            downside_risk_pct = max(0.005, (current_close - pred_low) / current_close)

            proj = {
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
                "is_tradable": is_tradable,
                "uncertainty_calibrated": uncertainty_calibrated,
                "measured_90_coverage": self.empirical_coverage_90.get(h),
            }
            if band_low is not None and band_high is not None:
                proj["uncertainty_band_low"] = round(band_low, 2)
                proj["uncertainty_band_high"] = round(band_high, 2)
                proj["uncertainty_width_pct"] = round(uncertainty_width_pct, 4)

            projections.append(proj)

        return {
            "current_price": current_close,
            "candle_timestamp": str(latest_row["time"].iloc[0]),
            "projections": projections,
        }
