from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, make_features

MODEL_VERSION = 3


def _safe_auc(y_true, probabilities):
    return float(roc_auc_score(y_true, probabilities)) if pd.Series(y_true).nunique() == 2 else None


def fit_and_evaluate(raw: pd.DataFrame, output_dir: str = "models"):
    data = make_features(raw)
    if len(data) < 1000:
        raise ValueError(f"Need at least 1,000 usable candles; currently have {len(data)}.")
    n = len(data)
    train_end, val_end = int(n * .70), int(n * .85)
    train, val, test = data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Not enough data for chronological train/validation/test splits.")

    clf_candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        ),
    }
    val_scores = {}
    for name, model in clf_candidates.items():
        model.fit(train[FEATURES], train["target_up"].astype(int))
        val_pred = model.predict(val[FEATURES])
        val_scores[name] = float(balanced_accuracy_score(val["target_up"].astype(int), val_pred))

    best_name = max(val_scores, key=val_scores.get)
    trainval = data.iloc[:val_end]
    classifier = clf_candidates[best_name]
    classifier.fit(trainval[FEATURES], trainval["target_up"].astype(int))
    test_prob = classifier.predict_proba(test[FEATURES])[:, 1]
    test_pred = (test_prob >= .5).astype(int)
    y_test = test["target_up"].astype(int)
    majority_baseline = max(float(y_test.mean()), float(1 - y_test.mean()))

    regressor = HistGradientBoostingRegressor(
        max_iter=180, learning_rate=.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42,
    )
    regressor.fit(trainval[FEATURES], trainval["target_close"])
    price_pred = regressor.predict(test[FEATURES])
    # Naive price baseline: next close equals current close.
    naive_pred = test["close"].to_numpy(dtype=float)

    cm = confusion_matrix(y_test, test_pred).tolist()

    metrics = {
        "model_version": MODEL_VERSION,
        "selected_classifier": best_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_rows": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "validation_balanced_accuracy_by_model": val_scores,
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_confusion_matrix": cm,
        "test_majority_direction_baseline_accuracy": majority_baseline,
        "test_roc_auc": _safe_auc(y_test, test_prob),
        "test_close_mae_inr": float(mean_absolute_error(test["target_close"], price_pred)),
        "test_close_rmse_inr": float(np.sqrt(mean_squared_error(test["target_close"], price_pred))),
        "test_close_mape_pct": float(mean_absolute_percentage_error(test["target_close"], price_pred)) * 100,
        "naive_close_baseline_mae_inr": float(mean_absolute_error(test["target_close"], naive_pred)),
        "naive_close_baseline_rmse_inr": float(np.sqrt(mean_squared_error(test["target_close"], naive_pred))),
        "naive_close_baseline_mape_pct": float(mean_absolute_percentage_error(test["target_close"], naive_pred)) * 100,
        "test_rows": int(len(test)),
        "test_start": str(test["time"].iloc[0]),
        "test_end": str(test["time"].iloc[-1]),
        "warning": "Historical metrics do not imply future profitability; fees, slippage, taxes and execution are not modeled.",
    }

    test_results = test[["time", "close", "target_close", "target_up"]].copy()
    test_results["predicted_up_probability"] = test_prob
    test_results["predicted_up"] = test_pred
    test_results["predicted_next_close"] = price_pred
    test_results["naive_next_close"] = naive_pred
    test_results["price_error_inr"] = price_pred - test["target_close"]
    test_results["direction_correct"] = test_pred == y_test

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": classifier, "features": FEATURES, "version": MODEL_VERSION, "name": best_name}, out / "direction_model.joblib")
    joblib.dump({"model": regressor, "features": FEATURES, "version": MODEL_VERSION, "name": "hist_gradient_boosting_regressor"}, out / "price_model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    test_results.to_csv(out / "test_predictions.csv", index=False)
    return metrics, test_results, classifier, regressor


def load_models(model_dir: str = "models"):
    directory = Path(model_dir)
    direction_path = directory / "direction_model.joblib"
    price_path = directory / "price_model.joblib"
    if not direction_path.exists() or not price_path.exists():
        raise FileNotFoundError("Models not found. Run `python train_model.py` first.")
    direction_bundle = joblib.load(direction_path)
    price_bundle = joblib.load(price_path)
    if direction_bundle.get("version") != MODEL_VERSION or price_bundle.get("version") != MODEL_VERSION:
        raise RuntimeError("Saved model version is outdated. Retrain with `python train_model.py`.")
    return direction_bundle["model"], price_bundle["model"]


def predict_next(raw: pd.DataFrame, classifier, regressor):
    # The final row can be predicted even though its future label is not known yet.
    featured = make_features(raw, include_unlabeled=True)
    if featured.empty:
        raise ValueError("Not enough clean candles to generate a prediction.")
    latest = featured.iloc[[-1]]
    probability_up = float(classifier.predict_proba(latest[FEATURES])[0, 1])
    predicted_close = float(regressor.predict(latest[FEATURES])[0])
    last_close = float(latest["close"].iloc[0])
    diff = predicted_close - last_close
    return {
        "last_close": last_close,
        "predicted_next_close": predicted_close,
        "predicted_change_inr": diff,
        "predicted_change_pct": (diff / last_close) * 100,
        "probability_up": probability_up,
        "direction": "UP" if probability_up >= .5 else "DOWN",
        "candle_time": str(latest["time"].iloc[0]),
        "next_candle_minutes": 15,
        "prediction_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
