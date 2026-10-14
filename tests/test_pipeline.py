import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, make_features
from src.modeling import fit_and_evaluate, predict_next, load_models
from src.data import data_quality_report, fetch_candles, save_candles, ingest_secondary_btcusd
from src.ocr_utils import parse_chart_entities


def synthetic_candles(n=1800, seed=7):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.00005, 0.002, n)
    close = 6500000 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.001, 0.0005, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(1.5, 0.5, n)
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="Asia/Kolkata"),
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


def test_features_are_finite_and_have_expected_columns():
    raw = synthetic_candles()
    featured = make_features(raw)
    assert set(FEATURES).issubset(featured.columns)
    assert len(featured) > 1000
    assert np.isfinite(featured[FEATURES].to_numpy()).all()


def test_latest_row_can_be_predicted_without_future_label(tmp_path):
    raw = synthetic_candles()
    metrics, results, clf, reg = fit_and_evaluate(raw, str(tmp_path / "models"))
    prediction = predict_next(raw, clf, reg)
    assert prediction["direction"] in {"UP", "DOWN"}
    assert 0 <= prediction["probability_up"] <= 1
    assert prediction["predicted_next_close"] > 0
    assert "test_close_mae_inr" in metrics
    assert "test_close_mape_pct" in metrics
    assert "test_precision" in metrics
    assert "test_recall" in metrics
    assert "test_f1" in metrics
    assert "test_confusion_matrix" in metrics
    assert "naive_close_baseline_mae_inr" in metrics
    assert len(results) > 100


def test_models_can_be_reloaded(tmp_path):
    raw = synthetic_candles()
    model_dir = tmp_path / "models"
    fit_and_evaluate(raw, str(model_dir))
    clf, reg = load_models(str(model_dir))
    result = predict_next(raw, clf, reg)
    assert result["predicted_next_close"] > 0


def test_quality_report_detects_interval_gaps_and_coverage():
    raw = synthetic_candles(100)
    raw = raw.drop(index=40).reset_index(drop=True)
    report = data_quality_report(raw)
    assert report["rows"] == 99
    assert report["gaps_over_22_5_minutes"] >= 1
    assert "observed_span_coverage_pct" in report
    assert report["missing_intervals"] >= 1


def test_fetch_candles_invalid_date_raises_error():
    with pytest.raises(ValueError):
        fetch_candles(pair="B-BTC_INR", start_date="invalid-date-format")


def test_fetch_candles_future_date_raises_error():
    future_date = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    with pytest.raises(ValueError, match="start_date must be earlier than the current time"):
        fetch_candles(pair="B-BTC_INR", start_date=future_date)


def test_fetch_candles_accepts_date_older_than_ten_years_and_handles_empty_windows(monkeypatch):
    import requests
    from src.data import CANDLE_MS

    start_date = "2010-03-17"
    start_ts = pd.Timestamp(start_date, tz="UTC")
    start_ms = int(start_ts.timestamp() * 1000)
    window_ms = 999 * CANDLE_MS

    chunk_2_start = start_ms + window_ms + 1
    fake_end_ms = chunk_2_start + (5 * CANDLE_MS)
    monkeypatch.setattr("src.data.time.time", lambda: fake_end_ms / 1000.0)

    test_candles = [
        {"time": chunk_2_start, "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1.5},
        {"time": chunk_2_start, "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1.5},
        {"time": chunk_2_start + CANDLE_MS, "open": 105.0, "high": 115.0, "low": 95.0, "close": 110.0, "volume": 2.0},
    ]

    call_count = 0

    class MockResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []
            return test_candles

    monkeypatch.setattr(requests.Session, "get", lambda self, url, params=None, timeout=None: MockResponse())

    df = fetch_candles(pair="B-BTC_INR", start_date=start_date, pause=0.0)

    assert not df.empty
    assert len(df) == 2
    assert df["time"].duplicated().sum() == 0


def test_fetch_candles_resumes_from_existing_data(monkeypatch, tmp_path):
    import requests
    from src.data import CANDLE_MS

    csv_file = tmp_path / "existing.csv"
    base_ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    existing = pd.DataFrame({
        "time": [base_ts, base_ts + pd.Timedelta(minutes=15)],
        "open": [100.0, 105.0], "high": [110.0, 115.0],
        "low": [95.0, 100.0], "close": [105.0, 110.0], "volume": [1.0, 1.2],
    })
    save_candles(existing, str(csv_file))

    next_candle_ms = int((base_ts + pd.Timedelta(minutes=30)).timestamp() * 1000)
    fake_end_ms = next_candle_ms + (2 * CANDLE_MS)
    monkeypatch.setattr("src.data.time.time", lambda: fake_end_ms / 1000.0)

    new_candles = [
        {"time": next_candle_ms, "open": 110.0, "high": 116.0, "low": 108.0, "close": 114.0, "volume": 1.5}
    ]

    class MockResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return new_candles

    monkeypatch.setattr(requests.Session, "get", lambda self, url, params=None, timeout=None: MockResponse())

    resumed_df = fetch_candles(pair="B-BTC_INR", resume_path=str(csv_file), pause=0.0)
    assert len(resumed_df) == 3
    assert resumed_df.attrs.get("resumed") is True


def test_secondary_btcusd_ingestion(tmp_path):
    src_csv = tmp_path / "btcusd.csv"
    dest_csv = tmp_path / "reconstructed.csv"
    raw_usd = pd.DataFrame({
        "time": pd.date_range("2015-01-01", periods=10, freq="15min", tz="UTC"),
        "open": [300.0] * 10, "high": [305.0] * 10, "low": [295.0] * 10,
        "close": [302.0] * 10, "volume": [50.0] * 10,
    })
    raw_usd.to_csv(src_csv, index=False)

    recon = ingest_secondary_btcusd(str(src_csv), str(dest_csv), usd_inr_rate=80.0)
    assert recon["close"].iloc[0] == 302.0 * 80.0
    assert recon["is_reconstructed"].all()
    assert dest_csv.exists()


def test_ocr_entity_parsing():
    sample_text = """
    BTC/INR 15m Binance / CoinDCX
    Current Price: ₹82,10,520.50
    24h High: 83,00,000.00
    24h Low: 81,50,000.00
    Time: 2026-10-14 22:30:00
    Volume: 12.5 BTC
    """
    res = parse_chart_entities(sample_text)
    assert len(res["prices"]) >= 3
    assert any(p["value"] == 8210520.50 for p in res["prices"])
    assert "BTC" in res["currencies"]
    assert "INR" in res["currencies"]
    assert any("2026-10-14" in t for t in res["timestamps"])
