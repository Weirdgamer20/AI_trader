import pytest
import numpy as np
import pandas as pd
from src.scanner.tagging import BehavioralMarketTagger
from src.scanner.coin_scanner import DynamicCoinScanner
from src.modeling_multicoin import MultiHorizonForecaster

def create_sample_candles(n=300):
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0001, 0.003, n)
    close = 5000000.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.002, 0.001, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.002, 0.001, n)))
    volume = rng.lognormal(2.0, 0.5, n)
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })

def test_behavioral_market_tagger():
    tagger = BehavioralMarketTagger()
    candles = create_sample_candles(50)
    tags = tagger.compute_tags(candles)
    assert len(tags) > 0
    assert any(t in tags for t in ["TRENDING_BULLISH", "TRENDING_BEARISH", "RANGE_BOUND"])

def test_dynamic_coin_scanner_filtering_and_ranking():
    scanner = DynamicCoinScanner(min_24h_volume_inr=50000.0, max_spread_pct=0.015, max_scan_candidates=50)

    # Simulated CoinDCX ticker with 3 markets: 1 liquid & tight, 1 wide spread, 1 tiny volume
    mock_ticker = [
        {"market": "I-BTC_INR", "last_price": "6000000", "bid": "5995000", "ask": "6005000", "volume": "5.0", "change_24_hour": "2.5"},
        {"market": "I-WIDE_INR", "last_price": "100", "bid": "90", "ask": "110", "volume": "1000", "change_24_hour": "0.5"}, # Spread 18%
        {"market": "I-TINY_INR", "last_price": "10", "bid": "9.99", "ask": "10.01", "volume": "10", "change_24_hour": "-1.0"}, # Volume ₹100 < 50k
        {"market": "ETHUSDT", "last_price": "3000", "bid": "2999", "ask": "3001", "volume": "100", "change_24_hour": "1.0"}, # USDT excluded
    ]

    results = scanner.scan_markets(raw_ticker=mock_ticker)
    assert len(results) <= 50
    assert len(results) == 3 # Only INR spot pairs considered

    btc = next(r for r in results if r["market_pair"] == "I-BTC_INR")
    assert btc["is_eligible"] is True
    assert btc["eligibility_status"] == "ELIGIBLE"

    wide = next(r for r in results if r["market_pair"] == "I-WIDE_INR")
    assert wide["is_eligible"] is False
    assert wide["eligibility_status"] == "HIGH_SPREAD"

    tiny = next(r for r in results if r["market_pair"] == "I-TINY_INR")
    assert tiny["is_eligible"] is False
    assert tiny["eligibility_status"] == "LOW_VOLUME"

def test_multi_horizon_forecaster_predictions():
    candles = create_sample_candles(250)
    forecaster = MultiHorizonForecaster(horizons=[1, 2])
    metrics = forecaster.fit(candles)
    assert "h1_mae" in metrics
    assert "h2_mae" in metrics

    latest = forecaster.predict_latest(candles)
    assert latest["current_price"] > 0
    assert len(latest["projections"]) == 2

    p1 = latest["projections"][0]
    assert p1["horizon_candles"] == 1
    assert p1["predicted_close"] > 0
    assert p1["predicted_high"] >= p1["predicted_close"]
    assert p1["predicted_low"] <= p1["predicted_close"]
    assert p1["direction"] in ["UP", "DOWN"]
    assert 0.0 <= p1["probability_up"] <= 1.0
    assert "uncertainty_band_low" in p1
    assert "uncertainty_band_high" in p1
    assert p1["uncertainty_band_high"] >= p1["uncertainty_band_low"]
    assert p1["downside_risk_pct"] > 0
