from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
import pandas as pd

TAGGER_VERSION = "1.0.0"

class BehavioralMarketTagger:
    """
    Evaluates historical candle dynamics to generate behavioral strategy tags.
    Every tag is traceable to explicit deterministic formulas.
    """
    def __init__(self, version: str = TAGGER_VERSION):
        self.version = version

    def compute_tags(self, df: pd.DataFrame) -> List[str]:
        """
        Given a candle DataFrame with columns: open, high, low, close, volume.
        Requires at least 20 candles.
        """
        if len(df) < 20:
            return ["DATA_INSUFFICIENT"]

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)

        tags = []

        # 1. Trend analysis (SMA-7 vs SMA-20)
        sma7 = np.mean(close[-7:])
        sma20 = np.mean(close[-20:])
        if sma7 > sma20 * 1.005:
            tags.append("TRENDING_BULLISH")
        elif sma7 < sma20 * 0.995:
            tags.append("TRENDING_BEARISH")
        else:
            tags.append("RANGE_BOUND")

        # 2. Volatility analysis (Standard deviation of returns)
        recent_close = close[-21:] if len(close) >= 21 else close
        returns = np.diff(recent_close) / recent_close[:-1]
        vol = float(np.std(returns))
        if vol > 0.015:
            tags.append("HIGH_VOLATILITY")
        elif vol < 0.004:
            tags.append("LOW_VOLATILITY")

        # 3. Mean Reversion indicator (RSI-14 estimate)
        diffs = np.diff(close[-15:])
        gains = diffs[diffs > 0]
        losses = -diffs[diffs < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 1e-6
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-6
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        if rsi < 30.0:
            tags.append("OVERSOLD_MEAN_REVERSION")
        elif rsi > 70.0:
            tags.append("OVERBOUGHT_MEAN_REVERSION")

        # 4. Breakout detection (Recent close near 20-period high)
        max_20 = np.max(high[-20:])
        if close[-1] >= max_20 * 0.998:
            tags.append("BREAKOUT_WATCH")

        # 5. Liquidity verification
        avg_vol = np.mean(volume[-10:])
        if avg_vol <= 0.01:
            tags.append("ILLIQUID_RISK")

        return tags or ["NEUTRAL"]
