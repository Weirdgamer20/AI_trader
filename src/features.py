from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_2", "ret_4", "ret_8", "ret_16",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "volume_change", "volatility_8", "volatility_32",
    "sma_gap_8", "sma_gap_32", "ema_gap_8", "ema_gap_32",
    "rsi_14", "macd_pct",
]


def make_features(raw: pd.DataFrame, include_unlabeled: bool = False) -> pd.DataFrame:
    """Create features known at candle close; target is the following candle's close."""
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required candle columns: {sorted(missing)}")

    df = raw.copy().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(df["open"], errors="coerce").astype(float)
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float)
    ret = close.pct_change()
    df["ret_1"] = ret
    for n in (2, 4, 8, 16):
        df[f"ret_{n}"] = close.pct_change(n)

    df["range_pct"] = (high - low) / close
    df["body_pct"] = (close - open_) / open_.replace(0, np.nan)
    df["upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / close
    df["lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / close
    df["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    df["volatility_8"] = ret.rolling(8, min_periods=8).std()
    df["volatility_32"] = ret.rolling(32, min_periods=32).std()
    df["sma_gap_8"] = close / close.rolling(8, min_periods=8).mean() - 1
    df["sma_gap_32"] = close / close.rolling(32, min_periods=32).mean() - 1
    df["ema_gap_8"] = close / close.ewm(span=8, adjust=False).mean() - 1
    df["ema_gap_32"] = close / close.ewm(span=32, adjust=False).mean() - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
    rsi = rsi.mask((gain == 0) & (loss == 0), 50.0)
    df["rsi_14"] = rsi

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_pct"] = (ema12 - ema26) / close

    # Row t features use only row t and earlier. Label is next candle close/direction/log_return.
    df["target_close"] = close.shift(-1)
    df["target_log_return"] = np.where(
        (df["target_close"].notna()) & (close > 0) & (df["target_close"] > 0),
        np.log(df["target_close"] / close),
        np.nan,
    )
    df["target_up"] = np.where(df["target_close"].notna(), (df["target_close"] > close).astype(float), np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    if include_unlabeled:
        return df.dropna(subset=FEATURES).reset_index(drop=True)
    return df.dropna(subset=FEATURES + ["target_close", "target_up", "target_log_return"]).reset_index(drop=True)
