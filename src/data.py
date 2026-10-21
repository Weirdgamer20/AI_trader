from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.coindcx.com"
MARKETS_URL = f"{BASE_URL}/exchange/v1/markets_details"
CANDLES_URL = f"{BASE_URL}/market_data/candles"
INTERVAL = "15m"
CANDLE_MS = 15 * 60 * 1000
REQUIRED_COLUMNS = {"time", "open", "high", "low", "close", "volume"}


def discover_btc_inr_pair(session: requests.Session | None = None) -> str:
    """Resolve the active BTC/INR spot pair from CoinDCX market metadata.

    Never assume a pair prefix: CoinDCX documents pair IDs as exchange-code based.
    """
    client = session or requests.Session()
    response = client.get(MARKETS_URL, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected markets metadata response: {type(payload).__name__}")

    matches: list[tuple[int, str]] = []
    for market in payload:
        if not isinstance(market, dict):
            continue
        pair = str(market.get("pair", "")).strip()
        target = str(market.get("target_currency_short_name", "")).upper()
        base = str(market.get("base_currency_short_name", "")).upper()
        status = str(market.get("status", "active")).lower()
        if target == "BTC" and base == "INR" and status == "active" and pair:
            # Prefer the spot-like B/I codes only if metadata returns more than one.
            priority = 0 if pair.startswith(("B-", "I-")) else 1
            matches.append((priority, pair))
    if not matches:
        raise RuntimeError(
            "CoinDCX market metadata did not list an active BTC/INR pair. "
            "Check current spot-market availability or pass the exact pair ID from markets_details."
        )
    matches.sort()
    return matches[0][1]


def _normalize_candles(payload: Any, pair: str) -> pd.DataFrame:
    if isinstance(payload, dict):
        raise RuntimeError(f"CoinDCX candle API returned an error object for {pair}: {payload}")
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected candle response type: {type(payload).__name__}")
    if not payload:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(payload)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise RuntimeError(f"CoinDCX candle response is missing fields: {sorted(missing)}")
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("Asia/Kolkata")
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
        & (df["volume"] >= 0)
    ]
    df.attrs["pair"] = pair
    return df.reset_index(drop=True)


def normalize_pair_symbol(pair: str) -> str:
    """Normalizes exchange pair codes (e.g., 'B-BTC_INR', 'I-BTC_INR', 'BTC_INR') to canonical symbol."""
    p = str(pair).strip().upper()
    if p.startswith(("B-", "I-")):
        p = p[2:]
    return p.replace("-", "").replace("_", "")


def fetch_candles(
    pair: str | None = None,
    start_date: str = "2018-10-18",
    pause: float = 0.3,
    max_retries: int = 5,
    resume_path: str | None = None,
) -> pd.DataFrame:
    """Download 15-minute candles using timestamp-bounded requests.

    Supports resuming from an existing dataset to avoid redundant downloads.
    The returned dataframe contains only data actually supplied by CoinDCX;
    no candles are synthesized.
    """
    from pathlib import Path

    session = requests.Session()
    session.headers.update({"User-Agent": "BTCINR-ML-Lab/1.0"})
    if not pair:
        pair = discover_btc_inr_pair(session)

    existing_df = None
    if resume_path:
        target_file = Path(resume_path)
        manifest_path = Path(f"{resume_path}.manifest.json")
        if manifest_path.exists():
            import json
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                manifest_pair = manifest.get("pair")
                if manifest_pair and pair:
                    if normalize_pair_symbol(manifest_pair) != normalize_pair_symbol(pair):
                        raise ValueError(
                            f"Dataset resume rejected: Manifest pair '{manifest_pair}' does not match requested pair '{pair}'. "
                            "Cross-pair data contamination strictly prevented."
                        )
            except ValueError:
                raise
            except Exception:
                pass

        if target_file.exists() and target_file.stat().st_size > 0:
            try:
                loaded = pd.read_csv(target_file, parse_dates=["time"])
                if REQUIRED_COLUMNS.issubset(loaded.columns) and not loaded.empty:
                    loaded["time"] = pd.to_datetime(loaded["time"], errors="coerce")
                    existing_df = loaded.dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
            except Exception:
                existing_df = None

    end_ms = int(time.time() * 1000)

    if existing_df is not None and not existing_df.empty:
        # Resume from the next interval after the latest saved candle
        last_saved_time = existing_df["time"].max()
        last_saved_utc = last_saved_time.tz_convert("UTC") if last_saved_time.tzinfo else last_saved_time.tz_localize("UTC")
        start_ms = int(last_saved_utc.timestamp() * 1000) + CANDLE_MS
        if start_ms >= end_ms:
            # Already up to date
            existing_df.attrs.update({"pair": pair, "requests_made": 0, "empty_chunks": 0, "resumed": True})
            return existing_df.reset_index(drop=True)
    else:
        start_timestamp = pd.Timestamp(start_date)
        if start_timestamp.tzinfo is None:
            start_timestamp = start_timestamp.tz_localize("UTC")
        else:
            start_timestamp = start_timestamp.tz_convert("UTC")
        start_ms = int(start_timestamp.timestamp() * 1000)
        if start_ms >= end_ms:
            raise ValueError("start_date must be earlier than the current time")

    # API limit is 1000. Request at most 999 expected intervals per chunk.
    window_ms = 999 * CANDLE_MS
    cursor = start_ms
    rows: list[dict[str, Any]] = []
    requests_made = 0
    empty_chunks = 0

    while cursor < end_ms:
        window_end = min(cursor + window_ms, end_ms)
        params = {
            "pair": pair,
            "interval": INTERVAL,
            "startTime": cursor,
            "endTime": window_end,
            "limit": 1000,
        }
        response = None
        for attempt in range(max_retries):
            try:
                response = session.get(CANDLES_URL, params=params, timeout=30)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    time.sleep(min(2 ** attempt, 20))
                    continue
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 20))
        if response is None:
            raise RuntimeError("No HTTP response received from CoinDCX.")
        requests_made += 1
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"CoinDCX returned non-JSON data: {response.text[:300]}") from exc

        normalized = _normalize_candles(payload, pair)
        if normalized.empty:
            empty_chunks += 1
        else:
            # Convert timestamps back to epoch milliseconds for range filtering.
            normalized["_time_ms"] = normalized["time"].values.astype("datetime64[ms]").astype("int64")
            normalized = normalized[
                (normalized["_time_ms"] >= cursor) & (normalized["_time_ms"] <= window_end)
            ]
            rows.extend(normalized.drop(columns=["_time_ms"]).to_dict("records"))

        cursor = window_end + 1
        if requests_made % 25 == 0:
            pct = min(100.0, (cursor - start_ms) / max(1, end_ms - start_ms) * 100)
            date_str = pd.to_datetime(window_end, unit="ms", utc=True).strftime("%Y-%m-%d")
            print(f"  [Progress] Chunk {requests_made} | {len(rows):,} new candles | reached {date_str} ({pct:.1f}%)", flush=True)
        time.sleep(max(0.0, pause))

    if not rows and existing_df is None:
        raise RuntimeError(
            f"No candles returned for pair {pair}. Verify pair metadata and that CoinDCX "
            "currently exposes historical spot candles for this market."
        )

    new_df = pd.DataFrame(rows)
    if not new_df.empty:
        new_df["time"] = pd.to_datetime(new_df["time"], errors="coerce")

    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True) if not new_df.empty else existing_df
    else:
        combined = new_df

    combined = combined.dropna(subset=["time"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    combined.attrs.update({
        "pair": pair,
        "requests_made": requests_made,
        "empty_chunks": empty_chunks,
        "resumed": existing_df is not None,
    })
    return combined


def get_dataset_path(pair: str, interval: str = INTERVAL, base_dir: str = "data/raw") -> Path:
    from pathlib import Path
    slug = pair.lower().replace("-", "_").replace("/", "_")
    return Path(base_dir) / f"{slug}_{interval}.csv"


def save_candles(df: pd.DataFrame, path: str = "data/raw/btc_inr_15m.csv", pair: str | None = None, interval: str = INTERVAL) -> None:
    from pathlib import Path
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)

    resolved_pair = pair or df.attrs.get("pair")
    manifest_file = Path(f"{path}.manifest.json")
    manifest_data = {
        "pair": resolved_pair,
        "interval": interval,
        "source": "CoinDCX",
        "schema_version": 1,
        "row_count": len(df),
        "first_candle": str(df["time"].min()) if not df.empty and "time" in df.columns else None,
        "last_candle": str(df["time"].max()) if not df.empty and "time" in df.columns else None,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)


def data_quality_report(df: pd.DataFrame) -> dict[str, object]:
    """Return deterministic integrity/coverage diagnostics for a candle dataframe."""
    if df.empty:
        return {"rows": 0, "error": "No candles"}
    data = df.copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data = data.dropna(subset=["time"]).sort_values("time")
    delta = data["time"].diff().dropna()
    expected_interval = pd.Timedelta(minutes=15)
    first_time = data["time"].min()
    last_time = data["time"].max()
    span_seconds = (last_time - first_time).total_seconds()
    expected_intervals = int(span_seconds / (15 * 60)) + 1 if span_seconds >= 0 else 0
    missing_intervals = max(0, expected_intervals - len(data))
    coverage_pct = round((len(data) / expected_intervals) * 100, 2) if expected_intervals > 0 else 100.0

    return {
        "rows": int(len(data)),
        "first_candle": str(first_time),
        "last_candle": str(last_time),
        "expected_intervals_in_span": expected_intervals,
        "missing_intervals": missing_intervals,
        "observed_span_coverage_pct": coverage_pct,
        "duplicate_timestamps": int(data["time"].duplicated().sum()),
        "invalid_or_nonpositive_close": int((pd.to_numeric(data["close"], errors="coerce") <= 0).sum()),
        "negative_volume_rows": int((pd.to_numeric(data["volume"], errors="coerce") < 0).sum()),
        "gaps_over_22_5_minutes": int((delta > expected_interval * 1.5).sum()),
        "median_interval": str(delta.median()) if not delta.empty else None,
        "expected_rows_per_year": 35040,
    }


def ingest_secondary_btcusd(
    input_path: str,
    output_path: str = "data/reconstructed/btc_inr_reconstructed.csv",
    usd_inr_rate: float = 83.5,
    rate_source_note: str = "Fixed benchmark conversion or RBI reference rate",
) -> pd.DataFrame:
    """Ingest external BTC/USD history, convert to INR, and preserve separate provenance.

    Reconstructed candles are strictly labeled as reconstructed and never mixed into genuine
    CoinDCX raw data.
    """
    from pathlib import Path

    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Secondary BTC/USD file not found: {input_path}")

    df = pd.read_csv(src, parse_dates=["time"])
    required = {"time", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Secondary dataset missing columns: {sorted(required - set(df.columns))}")

    reconstructed = df.copy()
    for col in ["open", "high", "low", "close"]:
        reconstructed[f"{col}_usd"] = pd.to_numeric(reconstructed[col], errors="coerce")
        reconstructed[col] = reconstructed[f"{col}_usd"] * usd_inr_rate

    reconstructed["is_reconstructed"] = True
    reconstructed["usd_inr_conversion_rate"] = usd_inr_rate
    reconstructed["conversion_note"] = rate_source_note

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    reconstructed.to_csv(dest, index=False)
    return reconstructed

