from __future__ import annotations

import sys
import requests

from src.data import CANDLES_URL, discover_btc_inr_pair


def main():
    print("Checking CoinDCX public API connectivity...")
    try:
        pair = discover_btc_inr_pair()
        print(f"Active BTC/INR pair discovered: {pair}")
        response = requests.get(
            CANDLES_URL,
            params={"pair": pair, "interval": "15m", "limit": 3},
            timeout=20,
            headers={"User-Agent": "BTCINR-ML-Lab/1.0"},
        )
        print(f"Candle endpoint HTTP status: {response.status_code}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            print(f"Unexpected/empty candle payload: {str(payload)[:500]}")
            raise SystemExit(2)
        print(f"Candles returned: {len(payload)}")
        print(f"Response fields: {sorted(payload[0].keys())}")
        for candle in payload[:3]:
            print(
                f"time_ms={candle.get('time')} open={candle.get('open')} "
                f"high={candle.get('high')} low={candle.get('low')} "
                f"close={candle.get('close')} volume={candle.get('volume')}"
            )
        print("\nAPI smoke test passed. This confirms recent candle access only; it does NOT confirm five-year retention.")
    except requests.RequestException as exc:
        print(f"Network/API request failed: {type(exc).__name__}: {exc}")
        print("Check internet access, DNS, VPN/firewall, and CoinDCX service status.")
        raise SystemExit(1)
    except Exception as exc:
        print(f"Validation failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
