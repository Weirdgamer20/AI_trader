from __future__ import annotations

import requests
from typing import List, Dict, Any, Optional
import pandas as pd
from .tagging import BehavioralMarketTagger

COINDCX_TICKER_URL = "https://api.coindcx.com/exchange/ticker"
COINDCX_MARKETS_URL = "https://api.coindcx.com/exchange/v1/markets"

class DynamicCoinScanner:
    """
    Dynamic Coin Scanner for CoinDCX Spot markets.
    Scans up to 50 eligible Spot candidates.
    Does NOT force a fixed number of coins or trades.
    Rejects illiquid, excessive spread, or unverified markets.
    """
    def __init__(
        self,
        min_24h_volume_inr: float = 50000.0,
        max_spread_pct: float = 0.02,
        max_scan_candidates: int = 50,
        tagger: BehavioralMarketTagger | None = None,
    ):
        self.min_24h_volume_inr = min_24h_volume_inr
        self.max_spread_pct = max_spread_pct
        self.max_scan_candidates = max_scan_candidates
        self.tagger = tagger or BehavioralMarketTagger()

    def fetch_spot_ticker(self, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
        """Fetches raw CoinDCX public ticker payload."""
        sess = session or requests.Session()
        try:
            resp = sess.get(COINDCX_TICKER_URL, timeout=15, headers={"User-Agent": "Spot-Scanner/1.0"})
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def scan_markets(
        self,
        raw_ticker: List[Dict[str, Any]] | None = None,
        candle_data_lookup: Dict[str, pd.DataFrame] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Filters and ranks eligible CoinDCX Spot pairs up to max_scan_candidates.
        """
        ticker = raw_ticker if raw_ticker is not None else self.fetch_spot_ticker()
        candle_lookup = candle_data_lookup or {}

        candidates = []
        for item in ticker:
            market = item.get("market", "")
            # Filter for CoinDCX Spot INR markets (typically prefixed with 'I-' or ending with 'INR')
            if not (market.startswith("I-") or market.endswith("INR")):
                continue

            last_price = float(item.get("last_price", 0.0) or 0.0)
            bid = float(item.get("bid", 0.0) or 0.0)
            ask = float(item.get("ask", 0.0) or 0.0)
            vol = float(item.get("volume", 0.0) or 0.0)
            change_24h = float(item.get("change_24_hour", 0.0) or 0.0)

            if last_price <= 0.0:
                continue

            # Compute spread with strict bid/ask validation (P1-7)
            if ask <= 0.0 or bid <= 0.0 or bid >= ask:
                # Crossed or missing book: ineligible
                spread_pct = None
                is_tight_spread = False
                eligibility_reason = "INVALID_OR_CROSSED_BOOK"
            else:
                midpoint = (ask + bid) / 2.0
                spread_pct = (ask - bid) / midpoint
                is_tight_spread = spread_pct <= self.max_spread_pct
                eligibility_reason = "HIGH_SPREAD" if not is_tight_spread else "OK"

            volume_inr = vol * last_price

            # Liquidity & Quality Checks
            is_liquid = volume_inr >= self.min_24h_volume_inr
            eligible = is_liquid and is_tight_spread and (spread_pct is not None)

            # Generate tags from candle data if available
            df_candles = candle_lookup.get(market)
            if df_candles is not None and not df_candles.empty:
                tags = self.tagger.compute_tags(df_candles)
            else:
                tags = ["LIQUID_SPOT" if eligible else "LOW_LIQUIDITY"]

            # Heuristic ranking scores (P1-8: explicitly labeled as heuristic, not validated forecasts)
            vol_score = min(1.0, volume_inr / 500000.0)
            spread_penalty = max(0.0, 1.0 - ((spread_pct or 0.05) / self.max_spread_pct))
            heuristic_signal_strength = round((0.6 * (change_24h / 100.0) + 0.2 * vol_score + 0.2 * spread_penalty), 4)
            heuristic_downside_risk_pct = round(max(0.02, min(0.12, abs(change_24h * 0.5) / 100.0)), 4)

            candidates.append({
                "market_pair": market,
                "price": last_price,
                "bid": bid,
                "ask": ask,
                "spread_pct": round(spread_pct, 4) if spread_pct is not None else None,
                "change_24h_pct": round(change_24h, 2),
                "volume_24h_inr": round(volume_inr, 2),
                "strategy_tags": tags,
                "heuristic_signal_strength": heuristic_signal_strength,
                "heuristic_downside_risk_pct": heuristic_downside_risk_pct,
                "signal_strength": heuristic_signal_strength, # Kept for backward compatibility
                "downside_risk_pct": heuristic_downside_risk_pct,
                "is_eligible": eligible,
                "eligibility_status": "ELIGIBLE" if eligible else (eligibility_reason if not is_tight_spread else "LOW_VOLUME"),
            })

        # Rank by liquidity and signal strength
        eligible_candidates = [c for c in candidates if c["is_eligible"]]
        ineligible_candidates = [c for c in candidates if not c["is_eligible"]]

        eligible_candidates.sort(key=lambda x: (x["signal_strength"], x["volume_24h_inr"]), reverse=True)
        # Cap at max_scan_candidates (up to 50)
        final_list = eligible_candidates[:self.max_scan_candidates]

        # If fewer than max qualify, append top watch candidates marked INELIGIBLE
        if len(final_list) < self.max_scan_candidates:
            ineligible_candidates.sort(key=lambda x: x["volume_24h_inr"], reverse=True)
            remaining_slots = self.max_scan_candidates - len(final_list)
            final_list.extend(ineligible_candidates[:remaining_slots])

        return final_list
