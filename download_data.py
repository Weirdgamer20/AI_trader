import argparse
import pandas as pd

from src.data import data_quality_report, discover_btc_inr_pair, fetch_candles, save_candles


def main():
    parser = argparse.ArgumentParser(description="Download CoinDCX BTC/INR 15-minute spot candles.")
    parser.add_argument("--pair", default="auto", help="Exact pair ID, or 'auto' to discover active BTC/INR market.")
    parser.add_argument(
        "--start-date",
        default="2010-03-17",
        help="UTC start date for historical data, in YYYY-MM-DD format (CoinDCX earliest candle is 2018-10-18)",
    )
    parser.add_argument("--output", default="data/raw/btc_inr_15m.csv")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output file and download full requested range from scratch",
    )
    args = parser.parse_args()
    pair = discover_btc_inr_pair() if args.pair.lower() == "auto" else args.pair
    print(f"Market pair: {pair}")
    resume_file = None if args.no_resume else args.output
    df = fetch_candles(pair=pair, start_date=args.start_date, resume_path=resume_file)
    save_candles(df, args.output)
    report = data_quality_report(df)
    start = pd.Timestamp(args.start_date, tz="UTC")
    end = pd.Timestamp.now(tz="UTC")
    expected = int((end - start).total_seconds() / (15 * 60))
    print(f"\nSaved {len(df):,} candles to {args.output}")
    print(f"Actual range: {df['time'].min()} -> {df['time'].max()}")
    print(f"Requested range: {start} -> {end}")
    print(f"Expected 15-minute intervals: approximately {expected:,}")
    print(f"Observed coverage by row count: {len(df) / expected:.1%}")
    print("\nData quality report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print("\nNo missing candles are fabricated. Review historical coverage and gaps before training.")


if __name__ == "__main__":
    main()
