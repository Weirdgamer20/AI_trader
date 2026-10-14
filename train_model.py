from pathlib import Path

import pandas as pd

from src.modeling import fit_and_evaluate


def main():
    path = Path("data/raw/btc_inr_15m.csv")
    if not path.exists():
        raise SystemExit("No candle data. Run `python download_data.py --pair auto --start-date 2018-10-18` first.")
    raw = pd.read_csv(path, parse_dates=["time"])
    metrics, _, _, _ = fit_and_evaluate(raw)
    print("\nChronological out-of-sample metrics (newest 15% untouched during model selection):")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nSaved models and test predictions under ./models/")
    print("These metrics are research diagnostics, not evidence of profitability.")


if __name__ == "__main__":
    main()
