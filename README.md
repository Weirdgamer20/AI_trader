# BTC/INR ML Predictor — CoinDCX Data + OCR (Windows 11 & WSL)

A complete research, prediction, and visual evaluation system for 15-minute BTC/INR candles using historical market data, technical indicators, machine learning, and chart-screenshot OCR.

---

## Important Boundaries & Scope

- **Non-Custodial / Read-Only**: CoinDCX remains your exchange and custodian. This application uses **public market data endpoints only**. It never asks for API keys, accesses wallet balances, or places trades.
- **Probabilistic Estimates**: Predictions are statistical estimates with uncertainty. High backtest scores are not guarantees of future profitability.
- **Genuine vs. Reconstructed History**: CoinDCX's active spot pair is **`I-BTC_INR`**, with historical candles dating back to **18 October 2018 08:00:00 UTC**. Earlier dates (2010–2018) require secondary BTC/USD reference data, which must be stored separately and labeled as reconstructed data.
- **No Synthesized Market Data**: The downloader records genuine exchange responses and reports missing intervals and coverage truthfully. It never invents candles.

---

## Deliverables Architecture

1. **Market Data Engine (`src/data.py`)**:
   - Dynamic pair discovery (`I-BTC_INR`).
   - Forward chunk pagination (999 intervals / ~10.4 days per chunk) preventing silent reverse truncation.
   - Interruption resumption: automatically picks up from the latest saved candle (`max(time) + 15m`).
   - Comprehensive data quality diagnostics (gaps >22.5m, missing intervals, span coverage percentage, deduplication).
   - Provenance-safe secondary BTC/USD ingestion (`ingest_secondary_btcusd`).
2. **Feature Engineering (`src/features.py`)**:
   - Lagged returns (1, 2, 4, 8, 16), candle range & body fractions, upper/lower wicks, volume change, rolling volatility (8, 32), SMA/EMA gaps, RSI-14, and MACD.
   - Strict temporal causality: feature row $t$ depends strictly on candle $t$ and earlier; targets use next candle ($t+1$).
3. **Machine Learning Pipeline (`src/modeling.py`)**:
   - Chronological splits: oldest 70% train, next 15% validation, newest 15% out-of-sample test.
   - Classification candidates: Logistic Regression, Random Forest, and Histogram Gradient Boosting.
   - Regression: Histogram Gradient Boosting regressor.
   - Baselines: Majority class direction baseline and Naive persistent close baseline ($C_{t+1} \approx C_t$).
   - Metrics: Direction Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Price MAE, RMSE, and MAPE.
4. **Chart OCR Engine (`src/ocr_utils.py`)**:
   - Preprocessing with contrast enhancement and sharpening.
   - Automated candidate extraction: prices (INR), timestamps, and currency tags.
   - Tesseract per-word mean confidence metric.
   - Sandbox verification: display-only; prevents OCR text from contaminating verified OHLCV datasets.
5. **Evaluation Dashboard (`app.py`)**:
   - Interactive Streamlit app with Market Overview, Live Candlestick/Line charts, Forecast Bridge, Out-of-Sample Holdout Evaluation, Confusion Matrix, OCR Workspace, and CSV/JSON Export facilities.

---

## Setup & Running

### Option A: Running in WSL (Ubuntu)
```bash
# 1. Open WSL
wsl

# 2. Navigate to project folder
cd /mnt/d/btc_inr_ml/btc_inr_ml

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Ingest CoinDCX candles (resumes automatically if data exists)
python download_data.py --pair auto --start-date 2018-10-18

# 5. Train baseline & advanced models
python train_model.py

# 6. Launch the Streamlit dashboard
streamlit run app.py
```

### Option B: Running natively on Windows 11
```powershell
# 1. Open PowerShell in D:\btc_inr_ml\btc_inr_ml
cd D:\btc_inr_ml\btc_inr_ml

# 2. Create virtual environment (Python 3.12+ recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download and train
python download_data.py --pair auto --start-date 2018-10-18
python train_model.py

# 5. Launch the dashboard
streamlit run app.py
```

---

## CLI Options

### `download_data.py`
```text
options:
  --pair PAIR             Exact pair ID, or 'auto' to resolve active BTC/INR market (default: auto)
  --start-date START_DATE UTC start date in YYYY-MM-DD format (default: 2010-03-17; CoinDCX starts 2018-10-18)
  --output OUTPUT         Destination CSV path (default: data/raw/btc_inr_15m.csv)
  --no-resume             Force full download from scratch instead of resuming
```

---

## Automated Test Suite

Run the automated regression test suite:
```bash
python -m pytest -v
```

Tests verify:
- Feature finiteness and column schema.
- Forecast generation on unlabeled live rows.
- Model persistence and bundle reloading.
- Interval gap detection and coverage percentage.
- Start-date validation (future date rejection, dates older than 10 years).
- Resumption from existing local datasets.
- Clean secondary BTC/USD ingestion and labeling.
- OCR price, timestamp, and confidence extraction.
