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

---

## Multi-Coin AI Spot Trading Platform (Milestones M1–M8)

A production-grade, risk-controlled spot trading architecture implementing the complete project specification:

### Milestone Architecture
1. **M1 — Ledger, Accounting & Indian Tax Engine (`src/ledger/`)**:
   - Double-entry portfolio accounting starting from ₹500 aiming for ₹100,000 target.
   - CoinDCX fee schedule (0.50% maker/taker), 18% GST on fees.
   - Section 194S 1% TDS deduction and Section 115BBH 30% flat VDA tax calculation (no loss set-off).
   - Relational database persistence supporting PostgreSQL and local transactional SQLite.
2. **M2 — Deterministic Risk Engine & Execution State Machine (`src/risk/`, `src/execution/`)**:
   - Strictly enforces maximum 2.0% planned risk per trade (e.g. ₹10 at ₹500 balance).
   - ₹250 hard affordable-loss ceiling tracker with automated trading halt.
   - Execution State Machine: `PROPOSED` → `RISK_CHECKED` → `AUTHORIZED` → `SUBMITTED` → `FILLED` → `RECONCILED`.
   - Emergency stop circuit breaker and safe mode.
   - Strict `LiveExecutionGuard` disabling live execution by default (REQ-28).
3. **M3 — Dynamic Coin Scanner & Multi-Horizon Forecasting (`src/scanner/`, `src/modeling_multicoin.py`)**:
   - Dynamically scans up to 50 eligible CoinDCX Spot candidates based on 24h volume and spread.
   - Behavioral market tagger (regime, trend, mean reversion, volatility tags).
   - Multi-horizon forecasting (1, 2, 4 candles) with wicks, direction, return, downside, and 90% calibrated uncertainty intervals.
4. **M4 — Mandatory Meta-Learning & Independent Strategy Critic (`src/meta_learning/`)**:
   - Meta-feature extraction without lookahead bias.
   - Meta-learner dynamically selecting optimal learner algorithm and horizon for active market conditions.
   - Champion/Challenger model registry with gated promotion and audit-logged rollback.
   - Independent Strategy Critic vetoing proposals with excessive uncertainty or weak returns.
5. **M5 — Paper Execution Engine & Divergence Monitor (`src/execution/`)**:
   - High-fidelity simulated execution against candle range with realistic volume-weighted slippage.
   - Backtest-to-paper performance divergence monitoring and alerting.
6. **M6 — Extensible Developer Mode, Terminal Dashboard & Chrome Overlay (`src/ui/`, `extension/`)**:
   - Interactive terminal dashboard and enhanced multi-tab Streamlit terminal.
   - Sandboxed plugin architecture with strict permission isolation.
   - Telegram notification dispatcher with automated secret redaction.
   - Click-through Chrome overlay (`extension/`) for non-interfering HUD on CoinDCX.
7. **M7/M8 — Security, Provenance & Release Qualification (`src/security/`, `tests/`)**:
   - Credential vault with recursive secret masking.
   - Provenance memory logger distinguishing facts, calculations, hypotheses, and configurations.
   - Full end-to-end regression test suite covering all requirements (REQ-01 to REQ-30).

### Quick CLI Usage
```bash
# View live terminal dashboard with active CoinDCX Spot scanner:
python run_system.py --status

# Scan up to 50 active Spot markets:
python run_system.py --scan

# Trigger emergency stop circuit breaker:
python run_system.py --emergency-stop

# Clear emergency stop with authorization:
python run_system.py --clear-stop "AUTHORIZED_LIVE_USER_CONFIRMED_2026"
```

### Full Automated Test Suite (All 36 Tests Passing)
```bash
python -m pytest -v
```
