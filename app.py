import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import data_quality_report, discover_btc_inr_pair, fetch_candles, save_candles
from src.modeling import fit_and_evaluate, load_models, predict_next
from src.ocr_utils import extract_chart_data

DATA_PATH = Path("data/raw/btc_inr_15m.csv")
MODEL_DIR = Path("models")

st.set_page_config(
    page_title="BTC/INR ML Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("BTC/INR ML Predictor")
st.caption("CoinDCX 15-Minute Spot Candle Research · Next-Candle Forecasts · Evaluated Holdouts · OCR Verification")

st.info(
    "**Boundary Notice**: Research & prediction tool only. Predictions are probabilistic model estimates and "
    "never guarantee market outcomes. CoinDCX holds exchange custody; this system does not store API secrets, "
    "read account balances, or execute live orders.",
    icon="ℹ️",
)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Market Data Engine")
    pair_input = st.text_input(
        "Market Pair",
        value="auto",
        help="Use 'auto' to resolve active BTC/INR spot market (e.g. I-BTC_INR on CoinDCX).",
    )
    start_date_input = st.date_input(
        "Historical Target Start",
        value=pd.to_datetime("2018-10-18"),
        help="CoinDCX's earliest available BTC/INR candle is 2018-10-18. Earlier dates return empty sets.",
    )
    force_fresh = st.checkbox("Download from scratch (no resumption)", value=False)

    if st.button("📥 Download / Refresh Candles", type="primary", use_container_width=True):
        try:
            with st.spinner("Connecting to CoinDCX and ingesting 15m candles..."):
                pair = discover_btc_inr_pair() if pair_input.strip().lower() == "auto" else pair_input.strip()
                resume_file = None if force_fresh else str(DATA_PATH)
                candles = fetch_candles(
                    pair=pair,
                    start_date=str(start_date_input),
                    resume_path=resume_file,
                )
                save_candles(candles, str(DATA_PATH))
                is_resumed = candles.attrs.get("resumed", False)
                mode_str = "Incrementally updated" if is_resumed else "Downloaded"
                st.session_state["data_message"] = f"✅ {mode_str} {len(candles):,} candles for {pair}."
                st.session_state.pop("data_error", None)
        except Exception as exc:
            st.session_state["data_error"] = f"Failed to ingest: {type(exc).__name__}: {exc}"

    if st.session_state.get("data_message"):
        st.success(st.session_state["data_message"])
    if st.session_state.get("data_error"):
        st.error(st.session_state["data_error"])

    st.divider()
    st.markdown("### 💾 Local State")
    if DATA_PATH.exists():
        st.caption(f"📁 Local storage: `{DATA_PATH}` ({DATA_PATH.stat().st_size / (1024*1024):.2f} MB)")
    if (MODEL_DIR / "direction_model.joblib").exists():
        st.caption("🤖 Trained models available in `./models`")

# ----------------- DATA LOADING -----------------
if not DATA_PATH.exists():
    st.warning("No candle dataset found locally. Use **Download / Refresh Candles** in the sidebar to ingest history.")
    st.code("python download_data.py --pair auto --start-date 2018-10-18\npython train_model.py\nstreamlit run app.py", language="bash")
    st.stop()

raw = pd.read_csv(DATA_PATH, parse_dates=["time"])
required = {"time", "open", "high", "low", "close", "volume"}
if not required.issubset(raw.columns):
    st.error(f"Dataset corrupted or missing required columns: {sorted(required - set(raw.columns))}")
    st.stop()

raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
for col in ["open", "high", "low", "close", "volume"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

raw = raw.dropna(subset=list(required)).drop_duplicates("time").sort_values("time").reset_index(drop=True)
if len(raw) < 100:
    st.error(f"Dataset too small ({len(raw)} candles). Minimum 100 rows required.")
    st.stop()

quality = data_quality_report(raw)

# ----------------- TOP METRICS BANNER -----------------
latest_close = raw["close"].iloc[-1]
first_close = raw["close"].iloc[0]
change_24h_candles = min(96, len(raw)) # 96 * 15m = 24h
day_change_pct = ((latest_close / raw["close"].iloc[-change_24h_candles]) - 1) * 100

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Latest BTC/INR", f"₹{latest_close:,.2f}", f"{day_change_pct:+.2f}% 24h")
m2.metric("Total Candles", f"{len(raw):,}")
m3.metric("Coverage in Span", f"{quality.get('observed_span_coverage_pct', 0.0):.1f}%")
m4.metric("Earliest Candle", str(raw["time"].min())[:16])
m5.metric("Latest Candle", str(raw["time"].max())[:16])

# ----------------- TABS -----------------
tab_market, tab_predict, tab_eval, tab_ocr, tab_data = st.tabs([
    "📈 Market Overview",
    "🔮 Next-Candle Forecast",
    "📊 Model Evaluation",
    "🔍 OCR Chart Inspector",
    "📋 Data Quality & Export",
])

# 1. MARKET OVERVIEW
with tab_market:
    st.subheader("BTC/INR Spot Price Chart (15-Minute Candles)")
    chart_col1, chart_col2 = st.columns([1, 3])
    with chart_col1:
        chart_mode = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
        display_n = st.slider("Visible candles", min_value=50, max_value=min(3000, len(raw)), value=min(300, len(raw)), step=50)

    sub_df = raw.tail(display_n)
    fig = go.Figure()
    if chart_mode == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=sub_df["time"], open=sub_df["open"], high=sub_df["high"],
            low=sub_df["low"], close=sub_df["close"], name="BTC/INR 15m"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=sub_df["time"], y=sub_df["close"], mode="lines", name="Close",
            line=dict(color="#f7931a", width=2)
        ))
    fig.update_layout(
        height=550,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Timestamp (IST)",
        yaxis_title="Price (₹ INR)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

# 2. NEXT-CANDLE FORECAST
with tab_predict:
    st.subheader("Next 15-Minute Candle Prediction")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        train_trigger = st.button("🚀 Train & Evaluate Models (Chronological)", type="primary", use_container_width=True)
    with btn_col2:
        predict_trigger = st.button("⚡ Generate Forecast with Saved Models", use_container_width=True)

    if train_trigger:
        try:
            with st.spinner("Training baseline, Random Forest, and Gradient Boosting models..."):
                metrics, test_res, clf, reg = fit_and_evaluate(raw, str(MODEL_DIR))
                pred = predict_next(raw, clf, reg)
                st.session_state["model_metrics"] = metrics
                st.session_state["test_results"] = test_res
                st.session_state["next_prediction"] = pred
                st.success(f"Models successfully trained and saved! Best classifier: **{metrics.get('selected_classifier')}**")
        except Exception as exc:
            st.error(f"Training failed: {type(exc).__name__}: {exc}")

    if predict_trigger:
        try:
            clf, reg = load_models(str(MODEL_DIR))
            pred = predict_next(raw, clf, reg)
            st.session_state["next_prediction"] = pred
            st.success("Loaded saved models and generated forecast.")
        except Exception as exc:
            st.error(f"Failed to generate prediction: {type(exc).__name__}: {exc}")

    pred = st.session_state.get("next_prediction")
    if pred:
        st.markdown("---")
        p1, p2, p3, p4 = st.columns(4)
        dir_color = "🟢" if pred["direction"] == "UP" else "🔴"
        p1.metric("Predicted Direction", f"{dir_color} {pred['direction']}", f"{pred['probability_up']:.1%} probability UP")
        p2.metric("Estimated Next Close", f"₹{pred['predicted_next_close']:,.2f}", f"{pred['predicted_change_pct']:+.3f}%")
        p3.metric("Expected Price Move", f"₹{pred['predicted_change_inr']:+,.2f}")
        p4.metric("Last Completed Close", f"₹{pred['last_close']:,.2f}")

        st.caption(
            f"📅 Base candle: **{pred['candle_time']}** | Forecast Horizon: **Next 15 minutes** | "
            f"Generated at: `{pred.get('prediction_timestamp_utc', 'N/A')}`"
        )

        # Visual forecast bridge
        bridge_df = raw.tail(80)
        f_fig = go.Figure()
        f_fig.add_trace(go.Scatter(
            x=bridge_df["time"], y=bridge_df["close"], mode="lines+markers",
            name="Observed Closes", line=dict(color="#00d2be", width=2)
        ))
        f_fig.add_trace(go.Scatter(
            x=[bridge_df["time"].iloc[-1], bridge_df["time"].iloc[-1] + pd.Timedelta(minutes=15)],
            y=[bridge_df["close"].iloc[-1], pred["predicted_next_close"]],
            mode="lines+markers", name="Forecast", line=dict(color="#f39c12", width=3, dash="dash"),
            marker=dict(size=[6, 12], color="#f39c12", symbol=["circle", "star"])
        ))
        f_fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Time",
            yaxis_title="BTC/INR (₹)",
            template="plotly_dark",
        )
        st.plotly_chart(f_fig, use_container_width=True)

# 3. MODEL EVALUATION
with tab_eval:
    st.subheader("Strict Chronological Out-of-Sample Holdout")
    metrics = st.session_state.get("model_metrics")
    if not metrics and (MODEL_DIR / "metrics.json").exists():
        try:
            metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            metrics = None

    if metrics:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Selected Classifier", metrics.get("selected_classifier", "N/A"))
        e2.metric("Test Direction Accuracy", f"{metrics.get('test_accuracy', 0):.2%}", f"Baseline: {metrics.get('test_majority_direction_baseline_accuracy', 0):.2%}")
        e3.metric("Test Close MAE", f"₹{metrics.get('test_close_mae_inr', 0):,.2f}", f"Naive: ₹{metrics.get('naive_close_baseline_mae_inr', 0):,.2f}")
        e4.metric("Close MAPE", f"{metrics.get('test_close_mape_pct', 0):.2f}%", f"Naive: {metrics.get('naive_close_baseline_mape_pct', 0):.2f}%")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Classification Metrics")
            clf_df = pd.DataFrame({
                "Metric": ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Majority Baseline"],
                "Value": [
                    f"{metrics.get('test_accuracy', 0):.4f}",
                    f"{metrics.get('test_balanced_accuracy', 0):.4f}",
                    f"{metrics.get('test_precision', 0):.4f}",
                    f"{metrics.get('test_recall', 0):.4f}",
                    f"{metrics.get('test_f1', 0):.4f}",
                    f"{metrics.get('test_roc_auc') or 0:.4f}",
                    f"{metrics.get('test_majority_direction_baseline_accuracy', 0):.4f}",
                ]
            })
            st.table(clf_df)

            cm = metrics.get("test_confusion_matrix")
            if cm and len(cm) == 2:
                st.markdown("#### Confusion Matrix")
                cm_df = pd.DataFrame(cm, index=["Actual DOWN", "Actual UP"], columns=["Pred DOWN", "Pred UP"])
                st.table(cm_df)

        with col_r:
            st.markdown("#### Regression Error Comparison")
            reg_df = pd.DataFrame({
                "Model": ["Machine Learning (HistGB)", "Naive Benchmark (Next = Current)"],
                "MAE (INR)": [f"₹{metrics.get('test_close_mae_inr', 0):,.2f}", f"₹{metrics.get('naive_close_baseline_mae_inr', 0):,.2f}"],
                "RMSE (INR)": [f"₹{metrics.get('test_close_rmse_inr', 0):,.2f}", f"₹{metrics.get('naive_close_baseline_rmse_inr', 0):,.2f}"],
                "MAPE (%)": [f"{metrics.get('test_close_mape_pct', 0):.2f}%", f"{metrics.get('naive_close_baseline_mape_pct', 0):.2f}%"],
            })
            st.table(reg_df)

            st.markdown("#### Validation Benchmark Scores")
            st.json(metrics.get("validation_balanced_accuracy_by_model", {}))

        test_pred_path = MODEL_DIR / "test_predictions.csv"
        if test_pred_path.exists():
            tdf = pd.read_csv(test_pred_path)
            st.markdown("#### Out-of-Sample Predictions vs Actual Outcomes")
            st.dataframe(tdf.tail(100), use_container_width=True)
            st.download_button(
                "📥 Download Out-of-Sample Predictions CSV",
                tdf.to_csv(index=False),
                file_name="test_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("No trained models found. Train models under the **Next-Candle Forecast** tab to view out-of-sample evaluation.")

# 4. OCR CHART INSPECTOR
with tab_ocr:
    st.subheader("Chart Screenshot OCR & Price Extraction")
    st.markdown(
        "Upload a chart screenshot to extract visible price labels and timestamps. "
        "**Safety Policy**: OCR results are display-only and cannot silently contaminate or alter the exchange candle dataset."
    )

    uploaded_img = st.file_uploader("Upload chart image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_img is not None:
        col_img, col_out = st.columns([1, 1])
        with col_img:
            st.image(uploaded_img, caption="Target Screenshot", use_container_width=True)

        with col_out:
            if st.button("🔍 Extract Text & Detect Prices", type="primary", use_container_width=True):
                with st.spinner("Processing image and running Tesseract OCR..."):
                    try:
                        ocr_data = extract_chart_data(uploaded_img)
                        st.session_state["ocr_data"] = ocr_data
                    except Exception as exc:
                        st.error(f"OCR failed: {type(exc).__name__}: {exc}")

            ocr_res = st.session_state.get("ocr_data")
            if ocr_res:
                st.metric("OCR Mean Confidence", f"{ocr_res.get('confidence_pct', 0)}%")
                if ocr_res.get("parsed_prices"):
                    st.markdown("**Detected Candidate Prices:**")
                    for p in ocr_res["parsed_prices"]:
                        st.write(f"- `{p['raw']}` → **₹{p['value']:,.2f}**")
                if ocr_res.get("parsed_timestamps"):
                    st.markdown(f"**Detected Timestamps:** {', '.join(ocr_res['parsed_timestamps'])}")

                user_correction = st.text_area(
                    "Manual Inspection & Correction Sandbox (Display Only)",
                    value=ocr_res.get("raw_text", ""),
                    height=180,
                )
                st.caption("Editing here does not write to historical market databases.")

# 5. DATA QUALITY & EXPORT
with tab_data:
    st.subheader("Dataset Integrity & Coverage Diagnostics")
    st.json(quality)

    st.markdown("#### Recent 100 Ingested Candles")
    st.dataframe(raw.tail(100), use_container_width=True)

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.download_button(
            "📥 Export Full Ingested Candles (CSV)",
            raw.to_csv(index=False),
            file_name="btc_inr_15m.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d_col2:
        st.download_button(
            "📥 Export Quality Report (JSON)",
            json.dumps(quality, indent=2),
            file_name="data_quality_report.json",
            mime="application/json",
            use_container_width=True,
        )

st.divider()
st.caption(
    "BTC/INR ML Predictor · Built for local execution on Windows 11 / WSL · "
    "Chronological holdout validation · Deterministic feature pipeline"
)
