import json
import pandas as pd
from src.modeling import load_models, predict_next

def main():
    df = pd.read_csv("data/raw/btc_inr_15m.csv", parse_dates=["time"])
    direction_model, price_model = load_models("models")
    prediction = predict_next(df, direction_model, price_model)
    print("=== LIVE PREDICTION ===")
    print(json.dumps(prediction, indent=2))

if __name__ == "__main__":
    main()
