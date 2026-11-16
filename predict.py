
# ==========================================
# TESLA STOCK PRICE FORECASTING USING LSTM
# File: predict.py
# Purpose: Predict the next closing price
# ==========================================

from pathlib import Path

import joblib
import pandas as pd

try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "tesla_stock_data.csv"
MODEL_PATH = BASE_DIR / "models" / "tesla_lstm_model.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
OUTPUT_PATH = BASE_DIR / "results" / "next_day_prediction.csv"


def predict_next_closing_price(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    scaler_path: str | Path = SCALER_PATH,
    output_path: str | Path = OUTPUT_PATH,
):
    """Use the last 60 closing prices to predict the next one-day close."""
    data_path = Path(data_path)
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nLoading Tesla stock data...")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data = data.dropna()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna()

    closing_prices = data[["Close"]].values
    if len(closing_prices) < 60:
        raise ValueError("Not enough data for prediction.")

    print("Loading trained LSTM model...")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    TIME_STEPS = 60
    last_60_prices = closing_prices[-TIME_STEPS:]
    scaled_input = scaler.transform(last_60_prices)
    X_input = scaled_input.reshape(1, TIME_STEPS, 1)

    print("\nGenerating next-observation forecast...")
    predicted_scaled = model.predict(X_input, verbose=0)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

    last_date = data.index[-1]
    last_close = closing_prices[-1][0]

    print("\n================================")
    print("TESLA STOCK PRICE FORECAST")
    print("================================")
    print("Last available date:", last_date.date())
    print(f"Last closing price: ${last_close:.2f}")
    print(f"Forecast next closing price: ${predicted_price:.2f}")

    prediction_result = pd.DataFrame(
        {
            "Last Available Date": [last_date.date()],
            "Last Closing Price": [last_close],
            "Forecast Next Closing Price": [predicted_price],
        }
    )
    prediction_result.to_csv(output_path, index=False)

    print("\nPrediction saved:", output_path)
    print("\nPrediction completed successfully!")
    return prediction_result


def main() -> None:
    predict_next_closing_price()


if __name__ == "__main__":
    main()
