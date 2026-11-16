
# ==========================================
# TESLA STOCK PRICE FORECASTING USING LSTM
# File: preprocessing.py
# Purpose: Clean, scale and prepare stock data
# ==========================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Define paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "tesla_stock_data.csv"
MODEL_PATH = BASE_DIR / "models" / "scaler.pkl"
OUTPUT_PATH = BASE_DIR / "dataset" / "preprocessed_data.npz"


def preprocess_stock_data(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    output_path: str | Path = OUTPUT_PATH,
):
    """Load, preprocess, and split the Tesla stock data for LSTM training."""
    data_path = Path(data_path)
    model_path = Path(model_path)
    output_path = Path(output_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nLoading Tesla stock dataset...")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data = data.dropna()

    if "Close" not in data.columns:
        raise ValueError("Closing price column not found in dataset.")

    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna()

    closing_prices = data[["Close"]].values

    print("Dataset loaded successfully!")
    print("Total valid records:", len(closing_prices))

    split_index = int(len(closing_prices) * 0.80)
    train_data = closing_prices[:split_index]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit(train_data).transform(closing_prices)
    joblib.dump(scaler, model_path)

    TIME_STEPS = 60
    X_train, y_train = [], []
    X_test, y_test = [], []

    for i in range(TIME_STEPS, len(scaled_data)):
        sequence = scaled_data[i - TIME_STEPS:i]
        target = scaled_data[i]

        if i < split_index:
            X_train.append(sequence)
            y_train.append(target)
        else:
            X_test.append(sequence)
            y_test.append(target)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    np.savez(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    print("\n================================")
    print("PREPROCESSING COMPLETED")
    print("================================")
    print("Training input shape:", X_train.shape)
    print("Training output shape:", y_train.shape)
    print("Testing input shape:", X_test.shape)
    print("Testing output shape:", y_test.shape)
    print("\nScaler saved:", model_path)
    print("Processed data saved:", output_path)
    print("\nData preprocessing completed successfully!")
    return X_train, y_train, X_test, y_test


def main() -> None:
    preprocess_stock_data()


if __name__ == "__main__":
    main()
