
# ==========================================
# TESLA STOCK PRICE FORECASTING USING LSTM
# File: evaluate.py
# Purpose: Evaluate model performance
# ==========================================

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "tesla_stock_data.csv"
PROCESSED_PATH = BASE_DIR / "dataset" / "preprocessed_data.npz"
MODEL_PATH = BASE_DIR / "models" / "tesla_lstm_model.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
RESULTS_DIR = BASE_DIR / "results"


def evaluate_model(
    data_path: str | Path = DATA_PATH,
    processed_path: str | Path = PROCESSED_PATH,
    model_path: str | Path = MODEL_PATH,
    scaler_path: str | Path = SCALER_PATH,
    results_dir: str | Path = RESULTS_DIR,
):
    """Load a trained model, generate predictions, and save evaluation outputs."""
    data_path = Path(data_path)
    processed_path = Path(processed_path)
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading test dataset...")
    data = np.load(processed_path)
    X_test = data["X_test"]
    y_test = data["y_test"]

    print("Loading trained LSTM model...")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    print("\nGenerating predictions...")
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    actual_prices = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(actual_prices, predictions)
    mse = mean_squared_error(actual_prices, predictions)
    rmse = np.sqrt(mse)

    print("\n================================")
    print("MODEL EVALUATION RESULTS")
    print("================================")
    print(f"Mean Absolute Error (MAE):  {mae:.4f}")
    print(f"Mean Squared Error (MSE):   {mse:.4f}")
    print(f"Root Mean Squared Error:    {rmse:.4f}")

    with open(results_dir / "evaluation_metrics.txt", "w") as file:
        file.write("TESLA LSTM MODEL EVALUATION\n")
        file.write("===========================\n")
        file.write(f"MAE: {mae:.4f}\n")
        file.write(f"MSE: {mse:.4f}\n")
        file.write(f"RMSE: {rmse:.4f}\n")

    stock_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    stock_data = stock_data.dropna()
    stock_data = stock_data.apply(pd.to_numeric, errors="coerce")
    stock_data = stock_data.dropna()

    split_index = int(len(stock_data) * 0.80)
    test_dates = stock_data.index[split_index:]

    results = pd.DataFrame(
        {
            "Date": test_dates,
            "Actual Price": actual_prices.flatten(),
            "Predicted Price": predictions.flatten(),
        }
    )

    predictions_path = results_dir / "predictions.csv"
    results.to_csv(predictions_path, index=False)

    plt.figure(figsize=(14, 6))
    plt.plot(actual_prices, label="Actual Tesla Price", linewidth=2)
    plt.plot(predictions, label="Predicted Tesla Price", linewidth=2)
    plt.title("Tesla Stock Price: Actual vs Predicted")
    plt.xlabel("Test Data Records")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n================================")
    print("EVALUATION COMPLETED")
    print("================================")
    print("Metrics saved:", results_dir / "evaluation_metrics.txt")
    print("Predictions saved:", predictions_path)
    print("Graph saved:", results_dir / "actual_vs_predicted.png")
    print("\nModel evaluation completed successfully!")
    return mae, mse, rmse


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()
