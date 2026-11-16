
# ==========================================
# TESLA STOCK PRICE FORECASTING USING LSTM
# File: data_collection.py
# Purpose: Download historical Tesla stock data
# ==========================================

from pathlib import Path

import pandas as pd
import yfinance as yf

# Step 1: Define stock information
STOCK_SYMBOL = "TSLA"
START_DATE = "2015-01-01"
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_PATH = DATASET_DIR / "tesla_stock_data.csv"


def download_stock_data(
    stock_symbol: str = STOCK_SYMBOL,
    start_date: str = START_DATE,
    output_path: str | Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Download Tesla stock data and save it to the project dataset directory."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("\nDownloading Tesla stock data...")

    data = yf.download(
        stock_symbol,
        start=start_date,
        auto_adjust=True,
        progress=True,
    )

    if data.empty:
        raise ValueError(
            "No stock data downloaded. Please check your internet connection."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()
    data.to_csv(output_file, index_label="Date")

    print("\n================================")
    print("TESLA DATA DOWNLOADED SUCCESSFULLY")
    print("================================")
    print("\nTotal Records:", len(data))
    print("Dataset Location:", output_file)
    print("\nAvailable Columns:")
    print(data.columns.tolist())
    print("\nFirst 5 Records:")
    print(data.head())
    print("\nLast 5 Records:")
    print(data.tail())
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nData collection completed!")
    return data


def main() -> None:
    download_stock_data()


if __name__ == "__main__":
    main()
