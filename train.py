
# ==========================================
# TESLA STOCK PRICE FORECASTING USING LSTM
# File: train.py
# Purpose: Train the LSTM neural network
# ==========================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential
except ImportError:
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.layers import Dense, Dropout, LSTM
    from keras.models import Sequential

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "preprocessed_data.npz"
MODEL_PATH = BASE_DIR / "models" / "tesla_lstm_model.keras"
PLOT_PATH = BASE_DIR / "results" / "training_loss.png"


def train_model(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    plot_path: str | Path = PLOT_PATH,
):
    """Train an LSTM model using the preprocessed Tesla stock data."""
    data_path = Path(data_path)
    model_path = Path(model_path)
    plot_path = Path(plot_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nLoading preprocessed Tesla data...")
    data = np.load(data_path)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    print("\nBuilding LSTM model...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()

    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    print("\nStarting LSTM training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        shuffle=False,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )

    model.save(model_path)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Tesla LSTM Model Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n================================")
    print("LSTM TRAINING COMPLETED")
    print("================================")
    print("Model saved:", model_path)
    print("Training graph saved:", plot_path)
    return model


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
