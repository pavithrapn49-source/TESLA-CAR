# Tesla Stock Price Forecasting Using LSTM

## 1. Project Overview

This project focuses on predicting Tesla (TSLA) stock closing prices using a Long Short-Term Memory (LSTM) neural network.

LSTM is a type of Recurrent Neural Network (RNN) designed to learn patterns and dependencies in sequential data.

The project uses historical Tesla stock market data to train an LSTM model and generate future one-step forecasts.

## 2. Objectives

- Collect historical Tesla stock market data.
- Clean and preprocess the dataset.
- Apply feature scaling using MinMaxScaler.
- Create time-series sequences for LSTM training.
- Build and train an LSTM neural network.
- Evaluate model performance using MAE, MSE and RMSE.
- Generate actual-versus-predicted visualizations.
- Predict the next available stock-price observation.

## 3. Technologies Used

- Python 3.12 (recommended)
- TensorFlow 2.16+ on Python 3.12
- Keras + JAX fallback for newer Python versions (for example Python 3.15)
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- yfinance
- Joblib

## 4. Dataset

The dataset contains historical Tesla stock market information downloaded using the Yahoo Finance Python library (`yfinance`).

**Stock Symbol:** TSLA

**Data Period:** January 2015 – November 2026

**Features:**
- Open
- High
- Low
- Close
- Volume

The closing price is selected as the target variable for prediction.

## 5. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded historical stock data.
2. Removed missing values.
3. Converted data into numeric format.
4. Selected the closing price.
5. Split the dataset chronologically into 80% training and 20% testing data.
6. Applied MinMaxScaler to normalize the values between 0 and 1.
7. Created sequences using the previous 60 observations to predict the next observation.

The scaler was fitted only on the training data to avoid data leakage.

## 6. LSTM Model Architecture

The model consists of:

- First LSTM layer with 50 units.
- Dropout layer with 20% dropout.
- Second LSTM layer with 50 units.
- Dropout layer with 20% dropout.
- Dense layer with 25 units.
- Output Dense layer with 1 unit.

**Optimizer:** Adam

**Loss Function:** Mean Squared Error

**Training Epochs:** 50

**Batch Size:** 32

Early stopping and model checkpointing were used during training.

## 7. Model Evaluation

The model was evaluated using the testing dataset.

| Evaluation Metric | Result |
|---|---:|
| MAE | 36.7192 |
| MSE | 1810.4856 |
| RMSE | 42.5498 |

### Metric Explanation

- **MAE:** Measures the average absolute difference between actual and predicted prices.
- **MSE:** Measures the average squared prediction error.
- **RMSE:** Measures prediction error in the original stock-price unit, USD.

The MAE indicates an average absolute prediction error of approximately $36.72 on the test dataset.

## 8. Next Observation Prediction

The trained model uses the latest 60 available closing-price observations to generate a one-step forecast.

Latest available closing price: $390.22

Model forecast: $345.30

This forecast is an experimental output and does not guarantee actual future stock performance.

## 9. Project Structure

```text
Tesla_Stock_Prediction/
│
├── dataset/
│   ├── tesla_stock_data.csv
│   └── preprocessed_data.npz
│
├── models/
│   ├── scaler.pkl
│   └── tesla_lstm_model.keras
│
├── results/
│   ├── training_loss.png
│   ├── actual_vs_predicted.png
│   ├── evaluation_metrics.txt
│   ├── predictions.csv
│   └── next_day_prediction.csv
│
├── data_collection.py
├── preprocessing.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
└── README.md
```

## 10. How to Run the Project

### Step 1: Clone or open the project

Open the project folder in Visual Studio Code.

### Step 2: Create a virtual environment

```bash
python3.12 -m venv venv
```

### Step 3: Activate the environment

```bash
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Collect stock data

```bash
python data_collection.py
```

### Step 6: Preprocess the dataset

```bash
python preprocessing.py
```

### Step 7: Train the LSTM model

```bash
python train.py
```

### Step 8: Evaluate the model

```bash
python evaluate.py
```

### Step 9: Generate a prediction

```bash
python predict.py
```

## 11. Results and Observations

The LSTM model successfully learned patterns from historical Tesla closing-price data.

The training and validation loss graph demonstrates how the model loss changed during training.

The actual-versus-predicted graph provides a visual comparison between the model predictions and actual test-period prices.

The evaluation metrics show that the model still has prediction errors. Further tuning and comparison with other forecasting methods may improve performance.

## 12. Limitations and Future Improvements

- Include additional market indicators and technical features.
- Experiment with different LSTM layer sizes and sequence lengths.
- Tune hyperparameters to improve model performance.
- Compare LSTM with GRU and other forecasting algorithms.
- Include market news and sentiment data.
- Evaluate performance against simple baseline forecasting methods.

## 13. Conclusion

This project demonstrates the application of deep learning and LSTM neural networks to time-series stock-price forecasting.

It covers the complete machine-learning workflow, including data collection, preprocessing, model training, evaluation, visualization and prediction.

The project provides practical experience in Python, data analysis, deep learning and time-series forecasting.

---

**Disclaimer:** This project is developed for educational purposes only. Predictions are experimental and should not be considered financial or investment advice.
 
 licence Pavithrapn49-source