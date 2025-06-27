import yfinance as yf
import pandas as pd
import os
import streamlit as st
from lstm_model import fetch_stock_prices, preprocess_data, build_lstm_model, predict_stock_price

# Function to fetch and save stock data
def fetch_and_save_stock_data(stock_symbol):
    csv_file = f'{stock_symbol}.csv'
    
    # Check if the CSV file already exists
    if os.path.exists(csv_file):
        # Load from CSV
        df = pd.read_csv(csv_file)
        print(f"Loaded {stock_symbol} data from CSV file.")
    else:
        # Fetch from yfinance and save to CSV
        stock = yf.Ticker(stock_symbol)
        df = stock.history(period='max')
        df.to_csv(csv_file)
        print(f"Saved {stock_symbol} data to CSV file.")
    
    return df

# Function to validate stock data
def validate_stock_data(df):
    # Add your validation logic (e.g., check date ranges, data completeness)
    if df.empty:
        raise ValueError("Stock data is empty!")
    # You can add more checks here
    return True

# Function to perform stock prediction
def perform_stock_prediction(stock_symbol):
    # Fetch historical data
    stock_data = fetch_stock_prices(stock_symbol)
    
    if stock_data is None or stock_data.empty:
        return None  # Handle cases where no data is available

    # Preprocess data
    closing_prices = stock_data['Close'].values.reshape(-1, 1)
    X_train, y_train, scaler = preprocess_data(closing_prices)

    # Build and train LSTM model
    model = build_lstm_model()
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Make predictions using the model
    X_test = X_train[-60:]  # Use last 60 days data for prediction
    predictions = predict_stock_price(model, X_test, scaler)

    return predictions[-1][0]  # Return the last prediction
