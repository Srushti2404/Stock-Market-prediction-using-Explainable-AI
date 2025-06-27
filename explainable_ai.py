import yfinance as yf
import xgboost as xgb
import pandas as pd
from sklearn.metrics import  mean_absolute_percentage_error, mean_squared_error
import shap
import streamlit as st
import os
import numpy as np

# Function to train the XGBoost model
def train_model(stock_symbol):
    csv_file = f'{stock_symbol}.csv'  # File name based on the symbol

    # Check if the file exists
    if os.path.exists(csv_file):
        st.write(f"Loading stock data from existing {csv_file}")
        hist = pd.read_csv(csv_file)
    else:
        st.write(f"Fetching stock data for {stock_symbol} and saving to {csv_file}")
        stock_data = yf.Ticker(stock_symbol)
        hist = stock_data.history(period="5y", interval="1d")
        if hist.empty:
            raise ValueError(f"No historical data found for stock symbol(in explainable ai): {stock_symbol}")
        hist.to_csv(csv_file, index=True)  # Save with index

    # Convert necessary columns to numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        hist[col] = pd.to_numeric(hist[col], errors='coerce')
    
    hist.fillna(method='ffill', inplace=True)  # Handle missing values
    hist.replace([np.inf, -np.inf], np.nan, inplace=True)  # Remove infinite values
    hist.dropna(inplace=True)  # Drop any remaining NaN values
    
    hist['Returns'] = hist['Close'].pct_change().fillna(0)
    X = hist[['Open', 'High', 'Low', 'Volume']].fillna(0)
    y = hist['Close']

    # Train an XGBoost regressor
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, enable_categorical=True)
    model.fit(X, y)

    # Test the model accuracy (RMSE)
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    st.write(f"Model RMSE: {rmse}")

    return model, X, y

# Function to explain the predictions
def explain_model(stock_symbol):
    try:
        model, X, y = train_model(stock_symbol)
        if X is None or X.empty:
            raise ValueError("Feature dataset is empty after processing.")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if shap_values is None:
            raise ValueError("SHAP values computation failed.")

        # For debugging, show the shape of SHAP values and X
        st.write(f"SHAP Values Shape: {shap_values.shape}, Features Shape: {X.shape}")
        return model, shap_values, X
    except Exception as e:
        st.error(f"Error in explain_model: {e}")
        return None, None, None

# Function to evaluate the model performance
def evaluate_model_performance(model, X, y):
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mape = mean_absolute_percentage_error(y, predictions) * 100  # Convert to percentage

    return rmse, mape

def main():
    st.title("Stock Price Prediction using Explainable AI")  # Update title
    
    stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
    
    if st.button("Fetch Data and Train XGBoost"):
        # Fetch stock data and train XGBoost
        try:
            model, X, y = train_model(stock_symbol)
            st.write(f"XGBoost model trained for {stock_symbol}")
            rmse, mape = evaluate_model_performance(model, X, y)
            st.write(f"RMSE: {rmse}, MAPE: {mape:.2f}%")
        except ValueError as e:
            st.error(f"Error: {e}")

    if st.button("Explain Predictions"):
        # Explain the predictions
        try:
            model, shap_values, X = explain_model(stock_symbol)
            if model is not None and shap_values is not None:
                st.write(f"Explanations for {stock_symbol} predictions:")
                # Display explanations (you'll need to add your SHAP visualizations here)
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
