import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import shap
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import custom modules
from sentiment_analysis import perform_sentiment_analysis
from explainable_ai import explain_model
from lstm_model import fetch_stock_prices, preprocess_data, build_lstm_model, predict_stock_price

# Initialize session state variables
if "validated" not in st.session_state:
    st.session_state.validated = False  # Track if ticker is validated
if "stock_symbol" not in st.session_state:
    st.session_state.stock_symbol = ""  # Store selected ticker
if "processing" not in st.session_state:
    st.session_state.processing = False  # Prevent multiple clicks

# Function to validate stock symbol
def validate_stock_symbol(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")

        return not hist.empty  # True if valid, False otherwise
    except Exception as e:
        st.error(f"Error validating stock symbol: {e}")
        return False

# 🔹 Reset function to restart app
def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]  # Properly clear session state
    # Use a workaround to trigger a UI refresh
    st.session_state["reset_trigger"] = True  # Set a flag instead of calling st.rerun()

# Streamlit UI
st.title("Stock Price Prediction System with XAI")

# 🔹 "Reset" button in the top-right corner
if st.sidebar.button("🔄 Reset"):
    reset_app()
    st.rerun()  # ✅ Forces a refresh without a callback issue

# Stock Symbol Input (Disabled after validation)
stock_symbol = st.text_input(
    "Enter Stock Symbol (e.g., AAPL for Apple):",
    value=st.session_state.stock_symbol,
    disabled=st.session_state.validated,
)

# 🔹 Validate Stock Symbol with Pop-up Message
if not st.session_state.validated and st.button("Validate Ticker"):
    if not stock_symbol.strip():
        st.warning("⚠ Please enter a stock symbol before validating.")
    elif validate_stock_symbol(stock_symbol):
        st.session_state.stock_symbol = stock_symbol  # Store ticker in session state
        st.session_state.validated = True  # Lock input field
        st.success(f"✅ {stock_symbol} is a valid stock ticker!")  # Show success message
        st.rerun()  # Refresh UI to disable input
    else:
        st.error(f"❌ '{stock_symbol}' is not a valid stock symbol. Please enter a valid one.")

# 🔹 If ticker is validated, show buttons
if st.session_state.validated:
    st.write(f"✅ Validated Stock Symbol: **{st.session_state.stock_symbol}**")


    # Fetch Prices Button
    if st.button("Fetch Prices"):
        hist = fetch_stock_prices(stock_symbol)  # Get historical data
        if hist is not None and not hist.empty:
            st.write(f"📊 Historical Stock Data for {stock_symbol}:")
        
        # Ensure numerical columns are properly converted
            for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
                hist[col] = pd.to_numeric(hist[col], errors='coerce')

            hist = hist.dropna()  # Remove any NaN values

        # Display cleaned data
            st.dataframe(hist.head())

        # Plot correctly formatted graph
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hist.index, hist['Close'], color='cyan', linewidth=2, label='Close Price')
            ax.set_title(f"Stock Price Chart for {stock_symbol}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error(f"⚠ No historical data found for {stock_symbol}.")


    # Stock Prediction Button
    if st.button("Stock Prediction"):
        hist = fetch_stock_prices(st.session_state.stock_symbol)

        if hist is not None and not hist.empty:
            st.write(f"Historical Stock Data for {st.session_state.stock_symbol}:")
            st.dataframe(hist.head())

            if "Close" in hist.columns:
                hist["Close"] = pd.to_numeric(hist["Close"], errors="coerce")
                hist = hist.dropna(subset=["Close"])

                closing_prices = hist["Close"].values.reshape(-1, 1)

                # Preprocess data
                X_train, y_train, scaler = preprocess_data(closing_prices)

                # Train LSTM model
                model = build_lstm_model()
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

                # Predict stock prices
                predictions = predict_stock_price(model, X_val, scaler)
                st.write(
                    f"Predicted stock price for {st.session_state.stock_symbol}: ${predictions[-1][0]:.2f}"
                )

                # Plot predictions
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(
                    scaler.inverse_transform(y_train.reshape(-1, 1)), label="Training Data", color="blue"
                )
                ax.plot(
                    scaler.inverse_transform(y_val.reshape(-1, 1)), label="Test Data", color="green"
                )
                ax.plot(predictions, label="Predictions", color="red")
                ax.set_xlabel("Time")
                ax.set_ylabel("Stock Price")
                ax.set_title(f"Stock Price Predictions for {st.session_state.stock_symbol}")
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("The 'Close' column is missing from the dataset.")

    # Explainable AI Button
    if st.button("Explain Stock Predictions"):
        try:
            model, shap_values, X = explain_model(st.session_state.stock_symbol)
            st.write("Explainable AI for Stock Predictions:")
            shap_fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X)
            st.pyplot(shap_fig)
        except Exception as e:
            st.write(f"Error in Explainable AI analysis: {e}")

