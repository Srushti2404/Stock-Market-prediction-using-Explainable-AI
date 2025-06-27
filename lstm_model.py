import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type:ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type:ignore
import streamlit as st
import requests
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# Function to fetch news from SERP API
def fetch_stock_news(stock_symbol, api_key):
    params = {
        'q': stock_symbol,
        'location': 'Mumbai',
        'num': 10,
        'api_key': api_key
    }
    try:
        response = requests.get("https://serpapi.com/search.json", params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        news = response.json().get("news_results", [])
        headlines = [item['title'] for item in news]
        return headlines
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news data from SERP API: {e}")
        return []

# Perform basic sentiment analysis (using TextBlob or another library)
def perform_sentiment_analysis(headlines):
    from textblob import TextBlob
    sentiments = []
    for headline in headlines:
        analysis = TextBlob(headline)
        sentiment = analysis.sentiment.polarity
        sentiments.append(sentiment)
    return sentiments

# Function to fetch stock price data and save to CSV
def fetch_stock_prices(stock_symbol):
    csv_file = f'{stock_symbol}.csv'

    if os.path.exists(csv_file):
        st.write(f"Loading stock data from existing {csv_file}")
        stock_data = pd.read_csv(csv_file, index_col=0)  # Ensure 'Date' remains the index
    else:
        st.write(f"Fetching stock data for {stock_symbol} and saving to {csv_file}")
        stock_data = yf.download(tickers=stock_symbol, period='5y', interval='1d')
        stock_data.to_csv(csv_file, index=True)

    # Ensure correct column names
    stock_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

     # **Fix: Remove any row where the index (Date) is not in datetime format**
    stock_data.index = pd.to_datetime(stock_data.index, errors='coerce')
    stock_data = stock_data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    return stock_data

# Function to preprocess the stock data for LSTM
def preprocess_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        X_train.append(scaled_data[i - time_step:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    # Check the shape before reshaping
    st.write(f"Shape before reshaping: {X_train.shape}, {y_train.shape}")
    
    # Reshaping X_train for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Check the shape after reshaping
    st.write(f"Shape after reshaping: {X_train.shape}")
    return X_train, y_train, scaler

# Build LSTM Model
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Predicting the next price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to make predictions
def predict_stock_price(model, X_test, scaler):
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions)

# Streamlit app
def main():
    st.title("Stock Price Prediction using SERP API and LSTM")

    # User input
    stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
    api_key = st.text_input("Enter SERP API Key", "your_serp_api_key_here")  

    if st.button("Stock Prediction"):
        # Fetch stock news using SERP API
        headlines = fetch_stock_news(stock_symbol, api_key)
        if headlines:
            st.write(f"Top News Headlines for {stock_symbol}:")
            for idx, headline in enumerate(headlines, start=1):
                st.write(f"{idx}. {headline}")

        # Perform sentiment analysis on headlines
        sentiments = perform_sentiment_analysis(headlines)
        sentiment_df = pd.DataFrame(sentiments, columns=["Sentiment"])
        st.write("Sentiment Scores:")
        st.dataframe(sentiment_df)

        # Fetch stock prices and save to CSV
        stock_data = fetch_stock_prices(stock_symbol)
        if not stock_data.empty:
            st.write(f"Historical Stock Data for {stock_symbol}:")
            st.dataframe(stock_data.head())

            # Preprocess data
            closing_prices = stock_data['Close'].values.reshape(-1, 1)
            X_train, y_train, scaler = preprocess_data(closing_prices)

            # Build and train LSTM model
            model = build_lstm_model()
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            model.fit(
                X_train,
                y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
            )

            # Make predictions
            predictions = predict_stock_price(model, X_val, scaler)
            st.write(
                f"Predicted stock price for {stock_symbol}: ${predictions[-1][0]:.2f}"
            )

            # Plot the predictions
            fig, ax = plt.subplots(figsize=(12, 6))  # Create figure and axes
            ax.plot(
                scaler.inverse_transform(y_train.reshape(-1, 1)), label="Training Data", color="blue"
            )
            ax.plot(
                scaler.inverse_transform(y_val.reshape(-1, 1)), label="Test Data", color="green"
            )
            ax.plot(
                predictions,  # predictions already inverse transformed
                label="Predictions",
                color="red",
            )
            ax.set_xlabel("Time")
            ax.set_ylabel("Stock Price")
            ax.set_title(f"Stock Price Predictions for {stock_symbol}")
            ax.legend()

            # Show the plot in Streamlit
            st.pyplot(fig)  # Pass the figure object

        else:
            st.write(f"No stock data found for {stock_symbol}.")

if __name__ == "__main__":
    main()