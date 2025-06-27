import streamlit as st
from serpapi import GoogleSearch
import pandas as pd
import yfinance as yf
from sentiment_analysis import perform_sentiment_analysis
import matplotlib.pyplot as plt
import shap
from explainable_ai import explain_model
from lstm_model import fetch_stock_news, perform_sentiment_analysis, fetch_stock_prices, preprocess_data, build_lstm_model, predict_stock_price
from prediction import fetch_and_save_stock_data, validate_stock_data, perform_stock_prediction

# Add your other functions here (validate_stock_symbol, fetch_stock_news, fetch_stock_prices, etc.)
def validate_stock_symbol(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")  # Fetch 1 day of historical data
        return not hist.empty  # Return True if data exists, otherwise False
    except Exception:
        return False  # If any exception occurs, assume the symbol is invalid

# Streamlit app layout
st.title("Stock News, Price Fetcher, and Sentiment Analysis")

# Input: User enters the stock symbol
if "stock_symbol" not in st.session_state:
    st.session_state["stock_symbol"] = ""

stock_symbol_input = st.text_input("Enter Stock Symbol (e.g., AAPL for Apple):", st.session_state["stock_symbol"])
st.session_state["stock_symbol"] = stock_symbol_input

# Validate stock symbol
if not validate_stock_symbol(st.session_state["stock_symbol"]):
    st.error(f"Invalid stock symbol: {st.session_state['stock_symbol']}. Please try again.")
else:
    if "headlines" not in st.session_state:
        st.session_state.headlines = []

    if "sentiment_df" not in st.session_state:
        st.session_state.sentiment_df = pd.DataFrame()

    # Add a button to fetch news and stock prices
    if st.button("Fetch News and Prices"):
        if st.session_state["stock_symbol"]:
            api_key = "YOUR_SERP_API_KEY"  # Replace with your actual SERP API key
            headlines = fetch_stock_news(st.session_state["stock_symbol"], api_key)
            st.session_state.headlines = headlines

            if headlines:
                st.write(f"Stock News Headlines for {st.session_state['stock_symbol']}:")
                for idx, headline in enumerate(headlines, start=1):
                    st.write(f"{idx}. {headline}")

                # Perform sentiment analysis
                sentiment_df = perform_sentiment_analysis(headlines)
                st.session_state.sentiment_df = sentiment_df

                st.write("Sentiment Analysis of Headlines:")
                st.dataframe(sentiment_df)

            else:
                st.write("No news headlines found or the stock symbol is incorrect.")

            try:
                hist, current_price = fetch_stock_prices(st.session_state["stock_symbol"])
                if not hist.empty:
                    st.write(f"Historical Stock Data for {st.session_state['stock_symbol']}:")
                    st.dataframe(hist.head())

                    if current_price:
                        st.write(f"Current Stock Price for {st.session_state['stock_symbol']}: ${current_price}")
                        st.line_chart(hist['Close'])
                    else:
                        st.write("Error fetching current stock price.")
                else:
                    st.write(f"No historical data found for stock symbol: {st.session_state['stock_symbol']}. Please try a different stock symbol.")
        
            except Exception as e:
                st.write(f"Error fetching stock prices: {e}")

    # Save historical data in a CSV and use for prediction
    if st.button("Stock Prediction"):
        if st.session_state["stock_symbol"]:
            stock_symbol = st.session_state["stock_symbol"]

            # Fetch and save stock data to CSV
            csv_filename = fetch_and_save_stock_data(stock_symbol)

            # Validate saved CSV
            if validate_stock_data(csv_filename):
                # Perform stock prediction using the saved CSV
                prediction = perform_stock_prediction(csv_filename)
                st.write(f"Stock prediction for {stock_symbol}: ${prediction:.2f}")
            else:
                st.write("Error in validating the stock data CSV. Please check the file.")

    # Option to save the results to CSV (existing code)
    if st.button("Save to CSV"):
        if not st.session_state.headlines:
            st.write("No data available to save. Please fetch news first.")
        else:
            df = pd.DataFrame(st.session_state.headlines, columns=["Headline"])
            sentiment_df = st.session_state.sentiment_df
            df = df.join(sentiment_df.set_index("headline"), on="Headline")
            csv = df.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv, file_name=f"{st.session_state['stock_symbol']}_stock_news.csv", mime='text/csv')
            st.write(f"Data has been saved to '{st.session_state['stock_symbol']}_stock_news.csv'.")

    # Explainable AI for Stock Predictions (existing code)
    if st.button("Explain Stock Predictions"):
        try:
            stock_symbol = st.session_state["stock_symbol"]
            if not stock_symbol:
                st.write("Stock symbol is empty. Please enter a valid stock symbol.")
            else:
                st.write(f"Stock symbol: {stock_symbol}")
                model, shap_values, X = explain_model(stock_symbol)
            
                st.write("Explainable AI for Stock Predictions:")
                shap_fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X)
                st.pyplot(shap_fig)

        except Exception as e:
            st.write(f"Error in Explainable AI analysis: {e}")
