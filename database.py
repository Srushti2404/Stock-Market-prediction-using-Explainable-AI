import streamlit as st
from serpapi import GoogleSearch
import pandas as pd

# Function to fetch stock news headlines from SERP API
def fetch_stock_news(stock_symbol, api_key, max_results=50):
    headlines = []
    num_results = 0
    current_page = 0
    results_per_page = 10  # Number of results per page

    while num_results < max_results:
        params = {
            "engine": "google",
            "q": f"{stock_symbol} stock news",
            "tbm": "nws",  # tbm=nws is for news search
            "api_key": api_key,
            "start": current_page * results_per_page  # Pagination parameter
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        news_results = results.get("news_results", [])

        if not news_results:
            break  # No more news results available

        # Append the headlines to the list
        for article in news_results:
            headlines.append(article["title"])
            num_results += 1
            if num_results >= max_results:
                break

        current_page += 1

    return headlines

# Streamlit app layout
st.title("Stock News Fetcher")

# Input: User enters the stock symbol
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for Apple):")

# Add a button to fetch news
if st.button("Fetch News"):
    if stock_symbol:
        # Fetch headlines using SERP API
        api_key = "efc288c346f518d31a9620d414f4e3158e1a0d6897daf34c0f5cb69242626b16"  # Replace with your SERP API key
        headlines = fetch_stock_news(stock_symbol, api_key, max_results=50)  # Fetch up to 50 results
        
        # Print fetched news in terminal for debugging
        print(f"Fetched News Headlines for {stock_symbol}:")
        for idx, headline in enumerate(headlines, start=1):
            print(f"{idx}. {headline}")

        # Display the fetched headlines
        if headlines:
            st.write(f"Stock News Headlines for {stock_symbol}:")
            for idx, headline in enumerate(headlines, start=1):
                st.write(f"{idx}. {headline}")
        else:
            st.write("No news headlines found or the stock symbol is incorrect.")
    else:
        st.write("Please enter a valid stock symbol.")

# Option to save the results to CSV
if st.button("Save to CSV"):
    if stock_symbol and headlines:
        df = pd.DataFrame(headlines, columns=["Headline"])
        df.to_csv(f"{stock_symbol}_stock_news.csv", index=False)
        st.write(f"Data has been saved to '{stock_symbol}_stock_news.csv'.")
    else:
        st.write("No data available to save. Please fetch news first.")
