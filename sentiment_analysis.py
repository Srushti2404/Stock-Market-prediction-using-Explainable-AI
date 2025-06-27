# sentiment_analysis.py

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import pandas as pd

# Function to perform sentiment analysis using FinBERT
def perform_sentiment_analysis(headlines):
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    sentiment_results = []
    for headline in headlines:
        result = sentiment_pipeline(headline)[0]
        sentiment_results.append({
            "headline": headline,
            "sentiment": result['label'].lower(),
            "score": result['score']
        })

    return pd.DataFrame(sentiment_results)

