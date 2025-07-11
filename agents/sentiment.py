from crewai import Agent
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import requests
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

class NewsSentiment(BaseModel):
    """Model for news sentiment analysis results."""
    title: str = Field(..., description="News article title")
    date: str = Field(..., description="Publication date")
    source: str = Field(..., description="News source")
    url: Optional[str] = Field(None, description="Article URL")
    sentiment_score: float = Field(..., description="Sentiment score (-1 to 1)")
    sentiment_label: str = Field(..., description="Sentiment label (positive, neutral, negative)")

class SentimentSummary(BaseModel):
    """Model for overall sentiment summary."""
    symbol: str = Field(..., description="Stock ticker symbol")
    period: str = Field(..., description="Analysis time period")
    average_sentiment_score: float = Field(..., description="Average sentiment score across all articles")
    sentiment_trend: str = Field(..., description="Trend in sentiment (improving, deteriorating, stable)")
    sentiment_distribution: Dict[str, float] = Field(..., description="Distribution of sentiment labels")
    top_positive_news: List[NewsSentiment] = Field(..., description="Top positive news articles")
    top_negative_news: List[NewsSentiment] = Field(..., description="Top negative news articles")
    overall_assessment: str = Field(..., description="Overall sentiment assessment and potential market impact")

class SentimentAnalysisAgent:
    """
    Agent responsible for analyzing news sentiment for stocks.
    Retrieves news articles and analyzes their sentiment to gauge market perception.
    """
    
    def __init__(self, llm):
        """Initialize the sentiment analysis agent."""
        self.agent = Agent(
            role="Market Sentiment Analyst",
            goal="Analyze news sentiment for stocks and provide insights on market perception",
            backstory="""You are a market sentiment expert who specializes in analyzing financial 
                       news and social media content to gauge market perception about stocks. Your 
                       insights help investors understand how public sentiment might influence 
                       stock prices and market movements.""",
            llm=llm,
            verbose=True,
            memory=True,
        )
    
    def generate_sentiment_analysis_code(self, symbol: str, period: str, news_count: int = 10) -> str:
        """
        Generate Python code for sentiment analysis of news articles.
        
        Args:
            symbol: The stock ticker symbol
            period: The time period for analysis
            news_count: Number of news articles to analyze
            
        Returns:
            Python code string for the sentiment analysis
        """
        return f"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
import re
from urllib.parse import quote
import time
import random

# Fetch stock data to get date range
stock_data = yf.download('{symbol}', period='{period}', interval='1d')
end_date = datetime.now()
start_date = stock_data.index[0]

# Function to fetch news from Yahoo Finance
def get_yahoo_finance_news(symbol, limit={news_count}):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={{quote(symbol)}}&newsCount={{limit}}"
    headers = {{'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}}
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        news = data.get('news', [])
        return news
    except Exception as e:
        print(f"Error fetching news: {{e}}")
        return []

# Function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    
    if sentiment_score > 0.1:
        sentiment_label = 'Positive'
    elif sentiment_score < -0.1:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    
    return sentiment_score, sentiment_label

# Fetch and analyze news
print("Fetching and analyzing news articles...")
news_list = get_yahoo_finance_news('{symbol}', {news_count})

# Process each news article
sentiment_data = []
for article in news_list:
    title = article.get('title', '')
    published_date = datetime.fromtimestamp(article.get('providerPublishTime', 0))
    source = article.get('publisher', '')
    url = article.get('link', '')
    
    # Skip if title is empty
    if not title:
        continue
    
    # Clean and analyze title
    cleaned_title = clean_text(title)
    sentiment_score, sentiment_label = analyze_sentiment(cleaned_title)
    
    sentiment_data.append({{
        'title': title,
        'date': published_date.strftime('%Y-%m-%d'),
        'source': source,
        'url': url,
        'sentiment_score': sentiment_score,
        'sentiment_label': sentiment_label
    }})

# Convert to DataFrame
df = pd.DataFrame(sentiment_data)

# Sort by sentiment score
df_sorted = df.sort_values(by='sentiment_score', ascending=False)

# Calculate sentiment metrics
avg_sentiment = df['sentiment_score'].mean()
sentiment_counts = df['sentiment_label'].value_counts()
sentiment_distribution = sentiment_counts / len(df) if len(df) > 0 else pd.Series()

# Create visualizations
plt.figure(figsize=(12, 12))

# 1. Sentiment Distribution Pie Chart
plt.subplot(2, 2, 1)
colors = ['green', 'gray', 'red']
labels = ['Positive', 'Neutral', 'Negative']
sizes = [sentiment_distribution.get(label, 0) for label in labels]
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title(f'Sentiment Distribution for {symbol} News')

# 2. Sentiment Scores Bar Chart
plt.subplot(2, 2, 2)
plt.bar(range(len(df)), df['sentiment_score'], color=df['sentiment_score'].apply(
    lambda x: 'green' if x > 0.1 else ('red' if x < -0.1 else 'gray')))
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title(f'Sentiment Scores for {symbol} News Articles')
plt.xlabel('Article Index')
plt.ylabel('Sentiment Score')

# 3. Sentiment Timeline (if dates are available)
plt.subplot(2, 1, 2)
df['date_parsed'] = pd.to_datetime(df['date'])
df_timeline = df.sort_values('date_parsed')
plt.scatter(df_timeline['date_parsed'], df_timeline['sentiment_score'], 
           c=df_timeline['sentiment_score'].apply(lambda x: 'green' if x > 0.1 else ('red' if x < -0.1 else 'gray')),
           s=100, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title(f'Sentiment Timeline for {symbol}')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print top positive and negative news
print(f"\\n===== SENTIMENT ANALYSIS SUMMARY FOR {symbol} =====")
print(f"\\nAnalysis Period: {period}")
print(f"Average Sentiment Score: {avg_sentiment:.4f}")

sentiment_trend = "IMPROVING" if avg_sentiment > 0.1 else ("DETERIORATING" if avg_sentiment < -0.1 else "STABLE")
print(f"Sentiment Trend: {sentiment_trend}")

print("\\nSentiment Distribution:")
for label in labels:
    print(f"  {label}: {sentiment_distribution.get(label, 0):.1%}")

print("\\nTop Positive News:")
for i, row in df_sorted.head(3).iterrows():
    print(f"  - {row['title']} (Score: {row['sentiment_score']:.2f}, Source: {row['source']})")

print("\\nTop Negative News:")
for i, row in df_sorted.tail(3).sort_values('sentiment_score').iterrows():
    print(f"  - {row['title']} (Score: {row['sentiment_score']:.2f}, Source: {row['source']})")

# Overall assessment
print("\\n===== MARKET SENTIMENT ASSESSMENT =====")
if avg_sentiment > 0.3:
    print(f"STRONGLY POSITIVE SENTIMENT: The news coverage for {symbol} is overwhelmingly positive, which could drive significant bullish momentum in the stock price. Investors appear optimistic about the company's prospects.")
elif avg_sentiment > 0.1:
    print(f"POSITIVE SENTIMENT: The news coverage for {symbol} leans positive, which may contribute to upward price movement. There seems to be general optimism about the company.")
elif avg_sentiment < -0.3:
    print(f"STRONGLY NEGATIVE SENTIMENT: The news coverage for {symbol} is predominantly negative, which could lead to significant selling pressure. Investors appear concerned about the company's prospects.")
elif avg_sentiment < -0.1:
    print(f"NEGATIVE SENTIMENT: The news coverage for {symbol} leans negative, which may contribute to downward price movement. There appears to be some pessimism or concern about the company.")
else:
    print(f"NEUTRAL SENTIMENT: The news coverage for {symbol} is relatively balanced or neutral. This suggests that there is no strong sentiment-driven momentum in either direction at the moment.")

# Alignment with price movement
recent_price_change = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]
print(f"\\nRecent Price Change: {recent_price_change:.2%}")

if (recent_price_change > 0 and avg_sentiment > 0) or (recent_price_change < 0 and avg_sentiment < 0):
    print(f"SENTIMENT ALIGNED WITH PRICE: The market sentiment appears to be aligned with the recent price movement, suggesting that the trend may continue.")
elif (recent_price_change > 0 and avg_sentiment < 0):
    print(f"POTENTIAL REVERSAL WARNING: Despite the positive price movement, the negative sentiment could indicate a potential upcoming reversal or correction.")
elif (recent_price_change < 0 and avg_sentiment > 0):
    print(f"POTENTIAL RECOVERY SIGNAL: Despite the negative price movement, the positive sentiment could indicate a potential upcoming recovery or bounce.")
"""
    
    def analyze_sentiment(self, symbol: str, period: str, news_count: int = 10) -> SentimentSummary:
        """
        Perform sentiment analysis and return structured results.
        
        Args:
            symbol: The stock ticker symbol
            period: The time period for analysis
            news_count: Number of news articles to analyze
            
        Returns:
            SentimentSummary: Structured result of the sentiment analysis
        """
        # This would typically involve executing the code and parsing results
        # Placeholder implementation
        return SentimentSummary(
            symbol=symbol,
            period=period,
            average_sentiment_score=0.0,
            sentiment_trend="stable",
            sentiment_distribution={"positive": 0.0, "neutral": 0.0, "negative": 0.0},
            top_positive_news=[],
            top_negative_news=[],
            overall_assessment="Analysis not implemented yet"
        )
