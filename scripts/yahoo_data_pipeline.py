# yahoo_data_pipeline.py
"""
Fetch financial news via Yahoo Finance RSS,
classify sentiment using HuggingFace transformers,
and save labeled data to CSV.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from transformers import pipeline
import torch

def load_sentiment_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device set to use {'cuda' if device == 0 else 'cpu'}")
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

def fetch_yahoo_finance_rss(ticker: str):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, "xml")

        articles = []
        for item in soup.find_all("item"):
            title = item.title.text if item.title else "No Title"
            link = item.link.text if item.link else "No Link"
            pub_date = item.pubDate.text if item.pubDate else datetime.utcnow().isoformat()
            articles.append({
                "title": title,
                "link": link,
                "published": pub_date,
                "publisher": "Yahoo Finance"
            })
        return articles
    except Exception as e:
        print("❌ Failed to fetch articles:", e)
        return []

def classify_sentiment(articles, classifier):
    for article in articles:
        try:
            result = classifier(article["title"])[0]
            article["sentiment"] = result["label"]
            article["confidence"] = round(result["score"], 4)
        except Exception as e:
            article["sentiment"] = "UNKNOWN"
            article["confidence"] = 0.0
    return articles

def save_to_csv(records, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    print(f"✅ Saved {len(df)} records to {path}")

def main():
    ticker = input("Enter company ticker (e.g., AAPL, TSLA): ").strip().upper()
    print(f"🔄 Fetching Yahoo Finance news for: {ticker}")
    articles = fetch_yahoo_finance_rss(ticker)
    print(f"📥 Retrieved {len(articles)} articles.")

    if not articles:
        print("⚠️ No matching articles found.")
        return

    print("🧠 Classifying sentiment...")
    classifier = load_sentiment_pipeline()
    labeled_articles = classify_sentiment(articles, classifier)

    output_path = f"data/real/yahoo_finance_{ticker.lower()}_labeled.csv"
    print("💾 Saving results...")
    save_to_csv(labeled_articles, output_path)

if __name__ == "__main__":
    main()
