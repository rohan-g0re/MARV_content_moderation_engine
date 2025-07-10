import re
import numpy as np
import string
from typing import List, Dict
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Define common scammy financial phrases (expand as needed)
SCAM_PHRASES = [
    "guaranteed returns", "double your money", "get rich quick",
    "millionaire", "risk free", "no investment needed", "act now",
    "quick money", "100% profit", "insider tip", "secret method",
    "overnight success", "only today", "become rich", "click here",
    "no experience needed", "huge payout", "investment opportunity",
    "limited spots", "passive income", "fast cash", "unbeatable offer",
    "cash prize", "dm me", "ping me", "withdraw instantly"
]

# Optional: Add regex patterns for numeric promises or crypto shilling
SCAM_REGEXES = [
    r"\b\d{2,}%\s+(return|profit|gain|yield)\b",     # e.g., 200% return
    r"\bbitcoin\b|\bethereum\b|\bcrypto\b",           # crypto-specific
    r"make \$\d+",                                   # make $5000
    r"\b(?:no|zero)\s+(risk|fees)\b",                # no risk/fees
    r"\b(win|won|winner|prize)\b.*\b(cash|money)\b", # prize money
]

def extract_features(text: str, keywords: List[str] = [], tfidf_vec=None) -> Dict[str, float]:
    """Extracts features for LR moderation, optimized for scam detection."""
    features = {}
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    num_words = len(words)
    num_chars = len(text)
    num_upper = sum(1 for c in text if c.isupper())
    num_exclaims = text.count('!')
    num_questions = text.count('?')
    num_links = len(re.findall(r'https?://\S+', text_lower))
    num_mentions = text.count('@')
    num_dollar = text.count('$')
    num_hashtags = text.count('#')
    num_digits = sum(1 for c in text if c.isdigit())
    num_punct = sum(1 for c in text if c in string.punctuation)
    stopword_count = sum(1 for w in words if w in STOPWORDS)
    unique_words = len(set(words))

    # Base features (as before)
    features.update({
        "num_words": num_words,
        "num_chars": num_chars,
        "avg_word_len": (num_chars / num_words) if num_words > 0 else 0,
        "num_upper": num_upper,
        # Down-weighted features:
        "num_exclaims": num_exclaims * 0.1,  # Down-weighted
        "num_questions": num_questions * 0.1,  # Down-weighted
        "num_links": num_links,
        "num_mentions": num_mentions,
        "num_dollar": num_dollar,
        "num_hashtags": num_hashtags,
        "num_digits": num_digits,
        "num_punct": num_punct,
        "stopword_count": stopword_count,
        "unique_words": unique_words,
        "all_caps": int(text.isupper()),
        "starts_with_http": int(text_lower.strip().startswith('http')),
        "frac_upper": num_upper / num_chars if num_chars > 0 else 0,
        "frac_stopwords": stopword_count / num_words if num_words > 0 else 0,
    })

    # Keyword (as before)
    features["keyword_present"] = int(any(kw.lower() in text_lower for kw in keywords))
    features["keyword_count"] = sum(text_lower.count(kw.lower()) for kw in keywords)

    # Scam phrase features
    for phrase in SCAM_PHRASES:
        featname = f"phrase_{phrase.replace(' ','_')}"
        features[featname] = int(phrase in text_lower)

    # Regex scam indicators
    for i, pattern in enumerate(SCAM_REGEXES):
        features[f"regex_match_{i}"] = int(bool(re.search(pattern, text_lower)))

    # Optional: TF-IDF features for important bigrams (requires fit vectorizer)
    # If using, pass tfidf_vec at train/infer time
    if tfidf_vec is not None:
        X_tfidf = tfidf_vec.transform([text])
        tfidf_feature_names = tfidf_vec.get_feature_names_out()
        for idx, fname in enumerate(tfidf_feature_names):
            features[f"tfidf_{fname}"] = float(X_tfidf[0, idx])

    return features
