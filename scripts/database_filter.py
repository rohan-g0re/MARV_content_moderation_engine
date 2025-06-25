# database_filter.py
"""
GuardianAI v1 - Hybrid Rule-Based + Context-Aware Filtering
Uses SQLite for keyword/severity lookup and Detoxify for toxicity scoring.
"""

import sqlite3
import re
import os
import unicodedata
from typing import List, Dict, Any

from detoxify import Detoxify

# === Setup DB ===
def initialize_filter_db(db_path="filters.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS profanity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL,
            severity INTEGER NOT NULL,
            is_regex BOOLEAN DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

# === Normalization Utility ===
def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    return text

# === Core Filter Class ===
class DatabaseFilter:
    def __init__(self, db_path="filters.db", detoxify_model_name="original"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.filters = self._load_filters()

        print(f"🔍 Loading Detoxify model: {detoxify_model_name}")
        self.detoxify_model = Detoxify(detoxify_model_name)

    def _load_filters(self) -> List[Dict[str, Any]]:
        self.cursor.execute("SELECT keyword, severity, is_regex FROM profanity")
        rows = self.cursor.fetchall()
        return [
            {"keyword": row[0], "severity": row[1], "is_regex": bool(row[2])}
            for row in rows
        ]

    def scan(self, text: str) -> Dict[str, Any]:
        normalized_text = normalize_text(text)

        # === Rule-based filtering (SQLite) ===
        total_severity = 0
        for entry in self.filters:
            if entry["is_regex"]:
                if re.search(entry["keyword"], normalized_text, re.IGNORECASE):
                    total_severity += entry["severity"]
            else:
                if entry["keyword"].lower() in normalized_text:
                    total_severity += entry["severity"]

        if total_severity >= 8:
            threat_level = "high"
        elif total_severity >= 4:
            threat_level = "medium"
        else:
            threat_level = "low"

        # === Detoxify filtering ===
        results = self.detoxify_model.predict(text)
        toxicity_score = results["toxicity"]
        label = "toxic" if toxicity_score > 0.5 else "non-toxic"

        # === Action logic ===
        if threat_level == "high" or toxicity_score >= 0.9:
            action = "Block"
        elif threat_level == "medium" or toxicity_score >= 0.6:
            action = "Flag"
        else:
            action = "Accept"

        return {
            "total_severity": total_severity,
            "threat_level": threat_level,
            "toxicity_score": toxicity_score,
            "context_prediction": label,
            "moderation_action": action
        }

    def close(self):
        self.conn.close()

# === Test Entrypoint ===
if __name__ == "__main__":
    print("🛡️ Initializing GuardianAI...")
    initialize_filter_db()
    filter = DatabaseFilter()

    sample_posts = [
        "I hate everything about this damn service!",
        "This is a great product!",
        "I'm going to destroy your account!",
        "Let's discuss the issue rationally.",
        "F**k this community.",
        "You are a genius!"
    ]

    for post in sample_posts:
        print(f"\n📝 Post: {post}")
        result = filter.scan(post)
        print(f"Severity Score: {result['total_severity']} ({result['threat_level']})")
        print(f"Toxicity Score: {result['toxicity_score']:.4f} ({result['context_prediction']})")
        print(f"🛡️ Moderation Action: {result['moderation_action']}")

    filter.close()
