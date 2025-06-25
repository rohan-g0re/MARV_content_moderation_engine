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

# === Normalize path to filters.db in /database ===
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "filters.db"))

# === Utility ===
def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    return text

# === Initialize DB only if it doesn't exist ===
def initialize_filter_db(db_path: str):
    if not os.path.exists(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
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
        print(f"✅ Database created at {db_path}")
    else:
        print(f"✅ Using existing database at {db_path}")

# === Main Filter Class ===
class DatabaseFilter:
    def __init__(self, db_path=DB_PATH, detoxify_model_name="original"):
        self.db_path = db_path
        print(f"📂 Connecting to DB at: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.filters = self._load_filters()

        print(f"🔍 Loading Detoxify model: {detoxify_model_name}")
        self.model = Detoxify(detoxify_model_name)

    def _load_filters(self) -> List[Dict[str, Any]]:
        self.cursor.execute("SELECT keyword, severity, is_regex FROM profanity")
        rows = self.cursor.fetchall()
        return [
            {"keyword": row[0], "severity": row[1], "is_regex": bool(row[2])}
            for row in rows
        ]

    def scan(self, text: str) -> Dict[str, Any]:
        norm_text = normalize_text(text)
        matched_severity = 0

        for entry in self.filters:
            try:
                if entry["is_regex"]:
                    if re.search(entry["keyword"], norm_text):
                        matched_severity += entry["severity"]
                else:
                    if entry["keyword"] in norm_text:
                        matched_severity += entry["severity"]
            except re.error:
                continue

        # Determine threat level from severity
        if matched_severity >= 8:
            threat_level = "high"
        elif matched_severity >= 4:
            threat_level = "medium"
        else:
            threat_level = "low"

        # Use Detoxify for toxicity scoring
        result = self.model.predict(text)
        toxicity_score = result["toxicity"]
        context_label = "toxic" if toxicity_score > 0.5 else "non-toxic"

        # Final moderation action logic
        if threat_level == "high" or toxicity_score >= 0.9:
            moderation_action = "Block"
        elif threat_level == "medium" or toxicity_score >= 0.6:
            moderation_action = "Flag"
        else:
            moderation_action = "Accept"

        return {
            "severity_score": matched_severity,
            "threat_level": threat_level,
            "toxicity_score": round(toxicity_score, 4),
            "context_prediction": context_label,
            "moderation_action": moderation_action
        }

    def close(self):
        self.conn.close()


# === Test Runner ===
if __name__ == "__main__":
    print("🛡️ Initializing GuardianAI...")
    initialize_filter_db(DB_PATH)
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
        print(f"Severity Score: {result['severity_score']} ({result['threat_level']})")
        print(f"Toxicity Score: {result['toxicity_score']} ({result['context_prediction']})")
        print(f"🛡️ Moderation Action: {result['moderation_action']}")

    filter.close()
