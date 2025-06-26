# scripts/moderation_engine.py

import os
import sqlite3
import datetime
from pathlib import Path
from database_filter import DatabaseFilter, initialize_filter_db

# === Constants ===
DB_PATH = os.path.join(Path(__file__).parent.parent, "database", "moderation.db")

# === Init Database ===
def init_moderation_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Posts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Moderation logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moderation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            action TEXT,
            severity_score INTEGER,
            toxicity_score REAL,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(post_id) REFERENCES posts(id)
        )
    """)

    conn.commit()
    conn.close()

# === Moderation Pipeline ===
def moderate_post(text: str) -> dict:
    # 1. Store post in DB (status pending)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO posts (text, status) VALUES (?, ?)", (text, "Pending"))
    post_id = cursor.lastrowid

    # 2. Apply rule-based + Detoxify
    filter = DatabaseFilter()
    result = filter.scan(text)
    filter.close()

    severity = result.get("severity_score", 0)
    toxicity = result.get("toxicity_score", 0.0)
    action = result.get("moderation_action", "Accept")

    explanation = f"Rule-based severity={severity}, Detoxify={toxicity:.2f}"

    # 3. Update post status
    cursor.execute("UPDATE posts SET status = ? WHERE id = ?", (action, post_id))

    # 4. Log moderation
    cursor.execute("""
        INSERT INTO moderation_logs (post_id, action, severity_score, toxicity_score, explanation)
        VALUES (?, ?, ?, ?, ?)
    """, (post_id, action, severity, toxicity, explanation))

    conn.commit()
    conn.close()

    # 5. Return full moderation result
    return {
        "post_id": post_id,
        "action": action,
        "severity_score": severity,
        "toxicity_score": toxicity,
        "explanation": explanation
    }

# === Run Demo ===
if __name__ == "__main__":
    init_moderation_db()

    sample_post = input("📝 Enter post to moderate (max 300 words):\n> ")
    result = moderate_post(sample_post)

    print("\n🛡️ Moderation Result:")
    print(f"Post ID:         {result['post_id']}")
    print(f"Action:          {result['action']}")
    print(f"Severity Score:  {result['severity_score']}")
    print(f"Toxicity Score:  {result['toxicity_score']:.4f}")
    print(f"Explanation:     {result['explanation']}")
