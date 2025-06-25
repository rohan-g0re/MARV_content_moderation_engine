# scripts/insert_keywords.py

import sqlite3
import os

DB_PATH = os.path.join("database", "filters.db")

def populate_filters_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS profanity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL,
            severity INTEGER NOT NULL,
            is_regex BOOLEAN DEFAULT 0
        )
    """)

    # Define keywords
    keywords = [
        ("fuck", 5, 0),
        ("shit", 4, 0),
        ("idiot", 3, 0),
        ("dumb", 2, 0),
        ("suck", 2, 0),
        ("hate", 3, 0),
        ("damn", 2, 0),
        ("f+u+c+k", 5, 1),
    ]

    # Insert keywords
    for word, severity, is_regex in keywords:
        cursor.execute("""
            INSERT INTO profanity (keyword, severity, is_regex)
            VALUES (?, ?, ?)
        """, (word, severity, is_regex))

    conn.commit()
    conn.close()
    print(f"✅ Populated profanity table in {DB_PATH}")

if __name__ == "__main__":
    populate_filters_db()
