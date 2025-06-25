import sqlite3
import json
import os

# Paths
json_path = "data/words.json"  # adjust as needed
db_path = "database/filters.db"

# Load words from JSON
with open(json_path, "r") as f:
    words = json.load(f)

# Connect to SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Insert logic
inserted = 0
for word in words:
    if not word.strip():
        continue  # skip empty

    is_regex = int("+" in word or "*" in word or "\\" in word)
    severity = 5 if is_regex else 3

    cursor.execute(
        "INSERT INTO profanity (keyword, severity, is_regex) VALUES (?, ?, ?)",
        (word, severity, is_regex)
    )
    inserted += 1

conn.commit()
conn.close()

print(f"✅ Inserted {inserted} words into profanity database.")
