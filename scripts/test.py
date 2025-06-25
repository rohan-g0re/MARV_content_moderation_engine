import sqlite3

conn = sqlite3.connect("database/filters.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM profanity LIMIT 5")
print(cursor.fetchall())
conn.close()
