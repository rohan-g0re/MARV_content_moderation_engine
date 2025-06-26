import csv
import json

# Load existing words.json
with open('words.json', 'r', encoding='utf-8') as f:
    words = set(json.load(f))

# Helper to add words from a CSV if mean_rating > 6
def add_from_csv(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if float(row['mean_rating']) > 6:
                    word = row['word'].strip()
                    if word:
                        words.add(word)
            except Exception:
                continue

add_from_csv('dictionary_7plus.csv')
add_from_csv('dictionary_5plus.csv')

# Save back to words.json
with open('words.json', 'w', encoding='utf-8') as f:
    json.dump(sorted(words), f, ensure_ascii=False, indent=2)

print(f"words.json updated! Total words: {len(words)}")