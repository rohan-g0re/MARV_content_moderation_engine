import pandas as pd
import json
import joblib
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.app.core.features import extract_features

# --- Load your data ---
df = pd.read_csv(r"data/raw/gemini_dataset_v1.csv")
texts = df['post'].tolist()

# --- Multiclass label mapping (update as needed) ---
def map_label(label):
    if label in {"INAPPROPRIATE", "HARMFUL", "ILLEGAL", "SCAM", "FRAUD"}:
        return 2  # BLOCK
    elif label in {"QUESTIONABLE", "SUSPICIOUS"}:
        return 1  # FLAG
    else:
        return 0  # PASS

y = [map_label(l) for l in df['label']]

# --- Load keywords ---
with open(r"data/words.json", "r", encoding="utf-8") as f:
    keywords = json.load(f)

# --- Fit TF-IDF vectorizer (1- and 2-grams) ---
tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=40)
tfidf_vec.fit(texts)

# --- Extract features ---
X = pd.DataFrame([extract_features(text, keywords, tfidf_vec) for text in texts])
feature_names = list(X.columns)

print(f"Training set shape: {X.shape}")
print("Example features:", X.head(2).to_dict())

# --- Train LightGBM model (multiclass) ---
train_data = lgb.Dataset(X, label=y)
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'learning_rate': 0.1,
    'class_weight': 'balanced',
    'seed': 42,
    'verbose': -1
}
model = lgb.train(params, train_data, num_boost_round=100)

# --- Save model and support files ---
model.save_model("backend/app/core/lgbm_moderation.txt")
joblib.dump((keywords, feature_names, tfidf_vec), "backend/app/core/lgbm_support.pkl")
print("✅ LightGBM model and support files saved!")
# --- Save features and labels for inspection ---
debug_df = X.copy()
debug_df['post'] = texts
debug_df['original_label'] = df['label']
debug_df['mapped_label'] = y
debug_df.to_csv("data/processed/lgbm_features_labels_debug.csv", index=False)
print("🔎 Feature debug CSV saved to data/debug/lgbm_features_labels_debug.csv")