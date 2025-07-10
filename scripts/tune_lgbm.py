import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, make_scorer, f1_score
import lightgbm as lgb

# Add backend/app/core to sys.path
core_path = Path(__file__).resolve().parent.parent / 'backend' / 'app' / 'core'
sys.path.append(str(core_path))

try:
    from features import extract_features, SCAM_PHRASES, SCAM_REGEXES  # type: ignore
except ImportError:
    try:
        from backend.app.core.features import extract_features, SCAM_PHRASES, SCAM_REGEXES  # type: ignore
    except ImportError as e:
        print("Could not import features.py. Please check your path.")
        raise e

# --- CONFIG ---
DATA_FILE = "synthetic_moderation_data.csv"
MODEL_OUT = Path("backend/app/core/lgbm_moderation.txt")
SUPPORT_OUT = Path("backend/app/core/lgbm_support.pkl")
SEED = 42
N_ITER = 30
N_SPLITS = 5

# --- Load Data ---
df = pd.read_csv(DATA_FILE)
label_map = {"PASS": 0, "FLAG": 1, "BLOCK": 2}
df['label_enc'] = df['label'].apply(lambda x: label_map[x]).astype(int).values

# Before fitting the TF-IDF vectorizer, drop or fill NaNs in the 'text' column
if 'text' in df.columns:
    df = df.dropna(subset=['text'])
    # Alternatively, to fill NaNs with empty string, use:
    # df['text'] = df['text'].fillna('')

# --- TF-IDF Vectorizer (fit on all text) ---
tfidf_vec = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=40,
    min_df=2,
    stop_words='english',
)
tfidf_vec.fit(df['text'])

# --- Feature Extraction ---
keywords = SCAM_PHRASES
feature_dicts = [extract_features(text, keywords, tfidf_vec) for text in df['text']]
X = pd.DataFrame(feature_dicts)
feature_names = list(X.columns)
y = df['label_enc']

# --- Hyperparameter Search Space ---
param_dist = {
    'num_leaves': [8, 16, 32, 64],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
    'min_child_samples': [10, 20, 25, 40],
    'lambda_l1': [0, 0.1, 0.5, 1, 2],
    'lambda_l2': [0, 0.1, 0.5, 1, 2],
    'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
    'bagging_freq': [1, 2, 4],
    'max_depth': [3, 5, 7, -1],
}

# --- Scorers ---
f1_macro = make_scorer(f1_score, average='macro')

# --- LGBMClassifier ---
clf = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    random_state=SEED,
    verbosity=-1,
    n_estimators=200
)

# --- RandomizedSearchCV ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring={'f1_macro': f1_macro, 'neg_log_loss': 'neg_log_loss'},
    refit='f1_macro',
    cv=skf,
    verbose=2,
    n_jobs=-1,
    random_state=SEED
)
search.fit(X, y)

print("Best parameters:")
print(search.best_params_)
print(f"Best macro F1: {search.best_score_:.4f}")

# --- Evaluate on all data ---
preds = search.best_estimator_.predict(X)
print(classification_report(y, preds, target_names=label_map.keys()))

# --- Save best model and support files ---
# Save as Booster for compatibility
booster = search.best_estimator_.booster_
booster.save_model(str(MODEL_OUT))
joblib.dump((keywords, feature_names, tfidf_vec), str(SUPPORT_OUT))
print(f"Saved best model to {MODEL_OUT}")
print(f"Saved support to {SUPPORT_OUT}")

# --- Feature Importance ---
importances = booster.feature_importance(importance_type='gain')
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False)
print("Top 20 features:")
print(imp_df.head(20)) 