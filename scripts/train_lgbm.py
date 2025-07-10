import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
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
N_SPLITS = 5
SEED = 42

# --- Load Data ---
df = pd.read_csv(DATA_FILE)
assert set(df['label']).issubset({"PASS", "FLAG", "BLOCK"}), "Unexpected labels in data!"

# --- Label Encoding ---
label_map = {"PASS": 0, "FLAG": 1, "BLOCK": 2}
df['label_enc'] = df['label'].apply(lambda x: label_map[x]).astype(int).values

# --- TF-IDF Vectorizer (fit on all text) ---
tfidf_vec = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=40,
    min_df=2,
    stop_words='english',
)
tfidf_vec.fit(df['text'])

# --- Feature Extraction ---
# Use scam phrases and regexes as keywords
keywords = SCAM_PHRASES
feature_dicts = [extract_features(text, keywords, tfidf_vec) for text in df['text']]
X = pd.DataFrame(feature_dicts)
feature_names = list(X.columns)
y = df['label_enc']

# --- LGBM Params (anti-overfitting) ---
lgbm_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'learning_rate': 0.08,
    'num_leaves': 16,
    'min_child_samples': 25,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'random_state': SEED,
    'verbosity': -1,
}

# --- Cross-validation with LGBMClassifier for early stopping ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
val_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    clf = lgb.LGBMClassifier(**lgbm_params, n_estimators=200)
    clf.fit(
        X.iloc[train_idx], y[train_idx],
        eval_set=[(X.iloc[val_idx], y[val_idx])],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)]
    )
    preds = clf.predict(X.iloc[val_idx])
    print(classification_report(y[val_idx], preds, target_names=label_map.keys()))
    # Use best_score_ if available, else skip
    if hasattr(clf, 'best_score_') and 'valid_0' in clf.best_score_ and 'multi_logloss' in clf.best_score_['valid_0']:
        val_scores.append(clf.best_score_['valid_0']['multi_logloss'])

if val_scores:
    print(f"Mean CV multi_logloss: {np.mean(val_scores):.4f}")

# --- Train final model on all data using Booster for saving as .txt ---
lgb_train = lgb.Dataset(X, label=y, feature_name=feature_names)
final_model = lgb.train(
    {k: v for k, v in lgbm_params.items() if k != 'random_state'},
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_train]
)

# --- Save model and support files ---
final_model.save_model(str(MODEL_OUT))
joblib.dump((keywords, feature_names, tfidf_vec), str(SUPPORT_OUT))
print(f"Saved model to {MODEL_OUT}")
print(f"Saved support to {SUPPORT_OUT}")

# --- Feature Importance ---
importances = final_model.feature_importance(importance_type='gain')
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False)
print("Top 20 features:")
print(imp_df.head(20)) 