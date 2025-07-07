import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np

# Load model + support
model = lgb.Booster(model_file="backend/app/core/lgbm_moderation.txt")
keywords, feature_names, tfidf_vec = joblib.load("backend/app/core/lgbm_support.pkl")

from backend.app.core.features import extract_features

# Test text
test_text = "I'm going to kill you"

# Extract features
features = extract_features(test_text, keywords, tfidf_vec)
X = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)

# Predict
probs = model.predict(X)[0]   # This should be a vector of 3 probabilities
print("Probabilities:", probs)
print("Predicted class:", np.argmax(probs))
