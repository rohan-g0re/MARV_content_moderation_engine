import sys
import joblib
import numpy as np
from pathlib import Path

# Add backend/app/core to sys.path
core_path = Path(__file__).resolve().parent.parent / 'backend' / 'app' / 'core'
sys.path.append(str(core_path))

try:
    from features import extract_features  # type: ignore
except ImportError:
    from backend.app.core.features import extract_features  # type: ignore

import lightgbm as lgb

# --- Load model and support files ---
MODEL_PATH = Path("backend/app/core/lgbm_moderation.txt")
SUPPORT_PATH = Path("backend/app/core/lgbm_support.pkl")

booster = lgb.Booster(model_file=str(MODEL_PATH))
keywords, feature_names, tfidf_vec = joblib.load(str(SUPPORT_PATH))

# --- Test posts ---
test_posts = [
    "Hi",
    "Click here to double your money instantly!",
    "I'm not sure if this is legit, but it sounds too good to be true...",
    "Looking for advice on saving for retirement.",
    "CONGRATULATIONS! You've won a cash prize. DM me to claim.",
    "The market is volatile, but patience pays off in the long run.",
    "Send $1000 in bitcoin to this address and get $2000 back!",
    "Is this investment opportunity real or a scam?",
    "Check out this link: http://scammy-site.biz",
    "What are your thoughts on the new tax regulations?"
]

label_map = {0: "PASS", 1: "FLAG", 2: "BLOCK"}

print("\nTesting LGBM moderation model on generic posts:\n")
for post in test_posts:
    features = extract_features(post, keywords, tfidf_vec)
    X = np.array([[features[f] for f in feature_names]])
    probs = booster.predict(X)
    probs = np.asarray(probs)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]
    pred_class = int(np.argmax(probs[0]))
    confidence = float(np.max(probs[0]))
    print(f"Post: {post}\n  Prediction: {label_map[pred_class]}  |  Confidence: {confidence:.2f}\n  Probabilities: PASS={probs[0][0]:.2f}, FLAG={probs[0][1]:.2f}, BLOCK={probs[0][2]:.2f}")

    # Feature contribution (explainability)
    contribs = booster.predict(X, pred_contrib=True)
    contribs = np.asarray(contribs)
    n_classes = probs.shape[1]
    n_feat = len(feature_names)
    if contribs.shape[1] == (n_feat + 1) * n_classes:
        # Multiclass: split contributions per class
        class_contribs = contribs[0].reshape(n_classes, n_feat + 1)[pred_class][:-1]
    else:
        # Binary or fallback: use as is
        class_contribs = contribs[0][:-1]
    feature_contrib_pairs = list(zip(feature_names, class_contribs, [features[f] for f in feature_names]))
    # Sort by absolute contribution
    top_feats = sorted(feature_contrib_pairs, key=lambda x: abs(x[1]), reverse=True)[:5]
    print("  Top feature contributions for this decision:")
    for fname, contrib, fval in top_feats:
        print(f"    {fname}: value={fval}, contrib={contrib:+.4f}")
    print() 